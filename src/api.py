"""HTTP surface. The web UI and the browser extension are both clients."""

import json
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, field_validator

from src.graph import build_graph, run_fact_check
from src.models import FactCheckResult

# Resolved from this file, not the process CWD: StaticFiles validates the
# directory at construction, so a relative path made `import src.api` raise
# RuntimeError from anywhere but the repo root.
STATIC_DIR = Path(__file__).parent.parent / "static"

# The two endpoints are two clients of one pipeline and must not have two
# input contracts. This mirrors ClaimRequest's validation for the query param.
MAX_CLAIM_LENGTH = 2000

# Without these, nginx and most PaaS proxies buffer the whole stream and
# deliver it as one blocking response -- exactly the single blocking spinner
# that throws away the visible-tradecraft point of streaming. Local uvicorn
# does not buffer, which is why this is invisible in development.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

app = FastAPI(title="Live Fact Checker")

# The browser extension calls this from arbitrary news pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _must_not_be_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("claim must not be empty")
    return value


class ClaimRequest(BaseModel):
    claim: str

    _validate_claim = field_validator("claim")(_must_not_be_blank)


# The streaming endpoint previously took a bare `claim: str` with no
# validation at all, so GET /fact-check/stream?claim= ran the whole graph on
# an empty string that POST /fact-check rejects. Sharing the validator makes
# one contract rather than two.
ClaimQuery = Annotated[
    str,
    Query(min_length=1, max_length=MAX_CLAIM_LENGTH),
    AfterValidator(_must_not_be_blank),
]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/fact-check", response_model=FactCheckResult)
def fact_check(request: ClaimRequest) -> FactCheckResult:
    return run_fact_check(request.claim)


async def stream_fact_check(claim: str) -> AsyncIterator[str]:
    """Emit one Server-Sent Event per completed graph node, then 'done'.

    If a node raises (rate limit, transient network error, a bad
    structured-output parse — any of these are normal operation, not
    edge cases), emit a single 'error' event and stop. 'error' is itself
    the terminal signal for the stream: no 'done' event follows it,
    because 'done' means the run completed successfully. A client only
    ever sees exactly one of the two.
    """
    graph = build_graph()
    try:
        async for update in graph.astream({"raw_input": claim}):
            for node_name, payload in update.items():
                yield "data: " + json.dumps(
                    {"node": node_name, "payload": _encode(payload)}
                ) + "\n\n"
    except Exception as exc:
        yield "data: " + json.dumps(
            {"event": "error", "detail": str(exc)}
        ) + "\n\n"
        return
    yield "data: " + json.dumps({"event": "done"}) + "\n\n"


def _encode(payload: object) -> object:
    """Make node output JSON-serializable (Pydantic models and enums)."""
    if isinstance(payload, dict):
        return {key: _encode(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_encode(item) for item in payload]
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, Enum):
        return payload.value
    return payload


@app.get("/fact-check/stream")
async def fact_check_stream(claim: ClaimQuery) -> StreamingResponse:
    return StreamingResponse(
        stream_fact_check(claim),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
