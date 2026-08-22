"""HTTP surface. The web UI and the browser extension are both clients."""

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.graph import build_graph, run_fact_check
from src.models import FactCheckResult

app = FastAPI(title="Live Fact Checker")

# The browser extension calls this from arbitrary news pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimRequest(BaseModel):
    claim: str

    @field_validator("claim")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim must not be empty")
        return value


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/fact-check", response_model=FactCheckResult)
def fact_check(request: ClaimRequest) -> FactCheckResult:
    return run_fact_check(request.claim)


async def stream_fact_check(claim: str) -> AsyncIterator[str]:
    """Emit one Server-Sent Event per completed graph node, then 'done'."""
    graph = build_graph()
    async for update in graph.astream({"raw_input": claim}):
        for node_name, payload in update.items():
            yield "data: " + json.dumps(
                {"node": node_name, "payload": _encode(payload)}
            ) + "\n\n"
    yield "data: " + json.dumps({"event": "done"}) + "\n\n"


def _encode(payload: object) -> object:
    """Make node output JSON-serializable (Pydantic models and enums)."""
    if isinstance(payload, dict):
        return {key: _encode(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_encode(item) for item in payload]
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "value"):
        return payload.value
    return payload


@app.get("/fact-check/stream")
async def fact_check_stream(claim: str) -> StreamingResponse:
    return StreamingResponse(
        stream_fact_check(claim), media_type="text/event-stream"
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
