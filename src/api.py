"""HTTP surface. The web UI and the browser extension are both clients."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.graph import run_fact_check
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
