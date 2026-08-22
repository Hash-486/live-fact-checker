"""Pydantic schemas for every payload crossing a node boundary.

These are the contract the CLI, the API, the web UI, and the browser
extension all depend on. Changing a field name here is a breaking change.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Stance(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    UNRELATED = "unrelated"


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    CONTESTED = "contested"
    UNVERIFIED = "unverified"


class SourceDocument(BaseModel):
    url: str
    title: str
    snippet: str
    content: str
    published_date: str | None = None


class StanceJudgment(BaseModel):
    url: str
    stance: Stance
    reasoning: str


class StanceReport(BaseModel):
    """Wrapper so one structured-output call returns all judgments at once."""

    judgments: list[StanceJudgment]


class FactCheckResult(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    judgments: list[StanceJudgment] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
