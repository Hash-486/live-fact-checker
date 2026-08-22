"""Shared graph state. Nodes return partial dicts that LangGraph merges in."""

from typing import TypedDict

from src.models import SourceDocument, StanceJudgment, Verdict


class FactCheckState(TypedDict, total=False):
    raw_input: str
    claim: str
    documents: list[SourceDocument]
    judgments: list[StanceJudgment]
    verdict: Verdict
    confidence: float
    reasoning: str
