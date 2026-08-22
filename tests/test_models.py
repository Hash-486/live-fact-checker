import pytest
from pydantic import ValidationError

from src.models import (
    FactCheckResult,
    SourceDocument,
    Stance,
    StanceJudgment,
    Verdict,
)


def test_source_document_allows_missing_date():
    doc = SourceDocument(
        url="https://example.com/a",
        title="A",
        snippet="s",
        content="full text",
        published_date=None,
    )
    assert doc.published_date is None


def test_stance_enum_values():
    assert Stance.SUPPORTS == "supports"
    assert Stance.REFUTES == "refutes"
    assert Stance.UNRELATED == "unrelated"


def test_verdict_includes_unverified_as_first_class_outcome():
    assert Verdict.UNVERIFIED == "unverified"
    assert Verdict.CONTESTED == "contested"


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        FactCheckResult(
            claim="c",
            verdict=Verdict.TRUE,
            confidence=1.5,
            reasoning="r",
            judgments=[],
            sources=[],
        )


def test_fact_check_result_round_trips():
    result = FactCheckResult(
        claim="the sky is blue",
        verdict=Verdict.TRUE,
        confidence=0.9,
        reasoning="well established",
        judgments=[
            StanceJudgment(
                url="https://example.com/a",
                stance=Stance.SUPPORTS,
                reasoning="states it directly",
            )
        ],
        sources=[],
    )
    assert FactCheckResult.model_validate_json(result.model_dump_json()) == result
