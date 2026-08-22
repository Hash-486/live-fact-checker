from unittest.mock import MagicMock, patch

from src.models import Stance, StanceJudgment, Verdict
from src.nodes.verdict import VerdictDecision, verdict_node

JUDGMENTS = [
    StanceJudgment(url="https://a.com", stance=Stance.REFUTES, reasoning="debunked"),
]


def test_no_evidence_returns_unverified_without_calling_the_llm():
    with patch("src.nodes.verdict.get_llm") as mock_llm:
        result = verdict_node({"claim": "c", "judgments": [], "documents": []})

    mock_llm.assert_not_called()
    assert result["verdict"] == Verdict.UNVERIFIED
    assert result["confidence"] == 0.0
    assert "insufficient evidence" in result["reasoning"].lower()


def test_synthesizes_verdict_from_judgments():
    structured = MagicMock()
    structured.invoke.return_value = VerdictDecision(
        verdict=Verdict.FALSE, confidence=0.92, reasoning="refuted by sources"
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    with patch("src.nodes.verdict.get_llm", return_value=llm):
        result = verdict_node({"claim": "c", "judgments": JUDGMENTS})

    assert result["verdict"] == Verdict.FALSE
    assert result["confidence"] == 0.92
    assert result["reasoning"] == "refuted by sources"
