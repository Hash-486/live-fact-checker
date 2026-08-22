from unittest.mock import MagicMock, patch

from src.models import Stance, StanceJudgment, StanceReport, SourceDocument
from src.nodes.stance import build_stance_prompt, stance_node

DOCS = [
    SourceDocument(url="https://a.com", title="A", snippet="s", content="body a"),
    SourceDocument(url="https://b.com", title="B", snippet="s", content="body b"),
]


def test_prompt_marks_retrieved_content_as_untrusted():
    prompt = build_stance_prompt("some claim", DOCS)
    assert "untrusted" in prompt.lower()
    assert "body a" in prompt
    assert "https://b.com" in prompt


def test_makes_exactly_one_llm_call_for_all_documents():
    structured = MagicMock()
    structured.invoke.return_value = StanceReport(
        judgments=[
            StanceJudgment(url="https://a.com", stance=Stance.SUPPORTS, reasoning="r"),
            StanceJudgment(url="https://b.com", stance=Stance.REFUTES, reasoning="r"),
        ]
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    with patch("src.nodes.stance.get_llm", return_value=llm):
        result = stance_node({"claim": "c", "documents": DOCS})

    assert structured.invoke.call_count == 1
    assert len(result["judgments"]) == 2


def test_no_documents_skips_the_llm_entirely():
    with patch("src.nodes.stance.get_llm") as mock_llm:
        result = stance_node({"claim": "c", "documents": []})

    mock_llm.assert_not_called()
    assert result == {"judgments": []}
