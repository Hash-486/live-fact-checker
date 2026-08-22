from unittest.mock import MagicMock, patch

from src.models import SourceDocument, Stance, StanceJudgment, StanceReport, Verdict
from src.graph import build_graph, run_fact_check
from src.nodes.verdict import VerdictDecision

DOC = SourceDocument(url="https://a.com", title="A", snippet="s", content="body")
JUDGMENT = StanceJudgment(
    url="https://a.com", stance=Stance.REFUTES, reasoning="contradicts"
)


def test_graph_compiles_with_all_four_nodes():
    graph = build_graph()
    assert set(graph.get_graph().nodes) >= {
        "normalize",
        "search",
        "stance",
        "verdict",
    }


def test_build_graph_compiles_once_per_process():
    assert build_graph() is build_graph()


def _structured_llm(*returns):
    """An LLM stub whose structured-output calls return `returns` in order."""
    structured = MagicMock()
    structured.invoke.side_effect = list(returns)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_end_to_end_flow_produces_a_result():
    # NOTE ON PATCH TARGETS: the graph is compiled once per process, so the
    # node callables are captured at compile time and patching
    # "src.graph.stance_node" no longer reaches the compiled graph. Each node
    # is therefore mocked at its own outside edge -- the seam that survives
    # compilation -- which also means the real node bodies run here rather
    # than the test asserting a mock's own return value.
    with patch("src.nodes.normalize.extract_claim_text", return_value="claim x"), patch(
        "src.nodes.search_node.search_evidence", return_value=[DOC]
    ), patch(
        "src.nodes.stance.get_llm",
        return_value=_structured_llm(StanceReport(judgments=[JUDGMENT])),
    ), patch(
        "src.nodes.verdict.get_llm",
        return_value=_structured_llm(
            VerdictDecision(verdict=Verdict.FALSE, confidence=0.9, reasoning="refuted")
        ),
    ):
        result = run_fact_check("claim x")

    assert result.claim == "claim x"
    assert result.verdict == Verdict.FALSE
    assert result.sources == [DOC]
    assert result.judgments == [JUDGMENT]
