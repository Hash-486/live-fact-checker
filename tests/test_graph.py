from unittest.mock import patch

from src.models import SourceDocument, Stance, StanceJudgment, Verdict
from src.graph import build_graph, run_fact_check

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


def test_end_to_end_flow_produces_a_result():
    # NOTE ON PATCH TARGETS: all four names are from-imports, so each must be
    # patched where the binding lives, not where it was originally defined.
    # `src/graph.py` does `from src.nodes.stance import stance_node` (and
    # likewise for verdict_node), so those are patched at
    # "src.graph.stance_node" / "src.graph.verdict_node". The normalize and
    # search nodes call `extract_claim_text` / `search_evidence` for real, so
    # those are patched at the modules that import them:
    # "src.nodes.normalize.extract_claim_text" and
    # "src.nodes.search_node.search_evidence".
    with patch("src.nodes.normalize.extract_claim_text", return_value="claim x"), patch(
        "src.nodes.search_node.search_evidence", return_value=[DOC]
    ), patch("src.graph.stance_node", return_value={"judgments": [JUDGMENT]}):
        with patch(
            "src.graph.verdict_node",
            return_value={
                "verdict": Verdict.FALSE,
                "confidence": 0.9,
                "reasoning": "refuted",
            },
        ):
            result = run_fact_check("claim x")

    assert result.claim == "claim x"
    assert result.verdict == Verdict.FALSE
    assert result.sources == [DOC]
