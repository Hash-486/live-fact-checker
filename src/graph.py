"""LangGraph assembly: normalize -> search -> stance -> verdict."""

from functools import cache

from langgraph.graph import END, START, StateGraph

from src.models import FactCheckResult, Verdict
from src.nodes.normalize import normalize_node
from src.nodes.search_node import search_node
from src.nodes.stance import stance_node
from src.nodes.verdict import verdict_node
from src.state import FactCheckState


@cache
def build_graph():
    """Return the compiled graph, wiring and compiling it on first use.

    The wiring never varies, so the StateGraph is compiled once for the
    process rather than on every request. Memoizing the existing function
    rather than assigning a module-level constant keeps `build_graph` itself
    the seam the callers and tests already patch and call.
    """
    builder = StateGraph(FactCheckState)

    builder.add_node("normalize", normalize_node)
    builder.add_node("search", search_node)
    builder.add_node("stance", stance_node)
    builder.add_node("verdict", verdict_node)

    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "search")
    builder.add_edge("search", "stance")
    builder.add_edge("stance", "verdict")
    builder.add_edge("verdict", END)

    return builder.compile()


def run_fact_check(raw_input: str) -> FactCheckResult:
    """Run the full pipeline and return a typed result."""
    final_state = build_graph().invoke({"raw_input": raw_input})

    return FactCheckResult(
        # Every sibling below reads defensively; this one raised KeyError
        # inside result construction if the node had not populated it.
        claim=final_state.get("claim", raw_input),
        verdict=final_state.get("verdict", Verdict.UNVERIFIED),
        confidence=final_state.get("confidence", 0.0),
        reasoning=final_state.get("reasoning", ""),
        judgments=final_state.get("judgments", []),
        sources=final_state.get("documents", []),
    )
