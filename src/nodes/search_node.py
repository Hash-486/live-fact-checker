"""Node 2 — retrieve live evidence for the claim."""

from src.search import search_evidence
from src.state import FactCheckState


def search_node(state: FactCheckState) -> dict:
    """Search the live web for documents relevant to the claim."""
    return {"documents": search_evidence(state["claim"])}
