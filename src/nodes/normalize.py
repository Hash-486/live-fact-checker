"""Node 1 — turn raw input into a claim string."""

from src.ingest import extract_claim_text
from src.state import FactCheckState


def normalize_node(state: FactCheckState) -> dict:
    """Resolve raw input (text or URL) into the claim under test."""
    return {"claim": extract_claim_text(state["raw_input"])}
