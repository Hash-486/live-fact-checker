"""Node 3 — classify each retrieved document's stance toward the claim.

All documents are judged in ONE structured-output call. One call per
document would mean six round-trips before the verdict node even starts.
"""

from src.llm import get_llm
from src.models import SourceDocument, Stance, StanceJudgment, StanceReport
from src.prompts import fence
from src.state import FactCheckState

# The claim is fenced alongside the documents: for URL input it is itself
# scraped text, so treating it as the authoritative task definition would
# hand an attacker-controlled page the top of the prompt.
_PROMPT_TEMPLATE = """You are assessing how sources relate to a claim.

CLAIM UNDER TEST:
{claim}

Retrieved web documents follow.

{documents}

For each document, decide its stance toward the claim:
- "supports": the document's content affirms the claim
- "refutes": the document's content contradicts the claim
- "unrelated": the document does not bear on the claim either way

Return one judgment per document, using the document's exact URL.
"""

# Full article bodies blow past the context window quickly; the opening
# section carries the stance in nearly all cases.
_MAX_CHARS_PER_DOCUMENT = 4000

# A document the model returned no judgment for is recorded rather than
# dropped, so `judgments` always lines up one-to-one with `documents` and
# confidence is not computed over a silently incomplete sample.
_OMITTED_REASONING = (
    "The stance classifier returned no judgment for this retrieved document, "
    "so it is recorded as unrelated rather than silently omitted."
)


def build_stance_prompt(claim: str, documents: list[SourceDocument]) -> str:
    """Build the batched stance prompt with untrusted content fenced off."""
    rendered = "\n\n".join(
        fence(
            "retrieved-document",
            doc.content[:_MAX_CHARS_PER_DOCUMENT],
            url=doc.url,
            title=doc.title,
        )
        for doc in documents
    )
    return _PROMPT_TEMPLATE.format(
        claim=fence("claim-under-test", claim), documents=rendered
    )


def reconcile_judgments(
    judgments: list[StanceJudgment], documents: list[SourceDocument]
) -> list[StanceJudgment]:
    """Force the model's judgments to line up with what was actually retrieved.

    Every URL in a judgment is a string the model produced. Unchecked, a
    hallucinated one flows straight into the result's citations. So: drop
    judgments for URLs that were never retrieved, keep the first of any
    duplicates rather than double-weighting one document, and stand in a
    neutral judgment for documents the model skipped.

    The document's URL is what survives, never the model's copy of it, so
    the returned list can only ever cite something that was retrieved.
    """
    documents_by_url = {doc.url: doc for doc in documents}
    first_by_url: dict[str, StanceJudgment] = {}
    for judgment in judgments:
        if judgment.url in documents_by_url and judgment.url not in first_by_url:
            first_by_url[judgment.url] = judgment

    return [
        first_by_url.get(url)
        or StanceJudgment(
            url=url, stance=Stance.UNRELATED, reasoning=_OMITTED_REASONING
        )
        for url in documents_by_url
    ]


def stance_node(state: FactCheckState) -> dict:
    """Judge every retrieved document's stance in a single LLM call."""
    documents = state.get("documents", [])
    if not documents:
        return {"judgments": []}

    llm = get_llm().with_structured_output(StanceReport)
    report = llm.invoke(build_stance_prompt(state.get("claim", ""), documents))
    return {"judgments": reconcile_judgments(report.judgments, documents)}
