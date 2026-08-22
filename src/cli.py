"""Terminal entry point: python -m src.cli "<claim or url>" """

import sys

from src.graph import run_fact_check
from src.models import FactCheckResult


def format_result(result: FactCheckResult) -> str:
    """Render a result as readable terminal output."""
    lines = [
        "",
        f"CLAIM:      {result.claim[:200]}",
        f"VERDICT:    {result.verdict.value.upper()}",
        f"CONFIDENCE: {result.confidence:.0%}",
        "",
        f"REASONING:  {result.reasoning}",
        "",
        "SOURCES:",
    ]
    # Judgments are rendered by joining them against the documents that were
    # actually retrieved, and every printed URL comes from the document, not
    # from the judgment. A URL the model invented therefore cannot be
    # displayed as a source: it simply has no document to join to.
    documents_by_url = {doc.url: doc for doc in result.sources}
    cited = [
        (judgment, documents_by_url[judgment.url])
        for judgment in result.judgments
        if judgment.url in documents_by_url
    ]
    for judgment, document in cited:
        lines.append(f"  [{judgment.stance.value:<9}] {document.title}")
        lines.append(f"              {document.url}")
        lines.append(f"              {judgment.reasoning}")
    if not cited:
        lines.append("  (none retrieved)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python -m src.cli "<claim or url>"', file=sys.stderr)
        return 1

    try:
        print(format_result(run_fact_check(argv[0])))
    except Exception as exc:
        # A failing node otherwise exits with a multi-frame LangGraph pregel
        # traceback. Upstream timeouts and API errors are normal operation
        # here, not exotic bugs, so report them as a message.
        print(f"Fact check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
