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
    for judgment in result.judgments:
        lines.append(f"  [{judgment.stance.value:<9}] {judgment.url}")
        lines.append(f"              {judgment.reasoning}")
    if not result.judgments:
        lines.append("  (none retrieved)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python -m src.cli "<claim or url>"', file=sys.stderr)
        return 1

    print(format_result(run_fact_check(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
