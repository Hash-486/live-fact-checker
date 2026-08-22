"""Node 4 — synthesize a verdict from the stance judgments.

"unverified" is a first-class outcome, not a failure. Most fake-news systems
force a confident label onto ambiguous claims; refusing to do that is the
point.
"""

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.models import Verdict
from src.prompts import fence
from src.state import FactCheckState

# Both slots below carry text this system did not author: for URL input the
# claim is scraped article text, and the findings quote retrieved URLs and
# reasoning about scraped bodies. Both are fenced as data.
_PROMPT_TEMPLATE = """You are issuing a fact-check verdict.

CLAIM:
{claim}

STANCE FINDINGS FROM RETRIEVED SOURCES:
{findings}

Choose exactly one verdict:
- "true": sources substantiate the claim
- "false": sources contradict the claim
- "misleading": literally defensible but creates a false impression, or
  substantially accurate with material imprecision (a claim of 5% against an
  actual 4.8% is not simply false)
- "contested": credible sources genuinely disagree
- "unverified": evidence is too thin or conflicting to judge

Do not force confidence you do not have. "unverified" is a correct and
expected answer when the evidence does not support a conclusion.

Give a confidence between 0.0 and 1.0 and explain your reasoning in two or
three sentences.
"""


class VerdictDecision(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def verdict_node(state: FactCheckState) -> dict:
    """Combine stance findings into a final verdict."""
    judgments = state.get("judgments", [])

    if not judgments:
        return {
            "verdict": Verdict.UNVERIFIED,
            "confidence": 0.0,
            "reasoning": (
                "No usable sources were retrieved, so there is insufficient "
                "evidence to judge this claim."
            ),
        }

    findings = "\n".join(
        f"- {j.url} -> {j.stance.value}: {j.reasoning}" for j in judgments
    )
    llm = get_llm().with_structured_output(VerdictDecision)
    decision = llm.invoke(
        _PROMPT_TEMPLATE.format(
            claim=fence("claim-under-test", state.get("claim", "")),
            findings=fence("stance-findings", findings),
        )
    )

    return {
        "verdict": decision.verdict,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
    }
