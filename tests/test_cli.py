from unittest.mock import patch

from src.cli import format_result, main
from src.models import (
    FactCheckResult,
    SourceDocument,
    Stance,
    StanceJudgment,
    Verdict,
)

NASA_DOC = SourceDocument(
    url="https://nasa.gov/a", title="Apollo record", snippet="s", content="c"
)

# `sources` is populated because the CLI now renders judgments joined against
# the retrieved documents; a judgment with no matching document is not a
# citation and is not displayed. (This test previously passed `sources=[]`
# and relied on the judgment's own URL being printed unchecked — exactly the
# unverified-citation path that join removes.)
RESULT = FactCheckResult(
    claim="the moon landing was faked",
    verdict=Verdict.FALSE,
    confidence=0.95,
    reasoning="Overwhelmingly refuted.",
    judgments=[
        StanceJudgment(
            url="https://nasa.gov/a", stance=Stance.REFUTES, reasoning="documented"
        )
    ],
    sources=[NASA_DOC],
)


def test_format_includes_verdict_confidence_and_sources():
    text = format_result(RESULT)
    assert "FALSE" in text
    assert "95%" in text
    assert "https://nasa.gov/a" in text


def test_format_never_prints_a_judgment_url_that_was_not_retrieved():
    result = RESULT.model_copy(
        update={
            "judgments": RESULT.judgments
            + [
                StanceJudgment(
                    url="https://hallucinated.com",
                    stance=Stance.SUPPORTS,
                    reasoning="invented",
                )
            ]
        }
    )

    text = format_result(result)

    assert "https://hallucinated.com" not in text
    assert "https://nasa.gov/a" in text


def test_main_reports_pipeline_failure_without_a_traceback(capsys):
    with patch("src.cli.run_fact_check", side_effect=RuntimeError("tavily is down")):
        exit_code = main(["some claim"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "tavily is down" in captured.err
    assert "Traceback" not in captured.err


def test_main_returns_zero_on_success(capsys):
    with patch("src.cli.run_fact_check", return_value=RESULT):
        exit_code = main(["the moon landing was faked"])

    assert exit_code == 0
    assert "FALSE" in capsys.readouterr().out


def test_main_errors_without_an_argument(capsys):
    assert main([]) == 1
    assert "Usage" in capsys.readouterr().err
