from unittest.mock import patch

from src.cli import format_result, main
from src.models import FactCheckResult, Stance, StanceJudgment, Verdict

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
    sources=[],
)


def test_format_includes_verdict_confidence_and_sources():
    text = format_result(RESULT)
    assert "FALSE" in text
    assert "95%" in text
    assert "https://nasa.gov/a" in text


def test_main_returns_zero_on_success(capsys):
    with patch("src.cli.run_fact_check", return_value=RESULT):
        exit_code = main(["the moon landing was faked"])

    assert exit_code == 0
    assert "FALSE" in capsys.readouterr().out


def test_main_errors_without_an_argument(capsys):
    assert main([]) == 1
    assert "Usage" in capsys.readouterr().err
