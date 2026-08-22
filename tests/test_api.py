from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.models import FactCheckResult, Verdict

client = TestClient(app)

RESULT = FactCheckResult(
    claim="c",
    verdict=Verdict.FALSE,
    confidence=0.9,
    reasoning="r",
    judgments=[],
    sources=[],
)


def test_fact_check_returns_the_result_schema():
    with patch("src.api.run_fact_check", return_value=RESULT):
        response = client.post("/fact-check", json={"claim": "c"})

    assert response.status_code == 200
    assert response.json()["verdict"] == "false"
    assert response.json()["confidence"] == 0.9


def test_empty_claim_is_rejected():
    assert client.post("/fact-check", json={"claim": "  "}).status_code == 422


def test_missing_claim_field_is_rejected():
    assert client.post("/fact-check", json={}).status_code == 422


def test_health_endpoint():
    assert client.get("/health").json() == {"status": "ok"}
