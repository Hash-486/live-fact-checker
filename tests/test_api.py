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


def test_stream_rejects_the_same_claims_the_post_endpoint_rejects():
    # Two clients of one pipeline must not have two input contracts.
    assert client.get("/fact-check/stream?claim=").status_code == 422
    assert client.get("/fact-check/stream?claim=%20%20").status_code == 422
    assert client.get("/fact-check/stream").status_code == 422


def test_stream_sets_anti_buffering_headers():
    class FakeGraph:
        async def astream(self, _payload):
            yield {"normalize": {"claim": "c"}}

    with patch("src.api.build_graph", return_value=FakeGraph()):
        response = client.get("/fact-check/stream?claim=c")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_static_files_are_resolved_relative_to_the_module(tmp_path, monkeypatch):
    # StaticFiles validates at construction, so a CWD-relative path made
    # `import src.api` raise from anywhere but the repo root.
    import importlib

    import src.api

    monkeypatch.chdir(tmp_path)
    importlib.reload(src.api)
    assert src.api.STATIC_DIR.is_dir()
