"""Direct unit tests for src.api._encode.

FakeGraph in test_api_stream.py only ever yields plain dicts/lists/
primitives, so the Pydantic-model and bare-enum branches of _encode are
never exercised by the stream tests (nor by the live smoke run, which
died at the search node before stance/verdict — the only real nodes that
emit Pydantic objects — ever ran). These tests call _encode directly with
the actual payload shapes the real graph nodes put on the wire.
"""

import json

from src.api import _encode
from src.models import SourceDocument, StanceJudgment, Stance, Verdict


def test_encode_source_document_is_json_serializable():
    doc = SourceDocument(
        url="https://example.com",
        title="t",
        snippet="s",
        content="c",
        published_date="2024-01-01",
    )

    encoded = _encode(doc)

    json.dumps(encoded)  # must not raise
    assert encoded == doc.model_dump()
    assert encoded["url"] == "https://example.com"


def test_encode_stance_judgment_flattens_the_nested_stance_enum():
    judgment = StanceJudgment(url="u", stance=Stance.SUPPORTS, reasoning="r")

    encoded = _encode(judgment)

    serialized = json.dumps(encoded)  # must not raise
    assert json.loads(serialized)["stance"] == "supports"


def test_encode_bare_verdict_enum():
    encoded = _encode(Verdict.FALSE)

    json.dumps(encoded)  # must not raise
    assert encoded == "false"


def test_encode_nested_search_node_payload():
    """Shape the real search node emits: {'documents': [SourceDocument, ...]}."""
    payload = {
        "documents": [
            SourceDocument(url="u1", title="t1", snippet="s1", content="c1"),
            SourceDocument(url="u2", title="t2", snippet="s2", content="c2"),
        ]
    }

    encoded = _encode(payload)

    serialized = json.dumps(encoded)  # must not raise
    decoded = json.loads(serialized)
    assert [doc["url"] for doc in decoded["documents"]] == ["u1", "u2"]


def test_encode_nested_verdict_node_payload():
    """Shape the real verdict node emits: verdict enum plus a judgments list."""
    payload = {
        "verdict": Verdict.FALSE,
        "confidence": 0.9,
        "judgments": [StanceJudgment(url="u", stance=Stance.REFUTES, reasoning="r")],
    }

    encoded = _encode(payload)

    serialized = json.dumps(encoded)  # must not raise
    decoded = json.loads(serialized)
    assert decoded["verdict"] == "false"
    assert decoded["confidence"] == 0.9
    assert decoded["judgments"][0]["stance"] == "refutes"
