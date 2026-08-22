from unittest.mock import patch

from src.nodes.normalize import normalize_node


def test_plain_text_becomes_the_claim():
    result = normalize_node({"raw_input": "  the sky is green  "})
    assert result == {"claim": "the sky is green"}


def test_url_input_is_extracted_into_the_claim():
    with patch(
        "src.nodes.normalize.extract_claim_text", return_value="Article body."
    ):
        result = normalize_node({"raw_input": "https://example.com/a"})
    assert result == {"claim": "Article body."}


def test_returns_only_the_keys_it_changed():
    result = normalize_node({"raw_input": "x"})
    assert set(result) == {"claim"}
