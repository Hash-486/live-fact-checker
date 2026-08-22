from unittest.mock import patch

from src.ingest import extract_claim_text, is_url


def test_detects_urls():
    assert is_url("https://example.com/story")
    assert is_url("http://example.com")


def test_plain_text_is_not_a_url():
    assert not is_url("the moon landing was faked")
    assert not is_url("visit example.com sometime")


def test_plain_text_passes_through_unchanged():
    assert extract_claim_text("the sky is green") == "the sky is green"


def test_url_is_fetched_and_extracted():
    with patch("src.ingest.trafilatura.fetch_url", return_value="<html/>"), patch(
        "src.ingest.trafilatura.extract", return_value="Extracted body text."
    ):
        assert extract_claim_text("https://example.com/a") == "Extracted body text."


def test_unfetchable_url_raises_clear_error():
    with patch("src.ingest.trafilatura.fetch_url", return_value=None):
        try:
            extract_claim_text("https://example.com/dead")
        except ValueError as exc:
            assert "Could not fetch" in str(exc)
        else:
            raise AssertionError("expected ValueError")
