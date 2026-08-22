from unittest.mock import MagicMock, patch

from src.ingest import MAX_CLAIM_CHARS, extract_claim_text, is_url, reduce_to_claim

LONG_ARTICLE = "Sentence number one is here. " * 400  # ~11,600 characters


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


def test_long_article_body_is_reduced_to_a_search_safe_claim():
    # Tavily rejects queries over 400 characters, so a whole article body
    # cannot be handed to it as a query.
    metadata = MagicMock(title="Mayor denies budget claim")
    with patch("src.ingest.trafilatura.fetch_url", return_value="<html/>"), patch(
        "src.ingest.trafilatura.extract", return_value=LONG_ARTICLE
    ), patch("src.ingest.trafilatura.extract_metadata", return_value=metadata):
        claim = extract_claim_text("https://example.com/long")

    assert len(claim) <= MAX_CLAIM_CHARS
    assert claim.startswith("Mayor denies budget claim.")


def test_short_claims_are_not_altered():
    assert reduce_to_claim(None, "the sky is green") == "the sky is green"


def test_reduction_cuts_on_a_word_boundary():
    reduced = reduce_to_claim(None, LONG_ARTICLE)

    assert len(reduced) <= MAX_CLAIM_CHARS
    assert not reduced.endswith("Sen")
    assert LONG_ARTICLE.startswith(reduced)


def test_unfetchable_url_raises_clear_error():
    with patch("src.ingest.trafilatura.fetch_url", return_value=None):
        try:
            extract_claim_text("https://example.com/dead")
        except ValueError as exc:
            assert "Could not fetch" in str(exc)
        else:
            raise AssertionError("expected ValueError")
