from unittest.mock import MagicMock, patch

from src.config import Settings
from src.search import MAX_CONTENT_CHARS, SEARCH_TIMEOUT_SECONDS, search_evidence


def _settings() -> Settings:
    return Settings(
        llm_backend="anthropic",
        llm_model="claude-opus-5",
        anthropic_api_key="k",
        tavily_api_key="k",
        ollama_model="llama3.1:8b",
        search_top_n=2,
    )


FAKE_RESPONSE = {
    "results": [
        {
            "url": "https://reuters.com/a",
            "title": "Reuters story",
            "content": "snippet one",
            "raw_content": "full body one",
            "published_date": "2026-01-05",
        },
        {
            "url": "https://example.com/b",
            "title": "Other story",
            "content": "snippet two",
            "raw_content": None,
        },
    ]
}


def test_maps_results_to_source_documents():
    client = MagicMock()
    client.search.return_value = FAKE_RESPONSE
    with patch("src.search.TavilyClient", return_value=client):
        docs = search_evidence("did X happen", _settings())

    assert len(docs) == 2
    assert docs[0].url == "https://reuters.com/a"
    assert docs[0].content == "full body one"
    assert docs[0].published_date == "2026-01-05"


def test_falls_back_to_snippet_when_raw_content_missing():
    client = MagicMock()
    client.search.return_value = FAKE_RESPONSE
    with patch("src.search.TavilyClient", return_value=client):
        docs = search_evidence("q", _settings())

    assert docs[1].content == "snippet two"
    assert docs[1].published_date is None


def test_passes_top_n_to_tavily():
    client = MagicMock()
    client.search.return_value = {"results": []}
    with patch("src.search.TavilyClient", return_value=client):
        search_evidence("q", _settings())

    assert client.search.call_args.kwargs["max_results"] == 2


def test_passes_an_explicit_timeout_to_tavily():
    client = MagicMock()
    client.search.return_value = {"results": []}
    with patch("src.search.TavilyClient", return_value=client):
        search_evidence("q", _settings())

    assert client.search.call_args.kwargs["timeout"] == SEARCH_TIMEOUT_SECONDS


def test_raw_content_is_bounded_before_it_enters_the_state():
    # Untruncated bodies otherwise ride in the API response and the SSE wire.
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://long.example/a",
                "title": "Long",
                "content": "snippet",
                "raw_content": "x" * 200_000,
            }
        ]
    }
    with patch("src.search.TavilyClient", return_value=client):
        docs = search_evidence("q", _settings())

    assert len(docs[0].content) == MAX_CONTENT_CHARS


def test_empty_results_return_empty_list():
    client = MagicMock()
    client.search.return_value = {"results": []}
    with patch("src.search.TavilyClient", return_value=client):
        assert search_evidence("q", _settings()) == []
