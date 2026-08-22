"""Live web search for corroborating or refuting evidence.

Publish dates are captured here because Stage 4's recirculated-content
detection needs them.
"""

from tavily import TavilyClient

from src.config import Settings, get_settings
from src.models import SourceDocument

# Bound the wait explicitly rather than trusting the client's default: a
# hung search otherwise leaves an SSE connection open with nothing on screen.
SEARCH_TIMEOUT_SECONDS = 60

# Tavily's raw_content is a full article body. The 4,000-character cap in the
# stance prompt applies only inside that prompt, so untruncated bodies were
# still riding in the graph state, the POST /fact-check response, and the
# `search` SSE event -- six full articles serialized to the page, and the
# Stage 6 extension will consume this over mobile. Bound it at ingest, well
# above what the prompt uses so nothing the model reads is lost.
MAX_CONTENT_CHARS = 8000


def search_evidence(
    query: str, settings: Settings | None = None
) -> list[SourceDocument]:
    """Search the live web and return candidate evidence documents."""
    settings = settings or get_settings()
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set.")

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        max_results=settings.search_top_n,
        include_raw_content=True,
        timeout=SEARCH_TIMEOUT_SECONDS,
    )

    documents: list[SourceDocument] = []
    for hit in response.get("results", []):
        snippet = hit.get("content") or ""
        documents.append(
            SourceDocument(
                url=hit.get("url", ""),
                title=hit.get("title", ""),
                snippet=snippet,
                # raw_content is richer but often absent; snippet is the floor.
                content=(hit.get("raw_content") or snippet)[:MAX_CONTENT_CHARS],
                published_date=hit.get("published_date"),
            )
        )
    return documents
