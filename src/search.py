"""Live web search for corroborating or refuting evidence.

Publish dates are captured here because Stage 4's recirculated-content
detection needs them.
"""

from tavily import TavilyClient

from src.config import Settings, get_settings
from src.models import SourceDocument


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
                content=hit.get("raw_content") or snippet,
                published_date=hit.get("published_date"),
            )
        )
    return documents
