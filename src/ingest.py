"""Turn raw user input into claim text.

A URL is fetched and reduced to a short claim string; anything else is
treated as the claim itself and passes through untouched.
"""

from urllib.parse import urlparse

import trafilatura
from trafilatura.settings import use_config

# Tavily rejects queries longer than 400 characters, and searching with a
# whole article body is semantically useless well before that limit. Article
# text from a URL is therefore reduced to its headline plus opening
# sentences, which is where the checkable claim almost always sits.
# Stage 5's LLM claim-extraction node replaces this; until then the reduction
# is deliberately deterministic, not a model call.
MAX_CLAIM_CHARS = 400
MAX_CLAIM_BODY_CHARS = 300

# A hung fetch otherwise pins an SSE connection with the UI showing nothing.
FETCH_TIMEOUT_SECONDS = 30

_FETCH_CONFIG = use_config()
_FETCH_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(FETCH_TIMEOUT_SECONDS))


def is_url(text: str) -> bool:
    """True only for input that starts with an http(s) scheme."""
    parsed = urlparse(text.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _truncate(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` characters, on a word boundary."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    boundary = head.rfind(" ")
    return (head[:boundary] if boundary > 0 else head).rstrip(" ,;:-")


def reduce_to_claim(title: str | None, body: str) -> str:
    """Reduce an extracted article to a search-safe claim string."""
    reduced_body = _truncate(" ".join(body.split()), MAX_CLAIM_BODY_CHARS)
    normalized_title = " ".join((title or "").split())
    combined = (
        f"{normalized_title}. {reduced_body}" if normalized_title else reduced_body
    )
    return _truncate(combined, MAX_CLAIM_CHARS)


def extract_claim_text(raw_input: str) -> str:
    """Return claim text, fetching and extracting first if input is a URL."""
    raw_input = raw_input.strip()
    if not is_url(raw_input):
        return raw_input

    downloaded = trafilatura.fetch_url(raw_input, config=_FETCH_CONFIG)
    if downloaded is None:
        raise ValueError(f"Could not fetch {raw_input}")

    extracted = trafilatura.extract(downloaded)
    if not extracted:
        raise ValueError(f"Could not extract article text from {raw_input}")

    metadata = trafilatura.extract_metadata(downloaded)
    return reduce_to_claim(getattr(metadata, "title", None), extracted)
