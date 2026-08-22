"""Turn raw user input into claim text.

A URL is fetched and reduced to article body text; anything else is treated
as the claim itself.
"""

from urllib.parse import urlparse

import trafilatura


def is_url(text: str) -> bool:
    """True only for input that starts with an http(s) scheme."""
    parsed = urlparse(text.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def extract_claim_text(raw_input: str) -> str:
    """Return claim text, fetching and extracting first if input is a URL."""
    raw_input = raw_input.strip()
    if not is_url(raw_input):
        return raw_input

    downloaded = trafilatura.fetch_url(raw_input)
    if downloaded is None:
        raise ValueError(f"Could not fetch {raw_input}")

    extracted = trafilatura.extract(downloaded)
    if not extracted:
        raise ValueError(f"Could not extract article text from {raw_input}")

    return extracted
