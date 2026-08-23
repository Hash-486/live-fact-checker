"""LLM backend factory.

`langchain-anthropic` wraps the official `anthropic` SDK — this is not an
OpenAI-compatible shim.

Do NOT pass `budget_tokens`: it is removed on claude-opus-5 and
claude-sonnet-5 and returns a 400. Do NOT use assistant-message prefill for
the same reason; use structured output instead.
"""

from langchain_core.language_models import BaseChatModel

from src.config import Settings, get_settings

# The Anthropic SDK's default request timeout is 600 seconds. One hung call
# would pin an SSE connection for ten minutes with the UI showing nothing,
# so the wait is bounded explicitly. Generous for this workload: the batched
# stance call is the longest, and it is one call over a handful of documents.
LLM_TIMEOUT_SECONDS = 120

# Ollama allocates a 4096-token context by default, regardless of what the
# model reports it can handle (llama3.1:8b advertises 131072). The batched
# stance prompt runs ~6800 tokens for six documents, so the default silently
# truncates it — and a truncated prompt does not fail, it just yields
# judgments for the handful of documents that survived, plus invented URLs
# assembled from fragments. Measured on llama3.1:8b, claim "the moon landing
# was faked": 1 of 6 documents judged at the default, 6 of 6 at 16384.
# 16384 tokens of KV cache is ~2GB on top of a 4.9GB Q4_K_M model, which
# fits the 8GB VRAM target with headroom.
OLLAMA_NUM_CTX = 16384

# Stance classification and verdict synthesis are classification tasks, not
# creative ones. Ollama's default temperature of 0.8 adds sampling noise to
# a decision that should be reproducible for the same evidence.
OLLAMA_TEMPERATURE = 0.0


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """Return the configured chat model. Nodes must not name a provider."""
    settings = settings or get_settings()

    if settings.llm_backend == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Note that a Claude Pro "
                "subscription does not include API credits — add billing at "
                "console.anthropic.com."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            max_tokens=8000,
            timeout=LLM_TIMEOUT_SECONDS,
        )

    if settings.llm_backend == "ollama":
        from langchain_ollama import ChatOllama

        # Fail loudly if unreachable — never silently fall back to Anthropic.
        # A silent swap makes results non-reproducible.
        # ChatOllama has no timeout field of its own; the httpx client it
        # builds takes one.
        return ChatOllama(
            model=settings.ollama_model,
            num_ctx=OLLAMA_NUM_CTX,
            temperature=OLLAMA_TEMPERATURE,
            client_kwargs={"timeout": LLM_TIMEOUT_SECONDS},
        )

    raise ValueError(
        f"Unknown LLM_BACKEND {settings.llm_backend!r}. "
        "Expected 'anthropic' or 'ollama'."
    )
