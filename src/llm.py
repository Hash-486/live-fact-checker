"""LLM backend factory.

`langchain-anthropic` wraps the official `anthropic` SDK — this is not an
OpenAI-compatible shim.

Do NOT pass `budget_tokens`: it is removed on claude-opus-5 and
claude-sonnet-5 and returns a 400. Do NOT use assistant-message prefill for
the same reason; use structured output instead.
"""

from langchain_core.language_models import BaseChatModel

from src.config import Settings, get_settings


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
        )

    if settings.llm_backend == "ollama":
        from langchain_ollama import ChatOllama

        # Fail loudly if unreachable — never silently fall back to Anthropic.
        # A silent swap makes results non-reproducible.
        return ChatOllama(model=settings.ollama_model)

    raise ValueError(
        f"Unknown LLM_BACKEND {settings.llm_backend!r}. "
        "Expected 'anthropic' or 'ollama'."
    )
