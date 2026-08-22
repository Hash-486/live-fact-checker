import pytest

from src.config import Settings
from src.llm import get_llm


def _settings(**overrides) -> Settings:
    base = dict(
        llm_backend="anthropic",
        llm_model="claude-opus-5",
        anthropic_api_key="test-key",
        tavily_api_key="test-key",
        ollama_model="llama3.1:8b",
        search_top_n=6,
    )
    base.update(overrides)
    return Settings(**base)


def test_returns_anthropic_client_for_anthropic_backend():
    llm = get_llm(_settings())
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model == "claude-opus-5"


def test_returns_ollama_client_for_ollama_backend():
    llm = get_llm(_settings(llm_backend="ollama"))
    assert type(llm).__name__ == "ChatOllama"


def test_unknown_backend_fails_loudly():
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_llm(_settings(llm_backend="gpt4"))


def test_anthropic_backend_without_key_fails_loudly():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_llm(_settings(anthropic_api_key=None))
