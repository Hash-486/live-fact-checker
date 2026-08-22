import os
from src.config import get_settings


def test_defaults_when_env_unset(monkeypatch):
    for key in ("LLM_BACKEND", "LLM_MODEL", "OLLAMA_MODEL", "SEARCH_TOP_N"):
        monkeypatch.delenv(key, raising=False)
    settings = get_settings()
    assert settings.llm_backend == "anthropic"
    assert settings.llm_model == "claude-opus-5"
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.search_top_n == 6


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("SEARCH_TOP_N", "10")
    settings = get_settings()
    assert settings.llm_backend == "ollama"
    assert settings.llm_model == "claude-sonnet-5"
    assert settings.search_top_n == 10
