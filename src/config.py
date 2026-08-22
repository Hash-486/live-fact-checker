"""Environment-driven configuration. No secrets are hardcoded here."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    llm_model: str
    anthropic_api_key: str | None
    tavily_api_key: str | None
    ollama_model: str
    search_top_n: int


def get_settings() -> Settings:
    """Read settings from the environment, applying defaults."""
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", "claude-opus-5"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        search_top_n=int(os.getenv("SEARCH_TOP_N", "6")),
    )
