# Live Fact Checker

Submit a claim or a news-article URL and it searches the live web, judges
each source's stance, and returns a cited verdict with a confidence score
(`unverified` is a real outcome, not a failure mode).

## Pipeline

`normalize` -> `search` -> `stance` -> `verdict`, built as a LangGraph
graph in `src/graph.py`.

- normalize: a URL gets fetched and reduced to a short claim, plain text
  passes through as-is
- search: Tavily returns the top N candidate documents
- stance: every document gets judged supports/refutes/unrelated in one
  batched LLM call
- verdict: stances get synthesized into a verdict, confidence, and citations

Payloads between nodes are Pydantic models (`src/models.py`), and any
retrieved web content gets fenced as untrusted data before it hits a prompt
(`src/prompts.py`).

## Running it

Set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` in a `.env` file, then:

```
pip install -r requirements.txt
pytest -v                                  # test suite
python -m src.cli "the moon landing was faked"
uvicorn src.api:app                        # API + web UI at http://127.0.0.1:8000
```

LLM backend is swappable between Anthropic and a local Ollama model via
`LLM_BACKEND` in `.env`, see `.env.example`.

`plan.md` has the original planning doc if you want the full history of
how this got built.
