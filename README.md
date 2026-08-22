# Live Fact Checker

An on-demand fact checker: submit a claim or a news-article URL, and a
LangGraph pipeline searches the live web, judges each source's stance, and
returns a cited verdict with calibrated confidence — `unverified` included.

## Pipeline

`normalize` → `search` → `stance` → `verdict`

- **normalize** — a URL is fetched and reduced to a short claim; plain text passes through.
- **search** — Tavily returns the top N candidate documents.
- **stance** — every document is judged supports/refutes/unrelated in one batched LLM call.
- **verdict** — the stances are synthesized into a verdict, confidence, and citations.

Every inter-node payload is a Pydantic model (`src/models.py`), and all
retrieved web content is fenced as untrusted data in prompts (`src/prompts.py`).

## Running it

Set `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` in a `.env` file, then:

```
pip install -r requirements.txt
pytest -v                                  # test suite
python -m src.cli "the moon landing was faked"
uvicorn src.api:app                        # API + web UI at http://127.0.0.1:8000
```

> Architecture diagram, design rationale, and known limitations are written
> at Stage 9 — see `plan.md`.
