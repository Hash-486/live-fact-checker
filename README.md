# Fake News Detection

> **Status:** scaffold only. The folder/file layout below exists, but every
> `src/*.py`, `data/*`, and `models/*` file is currently empty (0 bytes).
> This document describes the **intended** architecture inferred from that
> layout, as a design reference for implementation — it is not a description
> of working code yet.

## Goal

Classify a news article as **real** or **fake**, using a combination of
signals: the article's own text, its emotional tone, and how semantically
similar it is to known/trusted reporting.

## Pipeline (planned)

```
                 ┌────────────────┐
                 │  api_fetch.py  │  pull raw articles (news API) into
                 └───────┬────────┘  data/datasets/training_data.csv
                         │
                 ┌───────▼────────┐
                 │ preprocessing.py│ clean + tokenize text, build features
                 └───────┬────────┘  → data/processed/tfidf_features.pkl
                         │           → data/processed/bert_embeddings.pkl
        ┌────────────────┼─────────────────┐
        │                │                 │
┌───────▼──────┐ ┌───────▼────────┐ ┌──────▼────────────┐
│model_training │ │sentiment_      │ │similarity_         │
│    .py        │ │analysis.py     │ │analysis.py          │
└───────┬──────┘ └───────┬────────┘ └──────┬────────────┘
        │                │                 │
   text_classifier   sentiment_analysis   sentence_similarity_model.pkl
      .pkl              _model.pkl         + bert embeddings
   gnn_model.pth      → data/results/     → data/processed/
                        sentiment_          similarity_scores.json
                        analysis.json
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                 ┌───────▼─────────────┐
                 │ prediction_pipeline.py │  combine all signals →
                 └───────┬─────────────┘  data/results/classification_results.json
                         │
                 ┌───────▼────────┐
                 │    main.py     │  orchestrates the full run
                 └────────────────┘
```

## Component responsibilities (inferred from filenames)

| File | Purpose |
|---|---|
| `src/api_fetch.py` | Fetch news articles from an external API into `data/datasets/training_data.csv`. |
| `src/preprocessing.py` | Clean and tokenize article text; build TF-IDF features (`tfidf_features.pkl`) and BERT embeddings (`bert_embeddings.pkl`). |
| `src/model_training.py` | Train the core classifier (`text_classifier.pkl`) and a graph-based model (`gnn_model.pth`), likely modeling article/source relationships or propagation structure. Metrics land in `data/results/model_metrics.json`. |
| `src/sentiment_analysis.py` | Score emotional tone of an article (fake news tends to skew toward emotionally charged language) — `sentiment_analysis_model.pkl` → `data/results/sentiment_analysis.json`. |
| `src/similarity_analysis.py` | Compare an article's embedding against known/trusted articles to gauge semantic similarity — `sentence_similarity_model.pkl` → `data/processed/similarity_scores.json`. |
| `src/prediction_pipeline.py` | Run a new article through preprocessing + all three signal models (classifier, sentiment, similarity) and combine them into a final verdict — `data/results/classification_results.json`. |
| `src/main.py` | Entry point; orchestrates fetch → preprocess → train/predict end to end. |

## Data layout

- `data/datasets/training_data.csv` — labeled training set (real/fake).
- `data/processed/` — intermediate features (`tfidf_features.pkl`, `bert_embeddings.pkl`, `similarity_scores.json`).
- `data/results/` — pipeline outputs (`classification_results.json`, `model_metrics.json`, `sentiment_analysis.json`).
- `models/` — trained/serialized models (`text_classifier.pkl`, `gnn_model.pth`, `sentiment_analysis_model.pkl`, `sentence_similarity_model.pkl`).

## Setup

`requirements.txt` is currently empty — dependencies need to be pinned once
implementation starts. Expected stack based on the artifacts above:
`pandas`/`numpy`, `scikit-learn` (TF-IDF + classifier), `torch` + a GNN
library (e.g. `torch-geometric`) for `gnn_model.pth`, and a sentence-embedding
library (e.g. `sentence-transformers`) for BERT embeddings and similarity.

## Next steps

1. Pin dependencies in `requirements.txt`.
2. Implement `api_fetch.py` and populate `data/datasets/training_data.csv`.
3. Implement `preprocessing.py` → produce TF-IDF/BERT features.
4. Implement `model_training.py`, `sentiment_analysis.py`, `similarity_analysis.py`.
5. Implement `prediction_pipeline.py` to combine the three signals into one verdict.
6. Wire it all together in `main.py`.
