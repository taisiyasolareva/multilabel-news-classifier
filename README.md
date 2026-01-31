---
title: News Classification API
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "3.0.0"
app_file: app.py
pinned: false
---

# multilabel-news-classifier

**Multi-label news classification for Russian articles.** Transformer backbones (DistilBERT, RuBERT) + LoRA fine-tuning + reproducible model zoo protocol + FastAPI + Streamlit dashboards. 141 labels (snippet-aware).

**Proof points:** Opt F1 **0.4518** (DistilBERT + LoRA, 10k train / 1k val) · threshold 0.15 · deployed on HuggingFace Spaces (API) + Streamlit Cloud (UI).

**CTAs:** [Live Demo](https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier) · [Portfolio Blurb](docs/PORTFOLIO_BLURB.md) · [Results](docs/RESULTS.md) · [Architecture](docs/ARCHITECTURE.md) · [Demo Walkthrough](docs/DEMO.md)

## Live Deployment

- **Streamlit Classifier (UI)**: https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier
- **API base**: https://solarevat-multilabel-news-classifier.hf.space
  - **API docs**: https://solarevat-multilabel-news-classifier.hf.space/docs
  - **Health**: https://solarevat-multilabel-news-classifier.hf.space/health
- **Served model**: distilmbert_lora_10k_v1 (threshold 0.15)

## Results at a glance

Source of truth: `experiments/results/*.json` and [docs/RESULTS.md](docs/RESULTS.md).

| Model | Backbone | Snippet | Opt Threshold | Opt F1 | Precision | Recall |
|-------|----------|---------|---------------|--------|-----------|--------|
| **distilmbert_lora_10k_v1** (served) | DistilBERT + LoRA | ✅ | 0.15 | **0.4518** | 0.4338 | 0.4713 |
| rubert_base_lora_10k_v1 | RuBERT + LoRA | ✅ | 0.19 | 0.3702 | — | — |
| rubert_snippet_ablation_lora_10k_v1 | RuBERT + LoRA | ❌ | 0.09 | 0.2305 | — | — |

Protocol: frozen 10k train / 1k val split (`experiments/model_zoo/protocol_10k_1k`).

## API

- `GET /health` — health check, model info
- `POST /classify` — single classification
- `POST /classify/batch` — batch classification
- `GET /docs` — Swagger UI

## Run locally

### Quick (Docker)

```bash
docker compose up --build
```

- **API**: http://localhost:8000/docs · http://localhost:8000/health
- **Streamlit** (Classifier, Evaluation, Analytics, Model Comparison, Sentiment): http://localhost:8501

Requires `models/distilmbert_lora_10k_v1.pt` (download via `scripts/download_model.py`; see [docs/DEMO.md](docs/DEMO.md)).

### Without Docker

Requires `models/distilmbert_lora_10k_v1.pt` (see [docs/DEMO.md](docs/DEMO.md) for download).

```bash
pip install -r requirements.txt -r requirements-api.txt
export MODEL_PATH=models/distilmbert_lora_10k_v1.pt
export THRESHOLDS_PATH=config/thresholds.json
python scripts/start_api.py --reload
```

## Repo structure

```
├── api/              # FastAPI app (inference, monitoring)
├── pages/            # Streamlit multipage app (Classifier, Evaluation, Analytics, …)
├── models/           # Transformer architectures
├── config/           # thresholds.json, model configs
├── experiments/      # Model zoo protocol + results JSONs
├── scripts/          # train_model.py, evaluate.py, download_model.py
├── app.py            # HF Spaces entry point (FastAPI)
└── streamlit_app.py  # Streamlit multipage entry
```

## Where to read next

- **Results & protocol**: [docs/RESULTS.md](docs/RESULTS.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Demo walkthrough** (curl, dashboards): [docs/DEMO.md](docs/DEMO.md)
- **Portfolio narrative**: [docs/PORTFOLIO_BLURB.md](docs/PORTFOLIO_BLURB.md)
- **Deployment**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
