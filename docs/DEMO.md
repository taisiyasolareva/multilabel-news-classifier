# Demo Walkthrough

Copy/paste guide for trying the project without training.

---

## Option A: One-command local demo (recommended)

```bash
docker compose up --build
```

Then open:
- **API docs**: http://localhost:8000/docs
- **API health**: http://localhost:8000/health
- **Streamlit multipage app**: http://localhost:8501 (Classifier, Evaluation, Analytics, Model Comparison, Sentiment in sidebar)

**Note:** Docker Compose expects `models/distilmbert_lora_10k_v1.pt`. If missing, run:
```bash
python scripts/download_model.py --model-id distilmbert_lora_10k_v1 \
  --url "$MODEL_URL" --output-path models/distilmbert_lora_10k_v1.pt
```
**MODEL_URL:** Get the direct asset URL from the repo's GitHub Releases (see [docs/DEPLOYMENT.md](DEPLOYMENT.md)).

---

## Option B: Local (no Docker)

**Terminal 1 — API:**
```bash
export MODEL_PATH=models/distilmbert_lora_10k_v1.pt
export THRESHOLDS_PATH=config/thresholds.json
python scripts/start_api.py --reload
```

**Terminal 2 — Streamlit:**
```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and set `API_URL` to `http://localhost:8000` in the sidebar.

---

## Try the API (copy/paste)

### Classify (text only)

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Путин объявил о новых мерах поддержки экономики",
    "top_k": 5
  }'
```

### Classify (title + snippet)

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Рубль укрепился на фоне роста цен на нефть",
    "snippet": "Российский рубль укрепился на фоне роста мировых цен на нефть.",
    "top_k": 5
  }'
```

---

## Dashboards with existing artifacts (no training)

### Evaluation dashboard
Upload a sample predictions CSV:
- `experiments/sample_outputs/distilmbert_lora_10k_v1_val_preds_sample_50.csv`

### Model comparison dashboard
Reads experiment results from `experiments/results/` (already in repo).
