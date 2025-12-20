# Demo Walkthrough (Reviewers)

This is a copy/paste guide for trying the project without training anything.

---

## Option A: One-command local demo (recommended)

```bash
docker compose up --build
```

Then open:
- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`
- Evaluation dashboard: `http://localhost:8501`
- Analytics dashboard: `http://localhost:8502`
- Model comparison dashboard: `http://localhost:8503`
- Sentiment dashboard: `http://localhost:8504`

---

## Try the API (copy/paste)

### Classify (title only)

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

## Use the dashboards with existing artifacts (no training)

### Evaluation dashboard
- Upload a **small sample predictions CSV** (recommended for reviewers):
  - `experiments/sample_outputs/distilmbert_lora_10k_v1_val_preds_sample_50.csv`

### Model comparison dashboard
- Uses the experiment results JSONs in `experiments/results/` (already in repo).

---

## Option B: Local (no Docker)

In one terminal:

```bash
export MODEL_PATH="models/distilmbert_lora_10k_v1.pt"
export THRESHOLDS_PATH="config/thresholds.json"
python scripts/start_api.py --reload
```

In another terminal:

```bash
streamlit run dashboards/model_comparison_dashboard.py
```


