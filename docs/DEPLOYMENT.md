# Deployment

**Live URLs**: [Streamlit Classifier](https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier) · [API](https://solarevat-multilabel-news-classifier.hf.space/docs)

## Current deployment

- **API**: HuggingFace Spaces (CPU Basic tier, 16GB RAM). Model weights downloaded at startup from GitHub Release assets.
- **Streamlit UI**: Streamlit Community Cloud, calling the API via `API_URL`.

## HuggingFace Spaces (recommended for API)

1. Create a Space at https://huggingface.co/spaces → SDK: **Docker**
2. Connect GitHub repo
3. Set environment variables:
   - `MODEL_URL` — direct URL to fp16 checkpoint (e.g. GitHub Release asset)
   - `MODEL_DTYPE=float16`
   - `MODEL_PATH=models/distilmbert_lora_10k_v1.pt`
   - `THRESHOLDS_PATH=config/thresholds.json`
   - `TOKENIZER_NAME=distilbert-base-multilingual-cased`
   - `PYTHONPATH=/app`
4. Spaces auto-builds from `app.py` / `Dockerfile`

## Model weights

Do not commit `.pt` files. Publish via GitHub Release:

```bash
# Convert to fp16 for smaller footprint (optional)
python scripts/convert_checkpoint_fp16.py \
  --input models/distilmbert_lora_10k_v1.pt \
  --output models/distilmbert_lora_10k_v1_fp16.pt
```

Upload the asset and set `MODEL_URL` in your host environment.
