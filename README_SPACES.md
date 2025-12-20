# HuggingFace Spaces Deployment

This repository is configured for deployment on HuggingFace Spaces.

## Setup Instructions

1. **Create a HuggingFace Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select **"Docker"** as the SDK
   - Name your space (e.g., `russian-news-classifier`)
   - Set visibility to **Public** (required for free tier)

2. **Connect GitHub Repository:**
   - In Space settings, connect your GitHub repo
   - Set the repository to: `taisiyasolareva/multilabel-news-classifier`
   - Spaces will automatically detect `app.py` and `Dockerfile.api`

3. **Set Environment Variables:**
   - Go to Space → Settings → Variables
   - Add these variables:
     ```
     MODEL_URL=https://github.com/taisiyasolareva/multilabel-news-classifier/releases/download/v0.1.0/distilmbert_lora_10k_v1_fp16.pt
     MODEL_DTYPE=float16
     MODEL_PATH=models/distilmbert_lora_10k_v1.pt
     THRESHOLDS_PATH=config/thresholds.json
     TOKENIZER_NAME=distilbert-base-multilingual-cased
     PYTHONPATH=/app
     PYTHONUNBUFFERED=1
     ```

4. **Deploy:**
   - Spaces will automatically build and deploy
   - First build takes ~5-10 minutes (downloads dependencies + model)
   - Your API will be available at: `https://your-username-russian-news-classifier.hf.space`

## API Endpoints

Once deployed, your API will be available at:

- **Health Check:** `GET /health`
- **API Docs:** `GET /docs`
- **Classify:** `POST /classify`
- **Batch Classify:** `POST /classify/batch`

## Free Tier Benefits

- **16GB RAM** - Plenty of space for your fp16 model!
- **Free hosting** for public Spaces
- **Automatic HTTPS**
- **GPU access** (optional, requires upgrade)

## Troubleshooting

- **Build fails:** Check Space logs for errors
- **Model download fails:** Verify `MODEL_URL` is correct and public
- **Out of memory:** Shouldn't happen with 16GB, but check logs

## Local Testing

To test locally before deploying:

```bash
# Install dependencies
pip install -r requirements-api.txt

# Set environment variables
export MODEL_URL="https://github.com/taisiyasolareva/multilabel-news-classifier/releases/download/v0.1.0/distilmbert_lora_10k_v1_fp16.pt"
export MODEL_DTYPE="float16"
export MODEL_PATH="models/distilmbert_lora_10k_v1.pt"
export THRESHOLDS_PATH="config/thresholds.json"
export TOKENIZER_NAME="distilbert-base-multilingual-cased"
export PYTHONPATH="/app"

# Run the app
python app.py
```

Then visit `http://localhost:8000/docs` to test the API.

