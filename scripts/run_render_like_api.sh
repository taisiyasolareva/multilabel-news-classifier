#!/usr/bin/env bash
set -euo pipefail

# Run the API in a Docker container with a 512Mi memory limit (Render free-tier-ish),
# so you can debug OOM/import/startup issues locally with cached layers.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH_HOST="${MODEL_PATH_HOST:-$ROOT_DIR/models/distilmbert_lora_10k_v1.pt}"
THRESHOLDS_HOST="${THRESHOLDS_HOST:-$ROOT_DIR/config/thresholds.json}"

if [[ ! -f "$MODEL_PATH_HOST" ]]; then
  echo "ERROR: Model file not found at: $MODEL_PATH_HOST"
  echo ""
  echo "Download it first (example):"
  echo "  python scripts/download_model.py --model-id distilmbert_lora_10k_v1 \\"
  echo "    --url \"\$MODEL_URL\" --output-path models/distilmbert_lora_10k_v1.pt"
  exit 1
fi

echo "Building API image (cached)…"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
docker build --platform "$DOCKER_PLATFORM" -f Dockerfile.api -t nlp-api-renderlike .

echo "Starting container with 512Mi RAM limit…"
HOST_PORT="${HOST_PORT:-8000}"
docker run --rm \
  --name nlp-api-renderlike \
  --platform "$DOCKER_PLATFORM" \
  -p "${HOST_PORT}:8000" \
  --memory=512m --memory-swap=512m \
  -e PYTHONUNBUFFERED=1 \
  -e MODEL_PATH=models/distilmbert_lora_10k_v1.pt \
  -e THRESHOLDS_PATH=config/thresholds.json \
  -e TOKENIZER_NAME=distilbert-base-multilingual-cased \
  -e MODEL_DTYPE="${MODEL_DTYPE:-float32}" \
  -v "$MODEL_PATH_HOST:/app/models/distilmbert_lora_10k_v1.pt:ro" \
  -v "$THRESHOLDS_HOST:/app/config/thresholds.json:ro" \
  nlp-api-renderlike


