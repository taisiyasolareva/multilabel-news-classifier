# Russian News Tagging (Multi-label) — API + Model Zoo + Dashboards

A multi-label classification system for Russian news articles, built around transformer backbones and a reproducible **model zoo** protocol (fixed 10k train / 1k val split) for fair comparisons.

**What you can review quickly:**
- **FastAPI inference service** (thresholding + model hot-reload + health reporting)
- **Streamlit dashboards** (Evaluation, Analytics, Model Comparison, Sentiment)
- **Model zoo results** with **threshold optimization** (metrics JSON + predictions CSV)

---

## For Portfolio Reviewers (choose your path)

### Level A — 1–2 minutes (no running anything)
- **Results summary**: see `docs/RESULTS.md`
- **System overview**: see `docs/ARCHITECTURE.md`
- **Current served model + threshold**: see `config/thresholds.json`

### Level B — 5 minutes (local demo via Docker)

```bash
docker compose up --build
```

Open:
- **API docs**: `http://localhost:8000/docs`
- **API health** (shows loaded model + thresholds): `http://localhost:8000/health`
- **Streamlit multipage app** (Evaluation + Analytics + Model Comparison + Sentiment): `http://localhost:8501`

### Hosted Demo (Option B) — API + Streamlit (links)

When you deploy, paste the final URLs here:
- **API docs**: `https://<api-host>/docs`
- **API health**: `https://<api-host>/health`
- **Streamlit multipage app** (Evaluation + Analytics + Model Comparison + Sentiment): `https://<your-streamlit-app>`

### Level C — deep dive (reproduce training + evaluation)
- Train/evaluate with the frozen protocol: `experiments/model_zoo/protocol_10k_1k`
- Use:
  - `scripts/train_model.py` (supports LoRA)
  - `scripts/evaluate.py` (metrics + threshold optimization + predictions CSV)

Copy/paste walkthrough: see `docs/DEMO.md`

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [API](#api)
- [Monitoring](#monitoring)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a multi-label classification system for Russian news articles, predicting relevant tags (topics) for each article. The system uses state-of-the-art transformer models (Russian BERT) fine-tuned on a dataset of Russian news articles.

### Key Capabilities

- **Multi-label Classification**: Predicts multiple tags per article
- **Russian Language Support**: Optimized for Russian text using `DeepPavlov/rubert-base-cased`
- **Production Ready**: FastAPI REST API with async inference
- **MLOps**: Complete CI/CD, monitoring, and deployment pipeline
- **Experiment Tracking**: WandB, MLflow, and DVC integration
- **Hyperparameter Tuning**: Optuna and WandB sweeps

### Performance

- **Baseline (Simple Embeddings)**: F1 ~0.30-0.40
- **Russian BERT (Fine-tuned)**: F1 ~0.50-0.65
- **Ensemble Models**: F1 ~0.55-0.70

---

## ✨ Features

### Model Architectures
- ✅ **Simple Classifier**: Baseline embedding-based model
- ✅ **CNN Classifier**: Convolutional neural network
- ✅ **Russian BERT**: Fine-tuned `DeepPavlov/rubert-base-cased`
- ✅ **Multilingual BERT**: Comparison model
- ✅ **RoBERTa**: XLM-RoBERTa variant
- ✅ **DistilBERT**: Faster, smaller model
- ✅ **Multi-Head Attention**: Attention pooling classifier
- ✅ **Ensemble Methods**: Weighted, stacking, voting

### ML Engineering
- ✅ **Configuration Management**: Hydra + YAML configs
- ✅ **Experiment Tracking**: WandB, MLflow, DVC
- ✅ **Hyperparameter Tuning**: Optuna, WandB sweeps
- ✅ **Model Registry**: MLflow model versioning
- ✅ **Data Versioning**: DVC pipeline

### Production Features
- ✅ **REST API**: FastAPI with async inference
- ✅ **Containerization**: Docker with multi-stage builds
- ✅ **CI/CD**: GitHub Actions workflows
- ✅ **Monitoring**: Performance, drift detection, logging
- ✅ **Testing**: 50+ comprehensive tests

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Applications                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/REST
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    FastAPI REST API                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Monitoring Middleware                                │  │
│  │  - Prediction Logging                                  │  │
│  │  - Data Drift Detection                               │  │
│  │  - Performance Monitoring                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Inference Engine                                     │  │
│  │  - Async Processing                                   │  │
│  │  - Batch Support                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Model Loading
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Transformer Models                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Russian BERT / RoBERTa / DistilBERT                  │  │
│  │  - Pre-trained Embeddings                             │  │
│  │  - Fine-tuned Classifier Head                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input: Title + Snippet (Russian Text)
    │
    ├─► Tokenization (HuggingFace Tokenizer)
    │   └─► Subword Tokenization (WordPiece)
    │
    ├─► Russian BERT Encoder
    │   ├─► 12 Transformer Layers
    │   ├─► 768 Hidden Dimensions
    │   └─► Multi-Head Attention
    │
    ├─► Pooling ([CLS] token or Attention Pooling)
    │   └─► [batch_size, 768]
    │
    ├─► Classification Head
    │   ├─► Dropout (0.3)
    │   ├─► Linear(768 → 768) + ReLU
    │   ├─► Dropout (0.3)
    │   └─► Linear(768 → num_labels)
    │
    └─► Output: Multi-label Predictions
        └─► [batch_size, num_labels] with logits
```

### Data Pipeline

```
Raw Data (TSV)
    │
    ├─► Data Loading
    │   └─► Filter nulls, validate
    │
    ├─► Text Preprocessing
    │   ├─► Normalization (lowercase, punctuation)
    │   ├─► Russian text handling
    │   └─► Tokenization
    │
    ├─► Label Processing
    │   ├─► Tag frequency filtering
    │   ├─► Label mapping
    │   └─► Multi-hot encoding
    │
    ├─► Data Splitting
    │   ├─► Train (date-based)
    │   ├─► Validation (date-based)
    │   └─► Test (date-based + href exclusion)
    │
    └─► Dataset Creation
        └─► PyTorch Dataset / Transformer Dataset
```

### Training Pipeline

```
Configuration (Hydra)
    │
    ├─► Model Initialization
    │   └─► Load pre-trained BERT
    │
    ├─► Data Loading
    │   └─► DataLoader with batching
    │
    ├─► Training Loop (PyTorch Lightning)
    │   ├─► Forward pass
    │   ├─► Loss calculation (BCEWithLogitsLoss)
    │   ├─► Backward pass
    │   ├─► Optimizer step (AdamW)
    │   └─► LR scheduling (warmup + decay)
    │
    ├─► Validation
    │   ├─► Metrics calculation
    │   └─► Early stopping
    │
    ├─► Experiment Tracking
    │   ├─► WandB logging
    │   ├─► MLflow tracking
    │   └─► Model checkpointing
    │
    └─► Model Registry
        └─► MLflow model registration
```

### Deployment Architecture

```
GitHub Repository
    │
    ├─► CI/CD Pipeline (GitHub Actions)
    │   ├─► Automated Testing
    │   ├─► Docker Build
    │   └─► Security Scanning
    │
    ├─► Container Registry (GHCR)
    │   └─► Docker Images
    │
    ├─► Deployment
    │   ├─► Staging Environment
    │   └─► Production Environment
    │
    └─► Monitoring
        ├─► Performance Metrics
        ├─► Data Drift Detection
        └─► Prediction Logging
```

---

## 📦 Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM
- 10GB+ disk space (for models and data)

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/username/russian-news-classification.git
cd russian-news-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
```

### Option 2: Docker Installation

```bash
# Build Docker image
docker build --target production -t news-classifier-api:latest .

# Or use docker-compose
docker-compose up --build
```

### Option 3: Development Setup

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Verify Installation

```bash
# Test imports
python -c "from models.transformer_model import RussianNewsClassifier; print('OK')"

# Run tests
pytest tests/ -v

# Check API
python scripts/start_api.py &
curl http://localhost:8000/health
```

---

## 🚀 Quick Start

### 1. Prepare Data

```bash
# Place your data files in data/news_data/
# Required: ria_news.tsv
# Optional: vk_news.tsv, vk_comments.tsv
```

### 2. Generate Reference Statistics (for drift detection)

```bash
python scripts/generate_reference_stats.py \
    --data-path data/news_data/ria_news.tsv \
    --output monitoring/reference_stats.json
```

### 3. Train Model

```bash
# Using Hydra (recommended)
python training/train_with_hydra.py

# Or with custom config
python training/train_with_hydra.py \
    training.epochs=10 \
    training.batch_size=32 \
    model.transformer.model_name="xlm-roberta-base"
```

### 4. Start API

```bash
# Development
python scripts/start_api.py --reload

# Production
docker-compose -f docker-compose.prod.yml up -d
```

### 5. Test API

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Путин объявил о новых мерах поддержки экономики",
    "snippet": "Президент России объявил о новых мерах...",
    "threshold": 0.5,
    "top_k": 5
  }'
```

---

## 💡 Usage Examples

### Training a Model

#### Basic Training
```python
from training.train_with_hydra import train_with_hydra

# Train with default config
train_with_hydra()
```

#### Custom Configuration
```python
# Override config values
python training/train_with_hydra.py \
    training.epochs=10 \
    training.batch_size=32 \
    training.optimizer.learning_rate=2e-5 \
    model.transformer.dropout=0.5
```

#### Hyperparameter Tuning
```bash
# Optuna
python training/tune_hyperparameters.py \
    --method optuna \
    --n-trials 50

# WandB Sweep
wandb sweep config/wandb_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

### Using the API

#### Single Classification
```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={
        "title": "Тестовая новость",
        "snippet": "Описание новости",
        "threshold": 0.5,
        "top_k": 5
    }
)

result = response.json()
print(f"Predicted tags: {[p['tag'] for p in result['predictions']]}")
```

#### Batch Classification
```python
response = requests.post(
    "http://localhost:8000/classify/batch",
    json={
        "items": [
            {"title": "Новость 1"},
            {"title": "Новость 2", "snippet": "Описание"}
        ]
    }
)

for result in response.json()["results"]:
    print(f"Tags: {[p['tag'] for p in result['predictions']]}")
```

#### Async Client
```python
import httpx
import asyncio

async def classify_async():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/classify",
            json={"title": "Тестовая новость"}
        )
        return response.json()

result = asyncio.run(classify_async())
```

### Monitoring

#### View Dashboard
```bash
python scripts/monitoring_dashboard.py
```

#### Check Performance Metrics
```bash
curl http://localhost:8000/monitoring/performance
```

#### Check Data Drift
```bash
curl http://localhost:8000/monitoring/drift
```

#### Get Prediction Logs
```bash
curl http://localhost:8000/monitoring/predictions?limit=100
```

### Model Deployment

#### Register Model in MLflow
```bash
python scripts/register_model_mlflow.py \
    --model-path models/best_model.pt \
    --model-name news-classifier
```

#### Deploy via CI/CD
1. Push model to `models/` directory
2. GitHub Actions will automatically:
   - Validate model
   - Build Docker image
   - Deploy to staging
   - Run integration tests
   - Deploy to production (if approved)

---

## 📁 Project Structure

```
nlp/
├── api/                          # FastAPI application
│   ├── main.py                  # Main API application
│   ├── schemas.py               # Pydantic models
│   ├── inference.py             # Inference utilities
│   ├── monitoring_middleware.py # Monitoring middleware
│   └── monitoring_endpoints.py  # Monitoring API
│
├── config/                       # Configuration files
│   ├── config.yaml              # Main config
│   ├── model/                    # Model configs
│   ├── training/                 # Training configs
│   ├── data/                     # Data configs
│   └── logging/                  # Logging configs
│
├── data/                         # Data processing
│   ├── data_loader.py           # Data loading
│   ├── dataset.py                # PyTorch Dataset
│   └── transformer_dataset.py   # Transformer Dataset
│
├── models/                       # Model architectures
│   ├── simple_classifier.py     # Baseline model
│   ├── cnn_classifier.py        # CNN model
│   ├── transformer_model.py     # BERT models
│   ├── advanced_transformers.py # Advanced architectures
│   ├── ensemble.py              # Ensemble methods
│   └── lightning_module.py      # PyTorch Lightning
│
├── training/                     # Training scripts
│   ├── train_with_hydra.py      # Hydra training
│   ├── train_transformer.py     # Transformer training
│   ├── tune_hyperparameters.py  # Hyperparameter tuning
│   └── train_ensemble.py        # Ensemble training
│
├── monitoring/                   # Monitoring utilities
│   ├── performance_monitor.py   # Performance tracking
│   ├── data_drift.py            # Drift detection
│   └── prediction_logger.py    # Prediction logging
│
├── utils/                        # Utilities
│   ├── text_processing.py       # Text preprocessing
│   ├── data_processing.py      # Data utilities
│   ├── tokenization.py         # Tokenization
│   ├── config_manager.py       # Config management
│   └── experiment_tracking.py  # Experiment tracking
│
├── tests/                        # Test suite
│   ├── test_data_pipeline.py    # Data tests
│   ├── test_models.py          # Model tests
│   ├── test_training_integration.py # Training tests
│   └── test_api.py             # API tests
│
├── scripts/                      # Utility scripts
│   ├── start_api.py             # API startup
│   ├── generate_reference_stats.py # Drift detection setup
│   └── monitoring_dashboard.py # Monitoring dashboard
│
├── .github/workflows/            # CI/CD workflows
│   ├── ci.yml                   # Continuous Integration
│   ├── cd.yml                   # Continuous Deployment
│   └── model-deploy.yml         # Model deployment
│
├── Dockerfile                    # Production Dockerfile
├── docker-compose.yml            # Development compose
├── docker-compose.prod.yml       # Production compose
├── requirements.txt             # Dependencies
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Main Configuration

Edit `config/config.yaml`:

```yaml
# Model configuration
model:
  transformer:
    model_name: "DeepPavlov/rubert-base-cased"
    dropout: 0.3
    use_snippet: true

# Training configuration
training:
  epochs: 5
  batch_size: 16
  optimizer:
    learning_rate: 2e-5
    weight_decay: 0.01

# Data configuration
data:
  paths:
    train_path: "data/news_data/ria_news.tsv"
  preprocessing:
    min_tag_frequency: 30
```

### Environment Variables

Create `.env` file (see `config/env.example`):

```bash
# WandB
WANDB_API_KEY=your_key_here

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns

# API
MODEL_PATH=models/best_model.pt
```

### Threshold Configuration

Custom thresholds can be configured in `config/thresholds.json`:

```json
{
  "global_threshold": 0.5,
  "per_class_thresholds": {
    "политика": 0.6,
    "экономика": 0.4
  },
  "model_version": "best_model_v3.pt"
}
```

- **global_threshold**: Default threshold for all classes (used when request threshold is 0.5)
- **per_class_thresholds**: Class-specific thresholds (override global threshold)
- The API automatically loads this configuration on startup

---

## 🎓 Training

### Basic Training

```bash
# Using the training script (recommended for production)
python scripts/train_model.py \
    --data-path data/news_data/ria_news.tsv \
    --output-path models/best_model.pt \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --use-snippet false

# Using Hydra (alternative)
python training/train_with_hydra.py

# With config overrides
python training/train_with_hydra.py \
    training.epochs=10 \
    training.batch_size=32
```

### Scaling Up Training

After validating with a tiny dataset (100 train / 10 val samples), scale up gradually:

#### Step 1: Small Scale (1k train / 100 val)
```bash
python scripts/train_model.py \
    --data-path data/news_data/ria_news.tsv \
    --output-path models/best_model_v1.pt \
    --epochs 3 \
    --batch-size 16 \
    --max-train-samples 1000 \
    --max-val-samples 100
```

#### Step 2: Medium Scale (10k train / 1k val)
```bash
python scripts/train_model.py \
    --data-path data/news_data/ria_news.tsv \
    --output-path models/best_model_v2.pt \
    --epochs 5 \
    --batch-size 16 \
    --max-train-samples 10000 \
    --max-val-samples 1000
```

#### Step 3: Full Dataset
```bash
python scripts/train_model.py \
    --data-path data/news_data/ria_news.tsv \
    --output-path models/best_model_v3.pt \
    --epochs 10 \
    --batch-size 16
```

**Note**: By default, `use_snippet=False` (title-only). To use snippets, add `--use-snippet true`.

**Note**: Remove `--max-train-samples` and `--max-val-samples` flags for full dataset training.

### Threshold Optimization

After training, optimize thresholds using the evaluation dashboard:

1. **Generate predictions**:
```bash
python scripts/evaluate.py \
    --checkpoint models/best_model_v3.pt \
    --data-path data/news_data/ria_news.tsv \
    --threshold 0.5 \
    --output-csv experiments/full_eval_predictions.csv
```

2. **Use Streamlit dashboard** to find optimal thresholds:
   - Upload `experiments/full_eval_predictions.csv` to the evaluation dashboard
   - Navigate to "Threshold Optimization" tab
   - Select global or per-class threshold optimization
   - Export optimal thresholds to `config/thresholds.json`

3. **Update API configuration**:
   - The API will automatically load thresholds from `config/thresholds.json` if available
   - Or set custom threshold per request via the `threshold` parameter

---

## 📊 Preparing Dashboard Data

All Streamlit dashboards require specific CSV files. Use the master script to generate all data at once:

### Quick Start (All Dashboards)

```bash
# Generate ALL dashboard data files
python scripts/prepare_all_dashboard_data.py \
    --checkpoint models/best_model_v2.pt \
    --max-val-samples 1000 \
    --max-samples 5000 \
    --skip-sentiment  # Skip slow sentiment analysis for quick test
```

This generates:
- ✅ **Evaluation Dashboard**: `experiments/dashboard_eval_predictions.csv`
- ✅ **Analytics Dashboard - Category**: `experiments/analytics_category_data.csv`
- ✅ **Analytics Dashboard - Thread**: `experiments/analytics_thread_data.csv`
- ✅ **Analytics Dashboard - Predictive Intervals**: `experiments/analytics_sentiment_counts.csv` (if not skipped)

### Full Dataset (Including Sentiment Analysis)

```bash
# Full dataset with sentiment analysis (slow!)
python scripts/prepare_all_dashboard_data.py \
    --checkpoint models/best_model_v2.pt \
    --max-news-items 100 \
    --max-comments-per-item 2000
```

### Individual Dashboard Data

**Evaluation Dashboard:**
```bash
python scripts/evaluate.py \
    --checkpoint models/best_model_v2.pt \
    --data-path data/news_data/ria_news.tsv \
    --max-val-samples 1000 \
    --output-csv experiments/dashboard_eval_predictions.csv
```

**Analytics Dashboard:**
```bash
# Category Analytics + Thread Analysis (fast)
python scripts/prepare_analytics_data.py \
    --max-samples 5000 \
    --skip-sentiment

# Predictive Intervals (slow - sentiment analysis)
python scripts/prepare_analytics_data.py \
    --max-news-items 50 \
    --max-comments-per-item 1000
```

### Dashboard CSV Requirements

| Dashboard | Tab | Required CSV Format | Columns |
|-----------|-----|---------------------|---------|
| **Evaluation** | All | Predictions | `sample_id`, `class_0`, `class_1`, ..., `target_class_0`, `target_class_1`, ... |
| **Analytics** | Category Analytics | Category data | `category`, `text` |
| **Analytics** | Thread Analysis | Comments | `news_id`, `text` |
| **Analytics** | Predictive Intervals | Sentiment counts | `id`, `positive_count`, `negative_count`, `neutral_count` |
| **Sentiment** | All | None | Uses FastAPI endpoint |
| **Model Comparison** | All | Experiment results | Uses experiment tracker or upload CSV manually |

### Model Versioning

Models are saved with version suffixes (e.g., `best_model_v1.pt`, `best_model_v2.pt`). The API can be configured to use a specific version:

```bash
# Start API with specific model version
MODEL_PATH=models/best_model_v3.pt uvicorn api.main:app --reload
```

The `/health` endpoint reports the loaded model version and configuration.

### Hyperparameter Tuning

```bash
# Optuna
python training/tune_hyperparameters.py \
    --method optuna \
    --n-trials 50 \
    --study-name russian-news-classification

# WandB Sweep
wandb sweep config/wandb_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

### Training Different Models

```python
# Russian BERT
python training/train_with_hydra.py \
    model.transformer.architecture=russian_bert

# RoBERTa
python training/train_with_hydra.py \
    model.transformer.architecture=roberta

# DistilBERT
python training/train_with_hydra.py \
    model.transformer.architecture=distilbert
```

---

## 🌐 API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/classify` | POST | Single classification |
| `/classify/batch` | POST | Batch classification |
| `/model/reload` | POST | Reload model |
| `/monitoring/performance` | GET | Performance metrics |
| `/monitoring/drift` | GET | Data drift status |
| `/monitoring/predictions` | GET | Prediction logs |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

### API Documentation

Start the API and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

#### Single Classification
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Путин объявил о новых мерах",
    "threshold": 0.5,
    "top_k": 5
  }'
```

#### Batch Classification
```bash
curl -X POST "http://localhost:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"title": "Новость 1"},
      {"title": "Новость 2"}
    ]
  }'
```

---

## 📊 Monitoring

### Performance Monitoring

Tracks model performance over time:
- Precision, Recall, F1 Score
- Exact Match Rate
- Performance degradation alerts

### Data Drift Detection

Monitors input data distribution:
- Title/snippet length distributions
- Word count statistics
- Cyrillic character ratios
- Statistical tests (KS test)

### Prediction Logging

Logs all predictions for analysis:
- Input data
- Predictions and scores
- Metadata (latency, model version)
- Daily log rotation

### Monitoring Dashboard

```bash
python scripts/monitoring_dashboard.py
```

Output:
```
================================================================================
MONITORING DASHBOARD
================================================================================
Generated: 2024-01-15 10:30:00

PERFORMANCE METRICS
--------------------------------------------------------------------------------
Recent Metrics (last 100 predictions):
  Precision: 0.702
  Recall:    0.658
  F1 Score:  0.679
  Exact Match: 0.450
  Count:     100

DATA DRIFT DETECTION
--------------------------------------------------------------------------------
✓ No drift detected

PREDICTION LOGS
--------------------------------------------------------------------------------
Total Predictions (7 days): 1500
Unique Titles: 1200
Average Latency: 45.2 ms
```

---

## 🚢 Deployment

### Option B (recommended for portfolio): Hosted API + Streamlit dashboards

**Architecture**:
- FastAPI hosted on Render/Railway/Fly
- Streamlit dashboards hosted on Streamlit Community Cloud, calling the API via `API_URL`

#### 1) Publish model weights (recommended: GitHub Release asset)
Don’t commit `.pt` files to git history. Instead:
- Create a GitHub Release and upload `distilmbert_lora_10k_v1.pt` as an asset
- Copy the asset URL and set it as `MODEL_URL` in your API host

#### 2) Deploy API (Render)
This repo includes a `render.yaml` blueprint. In Render:
- Create a new **Web Service** from your repo
- Set environment variables:
  - `MODEL_URL` = the direct `.pt` download URL (Release asset)
  - `MODEL_PATH=models/distilmbert_lora_10k_v1.pt`
  - `THRESHOLDS_PATH=config/thresholds.json`
  - `TOKENIZER_NAME=distilbert-base-multilingual-cased`
  - `CORS_ALLOW_ORIGINS=https://<your-streamlit-app-domain>` (comma-separated allowed origins)
- Verify:
  - `https://<api-host>/health`
  - `https://<api-host>/docs`

#### 3) Deploy dashboards (Streamlit Community Cloud)
Create **one multipage** Streamlit app:
- entrypoint: `streamlit_app.py`
- pages: `pages/` (Evaluation, Analytics, Model Comparison, Sentiment)

Set Streamlit secrets/env:
- `API_URL=https://<api-host>`

#### Notes
- On macOS, local Docker runs PyTorch on CPU; hosted CPU is expected unless you pay for GPU hosting.

### Docker Deployment

```bash
# Build production image
docker build --target production -t news-classifier-api:latest .

# Run container
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    news-classifier-api:latest
```

### Docker Compose

```bash
# Development
docker compose up --build

# Production
docker compose up --build -d
```

### Kubernetes

```yaml
# Example deployment (k8s/deployment.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: news-classifier-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: ghcr.io/username/repo:latest
        ports:
        - containerPort: 8000
```

### Cloud Platforms

#### AWS (ECS/Fargate)
```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag news-classifier-api:latest <account>.dkr.ecr.<region>.amazonaws.com/news-classifier:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/news-classifier:latest
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/<project>/news-classifier
gcloud run deploy news-classifier \
    --image gcr.io/<project>/news-classifier \
    --platform managed \
    --region us-central1
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test
pytest tests/test_models.py::TestSimpleClassifier::test_forward_title_only -v
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=. --cov-report=term-missing

# HTML report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## 📈 Experiment Tracking

### WandB

```python
# Automatic logging with PyTorch Lightning
from models.lightning_module_tracking import create_tracking_loggers

loggers, callbacks = create_tracking_loggers(
    use_wandb=True,
    project_name="russian-news-classification"
)
```

### MLflow

```python
# Model registry
from utils.experiment_tracking import MLflowTracker

with MLflowTracker() as tracker:
    tracker.log_model(model, artifact_path="model")
    tracker.register_model("runs:/<run_id>/model", "news-classifier")
```

### DVC

```bash
# Track data
dvc add data/raw/ria_news.tsv

# Reproduce pipeline
dvc repro

# Push to remote
dvc push
```

---

## 🔧 Development

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .
pylint api/ models/ utils/

# Type checking
mypy api/ models/ utils/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

---

## 📚 Documentation

- **Architecture**: `docs/ARCHITECTURE.md`
- **Demo walkthrough**: `docs/DEMO.md`
- **Results summary**: `docs/RESULTS.md`
- **Project notes (dev history + remaining polish gaps)**: `docs/PROJECT_NOTES.md`
- **Portfolio polish plan**: `PORTFOLIO_REVIEWER_POLISH_PLAN.md`
- **API docs (local)**: `http://localhost:8000/docs`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for all classes and functions
- Add tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [DeepPavlov](https://github.com/deepmipt/DeepPavlov) for Russian BERT model
- [HuggingFace](https://huggingface.co/) for transformers library
- [PyTorch Lightning](https://lightning.ai/) for training framework
- [FastAPI](https://fastapi.tiangolo.com/) for API framework

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: 2024
