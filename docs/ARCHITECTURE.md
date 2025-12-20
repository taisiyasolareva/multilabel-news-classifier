# Architecture Documentation

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│  (Web Apps, Mobile Apps, Other Services)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ HTTP/REST API
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  API Gateway Layer                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FastAPI Application                                  │  │
│  │  - Request Validation                                 │  │
│  │  - Authentication (optional)                          │  │
│  │  - Rate Limiting                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Application Layer                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Monitoring Middleware                                │  │
│  │  - Prediction Logging                                 │  │
│  │  - Data Drift Detection                              │  │
│  │  - Performance Tracking                               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Inference Engine                                     │  │
│  │  - Async Processing                                   │  │
│  │  - Batch Handling                                     │  │
│  │  - Error Handling                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Model Calls
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Model Layer                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Transformer Models                                   │  │
│  │  - Russian BERT                                      │  │
│  │  - RoBERTa                                           │  │
│  │  - DistilBERT                                        │  │
│  │  - Ensemble Models                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Data Layer                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Tokenization                                         │  │
│  │  - HuggingFace Tokenizers                             │  │
│  │  - Subword Tokenization                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture Details

### Transformer Model Flow

```
Input Text: "Путин объявил о новых мерах поддержки экономики"
    │
    ├─► Text Preprocessing
    │   └─► Normalize: "путин объявил о новых мерах поддержки экономики"
    │
    ├─► Tokenization (HuggingFace)
    │   └─► Tokens: ["[CLS]", "путин", "объявил", "о", "новых", "мерах", ...]
    │   └─► Token IDs: [101, 1234, 5678, ...]
    │
    ├─► Embedding Layer
    │   └─► [batch, seq_len, 768]
    │
    ├─► BERT Encoder (12 layers)
    │   ├─► Multi-Head Self-Attention (12 heads)
    │   ├─► Feed-Forward Network
    │   ├─► Layer Normalization
    │   └─► Residual Connections
    │   └─► Output: [batch, seq_len, 768]
    │
    ├─► Pooling
    │   └─► [CLS] token or Attention Pooling
    │   └─► [batch, 768]
    │
    ├─► Classification Head
    │   ├─► Dropout(0.3)
    │   ├─► Linear(768 → 768) + ReLU
    │   ├─► Dropout(0.3)
    │   └─► Linear(768 → num_labels)
    │   └─► Output: [batch, num_labels]
    │
    ├─► Sigmoid Activation
    │   └─► Probabilities: [batch, num_labels]
    │
    └─► Threshold Filtering (0.5)
        └─► Final Tags: ["политика", "экономика"]
```

### Ensemble Architecture

```
Input: Title + Snippet
    │
    ├─► Model 1 (Russian BERT)
    │   └─► Predictions: [0.9, 0.7, 0.3, ...]
    │
    ├─► Model 2 (RoBERTa)
    │   └─► Predictions: [0.85, 0.75, 0.4, ...]
    │
    ├─► Model 3 (DistilBERT)
    │   └─► Predictions: [0.88, 0.72, 0.35, ...]
    │
    └─► Ensemble Combination
        ├─► Weighted Average (weights: [0.4, 0.3, 0.3])
        └─► Final Predictions: [0.88, 0.73, 0.35, ...]
```

## Data Flow

### Training Data Flow

```
Raw TSV Files
    │
    ├─► Load Data (pandas)
    │   └─► Filter nulls
    │
    ├─► Text Preprocessing
    │   ├─► Normalize text
    │   ├─► Lowercase
    │   └─► Remove special chars
    │
    ├─► Tag Processing
    │   ├─► Split tags
    │   ├─► Filter by frequency
    │   └─► Create label mapping
    │
    ├─► Data Splitting
    │   ├─► Train (dates < 2018-10-01)
    │   ├─► Validation (2018-10-01 to 2018-12-01)
    │   └─► Test (dates >= 2018-12-01)
    │
    ├─► Dataset Creation
    │   ├─► Tokenization
    │   ├─► Padding/Truncation
    │   └─► Multi-hot encoding
    │
    └─► DataLoader
        └─► Batches for training
```

### Inference Data Flow

```
API Request
    │
    ├─► Request Validation (Pydantic)
    │   └─► Validate title, snippet, threshold
    │
    ├─► Text Preprocessing
    │   └─► Normalize and clean
    │
    ├─► Tokenization
    │   └─► Convert to token IDs
    │
    ├─► Model Inference
    │   └─► Forward pass through BERT
    │
    ├─► Post-processing
    │   ├─► Sigmoid activation
    │   ├─► Threshold filtering
    │   └─► Top-K selection
    │
    ├─► Monitoring
    │   ├─► Log prediction
    │   ├─► Record for drift detection
    │   └─► Track performance
    │
    └─► Response
        └─► JSON with predictions
```

## Component Interactions

### Training Pipeline

```
Config (Hydra)
    │
    ├─► Data Loading
    │   └─► Dataset Creation
    │
    ├─► Model Initialization
    │   └─► Load Pre-trained BERT
    │
    ├─► Training Loop
    │   ├─► Forward Pass
    │   ├─► Loss Calculation
    │   ├─► Backward Pass
    │   └─► Optimizer Step
    │
    ├─► Validation
    │   └─► Metrics Calculation
    │
    ├─► Experiment Tracking
    │   ├─► WandB Logging
    │   ├─► MLflow Tracking
    │   └─► DVC Versioning
    │
    └─► Model Checkpointing
        └─► Save Best Model
```

### API Request Flow

```
HTTP Request
    │
    ├─► CORS Middleware
    │
    ├─► Monitoring Middleware
    │   └─► Start timer
    │
    ├─► Request Validation
    │   └─► Pydantic validation
    │
    ├─► Inference
    │   ├─► Text preprocessing
    │   ├─► Tokenization
    │   ├─► Model forward pass
    │   └─► Post-processing
    │
    ├─► Monitoring
    │   ├─► Log prediction
    │   ├─► Check drift
    │   └─► Update metrics
    │
    └─► HTTP Response
        └─► JSON with predictions
```

## Technology Stack

### Core ML
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training framework
- **Transformers**: HuggingFace transformers library
- **Russian BERT**: DeepPavlov/rubert-base-cased

### API & Web
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### MLOps
- **WandB**: Experiment tracking
- **MLflow**: Model registry
- **DVC**: Data versioning
- **Optuna**: Hyperparameter tuning
- **Hydra**: Configuration management

### Infrastructure
- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **Nginx**: Reverse proxy (optional)

### Monitoring
- **Custom Monitoring**: Performance, drift, logging
- **Prometheus** (optional): Metrics collection
- **Grafana** (optional): Visualization

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Load balancer support
- Multiple worker processes
- Container orchestration (Kubernetes)

### Performance Optimization
- Async inference
- Batch processing
- Model quantization (future)
- GPU acceleration
- Caching (future)

### High Availability
- Health checks
- Graceful degradation
- Circuit breakers (future)
- Retry mechanisms




