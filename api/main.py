"""FastAPI application for Russian news classification.

Notes:
- Supports configuring model + thresholds via environment variables:
  - MODEL_PATH: path to `.pt` checkpoint
  - THRESHOLDS_PATH: path to thresholds JSON (optional)
- `/classify` accepts `title` (canonical) or `text` (alias for convenience).
"""

import logging
import asyncio
import json
import hashlib
import inspect
import gc
import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, root_validator
import uvicorn

from models.transformer_model import RussianNewsClassifier
from utils.tokenization import create_tokenizer
from utils.russian_text_utils import prepare_text_for_tokenization
from monitoring.prediction_logger import PredictionLogger
from monitoring.data_drift import DataDriftDetector
from monitoring.performance_monitor import PerformanceMonitor
from api.monitoring_middleware import MonitoringMiddleware
from api.monitoring_endpoints import router as monitoring_router
from api.sentiment_endpoints import router as sentiment_router
from api.analytics_endpoints import router as analytics_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Russian News Classification API",
    description="Multi-label news tag classification using transformer models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
def _cors_allow_origins() -> list[str]:
    """
    Allow configuring CORS in production without code changes.

    - Default: "*" (open)
    - Set `CORS_ALLOW_ORIGINS` to a comma-separated list:
        CORS_ALLOW_ORIGINS=https://my-app.streamlit.app,https://my-portfolio.com
    """
    raw = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (at module level so they're available for tests)
app.include_router(monitoring_router)
app.include_router(sentiment_router)
app.include_router(analytics_router)

# Global model and tokenizer (loaded on startup)
model = None
tokenizer = None
tag_to_idx = None  # Tag mapping loaded from checkpoint

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _pick_device()
model_loaded = False
model_path = None  # Track loaded model path for versioning
threshold_config = None  # Custom thresholds from config file
thresholds_path = None  # Track thresholds path for debugging/versioning

# Model dtype (can help memory on small instances if you provide an fp16 checkpoint)
def _get_model_dtype() -> torch.dtype:
    raw = os.environ.get("MODEL_DTYPE", "float32").strip().lower()
    if raw in {"float16", "fp16", "half"}:
        return torch.float16
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32

# Monitoring components (initialized at import time so middleware can be added before startup)
prediction_logger = PredictionLogger(log_dir="monitoring/predictions")
performance_monitor = PerformanceMonitor(metrics_file="monitoring/performance_metrics.json")
drift_detector = DataDriftDetector()

# Load reference statistics for drift detector if available
ref_stats_path = Path("monitoring/reference_stats.json")
if ref_stats_path.exists():
    drift_detector.load_reference_stats(str(ref_stats_path))

def _resolve_path(path_str: str) -> Path:
    """
    Resolve path relative to project root unless already absolute.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent.parent / p).resolve()


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _load_thresholds_from_file(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else None
    except Exception as e:
        logger.warning(f"Failed to load threshold config from {path}: {e}")
        return None

# Add monitoring middleware at application setup time (required by Starlette/FastAPI)
app.add_middleware(
    MonitoringMiddleware,
    prediction_logger=prediction_logger,
    drift_detector=drift_detector,
    performance_monitor=performance_monitor,
)


# Request/Response Models
class ClassificationRequest(BaseModel):
    """Request model for classification."""
    
    title: str = Field(..., description="News article title", min_length=1, max_length=500)
    # Convenience alias (allows curl payloads with {"text": "..."}). We map it to title if title is missing.
    text: Optional[str] = Field(None, description="Alias for title (optional)", max_length=500)
    snippet: Optional[str] = Field(None, description="News article snippet", max_length=2000)
    threshold: float = Field(0.5, description="Classification threshold", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Return top K predictions", ge=1, le=100)

    @root_validator(pre=True)
    def _coerce_text_to_title(cls, values):
        # If caller provided "text" but not "title", treat it as title.
        if isinstance(values, dict):
            if not values.get("title") and values.get("text"):
                values["title"] = values["text"]
        return values
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()
    
    @validator('snippet')
    def validate_snippet(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Путин объявил о новых мерах поддержки экономики",
                "snippet": "Президент России объявил о новых мерах поддержки экономики в условиях санкций.",
                "threshold": 0.5,
                "top_k": 5
            }
        }


class TagPrediction(BaseModel):
    """Single tag prediction."""
    
    tag: str = Field(..., description="Tag name")
    score: float = Field(..., description="Prediction score", ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Response model for classification."""
    
    predictions: List[TagPrediction] = Field(..., description="List of tag predictions")
    title: str = Field(..., description="Processed title")
    snippet: Optional[str] = Field(None, description="Processed snippet")
    threshold: float = Field(..., description="Threshold used")
    model_version: str = Field(..., description="Model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"tag": "политика", "score": 0.95},
                    {"tag": "экономика", "score": 0.87}
                ],
                "title": "Путин объявил о новых мерах поддержки экономики",
                "snippet": "Президент России объявил о новых мерах поддержки экономики...",
                "threshold": 0.5,
                "model_version": "1.0.0"
            }
        }


class BatchClassificationRequest(BaseModel):
    """Request model for batch classification."""
    
    items: List[ClassificationRequest] = Field(..., description="List of classification requests", min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "title": "Новость 1",
                        "snippet": "Описание 1",
                        "threshold": 0.5
                    },
                    {
                        "title": "Новость 2",
                        "threshold": 0.5
                    }
                ]
            }
        }


class BatchClassificationResponse(BaseModel):
    """Response model for batch classification."""
    
    results: List[ClassificationResponse] = Field(..., description="List of classification results")
    total: int = Field(..., description="Total number of items processed")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device used for inference")
    model_version: Optional[str] = Field(None, description="Model version")
    model_path: Optional[str] = Field(None, description="Resolved model path")
    global_threshold: Optional[float] = Field(None, description="Global threshold from thresholds config (if loaded)")
    thresholds_path: Optional[str] = Field(None, description="Resolved thresholds path (if any)")
    thresholds_sha256: Optional[str] = Field(None, description="SHA256 of thresholds file (if any)")


# Model Loading
async def load_model(
    model_path: str = "models/best_model.pt",
    tokenizer_name: Optional[str] = None,
) -> None:
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_name: HuggingFace tokenizer name
    """
    global model, tokenizer, model_loaded, tag_to_idx
    
    try:
        logger.info(f"Loading model from {model_path}")

        # Load model checkpoint first (so we can infer tokenizer/model name from metadata).
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        # Reduce peak RSS for large checkpoints if supported by the installed torch.
        # (Render free tier is 512Mi; mmap can help avoid transient double-buffering.)
        try:
            sig = inspect.signature(torch.load)
            if "mmap" in sig.parameters:
                load_kwargs["mmap"] = True
            # NOTE: we intentionally do NOT force weights_only=True here because
            # some checkpoints store small metadata dicts; weights_only can reject them.
        except Exception:
            pass

        checkpoint = torch.load(model_path, **load_kwargs)

        # Infer tokenizer name from checkpoint if not provided.
        inferred_name = None
        if isinstance(checkpoint, dict):
            inferred_name = checkpoint.get("model_name")
        tokenizer_name_eff = tokenizer_name or inferred_name or "DeepPavlov/rubert-base-cased"

        # Load tokenizer
        tokenizer = create_tokenizer(tokenizer_name_eff)
        logger.info(f"Tokenizer loaded: {tokenizer_name_eff}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Reconstruct model from state dict
                num_labels = checkpoint.get('num_labels', 1000)
                use_snippet = checkpoint.get('use_snippet', False)  # Default to False for title-only
                tag_to_idx = checkpoint.get('tag_to_idx', None)  # Load tag mapping if available
                model_dtype = _get_model_dtype()
                model = RussianNewsClassifier(
                    model_name=tokenizer_name_eff,
                    num_labels=num_labels,
                    use_snippet=use_snippet,
                    # Crucial for low-memory deployments: don't load backbone weights
                    # from HuggingFace before applying the checkpoint.
                    load_pretrained_backbone=False,
                )
                # Move model to target dtype/device before loading weights (best effort).
                # If you upload an fp16 checkpoint and set MODEL_DTYPE=float16, this
                # significantly reduces RAM usage.
                model = model.to(device=device, dtype=model_dtype)

                state_dict = checkpoint['state_dict']
                # Drop the big checkpoint dict ASAP to reduce peak memory.
                checkpoint = None
                gc.collect()
                model.load_state_dict(state_dict)
                state_dict = None
                gc.collect()
            else:
                model = checkpoint
        else:
            model = checkpoint
        
        model.to(device)
        model.eval()
        model_loaded = True
        # Store model path for versioning (assign parameter to module-level variable)
        import sys
        current_module = sys.modules[__name__]
        current_module.model_path = model_path
        
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model path: {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        raise


# Inference Functions
async def predict_async(
    title: str,
    snippet: Optional[str] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    tag_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[List[TagPrediction], float]:
    """
    Async prediction function.
    
    Args:
        title: Article title
        snippet: Optional article snippet
        threshold: Classification threshold
        top_k: Return top K predictions
        tag_to_idx: Tag to index mapping
        
    Returns:
        List of tag predictions
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Run inference in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    predictions_and_threshold = await loop.run_in_executor(
        None,
        _predict_sync,
        title,
        snippet,
        threshold,
        top_k,
        tag_to_idx,  # Pass as tag_to_idx_param
    )
    
    return predictions_and_threshold


def _predict_sync(
    title: str,
    snippet: Optional[str] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    tag_to_idx_param: Optional[Dict[str, int]] = None,
) -> Tuple[List[TagPrediction], float]:
    """
    Synchronous prediction function (runs in thread pool).
    
    Args:
        title: Article title
        snippet: Optional article snippet
        threshold: Classification threshold
        top_k: Return top K predictions
        tag_to_idx_param: Tag to index mapping (if None, uses global tag_to_idx)
        
    Returns:
        List of tag predictions
    """
    global tag_to_idx
    
    # Use global tag_to_idx if not provided as parameter
    tag_mapping = tag_to_idx_param if tag_to_idx_param is not None else tag_to_idx
    
    # Prepare text
    title_clean = prepare_text_for_tokenization(title)
    snippet_clean = prepare_text_for_tokenization(snippet) if snippet else None
    
    # Tokenize
    title_encoded = tokenizer.encode(
        title_clean,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    # Tokenizer returns [1, seq_len] with return_tensors='pt', which is correct for batch_size=1
    title_input_ids = title_encoded['input_ids'].to(device)
    title_attention_mask = title_encoded['attention_mask'].to(device)
    
    snippet_input_ids = None
    snippet_attention_mask = None
    
    if snippet_clean:
        snippet_encoded = tokenizer.encode(
            snippet_clean,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        snippet_input_ids = snippet_encoded['input_ids'].to(device)
        snippet_attention_mask = snippet_encoded['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            snippet_input_ids=snippet_input_ids,
            snippet_attention_mask=snippet_attention_mask,
        )
        
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Apply custom thresholds if available
    global threshold_config
    effective_threshold = threshold
    per_class_thresholds = {}
    
    if threshold_config:
        # Use global threshold from config if request threshold is default (0.5)
        if threshold == 0.5 and 'global_threshold' in threshold_config:
            effective_threshold = threshold_config['global_threshold']
        
        # Load per-class thresholds if available
        if 'per_class_thresholds' in threshold_config:
            per_class_thresholds = threshold_config['per_class_thresholds']
    
    # Convert to predictions
    predictions = []
    
    if tag_mapping:
        # Use provided tag mapping
        idx_to_tag = {v: k for k, v in tag_mapping.items()}
        for idx, prob in enumerate(probs):
            # Use per-class threshold if available, otherwise use effective threshold
            class_threshold = per_class_thresholds.get(idx_to_tag.get(idx, f"tag_{idx}"), effective_threshold)
            if prob >= class_threshold:
                tag = idx_to_tag.get(idx, f"tag_{idx}")
                predictions.append(TagPrediction(tag=tag, score=float(prob)))
    else:
        # Generic tag indices
        for idx, prob in enumerate(probs):
            class_threshold = per_class_thresholds.get(f"tag_{idx}", effective_threshold)
            if prob >= class_threshold:
                predictions.append(TagPrediction(tag=f"tag_{idx}", score=float(prob)))
    
    # Sort by score and apply top_k
    predictions.sort(key=lambda x: x.score, reverse=True)
    
    if top_k:
        predictions = predictions[:top_k]
    
    return predictions, float(effective_threshold)


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model and initialize monitoring on startup."""
    global model, tokenizer, tag_to_idx, model_loaded, threshold_config, thresholds_path, device

    # Re-pick device at startup (helps if env/hardware changes between imports).
    device = _pick_device()
    
    # Resolve model path: env var wins, else default to best_model_v2.pt then best_model.pt
    model_path_env = os.environ.get("MODEL_PATH")
    if model_path_env:
        model_path_p = _resolve_path(model_path_env)
    else:
        model_path_p = _resolve_path("models/best_model_v2.pt")
        if not model_path_p.exists():
            model_path_p = _resolve_path("models/best_model.pt")
    
    async def _load_resources_background() -> None:
        """Load model (and later potentially other heavy resources) without blocking server startup."""
        try:
            if model_path_p.exists():
                await load_model(str(model_path_p))
            else:
                logger.warning(f"Model file not found: {model_path_p}. API will not work until model is loaded.")
        except Exception as e:
            # Keep the process alive; health will show model_not_loaded.
            logger.exception(f"Background model load failed: {e}")

    # IMPORTANT (Render): don't block server startup on large model/tokenizer load.
    # Render performs a port scan shortly after starting the process; heavy startup work can time out.
    asyncio.create_task(_load_resources_background())

    # Resolve thresholds path: env var wins, else config/thresholds.json (if present)
    thresholds_env = os.environ.get("THRESHOLDS_PATH")
    thresholds_path_p = _resolve_path(thresholds_env) if thresholds_env else _resolve_path("config/thresholds.json")
    if thresholds_path_p.exists():
        threshold_config = _load_thresholds_from_file(thresholds_path_p)
        thresholds_path = str(thresholds_path_p)
        if threshold_config is not None:
            logger.info(f"Loaded threshold configuration from {thresholds_path_p}")
    else:
        threshold_config = None
        thresholds_path = None
    
    logger.info("Monitoring initialized (middleware configured at app setup)")
    logger.info("Sentiment Analysis initialized")
    logger.info("Advanced Analytics initialized")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Russian News Classification API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global model_path, thresholds_path, threshold_config
    # Extract model version from path (e.g., "best_model_v3.pt" -> "v3" or "best_model.pt" -> "default")
    model_version_str = "default"
    if model_path:
        model_path_obj = Path(model_path)
        model_version_str = model_path_obj.stem  # e.g., "best_model_v3" or "best_model"

    thresholds_sha = None
    if thresholds_path:
        thresholds_sha = _file_sha256(Path(thresholds_path))
    
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        device=str(device),
        model_version=model_version_str if model_loaded else None,
        model_path=str(model_path) if model_loaded and model_path else None,
        global_threshold=(threshold_config or {}).get("global_threshold") if threshold_config else None,
        thresholds_path=thresholds_path,
        thresholds_sha256=thresholds_sha,
    )


@app.post("/thresholds/reload")
async def reload_thresholds():
    """Reload thresholds file from THRESHOLDS_PATH (or config/thresholds.json)."""
    global threshold_config, thresholds_path

    thresholds_env = os.environ.get("THRESHOLDS_PATH")
    thresholds_path_p = _resolve_path(thresholds_env) if thresholds_env else _resolve_path("config/thresholds.json")
    if not thresholds_path_p.exists():
        threshold_config = None
        thresholds_path = None
        return {"ok": False, "message": f"Thresholds file not found: {thresholds_path_p}"}

    threshold_config = _load_thresholds_from_file(thresholds_path_p)
    thresholds_path = str(thresholds_path_p)
    return {
        "ok": threshold_config is not None,
        "thresholds_path": thresholds_path,
        "thresholds_sha256": _file_sha256(thresholds_path_p),
        "global_threshold": (threshold_config or {}).get("global_threshold") if threshold_config else None,
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Classify a single news article.
    
    Args:
        request: Classification request
        background_tasks: Background tasks for monitoring
        
    Returns:
        Classification response with predictions
    """
    try:
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        preds, effective_threshold = await predict_async(
            title=request.title,
            snippet=request.snippet,
            threshold=request.threshold,
            top_k=request.top_k,
        )
        
        # Get model version from path
        global model_path
        model_version_str = "default"
        if model_path:
            model_path_obj = Path(model_path)
            model_version_str = model_path_obj.stem  # e.g., "best_model_v3" or "best_model"
        
        response = ClassificationResponse(
            predictions=preds,
            title=request.title,
            snippet=request.snippet,
            threshold=effective_threshold,
            model_version=model_version_str,
        )
        
        # Log prediction in background
        if prediction_logger:
            background_tasks.add_task(
                _log_prediction,
                input_data={
                    "title": request.title,
                    "snippet": request.snippet,
                },
                prediction={
                    "tags": [p.tag for p in preds],
                    "scores": {p.tag: p.score for p in preds},
                },
                metadata={
                    "model_version": model_version_str,
                },
            )
        
        # Record sample for drift detection
        if drift_detector:
            drift_detector.record_sample(
                title=request.title,
                snippet=request.snippet,
            )
        
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 503)
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _log_prediction(input_data: Dict, prediction: Dict, metadata: Dict) -> None:
    """Helper function to log prediction."""
    if prediction_logger:
        prediction_logger.log_prediction(input_data, prediction, metadata)


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify multiple news articles in batch.
    
    Args:
        request: Batch classification request
        
    Returns:
        Batch classification response
    """
    try:
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        results = []
        
        # Process items concurrently
        tasks = [
            predict_async(
                title=item.title,
                snippet=item.snippet,
                threshold=item.threshold,
                top_k=item.top_k,
            )
            for item in request.items
        ]
        
        predictions_list = await asyncio.gather(*tasks)
        
        # Create responses
        for item, predictions in zip(request.items, predictions_list):
            results.append(
                ClassificationResponse(
                    predictions=predictions,
                    title=item.title,
                    snippet=item.snippet,
                    threshold=item.threshold,
                    model_version="1.0.0",
                )
            )
        
        return BatchClassificationResponse(
            results=results,
            total=len(results),
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 503)
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks, model_path: str = "models/best_model.pt"):
    """
    Reload model from file.
    
    Args:
        background_tasks: FastAPI background tasks
        model_path: Path to model file
        
    Returns:
        Status message
    """
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
    
    background_tasks.add_task(load_model, model_path)
    
    return {"message": "Model reload initiated", "model_path": model_path}


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

