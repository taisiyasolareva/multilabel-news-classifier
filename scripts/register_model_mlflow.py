"""Script to register model in MLflow model registry."""

import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_model(
    model_path: str,
    model_name: str,
    run_id: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    tags: Optional[dict] = None,
) -> None:
    """
    Register model in MLflow model registry.
    
    Args:
        model_path: Path to model file or MLflow run URI
        model_name: Name for model in registry
        run_id: MLflow run ID (if model_path is not a URI)
        tracking_uri: MLflow tracking URI
        tags: Dictionary of tags
    """
    try:
        import mlflow
        import mlflow.pytorch
    except ImportError:
        raise ImportError("mlflow not installed. Install with: pip install mlflow")
    
    # Set tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Determine model URI
    if model_path.startswith("runs:/"):
        model_uri = model_path
    elif run_id:
        model_uri = f"runs:/{run_id}/{model_path}"
    else:
        # Assume model is in current MLflow run
        model_uri = model_path
    
    # Register model
    logger.info(f"Registering model: {model_name}")
    logger.info(f"Model URI: {model_uri}")
    
    try:
        mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags or {},
        )
        logger.info(f"Model '{model_name}' registered successfully!")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model or MLflow run URI (runs:/<run_id>/model)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name for model in registry"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run ID (if model_path is not a URI)"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI"
    )
    
    args = parser.parse_args()
    
    register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        run_id=args.run_id,
        tracking_uri=args.tracking_uri,
    )

