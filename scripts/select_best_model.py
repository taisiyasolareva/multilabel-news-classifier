"""Script to automatically select best model from multiple runs."""

import logging
import argparse
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_best_model(
    results_dir: str,
    metric_name: str = "val_f1",
    mode: str = "max",
) -> dict:
    """
    Select best model from results directory.
    
    Args:
        results_dir: Directory containing model results
        metric_name: Metric to use for selection
        mode: "max" or "min"
        
    Returns:
        Dictionary with best model information
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    best_value = float("-inf") if mode == "max" else float("inf")
    best_model = None
    best_run = None
    
    # Search for results files
    for result_file in results_path.rglob("*.json"):
        try:
            with open(result_file) as f:
                result = json.load(f)
            
            metric_value = result.get(metric_name)
            if metric_value is None:
                continue
            
            is_best = False
            if mode == "max":
                if metric_value > best_value:
                    is_best = True
            else:
                if metric_value < best_value:
                    is_best = True
            
            if is_best:
                best_value = metric_value
                best_model = result.get("model_path")
                best_run = result_file.stem
                
        except Exception as e:
            logger.warning(f"Failed to read {result_file}: {e}")
    
    if best_model is None:
        raise ValueError("No valid results found")
    
    result = {
        "best_model_path": best_model,
        "best_metric_value": best_value,
        "best_run": best_run,
        "metric_name": metric_name,
    }
    
    logger.info("=" * 60)
    logger.info("Best Model Selection Results")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_model}")
    logger.info(f"Best {metric_name}: {best_value:.4f}")
    logger.info(f"Best run: {best_run}")
    
    return result


def select_from_optuna_study(
    study_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Select best model from Optuna study.
    
    Args:
        study_path: Path to Optuna study file
        output_path: Path to save best model info
        
    Returns:
        Dictionary with best model information
    """
    import joblib
    
    study = joblib.load(study_path)
    
    best_trial = study.best_trial
    best_params = study.best_params
    best_value = study.best_value
    
    result = {
        "best_trial": best_trial.number,
        "best_value": best_value,
        "best_params": best_params,
    }
    
    logger.info("=" * 60)
    logger.info("Optuna Study Results")
    logger.info("=" * 60)
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best value: {best_value:.4f}")
    logger.info("Best parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return result


def select_from_wandb_sweep(
    project: str,
    sweep_id: str,
    entity: Optional[str] = None,
) -> dict:
    """
    Select best model from WandB sweep.
    
    Args:
        project: WandB project name
        sweep_id: Sweep ID
        entity: WandB entity
        
    Returns:
        Dictionary with best model information
    """
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    sweep = api.sweep(f"{entity or ''}/{project}/{sweep_id}".lstrip('/'))
    
    # Get best run
    runs = sorted(
        sweep.runs,
        key=lambda r: r.summary.get("val_f1", 0),
        reverse=True,
    )
    
    if not runs:
        raise ValueError("No runs found in sweep")
    
    best_run = runs[0]
    
    result = {
        "run_id": best_run.id,
        "run_name": best_run.name,
        "config": dict(best_run.config),
        "metrics": dict(best_run.summary),
    }
    
    logger.info("=" * 60)
    logger.info("WandB Sweep Results")
    logger.info("=" * 60)
    logger.info(f"Best run: {best_run.name}")
    logger.info(f"Best val_f1: {best_run.summary.get('val_f1', 'N/A')}")
    logger.info("Best config:")
    for key, value in best_run.config.items():
        logger.info(f"  {key}: {value}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best model")
    parser.add_argument(
        "--method",
        type=str,
        choices=["results", "optuna", "wandb"],
        default="results",
        help="Selection method"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Results directory (for results method)"
    )
    parser.add_argument(
        "--study-path",
        type=str,
        help="Path to Optuna study (for optuna method)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="WandB project (for wandb method)"
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="WandB sweep ID (for wandb method)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_f1",
        help="Metric name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Optimization mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    if args.method == "results":
        result = select_best_model(
            results_dir=args.results_dir,
            metric_name=args.metric,
            mode=args.mode,
        )
    elif args.method == "optuna":
        if not args.study_path:
            raise ValueError("--study-path required for optuna method")
        result = select_from_optuna_study(
            study_path=args.study_path,
            output_path=args.output,
        )
    elif args.method == "wandb":
        if not args.project or not args.sweep_id:
            raise ValueError("--project and --sweep-id required for wandb method")
        result = select_from_wandb_sweep(
            project=args.project,
            sweep_id=args.sweep_id,
        )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")

