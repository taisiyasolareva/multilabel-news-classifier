"""Systematic model comparison framework."""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from evaluation.metrics import (
    precision,
    recall,
    f1_score,
    exact_match,
    get_predict,
    per_class_metrics
)
from experiments.experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Systematic comparison of multiple model architectures.
    
    Trains and evaluates multiple models on the same dataset,
    tracks results, and generates comparison reports.
    """
    
    def __init__(
        self,
        tracker: Optional[ExperimentTracker] = None,
        results_dir: str = "experiments/comparisons"
    ):
        """
        Initialize model comparison framework.
        
        Args:
            tracker: ExperimentTracker instance (creates new if None)
            results_dir: Directory to store comparison results
        """
        if tracker is None:
            self.tracker = ExperimentTracker()
        else:
            self.tracker = tracker
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison_results = []
    
    def compare_models(
        self,
        models_config: List[Dict[str, Any]],
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        train_func: Optional[Callable] = None,
        epochs: int = 3,
        batch_size: int = 16
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same datasets.
        
        Args:
            models_config: List of model configurations
                Each config should have: model_name, model_class, model_kwargs
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
            train_func: Optional custom training function
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            DataFrame with comparison results
            
        Example:
            >>> comparison = ModelComparison()
            >>> models = [
            ...     {
            ...         "model_name": "RussianBERT",
            ...         "model_class": RussianNewsClassifier,
            ...         "model_kwargs": {"num_labels": 100}
            ...     },
            ...     {
            ...         "model_name": "RoBERTa",
            ...         "model_class": RoBERTaNewsClassifier,
            ...         "model_kwargs": {"num_labels": 100}
            ...     }
            ... ]
            >>> results = comparison.compare_models(models, train_ds, val_ds)
        """
        logger.info("=" * 80)
        logger.info("Starting Model Comparison")
        logger.info("=" * 80)
        logger.info(f"Comparing {len(models_config)} models")
        
        results = []
        
        for i, model_config in enumerate(models_config, 1):
            model_name = model_config.get("model_name", f"model_{i}")
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Model {i}/{len(models_config)}: {model_name}")
            logger.info(f"{'=' * 80}")
            
            try:
                # Start experiment
                experiment_id = self.tracker.start_experiment(
                    experiment_name=f"comparison_{model_name}",
                    model_name=model_name,
                    config={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        **model_config.get("model_kwargs", {})
                    },
                    tags=["model_comparison", model_name]
                )
                
                # Train model (if train_func provided)
                model = None
                if train_func:
                    logger.info(f"Training {model_name}...")
                    model = train_func(
                        model_config=model_config,
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                else:
                    logger.warning("No training function provided, skipping training")
                
                # Evaluate on validation set
                if model and val_dataset:
                    val_metrics = self._evaluate_model(
                        model,
                        val_dataset,
                        model_config.get("use_snippet", False),
                        prefix="val_"
                    )
                    self.tracker.log_metrics(experiment_id, val_metrics)
                    logger.info(f"Validation metrics: {val_metrics}")
                
                # Evaluate on test set
                test_metrics = {}
                if model and test_dataset:
                    test_metrics = self._evaluate_model(
                        model,
                        test_dataset,
                        model_config.get("use_snippet", False),
                        prefix="test_"
                    )
                    self.tracker.log_metrics(experiment_id, test_metrics)
                    logger.info(f"Test metrics: {test_metrics}")
                
                # Finish experiment
                self.tracker.finish_experiment(experiment_id, test_metrics)
                
                # Store results
                result = {
                    "model_name": model_name,
                    "experiment_id": experiment_id,
                    **val_metrics,
                    **test_metrics,
                    "status": "completed"
                }
                results.append(result)
                self.comparison_results.append(result)
                
            except Exception as e:
                logger.error(f"Error comparing {model_name}: {e}")
                result = {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(e)
                }
                results.append(result)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Save comparison results
        comparison_file = self.results_dir / f"comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"\nComparison results saved to: {comparison_file}")
        
        # Generate comparison report
        self._generate_report(comparison_df)
        
        return comparison_df
    
    def _evaluate_model(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        use_snippet: bool = False,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            model: Trained model
            dataset: Dataset to evaluate on
            use_snippet: Whether model uses snippets
            prefix: Prefix for metric names (e.g., "val_", "test_")
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        pred_prob, target = get_predict(model, dataset, use_snippet)
        
        # Use default threshold for now (can be optimized)
        threshold = 0.5
        y_pred = (pred_prob > threshold).float()
        
        # Calculate metrics
        metrics = {
            f"{prefix}precision": precision(target, y_pred),
            f"{prefix}recall": recall(target, y_pred),
            f"{prefix}f1": f1_score(target, y_pred),
            f"{prefix}exact_match": exact_match(target, y_pred)
        }
        
        return metrics
    
    def _generate_report(self, comparison_df: pd.DataFrame) -> None:
        """
        Generate comparison report.
        
        Args:
            comparison_df: DataFrame with comparison results
        """
        report_file = self.results_dir / f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Models compared: {len(comparison_df)}\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            
            metric_cols = [col for col in comparison_df.columns if any(
                m in col for m in ["precision", "recall", "f1", "exact_match"]
            )]
            
            for metric_col in metric_cols:
                f.write(f"\n{metric_col.upper()}:\n")
                sorted_df = comparison_df.sort_values(metric_col, ascending=False, na_last=True)
                for _, row in sorted_df.iterrows():
                    model_name = row.get("model_name", "Unknown")
                    value = row.get(metric_col, "N/A")
                    f.write(f"  {model_name}: {value}\n")
            
            # Best model
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST MODEL\n")
            f.write("=" * 80 + "\n")
            
            if "val_f1" in comparison_df.columns:
                best = comparison_df.nlargest(1, "val_f1")
                if not best.empty:
                    best_model = best.iloc[0]
                    f.write(f"Model: {best_model['model_name']}\n")
                    f.write(f"Validation F1: {best_model.get('val_f1', 'N/A')}\n")
                    f.write(f"Validation Precision: {best_model.get('val_precision', 'N/A')}\n")
                    f.write(f"Validation Recall: {best_model.get('val_recall', 'N/A')}\n")
        
        logger.info(f"Comparison report saved to: {report_file}")
    
    def get_best_model(
        self,
        metric_name: str = "val_f1",
        comparison_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get best model from comparison.
        
        Args:
            metric_name: Metric to use for selection
            comparison_df: Optional comparison DataFrame (uses stored if None)
            
        Returns:
            Dictionary with best model information
        """
        if comparison_df is None:
            if not self.comparison_results:
                logger.warning("No comparison results available")
                return None
            comparison_df = pd.DataFrame(self.comparison_results)
        
        if metric_name not in comparison_df.columns:
            logger.warning(f"Metric {metric_name} not found in comparison results")
            return None
        
        # Filter completed models
        completed = comparison_df[comparison_df["status"] == "completed"]
        if completed.empty:
            logger.warning("No completed models found")
            return None
        
        # Get best model
        best = completed.nlargest(1, metric_name)
        if best.empty:
            return None
        
        best_model = best.iloc[0].to_dict()
        logger.info(f"Best model: {best_model['model_name']} ({metric_name}={best_model.get(metric_name, 'N/A')})")
        
        return best_model
    
    def compare_from_checkpoints(
        self,
        checkpoint_paths: List[Dict[str, str]],
        test_dataset: Dataset,
        model_classes: Dict[str, type]
    ) -> pd.DataFrame:
        """
        Compare models from saved checkpoints.
        
        Args:
            checkpoint_paths: List of dicts with model_name and checkpoint_path
            test_dataset: Test dataset for evaluation
            model_classes: Dictionary mapping model_name to model class
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("=" * 80)
        logger.info("Comparing Models from Checkpoints")
        logger.info("=" * 80)
        
        results = []
        
        for checkpoint_info in checkpoint_paths:
            model_name = checkpoint_info["model_name"]
            checkpoint_path = checkpoint_info["checkpoint_path"]
            
            logger.info(f"\nEvaluating {model_name} from {checkpoint_path}")
            
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Get model class
                model_class = model_classes.get(model_name)
                if model_class is None:
                    logger.warning(f"Model class not found for {model_name}, skipping")
                    continue
                
                # Reconstruct model
                model_kwargs = {
                    "num_labels": checkpoint.get("num_labels", 1000),
                    "use_snippet": checkpoint.get("use_snippet", False),
                    "dropout": checkpoint.get("dropout", 0.3)
                }
                
                if "model_name" in checkpoint:
                    model_kwargs["model_name"] = checkpoint["model_name"]
                
                model = model_class(**model_kwargs)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                
                # Evaluate
                use_snippet = checkpoint.get("use_snippet", False)
                test_metrics = self._evaluate_model(
                    model,
                    test_dataset,
                    use_snippet,
                    prefix="test_"
                )
                
                result = {
                    "model_name": model_name,
                    "checkpoint_path": checkpoint_path,
                    **test_metrics,
                    "status": "completed"
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results.append({
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        comparison_df = pd.DataFrame(results)
        
        # Save results
        comparison_file = self.results_dir / f"checkpoint_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        return comparison_df

