"""Experiment tracking system for model training and evaluation."""

import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Track and store experiment results for model comparison.
    
    Stores experiment metadata, metrics, and model information
    in a structured format for easy comparison and analysis.
    """
    
    def __init__(
        self,
        results_dir: str = "experiments/results",
        use_wandb: bool = False,
        use_mlflow: bool = False
    ):
        """
        Initialize experiment tracker.
        
        Args:
            results_dir: Directory to store experiment results
            use_wandb: Whether to log to WandB
            use_mlflow: Whether to log to MLflow
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        
        # Initialize tracking services
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, WandB tracking disabled")
                self.use_wandb = False
        
        self.mlflow_run = None
        if use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
            except ImportError:
                logger.warning("mlflow not installed, MLflow tracking disabled")
                self.use_mlflow = False
    
    def start_experiment(
        self,
        experiment_name: str,
        model_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model architecture
            config: Experiment configuration (hyperparameters, etc.)
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID (unique identifier)
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_data = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "config": config,
            "tags": tags or [],
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "model_path": None,
            "checkpoint_path": None
        }
        
        # Save experiment metadata
        experiment_file = self.results_dir / f"{experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Initialize WandB
        if self.use_wandb:
            try:
                self.wandb_run = self.wandb.init(
                    project="russian-news-classification",
                    name=experiment_id,
                    config=config,
                    tags=tags or []
                )
                logger.info(f"WandB run started: {self.wandb_run.id}")
            except Exception as e:
                logger.warning(f"Failed to start WandB run: {e}")
        
        # Initialize MLflow
        if self.use_mlflow:
            try:
                self.mlflow_run = self.mlflow.start_run(run_name=experiment_id)
                self.mlflow.log_params(config)
                if tags:
                    self.mlflow.set_tags({f"tag_{i}": tag for i, tag in enumerate(tags)})
                logger.info(f"MLflow run started: {self.mlflow_run.info.run_id}")
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        
        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metrics: Dictionary of metric names to values
            step: Optional step number (for training epochs)
        """
        # Update local file
        experiment_file = self.results_dir / f"{experiment_id}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            # Merge metrics
            experiment_data["metrics"].update(metrics)
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
        
        # Log to WandB
        if self.use_wandb and self.wandb_run:
            try:
                if step is not None:
                    self.wandb.log(metrics, step=step)
                else:
                    self.wandb.log(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")
        
        # Log to MLflow
        if self.use_mlflow and self.mlflow_run:
            try:
                if step is not None:
                    self.mlflow.log_metrics(metrics, step=step)
                else:
                    self.mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
    
    def log_model_path(
        self,
        experiment_id: str,
        model_path: str,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Log model file paths.
        
        Args:
            experiment_id: Experiment ID
            model_path: Path to saved model
            checkpoint_path: Optional path to checkpoint
        """
        experiment_file = self.results_dir / f"{experiment_id}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data["model_path"] = model_path
            if checkpoint_path:
                experiment_data["checkpoint_path"] = checkpoint_path
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
        
        # Log to MLflow
        if self.use_mlflow and self.mlflow_run:
            try:
                self.mlflow.log_artifact(model_path)
                if checkpoint_path:
                    self.mlflow.log_artifact(checkpoint_path)
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
    
    def finish_experiment(
        self,
        experiment_id: str,
        final_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Finish an experiment and save final results.
        
        Args:
            experiment_id: Experiment ID
            final_metrics: Optional final metrics to log
        """
        experiment_file = self.results_dir / f"{experiment_id}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                experiment_data = json.load(f)
            
            experiment_data["status"] = "completed"
            experiment_data["end_time"] = datetime.now().isoformat()
            
            if final_metrics:
                experiment_data["metrics"].update(final_metrics)
            
            with open(experiment_file, 'w') as f:
                json.dump(experiment_data, f, indent=2)
        
        # Finish WandB run
        if self.use_wandb and self.wandb_run:
            try:
                if final_metrics:
                    self.wandb.log(final_metrics)
                self.wandb.finish()
                self.wandb_run = None
            except Exception as e:
                logger.warning(f"Failed to finish WandB run: {e}")
        
        # Finish MLflow run
        if self.use_mlflow and self.mlflow_run:
            try:
                if final_metrics:
                    self.mlflow.log_metrics(final_metrics)
                self.mlflow.end_run()
                self.mlflow_run = None
            except Exception as e:
                logger.warning(f"Failed to finish MLflow run: {e}")
        
        logger.info(f"Finished experiment: {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment data by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment data dictionary or None if not found
        """
        experiment_file = self.results_dir / f"{experiment_id}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                return json.load(f)
        return None
    
    def list_experiments(
        self,
        model_name: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all experiments with optional filtering.
        
        Args:
            model_name: Filter by model name
            status: Filter by status ('running', 'completed', 'failed')
            
        Returns:
            List of experiment data dictionaries
        """
        experiments = []
        
        for experiment_file in self.results_dir.glob("*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    experiment_data = json.load(f)
                
                # Apply filters
                if model_name and experiment_data.get("model_name") != model_name:
                    continue
                if status and experiment_data.get("status") != status:
                    continue
                
                experiments.append(experiment_data)
            except Exception as e:
                logger.warning(f"Failed to read {experiment_file}: {e}")
        
        return experiments
    
    def get_comparison_dataframe(
        self,
        metric_name: str = "val_f1",
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get comparison DataFrame for all experiments.
        
        Args:
            metric_name: Metric to compare
            model_names: Optional list of model names to filter
            
        Returns:
            DataFrame with experiment comparison data
        """
        experiments = self.list_experiments()
        
        if model_names:
            experiments = [e for e in experiments if e.get("model_name") in model_names]
        
        data = []
        for exp in experiments:
            metrics = exp.get("metrics", {})
            row = {
                "experiment_id": exp.get("experiment_id"),
                "experiment_name": exp.get("experiment_name"),
                "model_name": exp.get("model_name"),
                "metric_value": metrics.get(metric_name),
                "status": exp.get("status"),
                "start_time": exp.get("start_time"),
                "config": exp.get("config", {})
            }
            # Also include all metrics as separate columns
            for key, value in metrics.items():
                if key not in row:
                    row[key] = value
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values("metric_value", ascending=False, na_position='last')

