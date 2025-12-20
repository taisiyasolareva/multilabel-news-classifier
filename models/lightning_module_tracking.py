"""PyTorch Lightning module with enhanced experiment tracking."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger

from utils.experiment_tracking import WandBTracker, MLflowTracker, ExperimentTracker
import logging

logger = logging.getLogger(__name__)


class WandBCallback(Callback):
    """Enhanced WandB callback for PyTorch Lightning."""

    def __init__(self, log_model: bool = True, log_artifacts: bool = True):
        """
        Initialize WandB callback.
        
        Args:
            log_model: Whether to log model checkpoints
            log_artifacts: Whether to log artifacts
        """
        super().__init__()
        self.log_model = log_model
        self.log_artifacts = log_artifacts

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log metrics at end of training epoch."""
        metrics = {f"train/{k}": v for k, v in trainer.callback_metrics.items()}
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log metrics at end of validation epoch."""
        metrics = {f"val/{k}": v for k, v in trainer.callback_metrics.items()}
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(metrics)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log artifacts at end of training."""
        if self.log_artifacts and hasattr(trainer, 'logger'):
            if isinstance(trainer.logger, WandbLogger):
                # Log best model
                if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                    trainer.logger.experiment.log_artifact(
                        trainer.checkpoint_callback.best_model_path,
                        name="best_model"
                    )


class MLflowCallback(Callback):
    """MLflow callback for PyTorch Lightning."""

    def __init__(self, log_model: bool = True):
        """
        Initialize MLflow callback.
        
        Args:
            log_model: Whether to log model
        """
        super().__init__()
        self.log_model = log_model

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log metrics at end of training epoch."""
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, MLFlowLogger):
            metrics = {f"train_{k}": v for k, v in trainer.callback_metrics.items()}
            trainer.logger.experiment.log_metrics(metrics, step=trainer.current_epoch)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log metrics at end of validation epoch."""
        if hasattr(trainer, 'logger') and isinstance(trainer.logger, MLFlowLogger):
            metrics = {f"val_{k}": v for k, v in trainer.callback_metrics.items()}
            trainer.logger.experiment.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log model at end of training."""
        if self.log_model and hasattr(trainer, 'logger'):
            if isinstance(trainer.logger, MLFlowLogger):
                # Log model
                trainer.logger.experiment.log_model(
                    pl_module.model,
                    artifact_path="model"
                )


def create_tracking_loggers(
    use_wandb: bool = True,
    use_mlflow: bool = True,
    project_name: str = "russian-news-classification",
    experiment_name: Optional[str] = None,
    **kwargs
) -> tuple[list, list]:
    """
    Create tracking loggers and callbacks.
    
    Args:
        use_wandb: Enable WandB
        use_mlflow: Enable MLflow
        project_name: Project name
        experiment_name: Experiment name
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (loggers, callbacks)
    """
    loggers = []
    callbacks = []
    
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=project_name,
                name=experiment_name,
                **kwargs.get('wandb', {})
            )
            loggers.append(wandb_logger)
            callbacks.append(WandBCallback())
            logger.info("WandB logger created")
        except Exception as e:
            logger.warning(f"Failed to create WandB logger: {e}")
    
    if use_mlflow:
        try:
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name or project_name,
                tracking_uri=kwargs.get('mlflow', {}).get('tracking_uri'),
            )
            loggers.append(mlflow_logger)
            callbacks.append(MLflowCallback())
            logger.info("MLflow logger created")
        except Exception as e:
            logger.warning(f"Failed to create MLflow logger: {e}")
    
    return loggers, callbacks

