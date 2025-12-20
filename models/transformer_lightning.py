"""PyTorch Lightning module for transformer-based models."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import logging

logger = logging.getLogger(__name__)


class TransformerClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for transformer-based news classification.
    
    Handles fine-tuning of BERT models with proper learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        num_training_steps: Optional[int] = None,
        use_snippet: bool = True,
    ):
        """
        Initialize transformer training module.
        
        Args:
            model: Transformer model (RussianNewsClassifier or similar)
            learning_rate: Learning rate (typically 2e-5 for BERT fine-tuning)
            warmup_steps: Number of warmup steps for LR scheduler
            weight_decay: Weight decay for optimizer
            num_training_steps: Total training steps (for LR scheduler)
            use_snippet: Whether model uses snippets
            
        Example:
            >>> model = RussianNewsClassifier(num_labels=1000)
            >>> module = TransformerClassificationModule(
            ...     model=model,
            ...     learning_rate=2e-5,
            ...     num_training_steps=1000
            ... )
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps
        self.use_snippet = use_snippet
        
        # Loss function (BCE with logits for multi-label)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Store validation outputs for epoch-end metrics
        self._val_outputs = []
        
        self.save_hyperparameters(ignore=['model'])
        
        logger.info(
            f"Initialized TransformerClassificationModule: "
            f"lr={learning_rate}, warmup_steps={warmup_steps}, "
            f"use_snippet={use_snippet}"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, etc.
            
        Returns:
            Model logits
        """
        return self.model(
            title_input_ids=batch['title_input_ids'],
            title_attention_mask=batch['title_attention_mask'],
            snippet_input_ids=batch.get('snippet_input_ids'),
            snippet_attention_mask=batch.get('snippet_attention_mask'),
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        logits = self.forward(batch)
        loss = self.criterion(logits, batch['labels'])
        
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Batch of validation data
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss, predictions, and labels
        """
        logits = self.forward(batch)
        loss = self.criterion(logits, batch['labels'])
        
        # Calculate predictions (sigmoid + threshold)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        
        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        
        # Store outputs for epoch-end metrics
        output = {
            'loss': loss,
            'predictions': predictions,
            'labels': batch['labels'],
        }
        self._val_outputs.append(output)
        
        return output

    def on_validation_epoch_start(self) -> None:
        """Initialize validation outputs at start of epoch."""
        self._val_outputs = []
    
    def on_validation_epoch_end(self) -> None:
        """
        Calculate epoch-end metrics.
        
        Uses stored validation outputs from validation_step.
        """
        if not hasattr(self, '_val_outputs') or not self._val_outputs:
            return
        
        all_preds = torch.cat([o['predictions'] for o in self._val_outputs])
        all_labels = torch.cat([o['labels'] for o in self._val_outputs])
        
        # Calculate precision, recall, F1
        precision = self._calculate_precision(all_preds, all_labels)
        recall = self._calculate_recall(all_preds, all_labels)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)

    def _calculate_precision(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate precision metric."""
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        return (tp / (tp + fp + 1e-8)).item()

    def _calculate_recall(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """Calculate recall metric."""
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        return (tp / (tp + fn + 1e-8)).item()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Use AdamW for transformer fine-tuning
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        if self.num_training_steps:
            # Linear warmup + linear decay scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                },
            }
        
        return {'optimizer': optimizer}

