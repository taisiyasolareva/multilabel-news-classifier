"""PyTorch Lightning module for training."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import logging

logger = logging.getLogger(__name__)


class NewsClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for news classification training.
    
    Handles both title-only and title+snippet models.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize training module.
        
        Args:
            model: The neural network model to train
            learning_rate: Learning rate for optimizer
            criterion: Loss function. If None, uses CrossEntropyLoss
            
        Example:
            >>> model = SimpleClassifier(vocab_size=10000, embedding_dim=300, output_dim=1000)
            >>> lightning_module = NewsClassificationModule(model, learning_rate=1e-3)
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Detect if model uses snippets
        # Check if model has use_snippet attribute or if forward() accepts snippet parameter
        import inspect
        if hasattr(model, 'use_snippet'):
            self.use_snippet = model.use_snippet
        else:
            # Check forward signature for snippet parameter
            sig = inspect.signature(model.forward)
            self.use_snippet = 'snippet' in sig.parameters
        
        logger.info(
            f"Initialized NewsClassificationModule: "
            f"lr={learning_rate}, use_snippet={self.use_snippet}"
        )

    def forward(
        self,
        title: torch.Tensor,
        snippet: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            title: Title token indices
            snippet: Optional snippet token indices
            
        Returns:
            Model logits
        """
        if self.use_snippet and snippet is not None:
            return self.model(title, snippet)
        else:
            return self.model(title)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer.
        
        Returns:
            Dictionary with optimizer configuration
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(
        self,
        train_batch: tuple,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.
        
        Args:
            train_batch: Batch of training data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        if self.use_snippet:
            title, snippet, target = train_batch
            logits = self.forward(title, snippet)
        else:
            title, target = train_batch
            logits = self.forward(title)
        
        loss = self.criterion(logits, target)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        val_batch: tuple,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            val_batch: Batch of validation data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        if self.use_snippet:
            title, snippet, target = val_batch
            logits = self.forward(title, snippet)
        else:
            title, target = val_batch
            logits = self.forward(title)
        
        loss = self.criterion(logits, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

