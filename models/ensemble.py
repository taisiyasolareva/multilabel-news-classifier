"""Ensemble methods for combining multiple models."""

from typing import List, Dict, Optional, Callable
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WeightedEnsemble(nn.Module):
    """
    Weighted ensemble of multiple models.
    
    Combines predictions using learned or fixed weights.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        learn_weights: bool = False,
    ):
        """
        Initialize weighted ensemble.
        
        Args:
            models: List of trained models
            weights: Initial weights (default: equal weights)
            learn_weights: If True, learn weights during training
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.learn_weights = learn_weights
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if learn_weights:
            # Learnable weights
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            # Fixed weights
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(
            f"Initialized WeightedEnsemble: {len(models)} models, "
            f"learn_weights={learn_weights}"
        )

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if self.learn_weights:
            # Softmax normalization
            self.weights.data = torch.softmax(self.weights.data, dim=0)
        else:
            # Simple normalization
            total = self.weights.sum()
            if total > 0:
                self.weights.data = self.weights.data / total

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    # Check if model accepts snippet parameters
                    import inspect
                    sig = inspect.signature(model.forward)
                    params = sig.parameters
                    
                    # Build kwargs based on what model accepts
                    kwargs = {
                        'title_input_ids': title_input_ids,
                        'title_attention_mask': title_attention_mask,
                    }
                    
                    # Only add snippet params if model accepts them and they're provided
                    if 'snippet_input_ids' in params and snippet_input_ids is not None:
                        kwargs['snippet_input_ids'] = snippet_input_ids
                    if 'snippet_attention_mask' in params and snippet_attention_mask is not None:
                        kwargs['snippet_attention_mask'] = snippet_attention_mask
                    
                    pred = model(**kwargs)
                else:
                    pred = model(title_input_ids, title_attention_mask)
                predictions.append(pred)
        
        # Stack and weight
        stacked = torch.stack(predictions, dim=0)  # [num_models, batch, num_labels]
        weights = self.weights.view(-1, 1, 1)  # [num_models, 1, 1]
        
        ensemble_pred = (stacked * weights).sum(dim=0)
        
        return ensemble_pred


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner.
    
    Uses a second-level model to learn how to combine base models.
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_labels: int,
        meta_hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            models: List of base models
            num_labels: Number of output classes
            meta_hidden_dim: Hidden dimension for meta-learner
            dropout: Dropout for meta-learner
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        # Meta-learner (learns to combine base models)
        input_dim = len(models) * num_labels
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim // 2, num_labels),
        )
        
        logger.info(
            f"Initialized StackingEnsemble: {len(models)} base models, "
            f"meta_hidden_dim={meta_hidden_dim}"
        )

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through stacking ensemble.
        
        Returns:
            Meta-learner predictions
        """
        base_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    # Check if model accepts snippet parameters
                    import inspect
                    sig = inspect.signature(model.forward)
                    params = sig.parameters
                    
                    # Build kwargs based on what model accepts
                    kwargs = {
                        'title_input_ids': title_input_ids,
                        'title_attention_mask': title_attention_mask,
                    }
                    
                    # Only add snippet params if model accepts them and they're provided
                    if 'snippet_input_ids' in params and snippet_input_ids is not None:
                        kwargs['snippet_input_ids'] = snippet_input_ids
                    if 'snippet_attention_mask' in params and snippet_attention_mask is not None:
                        kwargs['snippet_attention_mask'] = snippet_attention_mask
                    
                    pred = model(**kwargs)
                else:
                    pred = model(title_input_ids, title_attention_mask)
                base_predictions.append(pred)
        
        # Concatenate base predictions
        stacked = torch.cat(base_predictions, dim=1)  # [batch, num_models * num_labels]
        
        # Meta-learner prediction
        ensemble_pred = self.meta_learner(stacked)
        
        return ensemble_pred


class VotingEnsemble(nn.Module):
    """
    Hard/soft voting ensemble.
    
    Combines predictions using majority voting or probability averaging.
    """

    def __init__(
        self,
        models: List[nn.Module],
        voting_type: str = "soft",  # "soft" or "hard"
        threshold: float = 0.5,
    ):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of trained models
            voting_type: "soft" (average probabilities) or "hard" (majority vote)
            threshold: Threshold for hard voting
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting_type = voting_type
        self.threshold = threshold
        
        logger.info(
            f"Initialized VotingEnsemble: {len(models)} models, "
            f"voting_type={voting_type}"
        )

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through voting ensemble.
        
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    # Check if model accepts snippet parameters
                    import inspect
                    sig = inspect.signature(model.forward)
                    params = sig.parameters
                    
                    # Build kwargs based on what model accepts
                    kwargs = {
                        'title_input_ids': title_input_ids,
                        'title_attention_mask': title_attention_mask,
                    }
                    
                    # Only add snippet params if model accepts them and they're provided
                    if 'snippet_input_ids' in params and snippet_input_ids is not None:
                        kwargs['snippet_input_ids'] = snippet_input_ids
                    if 'snippet_attention_mask' in params and snippet_attention_mask is not None:
                        kwargs['snippet_attention_mask'] = snippet_attention_mask
                    
                    pred = model(**kwargs)
                else:
                    pred = model(title_input_ids, title_attention_mask)
                predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=0)  # [num_models, batch, num_labels]
        
        if self.voting_type == "soft":
            # Average probabilities
            probs = torch.sigmoid(stacked)
            ensemble_pred = probs.mean(dim=0)
            # Convert back to logits
            ensemble_pred = torch.logit(ensemble_pred.clamp(min=1e-7, max=1-1e-7))
        else:
            # Hard voting (majority)
            probs = torch.sigmoid(stacked)
            votes = (probs > self.threshold).float()
            ensemble_pred = votes.mean(dim=0) * stacked.sum(dim=0) / votes.sum(dim=0).clamp(min=1)
        
        return ensemble_pred


def create_ensemble(
    models: List[nn.Module],
    method: str = "weighted",
    **kwargs
) -> nn.Module:
    """
    Factory function to create ensemble.
    
    Args:
        models: List of models
        method: "weighted", "stacking", or "voting"
        **kwargs: Additional arguments for ensemble
        
    Returns:
        Ensemble model
    """
    if method == "weighted":
        return WeightedEnsemble(models, **kwargs)
    elif method == "stacking":
        return StackingEnsemble(models, **kwargs)
    elif method == "voting":
        return VotingEnsemble(models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

