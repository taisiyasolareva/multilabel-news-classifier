"""Advanced transformer architectures for news classification."""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    RobertaModel,
    RobertaConfig,
    DistilBertModel,
    DistilBertConfig,
)
import logging

logger = logging.getLogger(__name__)


class RoBERTaNewsClassifier(nn.Module):
    """
    RoBERTa-based classifier for Russian news.
    
    RoBERTa is an optimized version of BERT with better performance.
    Uses Russian RoBERTa if available, otherwise multilingual.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",  # Multilingual RoBERTa
        num_labels: int = 1000,
        dropout: float = 0.3,
        use_snippet: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Initialize RoBERTa classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            dropout: Dropout probability
            use_snippet: Whether to use snippets
            freeze_backbone: Freeze RoBERTa weights
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_snippet = use_snippet
        
        logger.info(f"Loading RoBERTa model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size
        
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False
            logger.info("RoBERTa weights frozen")
        
        # Classification head
        if use_snippet:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels),
            )
        
        logger.info(f"Initialized RoBERTaNewsClassifier: hidden_size={hidden_size}")

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Title representation
        title_outputs = self.roberta(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
        )
        title_repr = title_outputs.pooler_output if hasattr(title_outputs, 'pooler_output') else title_outputs.last_hidden_state[:, 0]
        
        if self.use_snippet and snippet_input_ids is not None:
            snippet_outputs = self.roberta(
                input_ids=snippet_input_ids,
                attention_mask=snippet_attention_mask,
            )
            snippet_repr = snippet_outputs.pooler_output if hasattr(snippet_outputs, 'pooler_output') else snippet_outputs.last_hidden_state[:, 0]
            combined = torch.cat([title_repr, snippet_repr], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(title_repr)
        
        return logits


class DistilBERTNewsClassifier(nn.Module):
    """
    DistilBERT-based classifier (faster, smaller than BERT).
    
    DistilBERT is 60% faster and 60% smaller than BERT while
    retaining 97% of BERT's performance.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        num_labels: int = 1000,
        dropout: float = 0.3,
        use_snippet: bool = True,
    ):
        """
        Initialize DistilBERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            dropout: Dropout probability
            use_snippet: Whether to use snippets
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_snippet = use_snippet
        
        logger.info(f"Loading DistilBERT model: {model_name}")
        config = DistilBertConfig.from_pretrained(model_name)
        self.distilbert = DistilBertModel.from_pretrained(model_name, config=config)
        
        hidden_size = config.dim
        
        # Classification head
        if use_snippet:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels),
            )
        
        logger.info(f"Initialized DistilBERTNewsClassifier: hidden_size={hidden_size}")

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        title_outputs = self.distilbert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
        )
        title_repr = title_outputs.last_hidden_state[:, 0]  # [CLS] token
        
        if self.use_snippet and snippet_input_ids is not None:
            snippet_outputs = self.distilbert(
                input_ids=snippet_input_ids,
                attention_mask=snippet_attention_mask,
            )
            snippet_repr = snippet_outputs.last_hidden_state[:, 0]
            combined = torch.cat([title_repr, snippet_repr], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(title_repr)
        
        return logits


class MultiHeadAttentionClassifier(nn.Module):
    """
    BERT with multi-head attention pooling instead of [CLS] token.
    
    Uses attention mechanism to aggregate token representations.
    """

    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        num_labels: int = 1000,
        dropout: float = 0.3,
        use_snippet: bool = True,
        num_attention_heads: int = 8,
    ):
        """
        Initialize multi-head attention classifier.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of output classes
            dropout: Dropout probability
            use_snippet: Whether to use snippets
            num_attention_heads: Number of attention heads for pooling
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_snippet = use_snippet
        
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        
        # Multi-head attention for pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Classification head
        if use_snippet:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels),
            )
        
        logger.info(f"Initialized MultiHeadAttentionClassifier: hidden_size={hidden_size}")

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Forward pass with attention pooling.
        
        Args:
            return_attention: If True, return attention weights for visualization
            
        Returns:
            Logits or (logits, attention_weights)
        """
        # Get BERT outputs
        title_outputs = self.bert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
        )
        title_hidden = title_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Attention pooling
        query = self.query.expand(title_hidden.size(0), -1, -1)
        title_repr, title_attn = self.attention_pooling(
            query, title_hidden, title_hidden,
            key_padding_mask=~title_attention_mask.bool()
        )
        title_repr = title_repr.squeeze(1)  # [batch, hidden_size]
        
        if self.use_snippet and snippet_input_ids is not None:
            snippet_outputs = self.bert(
                input_ids=snippet_input_ids,
                attention_mask=snippet_attention_mask,
            )
            snippet_hidden = snippet_outputs.last_hidden_state
            snippet_repr, snippet_attn = self.attention_pooling(
                query, snippet_hidden, snippet_hidden,
                key_padding_mask=~snippet_attention_mask.bool()
            )
            snippet_repr = snippet_repr.squeeze(1)
            
            combined = torch.cat([title_repr, snippet_repr], dim=1)
            logits = self.classifier(combined)
            
            if return_attention:
                return logits, {'title': title_attn, 'snippet': snippet_attn}
        else:
            logits = self.classifier(title_repr)
            if return_attention:
                return logits, {'title': title_attn}
        
        return logits if not return_attention else (logits, {})


class EnsembleClassifier(nn.Module):
    """
    Ensemble of multiple transformer models.
    
    Combines predictions from multiple models using weighted averaging.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted_average",
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
            ensemble_method: "weighted_average" or "voting"
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        logger.info(
            f"Initialized EnsembleClassifier: {len(models)} models, "
            f"method={ensemble_method}, weights={self.weights}"
        )

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
            Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            # Get predictions from each model
            if hasattr(model, 'forward'):
                pred = model(
                    title_input_ids=title_input_ids,
                    title_attention_mask=title_attention_mask,
                    snippet_input_ids=snippet_input_ids,
                    snippet_attention_mask=snippet_attention_mask,
                )
            else:
                # Handle different model interfaces
                pred = model(title_input_ids, title_attention_mask)
            
            predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=0)  # [num_models, batch, num_labels]
        
        if self.ensemble_method == "weighted_average":
            # Weighted average
            weights_tensor = torch.tensor(
                self.weights,
                device=stacked.device,
                dtype=stacked.dtype
            ).view(-1, 1, 1)
            ensemble_pred = (stacked * weights_tensor).sum(dim=0)
        elif self.ensemble_method == "voting":
            # Hard voting (majority)
            probs = torch.sigmoid(stacked)
            votes = (probs > 0.5).float()
            ensemble_pred = votes.mean(dim=0) * stacked.sum(dim=0) / votes.sum(dim=0).clamp(min=1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_pred

