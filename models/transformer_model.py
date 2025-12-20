"""Transformer-based models for Russian news classification."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    BertConfig,
)
import logging

logger = logging.getLogger(__name__)


class RussianNewsClassifier(nn.Module):
    """
    BERT-based classifier for Russian news tag classification.
    
    Uses pre-trained Russian BERT models (rubert-base-cased) and fine-tunes
    for multi-label classification task.
    """

    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        num_labels: int = 1000,
        dropout: float = 0.3,
        use_snippet: bool = True,
        freeze_bert: bool = False,
    ):
        """
        Initialize transformer-based classifier.
        
        Args:
            model_name: HuggingFace model name or path
                Options:
                - "DeepPavlov/rubert-base-cased" (Russian BERT)
                - "bert-base-multilingual-cased" (Multilingual BERT)
            num_labels: Number of output classes (tags)
            dropout: Dropout probability for classification head
            use_snippet: Whether to use snippet in addition to title
            freeze_bert: If True, freeze BERT weights (only train classifier)
            
        Example:
            >>> model = RussianNewsClassifier(
            ...     model_name="DeepPavlov/rubert-base-cased",
            ...     num_labels=1000,
            ...     use_snippet=True
            ... )
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_snippet = use_snippet
        self.freeze_bert = freeze_bert
        
        # Load pre-trained BERT
        logger.info(f"Loading model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size
        
        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT weights frozen")
        
        # Classification head
        if use_snippet:
            # Concatenate title and snippet representations
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            # Title-only classification
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels),
            )
        
        logger.info(
            f"Initialized RussianNewsClassifier: "
            f"model={model_name}, num_labels={num_labels}, "
            f"hidden_size={hidden_size}, use_snippet={use_snippet}"
        )

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            title_input_ids: Title token IDs [batch_size, seq_len]
            title_attention_mask: Title attention mask [batch_size, seq_len]
            snippet_input_ids: Optional snippet token IDs [batch_size, seq_len]
            snippet_attention_mask: Optional snippet attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, num_labels]
        """
        def _pool(outputs: Any) -> torch.Tensor:
            """
            Get a single vector representation per sample.
            - BERT-like models: prefer `pooler_output` when present
            - DistilBERT-like models: fall back to CLS token from `last_hidden_state[:, 0]`
            """
            pooler = getattr(outputs, "pooler_output", None)
            if pooler is not None:
                return pooler
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise AttributeError("Model outputs missing both pooler_output and last_hidden_state")
            return last_hidden[:, 0]

        # Get title representation
        title_outputs = self.bert(
            input_ids=title_input_ids,
            attention_mask=title_attention_mask,
        )
        title_repr = _pool(title_outputs)  # [batch_size, hidden_size]
        
        if self.use_snippet:
            # If the model was trained with snippet support, the classifier expects
            # concatenated (title_repr, snippet_repr). For requests where snippet is
            # missing, we fall back to an all-zeros snippet vector so title-only
            # inference still works.
            if snippet_input_ids is not None:
                snippet_outputs = self.bert(
                    input_ids=snippet_input_ids,
                    attention_mask=snippet_attention_mask,
                )
                snippet_repr = _pool(snippet_outputs)  # [batch_size, hidden_size]
            else:
                snippet_repr = torch.zeros_like(title_repr)

            combined = torch.cat([title_repr, snippet_repr], dim=1)
            logits = self.classifier(combined)
        else:
            # Title-only architecture
            logits = self.classifier(title_repr)
        
        return logits


class MultilingualBERTClassifier(nn.Module):
    """
    Multilingual BERT classifier for comparison with Russian BERT.
    
    Uses bert-base-multilingual-cased for Russian text classification.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_labels: int = 1000,
        dropout: float = 0.3,
        use_snippet: bool = True,
    ):
        """
        Initialize multilingual BERT classifier.
        
        Args:
            model_name: HuggingFace model name (default: multilingual BERT)
            num_labels: Number of output classes
            dropout: Dropout probability
            use_snippet: Whether to use snippets
            
        Example:
            >>> model = MultilingualBERTClassifier(num_labels=1000)
        """
        super().__init__()
        # Reuse RussianNewsClassifier architecture
        self.base_model = RussianNewsClassifier(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            use_snippet=use_snippet,
        )
        
        logger.info(f"Initialized MultilingualBERTClassifier: {model_name}")

    def forward(
        self,
        title_input_ids: torch.Tensor,
        title_attention_mask: torch.Tensor,
        snippet_input_ids: Optional[torch.Tensor] = None,
        snippet_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass (delegates to base model)."""
        return self.base_model(
            title_input_ids=title_input_ids,
            title_attention_mask=title_attention_mask,
            snippet_input_ids=snippet_input_ids,
            snippet_attention_mask=snippet_attention_mask,
        )

