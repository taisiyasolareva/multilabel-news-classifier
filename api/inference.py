"""Inference utilities for API."""

import torch
from typing import List, Optional, Dict
import logging

from models.transformer_model import RussianNewsClassifier
from utils.tokenization import RussianTextTokenizer
from utils.russian_text_utils import prepare_text_for_tokenization
from api.schemas import TagPrediction

logger = logging.getLogger(__name__)


class ModelInference:
    """
    Model inference handler.
    
    Handles model loading, caching, and async inference.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "DeepPavlov/rubert-base-cased",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize inference handler.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer_name: HuggingFace tokenizer name
            device: Device for inference
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        self.tag_to_idx = None
        self.loaded = False

    def load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            from utils.tokenization import create_tokenizer
            self.tokenizer = create_tokenizer(self.tokenizer_name)
            logger.info("Tokenizer loaded")
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    num_labels = checkpoint.get('num_labels', 1000)
                    self.model = RussianNewsClassifier(
                        model_name=self.tokenizer_name,
                        num_labels=num_labels,
                        use_snippet=True,
                    )
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model = checkpoint
            else:
                self.model = checkpoint
            
            # Load tag mapping if available
            if isinstance(checkpoint, dict) and 'tag_to_idx' in checkpoint:
                self.tag_to_idx = checkpoint['tag_to_idx']
            
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.loaded = False
            raise

    def predict(
        self,
        title: str,
        snippet: Optional[str] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> List[TagPrediction]:
        """
        Run inference.
        
        Args:
            title: Article title
            snippet: Optional article snippet
            threshold: Classification threshold
            top_k: Return top K predictions
            
        Returns:
            List of tag predictions
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Prepare text
        title_clean = prepare_text_for_tokenization(title)
        snippet_clean = prepare_text_for_tokenization(snippet) if snippet else None
        
        # Tokenize
        title_encoded = self.tokenizer.encode(
            title_clean,
            max_length=128,
            padding='max_length',
            truncation=True,
        )
        
        title_input_ids = title_encoded['input_ids'].unsqueeze(0).to(self.device)
        title_attention_mask = title_encoded['attention_mask'].unsqueeze(0).to(self.device)
        
        snippet_input_ids = None
        snippet_attention_mask = None
        
        if snippet_clean:
            snippet_encoded = self.tokenizer.encode(
                snippet_clean,
                max_length=256,
                padding='max_length',
                truncation=True,
            )
            snippet_input_ids = snippet_encoded['input_ids'].unsqueeze(0).to(self.device)
            snippet_attention_mask = snippet_encoded['attention_mask'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(
                title_input_ids=title_input_ids,
                title_attention_mask=title_attention_mask,
                snippet_input_ids=snippet_input_ids,
                snippet_attention_mask=snippet_attention_mask,
            )
            
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Convert to predictions
        predictions = []
        
        if self.tag_to_idx:
            # Use provided tag mapping
            idx_to_tag = {v: k for k, v in self.tag_to_idx.items()}
            for idx, prob in enumerate(probs):
                if prob >= threshold:
                    tag = idx_to_tag.get(idx, f"tag_{idx}")
                    predictions.append(TagPrediction(tag=tag, score=float(prob)))
        else:
            # Generic tag indices
            for idx, prob in enumerate(probs):
                if prob >= threshold:
                    predictions.append(TagPrediction(tag=f"tag_{idx}", score=float(prob)))
        
        # Sort by score and apply top_k
        predictions.sort(key=lambda x: x.score, reverse=True)
        
        if top_k:
            predictions = predictions[:top_k]
        
        return predictions

