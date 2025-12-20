"""Sentiment analysis using HuggingFace pipeline for Russian text."""

import logging
from typing import List, Dict, Optional
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis for Russian text using HuggingFace pipeline.
    
    Uses the seara/rubert-tiny2-russian-sentiment model for sentiment
    classification of Russian comments and text.
    
    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze("Отличная новость!")
        >>> print(result)
        {'label': 'POSITIVE', 'score': 0.95}
    """
    
    def __init__(
        self,
        model_name: str = "seara/rubert-tiny2-russian-sentiment",
        device: Optional[str] = None
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self._pipeline = None
        
        logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
        logger.info(f"Using device: {device}")
    
    @property
    def pipeline(self):
        """Lazy load pipeline to avoid loading on import."""
        if self._pipeline is None:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Sentiment model loaded successfully")
        return self._pipeline
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze (Russian)
            
        Returns:
            Dictionary with 'label' (POSITIVE/NEGATIVE/NEUTRAL) and 'score' (confidence)
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze("Это отличная новость!")
            >>> result['label']
            'POSITIVE'
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "text": text
            }
        
        try:
            result = self.pipeline(text)[0]
            
            # Normalize label names (model might return different formats)
            label = result["label"].upper()
            if "POSITIVE" in label or "POS" in label:
                label = "POSITIVE"
            elif "NEGATIVE" in label or "NEG" in label:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "label": label,
                "score": float(result["score"]),
                "text": text
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "label": "NEUTRAL",
                "score": 0.5,
                "text": text,
                "error": str(e)
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Отлично!", "Ужасно!"]
            >>> results = analyzer.analyze_batch(texts)
            >>> len(results)
            2
        """
        if not texts:
            return []
        
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        
        return results
    
    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, int]:
        """
        Get sentiment distribution for a list of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with counts for each sentiment label
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Отлично!", "Ужасно!", "Нормально"]
            >>> dist = analyzer.get_sentiment_distribution(texts)
            >>> dist['POSITIVE']
            1
        """
        results = self.analyze_batch(texts)
        
        distribution = {
            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0
        }
        
        for result in results:
            label = result.get("label", "NEUTRAL")
            distribution[label] = distribution.get(label, 0) + 1
        
        return distribution
    
    def get_average_sentiment_score(self, texts: List[str]) -> float:
        """
        Get average sentiment score across texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Average sentiment score (0.0 to 1.0)
        """
        if not texts:
            return 0.5
        
        results = self.analyze_batch(texts)
        scores = [r.get("score", 0.5) for r in results]
        
        # Normalize: POSITIVE = score, NEGATIVE = 1 - score, NEUTRAL = 0.5
        normalized_scores = []
        for result in results:
            label = result.get("label", "NEUTRAL")
            score = result.get("score", 0.5)
            
            if label == "POSITIVE":
                normalized_scores.append(score)
            elif label == "NEGATIVE":
                normalized_scores.append(1.0 - score)
            else:
                normalized_scores.append(0.5)
        
        return sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.5



