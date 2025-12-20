"""Thread length and sentiment correlation analysis."""

import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from analysis.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadAnalyzer:
    """
    Analyze correlation between thread length and sentiment.
    
    Thread length is the number of comments under a news article.
    Temperature is the probability that a comment is negative.
    """
    
    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize thread analyzer.
        
        Args:
            sentiment_analyzer: SentimentAnalyzer instance (creates new if None)
        """
        if sentiment_analyzer is None:
            self.analyzer = SentimentAnalyzer()
        else:
            self.analyzer = sentiment_analyzer
    
    def calculate_thread_lengths(
        self,
        data: List[Dict],
        news_id_key: str = "news_id",
        comment_id_key: str = "id"
    ) -> Dict[str, int]:
        """
        Calculate thread length (number of comments) for each news item.
        
        Args:
            data: List of comment dictionaries
            news_id_key: Key for news ID in data dict
            comment_id_key: Key for comment ID in data dict
            
        Returns:
            Dictionary mapping news_id to thread length
            
        Example:
            >>> analyzer = ThreadAnalyzer()
            >>> data = [
            ...     {"news_id": 1, "id": 1},
            ...     {"news_id": 1, "id": 2},
            ...     {"news_id": 2, "id": 3},
            ... ]
            >>> lengths = analyzer.calculate_thread_lengths(data)
            >>> lengths[1]
            2
        """
        thread_lengths = {}
        
        for item in data:
            news_id = item.get(news_id_key)
            if news_id:
                thread_lengths[news_id] = thread_lengths.get(news_id, 0) + 1
        
        return thread_lengths
    
    def calculate_temperature(
        self,
        data: List[Dict],
        news_id_key: str = "news_id",
        text_key: str = "text"
    ) -> Dict[str, float]:
        """
        Calculate temperature (negative sentiment probability) for each news item.
        
        Temperature is the probability that a comment is negative.
        
        Args:
            data: List of comment dictionaries
            news_id_key: Key for news ID in data dict
            text_key: Key for text in data dict
            
        Returns:
            Dictionary mapping news_id to average temperature
        """
        # Group comments by news_id
        news_comments = {}
        for item in data:
            news_id = item.get(news_id_key)
            text = item.get(text_key)
            if news_id and text:
                if news_id not in news_comments:
                    news_comments[news_id] = []
                news_comments[news_id].append(text)
        
        # Calculate temperature for each news item
        temperatures = {}
        
        for news_id, texts in news_comments.items():
            # Analyze sentiment
            results = self.analyzer.analyze_batch(texts)
            
            # Calculate average temperature (probability of negative)
            negative_scores = []
            for result in results:
                label = result["label"]
                score = result["score"]
                
                if label == "NEGATIVE":
                    # High confidence negative = high temperature
                    negative_scores.append(score)
                elif label == "POSITIVE":
                    # High confidence positive = low temperature
                    negative_scores.append(1.0 - score)
                else:
                    # Neutral = medium temperature
                    negative_scores.append(0.5)
            
            avg_temperature = np.mean(negative_scores) if negative_scores else 0.5
            temperatures[news_id] = avg_temperature
        
        return temperatures
    
    def analyze_correlation(
        self,
        thread_lengths: Dict[str, int],
        temperatures: Dict[str, float]
    ) -> Dict:
        """
        Analyze correlation between thread length and temperature.
        
        Args:
            thread_lengths: Dictionary mapping news_id to thread length
            temperatures: Dictionary mapping news_id to temperature
            
        Returns:
            Dictionary with correlation statistics
        """
        # Get common news_ids
        common_ids = set(thread_lengths.keys()) & set(temperatures.keys())
        
        if len(common_ids) < 2:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "significant": False,
                "sample_size": len(common_ids),
                "error": "Insufficient data for correlation analysis"
            }
        
        # Prepare data
        lengths = [thread_lengths[id] for id in common_ids]
        temps = [temperatures[id] for id in common_ids]
        
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(lengths, temps)
        
        # Linear regression
        X = np.array(lengths).reshape(-1, 1)
        y = np.array(temps)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        slope = reg.coef_[0]
        intercept = reg.intercept_
        r_squared = reg.score(X, y)
        
        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "sample_size": len(common_ids),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "interpretation": self._interpret_correlation(correlation, p_value)
        }
    
    def _interpret_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret correlation results."""
        if p_value >= 0.05:
            return "No significant correlation (p >= 0.05)"
        
        if abs(correlation) < 0.1:
            strength = "negligible"
        elif abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.5:
            strength = "moderate"
        elif abs(correlation) < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        return f"{strength.capitalize()} {direction} correlation (r={correlation:.3f}, p={p_value:.4f})"
    
    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        news_id_col: str = "news_id",
        text_col: str = "text"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze thread-sentiment correlation from DataFrame.
        
        Args:
            df: DataFrame with news_id and text columns
            news_id_col: Name of news_id column
            text_col: Name of text column
            
        Returns:
            Tuple of (DataFrame with thread stats, correlation results)
        """
        # Convert to list of dicts
        data = df[[news_id_col, text_col]].to_dict('records')
        
        # Calculate thread lengths and temperatures
        thread_lengths = self.calculate_thread_lengths(
            data,
            news_id_key=news_id_col
        )
        temperatures = self.calculate_temperature(
            data,
            news_id_key=news_id_col,
            text_key=text_col
        )
        
        # Analyze correlation
        correlation_results = self.analyze_correlation(thread_lengths, temperatures)
        
        # Create DataFrame with thread statistics
        common_ids = set(thread_lengths.keys()) & set(temperatures.keys())
        thread_stats = pd.DataFrame([
            {
                "news_id": news_id,
                "thread_length": thread_lengths[news_id],
                "temperature": temperatures[news_id]
            }
            for news_id in common_ids
        ])
        
        return thread_stats, correlation_results



