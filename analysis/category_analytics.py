"""Category-level sentiment analytics."""

import logging
from typing import List, Dict, Optional
import pandas as pd
from collections import defaultdict

from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.predictive_intervals import (
    calculate_predictive_interval,
    rank_by_predictive_interval,
    get_top_positive_by_interval,
    get_top_negative_by_interval
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoryAnalytics:
    """
    Analytics for category-level sentiment analysis.
    
    Analyzes sentiment distribution across different categories
    and ranks them using predictive intervals.
    """
    
    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize category analytics.
        
        Args:
            sentiment_analyzer: SentimentAnalyzer instance (creates new if None)
        """
        if sentiment_analyzer is None:
            self.analyzer = SentimentAnalyzer()
        else:
            self.analyzer = sentiment_analyzer
    
    def analyze_category_sentiment(
        self,
        data: List[Dict],
        category_key: str = "category",
        text_key: str = "text"
    ) -> Dict[str, Dict]:
        """
        Analyze sentiment for each category.
        
        Args:
            data: List of dictionaries with category and text
            category_key: Key for category in data dict
            text_key: Key for text in data dict
            
        Returns:
            Dictionary mapping category to sentiment statistics
            
        Example:
            >>> analytics = CategoryAnalytics()
            >>> data = [
            ...     {"category": "politics", "text": "Отлично!"},
            ...     {"category": "politics", "text": "Ужасно!"},
            ... ]
            >>> stats = analytics.analyze_category_sentiment(data)
            >>> "politics" in stats
            True
        """
        # Group by category
        category_data = defaultdict(list)
        for item in data:
            category = item.get(category_key)
            text = item.get(text_key)
            if category and text:
                category_data[category].append(text)
        
        # Analyze sentiment for each category
        category_stats = {}
        
        for category, texts in category_data.items():
            # Analyze all texts in category
            results = self.analyzer.analyze_batch(texts)
            
            # Count sentiments
            positive_count = sum(1 for r in results if r["label"] == "POSITIVE")
            negative_count = sum(1 for r in results if r["label"] == "NEGATIVE")
            neutral_count = sum(1 for r in results if r["label"] == "NEUTRAL")
            
            # Calculate predictive interval
            predictive_interval = calculate_predictive_interval(
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count
            )
            
            # Calculate average score
            avg_score = self.analyzer.get_average_sentiment_score(texts)
            
            category_stats[category] = {
                "category": category,
                "total_comments": len(texts),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_ratio": positive_count / len(texts) if texts else 0.0,
                "negative_ratio": negative_count / len(texts) if texts else 0.0,
                "neutral_ratio": neutral_count / len(texts) if texts else 0.0,
                "predictive_interval": predictive_interval,
                "average_score": avg_score
            }
        
        return category_stats
    
    def rank_categories_by_sentiment(
        self,
        category_stats: Dict[str, Dict],
        sort_by: str = "predictive_interval"
    ) -> List[Dict]:
        """
        Rank categories by sentiment metric.
        
        Args:
            category_stats: Dictionary of category statistics
            sort_by: Metric to sort by ('predictive_interval', 'positive_ratio', 'average_score')
            
        Returns:
            List of category stats sorted by metric (descending)
        """
        categories = list(category_stats.values())
        
        # Sort by specified metric
        if sort_by == "predictive_interval":
            categories.sort(key=lambda x: x.get("predictive_interval", 0), reverse=True)
        elif sort_by == "positive_ratio":
            categories.sort(key=lambda x: x.get("positive_ratio", 0), reverse=True)
        elif sort_by == "average_score":
            categories.sort(key=lambda x: x.get("average_score", 0), reverse=True)
        else:
            logger.warning(f"Unknown sort_by: {sort_by}, using predictive_interval")
            categories.sort(key=lambda x: x.get("predictive_interval", 0), reverse=True)
        
        return categories
    
    def get_top_positive_categories(
        self,
        category_stats: Dict[str, Dict],
        top_k: int = 10,
        min_comments: int = 1
    ) -> List[Dict]:
        """
        Get top K categories by positive sentiment.
        
        Args:
            category_stats: Dictionary of category statistics
            top_k: Number of top categories to return
            min_comments: Minimum number of comments required
            
        Returns:
            Top K categories sorted by predictive interval
        """
        # Convert to list format for ranking
        data = [
            {
                "category": stats["category"],
                "positive_count": stats["positive_count"],
                "negative_count": stats["negative_count"],
                "neutral_count": stats["neutral_count"]
            }
            for stats in category_stats.values()
        ]
        
        return get_top_positive_by_interval(
            data,
            top_k=top_k,
            min_comments=min_comments
        )
    
    def get_top_negative_categories(
        self,
        category_stats: Dict[str, Dict],
        top_k: int = 10,
        min_comments: int = 1
    ) -> List[Dict]:
        """
        Get top K categories by negative sentiment.
        
        Args:
            category_stats: Dictionary of category statistics
            top_k: Number of top categories to return
            min_comments: Minimum number of comments required
            
        Returns:
            Top K categories with most negative sentiment
        """
        # Convert to list format for ranking
        data = [
            {
                "category": stats["category"],
                "positive_count": stats["positive_count"],
                "negative_count": stats["negative_count"],
                "neutral_count": stats["neutral_count"]
            }
            for stats in category_stats.values()
        ]
        
        return get_top_negative_by_interval(
            data,
            top_k=top_k,
            min_comments=min_comments
        )
    
    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
        text_col: str = "text"
    ) -> pd.DataFrame:
        """
        Analyze category sentiment from DataFrame.
        
        Args:
            df: DataFrame with category and text columns
            category_col: Name of category column
            text_col: Name of text column
            
        Returns:
            DataFrame with category sentiment statistics
        """
        # Convert DataFrame to list of dicts
        data = df[[category_col, text_col]].to_dict('records')
        
        # Analyze sentiment
        category_stats = self.analyze_category_sentiment(
            data,
            category_key=category_col,
            text_key=text_col
        )
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(list(category_stats.values()))
        
        return stats_df



