"""Predictive intervals for sentiment analysis using Beta distribution."""

import math
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_predictive_interval(
    positive_count: int,
    negative_count: int,
    neutral_count: int = 0,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate lower bound of predictive interval for positive comment ratio.
    
    Uses Beta distribution to model the proportion of positive comments.
    This accounts for uncertainty when sample size is small.
    
    Formula:
        a = 1 + u (positive comments)
        b = 1 + d (negative + neutral comments)
        Lower bound = mean - z_score * std_dev
    
    Args:
        positive_count: Number of positive comments
        negative_count: Number of negative comments
        neutral_count: Number of neutral comments (default: 0)
        confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
        
    Returns:
        Lower bound of predictive interval (0.0 to 1.0)
        
    Example:
        >>> # 80 positive, 20 negative out of 100 comments
        >>> lower_bound = calculate_predictive_interval(80, 20)
        >>> print(f"Lower bound: {lower_bound:.3f}")
        Lower bound: 0.742
    """
    u = positive_count
    d = negative_count + neutral_count
    
    # Beta distribution parameters
    a = 1 + u
    b = 1 + d
    
    # Mean of Beta distribution
    mean = a / (a + b)
    
    # Variance of Beta distribution
    variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
    std_dev = math.sqrt(variance)
    
    # Z-score for confidence level
    # 95% confidence: z = 1.65 (one-sided)
    # 99% confidence: z = 2.33 (one-sided)
    z_scores = {
        0.90: 1.28,
        0.95: 1.65,
        0.99: 2.33
    }
    z_score = z_scores.get(confidence_level, 1.65)
    
    # Lower bound of predictive interval
    lower_bound = mean - z_score * std_dev
    
    # Ensure non-negative and within [0, 1]
    lower_bound = max(0.0, min(1.0, lower_bound))
    
    return lower_bound


def rank_by_predictive_interval(
    data: List[Dict],
    positive_key: str = "positive_count",
    negative_key: str = "negative_count",
    neutral_key: str = "neutral_count",
    confidence_level: float = 0.95
) -> List[Dict]:
    """
    Rank items by predictive interval lower bound.
    
    This is useful for ranking news articles or categories by positive
    sentiment while accounting for sample size uncertainty.
    
    Args:
        data: List of dictionaries with sentiment counts
        positive_key: Key for positive count in data dict
        negative_key: Key for negative count in data dict
        neutral_key: Key for neutral count in data dict
        confidence_level: Confidence level for interval
        
    Returns:
        List of dictionaries sorted by predictive interval (descending)
        Each dict includes 'predictive_interval' field
        
    Example:
        >>> data = [
        ...     {"id": 1, "positive_count": 80, "negative_count": 20},
        ...     {"id": 2, "positive_count": 1, "negative_count": 0},
        ... ]
        >>> ranked = rank_by_predictive_interval(data)
        >>> ranked[0]["id"]  # First item has higher interval
        1
    """
    results = []
    
    for item in data:
        positive = item.get(positive_key, 0)
        negative = item.get(negative_key, 0)
        neutral = item.get(neutral_key, 0)
        
        interval = calculate_predictive_interval(
            positive_count=positive,
            negative_count=negative,
            neutral_count=neutral,
            confidence_level=confidence_level
        )
        
        # Create new dict with interval
        result = item.copy()
        result["predictive_interval"] = interval
        result["total_comments"] = positive + negative + neutral
        result["positive_ratio"] = positive / (positive + negative + neutral) if (positive + negative + neutral) > 0 else 0.0
        
        results.append(result)
    
    # Sort by predictive interval (descending)
    results.sort(key=lambda x: x["predictive_interval"], reverse=True)
    
    return results


def calculate_intervals_for_dataframe(
    df: pd.DataFrame,
    positive_col: str = "positive_count",
    negative_col: str = "negative_count",
    neutral_col: str = "neutral_count",
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate predictive intervals for DataFrame.
    
    Args:
        df: DataFrame with sentiment counts
        positive_col: Column name for positive counts
        negative_col: Column name for negative counts
        neutral_col: Column name for neutral counts
        confidence_level: Confidence level
        
    Returns:
        DataFrame with added 'predictive_interval' column
        
    Example:
        >>> df = pd.DataFrame({
        ...     "positive_count": [80, 1],
        ...     "negative_count": [20, 0]
        ... })
        >>> df_with_intervals = calculate_intervals_for_dataframe(df)
        >>> "predictive_interval" in df_with_intervals.columns
        True
    """
    df = df.copy()
    
    df["predictive_interval"] = df.apply(
        lambda row: calculate_predictive_interval(
            positive_count=row.get(positive_col, 0),
            negative_count=row.get(negative_col, 0),
            neutral_count=row.get(neutral_col, 0),
            confidence_level=confidence_level
        ),
        axis=1
    )
    
    return df


def get_top_positive_by_interval(
    data: List[Dict],
    top_k: int = 10,
    min_comments: int = 1,
    **kwargs
) -> List[Dict]:
    """
    Get top K items ranked by predictive interval.
    
    Args:
        data: List of dictionaries with sentiment counts
        top_k: Number of top items to return
        min_comments: Minimum number of comments required
        **kwargs: Additional arguments for rank_by_predictive_interval
        
    Returns:
        Top K items sorted by predictive interval
        
    Example:
        >>> data = [
        ...     {"id": 1, "positive_count": 80, "negative_count": 20},
        ...     {"id": 2, "positive_count": 1, "negative_count": 0},
        ... ]
        >>> top = get_top_positive_by_interval(data, top_k=1)
        >>> len(top)
        1
    """
    # Filter by minimum comments
    filtered = [
        item for item in data
        if (item.get("positive_count", 0) + 
            item.get("negative_count", 0) + 
            item.get("neutral_count", 0)) >= min_comments
    ]
    
    # Rank by predictive interval
    ranked = rank_by_predictive_interval(filtered, **kwargs)
    
    # Return top K
    return ranked[:top_k]


def get_top_negative_by_interval(
    data: List[Dict],
    top_k: int = 10,
    min_comments: int = 1,
    **kwargs
) -> List[Dict]:
    """
    Get top K items ranked by negative sentiment (lowest predictive interval).
    
    Args:
        data: List of dictionaries with sentiment counts
        top_k: Number of top items to return
        min_comments: Minimum number of comments required
        **kwargs: Additional arguments for rank_by_predictive_interval
        
    Returns:
        Top K items with lowest predictive intervals (most negative)
    """
    # Filter by minimum comments
    filtered = [
        item for item in data
        if (item.get("positive_count", 0) + 
            item.get("negative_count", 0) + 
            item.get("neutral_count", 0)) >= min_comments
    ]
    
    # Rank by predictive interval
    ranked = rank_by_predictive_interval(filtered, **kwargs)
    
    # Return bottom K (most negative)
    return ranked[-top_k:][::-1]  # Reverse to get most negative first



