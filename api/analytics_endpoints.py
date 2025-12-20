"""API endpoints for advanced analytics."""

import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from analysis.predictive_intervals import (
    calculate_predictive_interval,
    rank_by_predictive_interval,
    get_top_positive_by_interval,
    get_top_negative_by_interval
)
from analysis.category_analytics import CategoryAnalytics
from analysis.thread_analysis import ThreadAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Global analyzers (lazy loaded)
category_analytics: Optional[CategoryAnalytics] = None
thread_analyzer: Optional[ThreadAnalyzer] = None


def get_category_analytics() -> CategoryAnalytics:
    """Get or create category analytics instance."""
    global category_analytics
    if category_analytics is None:
        category_analytics = CategoryAnalytics()
    return category_analytics


def get_thread_analyzer() -> ThreadAnalyzer:
    """Get or create thread analyzer instance."""
    global thread_analyzer
    if thread_analyzer is None:
        thread_analyzer = ThreadAnalyzer()
    return thread_analyzer


# Request/Response Models
class SentimentCounts(BaseModel):
    """Sentiment counts for an item."""
    
    id: str = Field(..., description="Item identifier")
    positive_count: int = Field(..., description="Number of positive comments", ge=0)
    negative_count: int = Field(..., description="Number of negative comments", ge=0)
    neutral_count: int = Field(0, description="Number of neutral comments", ge=0)


class PredictiveIntervalRequest(BaseModel):
    """Request model for predictive interval calculation."""
    
    data: List[SentimentCounts] = Field(..., description="List of items with sentiment counts", min_items=1)
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.90, le=0.99)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"id": "news_1", "positive_count": 80, "negative_count": 20, "neutral_count": 0},
                    {"id": "news_2", "positive_count": 1, "negative_count": 0, "neutral_count": 0}
                ],
                "confidence_level": 0.95
            }
        }


class PredictiveIntervalResponse(BaseModel):
    """Response model for predictive interval calculation."""
    
    ranked_data: List[Dict] = Field(..., description="Items ranked by predictive interval")
    top_positive: List[Dict] = Field(..., description="Top positive items")
    top_negative: List[Dict] = Field(..., description="Top negative items")


class CategorySentimentRequest(BaseModel):
    """Request model for category sentiment analysis."""
    
    data: List[Dict[str, str]] = Field(..., description="List of items with category and text", min_items=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"category": "politics", "text": "Отличная новость!"},
                    {"category": "politics", "text": "Ужасная ситуация..."},
                    {"category": "economy", "text": "Нормально"}
                ]
            }
        }


class CategorySentimentResponse(BaseModel):
    """Response model for category sentiment analysis."""
    
    category_stats: Dict[str, Dict] = Field(..., description="Statistics per category")
    top_positive_categories: List[Dict] = Field(..., description="Top positive categories")
    top_negative_categories: List[Dict] = Field(..., description="Top negative categories")


class ThreadAnalysisRequest(BaseModel):
    """Request model for thread analysis."""
    
    data: List[Dict[str, str]] = Field(..., description="List of comments with news_id and text", min_items=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"news_id": "1", "text": "Отлично!"},
                    {"news_id": "1", "text": "Ужасно!"},
                    {"news_id": "2", "text": "Нормально"}
                ]
            }
        }


class ThreadAnalysisResponse(BaseModel):
    """Response model for thread analysis."""
    
    thread_stats: List[Dict] = Field(..., description="Thread statistics per news item")
    correlation: Dict = Field(..., description="Correlation analysis results")


# API Endpoints
@router.post("/predictive-intervals", response_model=PredictiveIntervalResponse)
async def calculate_predictive_intervals(request: PredictiveIntervalRequest):
    """
    Calculate predictive intervals for ranking items by positive sentiment.
    
    Uses Beta distribution to account for uncertainty when sample sizes are small.
    Useful for ranking news articles or categories by positive sentiment.
    
    Args:
        request: Request with sentiment counts and confidence level
        
    Returns:
        Ranked items with predictive intervals
    """
    try:
        # Convert to list of dicts
        data = [
            {
                "id": item.id,
                "positive_count": item.positive_count,
                "negative_count": item.negative_count,
                "neutral_count": item.neutral_count
            }
            for item in request.data
        ]
        
        # Rank by predictive interval
        ranked_data = rank_by_predictive_interval(
            data,
            confidence_level=request.confidence_level
        )
        
        # Get top positive and negative
        top_positive = get_top_positive_by_interval(
            data,
            top_k=10,
            confidence_level=request.confidence_level
        )
        top_negative = get_top_negative_by_interval(
            data,
            top_k=10,
            confidence_level=request.confidence_level
        )
        
        return PredictiveIntervalResponse(
            ranked_data=ranked_data,
            top_positive=top_positive,
            top_negative=top_negative
        )
    except Exception as e:
        logger.error(f"Error calculating predictive intervals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Predictive interval calculation failed: {str(e)}"
        )


@router.post("/category-sentiment", response_model=CategorySentimentResponse)
async def analyze_category_sentiment(request: CategorySentimentRequest):
    """
    Analyze sentiment distribution across categories.
    
    Calculates sentiment statistics for each category and ranks them
    using predictive intervals.
    
    Args:
        request: Request with category and text data
        
    Returns:
        Category sentiment statistics and rankings
    """
    try:
        analytics = get_category_analytics()
        
        # Analyze category sentiment
        category_stats = analytics.analyze_category_sentiment(
            request.data,
            category_key="category",
            text_key="text"
        )
        
        # Get top categories
        top_positive = analytics.get_top_positive_categories(
            category_stats,
            top_k=10
        )
        top_negative = analytics.get_top_negative_categories(
            category_stats,
            top_k=10
        )
        
        return CategorySentimentResponse(
            category_stats=category_stats,
            top_positive_categories=top_positive,
            top_negative_categories=top_negative
        )
    except Exception as e:
        logger.error(f"Error analyzing category sentiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Category sentiment analysis failed: {str(e)}"
        )


@router.post("/thread-analysis", response_model=ThreadAnalysisResponse)
async def analyze_thread_correlation(request: ThreadAnalysisRequest):
    """
    Analyze correlation between thread length and sentiment temperature.
    
    Thread length is the number of comments under a news article.
    Temperature is the probability that a comment is negative.
    
    Args:
        request: Request with news_id and text data
        
    Returns:
        Thread statistics and correlation analysis
    """
    try:
        analyzer = get_thread_analyzer()
        
        # Calculate thread lengths and temperatures
        thread_lengths = analyzer.calculate_thread_lengths(
            request.data,
            news_id_key="news_id"
        )
        temperatures = analyzer.calculate_temperature(
            request.data,
            news_id_key="news_id",
            text_key="text"
        )
        
        # Analyze correlation
        correlation_results = analyzer.analyze_correlation(
            thread_lengths,
            temperatures
        )
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_python(obj):
            """Recursively convert numpy types to Python native types."""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python(item) for item in obj]
            return obj
        
        correlation_results = convert_to_python(correlation_results)
        
        # Create thread statistics
        common_ids = set(thread_lengths.keys()) & set(temperatures.keys())
        thread_stats = [
            {
                "news_id": news_id,
                "thread_length": int(thread_lengths[news_id]),
                "temperature": float(temperatures[news_id])
            }
            for news_id in common_ids
        ]
        
        return ThreadAnalysisResponse(
            thread_stats=thread_stats,
            correlation=correlation_results
        )
    except Exception as e:
        logger.error(f"Error analyzing thread correlation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Thread analysis failed: {str(e)}"
        )


@router.get("/health")
async def analytics_health():
    """
    Health check for analytics service.
    
    Returns:
        Status of analytics components
    """
    try:
        return {
            "status": "healthy",
            "category_analytics_loaded": category_analytics is not None,
            "thread_analyzer_loaded": thread_analyzer is not None
        }
    except Exception as e:
        logger.error(f"Error checking analytics health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

