"""API endpoints for sentiment analysis."""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator

from analysis.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Global sentiment analyzer (lazy loaded)
sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create sentiment analyzer instance."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentAnalyzer()
    return sentiment_analyzer


# Request/Response Models
class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Это отличная новость! Очень позитивная информация."
            }
        }


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    
    label: str = Field(..., description="Sentiment label (POSITIVE/NEGATIVE/NEUTRAL)")
    score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    text: str = Field(..., description="Analyzed text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "POSITIVE",
                "score": 0.95,
                "text": "Это отличная новость!"
            }
        }


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment analysis."""
    
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=100)
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        # Filter out empty texts
        filtered = [text.strip() for text in v if text and text.strip()]
        if not filtered:
            raise ValueError("All texts are empty")
        return filtered
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Это отличная новость!",
                    "Ужасная ситуация...",
                    "Обычная информация."
                ]
            }
        }


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    
    results: List[SentimentResponse] = Field(..., description="List of sentiment analysis results")
    total: int = Field(..., description="Total number of texts analyzed")
    distribution: dict = Field(..., description="Sentiment distribution")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {"label": "POSITIVE", "score": 0.95, "text": "Отлично!"},
                    {"label": "NEGATIVE", "score": 0.87, "text": "Ужасно!"}
                ],
                "total": 2,
                "distribution": {"POSITIVE": 1, "NEGATIVE": 1, "NEUTRAL": 0}
            }
        }


class SentimentDistributionResponse(BaseModel):
    """Response model for sentiment distribution."""
    
    distribution: dict = Field(..., description="Sentiment distribution counts")
    total: int = Field(..., description="Total number of texts")
    average_score: float = Field(..., description="Average sentiment score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "distribution": {"POSITIVE": 50, "NEGATIVE": 30, "NEUTRAL": 20},
                "total": 100,
                "average_score": 0.65
            }
        }


# API Endpoints
@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of a single text.
    
    Uses Russian sentiment model (seara/rubert-tiny2-russian-sentiment)
    to classify text as POSITIVE, NEGATIVE, or NEUTRAL.
    
    Args:
        request: Sentiment analysis request with text
        
    Returns:
        Sentiment analysis result with label and confidence score
        
    Example:
        ```json
        {
            "text": "Это отличная новость!"
        }
        ```
    """
    try:
        analyzer = get_sentiment_analyzer()
        result = analyzer.analyze(request.text)
        
        return SentimentResponse(
            label=result["label"],
            score=result["score"],
            text=result["text"]
        )
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(request: BatchSentimentRequest):
    """
    Analyze sentiment of multiple texts in batch.
    
    Processes up to 100 texts at once for efficient batch processing.
    
    Args:
        request: Batch sentiment analysis request with list of texts
        
    Returns:
        Batch sentiment analysis results with distribution
        
    Example:
        ```json
        {
            "texts": [
                "Отличная новость!",
                "Ужасная ситуация..."
            ]
        }
        ```
    """
    try:
        analyzer = get_sentiment_analyzer()
        results = analyzer.analyze_batch(request.texts)
        
        # Get distribution
        distribution = analyzer.get_sentiment_distribution(request.texts)
        
        # Convert to response format
        sentiment_results = [
            SentimentResponse(
                label=r["label"],
                score=r["score"],
                text=r["text"]
            )
            for r in results
        ]
        
        return BatchSentimentResponse(
            results=sentiment_results,
            total=len(results),
            distribution=distribution
        )
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch sentiment analysis failed: {str(e)}"
        )


@router.post("/distribution", response_model=SentimentDistributionResponse)
async def get_sentiment_distribution(request: BatchSentimentRequest):
    """
    Get sentiment distribution for a list of texts.
    
    Calculates the distribution of POSITIVE, NEGATIVE, and NEUTRAL
    sentiments across the provided texts.
    
    Args:
        request: Batch sentiment analysis request with list of texts
        
    Returns:
        Sentiment distribution statistics
        
    Example:
        ```json
        {
            "texts": [
                "Отлично!",
                "Ужасно!",
                "Нормально"
            ]
        }
        ```
    """
    try:
        analyzer = get_sentiment_analyzer()
        distribution = analyzer.get_sentiment_distribution(request.texts)
        average_score = analyzer.get_average_sentiment_score(request.texts)
        
        return SentimentDistributionResponse(
            distribution=distribution,
            total=len(request.texts),
            average_score=average_score
        )
    except Exception as e:
        logger.error(f"Error calculating sentiment distribution: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment distribution calculation failed: {str(e)}"
        )


@router.get("/health")
async def sentiment_health():
    """
    Health check for sentiment analysis service.
    
    Returns:
        Status of sentiment analyzer
    """
    try:
        analyzer = get_sentiment_analyzer()
        return {
            "status": "healthy",
            "model": analyzer.model_name,
            "device": analyzer.device,
            "loaded": analyzer._pipeline is not None
        }
    except Exception as e:
        logger.error(f"Error checking sentiment health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }



