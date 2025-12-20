"""Pydantic schemas for API request/response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ClassificationRequest(BaseModel):
    """Request model for classification."""
    
    title: str = Field(..., description="News article title", min_length=1, max_length=500)
    snippet: Optional[str] = Field(None, description="News article snippet", max_length=2000)
    threshold: float = Field(0.5, description="Classification threshold", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, description="Return top K predictions", ge=1, le=100)
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()
    
    @validator('snippet')
    def validate_snippet(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return None


class TagPrediction(BaseModel):
    """Single tag prediction."""
    
    tag: str = Field(..., description="Tag name")
    score: float = Field(..., description="Prediction score", ge=0.0, le=1.0)


class ClassificationResponse(BaseModel):
    """Response model for classification."""
    
    predictions: List[TagPrediction] = Field(..., description="List of tag predictions")
    title: str = Field(..., description="Processed title")
    snippet: Optional[str] = Field(None, description="Processed snippet")
    threshold: float = Field(..., description="Threshold used")
    model_version: str = Field(..., description="Model version")

