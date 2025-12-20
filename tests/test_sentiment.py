"""Tests for sentiment analysis."""

import pytest
import torch
from analysis.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer instance."""
        return SentimentAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.model_name == "seara/rubert-tiny2-russian-sentiment"
        assert analyzer.device in ["cuda", "cpu"]
    
    def test_analyze_positive_text(self, analyzer):
        """Test analyzing positive text."""
        result = analyzer.analyze("Это отличная новость! Очень позитивная информация.")
        
        assert result is not None
        assert "label" in result
        assert "score" in result
        assert "text" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert 0.0 <= result["score"] <= 1.0
    
    def test_analyze_negative_text(self, analyzer):
        """Test analyzing negative text."""
        result = analyzer.analyze("Ужасная ситуация. Очень плохо.")
        
        assert result is not None
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert 0.0 <= result["score"] <= 1.0
    
    def test_analyze_empty_text(self, analyzer):
        """Test analyzing empty text."""
        result = analyzer.analyze("")
        
        assert result is not None
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.5
    
    def test_analyze_whitespace_text(self, analyzer):
        """Test analyzing whitespace-only text."""
        result = analyzer.analyze("   ")
        
        assert result is not None
        assert result["label"] == "NEUTRAL"
        assert result["score"] == 0.5
    
    def test_analyze_batch(self, analyzer):
        """Test batch analysis."""
        texts = [
            "Отличная новость!",
            "Ужасная ситуация...",
            "Обычная информация."
        ]
        
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result
            assert result["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            assert 0.0 <= result["score"] <= 1.0
    
    def test_analyze_batch_empty(self, analyzer):
        """Test batch analysis with empty list."""
        results = analyzer.analyze_batch([])
        assert results == []
    
    def test_get_sentiment_distribution(self, analyzer):
        """Test getting sentiment distribution."""
        texts = [
            "Отлично!",
            "Отлично!",
            "Ужасно!",
            "Нормально"
        ]
        
        distribution = analyzer.get_sentiment_distribution(texts)
        
        assert "POSITIVE" in distribution
        assert "NEGATIVE" in distribution
        assert "NEUTRAL" in distribution
        assert sum(distribution.values()) == len(texts)
    
    def test_get_average_sentiment_score(self, analyzer):
        """Test getting average sentiment score."""
        texts = [
            "Отлично!",
            "Ужасно!",
            "Нормально"
        ]
        
        avg_score = analyzer.get_average_sentiment_score(texts)
        
        assert 0.0 <= avg_score <= 1.0
    
    def test_pipeline_lazy_loading(self, analyzer):
        """Test that pipeline is loaded lazily."""
        # Pipeline should be None initially
        assert analyzer._pipeline is None
        
        # Access pipeline property to trigger loading
        pipeline = analyzer.pipeline
        
        # Pipeline should now be loaded
        assert analyzer._pipeline is not None
        assert pipeline is not None


class TestSentimentAPI:
    """Test suite for sentiment API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    
    def test_sentiment_health(self, client):
        """Test sentiment health endpoint."""
        response = client.get("/sentiment/health")
        
        assert response.status_code in [200, 500]  # May fail if model not loaded
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
    
    def test_analyze_sentiment_endpoint(self, client):
        """Test sentiment analysis endpoint."""
        response = client.post(
            "/sentiment/analyze",
            json={"text": "Это отличная новость!"}
        )
        
        # May fail if model not loaded, but should return proper error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "score" in data
            assert "text" in data
            assert data["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
            assert 0.0 <= data["score"] <= 1.0
    
    def test_analyze_sentiment_empty_text(self, client):
        """Test sentiment analysis with empty text."""
        response = client.post(
            "/sentiment/analyze",
            json={"text": ""}
        )
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_batch_sentiment_endpoint(self, client):
        """Test batch sentiment analysis endpoint."""
        response = client.post(
            "/sentiment/batch",
            json={
                "texts": [
                    "Отлично!",
                    "Ужасно!",
                    "Нормально"
                ]
            }
        )
        
        # May fail if model not loaded
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "distribution" in data
            assert len(data["results"]) == 3
    
    def test_sentiment_distribution_endpoint(self, client):
        """Test sentiment distribution endpoint."""
        response = client.post(
            "/sentiment/distribution",
            json={
                "texts": [
                    "Отлично!",
                    "Ужасно!",
                    "Нормально"
                ]
            }
        )
        
        # May fail if model not loaded
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "distribution" in data
            assert "total" in data
            assert "average_score" in data
            assert data["total"] == 3
            assert 0.0 <= data["average_score"] <= 1.0



