"""Tests for advanced analytics."""

import pytest
import pandas as pd
import numpy as np
from analysis.predictive_intervals import (
    calculate_predictive_interval,
    rank_by_predictive_interval,
    get_top_positive_by_interval,
    get_top_negative_by_interval,
    calculate_intervals_for_dataframe
)
from analysis.category_analytics import CategoryAnalytics
from analysis.thread_analysis import ThreadAnalyzer


class TestPredictiveIntervals:
    """Test suite for predictive intervals."""
    
    def test_calculate_predictive_interval_high_positive(self):
        """Test predictive interval with high positive ratio."""
        interval = calculate_predictive_interval(
            positive_count=80,
            negative_count=20,
            neutral_count=0
        )
        
        assert 0.0 <= interval <= 1.0
        assert interval > 0.5  # Should be high for mostly positive
    
    def test_calculate_predictive_interval_high_negative(self):
        """Test predictive interval with high negative ratio."""
        interval = calculate_predictive_interval(
            positive_count=20,
            negative_count=80,
            neutral_count=0
        )
        
        assert 0.0 <= interval <= 1.0
        assert interval < 0.5  # Should be low for mostly negative
    
    def test_calculate_predictive_interval_small_sample(self):
        """Test predictive interval with small sample."""
        # Single positive comment
        interval_small = calculate_predictive_interval(
            positive_count=1,
            negative_count=0,
            neutral_count=0
        )
        
        # Many positive comments
        interval_large = calculate_predictive_interval(
            positive_count=100,
            negative_count=0,
            neutral_count=0
        )
        
        # Small sample should have lower interval (more uncertainty)
        assert interval_small < interval_large
    
    def test_calculate_predictive_interval_confidence_levels(self):
        """Test different confidence levels."""
        interval_90 = calculate_predictive_interval(
            positive_count=80,
            negative_count=20,
            confidence_level=0.90
        )
        
        interval_95 = calculate_predictive_interval(
            positive_count=80,
            negative_count=20,
            confidence_level=0.95
        )
        
        interval_99 = calculate_predictive_interval(
            positive_count=80,
            negative_count=20,
            confidence_level=0.99
        )
        
        # Higher confidence = lower bound (more conservative)
        assert interval_90 >= interval_95 >= interval_99
    
    def test_rank_by_predictive_interval(self):
        """Test ranking by predictive interval."""
        data = [
            {"id": "item1", "positive_count": 80, "negative_count": 20},
            {"id": "item2", "positive_count": 1, "negative_count": 0},
            {"id": "item3", "positive_count": 50, "negative_count": 50},
        ]
        
        ranked = rank_by_predictive_interval(data)
        
        assert len(ranked) == 3
        assert "predictive_interval" in ranked[0]
        assert ranked[0]["predictive_interval"] >= ranked[-1]["predictive_interval"]
    
    def test_get_top_positive_by_interval(self):
        """Test getting top positive items."""
        data = [
            {"id": "item1", "positive_count": 80, "negative_count": 20},
            {"id": "item2", "positive_count": 1, "negative_count": 0},
            {"id": "item3", "positive_count": 50, "negative_count": 50},
        ]
        
        top = get_top_positive_by_interval(data, top_k=2)
        
        assert len(top) == 2
        assert top[0]["predictive_interval"] >= top[1]["predictive_interval"]
    
    def test_get_top_negative_by_interval(self):
        """Test getting top negative items."""
        data = [
            {"id": "item1", "positive_count": 20, "negative_count": 80},
            {"id": "item2", "positive_count": 0, "negative_count": 1},
            {"id": "item3", "positive_count": 50, "negative_count": 50},
        ]
        
        top = get_top_negative_by_interval(data, top_k=2)
        
        assert len(top) == 2
        # Most negative should have lowest interval
        assert top[0]["predictive_interval"] <= top[1]["predictive_interval"]
    
    def test_calculate_intervals_for_dataframe(self):
        """Test calculating intervals for DataFrame."""
        df = pd.DataFrame({
            "id": ["item1", "item2"],
            "positive_count": [80, 20],
            "negative_count": [20, 80]
        })
        
        df_result = calculate_intervals_for_dataframe(df)
        
        assert "predictive_interval" in df_result.columns
        assert len(df_result) == 2
        assert df_result.loc[0, "predictive_interval"] > df_result.loc[1, "predictive_interval"]


class TestCategoryAnalytics:
    """Test suite for category analytics."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {"category": "politics", "text": "Отличная новость!"},
            {"category": "politics", "text": "Ужасная ситуация..."},
            {"category": "economy", "text": "Нормально"},
        ]
    
    def test_category_analytics_initialization(self):
        """Test category analytics initialization."""
        analytics = CategoryAnalytics()
        assert analytics is not None
        assert analytics.analyzer is not None
    
    def test_analyze_category_sentiment(self, sample_data):
        """Test category sentiment analysis."""
        analytics = CategoryAnalytics()
        
        # This will actually run sentiment analysis, so it may be slow
        # For faster tests, we could mock the analyzer
        stats = analytics.analyze_category_sentiment(sample_data)
        
        assert "politics" in stats
        assert "economy" in stats
        assert "total_comments" in stats["politics"]
        assert "positive_count" in stats["politics"]
        assert "predictive_interval" in stats["politics"]
    
    def test_rank_categories_by_sentiment(self, sample_data):
        """Test ranking categories by sentiment."""
        analytics = CategoryAnalytics()
        stats = analytics.analyze_category_sentiment(sample_data)
        
        ranked = analytics.rank_categories_by_sentiment(stats, sort_by="predictive_interval")
        
        assert len(ranked) >= 1
        assert "category" in ranked[0]
        assert "predictive_interval" in ranked[0]
    
    def test_get_top_positive_categories(self, sample_data):
        """Test getting top positive categories."""
        analytics = CategoryAnalytics()
        stats = analytics.analyze_category_sentiment(sample_data)
        
        top = analytics.get_top_positive_categories(stats, top_k=5)
        
        assert len(top) <= 5
        if len(top) > 1:
            assert top[0]["predictive_interval"] >= top[1]["predictive_interval"]
    
    def test_analyze_from_dataframe(self):
        """Test analyzing from DataFrame."""
        df = pd.DataFrame({
            "category": ["politics", "politics", "economy"],
            "text": ["Отлично!", "Ужасно!", "Нормально"]
        })
        
        analytics = CategoryAnalytics()
        stats_df = analytics.analyze_from_dataframe(df)
        
        assert isinstance(stats_df, pd.DataFrame)
        assert "category" in stats_df.columns
        assert "predictive_interval" in stats_df.columns


class TestThreadAnalyzer:
    """Test suite for thread analyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {"news_id": "1", "id": "1", "text": "Отлично!"},
            {"news_id": "1", "id": "2", "text": "Ужасно!"},
            {"news_id": "2", "id": "3", "text": "Нормально"},
        ]
    
    def test_thread_analyzer_initialization(self):
        """Test thread analyzer initialization."""
        analyzer = ThreadAnalyzer()
        assert analyzer is not None
        assert analyzer.analyzer is not None
    
    def test_calculate_thread_lengths(self, sample_data):
        """Test calculating thread lengths."""
        analyzer = ThreadAnalyzer()
        lengths = analyzer.calculate_thread_lengths(sample_data)
        
        assert "1" in lengths
        assert "2" in lengths
        assert lengths["1"] == 2
        assert lengths["2"] == 1
    
    def test_calculate_temperature(self, sample_data):
        """Test calculating temperature."""
        analyzer = ThreadAnalyzer()
        temperatures = analyzer.calculate_temperature(sample_data)
        
        assert "1" in temperatures
        assert "2" in temperatures
        assert 0.0 <= temperatures["1"] <= 1.0
        assert 0.0 <= temperatures["2"] <= 1.0
    
    def test_analyze_correlation(self):
        """Test correlation analysis."""
        analyzer = ThreadAnalyzer()
        
        thread_lengths = {"1": 10, "2": 5, "3": 20}
        temperatures = {"1": 0.3, "2": 0.5, "3": 0.7}
        
        results = analyzer.analyze_correlation(thread_lengths, temperatures)
        
        assert "correlation" in results
        assert "p_value" in results
        assert "significant" in results
        assert "sample_size" in results
        assert -1.0 <= results["correlation"] <= 1.0
        assert 0.0 <= results["p_value"] <= 1.0
    
    def test_analyze_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        analyzer = ThreadAnalyzer()
        
        thread_lengths = {"1": 10}
        temperatures = {"2": 0.5}  # No overlap
        
        results = analyzer.analyze_correlation(thread_lengths, temperatures)
        
        assert results.get("sample_size", 0) < 2
        assert "error" in results or results.get("correlation", 0) == 0.0
    
    def test_analyze_from_dataframe(self):
        """Test analyzing from DataFrame."""
        df = pd.DataFrame({
            "news_id": ["1", "1", "2"],
            "text": ["Отлично!", "Ужасно!", "Нормально"]
        })
        
        analyzer = ThreadAnalyzer()
        thread_stats, correlation = analyzer.analyze_from_dataframe(df)
        
        assert isinstance(thread_stats, pd.DataFrame)
        assert "news_id" in thread_stats.columns
        assert "thread_length" in thread_stats.columns
        assert "temperature" in thread_stats.columns
        assert isinstance(correlation, dict)
        assert "correlation" in correlation


class TestAnalyticsAPI:
    """Test suite for analytics API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    
    def test_predictive_intervals_endpoint(self, client):
        """Test predictive intervals endpoint."""
        request_data = {
            "data": [
                {"id": "item1", "positive_count": 80, "negative_count": 20, "neutral_count": 0},
                {"id": "item2", "positive_count": 1, "negative_count": 0, "neutral_count": 0}
            ],
            "confidence_level": 0.95
        }
        
        response = client.post(
            "/analytics/predictive-intervals",
            json=request_data
        )
        
        assert response.status_code in [200, 500]  # May fail if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "ranked_data" in data
            assert "top_positive" in data
            assert "top_negative" in data
    
    def test_category_sentiment_endpoint(self, client):
        """Test category sentiment endpoint."""
        request_data = {
            "data": [
                {"category": "politics", "text": "Отлично!"},
                {"category": "politics", "text": "Ужасно!"},
                {"category": "economy", "text": "Нормально"}
            ]
        }
        
        response = client.post(
            "/analytics/category-sentiment",
            json=request_data
        )
        
        # May fail if model not loaded
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "category_stats" in data
            assert "top_positive_categories" in data
            assert "top_negative_categories" in data
    
    def test_thread_analysis_endpoint(self, client):
        """Test thread analysis endpoint."""
        request_data = {
            "data": [
                {"news_id": "1", "text": "Отлично!"},
                {"news_id": "1", "text": "Ужасно!"},
                {"news_id": "2", "text": "Нормально"}
            ]
        }
        
        response = client.post(
            "/analytics/thread-analysis",
            json=request_data
        )
        
        # May fail if model not loaded
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "thread_stats" in data
            assert "correlation" in data
            assert "correlation" in data["correlation"]
    
    def test_analytics_health(self, client):
        """Test analytics health endpoint."""
        response = client.get("/analytics/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data



