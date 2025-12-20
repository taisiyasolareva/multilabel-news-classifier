"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_classify_invalid_request():
    """Test classification with invalid request."""
    # Empty title
    response = client.post(
        "/classify",
        json={"title": ""}
    )
    assert response.status_code == 422  # Validation error
    
    # Missing title
    response = client.post(
        "/classify",
        json={}
    )
    assert response.status_code == 422


def test_classify_valid_request():
    """Test classification with valid request."""
    # Note: This will fail if model is not loaded
    response = client.post(
        "/classify",
        json={
            "title": "Тестовая новость",
            "snippet": "Описание новости",
            "threshold": 0.5
        }
    )
    
    # Should either succeed (if model loaded) or return 503 (if not)
    assert response.status_code in [200, 503]


def test_classify_batch():
    """Test batch classification."""
    response = client.post(
        "/classify/batch",
        json={
            "items": [
                {"title": "Новость 1"},
                {"title": "Новость 2"}
            ]
        }
    )
    
    # Should either succeed or return 503
    assert response.status_code in [200, 503]


def test_classify_with_top_k():
    """Test classification with top_k parameter."""
    response = client.post(
        "/classify",
        json={
            "title": "Тестовая новость",
            "top_k": 5
        }
    )
    
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert len(data["predictions"]) <= 5


def test_classify_threshold_validation():
    """Test threshold validation."""
    # Invalid threshold (> 1.0)
    response = client.post(
        "/classify",
        json={
            "title": "Тест",
            "threshold": 1.5
        }
    )
    assert response.status_code == 422
    
    # Invalid threshold (< 0.0)
    response = client.post(
        "/classify",
        json={
            "title": "Тест",
            "threshold": -0.1
        }
    )
    assert response.status_code == 422


def test_classify_top_k_validation():
    """Test top_k validation."""
    # Invalid top_k (> 100)
    response = client.post(
        "/classify",
        json={
            "title": "Тест",
            "top_k": 200
        }
    )
    assert response.status_code == 422
    
    # Invalid top_k (< 1)
    response = client.post(
        "/classify",
        json={
            "title": "Тест",
            "top_k": 0
        }
    )
    assert response.status_code == 422

