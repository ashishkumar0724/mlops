import os
import sys
import pytest
from pathlib import Path

# 🔑 Ensure src is importable regardless of cwd
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path.parent))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    """Verify the API is running and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_spam():
    """Verify spam detection works with high confidence."""
    response = client.post(
        "/predict",
        json={"message": "congratulations you won a free iphone call now"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["confidence"] > 0.8
    assert isinstance(data["latency_s"], (int, float))

def test_predict_ham():
    """Verify ham (non-spam) detection works."""
    response = client.post(
        "/predict",
        json={"message": "hey can we meet tomorrow for lunch"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "ham"
    assert data["confidence"] > 0.8

def test_predict_empty_message():
    """Verify API rejects empty input with 400 error."""
    response = client.post("/predict", json={"message": ""})
    assert response.status_code == 400

def test_predict_missing_field():
    """Verify API rejects malformed JSON."""
    response = client.post("/predict", json={"wrong_field": "test"})
    assert response.status_code == 422  # Pydantic validation error