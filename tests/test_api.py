"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Using a fixture with a context manager ensures the 'lifespan' 
# (which loads the MLflow model) runs properly during our tests.
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_predict_successful_response(client):
    """Test a successful /predict response with the full result schema."""
    response = client.post("/predict", json={"text": "Udacity's stock is up today!"})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "text" in data
    assert "sentiment" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert data["text"] == "Udacity's stock is up today!"

def test_predict_batch_length_matches(client):
    """Test a /predict/batch response whose length matches the input length."""
    input_texts = ["Market is up", "Market is down", "Market is sideways"]
    response = client.post("/predict/batch", json={"texts": input_texts})
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == len(input_texts)

def test_predict_batch_empty_list(client):
    """Test a /predict/batch 422 response for an empty list."""
    response = client.post("/predict/batch", json={"texts": []})
    
    # FastAPI returns 422 Unprocessable Entity when validation fails
    assert response.status_code == 422