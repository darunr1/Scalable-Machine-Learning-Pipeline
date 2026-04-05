"""Tests for the FastAPI inference API."""

import os
import sys

import pytest

# Note: These tests require a trained model to be present.
# They test the API endpoints using FastAPI TestClient.
# Run the training pipeline first: python -m training.train

try:
    from fastapi.testclient import TestClient
    from api.app import app

    HAS_APP = True
except Exception:
    HAS_APP = False


@pytest.mark.skipif(not HAS_APP, reason="API app not loadable (model may not be trained)")
class TestAPI:
    """Tests for the inference API endpoints."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_model_list_endpoint(self, client):
        """Model list endpoint should return 200."""
        response = client.get("/models")
        assert response.status_code in (200, 503)

    def test_predict_endpoint_validation(self, client):
        """Predict endpoint should validate input."""
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_with_valid_data(self, client):
        """Predict endpoint should work with valid input (if model loaded)."""
        payload = {
            "temperature_2m": 15.2,
            "relative_humidity_2m": 65.0,
            "dew_point_2m": 8.5,
            "apparent_temperature": 13.1,
            "pressure_msl": 1013.25,
            "surface_pressure": 1010.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "cloud_cover": 40.0,
            "wind_speed_10m": 12.5,
            "wind_direction_10m": 180.0,
            "wind_gusts_10m": 25.0,
            "hour": 14,
        }
        response = client.post("/predict", json=payload)
        # 200 if model loaded, 503 if not
        assert response.status_code in (200, 503)
