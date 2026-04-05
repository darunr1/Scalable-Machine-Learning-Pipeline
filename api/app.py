"""
FastAPI Inference API.

Endpoints:
    POST /predict       — Get a temperature prediction
    GET  /health        — Health check with model info
    GET  /ready         — Kubernetes-style readiness probe
    GET  /metrics       — API performance metrics
    GET  /model/info    — Current production model metadata
    GET  /models        — List all registered models
    POST /model/reload  — Hot-reload the production model
"""

import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from features.pipeline import FeaturePipeline
from models.registry import ModelRegistry
from monitoring.monitor import PredictionLogger
from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


# ── Global state ────────────────────────────────────────────────────────
class AppState:
    """Encapsulates all mutable application state."""
    model = None
    feature_pipeline = None
    registry = None
    prediction_logger = None
    config = None
    model_loaded_at: Optional[str] = None
    # Metrics tracking
    request_count: int = 0
    error_count: int = 0
    latencies: deque = deque(maxlen=1000)  # Last 1000 request latencies


state = AppState()


def _load_model_and_pipeline():
    """Load model and feature pipeline into app state."""
    state.config = load_config()
    state.registry = ModelRegistry(state.config)
    state.prediction_logger = PredictionLogger(state.config)

    try:
        state.model = state.registry.load_model()  # loads production model
        state.feature_pipeline = FeaturePipeline(state.config)
        state.feature_pipeline.load()
        state.model_loaded_at = datetime.now().isoformat()
        logger.info("Model and feature pipeline loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model/pipeline at startup: {e}")
        logger.warning("API will return errors until a model is trained")


# ── Lifespan (modern replacement for @app.on_event) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    _load_model_and_pipeline()
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="ML Pipeline — Inference API",
    description="Production-style weather prediction API with model versioning and monitoring.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ───────────────────────────────────────────
@app.middleware("http")
async def track_latency(request: Request, call_next):
    """Track request latency for /metrics endpoint."""
    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    state.request_count += 1
    if response.status_code >= 400:
        state.error_count += 1

    # Only track prediction endpoint latency
    if request.url.path == "/predict":
        state.latencies.append(duration_ms)

    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


# ── Prediction Endpoint ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a temperature prediction from weather features."""
    if state.model is None or state.feature_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )

    try:
        # Build input DataFrame matching expected format
        input_data = request.model_dump()
        hour = input_data.pop("hour", 12)

        # Create a DataFrame with enough rows for lag/rolling features
        # In production, we'd use recent history; here we replicate current input
        n_history = max(state.feature_pipeline.lag_steps) + max(state.feature_pipeline.rolling_windows)
        rows = [input_data.copy() for _ in range(n_history + 1)]
        df = pd.DataFrame(rows)

        # Create time index
        now = datetime.now()
        time_index = pd.date_range(end=now, periods=len(df), freq="h")
        df.index = time_index

        # Transform using pipeline
        features_df = state.feature_pipeline.transform(df, include_target=False)

        if len(features_df) == 0:
            raise ValueError("Feature transform produced empty result")

        # Use last row (current) for prediction
        X = features_df.iloc[[-1]]
        prediction = float(state.model.predict(X)[0])

        # Get model confidence (tree variance for RF)
        tree_preds = np.array([tree.predict(X)[0] for tree in state.model.estimators_])
        confidence = {
            "std": round(float(tree_preds.std()), 4),
            "min": round(float(tree_preds.min()), 2),
            "max": round(float(tree_preds.max()), 2),
        }

        version = state.registry.get_production_version()
        timestamp = datetime.now().isoformat()

        # Log prediction
        state.prediction_logger.log(
            input_features=input_data,
            prediction=prediction,
            model_version=version,
            confidence=confidence,
        )

        return PredictionResponse(
            prediction=round(prediction, 2),
            model_version=version,
            timestamp=timestamp,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health Endpoint ─────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    if state.model is None:
        return HealthResponse(status="degraded", model_version=None)

    version = state.registry.get_production_version()
    info = state.registry.get_model_info(version)

    return HealthResponse(
        status="healthy",
        model_version=version,
        model_metrics=info["metrics"],
    )


# ── Readiness Probe ────────────────────────────────────────────────────
@app.get("/ready")
def readiness():
    """
    Kubernetes-style readiness probe.
    Returns 200 only when the model is loaded and ready to serve.
    """
    if state.model is None or state.feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Not ready — model not loaded")
    return {"ready": True, "model_loaded_at": state.model_loaded_at}


# ── API Metrics ─────────────────────────────────────────────────────────
@app.get("/metrics")
def metrics():
    """API performance metrics: request count, error rate, latency stats."""
    latencies = list(state.latencies)
    latency_stats = {}
    if latencies:
        latency_stats = {
            "count": len(latencies),
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
            "max_ms": round(max(latencies), 2),
        }

    return {
        "total_requests": state.request_count,
        "total_errors": state.error_count,
        "error_rate": round(state.error_count / max(state.request_count, 1), 4),
        "prediction_latency": latency_stats,
    }


# ── Model Info ──────────────────────────────────────────────────────────
@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Get current production model metadata."""
    if state.registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    version = state.registry.get_production_version()
    if version is None:
        raise HTTPException(status_code=404, detail="No production model")

    info = state.registry.get_model_info(version)
    return ModelInfoResponse(**info)


# ── Model Hot-Reload ───────────────────────────────────────────────────
@app.post("/model/reload")
def reload_model():
    """
    Hot-reload the production model without restarting the server.
    Useful after retraining produces a new model version.
    """
    try:
        _load_model_and_pipeline()
        version = state.registry.get_production_version()
        return {
            "status": "reloaded",
            "model_version": version,
            "loaded_at": state.model_loaded_at,
        }
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── List Models ─────────────────────────────────────────────────────────
@app.get("/models")
def list_models():
    """List all registered models."""
    if state.registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return state.registry.list_models()
