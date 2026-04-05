"""
FastAPI Inference API.

Endpoints:
    POST /predict    — Get a temperature prediction
    GET  /health     — Health check with model info
    GET  /model/info — Current production model metadata
"""

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
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

app = FastAPI(
    title="ML Pipeline — Inference API",
    description="Production-style weather prediction API with model versioning and monitoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state loaded at startup ──────────────────────────────────────
model = None
feature_pipeline = None
registry = None
prediction_logger = None
config = None


@app.on_event("startup")
def startup():
    """Load model and feature pipeline on startup."""
    global model, feature_pipeline, registry, prediction_logger, config

    config = load_config()
    registry = ModelRegistry(config)
    prediction_logger = PredictionLogger(config)

    try:
        model = registry.load_model()  # loads production model
        feature_pipeline = FeaturePipeline(config)
        feature_pipeline.load()
        logger.info("Model and feature pipeline loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model/pipeline at startup: {e}")
        logger.warning("API will return errors until a model is trained")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a temperature prediction from weather features."""
    if model is None or feature_pipeline is None:
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
        n_history = max(feature_pipeline.lag_steps) + max(feature_pipeline.rolling_windows)
        rows = [input_data.copy() for _ in range(n_history + 1)]
        df = pd.DataFrame(rows)

        # Create time index
        now = datetime.now()
        time_index = pd.date_range(end=now, periods=len(df), freq="h")
        df.index = time_index

        # Transform using pipeline
        features_df = feature_pipeline.transform(df, include_target=False)

        if len(features_df) == 0:
            raise ValueError("Feature transform produced empty result")

        # Use last row (current) for prediction
        X = features_df.iloc[[-1]]
        prediction = float(model.predict(X)[0])

        # Get model confidence (tree variance for RF)
        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
        confidence = {
            "std": round(float(tree_preds.std()), 4),
            "min": round(float(tree_preds.min()), 2),
            "max": round(float(tree_preds.max()), 2),
        }

        version = registry.get_production_version()
        timestamp = datetime.now().isoformat()

        # Log prediction
        prediction_logger.log(
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


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    if model is None:
        return HealthResponse(status="degraded", model_version=None)

    version = registry.get_production_version()
    info = registry.get_model_info(version)

    return HealthResponse(
        status="healthy",
        model_version=version,
        model_metrics=info["metrics"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Get current production model metadata."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    version = registry.get_production_version()
    if version is None:
        raise HTTPException(status_code=404, detail="No production model")

    info = registry.get_model_info(version)
    return ModelInfoResponse(**info)


@app.get("/models")
def list_models():
    """List all registered models."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return registry.list_models()
