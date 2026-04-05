"""
End-to-end integration test.

Tests the full pipeline: ingest → features → train → registry → predict → monitor → drift.
Uses synthetic data to avoid external API dependencies.
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from tests.test_features import make_sample_weather_data
from features.pipeline import FeaturePipeline
from models.registry import ModelRegistry
from monitoring.monitor import PredictionLogger
from drift.detector import DataDriftDetector
from training.train import compute_metrics, train_test_split_temporal
from utils.config import load_config


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_full_pipeline(self, tmp_path):
        """
        End-to-end: generate data → engineer features → train model
        → register → predict → log → check drift.
        """
        config = load_config()

        # Override paths to use temp directory
        config["registry"]["model_dir"] = str(tmp_path / "models")
        config["registry"]["metadata_file"] = str(tmp_path / "models" / "metadata.json")
        config["features"]["pipeline_artifact"] = str(tmp_path / "pipeline.pkl")
        config["monitoring"]["prediction_log"] = str(tmp_path / "predictions.jsonl")
        config["drift"]["baseline_file"] = str(tmp_path / "baseline.json")
        config["drift"]["reports_dir"] = str(tmp_path / "drift_reports")

        # 1. Generate synthetic data (simulates ingestion)
        df = make_sample_weather_data(500)
        assert len(df) == 500

        # 2. Feature engineering
        pipeline = FeaturePipeline(config)
        features_df = pipeline.fit_transform(df)
        assert len(features_df) > 0
        assert features_df.isnull().sum().sum() == 0

        # Save pipeline
        pipeline.save(config["features"]["pipeline_artifact"])

        # 3. Train/test split
        target = config["features"]["target_column"]
        train_df, test_df = train_test_split_temporal(features_df, test_size=0.2)

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        # 4. Train model
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1.0

        # 6. Register model
        registry = ModelRegistry(config)
        version = registry.register_model(model, metrics,
                                          {"training_date": "2024-01-01"})
        assert version == 1
        assert registry.get_production_version() == 1

        # 7. Load and predict
        loaded_model = registry.load_model()
        pred = loaded_model.predict(X_test.iloc[:1])
        assert len(pred) == 1

        # 8. Log prediction
        pred_logger = PredictionLogger(config)
        pred_logger.log(
            input_features={"temperature_2m": 15.0, "humidity": 60.0},
            prediction=float(pred[0]),
            model_version=1,
            confidence={"std": 0.5},
        )
        logs = pred_logger.get_logs()
        assert len(logs) == 1

        # 9. Drift detection
        drift_detector = DataDriftDetector(config)
        baseline_data = {
            "temperature_2m": df["temperature_2m"].values[:200],
        }
        drift_detector.save_baseline(baseline_data)

        current_data = {
            "temperature_2m": df["temperature_2m"].values[200:400],
        }
        report = drift_detector.detect(current_data)
        assert "drift_detected" in report

        # Pipeline complete!
        assert True
