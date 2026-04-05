"""Tests for the training pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tests.test_features import make_sample_weather_data
from features.pipeline import FeaturePipeline
from training.train import train_test_split_temporal, compute_metrics


class TestTrainTestSplit:
    """Tests for temporal train/test split."""

    def test_split_sizes(self):
        """Split should respect the test_size ratio."""
        df = pd.DataFrame({"a": range(100)})
        train, test = train_test_split_temporal(df, test_size=0.2)

        assert len(train) == 80
        assert len(test) == 20

    def test_split_no_overlap(self):
        """Train and test should not overlap."""
        df = pd.DataFrame({"a": range(100)})
        train, test = train_test_split_temporal(df)

        assert train.index[-1] < test.index[0]


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should give MAE=0, R²=1."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_metrics_keys(self):
        """Metrics dict should contain mae, rmse, r2."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])

        metrics = compute_metrics(y_true, y_pred)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics


class TestTrainingIntegration:
    """Integration test for feature pipeline + training."""

    def test_feature_pipeline_produces_trainable_data(self):
        """Feature pipeline output should be suitable for training."""
        df = make_sample_weather_data(300)
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(df)

        # Should have target column
        assert "temperature_2m" in result.columns

        # Should have features
        feature_cols = [c for c in result.columns if c != "temperature_2m"]
        assert len(feature_cols) > 5

        # No nulls
        assert result.isnull().sum().sum() == 0
