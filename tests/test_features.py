"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from features.pipeline import FeaturePipeline
from utils.config import load_config


def make_sample_weather_data(n_rows=200):
    """Create synthetic weather data for testing."""
    np.random.seed(42)
    time_index = pd.date_range("2024-01-01", periods=n_rows, freq="h")

    df = pd.DataFrame({
        "temperature_2m": 15 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_rows)) + np.random.randn(n_rows),
        "relative_humidity_2m": 60 + 10 * np.random.randn(n_rows),
        "dew_point_2m": 8 + 3 * np.random.randn(n_rows),
        "apparent_temperature": 13 + 5 * np.random.randn(n_rows),
        "pressure_msl": 1013 + 5 * np.random.randn(n_rows),
        "surface_pressure": 1010 + 5 * np.random.randn(n_rows),
        "precipitation": np.abs(np.random.randn(n_rows) * 0.5),
        "rain": np.abs(np.random.randn(n_rows) * 0.3),
        "snowfall": np.zeros(n_rows),
        "cloud_cover": np.clip(50 + 20 * np.random.randn(n_rows), 0, 100),
        "wind_speed_10m": np.abs(10 + 5 * np.random.randn(n_rows)),
        "wind_direction_10m": np.random.uniform(0, 360, n_rows),
        "wind_gusts_10m": np.abs(15 + 8 * np.random.randn(n_rows)),
    }, index=time_index)

    return df


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_fit_transform(self):
        """Pipeline should fit and transform without errors."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(df)

        assert len(result) > 0
        assert "temperature_2m" in result.columns  # target included

    def test_transform_without_fit_raises(self):
        """Transform without fit should raise RuntimeError."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()

        with pytest.raises(RuntimeError):
            pipeline.transform(df)

    def test_lag_features_created(self):
        """Lag features should be present after transform."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        pipeline.fit(df)

        result = pipeline.transform(df, include_target=False)
        lag_cols = [c for c in result.columns if "lag" in c]
        assert len(lag_cols) > 0

    def test_rolling_features_created(self):
        """Rolling features should be present after transform."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        pipeline.fit(df)

        result = pipeline.transform(df, include_target=False)
        roll_cols = [c for c in result.columns if "roll" in c]
        assert len(roll_cols) > 0

    def test_cyclical_features_created(self):
        """Cyclical hour encoding should produce sin/cos features."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        pipeline.fit(df)

        result = pipeline.transform(df, include_target=False)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_no_nulls_in_output(self):
        """Output should have no null values."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(df)

        assert result.isnull().sum().sum() == 0

    def test_train_inference_parity(self):
        """Fit on train, transform on both → same columns."""
        df = make_sample_weather_data(300)
        train = df.iloc[:200]
        test = df.iloc[200:]

        pipeline = FeaturePipeline()
        train_result = pipeline.fit_transform(train, include_target=False)
        test_result = pipeline.transform(test, include_target=False)

        assert list(train_result.columns) == list(test_result.columns)

    def test_save_load(self, tmp_path):
        """Pipeline should save and load correctly."""
        df = make_sample_weather_data()
        pipeline = FeaturePipeline()
        pipeline.fit(df)

        save_path = str(tmp_path / "test_pipeline.pkl")
        pipeline.save(save_path)

        # Load into new pipeline
        new_pipeline = FeaturePipeline()
        new_pipeline.load(save_path)

        # Transform should produce same result
        result1 = pipeline.transform(df, include_target=False)
        result2 = new_pipeline.transform(df, include_target=False)

        pd.testing.assert_frame_equal(result1, result2)
