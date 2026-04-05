"""
Feature Engineering Pipeline.

Provides a FeaturePipeline class with fit/transform pattern to ensure
train-time and inference-time feature engineering is identical.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Production feature engineering pipeline.

    Transforms raw weather data into ML-ready features:
        - Null imputation (median fill)
        - Lag features (previous N timesteps)
        - Rolling averages (windowed means)
        - Cyclical encoding (hour of day → sin/cos)
        - Standard scaling (zero mean, unit variance)

    Uses fit/transform pattern for train-inference parity.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        feat_config = config["features"]
        self.target_column = feat_config["target_column"]
        self.lag_steps = feat_config["lag_features"]
        self.rolling_windows = feat_config["rolling_windows"]
        self.cyclical_features = feat_config["cyclical_features"]
        self.numerical_features = feat_config["numerical_features"]
        self.pipeline_artifact_path = resolve_path(feat_config["pipeline_artifact"])

        # Fitted state
        self.scaler = StandardScaler()
        self.median_values: dict = {}
        self.feature_columns: List[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit the pipeline on training data. Learns medians and scaler params.

        Args:
            df: Raw DataFrame with time index and weather columns.

        Returns:
            self (for chaining).
        """
        logger.info(f"Fitting feature pipeline on {len(df)} rows")

        # Compute medians for imputation
        all_numeric = self.numerical_features + [self.target_column]
        for col in all_numeric:
            if col in df.columns:
                self.median_values[col] = float(df[col].median())

        # Build features to learn scaler
        features_df = self._build_features(df)
        self.feature_columns = [c for c in features_df.columns if c != self.target_column]

        # Fit scaler on feature columns
        self.scaler.fit(features_df[self.feature_columns])

        self._is_fitted = True
        logger.info(f"Pipeline fitted. {len(self.feature_columns)} feature columns.")
        return self

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Transform raw data using the fitted pipeline.

        Args:
            df: Raw DataFrame.
            include_target: Whether to include target column in output.

        Returns:
            Transformed DataFrame ready for ML.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform. Call fit() first.")

        logger.info(f"Transforming {len(df)} rows")

        # Build features
        features_df = self._build_features(df)

        # Scale features
        scaled = self.scaler.transform(features_df[self.feature_columns])
        result = pd.DataFrame(scaled, columns=self.feature_columns, index=features_df.index)

        if include_target and self.target_column in features_df.columns:
            result[self.target_column] = features_df[self.target_column].values

        logger.info(f"Transform complete: {result.shape}")
        return result

    def fit_transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, include_target=include_target)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: build all feature columns from raw data."""
        result = df.copy()

        # 1. Impute nulls with medians
        for col, median_val in self.median_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(median_val)

        # 2. Lag features (for target column)
        if self.target_column in result.columns:
            for lag in self.lag_steps:
                result[f"{self.target_column}_lag_{lag}"] = result[self.target_column].shift(lag)

        # 3. Rolling averages (for target column)
        if self.target_column in result.columns:
            for window in self.rolling_windows:
                result[f"{self.target_column}_roll_mean_{window}"] = (
                    result[self.target_column].rolling(window=window, min_periods=1).mean()
                )
                result[f"{self.target_column}_roll_std_{window}"] = (
                    result[self.target_column].rolling(window=window, min_periods=1).std().fillna(0)
                )

        # 4. Cyclical encoding for hour of day
        if "hour" in self.cyclical_features and hasattr(result.index, "hour"):
            hour = result.index.hour
            result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # 5. Drop rows with NaN from lags
        result = result.dropna()

        return result

    def save(self, path: Optional[str] = None):
        """Save fitted pipeline to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        path = path or self.pipeline_artifact_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "scaler": self.scaler,
            "median_values": self.median_values,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "lag_steps": self.lag_steps,
            "rolling_windows": self.rolling_windows,
            "cyclical_features": self.cyclical_features,
            "numerical_features": self.numerical_features,
        }
        joblib.dump(state, path)
        logger.info(f"Saved fitted pipeline to {path}")

    def load(self, path: Optional[str] = None) -> "FeaturePipeline":
        """Load a fitted pipeline from disk."""
        path = path or self.pipeline_artifact_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Pipeline artifact not found: {path}")

        state = joblib.load(path)
        self.scaler = state["scaler"]
        self.median_values = state["median_values"]
        self.feature_columns = state["feature_columns"]
        self.target_column = state["target_column"]
        self.lag_steps = state["lag_steps"]
        self.rolling_windows = state["rolling_windows"]
        self.cyclical_features = state["cyclical_features"]
        self.numerical_features = state["numerical_features"]
        self._is_fitted = True

        logger.info(f"Loaded fitted pipeline from {path}")
        return self
