"""
Model Training Pipeline.

Handles the full training workflow:
    - Load data
    - Feature engineering
    - Train/test split (time-based)
    - Cross-validation with TimeSeriesSplit
    - Hyperparameter tuning with GridSearchCV
    - Save model, metrics, and config
    - Register in model registry
"""

import json
import os
import glob
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.pipeline import FeaturePipeline
from models.registry import ModelRegistry
from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


def load_latest_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load the most recent Parquet file from the data directory."""
    if config is None:
        config = load_config()

    data_dir = resolve_path(config["ingestion"]["data_dir"])

    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    latest = parquet_files[-1]
    logger.info(f"Loading data from: {latest}")
    df = pd.read_parquet(latest)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")

    return df


def train_test_split_temporal(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split (no shuffling for time series)."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    logger.info(f"Split: {len(train)} train, {len(test)} test")
    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run_training(config: Optional[dict] = None) -> dict:
    """
    Run the full training pipeline.

    Returns:
        Dict with model version, metrics, and file paths.
    """
    if config is None:
        config = load_config()

    train_config = config["training"]
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load data
    df = load_latest_data(config)
    logger.info(f"Loaded {len(df)} rows of data")

    # 2. Feature engineering
    pipeline = FeaturePipeline(config)
    features_df = pipeline.fit_transform(df, include_target=True)
    logger.info(f"Features shape: {features_df.shape}")

    # 3. Train/test split
    train_df, test_df = train_test_split_temporal(features_df, train_config["test_size"])

    target = config["features"]["target_column"]
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # 4. Hyperparameter tuning with cross-validation
    logger.info("Starting GridSearchCV...")

    # Convert None strings to actual None in param_grid
    param_grid = {}
    for key, values in train_config["param_grid"].items():
        param_grid[key] = [None if v is None else v for v in values]

    model = RandomForestRegressor(random_state=train_config["random_state"])
    tscv = TimeSeriesSplit(n_splits=train_config["cv_folds"])

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring=train_config["scoring"],
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # 5. Evaluate on test set
    y_pred = best_model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)
    logger.info(f"Test metrics: {metrics}")

    # 6. Save feature pipeline
    pipeline.save()

    # 7. Register model
    registry = ModelRegistry(config)
    training_info = {
        "hyperparameters": {k: str(v) for k, v in best_params.items()},
        "training_date": datetime.now().isoformat(),
        "dataset_rows": len(df),
        "feature_count": len(X_train.columns),
        "cv_score": round(float(grid_search.best_score_), 4),
        "feature_columns": list(X_train.columns),
    }

    version = registry.register_model(
        model=best_model,
        metrics=metrics,
        training_info=training_info,
    )

    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE — Model v{version}")
    logger.info(f"MAE: {metrics['mae']} | RMSE: {metrics['rmse']} | R²: {metrics['r2']}")
    logger.info("=" * 60)

    return {
        "version": version,
        "metrics": metrics,
        "best_params": best_params,
    }


def main():
    """CLI entry point for training."""
    run_training()


if __name__ == "__main__":
    main()
