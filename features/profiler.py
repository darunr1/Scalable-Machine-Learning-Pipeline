"""
Data Profiler.

Generates comprehensive data quality reports including:
    - Missing value analysis
    - Outlier detection (IQR method)
    - Distribution statistics (skewness, kurtosis)
    - Correlation matrix
    - Temporal coverage analysis

Reports are saved to data/profiles/ as JSON.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config import resolve_path

logger = get_logger(__name__)


class DataProfiler:
    """
    Generates a comprehensive profile report for a DataFrame.

    Useful for understanding data quality before feature engineering
    and for comparing data across ingestion batches.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or resolve_path("data/profiles")
        os.makedirs(self.output_dir, exist_ok=True)

    def profile(self, df: pd.DataFrame, name: str = "dataset") -> Dict:
        """
        Generate a full data profile report.

        Args:
            df: DataFrame to profile.
            name: Name/label for this dataset.

        Returns:
            Profile report dict.
        """
        logger.info(f"Profiling dataset '{name}': {df.shape}")

        report = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": self._analyze_missing(df),
            "numeric_stats": self._compute_numeric_stats(df),
            "outliers": self._detect_outliers(df),
            "correlations": self._compute_correlations(df),
            "temporal": self._analyze_temporal(df),
        }

        # Save report
        report_file = os.path.join(
            self.output_dir,
            f"profile_{name}_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Data profile saved to {report_file}")
        return report

    def _analyze_missing(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values per column."""
        missing = {}
        total = len(df)

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            missing[col] = {
                "count": null_count,
                "percentage": round(null_count / total * 100, 2) if total > 0 else 0,
            }

        total_missing = sum(m["count"] for m in missing.values())
        return {
            "per_column": missing,
            "total_missing_cells": total_missing,
            "total_cells": total * len(df.columns),
            "overall_percentage": round(
                total_missing / (total * len(df.columns)) * 100, 2
            ) if total > 0 else 0,
        }

    def _compute_numeric_stats(self, df: pd.DataFrame) -> Dict:
        """Compute distribution statistics for numeric columns."""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            stats[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "25%": round(float(series.quantile(0.25)), 4),
                "50%": round(float(series.median()), 4),
                "75%": round(float(series.quantile(0.75)), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "zeros": int((series == 0).sum()),
                "negatives": int((series < 0).sum()),
            }

        return stats

    def _detect_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict:
        """Detect outliers using the IQR method."""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())

            outliers[col] = {
                "count": outlier_count,
                "percentage": round(outlier_count / len(series) * 100, 2),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
            }

        return outliers

    def _compute_correlations(self, df: pd.DataFrame, top_n: int = 10) -> Dict:
        """Compute correlation matrix and return top correlated pairs."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {"top_pairs": [], "matrix": {}}

        corr_matrix = numeric_df.corr()

        # Find top correlated pairs (excluding self-correlation)
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "correlation": round(float(corr_matrix.iloc[i, j]), 4),
                })

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "top_pairs": pairs[:top_n],
            "matrix": {
                col: {c: round(float(v), 4) for c, v in corr_matrix[col].items()}
                for col in corr_matrix.columns
            },
        }

    def _analyze_temporal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Analyze temporal coverage if index is a DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return None

        return {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "duration_days": (df.index.max() - df.index.min()).days,
            "frequency": str(pd.infer_freq(df.index) or "irregular"),
            "gaps": int(df.index.to_series().diff().gt(pd.Timedelta(hours=2)).sum()),
        }
