"""
Data validation module.
Validates schema, data types, null values, and value ranges.
"""

from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class SchemaValidator:
    """
    Validates a DataFrame against an expected schema.

    Checks:
        - Required columns exist
        - Data types are correct
        - Null values within acceptable thresholds
        - Value ranges are valid
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        max_null_fraction: float = 0.1,
        value_ranges: Optional[Dict[str, tuple]] = None,
    ):
        """
        Args:
            required_columns: List of columns that must be present.
            max_null_fraction: Maximum fraction of nulls allowed per column (0-1).
            value_ranges: Dict of column_name -> (min, max) acceptable ranges.
        """
        self.required_columns = required_columns or []
        self.max_null_fraction = max_null_fraction
        self.value_ranges = value_ranges or {}

    def validate(self, df: pd.DataFrame, raise_on_error: bool = True) -> Dict:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate.
            raise_on_error: If True, raise ValidationError on first failure.

        Returns:
            Dict with validation results.
        """
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "null_counts": {},
            },
        }

        # Check empty DataFrame
        if len(df) == 0:
            results["is_valid"] = False
            results["errors"].append("DataFrame is empty")
            if raise_on_error:
                raise ValidationError("DataFrame is empty")
            return results

        # Check required columns
        self._check_columns(df, results)

        # Check null values
        self._check_nulls(df, results)

        # Check value ranges
        self._check_ranges(df, results)

        if results["errors"]:
            results["is_valid"] = False
            logger.error(f"Validation failed with {len(results['errors'])} errors")
            if raise_on_error:
                raise ValidationError(
                    f"Validation failed: {'; '.join(results['errors'])}"
                )
        else:
            logger.info("Validation passed ✓")

        if results["warnings"]:
            for warning in results["warnings"]:
                logger.warning(warning)

        return results

    def _check_columns(self, df: pd.DataFrame, results: Dict):
        """Check that all required columns exist."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            results["errors"].append(f"Missing required columns: {missing}")
            logger.error(f"Missing columns: {missing}")

    def _check_nulls(self, df: pd.DataFrame, results: Dict):
        """Check null value fractions."""
        for col in df.columns:
            null_frac = df[col].isnull().mean()
            results["stats"]["null_counts"][col] = {
                "null_count": int(df[col].isnull().sum()),
                "null_fraction": round(float(null_frac), 4),
            }

            if null_frac > self.max_null_fraction:
                msg = (
                    f"Column '{col}' has {null_frac:.1%} nulls "
                    f"(threshold: {self.max_null_fraction:.1%})"
                )
                results["warnings"].append(msg)

            if null_frac == 1.0:
                results["errors"].append(f"Column '{col}' is entirely null")

    def _check_ranges(self, df: pd.DataFrame, results: Dict):
        """Check that numeric columns are within expected ranges."""
        for col, (min_val, max_val) in self.value_ranges.items():
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_min = df[col].min()
            col_max = df[col].max()

            if col_min < min_val or col_max > max_val:
                msg = (
                    f"Column '{col}' out of range: "
                    f"[{col_min}, {col_max}] vs expected [{min_val}, {max_val}]"
                )
                results["warnings"].append(msg)


def get_weather_validator() -> SchemaValidator:
    """Get a pre-configured validator for weather data."""
    return SchemaValidator(
        required_columns=[
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
        ],
        max_null_fraction=0.1,
        value_ranges={
            "temperature_2m": (-60, 60),
            "relative_humidity_2m": (0, 100),
            "pressure_msl": (870, 1084),
            "wind_speed_10m": (0, 200),
            "precipitation": (0, 500),
        },
    )
