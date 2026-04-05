"""Tests for the data ingestion layer."""

import os
import tempfile
import json

import pandas as pd
import numpy as np
import pytest

from ingestion.validation import SchemaValidator, ValidationError, get_weather_validator
from ingestion.sources import CSVSource, JSONSource


# ── Schema Validation Tests ─────────────────────────────────────────────

class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_valid_data_passes(self):
        """Valid data should pass validation."""
        df = pd.DataFrame({
            "temperature_2m": [15.0, 16.0, 17.0],
            "relative_humidity_2m": [60.0, 65.0, 70.0],
            "pressure_msl": [1013.0, 1012.0, 1014.0],
            "wind_speed_10m": [10.0, 12.0, 8.0],
        })

        validator = SchemaValidator(
            required_columns=["temperature_2m", "relative_humidity_2m"],
        )
        result = validator.validate(df)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_columns_fails(self):
        """Missing required columns should fail validation."""
        df = pd.DataFrame({"temperature_2m": [15.0]})

        validator = SchemaValidator(
            required_columns=["temperature_2m", "pressure_msl"],
        )

        with pytest.raises(ValidationError):
            validator.validate(df, raise_on_error=True)

    def test_empty_dataframe_fails(self):
        """Empty DataFrame should fail validation."""
        df = pd.DataFrame()

        validator = SchemaValidator()
        with pytest.raises(ValidationError):
            validator.validate(df, raise_on_error=True)

    def test_null_fraction_warning(self):
        """High null fraction should generate warnings."""
        df = pd.DataFrame({
            "col_a": [1.0, None, None, None, 5.0],
            "col_b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        validator = SchemaValidator(max_null_fraction=0.1)
        result = validator.validate(df, raise_on_error=False)
        assert len(result["warnings"]) > 0

    def test_value_range_warning(self):
        """Out-of-range values should generate warnings."""
        df = pd.DataFrame({
            "temperature_2m": [15.0, 100.0],  # 100 is out of range
        })

        validator = SchemaValidator(
            value_ranges={"temperature_2m": (-60, 60)},
        )
        result = validator.validate(df, raise_on_error=False)
        assert len(result["warnings"]) > 0


class TestWeatherValidator:
    """Tests for the pre-configured weather validator."""

    def test_weather_validator_creation(self):
        """Weather validator should be created successfully."""
        validator = get_weather_validator()
        assert validator is not None
        assert len(validator.required_columns) > 0

    def test_weather_validator_valid_data(self):
        """Valid weather data should pass."""
        df = pd.DataFrame({
            "temperature_2m": [15.0, 16.0],
            "relative_humidity_2m": [60.0, 65.0],
            "pressure_msl": [1013.0, 1012.0],
            "wind_speed_10m": [10.0, 12.0],
            "precipitation": [0.0, 1.0],
        })

        validator = get_weather_validator()
        result = validator.validate(df)
        assert result["is_valid"] is True


# ── CSV Source Tests ────────────────────────────────────────────────────

class TestCSVSource:
    """Tests for CSVSource."""

    def test_csv_load(self):
        """CSV file should load successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("time,temperature_2m,humidity\n")
            f.write("2024-01-01 00:00,15.0,60.0\n")
            f.write("2024-01-01 01:00,14.5,62.0\n")
            temp_path = f.name

        try:
            source = CSVSource(temp_path)
            df = source.fetch()
            assert len(df) == 2
            assert "temperature_2m" in df.columns
        finally:
            os.unlink(temp_path)

    def test_csv_file_not_found(self):
        """Missing CSV file should raise error."""
        source = CSVSource("/nonexistent/file.csv")
        with pytest.raises(FileNotFoundError):
            source.fetch()


# ── JSON Source Tests ───────────────────────────────────────────────────

class TestJSONSource:
    """Tests for JSONSource."""

    def test_json_load(self):
        """JSON file should load successfully."""
        data = [
            {"time": "2024-01-01 00:00", "temperature_2m": 15.0},
            {"time": "2024-01-01 01:00", "temperature_2m": 14.5},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            source = JSONSource(temp_path)
            df = source.fetch()
            assert len(df) == 2
        finally:
            os.unlink(temp_path)
