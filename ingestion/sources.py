"""
Data source classes for the ingestion layer.
Each source implements a common interface for fetching raw data.
"""

import os
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def fetch(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        Fetch raw data for the given date range.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with raw data.
        """
        pass


class WeatherAPISource(DataSource):
    """
    Fetches historical weather data from the Open-Meteo Archive API.
    Free, no API key required.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.api_config = config["ingestion"]["api"]

    def fetch(self, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """Fetch hourly weather data from Open-Meteo."""
        logger.info(f"Fetching weather data: {start_date} to {end_date}")

        params = {
            "latitude": self.api_config["latitude"],
            "longitude": self.api_config["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.api_config["hourly_variables"]),
            "timezone": self.api_config["timezone"],
        }

        try:
            response = requests.get(self.api_config["base_url"], params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

        if "hourly" not in data:
            raise ValueError("No hourly data returned from API")

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        logger.info(f"Fetched {len(df)} rows, {len(df.columns)} columns")
        return df


class CSVSource(DataSource):
    """Reads data from a local CSV file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch(self, start_date: str = None, end_date: str = None, **kwargs) -> pd.DataFrame:
        """Load CSV data, optionally filtering by date range."""
        logger.info(f"Reading CSV: {self.file_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        df = pd.read_csv(self.file_path)

        # Try to parse time column and filter by date range
        time_cols = [c for c in df.columns if c.lower() in ("time", "date", "timestamp", "datetime")]
        if time_cols:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
            df = df.set_index(time_cols[0])

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

        logger.info(f"Loaded {len(df)} rows from CSV")
        return df


class JSONSource(DataSource):
    """Reads data from a local JSON file."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def fetch(self, start_date: str = None, end_date: str = None, **kwargs) -> pd.DataFrame:
        """Load JSON data."""
        logger.info(f"Reading JSON: {self.file_path}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")

        with open(self.file_path, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        time_cols = [c for c in df.columns if c.lower() in ("time", "date", "timestamp", "datetime")]
        if time_cols:
            df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
            df = df.set_index(time_cols[0])

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

        logger.info(f"Loaded {len(df)} rows from JSON")
        return df


def get_source(source_type: str, **kwargs) -> DataSource:
    """
    Factory function to get a data source by type.

    Args:
        source_type: One of 'weather_api', 'csv', 'json'.
        **kwargs: Additional arguments (e.g., file_path for csv/json).

    Returns:
        DataSource instance.
    """
    sources = {
        "weather_api": lambda: WeatherAPISource(),
        "csv": lambda: CSVSource(kwargs.get("file_path", "")),
        "json": lambda: JSONSource(kwargs.get("file_path", "")),
    }

    if source_type not in sources:
        raise ValueError(f"Unknown source type: {source_type}. Choose from: {list(sources.keys())}")

    return sources[source_type]()
