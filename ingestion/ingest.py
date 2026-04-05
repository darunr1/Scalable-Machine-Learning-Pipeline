"""
Data Ingestion CLI.

Usage:
    python -m ingestion.ingest --source weather_api --start-date 2024-01-01 --end-date 2024-01-31
    python -m ingestion.ingest --source csv --file-path data/raw/my_data.csv
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ingestion.sources import get_source
from ingestion.validation import get_weather_validator
from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


def save_versioned_data(df: pd.DataFrame, data_dir: str, source: str, date_tag: str) -> str:
    """
    Save DataFrame as a versioned Parquet file.

    Args:
        df: DataFrame to save.
        data_dir: Base directory for raw data.
        source: Source identifier.
        date_tag: Date tag for the filename.

    Returns:
        Path to the saved file.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Find next version number
    existing = [
        f for f in os.listdir(data_dir)
        if f.startswith(f"{source}_{date_tag}_v") and f.endswith(".parquet")
    ]
    version = len(existing) + 1

    filename = f"{source}_{date_tag}_v{version}.parquet"
    filepath = os.path.join(data_dir, filename)

    df.to_parquet(filepath, engine="pyarrow")
    logger.info(f"Saved data to {filepath} ({len(df)} rows)")
    return filepath


def log_ingestion_run(
    log_file: str,
    source: str,
    start_date: str,
    end_date: str,
    filepath: str,
    row_count: int,
    validation_results: dict,
):
    """Append ingestion run metadata to the log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "start_date": start_date,
        "end_date": end_date,
        "output_file": filepath,
        "row_count": row_count,
        "validation": {
            "is_valid": validation_results["is_valid"],
            "error_count": len(validation_results["errors"]),
            "warning_count": len(validation_results["warnings"]),
        },
    }

    # Load existing log or create new
    log = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log = json.load(f)

    log.append(entry)

    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)

    logger.info(f"Logged ingestion run to {log_file}")


def run_ingestion(
    source_type: str,
    start_date: str,
    end_date: str,
    file_path: Optional[str] = None,
    config: Optional[dict] = None,
) -> str:
    """
    Run the full ingestion pipeline.

    Args:
        source_type: Data source type ('weather_api', 'csv', 'json').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        file_path: Path to local file (for csv/json sources).
        config: Optional config dict.

    Returns:
        Path to saved Parquet file.
    """
    if config is None:
        config = load_config()

    logger.info(f"Starting ingestion: source={source_type}, dates={start_date} to {end_date}")

    # 1. Fetch data
    source = get_source(source_type, file_path=file_path)
    df = source.fetch(start_date=start_date, end_date=end_date)

    # 2. Validate
    validator = get_weather_validator()
    validation_results = validator.validate(df, raise_on_error=False)

    if not validation_results["is_valid"]:
        logger.warning("Data validation failed — saving anyway for inspection")

    # 3. Save versioned data
    data_dir = resolve_path(config["ingestion"]["data_dir"])
    date_tag = start_date.replace("-", "")
    filepath = save_versioned_data(df, data_dir, source_type, date_tag)

    # 4. Log run
    log_file = resolve_path(config["ingestion"]["log_file"])
    log_ingestion_run(
        log_file=log_file,
        source=source_type,
        start_date=start_date,
        end_date=end_date,
        filepath=filepath,
        row_count=len(df),
        validation_results=validation_results,
    )

    logger.info("Ingestion complete ✓")
    return filepath


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Pipeline — Data Ingestion")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["weather_api", "csv", "json"],
        help="Data source type",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD). Default: 30 days ago.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to local file (for csv/json sources)",
    )

    args = parser.parse_args()

    run_ingestion(
        source_type=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
        file_path=args.file_path,
    )


if __name__ == "__main__":
    main()
