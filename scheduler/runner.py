"""
Automation Scheduler.

Runs pipeline tasks on a configurable schedule:
    - Daily: data ingestion, monitoring report, drift check
    - Weekly: retraining pipeline

Uses the `schedule` library for simple cron-like scheduling.
"""

import time
from datetime import datetime, timedelta

import schedule

from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


def daily_ingestion():
    """Run daily data ingestion."""
    from ingestion.ingest import run_ingestion

    logger.info("[SCHEDULER] Running daily ingestion...")
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        run_ingestion(source_type="weather_api", start_date=start_date, end_date=end_date)
        logger.info("[SCHEDULER] Daily ingestion complete ✓")
    except Exception as e:
        logger.error(f"[SCHEDULER] Daily ingestion failed: {e}")


def daily_monitoring():
    """Run daily monitoring report."""
    from monitoring.monitor import run_monitoring_report

    logger.info("[SCHEDULER] Running daily monitoring report...")
    try:
        run_monitoring_report()
        logger.info("[SCHEDULER] Daily monitoring complete ✓")
    except Exception as e:
        logger.error(f"[SCHEDULER] Daily monitoring failed: {e}")


def daily_drift_check():
    """Run daily drift check."""
    from drift.detector import run_drift_check

    logger.info("[SCHEDULER] Running daily drift check...")
    try:
        report = run_drift_check()
        if report.get("drift_detected"):
            logger.warning("[SCHEDULER] Drift detected! Triggering retraining...")
            weekly_retraining()
        else:
            logger.info("[SCHEDULER] No drift detected ✓")
    except Exception as e:
        logger.error(f"[SCHEDULER] Daily drift check failed: {e}")


def weekly_retraining():
    """Run weekly retraining pipeline."""
    from training.retrain import run_retraining

    logger.info("[SCHEDULER] Running weekly retraining...")
    try:
        result = run_retraining(force=True)
        logger.info(f"[SCHEDULER] Retraining result: {result['status']}")
    except Exception as e:
        logger.error(f"[SCHEDULER] Retraining failed: {e}")


def start_scheduler():
    """Start the automation scheduler."""
    config = load_config()

    logger.info("=" * 60)
    logger.info("STARTING ML PIPELINE SCHEDULER")
    logger.info("=" * 60)
    logger.info("Scheduled tasks:")
    logger.info("  - Daily 02:00: Data ingestion")
    logger.info("  - Daily 03:00: Monitoring report")
    logger.info("  - Daily 04:00: Drift check")
    logger.info("  - Sunday 05:00: Retraining pipeline")
    logger.info("=" * 60)

    # Configure schedule
    schedule.every().day.at("02:00").do(daily_ingestion)
    schedule.every().day.at("03:00").do(daily_monitoring)
    schedule.every().day.at("04:00").do(daily_drift_check)
    schedule.every().sunday.at("05:00").do(weekly_retraining)

    # Run loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    start_scheduler()
