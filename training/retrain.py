"""
Retraining Pipeline.

Triggered when drift is detected. Steps:
    1. Pull latest data (re-run ingestion)
    2. Rebuild features
    3. Train new model
    4. Compare new vs current production model
    5. Promote new model ONLY if metrics are better
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional

from utils.logger import get_logger
from utils.config import load_config

logger = get_logger(__name__)


def run_retraining(config: Optional[dict] = None, force: bool = False) -> dict:
    """
    Run the full retraining pipeline.

    Args:
        config: Optional config dict.
        force: If True, retrain even if no drift detected.

    Returns:
        Dict with retraining results.
    """
    from drift.detector import run_drift_check
    from ingestion.ingest import run_ingestion
    from training.train import run_training
    from models.registry import ModelRegistry

    if config is None:
        config = load_config()

    logger.info("=" * 60)
    logger.info("RETRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    # 1. Check if retraining is needed (drift check)
    if not force:
        try:
            drift_report = run_drift_check(config)
            if not drift_report.get("drift_detected", False):
                logger.info("No drift detected — skipping retraining")
                return {
                    "status": "skipped",
                    "reason": "no_drift",
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.warning(f"Drift check failed: {e}. Proceeding with retraining anyway.")

    # 2. Ingest latest data
    logger.info("Step 1/4: Ingesting latest data...")
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        run_ingestion(
            source_type="weather_api",
            start_date=start_date,
            end_date=end_date,
            config=config,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {"status": "failed", "stage": "ingestion", "error": str(e)}

    # 3. Get current production model metrics
    registry = ModelRegistry(config)
    old_version = registry.get_production_version()
    old_metrics = None
    if old_version:
        old_info = registry.get_model_info(old_version)
        old_metrics = old_info["metrics"]
        logger.info(f"Current production model: v{old_version}, MAE={old_metrics.get('mae')}")

    # 4. Train new model
    logger.info("Step 2/4: Training new model...")
    try:
        training_result = run_training(config)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"status": "failed", "stage": "training", "error": str(e)}

    new_version = training_result["version"]
    new_metrics = training_result["metrics"]

    # 5. Compare and decide
    logger.info("Step 3/4: Comparing models...")
    promoted = False

    if old_metrics is None:
        # First model — auto-promote
        registry.promote_model(new_version)
        promoted = True
        logger.info(f"Promoted v{new_version} (first model)")
    elif new_metrics["mae"] < old_metrics["mae"]:
        # New model is better
        improvement = old_metrics["mae"] - new_metrics["mae"]
        registry.promote_model(new_version)
        promoted = True
        logger.info(
            f"Promoted v{new_version} — MAE improved by {improvement:.4f} "
            f"({old_metrics['mae']:.4f} → {new_metrics['mae']:.4f})"
        )
    else:
        logger.info(
            f"Keeping v{old_version} — new model v{new_version} is not better "
            f"(old MAE={old_metrics['mae']:.4f}, new MAE={new_metrics['mae']:.4f})"
        )

    result = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "new_version": new_version,
        "new_metrics": new_metrics,
        "old_version": old_version,
        "old_metrics": old_metrics,
        "promoted": promoted,
    }

    logger.info("=" * 60)
    logger.info(f"RETRAINING COMPLETE — {'PROMOTED' if promoted else 'NOT PROMOTED'}")
    logger.info("=" * 60)

    return result


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ML Pipeline — Retraining")
    parser.add_argument("--force", action="store_true", help="Force retraining even without drift")
    args = parser.parse_args()

    run_retraining(force=args.force)


if __name__ == "__main__":
    main()
