"""
Monitoring system for the ML pipeline.

Tracks predictions, input distributions, model confidence,
and generates periodic monitoring reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


class PredictionLogger:
    """
    Logs every prediction to a JSONL file for monitoring.

    Each log entry contains:
        - input features
        - prediction value
        - model version
        - confidence metrics
        - timestamp
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.log_file = resolve_path(config["monitoring"]["prediction_log"])
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(
        self,
        input_features: Dict,
        prediction: float,
        model_version: int,
        confidence: Optional[Dict] = None,
        actual: Optional[float] = None,
    ):
        """
        Log a single prediction.

        Args:
            input_features: Input feature dict.
            prediction: Model prediction value.
            model_version: Version of the model used.
            confidence: Optional confidence metrics.
            actual: Optional ground truth (for error tracking).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "prediction": round(prediction, 4),
            "input_features": input_features,
            "confidence": confidence,
            "actual": actual,
            "error": round(abs(actual - prediction), 4) if actual is not None else None,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self, limit: Optional[int] = None) -> List[Dict]:
        """Read prediction logs, optionally limiting to last N entries."""
        if not os.path.exists(self.log_file):
            return []

        logs = []
        with open(self.log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        if limit:
            logs = logs[-limit:]

        return logs


class MonitoringReport:
    """
    Generates monitoring reports from prediction logs.

    Reports include:
        - Input feature distribution stats
        - Prediction distribution
        - Confidence stats
        - Error rate (when ground truth available)
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        self.reports_dir = resolve_path(config["monitoring"]["reports_dir"])
        self.prediction_logger = PredictionLogger(config)
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate(self, period_days: int = 1) -> Dict:
        """
        Generate a monitoring report for the specified period.

        Args:
            period_days: Number of days to include in the report.

        Returns:
            Report dict with distribution stats.
        """
        logs = self.prediction_logger.get_logs()

        if not logs:
            logger.warning("No prediction logs found")
            return {"status": "no_data", "timestamp": datetime.now().isoformat()}

        # Filter by period
        cutoff = datetime.now().timestamp() - (period_days * 86400)
        recent_logs = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff
        ]

        if not recent_logs:
            recent_logs = logs  # Use all logs if none in period

        # Compute prediction stats
        predictions = [log["prediction"] for log in recent_logs]
        pred_stats = {
            "count": len(predictions),
            "mean": round(float(np.mean(predictions)), 4),
            "std": round(float(np.std(predictions)), 4),
            "min": round(float(np.min(predictions)), 4),
            "max": round(float(np.max(predictions)), 4),
            "median": round(float(np.median(predictions)), 4),
        }

        # Compute input feature stats
        feature_stats = {}
        if recent_logs[0].get("input_features"):
            feature_keys = recent_logs[0]["input_features"].keys()
            for key in feature_keys:
                values = [
                    log["input_features"][key]
                    for log in recent_logs
                    if key in log.get("input_features", {})
                    and isinstance(log["input_features"][key], (int, float))
                ]
                if values:
                    feature_stats[key] = {
                        "mean": round(float(np.mean(values)), 4),
                        "std": round(float(np.std(values)), 4),
                        "min": round(float(np.min(values)), 2),
                        "max": round(float(np.max(values)), 2),
                    }

        # Compute confidence stats
        confidence_stats = None
        conf_stds = [
            log["confidence"]["std"]
            for log in recent_logs
            if log.get("confidence") and "std" in log["confidence"]
        ]
        if conf_stds:
            confidence_stats = {
                "mean_std": round(float(np.mean(conf_stds)), 4),
                "max_std": round(float(np.max(conf_stds)), 4),
            }

        # Compute error stats
        error_stats = None
        errors = [log["error"] for log in recent_logs if log.get("error") is not None]
        if errors:
            error_stats = {
                "count": len(errors),
                "mean_error": round(float(np.mean(errors)), 4),
                "max_error": round(float(np.max(errors)), 4),
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": period_days,
            "total_predictions": len(recent_logs),
            "prediction_stats": pred_stats,
            "feature_stats": feature_stats,
            "confidence_stats": confidence_stats,
            "error_stats": error_stats,
        }

        # Save report
        report_file = os.path.join(
            self.reports_dir,
            f"report_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {report_file}")
        return report


def run_monitoring_report(config: Optional[dict] = None):
    """CLI entry point for generating monitoring reports."""
    reporter = MonitoringReport(config)
    report = reporter.generate()
    logger.info(f"Report: {json.dumps(report, indent=2)}")
    return report


if __name__ == "__main__":
    run_monitoring_report()
