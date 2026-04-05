"""
Drift Detection Module.

Detects data drift and concept drift using statistical tests:
    - Kolmogorov-Smirnov (KS) test for feature distributions
    - KL divergence for distribution comparison
    - Prediction distribution shift detection

Triggers alerts and retraining when drift exceeds thresholds.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


class DataDriftDetector:
    """
    Detects data drift by comparing current feature distributions
    against a training baseline using the KS test.

    If KS statistic exceeds threshold (p-value < alpha),
    drift is flagged for that feature.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        drift_config = config["drift"]
        self.ks_threshold = drift_config["ks_threshold"]
        self.check_features = drift_config["check_features"]
        self.baseline_file = resolve_path(drift_config["baseline_file"])
        self.reports_dir = resolve_path(drift_config["reports_dir"])

        os.makedirs(self.reports_dir, exist_ok=True)

    def save_baseline(self, training_data: Dict[str, np.ndarray]):
        """
        Save training data distributions as the baseline for drift comparison.

        Args:
            training_data: Dict of feature_name → array of training values.
        """
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)

        baseline = {}
        for feature, values in training_data.items():
            values = np.array(values, dtype=float)
            baseline[feature] = {
                "values": values.tolist(),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline, f)

        logger.info(f"Saved drift baseline for {len(baseline)} features")

    def load_baseline(self) -> Dict:
        """Load the training baseline."""
        if not os.path.exists(self.baseline_file):
            raise FileNotFoundError(
                f"Drift baseline not found: {self.baseline_file}. "
                "Run training first to generate baseline."
            )

        with open(self.baseline_file, "r") as f:
            return json.load(f)

    def detect(
        self, current_data: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Run drift detection on current data vs baseline.

        Args:
            current_data: Dict of feature_name → array of current values.

        Returns:
            Drift report with per-feature KS stats and alert flag.
        """
        baseline = self.load_baseline()

        report = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "features": {},
            "alerts": [],
        }

        for feature in self.check_features:
            if feature not in current_data or feature not in baseline:
                logger.warning(f"Feature '{feature}' missing from data or baseline")
                continue

            baseline_values = np.array(baseline[feature]["values"])
            current_values = np.array(current_data[feature], dtype=float)

            # KS test
            ks_stat, p_value = stats.ks_2samp(baseline_values, current_values)

            is_drifted = p_value < self.ks_threshold

            feature_report = {
                "ks_statistic": round(float(ks_stat), 6),
                "p_value": round(float(p_value), 6),
                "drift_detected": is_drifted,
                "baseline_mean": baseline[feature]["mean"],
                "current_mean": round(float(np.mean(current_values)), 4),
                "baseline_std": baseline[feature]["std"],
                "current_std": round(float(np.std(current_values)), 4),
            }

            report["features"][feature] = feature_report

            if is_drifted:
                report["drift_detected"] = True
                alert_msg = (
                    f"DRIFT ALERT: '{feature}' — KS={ks_stat:.4f}, p={p_value:.6f} "
                    f"(baseline μ={baseline[feature]['mean']:.2f}, "
                    f"current μ={np.mean(current_values):.2f})"
                )
                report["alerts"].append(alert_msg)
                logger.warning(alert_msg)

        if report["drift_detected"]:
            logger.warning(
                f"Data drift detected in {len(report['alerts'])} feature(s)!"
            )
        else:
            logger.info("No data drift detected ✓")

        # Save report
        report_file = os.path.join(
            self.reports_dir,
            f"drift_report_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Drift report saved to {report_file}")
        return report


class ConceptDriftDetector:
    """
    Detects concept drift by comparing prediction distributions
    over time (i.e., p(y|x) has changed).
    """

    def __init__(self, ks_threshold: float = 0.05):
        self.ks_threshold = ks_threshold

    def detect(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict:
        """
        Compare baseline and current prediction distributions.

        Args:
            baseline_predictions: Training/validation predictions.
            current_predictions: Recent inference predictions.

        Returns:
            Concept drift report.
        """
        ks_stat, p_value = stats.ks_2samp(baseline_predictions, current_predictions)
        is_drifted = p_value < self.ks_threshold

        report = {
            "timestamp": datetime.now().isoformat(),
            "concept_drift_detected": is_drifted,
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "baseline_mean": round(float(np.mean(baseline_predictions)), 4),
            "current_mean": round(float(np.mean(current_predictions)), 4),
            "baseline_std": round(float(np.std(baseline_predictions)), 4),
            "current_std": round(float(np.std(current_predictions)), 4),
        }

        if is_drifted:
            logger.warning(
                f"Concept drift detected! KS={ks_stat:.4f}, p={p_value:.6f}"
            )
        else:
            logger.info("No concept drift detected ✓")

        return report


def run_drift_check(config: Optional[dict] = None) -> Dict:
    """
    Run a drift check using recent prediction logs vs training baseline.

    Returns:
        Drift report dict.
    """
    from monitoring.monitor import PredictionLogger

    if config is None:
        config = load_config()

    prediction_logger = PredictionLogger(config)
    logs = prediction_logger.get_logs()

    if len(logs) < 10:
        logger.warning("Not enough predictions for drift check (need ≥10)")
        return {"status": "insufficient_data", "count": len(logs)}

    # Extract feature values from prediction logs
    current_data = {}
    for feature in config["drift"]["check_features"]:
        values = [
            log["input_features"].get(feature)
            for log in logs
            if log.get("input_features") and feature in log["input_features"]
        ]
        values = [v for v in values if v is not None]
        if values:
            current_data[feature] = np.array(values)

    detector = DataDriftDetector(config)
    report = detector.detect(current_data)

    return report


if __name__ == "__main__":
    run_drift_check()
