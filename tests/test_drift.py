"""Tests for drift detection."""

import numpy as np
import pytest

from drift.detector import DataDriftDetector, ConceptDriftDetector


class TestDataDriftDetector:
    """Tests for DataDriftDetector."""

    def test_no_drift_same_distribution(self, tmp_path):
        """Same distribution should not trigger drift."""
        np.random.seed(42)
        baseline = {"feature_a": np.random.normal(0, 1, 1000)}
        current = {"feature_a": np.random.normal(0, 1, 1000)}

        detector = DataDriftDetector()
        detector.baseline_file = str(tmp_path / "baseline.json")
        detector.reports_dir = str(tmp_path / "reports")
        detector.check_features = ["feature_a"]

        detector.save_baseline(baseline)
        report = detector.detect(current)

        assert report["drift_detected"] is False

    def test_drift_detected_shifted_distribution(self, tmp_path):
        """Shifted distribution should trigger drift."""
        np.random.seed(42)
        baseline = {"feature_a": np.random.normal(0, 1, 1000)}
        current = {"feature_a": np.random.normal(5, 1, 1000)}  # shifted mean

        detector = DataDriftDetector()
        detector.baseline_file = str(tmp_path / "baseline.json")
        detector.reports_dir = str(tmp_path / "reports")
        detector.check_features = ["feature_a"]

        detector.save_baseline(baseline)
        report = detector.detect(current)

        assert report["drift_detected"] is True
        assert len(report["alerts"]) > 0

    def test_drift_report_saved(self, tmp_path):
        """Drift report should be saved to disk."""
        import os

        np.random.seed(42)
        baseline = {"feature_a": np.random.normal(0, 1, 100)}
        current = {"feature_a": np.random.normal(0, 1, 100)}

        detector = DataDriftDetector()
        detector.baseline_file = str(tmp_path / "baseline.json")
        detector.reports_dir = str(tmp_path / "reports")
        detector.check_features = ["feature_a"]

        detector.save_baseline(baseline)
        detector.detect(current)

        reports = os.listdir(str(tmp_path / "reports"))
        assert len(reports) > 0

    def test_multiple_features(self, tmp_path):
        """Should check drift for multiple features independently."""
        np.random.seed(42)
        baseline = {
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.normal(10, 2, 1000),
        }
        current = {
            "feature_a": np.random.normal(0, 1, 1000),  # no drift
            "feature_b": np.random.normal(20, 2, 1000),  # drifted
        }

        detector = DataDriftDetector()
        detector.baseline_file = str(tmp_path / "baseline.json")
        detector.reports_dir = str(tmp_path / "reports")
        detector.check_features = ["feature_a", "feature_b"]

        detector.save_baseline(baseline)
        report = detector.detect(current)

        assert report["drift_detected"] is True
        assert report["features"]["feature_a"]["drift_detected"] is False
        assert report["features"]["feature_b"]["drift_detected"] is True


class TestConceptDriftDetector:
    """Tests for ConceptDriftDetector."""

    def test_no_concept_drift(self):
        """Same prediction distribution should not trigger concept drift."""
        np.random.seed(42)
        baseline = np.random.normal(15, 3, 500)
        current = np.random.normal(15, 3, 500)

        detector = ConceptDriftDetector()
        report = detector.detect(baseline, current)

        assert report["concept_drift_detected"] is False

    def test_concept_drift_detected(self):
        """Shifted prediction distribution should trigger concept drift."""
        np.random.seed(42)
        baseline = np.random.normal(15, 3, 500)
        current = np.random.normal(25, 3, 500)  # shifted predictions

        detector = ConceptDriftDetector()
        report = detector.detect(baseline, current)

        assert report["concept_drift_detected"] is True
