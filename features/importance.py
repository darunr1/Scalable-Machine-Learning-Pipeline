"""
Feature Importance Analysis.

Extracts and compares feature importances using multiple methods:
    - Built-in tree-based importances (Gini / impurity)
    - Permutation importance (model-agnostic)

Saves reports to experiments/importance/ as JSON.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from utils.logger import get_logger
from utils.config import resolve_path

logger = get_logger(__name__)

IMPORTANCE_DIR = resolve_path("experiments/importance")


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importances using multiple methods and
    provides a comprehensive report for model interpretability.
    """

    def __init__(self):
        os.makedirs(IMPORTANCE_DIR, exist_ok=True)

    def analyze(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_version: int,
        n_repeats: int = 5,
    ) -> Dict:
        """
        Run full feature importance analysis.

        Args:
            model: Trained model (must support predict()).
            X_test: Test feature DataFrame.
            y_test: Test target Series.
            model_version: Model version number for labeling.
            n_repeats: Number of repeats for permutation importance.

        Returns:
            Importance report dict.
        """
        logger.info(f"Analyzing feature importance for model v{model_version}")

        feature_names = list(X_test.columns)

        report = {
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        }

        # 1. Built-in importance (tree-based models)
        builtin = self._builtin_importance(model, feature_names)
        if builtin is not None:
            report["builtin_importance"] = builtin

        # 2. Permutation importance (model-agnostic)
        perm = self._permutation_importance(
            model, X_test, y_test, feature_names, n_repeats
        )
        report["permutation_importance"] = perm

        # 3. Combined ranking
        report["combined_ranking"] = self._combined_ranking(report)

        # Save report
        report_file = os.path.join(
            IMPORTANCE_DIR,
            f"importance_v{model_version}_{datetime.now():%Y%m%d_%H%M%S}.json",
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"Feature importance report saved to {report_file} "
            f"(top feature: {report['combined_ranking'][0]['feature']})"
        )
        return report

    def _builtin_importance(
        self, model: Any, feature_names: List[str]
    ) -> Optional[List[Dict]]:
        """Extract built-in feature importances from tree-based models."""
        if not hasattr(model, "feature_importances_"):
            logger.info("Model does not have built-in feature_importances_")
            return None

        importances = model.feature_importances_
        result = [
            {"feature": name, "importance": round(float(imp), 6)}
            for name, imp in zip(feature_names, importances)
        ]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result

    def _permutation_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        n_repeats: int = 5,
    ) -> List[Dict]:
        """Compute permutation importance (model-agnostic)."""
        logger.info(f"Computing permutation importance ({n_repeats} repeats)...")

        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        result = [
            {
                "feature": name,
                "importance_mean": round(float(-perm_result.importances_mean[i]), 6),
                "importance_std": round(float(perm_result.importances_std[i]), 6),
            }
            for i, name in enumerate(feature_names)
        ]
        result.sort(key=lambda x: x["importance_mean"], reverse=True)
        return result

    def _combined_ranking(self, report: Dict) -> List[Dict]:
        """
        Create a combined ranking by averaging normalized rankings
        from built-in and permutation importance.
        """
        features = report["feature_names"]
        n = len(features)

        # Build ranking from permutation importance
        perm_ranks = {}
        for rank, item in enumerate(report["permutation_importance"]):
            perm_ranks[item["feature"]] = rank

        # Build ranking from built-in importance (if available)
        builtin_ranks = {}
        if "builtin_importance" in report and report["builtin_importance"]:
            for rank, item in enumerate(report["builtin_importance"]):
                builtin_ranks[item["feature"]] = rank

        # Average ranks
        combined = []
        for feature in features:
            ranks = [perm_ranks.get(feature, n)]
            if builtin_ranks:
                ranks.append(builtin_ranks.get(feature, n))
            avg_rank = sum(ranks) / len(ranks)
            combined.append({"feature": feature, "avg_rank": round(avg_rank, 2)})

        combined.sort(key=lambda x: x["avg_rank"])

        # Add position
        for i, item in enumerate(combined):
            item["position"] = i + 1

        return combined

    @staticmethod
    def load_latest_report(model_version: Optional[int] = None) -> Optional[Dict]:
        """
        Load the most recent importance report.

        Args:
            model_version: If specified, load the latest for this version.

        Returns:
            Report dict, or None if not found.
        """
        if not os.path.exists(IMPORTANCE_DIR):
            return None

        import glob
        pattern = f"importance_v{model_version}_*.json" if model_version \
            else "importance_*.json"
        files = sorted(glob.glob(os.path.join(IMPORTANCE_DIR, pattern)))

        if not files:
            return None

        with open(files[-1], "r") as f:
            return json.load(f)
