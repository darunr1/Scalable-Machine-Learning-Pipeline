"""
Model Registry.

Manages versioned model storage with metadata tracking.
Supports: register, list, load, promote, compare operations.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

from utils.logger import get_logger
from utils.config import load_config, resolve_path

logger = get_logger(__name__)


class ModelRegistry:
    """
    Production model registry with versioning and metadata.

    Models are stored as:
        models/
        ├── model_v1.pkl
        ├── model_v2.pkl
        └── metadata.json

    metadata.json tracks:
        - version, metrics, training date, dataset info,
          hyperparams, is_production flag
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()

        reg_config = config["registry"]
        self.model_dir = resolve_path(reg_config["model_dir"])
        self.metadata_file = resolve_path(reg_config["metadata_file"])

        os.makedirs(self.model_dir, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Load or initialize metadata file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"models": [], "production_version": None}

    def _save_metadata(self, metadata: Dict):
        """Save metadata file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def register_model(
        self,
        model: Any,
        metrics: Dict,
        training_info: Optional[Dict] = None,
    ) -> int:
        """
        Register a new model version.

        Args:
            model: Trained model object (must be pickle-serializable).
            metrics: Dict of metric name → value (e.g., {'mae': 1.23}).
            training_info: Optional dict with hyperparameters, dates, etc.

        Returns:
            Version number of the registered model.
        """
        metadata = self._load_metadata()

        # Determine next version
        version = len(metadata["models"]) + 1

        # Save model artifact
        model_path = os.path.join(self.model_dir, f"model_v{version}.pkl")
        joblib.dump(model, model_path)

        # Create model entry
        entry = {
            "version": version,
            "model_path": model_path,
            "metrics": metrics,
            "training_info": training_info or {},
            "registered_at": datetime.now().isoformat(),
            "is_production": False,
        }

        metadata["models"].append(entry)

        # Auto-promote first model, or promote if better than current
        if metadata["production_version"] is None:
            entry["is_production"] = True
            metadata["production_version"] = version
            logger.info(f"Auto-promoted model v{version} to production (first model)")

        self._save_metadata(metadata)
        logger.info(f"Registered model v{version} at {model_path}")
        logger.info(f"Metrics: {metrics}")

        return version

    def list_models(self) -> List[Dict]:
        """List all registered model versions with metadata."""
        metadata = self._load_metadata()
        return metadata["models"]

    def load_model(self, version: Optional[int] = None) -> Any:
        """
        Load a model by version. If version is None, loads the production model.

        Args:
            version: Model version number. None = production model.

        Returns:
            Loaded model object.
        """
        metadata = self._load_metadata()

        if version is None:
            version = metadata.get("production_version")
            if version is None:
                raise ValueError("No production model set")

        model_path = os.path.join(self.model_dir, f"model_v{version}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model v{version} not found at {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Loaded model v{version}")
        return model

    def get_model_info(self, version: Optional[int] = None) -> Dict:
        """Get metadata for a specific model version."""
        metadata = self._load_metadata()

        if version is None:
            version = metadata.get("production_version")

        for entry in metadata["models"]:
            if entry["version"] == version:
                return entry

        raise ValueError(f"Model v{version} not found in registry")

    def promote_model(self, version: int):
        """
        Promote a model version to production.

        Args:
            version: Version number to promote.
        """
        metadata = self._load_metadata()

        # Demote current production model
        for entry in metadata["models"]:
            entry["is_production"] = False

        # Promote specified version
        found = False
        for entry in metadata["models"]:
            if entry["version"] == version:
                entry["is_production"] = True
                metadata["production_version"] = version
                found = True
                break

        if not found:
            raise ValueError(f"Model v{version} not found in registry")

        self._save_metadata(metadata)
        logger.info(f"Promoted model v{version} to production")

    def compare_models(self, version_a: int, version_b: int) -> Dict:
        """
        Compare metrics of two model versions.

        Returns:
            Dict with side-by-side comparison.
        """
        info_a = self.get_model_info(version_a)
        info_b = self.get_model_info(version_b)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_a": info_a["metrics"],
            "metrics_b": info_b["metrics"],
            "diff": {},
        }

        for metric in info_a["metrics"]:
            if metric in info_b["metrics"]:
                diff = info_b["metrics"][metric] - info_a["metrics"][metric]
                comparison["diff"][metric] = round(diff, 4)

        return comparison

    def get_production_version(self) -> Optional[int]:
        """Get the current production model version number."""
        metadata = self._load_metadata()
        return metadata.get("production_version")
