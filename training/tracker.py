"""
Experiment Tracker.

Logs every training run with full reproducibility info:
    - Hyperparameters used
    - Metrics achieved
    - Data hash (for reproducibility)
    - Training duration
    - Git commit (if available)

Stores runs in experiments/runs.jsonl for easy querying.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.config import resolve_path

logger = get_logger(__name__)

EXPERIMENTS_DIR = resolve_path("experiments")
RUNS_FILE = os.path.join(EXPERIMENTS_DIR, "runs.jsonl")


def _get_git_commit() -> Optional[str]:
    """Get the current git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _compute_data_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic hash of a DataFrame for reproducibility."""
    content = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:12]


class ExperimentTracker:
    """
    Lightweight experiment tracker.

    Logs each training run as a JSON line in experiments/runs.jsonl,
    capturing everything needed to reproduce or compare runs.
    """

    def __init__(self, experiment_name: str = "default"):
        self.experiment_name = experiment_name
        self._run_id: Optional[str] = None
        self._start_time: Optional[float] = None
        self._params: Dict[str, Any] = {}
        self._metrics: Dict[str, float] = {}
        self._tags: Dict[str, str] = {}
        self._artifacts: List[str] = []

        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    def start_run(self, run_name: Optional[str] = None) -> "ExperimentTracker":
        """Begin a new experiment run."""
        self._start_time = time.time()
        self._run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self._params = {}
        self._metrics = {}
        self._tags = {"run_name": run_name or self._run_id}
        self._artifacts = []

        logger.info(f"Started experiment run: {self._run_id}")
        return self

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters for this run."""
        # Convert non-serializable values to strings
        for key, value in params.items():
            if value is None:
                self._params[key] = "None"
            elif isinstance(value, (int, float, str, bool)):
                self._params[key] = value
            else:
                self._params[key] = str(value)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics for this run."""
        self._metrics.update(metrics)

    def log_tag(self, key: str, value: str):
        """Log a tag (arbitrary metadata) for this run."""
        self._tags[key] = value

    def log_data_info(self, df: pd.DataFrame):
        """Log dataset information for reproducibility."""
        self._tags["data_hash"] = _compute_data_hash(df)
        self._tags["data_rows"] = str(len(df))
        self._tags["data_cols"] = str(len(df.columns))

    def log_artifact(self, path: str):
        """Record path of a saved artifact (model, pipeline, etc.)."""
        self._artifacts.append(path)

    def end_run(self) -> Dict:
        """
        End the current run and save it to the runs file.

        Returns:
            The complete run record.
        """
        if self._start_time is None:
            raise RuntimeError("No active run. Call start_run() first.")

        duration = round(time.time() - self._start_time, 2)
        git_commit = _get_git_commit()

        record = {
            "run_id": self._run_id,
            "experiment": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "params": self._params,
            "metrics": self._metrics,
            "tags": self._tags,
            "artifacts": self._artifacts,
            "git_commit": git_commit,
        }

        # Append to JSONL
        with open(RUNS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(
            f"Run {self._run_id} complete — "
            f"duration={duration}s, metrics={self._metrics}"
        )

        # Reset state
        self._start_time = None
        return record

    @staticmethod
    def list_runs(experiment: Optional[str] = None) -> List[Dict]:
        """
        List all experiment runs, optionally filtered by experiment name.

        Returns:
            List of run records, newest first.
        """
        if not os.path.exists(RUNS_FILE):
            return []

        runs = []
        with open(RUNS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    run = json.loads(line)
                    if experiment is None or run.get("experiment") == experiment:
                        runs.append(run)

        return list(reversed(runs))

    @staticmethod
    def get_best_run(metric: str = "mae", lower_is_better: bool = True) -> Optional[Dict]:
        """
        Get the best run by a specific metric.

        Args:
            metric: Metric name to compare.
            lower_is_better: If True, lower metric = better.

        Returns:
            Best run record, or None.
        """
        runs = ExperimentTracker.list_runs()
        scored_runs = [r for r in runs if metric in r.get("metrics", {})]

        if not scored_runs:
            return None

        return min(scored_runs, key=lambda r: r["metrics"][metric]) if lower_is_better \
            else max(scored_runs, key=lambda r: r["metrics"][metric])

    @staticmethod
    def compare_runs(run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs side by side.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            DataFrame with runs as rows and metrics/params as columns.
        """
        all_runs = ExperimentTracker.list_runs()
        selected = [r for r in all_runs if r["run_id"] in run_ids]

        if not selected:
            return pd.DataFrame()

        rows = []
        for run in selected:
            row = {
                "run_id": run["run_id"],
                "timestamp": run["timestamp"][:19],
                "duration_s": run["duration_seconds"],
                "git_commit": run.get("git_commit", "N/A"),
            }
            row.update({f"metric_{k}": v for k, v in run.get("metrics", {}).items()})
            row.update({f"param_{k}": v for k, v in run.get("params", {}).items()})
            rows.append(row)

        return pd.DataFrame(rows)
