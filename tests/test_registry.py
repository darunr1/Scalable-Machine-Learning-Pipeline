"""Tests for the model registry."""

import os
import tempfile
import json

import pytest
from sklearn.linear_model import LinearRegression

from models.registry import ModelRegistry
from utils.config import load_config


def make_test_registry(tmp_path):
    """Create a registry backed by a temp directory."""
    config = load_config()
    config["registry"]["model_dir"] = str(tmp_path / "models")
    config["registry"]["metadata_file"] = str(tmp_path / "models" / "metadata.json")
    return ModelRegistry(config)


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self, tmp_path):
        """Registering a model should save artifact and metadata."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()
        metrics = {"mae": 1.5, "rmse": 2.0, "r2": 0.85}

        version = registry.register_model(model, metrics)
        assert version == 1

        # Check file exists
        model_path = os.path.join(str(tmp_path / "models"), "model_v1.pkl")
        assert os.path.exists(model_path)

    def test_register_multiple_models(self, tmp_path):
        """Versions should increment with each registration."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()

        v1 = registry.register_model(model, {"mae": 2.0})
        v2 = registry.register_model(model, {"mae": 1.5})
        v3 = registry.register_model(model, {"mae": 1.0})

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_list_models(self, tmp_path):
        """Should list all registered models."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()

        registry.register_model(model, {"mae": 2.0})
        registry.register_model(model, {"mae": 1.5})

        models = registry.list_models()
        assert len(models) == 2

    def test_load_model(self, tmp_path):
        """Should load a specific model version."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()
        model.coef_ = [1.0, 2.0]

        registry.register_model(model, {"mae": 1.0})
        loaded = registry.load_model(version=1)
        assert loaded is not None

    def test_promote_model(self, tmp_path):
        """Should promote a specific version to production."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()

        registry.register_model(model, {"mae": 2.0})
        registry.register_model(model, {"mae": 1.0})

        registry.promote_model(2)
        assert registry.get_production_version() == 2

        # Check v1 is no longer production
        info = registry.get_model_info(1)
        assert info["is_production"] is False

    def test_compare_models(self, tmp_path):
        """Should compare metrics of two versions."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()

        registry.register_model(model, {"mae": 2.0, "rmse": 3.0})
        registry.register_model(model, {"mae": 1.5, "rmse": 2.5})

        comparison = registry.compare_models(1, 2)
        assert comparison["diff"]["mae"] == -0.5
        assert comparison["diff"]["rmse"] == -0.5

    def test_auto_promote_first(self, tmp_path):
        """First model should be auto-promoted to production."""
        registry = make_test_registry(tmp_path)
        model = LinearRegression()

        registry.register_model(model, {"mae": 1.0})
        assert registry.get_production_version() == 1
