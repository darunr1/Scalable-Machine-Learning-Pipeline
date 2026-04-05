"""
Configuration Validator.

Validates the YAML config against expected schema using Pydantic models.
Catches missing fields, wrong types, and invalid ranges at load time
instead of failing deep inside the pipeline with cryptic errors.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class APIConfig(BaseModel):
    """Open-Meteo API configuration."""
    base_url: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    hourly_variables: List[str] = Field(min_length=1)
    timezone: str


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""
    api: APIConfig
    data_dir: str
    log_file: str


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    target_column: str
    lag_features: List[int] = Field(min_length=1)
    rolling_windows: List[int] = Field(min_length=1)
    cyclical_features: List[str]
    numerical_features: List[str] = Field(min_length=1)
    pipeline_artifact: str

    @field_validator("lag_features", "rolling_windows")
    @classmethod
    def must_be_positive(cls, v: List[int]) -> List[int]:
        for val in v:
            if val <= 0:
                raise ValueError(f"Values must be positive, got {val}")
        return v


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""
    test_size: float = Field(gt=0, lt=1)
    cv_folds: int = Field(ge=2, le=20)
    model_type: str
    param_grid: Dict
    scoring: str
    random_state: int


class RegistryConfig(BaseModel):
    """Model registry configuration."""
    model_dir: str
    metadata_file: str


class APIServerConfig(BaseModel):
    """API server configuration."""
    host: str
    port: int = Field(ge=1, le=65535)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prediction_log: str
    reports_dir: str


class DriftConfig(BaseModel):
    """Drift detection configuration."""
    ks_threshold: float = Field(gt=0, lt=1)
    check_features: List[str] = Field(min_length=1)
    baseline_file: str
    reports_dir: str


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""
    ingestion_interval: str
    monitoring_interval: str
    drift_check_interval: str
    retraining_interval: str


class PipelineConfig(BaseModel):
    """Complete pipeline configuration schema."""
    ingestion: IngestionConfig
    features: FeaturesConfig
    training: TrainingConfig
    registry: RegistryConfig
    api: APIServerConfig
    monitoring: MonitoringConfig
    drift: DriftConfig
    scheduler: SchedulerConfig


def validate_config(config: dict) -> dict:
    """
    Validate a config dict against the expected schema.

    Args:
        config: Raw config dict from YAML.

    Returns:
        The validated config dict (unchanged if valid).

    Raises:
        pydantic.ValidationError: If config is invalid, with clear
        field-level error messages.
    """
    PipelineConfig(**config)
    return config
