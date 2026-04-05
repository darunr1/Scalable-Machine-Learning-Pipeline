"""
Pydantic schemas for the inference API.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""

    temperature_2m: float = Field(..., description="Current temperature (°C)")
    relative_humidity_2m: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    dew_point_2m: float = Field(..., description="Dew point temperature (°C)")
    apparent_temperature: float = Field(..., description="Apparent temperature (°C)")
    pressure_msl: float = Field(..., ge=870, le=1084, description="Mean sea level pressure (hPa)")
    surface_pressure: float = Field(..., description="Surface pressure (hPa)")
    precipitation: float = Field(..., ge=0, description="Precipitation (mm)")
    rain: float = Field(..., ge=0, description="Rain (mm)")
    snowfall: float = Field(..., ge=0, description="Snowfall (cm)")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    wind_speed_10m: float = Field(..., ge=0, description="Wind speed at 10m (km/h)")
    wind_direction_10m: float = Field(..., ge=0, le=360, description="Wind direction (°)")
    wind_gusts_10m: float = Field(..., ge=0, description="Wind gusts at 10m (km/h)")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of day (0-23)")

    class Config:
        json_schema_extra = {
            "example": {
                "temperature_2m": 15.2,
                "relative_humidity_2m": 65.0,
                "dew_point_2m": 8.5,
                "apparent_temperature": 13.1,
                "pressure_msl": 1013.25,
                "surface_pressure": 1010.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "snowfall": 0.0,
                "cloud_cover": 40.0,
                "wind_speed_10m": 12.5,
                "wind_direction_10m": 180.0,
                "wind_gusts_10m": 25.0,
                "hour": 14,
            }
        }


class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""

    prediction: float = Field(..., description="Predicted temperature (°C)")
    model_version: int = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    confidence: Optional[Dict] = Field(default=None, description="Prediction confidence info")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_version: Optional[int] = None
    model_metrics: Optional[Dict] = None


class ModelInfoResponse(BaseModel):
    """Response body for the /model/info endpoint."""

    version: int
    metrics: Dict
    training_info: Dict
    registered_at: str
    is_production: bool
