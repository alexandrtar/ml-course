from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    utm_source_type: str
    utm_medium: str
    device_category: str
    os_type: str
    device_browser: str
    day_of_week: int
    hour_of_day: int
    hour_sin: float
    hour_cos: float
    time_of_day: str
    is_weekend: int
    month: int
    is_peak_hours: int
    session_hits_count: int
    unique_event_categories: int
    unique_event_actions: int
    unique_pages: int
    is_returning_user: int
    event_category_group: str
    country_region: str

class PredictionResponse(BaseModel):
    conversion_prediction: int
    conversion_probability: float
    no_conversion_probability: float
    threshold_used: float
    success: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    categorical_features: int
    numerical_features: int
    optimal_threshold: float
    feature_columns: List[str]

class CategoriesResponse(BaseModel):
    available_categories: Dict[str, List[str]]
    message: str

class ExampleResponse(BaseModel):
    example_request: Dict[str, Any]
    available_categories: Dict[str, List[str]]

class BatchPredictionResponse(BaseModel):
    total_processed: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]