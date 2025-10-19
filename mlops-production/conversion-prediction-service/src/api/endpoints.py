from fastapi import APIRouter, HTTPException
from datetime import datetime
from .schemas import (
    PredictionRequest, PredictionResponse, HealthResponse,
    ModelInfoResponse, CategoriesResponse, ExampleResponse,
    BatchPredictionResponse
)
from ..models.predictor import ConversionPredictor
from ..utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize predictor
try:
    predictor = ConversionPredictor()
    MODEL_LOADED = True
    logger.info("✅ Predictor initialized successfully")
except Exception as e:
    logger.error(f"❌ Predictor initialization failed: {e}")
    MODEL_LOADED = False
    predictor = None

@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        model_version="1.1.0",
        timestamp=datetime.now().isoformat()
    )

@router.get("/model/info", response_model=ModelInfoResponse, summary="Model Information")
async def model_info():
    """Get information about the trained model"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=type(predictor.model).__name__,
        feature_count=len(predictor.feature_info['feature_columns']),
        categorical_features=len(predictor.feature_info['categorical_columns']),
        numerical_features=len(predictor.feature_info['numerical_columns']),
        optimal_threshold=predictor.optimal_threshold,
        feature_columns=predictor.feature_info['feature_columns']
    )

@router.get("/categories", response_model=CategoriesResponse, summary="Available Categories")
async def get_categories():
    """Get available categories for categorical features"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return CategoriesResponse(
        available_categories=predictor.get_available_categories(),
        message="Use these values in requests to avoid errors"
    )

@router.get("/example", response_model=ExampleResponse, summary="Example Request")
async def get_example():
    """Get example request with valid data"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    example_data = {
        "utm_source_type": "social",
        "utm_medium": "cpc",
        "device_category": "mobile",
        "os_type": "Android",
        "device_browser": "Chrome",
        "day_of_week": 2,
        "hour_of_day": 14,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "time_of_day": "afternoon",
        "is_weekend": 0,
        "month": 6,
        "is_peak_hours": 1,
        "session_hits_count": 15,
        "unique_event_categories": 5,
        "unique_event_actions": 8,
        "unique_pages": 10,
        "is_returning_user": 0,
        "event_category_group": "submission_click",
        "country_region": "cis"
    }
    
    return ExampleResponse(
        example_request=example_data,
        available_categories=predictor.get_available_categories()
    )

@router.post("/predict", response_model=PredictionResponse, summary="Single Prediction")
async def predict_conversion(request: PredictionRequest):
    """Predict conversion probability for a single sample"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = predictor.predict(request.dict())
    
    if not result.get('success', False):
        raise HTTPException(
            status_code=400, 
            detail=result.get('error', 'Unknown error')
        )
    
    return PredictionResponse(**result)

@router.post("/batch_predict", response_model=BatchPredictionResponse, summary="Batch Prediction")
async def batch_predict(requests: list[PredictionRequest]):
    """Predict conversion probabilities for multiple samples"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_list = [request.dict() for request in requests]
        result = predictor.batch_predict(features_list)
        return BatchPredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")