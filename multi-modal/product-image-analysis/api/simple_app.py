from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Any, Optional
import logging
from fastapi.responses import HTMLResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Простая загрузка моделей при старте
def load_models_simple():
    """Load models at startup"""
    try:
        logger.info("🔄 Loading models...")
        
        # Load model
        model_path = "models_improved/product_success_model.joblib"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None, None
            
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded: {type(model).__name__}")
        
        # Load features
        features_path = "models_improved/feature_names.txt"
        if not os.path.exists(features_path):
            logger.error(f"❌ Features file not found: {features_path}")
            return model, None
            
        with open(features_path, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Features loaded: {len(feature_names)} features")
        
        return model, feature_names
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None, None

# Загружаем модели сразу
MODEL, FEATURE_NAMES = load_models_simple()
MODEL_LOADED = MODEL is not None and FEATURE_NAMES is not None

class ProductData(BaseModel):
    id: int
    gender: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: str
    season: str
    year: int
    usage: str
    productDisplayName: str

class PredictionRequest(BaseModel):
    product_data: ProductData
    image_url: Optional[str] = ""

class PredictionResponse(BaseModel):
    product_id: int
    success_probability: float
    prediction: int
    confidence: float
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_loaded: bool

app = FastAPI(
    title="Product Success Prediction API",
    description="API for predicting product success using multi-modal data",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
    <html>
    <body>
        <h1>🎯 Product Success Prediction API</h1>
        <p>Model Loaded: {'✅ Yes' if MODEL_LOADED else '❌ No'}</p>
        <p>Features Loaded: {'✅ Yes' if FEATURE_NAMES is not None else '❌ No'}</p>
        <p><a href="/docs">API Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_LOADED,
        features_loaded=FEATURE_NAMES is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create features
        features = [0.0] * len(FEATURE_NAMES)
        for i, feature_name in enumerate(FEATURE_NAMES):
            if '_' in feature_name:
                prefix, value = feature_name.split('_', 1)
                product_value = request.product_data.dict().get(prefix, "")
                if product_value == value:
                    features[i] = 1.0
        
        # Make prediction
        probability = MODEL.predict_proba([features])[0, 1]
        prediction = 1 if probability > 0.5 else 0
        
        return PredictionResponse(
            product_id=request.product_data.id,
            success_probability=float(probability),
            prediction=prediction,
            confidence=float(probability if prediction == 1 else 1 - probability),
            message="✅ Prediction successful" if prediction == 1 else "⚠️ Needs improvement"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)