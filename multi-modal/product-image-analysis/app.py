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

# Глобальные переменные для моделей
MODEL = None
FEATURE_NAMES = None
MODELS_LOADED = False

def load_models():
    """Load models at startup"""
    global MODEL, FEATURE_NAMES, MODELS_LOADED
    
    try:
        logger.info("🔄 Loading models...")
        
        # Load model
        model_path = "models_improved/product_success_model.joblib"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
            
        MODEL = joblib.load(model_path)
        logger.info(f"✅ Model loaded: {type(MODEL).__name__}")
        logger.info(f"   Model features: {MODEL.n_features_in_}")
        
        # Load features
        features_path = "models_improved/feature_names.txt"
        if not os.path.exists(features_path):
            logger.error(f"❌ Features file not found: {features_path}")
            return False
            
        with open(features_path, 'r', encoding='utf-8') as f:
            FEATURE_NAMES = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Features loaded: {len(FEATURE_NAMES)} features")
        logger.info(f"   First 5 features: {FEATURE_NAMES[:5]}")
        
        # Verify consistency
        if MODEL.n_features_in_ != len(FEATURE_NAMES):
            logger.warning(f"⚠️ Feature count mismatch: model expects {MODEL.n_features_in_}, features file has {len(FEATURE_NAMES)}")
        
        MODELS_LOADED = True
        logger.info("🎉 All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Загружаем модели при старте
load_models()

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
    details: Optional[Dict] = None

app = FastAPI(
    title="Product Success Prediction API",
    description="API for predicting product success using multi-modal data",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with HTML interface"""
    model_status = "✅ Loaded" if MODELS_LOADED else "❌ Not Loaded"
    features_status = "✅ Loaded" if FEATURE_NAMES is not None else "❌ Not Loaded"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Product Success Prediction API</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h1 {{ 
                color: #333; 
                text-align: center;
                margin-bottom: 30px;
            }}
            .status {{ 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 10px;
                background: #f8f9fa;
            }}
            .status-success {{ border-left: 5px solid #28a745; }}
            .status-error {{ border-left: 5px solid #dc3545; }}
            .endpoint {{ 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px;
                border-left: 5px solid #007bff;
            }}
            .method {{ 
                display: inline-block; 
                padding: 8px 15px; 
                background: #007bff; 
                color: white; 
                border-radius: 5px; 
                font-weight: bold;
                margin-right: 10px;
            }}
            .url {{ 
                font-family: 'Courier New', monospace; 
                color: #28a745; 
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Product Success Prediction API</h1>
            
            <div class="status {'status-success' if MODELS_LOADED else 'status-error'}">
                <h3>System Status</h3>
                <p><strong>Model:</strong> {model_status}</p>
                <p><strong>Features:</strong> {features_status}</p>
                <p><strong>Features Count:</strong> {len(FEATURE_NAMES) if FEATURE_NAMES else 0}</p>
                <p><strong>API Status:</strong> ✅ Running</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> 
                <span class="url">/health</span> 
                <p>Check API health and model status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> 
                <span class="url">/model/info</span>
                <p>Get detailed information about the loaded model</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> 
                <span class="url">/predict</span>
                <p>Predict product success probability for a single product</p>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 10px;">
                <h3>🚀 Quick Start</h3>
                <p><strong>Interactive Documentation:</strong> Visit <a href="/docs" target="_blank">/docs</a> for Swagger UI with live testing</p>
                <p><strong>API Base URL:</strong> <code>http://localhost:8000</code></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    details = {}
    if MODEL is not None:
        details["model_type"] = type(MODEL).__name__
        if hasattr(MODEL, 'n_features_in_'):
            details["model_features"] = MODEL.n_features_in_
    if FEATURE_NAMES is not None:
        details["feature_count"] = len(FEATURE_NAMES)
        details["feature_sample"] = FEATURE_NAMES[:3]
    
    return HealthResponse(
        status="healthy",
        model_loaded=MODELS_LOADED,
        features_loaded=FEATURE_NAMES is not None,
        details=details
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict success probability for a single product"""
    
    if not MODELS_LOADED or MODEL is None:
        logger.error("❌ Model not loaded for prediction")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if FEATURE_NAMES is None:
        logger.error("❌ Feature names not loaded for prediction")
        raise HTTPException(status_code=503, detail="Feature names not loaded")
    
    try:
        logger.info(f"📦 Processing prediction for product {request.product_data.id}")
        
        # Create features from product data
        features = [0.0] * len(FEATURE_NAMES)
        matched_features = 0
        
        product_dict = request.product_data.dict()
        logger.info(f"🔍 Product data: {product_dict}")
        
        for i, feature_name in enumerate(FEATURE_NAMES):
            # Check for categorical features
            if '_' in feature_name:
                try:
                    prefix, value = feature_name.split('_', 1)
                    
                    # Handle different feature types
                    product_value = product_dict.get(prefix, "")
                    if product_value == value:
                        features[i] = 1.0
                        matched_features += 1
                        logger.debug(f"   Matched feature: {feature_name}")
                except Exception as e:
                    logger.warning(f"   Error processing feature {feature_name}: {e}")
                    continue
        
        logger.info(f"🔧 Created feature vector: matched {matched_features} features")
        logger.info(f"   Feature vector sample: {features[:10]}")
        
        # Make prediction
        probability = MODEL.predict_proba([features])[0, 1]
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else 1 - probability
        
        # Generate message based on probability
        if prediction == 1:
            if probability > 0.8:
                message = "🎉 Excellent! Very high probability of success!"
            elif probability > 0.6:
                message = "✅ Good! High probability of success."
            else:
                message = "📊 Fair. Moderate probability of success."
        else:
            if probability < 0.2:
                message = "⚠️ Low success probability. Major improvements needed."
            elif probability < 0.4:
                message = "📉 Below average. Significant improvements recommended."
            else:
                message = "📈 Close to success. Minor optimizations could help."
        
        logger.info(f"🎯 Prediction result: {probability:.3f} -> {'Success' if prediction == 1 else 'Needs Improvement'}")
        
        return PredictionResponse(
            product_id=request.product_data.id,
            success_probability=float(probability),
            prediction=prediction,
            confidence=float(confidence),
            message=message
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get information about the loaded model"""
    
    if not MODELS_LOADED or MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = type(MODEL).__name__
    n_features = MODEL.n_features_in_ if hasattr(MODEL, 'n_features_in_') else "Unknown"
    
    return {
        "model_type": model_type,
        "n_features": n_features,
        "feature_names_loaded": FEATURE_NAMES is not None,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "classes": MODEL.classes_.tolist() if hasattr(MODEL, 'classes_') else "Unknown",
        "model_loaded": MODELS_LOADED
    }

@app.get("/model/metrics", response_model=Dict[str, Any])
async def model_metrics():
    """Get model performance metrics"""
    try:
        metrics_path = "results_improved/metrics.json"
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            return metrics
        else:
            raise HTTPException(status_code=404, detail="Metrics file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {e}")

@app.get("/features/importance", response_model=List[Dict[str, Any]])
async def feature_importance():
    """Get feature importance ranking"""
    try:
        importance_path = "results_improved/feature_importance.csv"
        if os.path.exists(importance_path):
            df = pd.read_csv(importance_path)
            return df.head(10).to_dict('records')
        else:
            raise HTTPException(status_code=404, detail="Feature importance file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading feature importance: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Product Success Prediction API...")
    print("📊 Model: RandomForest with 50 features")
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("-" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)