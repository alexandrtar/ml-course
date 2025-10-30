from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
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

# Используем класс для хранения состояния вместо глобальных переменных
class ModelState:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.loaded = False
    
    def load_models(self):
        """Load models and resources"""
        try:
            logger.info("🔄 Loading models...")
            
            # Load trained model
            model_path = "models_improved/product_success_model.joblib"
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"✅ Model loaded successfully from {model_path}")
                logger.info(f"   Model type: {type(self.model).__name__}")
                if hasattr(self.model, 'n_features_in_'):
                    logger.info(f"   Model features: {self.model.n_features_in_}")
            else:
                logger.error(f"❌ Model file not found: {model_path}")
                return False
                
            # Load feature names
            feature_names_path = "models_improved/feature_names.txt"
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r', encoding='utf-8') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"✅ Loaded {len(self.feature_names)} feature names")
                logger.info(f"   First 5 features: {self.feature_names[:5]}")
            else:
                logger.error(f"❌ Feature names file not found: {feature_names_path}")
                return False
                
            # Verify model and features match
            if (hasattr(self.model, 'n_features_in_') and 
                self.model.n_features_in_ != len(self.feature_names)):
                logger.warning(f"⚠️ Feature count mismatch: model expects {self.model.n_features_in_}, but we have {len(self.feature_names)} features")
            
            self.loaded = True
            logger.info("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# Создаем экземпляр состояния
model_state = ModelState()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting up API...")
    success = model_state.load_models()
    if not success:
        logger.error("❌ CRITICAL: Failed to load models during startup!")
    yield
    # Shutdown
    logger.info("🛑 Shutting down API...")

app = FastAPI(
    title="Product Success Prediction API",
    description="API for predicting product success using multi-modal data",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with HTML interface"""
    model_status = "✅ Loaded" if model_state.loaded else "❌ Not Loaded"
    features_status = "✅ Loaded" if model_state.feature_names is not None else "❌ Not Loaded"
    
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
            .status-warning {{ border-left: 5px solid #ffc107; }}
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
            
            <div class="status {'status-success' if model_state.loaded else 'status-error'}">
                <h3>System Status</h3>
                <p><strong>Model:</strong> {model_status}</p>
                <p><strong>Features:</strong> {features_status}</p>
                <p><strong>Features Count:</strong> {len(model_state.feature_names) if model_state.feature_names else 0}</p>
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
                <span class="method">GET</span> 
                <span class="url">/model/metrics</span>
                <p>Get model performance metrics and evaluation results</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> 
                <span class="url">/predict</span>
                <p>Predict product success probability for a single product</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> 
                <span class="url">/features/importance</span>
                <p>Get feature importance ranking and analysis</p>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #e7f3ff; border-radius: 10px;">
                <h3>🚀 Quick Start</h3>
                <p><strong>Interactive Documentation:</strong> Visit <a href="/docs" target="_blank">/docs</a> for Swagger UI with live testing</p>
                <p><strong>API Base URL:</strong> <code>http://localhost:8000</code></p>
                <p><strong>Test Prediction:</strong> Use the form below to test the API</p>
                
                <form id="predictionForm" style="margin-top: 15px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <label>Gender:</label>
                            <select name="gender" style="width: 100%; padding: 8px;">
                                <option value="Men">Men</option>
                                <option value="Women">Women</option>
                                <option value="Unisex">Unisex</option>
                            </select>
                        </div>
                        <div>
                            <label>Article Type:</label>
                            <select name="articleType" style="width: 100%; padding: 8px;">
                                <option value="Tshirts">Tshirts</option>
                                <option value="Shirts">Shirts</option>
                                <option value="Dresses">Dresses</option>
                                <option value="Shoes">Shoes</option>
                                <option value="Jeans">Jeans</option>
                            </select>
                        </div>
                        <div>
                            <label>Color:</label>
                            <select name="baseColour" style="width: 100%; padding: 8px;">
                                <option value="Black">Black</option>
                                <option value="White">White</option>
                                <option value="Blue">Blue</option>
                                <option value="Red">Red</option>
                            </select>
                        </div>
                        <div>
                            <label>Season:</label>
                            <select name="season" style="width: 100%; padding: 8px;">
                                <option value="Summer">Summer</option>
                                <option value="Winter">Winter</option>
                                <option value="All">All</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" style="margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Test Prediction
                    </button>
                </form>
                
                <div id="result" style="margin-top: 15px; padding: 15px; border-radius: 5px; display: none;"></div>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const formData = new FormData(this);
                
                const requestData = {{
                    product_data: {{
                        id: 1,
                        gender: formData.get('gender'),
                        masterCategory: "Apparel",
                        subCategory: "Topwear",
                        articleType: formData.get('articleType'),
                        baseColour: formData.get('baseColour'),
                        season: formData.get('season'),
                        year: 2023,
                        usage: "Casual",
                        productDisplayName: "Test Product"
                    }},
                    image_url: ""
                }};
                
                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify(requestData)
                    }});
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (response.ok) {{
                        resultDiv.style.display = 'block';
                        resultDiv.style.background = result.success_probability > 0.5 ? '#d4edda' : '#f8d7da';
                        resultDiv.style.border = result.success_probability > 0.5 ? '1px solid #c3e6cb' : '1px solid #f5c6cb';
                        resultDiv.innerHTML = `
                            <h4>🎯 Prediction Result</h4>
                            <p><strong>Success Probability:</strong> ${(result.success_probability * 100).toFixed(1)}%</p>
                            <p><strong>Prediction:</strong> ${{result.prediction === 1 ? '✅ SUCCESS' : '⚠️ NEEDS IMPROVEMENT'}}</p>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Message:</strong> ${{result.message}}</p>
                        `;
                    }} else {{
                        resultDiv.style.display = 'block';
                        resultDiv.style.background = '#f8d7da';
                        resultDiv.style.border = '1px solid #f5c6cb';
                        resultDiv.innerHTML = `<p>❌ Error: ${{result.detail}}</p>`;
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    details = {}
    if model_state.model is not None:
        details["model_type"] = type(model_state.model).__name__
        if hasattr(model_state.model, 'n_features_in_'):
            details["model_features"] = model_state.model.n_features_in_
    if model_state.feature_names is not None:
        details["feature_count"] = len(model_state.feature_names)
        details["feature_sample"] = model_state.feature_names[:3]
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_state.loaded,
        features_loaded=model_state.feature_names is not None,
        details=details
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Predict success probability for a single product"""
    
    if not model_state.loaded or model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    if model_state.feature_names is None:
        raise HTTPException(status_code=503, detail="Feature names not loaded. Please check server logs.")
    
    try:
        logger.info(f"📦 Processing prediction for product {request.product_data.id}")
        
        # Create features from product data
        features = create_features_from_product(request.product_data.dict())
        logger.info(f"🔧 Created feature vector with {len(features)} values")
        
        # Make prediction
        probability = model_state.model.predict_proba([features])[0, 1]
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

def create_features_from_product(product_data: dict) -> list:
    """Create feature vector from product data"""
    try:
        # Initialize feature vector with zeros
        features = [0.0] * len(model_state.feature_names)
        
        # Map product data to features
        matched_features = 0
        for i, feature_name in enumerate(model_state.feature_names):
            # Check for categorical features
            if '_' in feature_name:
                try:
                    prefix, value = feature_name.split('_', 1)
                    
                    # Handle different feature types
                    product_value = product_data.get(prefix, "")
                    if product_value == value:
                        features[i] = 1.0
                        matched_features += 1
                except Exception as e:
                    continue
        
        logger.info(f"🔍 Matched {matched_features} features from product data")
        return features
        
    except Exception as e:
        logger.error(f"❌ Error creating features: {e}")
        return [0.0] * len(model_state.feature_names)

@app.get("/model/info", response_model=Dict[str, Any])
async def model_info():
    """Get information about the loaded model"""
    
    if not model_state.loaded or model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = type(model_state.model).__name__
    n_features = model_state.model.n_features_in_ if hasattr(model_state.model, 'n_features_in_') else "Unknown"
    
    return {
        "model_type": model_type,
        "n_features": n_features,
        "feature_names_loaded": model_state.feature_names is not None,
        "feature_count": len(model_state.feature_names) if model_state.feature_names else 0,
        "classes": model_state.model.classes_.tolist() if hasattr(model_state.model, 'classes_') else "Unknown",
        "model_loaded": model_state.loaded
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)