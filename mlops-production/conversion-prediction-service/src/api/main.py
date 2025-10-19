from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import List, Optional
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GA Conversion Prediction API",
    description="API для предсказания конверсий на основе Google Analytics данных",
    version="1.0.0"
)

# Модели данных
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
    model_version: str = "1.0.0"
    timestamp: str

class ExampleResponse(BaseModel):
    example_request: dict
    available_categories: dict

# Загрузка модели
try:
    model = joblib.load('models/conversion_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    optimal_threshold = feature_info.get('optimal_threshold', 0.5)

    MODEL_LOADED = True
    logger.info("✅ Модель и артефакты успешно загружены")

    # Получаем доступные категории для каждого энкодера
    available_categories = {}
    for col, encoder in label_encoders.items():
        available_categories[col] = list(encoder.classes_)
    logger.info("✅ Категории загружены")

except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    MODEL_LOADED = False
    model = None
    label_encoders = {}
    feature_info = {}
    available_categories = {}
    optimal_threshold = 0.5

def safe_label_transform(encoder, values):
    """Безопасное преобразование label encoder с обработкой неизвестных значений"""
    try:
        if not hasattr(values, '__iter__') or isinstance(values, str):
            values = [values]

        transformed = []
        for val in values:
            val_str = str(val)
            if val_str in encoder.classes_:
                transformed.append(encoder.transform([val_str])[0])
            else:
                # Используем самое частое значение как fallback
                transformed.append(0)
                logger.warning(f"⚠️ Неизвестное значение '{val_str}'. Использовано значение по умолчанию.")

        return transformed[0] if len(transformed) == 1 else transformed

    except Exception as e:
        logger.error(f"Ошибка преобразования: {e}")
        return 0

def predict_conversion(features_dict):
    """Функция для предсказания конверсии"""
    if not MODEL_LOADED:
        return {"error": "Модель не загружена", "success": False}

    try:
        # Создаем DataFrame из входных данных
        input_data = pd.DataFrame([features_dict])

        # Применяем Label Encoding с безопасной обработкой
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                encoded_value = safe_label_transform(encoder, input_data[col].iloc[0])
                input_data[col + '_encoded'] = encoded_value

        # Подготавливаем фичи
        available_columns = [col for col in feature_info['feature_columns'] if col in input_data.columns]

        # Проверяем наличие всех необходимых фич
        missing_features = set(feature_info['feature_columns']) - set(available_columns)
        if missing_features:
            return {
                "error": f"Отсутствуют фичи: {missing_features}",
                "success": False
            }

        X_input = input_data[available_columns]

        # Предсказание
        prediction_proba = model.predict_proba(X_input)[0]
        conversion_prob = prediction_proba[1]
        no_conversion_prob = prediction_proba[0]

        # Применяем оптимальный порог
        prediction = 1 if conversion_prob >= optimal_threshold else 0

        return {
            "conversion_prediction": prediction,
            "conversion_probability": conversion_prob,
            "no_conversion_probability": no_conversion_prob,
            "threshold_used": optimal_threshold,
            "success": True
        }

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        return {"error": str(e), "success": False}

@app.get("/", summary="Главная страница")
async def root():
    return {
        "message": "GA Conversion Prediction API",
        "status": "active" if MODEL_LOADED else "error",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
        "endpoints": [
            "/docs - документация",
            "/health - статус API",
            "/model/info - информация о модели",
            "/example - пример запроса",
            "/predict - предсказание конверсии",
            "/categories - доступные категории"
        ]
    }

@app.get("/health", response_model=HealthResponse, summary="Проверка здоровья API")
async def health_check():
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "unhealthy",
        model_loaded=MODEL_LOADED,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", summary="Информация о модели")
async def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_info['feature_columns']),
        "categorical_features": len(feature_info['categorical_columns']),
        "numerical_features": len(feature_info['numerical_columns']),
        "optimal_threshold": optimal_threshold,
        "feature_columns": feature_info['feature_columns']
    }

@app.get("/categories", summary="Доступные категории")
async def get_categories():
    """Возвращает доступные категории для каждого поля"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    return {
        "available_categories": available_categories,
        "message": "Используйте эти значения в запросах для избежания ошибок"
    }

@app.get("/example", response_model=ExampleResponse, summary="Пример запроса")
async def get_example():
    """Возвращает пример валидного запроса"""
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
        "event_category_group": "engagement_click",
        "country_region": "cis"
    }

    return ExampleResponse(
        example_request=example_data,
        available_categories=available_categories
    )

@app.post("/predict", response_model=PredictionResponse, summary="Предсказание конверсии")
async def predict_conversion_endpoint(request: PredictionRequest):
    """
    Предсказание вероятности конверсии на основе входных параметров

    Важно: Используйте только категории из /categories эндпоинта
    """
    result = predict_conversion(request.dict())

    if not result.get('success', False):
        raise HTTPException(
            status_code=400,
            detail=result.get('error', 'Unknown error')
        )

    return PredictionResponse(**result)

if __name__ == "__main__":
    import uvicorn

    print("🚀 Запуск GA Conversion Prediction API...")
    print("=" * 50)
    print("📊 ИНФОРМАЦИЯ О МОДЕЛИ:")
    if MODEL_LOADED:
        print(f"   • Модель: {type(model).__name__}")
        print(f"   • Фичи: {len(feature_info['feature_columns'])}")
        print(f"   • Порог: {optimal_threshold:.4f}")
    print("🌐 ДОСТУПНЫЕ ЭНДПОИНТЫ:")
    print("   • http://localhost:8000 - главная")
    print("   • http://localhost:8000/docs - документация")
    print("   • http://localhost:8000/health - статус")
    print("   • http://localhost:8000/model/info - информация о модели")
    print("   • http://localhost:8000/example - пример запроса")
    print("   • http://localhost:8000/categories - доступные категории")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")