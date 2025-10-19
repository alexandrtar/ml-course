"""
Пример API для предсказания конверсий
"""

import joblib
import pandas as pd

# Загрузка модели (выполняется один раз при старте приложения)
try:
    model = joblib.load('models/conversion_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl') 
    feature_info = joblib.load('models/feature_info.pkl')
    print("✅ Модель загружена")
except:
    print("❌ Ошибка загрузки модели")
    model = None

def predict_conversion(features_dict):
    """
    Предсказание конверсии на основе входных features
    
    Parameters:
    features_dict (dict): Словарь с фичами
    
    Returns:
    dict: Результаты предсказания
    """
    if model is None:
        return {"error": "Модель не загружена"}
    
    try:
        # Создаем DataFrame из входных данных
        input_data = pd.DataFrame([features_dict])
        
        # Применяем Label Encoding к категориальным фичам
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)
                known_categories = set(encoder.classes_)
                input_data[col] = input_data[col].apply(lambda x: x if x in known_categories else 'unknown')
                input_data[col + '_encoded'] = encoder.transform(input_data[col])
        
        # Подготавливаем фичи для модели
        X = input_data[feature_info['feature_columns']]
        
        # Предсказание
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        return {
            'conversion_prediction': int(prediction),
            'conversion_probability': float(prediction_proba[1]),
            'no_conversion_probability': float(prediction_proba[0]),
            'success': True
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

# Пример использования
if __name__ == "__main__":
    example_data = {
        'utm_source_type': 'social',
        'utm_medium_type': 'paid',
        'device_category': 'mobile', 
        'os_type': 'Android',
        'time_of_day': 'afternoon',
        'session_hits_count': 15,
        'unique_event_categories': 5,
        'event_category_group': 'submission_click'
    }
    
    result = predict_conversion(example_data)
    print("Результат:", result)
