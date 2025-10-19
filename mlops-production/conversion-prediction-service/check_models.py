import os
import joblib

def check_models():
    print("🔍 ДИАГНОСТИКА МОДЕЛИ")
    print("=" * 50)
    
    model_files = {
        'conversion_model.pkl': 'models/conversion_model.pkl',
        'label_encoders.pkl': 'models/label_encoders.pkl',
        'feature_info.pkl': 'models/feature_info.pkl'
    }
    
    # Check file existence
    for name, path in model_files.items():
        exists = os.path.exists(path)
        status = "✅ найден" if exists else "❌ не найден"
        print(f"{name}: {status}")
        
        if exists:
            try:
                data = joblib.load(path)
                size = len(str(data))
                print(f"     Размер: ~{size} байт")
                
                if name == 'feature_info.pkl':
                    if 'feature_columns' in data:
                        print(f"     Фичи: {len(data['feature_columns'])}")
                    if 'optimal_threshold' in data:
                        print(f"     Порог: {data['optimal_threshold']}")
                        
            except Exception as e:
                print(f"     ❌ Ошибка загрузки: {e}")
    
    print("=" * 50)
    
    # Check if all files exist
    all_exist = all(os.path.exists(p) for p in model_files.values())
    if all_exist:
        print("🎉 Все файлы модели присутствуют и валидны!")
    else:
        print("❌ Некоторые файлы модели отсутствуют!")
        print("💡 Решение: Запустите ноутбук 03_model_training.ipynb для обучения модели")

if __name__ == "__main__":
    check_models()