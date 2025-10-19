#!/usr/bin/env python3
"""
Скрипт запуска API сервиса предсказания конверсий
"""

import uvicorn
import os
import sys

# Добавляем путь к src для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Основная функция запуска"""
    print("🚀 Запуск GA Conversion Prediction API Service")
    print("=" * 60)
    
    # Проверяем наличие необходимых файлов
    required_files = [
        'models/conversion_model.pkl',
        'models/label_encoders.pkl',
        'models/feature_info.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ ОТСУТСТВУЮТ ФАЙЛЫ МОДЕЛИ:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 РЕШЕНИЕ: Запустите ноутбуки в правильном порядке:")
        print("   1. 01_ga_data_analysis.ipynb")
        print("   2. 02_feature_engineering.ipynb") 
        print("   3. 03_model_training.ipynb")
        print("   4. 04_model_evaluation.ipynb")
        return
    
    print("✅ Все файлы модели присутствуют")
    print("🌐 ЗАПУСК API СЕРВЕРА...")
    print("=" * 60)
    
    # Запускаем сервер
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()