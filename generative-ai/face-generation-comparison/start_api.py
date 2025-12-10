#!/usr/bin/env python3
"""
Альтернативный скрипт для запуска API без проблем с protobuf
"""

import os
import sys
import uvicorn
from pathlib import Path

# Устанавливаем переменные окружения для решения проблемы protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключаем логи TensorFlow

# Добавляем путь к проекту
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def start_api():
    """Запуск API сервера"""
    print(">>> Запуск API сервера...")
    print("[INFO] Установлена переменная окружения: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")
    
    # Импортируем настройки после установки переменных окружения
    from app.config.settings import settings
    
    print(f"[NETWORK] API будет доступен по адресу: http://{settings.API_HOST}:{settings.API_PORT}")
    print("[DOCS] Документация: http://localhost:8000/docs")
    print("[CTRL+C] Для остановки нажмите Ctrl+C")
    
    # Запускаем сервер
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        start_api()
    except KeyboardInterrupt:
        print("\n[STOP] Остановка API сервера...")
    except Exception as e:
        print(f"\n[ERROR] Ошибка запуска API: {e}")
        print("\n[TROUBLESHOOTING] Возможные решения:")
        print("1. Попробуйте обновить protobuf: pip install --upgrade protobuf")
        print("2. Или установить старую версию: pip install protobuf==3.20.0")
        print("3. Или запустить без переменной окружения")