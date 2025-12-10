#!/usr/bin/env python3
"""
Исправленный запуск системы
"""

import os
import sys
import subprocess
import time
import socket
import requests
from pathlib import Path

# Устанавливаем переменные окружения для решения проблемы protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def is_port_in_use(port, host='localhost'):
    """Проверка, используется ли порт"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False
        except Exception:
            return False

def start_api():
    """Запуск API сервера"""
    print("\n[1/3] Запуск API сервера...")
    
    # Проверяем порт
    if is_port_in_use(8000):
        print("   [WARN] Порт 8000 уже занят. Проверяем API...")
        try:
            response = requests.get("http://localhost:8000/health", timeout=3)
            if response.status_code == 200:
                print("   [OK] API уже запущен")
                return None, True
        except:
            print("   [ERROR] Порт занят, но API не отвечает")
            return None, False
    
    # Запускаем API
    api_process = subprocess.Popen(
        [sys.executable, "start_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Ждем запуска
    print("   [WAIT] Ожидание запуска API (15 секунд)...")
    for i in range(15):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print(f"   [OK] API запущен через {i+1} секунд")
                return api_process, True
        except:
            pass
        
        time.sleep(1)
    
    # Если не запустился, показываем ошибки
    print("   [ERROR] API не запустился за 15 секунд")
    try:
        stdout, stderr = api_process.communicate(timeout=3)
        if stderr:
            print("   [LOG] Ошибки API:")
            for line in stderr.split('\n')[-10:]:
                if line.strip():
                    print(f"      {line}")
    except:
        pass
    
    api_process.terminate()
    return api_process, False

def start_web_interface():
    """Запуск веб-интерфейса"""
    print("\n[2/3] Запуск веб-интерфейса...")
    
    # Проверяем порт
    if is_port_in_use(8501):
        print("   [WARN] Порт 8501 уже занят")
        print("   [INFO] Проверьте http://localhost:8501")
        return None, True
    
    # Создаем конфигурацию Streamlit
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    config = streamlit_dir / "config.toml"
    if not config.exists():
        config.write_text("""
[server]
port = 8501
address = "localhost"
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "localhost"

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#31333F"
font = "sans serif"
""", encoding='utf-8')
    
    # Запускаем Streamlit
    web_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "web_interface.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Ждем запуска
    print("   [WAIT] Ожидание запуска Streamlit (10 секунд)...")
    time.sleep(10)
    
    # Проверяем запуск
    try:
        response = requests.get("http://localhost:8501", timeout=3)
        if response.status_code in [200, 403]:
            print("   [OK] Веб-интерфейс запущен")
            return web_process, True
    except:
        pass
    
    print("   [INFO] Streamlit запускается...")
    print("   [URL] Откройте http://localhost:8501 в браузере")
    return web_process, True

def monitor_processes(api_process, web_process):
    """Мониторинг процессов"""
    print("\n[3/3] Система запущена!")
    print("=" * 60)
    print("🌐 Веб-интерфейс: http://localhost:8501")
    print("📚 Документация API: http://localhost:8000/docs")
    print("\n🛑 Для остановки нажмите Ctrl+C")
    print("=" * 60)
    
    try:
        # Ждем завершения или прерывания
        while True:
            # Проверяем, живы ли процессы
            if api_process and api_process.poll() is not None:
                print("\n[ERROR] API сервер завершился")
                break
            
            if web_process and web_process.poll() is not None:
                print("\n[ERROR] Веб-интерфейс завершился")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[STOP] Остановка системы...")
    except Exception as e:
        print(f"\n[ERROR] {e}")

def main():
    print("=" * 60)
    print("Запуск системы генерации лиц: GAN vs VAE")
    print("=" * 60)
    
    processes = []
    
    try:
        # Запускаем API
        api_process, api_ok = start_api()
        if api_process:
            processes.append(api_process)
        
        if not api_ok:
            print("\n[ERROR] Не удалось запустить API")
            print("Запустите вручную: python start_api.py")
            if api_process:
                api_process.terminate()
            return
        
        # Ждем 2 секунды перед запуском веб-интерфейса
        time.sleep(2)
        
        # Запускаем веб-интерфейс
        web_process, web_ok = start_web_interface()
        if web_process:
            processes.append(web_process)
        
        # Мониторим процессы
        monitor_processes(api_process, web_process)
        
    except KeyboardInterrupt:
        print("\n[STOP] Прервано пользователем")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        # Останавливаем все процессы
        print("\n[STOP] Остановка процессов...")
        for process in processes:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        print("[OK] Все процессы остановлены")

if __name__ == "__main__":
    main()