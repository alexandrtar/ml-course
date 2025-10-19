# 🎯 GA Conversion Prediction Service

Сервис предсказания конверсий на основе данных Google Analytics для "СберАвтоподписка".

## 📊 Business Value

- **Прогнозирование** вероятности конверсии пользователя в реальном времени
- **Оптимизация** маркетинговых бюджетов через focus на high-value сессии
- **Увеличение** ROI рекламных кампаний
- **Персонализация** пользовательского опыта

## 🏗️ Архитектура решения
Google Analytics Data → Feature Engineering → RandomForest Model → FastAPI Service

## 📈 Результаты модели

- **ROC-AUC**: 1.0
- **F1-Score**: 0.996  
- **Precision**: 0.993
- **Recall**: 0.999

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements_simple.txt
```
### 2. Подготовка модели
Запустите ноутбуки в порядке:

```bash
01_ga_data_analysis.ipynb    # Анализ данных
02_feature_engineering.ipynb # Feature engineering  
03_model_training.ipynb      # Обучение модели
04_model_evaluation.ipynb    # Оценка модели
```
3. Запуск API
```bash
python run_api.py
```
4. Тестирование
```bash
python -m pytest tests/test_simple.py -v
```
📡 API Endpoints
Основные эндпоинты:

GET / - Главная страница

GET /health - Статус сервиса

GET /model/info - Информация о модели

GET /categories - Доступные категории фичей

GET /example - Пример запроса

POST /predict - Предсказание конверсии

Пример запроса:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
  ```
🎯 Использование в продакшене

Интеграция с веб-приложением:

javascript

Пример JavaScript вызова
```bash
async function predictConversion(sessionData) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(sessionData)
    });
    const result = await response.json();
    if (result.conversion_prediction === 1) {
        // Показать персонализированное предложение
        showPersonalizedOffer();
    }
    return result.conversion_probability;
}
```
Мониторинг:
```bash
# Проверка здоровья сервиса
curl http://localhost:8000/health

# Информация о модели
curl http://localhost:8000/model/info
```
📊 Ключевые фичи

UTM параметры:

utm_source_type - Тип источника (social, direct, organic, other)

utm_medium - Тип канала (cpc, organic, referral)

Device характеристики:

device_category - Тип устройства (mobile, desktop, tablet)

os_type - Операционная система

device_browser - Браузер

Временные фичи:

day_of_week - День недели (0-6)

hour_of_day - Час дня (0-23)

time_of_day - Время дня (morning, afternoon, evening, night)

is_weekend - Выходной день

is_peak_hours - Пиковые часы

Поведенческие фичи:

session_hits_count - Количество хитов в сессии

unique_event_categories - Уникальные категории событий

unique_event_actions - Уникальные действия

unique_pages - Уникальные страницы

is_returning_user - Возвращающийся пользователь

Гео фичи:

country_region - Регион страны (cis, europe, other)

🔧 Технические детали

Модель:

Алгоритм: RandomForestClassifier

Количество деревьев: 100

Глубина: 15

Балансировка классов: class_weight='balanced'

Производительность:

Время предсказания: < 100ms

Поддержка batch запросов: Да

Память: ~500MB

🛠️ Разработка

Структура проекта:

```bash
conversion-prediction-service/
├── 📁 models/           # Артефакты модели
├── 📁 notebooks/        # Jupyter ноутбуки анализа
├── 📁 src/             # Исходный код
│   └── 📁 api/         # FastAPI приложение
├── 📁 tests/           # Тесты
├── run_api.py          # Скрипт запуска
└── requirements_simple.txt
```
Добавление новых фич:

Обновите feature engineering в 02_feature_engineering.ipynb

Переобучите модель в 03_model_training.ipynb

Обновите Pydantic схемы в src/api/main.py

📝 Лицензия

MIT License

👥 Команда

Data Science Team - СберАвтоподписка

## 🚀 Запуск проекта

1. **Активируйте виртуальное окружение:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Запустите ноутбуки для подготовки модели:
```bash
jupyter notebook notebooks/01_ga_data_analysis.ipynb
# затем последовательно 02, 03, 04
```
4. Запустите API:
```bash
python run_api.py
```
5. Протестируйте:
```bash
python -m pytest tests/test_simple.py -v
```
