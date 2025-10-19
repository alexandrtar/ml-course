# create_sample_data.py
import pandas as pd
import numpy as np
import os

print("📊 Создание sample данных для тестирования...")

# Создаем папку если нет
os.makedirs('models', exist_ok=True)

# Создаем sample данных для быстрого тестирования
np.random.seed(42)

# Генерируем sample данных
n_samples = 1000

sample_data = {
    'utm_source_type': np.random.choice(['social', 'direct', 'organic', 'other'], n_samples),
    'utm_medium': np.random.choice(['cpc', 'organic', 'referral'], n_samples),
    'device_category': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
    'os_type': np.random.choice(['Android', 'iOS', 'Windows', 'Mac'], n_samples),
    'device_browser': np.random.choice(['Chrome', 'Safari', 'Firefox'], n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'hour_of_day': np.random.randint(0, 24, n_samples),
    'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
    'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, n_samples) / 24),
    'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
    'is_weekend': np.random.randint(0, 2, n_samples),
    'month': np.random.randint(1, 13, n_samples),
    'is_peak_hours': np.random.randint(0, 2, n_samples),
    'session_hits_count': np.random.randint(1, 50, n_samples),
    'unique_event_categories': np.random.randint(1, 10, n_samples),
    'unique_event_actions': np.random.randint(1, 15, n_samples),
    'unique_pages': np.random.randint(1, 20, n_samples),
    'is_returning_user': np.random.randint(0, 2, n_samples),
    'event_category_group': np.random.choice(['engagement_click', 'page_view', 'submission_click'], n_samples),
    'country_region': np.random.choice(['cis', 'europe', 'other'], n_samples),
    'target': np.random.randint(0, 2, n_samples)
}

# Создаем DataFrame
df = pd.DataFrame(sample_data)

# Сохраняем sample данные
df.to_csv('model_data_sample.csv', index=False)
print(f"✅ Sample данные созданы: {df.shape}")
print(f"   Conversion rate: {df['target'].mean():.4f}")

print("📊 Статистика данных:")
print(df.describe())