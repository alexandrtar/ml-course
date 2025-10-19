# train_simple_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

print("🎯 Обучение простой модели для демонстрации...")

# Загружаем sample данные
try:
    data = pd.read_csv('model_data_sample.csv')
    print(f"✅ Данные загружены: {data.shape}")
except FileNotFoundError:
    print("❌ Файл данных не найден. Сначала запустите create_sample_data.py")
    exit()

# Подготовка данных
X = data.drop('target', axis=1)
y = data['target']

# Кодируем категориальные переменные
label_encoders = {}
X_encoded = X.copy()

categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X_encoded[col + '_encoded'] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le
    print(f"  Закодирована колонка: {col}")

# Удаляем оригинальные категориальные колонки
X_final = X_encoded.drop(categorical_columns, axis=1)

print(f"Финальные фичи: {X_final.columns.tolist()}")

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Обучаем модель
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

print("🔄 Обучение модели...")
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()

print("📊 Результаты модели:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Сохраняем модель и артефакты
os.makedirs('models', exist_ok=True)

# Сохраняем модель
joblib.dump(model, 'models/conversion_model.pkl')
print("✅ Модель сохранена: models/conversion_model.pkl")

# Сохраняем label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✅ Label encoders сохранены: models/label_encoders.pkl")

# Сохраняем информацию о фичах
feature_info = {
    'feature_columns': X_final.columns.tolist(),
    'categorical_columns': categorical_columns.tolist(),
    'numerical_columns': X_final.select_dtypes(include=[np.number]).columns.tolist(),
    'optimal_threshold': 0.5
}

joblib.dump(feature_info, 'models/feature_info.pkl')
print("✅ Feature info сохранена: models/feature_info.pkl")

print("\n🎉 МОДЕЛЬ УСПЕШНО СОЗДАНА!")
print("Теперь можно запускать API сервис:")
print("python run_api.py")