"""
Скрипт для построения, обучения и оценки модели регрессии на основе TensorFlow.
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Пути
DATA_PATH = "data/house_prices.csv"
RESULTS_PATH = "results/house_price_tensorflow_metrics.json"

def load_data():
    """
    Загружает данные из файла CSV.

    Возвращает:
        pd.DataFrame: Исходные данные.
    """
    data = pd.read_csv(DATA_PATH)
    return data

def preprocess_data(data):
    """
    Предобрабатывает данные для модели.

    Параметры:
        data (pd.DataFrame): Исходные данные.

    Возвращает:
        tuple: Признаки (X) и целевая переменная (y).
    """
    if "Id" in data.columns:
        data = data.drop("Id", axis=1)
    # Обработка пропусков
    for column in data.select_dtypes(include=["number"]).columns:
        data[column] = data[column].fillna(data[column].median())
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    # Преобразование категориальных данных
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    # Нормализация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def build_model(input_dim):
    """
    Создает модель регрессии на основе TensorFlow.

    Параметры:
        input_dim (int): Размерность входных данных.

    Возвращает:
        tf.keras.Model: Скомпилированная модель TensorFlow.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Выходной слой для регрессии
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (tf.keras.Model): Обученная модель.
        X_test (np.ndarray): Признаки тестовой выборки.
        y_test (np.ndarray): Целевая переменная тестовой выборки.

    Возвращает:
        dict: Метрики производительности (MSE, MAE, R²).
    """
    y_pred = model.predict(X_test).squeeze()
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }
    return metrics

def save_metrics(metrics, path):
    """
    Сохраняет метрики в JSON-файл.

    Параметры:
        metrics (dict): Метрики производительности.
        path (str): Путь для сохранения.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    """
    Основной скрипт для выполнения предобработки данных, обучения и оценки модели.
    """
    # Загрузка данных
    data = load_data()

    # Предобработка данных
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание модели
    model = build_model(input_dim=X_train.shape[1])

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели TensorFlow:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
