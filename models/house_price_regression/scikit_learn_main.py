"""
Скрипт для построения, обучения и оценки модели регрессии на основе Scikit-learn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Пути
DATA_PATH = "data/house_prices.csv"
RESULTS_PATH = "results/house_price_sklearn_metrics.json"

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

    # Заполнение пропусков для числовых и категориальных признаков
    for column in data.select_dtypes(include=["number"]).columns:
        data[column] = data[column].fillna(data[column].median())  # Среднее для числовых
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].fillna(data[column].mode()[0])  # Мода для категориальных

    # Преобразование категориальных данных
    data = pd.get_dummies(data, drop_first=True)  # One-hot encoding

    # Разделение на признаки и целевую переменную
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]
    return X, y

def train_model(X_train, y_train):
    """
    Обучает линейную регрессию на тренировочных данных.

    Параметры:
        X_train (pd.DataFrame): Признаки тренировочной выборки.
        y_train (pd.Series): Целевая переменная тренировочной выборки.

    Возвращает:
        LinearRegression: Обученная модель.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (LinearRegression): Обученная модель.
        X_test (pd.DataFrame): Признаки тестовой выборки.
        y_test (pd.Series): Целевая переменная тестовой выборки.

    Возвращает:
        dict: Метрики производительности (MSE, MAE, R2).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred),
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
    print("Первые строки данных:")
    print(data.head())

    # Предобработка данных
    X, y = preprocess_data(data)
    print("Размеры данных после предобработки:")
    print("X:", X.shape, "y:", y.shape)

    # Разделение данных на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = train_model(X_train, y_train)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
