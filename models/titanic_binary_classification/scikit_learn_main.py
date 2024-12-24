"""
Скрипт для классификации Titanic с использованием Scikit-learn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Путь к данным
DATA_PATH = "data/titanic.csv"
RESULTS_PATH = "results/titanic_sklearn_metrics.json"


# 1. Загрузка данных
def load_data():
    """
    Загружает данные из CSV-файла.

    Возвращает:
        pd.DataFrame: Исходные данные.
    """
    data = pd.read_csv(DATA_PATH)
    return data


# 2. Предобработка данных
def preprocess_data(data):
    """
    Предобрабатывает данные для классификации.

    Параметры:
        data (pd.DataFrame): Исходные данные.

    Возвращает:
        pd.DataFrame: Предобработанные данные.
    """
    # Удаление ненужных столбцов
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    # Заполнение пропусков
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    # Преобразование категориальных данных
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    return data


# 3. Обучение модели
def train_model(X_train, y_train):
    """
    Обучает модель логистической регрессии.

    Параметры:
        X_train (pd.DataFrame): Признаки тренировочной выборки.
        y_train (pd.Series): Целевая переменная тренировочной выборки.

    Возвращает:
        LogisticRegression: Обученная модель.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# 4. Оценка модели
def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (LogisticRegression): Обученная модель.
        X_test (pd.DataFrame): Признаки тестовой выборки.
        y_test (pd.Series): Целевая переменная тестовой выборки.

    Возвращает:
        dict: Метрики производительности (Accuracy, Precision, Recall, F1-Score).
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    return metrics


# 5. Сохранение метрик
def save_metrics(metrics, path):
    """
    Сохраняет метрики в JSON-файл.

    Параметры:
        metrics (dict): Метрики производительности.
        path (str): Путь для сохранения файла.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


# Основной скрипт
if __name__ == "__main__":
    """
    Основной блок выполнения:
    1. Загрузка данных.
    2. Предобработка данных.
    3. Обучение модели.
    4. Оценка модели.
    5. Сохранение метрик.
    """
    data = load_data()
    data = preprocess_data(data)

    # Разделение данных на признаки и целевую переменную
    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    # Разделение на тренировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = train_model(X_train, y_train)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели:", metrics)

    # Сохранение метрик
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
