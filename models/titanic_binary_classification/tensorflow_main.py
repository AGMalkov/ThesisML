"""
Скрипт для классификации Titanic с использованием TensorFlow.
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Пути
DATA_PATH = "data/titanic.csv"
RESULTS_PATH = "results/titanic_tensorflow_metrics.json"


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
        tuple: Признаки (X) и целевая переменная (y).
    """
    # Удаление ненужных столбцов
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    # Заполнение пропусков
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    # Преобразование категориальных данных
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# 3. Построение модели
def build_model(input_dim):
    """
    Создает модель бинарной классификации с использованием TensorFlow.

    Параметры:
        input_dim (int): Размерность входных данных.

    Возвращает:
        tf.keras.Model: Скомпилированная модель.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),  # Явно указываем форму входных данных
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 4. Оценка модели
def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (tf.keras.Model): Обученная модель.
        X_test (np.ndarray): Признаки тестовой выборки.
        y_test (np.ndarray): Целевая переменная тестовой выборки.

    Возвращает:
        dict: Метрики производительности (Accuracy, Precision, Recall, F1-Score).
    """
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


# Основной скрипт
if __name__ == "__main__":
    """
    Основной блок выполнения:
    1. Загрузка данных.
    2. Предобработка данных.
    3. Построение и обучение модели.
    4. Оценка модели.
    5. Сохранение метрик.
    """
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Построение модели
    model = build_model(input_dim=X_train.shape[1])

    # Обучение модели
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели TensorFlow:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
