"""
Скрипт для классификации Iris с использованием PyTorch.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Пути
DATA_PATH = "data/iris.csv"
RESULTS_PATH = "results/iris_pytorch_metrics.json"

def load_data():
    """
    Загружает данные из CSV-файла.

    Возвращает:
        pd.DataFrame: Исходные данные.
    """
    data = pd.read_csv(DATA_PATH)
    return data

def preprocess_data(data):
    """
    Предобрабатывает данные для классификации.

    Параметры:
        data (pd.DataFrame): Исходные данные.

    Возвращает:
        tuple: Признаки (X) и метки классов (y).
    """
    X = data.drop("target", axis=1).values
    y = pd.Categorical(data["target"]).codes
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

class IrisModel(nn.Module):
    """
    Модель для классификации Iris с использованием PyTorch.

    Архитектура:
    - Два полносвязных слоя с функцией активации ReLU.
    - Выходной слой для классификации (количество нейронов соответствует числу классов).
    """
    def __init__(self, input_dim, output_dim):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

def train_model(model, X_train, y_train, criterion, optimizer, epochs=50):
    """
    Обучает модель с использованием PyTorch.

    Параметры:
        model (nn.Module): Модель для обучения.
        X_train (torch.Tensor): Признаки тренировочной выборки.
        y_train (torch.Tensor): Метки классов тренировочной выборки.
        criterion: Функция потерь.
        optimizer: Оптимизатор.
        epochs (int): Количество эпох обучения.
    """
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1}/{epochs}, Потеря: {loss.item()}")

def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (nn.Module): Обученная модель.
        X_test (torch.Tensor): Признаки тестовой выборки.
        y_test (torch.Tensor): Метки классов тестовой выборки.

    Возвращает:
        dict: Метрики производительности (Accuracy, Precision, Recall, F1-Score).
    """
    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        y_pred = torch.argmax(outputs, dim=1).numpy()
        y_test = y_test.numpy()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted")
    }
    return metrics

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

if __name__ == "__main__":
    """
    Основной скрипт для выполнения предобработки данных, обучения и оценки модели.
    """
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразуем данные в тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Создаем модель
    model = IrisModel(input_dim=X_train.shape[1], output_dim=len(pd.Categorical(data["target"]).categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, X_train, y_train, criterion, optimizer, epochs=50)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели PyTorch:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
