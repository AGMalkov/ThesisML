"""
Скрипт для классификации Titanic с использованием PyTorch.
"""

import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Пути
DATA_PATH = "data/titanic.csv"
RESULTS_PATH = "results/titanic_pytorch_metrics.json"


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
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# 3. Построение модели
class TitanicModel(nn.Module):
    """
    Модель для бинарной классификации Titanic с использованием PyTorch.

    Архитектура:
    - Два полносвязных слоя с функцией активации ReLU.
    - Выходной слой с функцией активации Sigmoid для бинарной классификации.
    """
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


# 4. Обучение модели
def train_model(model, X_train, y_train, criterion, optimizer, epochs=50):
    """
    Обучает модель с использованием PyTorch.

    Параметры:
        model (nn.Module): Модель для обучения.
        X_train (torch.Tensor): Признаки тренировочной выборки.
        y_train (torch.Tensor): Целевая переменная тренировочной выборки.
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


# 5. Оценка модели
def evaluate_model(model, X_test, y_test):
    """
    Оценивает производительность модели.

    Параметры:
        model (nn.Module): Обученная модель.
        X_test (torch.Tensor): Признаки тестовой выборки.
        y_test (torch.Tensor): Целевая переменная тестовой выборки.

    Возвращает:
        dict: Метрики производительности (Accuracy, Precision, Recall, F1-Score).
    """
    with torch.no_grad():
        y_pred = model(X_test).round()  # Предсказания в виде тензоров
        y_pred = y_pred.numpy()  # Преобразуем в NumPy массив
        y_test = y_test.numpy()  # Преобразуем в NumPy массив
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics


# 6. Сохранение метрик
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

    # Преобразуем данные в тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Создаем модель
    model = TitanicModel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, X_train, y_train, criterion, optimizer, epochs=50)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели PyTorch:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
