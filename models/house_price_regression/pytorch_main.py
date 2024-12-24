"""
Скрипт для построения, обучения и оценки модели регрессии на основе PyTorch.
"""

import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Пути
DATA_PATH = "data/house_prices.csv"
RESULTS_PATH = "results/house_price_pytorch_metrics.json"

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
        tuple: Признаки (X), целевая переменная (y), среднее и стандартное отклонение целевой переменной.
    """
    if "Id" in data.columns:
        data = data.drop("Id", axis=1)

    # Заполнение пропусков в числовых столбцах медианой
    for column in data.select_dtypes(include=["number"]).columns:
        data[column] = data[column].fillna(data[column].median())

    # Заполнение пропусков в категориальных столбцах модой
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    # Преобразование категориальных признаков в числовые (one-hot encoding)
    data = pd.get_dummies(data, drop_first=True)

    # Разделение на признаки и целевую переменную
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # Нормализация числовых данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Нормализация целевой переменной
    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / y_std

    return X, y, y_mean, y_std

class HousePriceModel(nn.Module):
    """
    Модель для прогнозирования цен на жильё с использованием PyTorch.

    Архитектура:
    - Три скрытых слоя с функцией активации ReLU.
    - Выходной слой для регрессии.
    """
    def __init__(self, input_dim):
        super(HousePriceModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

def train_model(model, X_train, y_train, criterion, optimizer, epochs=200):
    """
    Обучает модель на тренировочных данных.

    Параметры:
        model (nn.Module): Модель для обучения.
        X_train (torch.Tensor): Признаки тренировочной выборки.
        y_train (torch.Tensor): Целевая переменная тренировочной выборки.
        criterion: Функция потерь.
        optimizer: Оптимизатор.
        epochs (int): Количество эпох.
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Диагностика каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1}/{epochs}, Потери: {loss.item()}")

def evaluate_model(model, X_test, y_test, y_mean, y_std):
    """
    Оценивает производительность модели.

    Параметры:
        model (nn.Module): Обученная модель.
        X_test (torch.Tensor): Признаки тестовой выборки.
        y_test (torch.Tensor): Целевая переменная тестовой выборки.
        y_mean (float): Среднее целевой переменной.
        y_std (float): Стандартное отклонение целевой переменной.

    Возвращает:
        dict: Метрики производительности (MSE, MAE, R2).
    """
    with torch.no_grad():
        model.eval()
        y_pred = model(X_test).squeeze().detach().numpy()
        y_pred = (y_pred * y_std) + y_mean  # Обратная трансформация
        y_test = (y_test.numpy() * y_std) + y_mean  # Обратная трансформация

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
    X, y, y_mean, y_std = preprocess_data(data)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразование данных в PyTorch тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)  # Исправлено
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).view(-1, 1)  # Исправлено

    # Создание модели
    model = HousePriceModel(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Обучение
    train_model(model, X_train, y_train, criterion, optimizer, epochs=200)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test, y_mean, y_std)
    print("Метрики модели PyTorch:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
