import pandas as pd
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

# 1. Загрузка данных
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

# 2. Предобработка данных
def preprocess_data(data):
    X = data.drop("target", axis=1).values  # Удаляем столбец 'target', чтобы получить признаки
    y = pd.Categorical(data["target"]).codes  # Преобразуем 'target' в числовые метки
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# 3. Построение модели
class IrisModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(torch.relu(self.layer2(x)))
        x = self.output(x)
        return x

# 4. Обучение модели
def train_model(model, X_train, y_train, criterion, optimizer, epochs=50, patience=5):
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Early Stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Раннее завершение обучения на эпохе {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch + 1}/{epochs}, Потеря: {loss.item()}")

# 5. Оценка модели
def evaluate_model(model, X_test, y_test):
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

# 6. Сохранение метрик
def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

# Основной скрипт
if __name__ == "__main__":
    data = load_data()

    # Проверка данных
    assert not pd.isnull(data).any().any(), "Данные содержат пропуски!"
    print(data.head())

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразуем данные в PyTorch тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Создаём модель
    model = IrisModel(input_dim=X_train.shape[1], output_dim=len(pd.Categorical(data["target"]).categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    train_model(model, X_train, y_train, criterion, optimizer, epochs=50, patience=5)

    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели PyTorch:", metrics)

    # Сохранение метрик
    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
