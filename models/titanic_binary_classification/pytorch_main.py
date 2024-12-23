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
    data = pd.read_csv(DATA_PATH)
    return data


# 2. Предобработка данных
def preprocess_data(data):
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
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()


# 5. Оценка модели
def evaluate_model(model, X_test, y_test):
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


# Основной скрипт
if __name__ == "__main__":
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    model = TitanicModel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, X_train, y_train, criterion, optimizer, epochs=50)
    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели PyTorch:", metrics)

    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
