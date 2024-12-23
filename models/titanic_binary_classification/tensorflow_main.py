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
def build_model(input_dim):
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


# Основной скрипт
if __name__ == "__main__":
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели TensorFlow:", metrics)

    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
