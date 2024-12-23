import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import numpy as np  # Импортируем numpy для подсчета уникальных классов

# Пути
DATA_PATH = "data/iris.csv"
RESULTS_PATH = "results/iris_tensorflow_metrics.json"


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

    # Преобразуем метки в формат one-hot
    y = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))  # Преобразование в one-hot

    return X, y


# 3. Построение модели
def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 4. Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
    y_test_classes = tf.argmax(y_test, axis=1).numpy()
    metrics = {
        "accuracy": accuracy_score(y_test_classes, y_pred_classes),
        "precision": precision_score(y_test_classes, y_pred_classes, average="weighted"),
        "recall": recall_score(y_test_classes, y_pred_classes, average="weighted"),
        "f1_score": f1_score(y_test_classes, y_pred_classes, average="weighted")
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

    # Получаем количество классов для output_dim
    output_dim = y_train.shape[1]  # Количество классов теперь определяется из y_train в формате one-hot

    model = build_model(input_dim=X_train.shape[1], output_dim=output_dim)
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    metrics = evaluate_model(model, X_test, y_test)
    print("Метрики модели TensorFlow:", metrics)

    save_metrics(metrics, RESULTS_PATH)
    print(f"Метрики сохранены в {RESULTS_PATH}")
