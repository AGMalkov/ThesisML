import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

# Пути
DATA_PATH = "data/house_prices.csv"
RESULTS_PATH = "results/house_price_tensorflow_metrics.json"

# 1. Загрузка данных
def load_data():
    data = pd.read_csv(DATA_PATH)
    return data

# 2. Предобработка данных
def preprocess_data(data):
    # Удаление ненужного столбца Id
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

    return X, y

# 3. Построение модели
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Линейная активация для регрессии
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 4. Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
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
