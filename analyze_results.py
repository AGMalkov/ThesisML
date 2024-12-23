import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Пути
RESULTS_DIR = "results/"
OUTPUT_FILE = "results_summary.csv"

# Сбор метрик из файлов
def collect_metrics(results_dir):
    all_metrics = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            task, library = filename.replace("_metrics.json", "").split("_", 1)
            with open(os.path.join(results_dir, filename), "r") as f:
                metrics = json.load(f)
                # Добавляем задачу и библиотеку
                metrics["task"] = task
                metrics["library"] = library
                # Убедимся, что метрики корректно читаются
                metrics.setdefault("mean_squared_error", 0)
                metrics.setdefault("mean_absolute_error", 0)
                metrics.setdefault("r2_score", 0)
                all_metrics.append(metrics)
    df = pd.DataFrame(all_metrics)
    print("Собранные метрики:\n", df)  # Печатает собранные метрики
    return df

# Нормализация данных (Min-Max Scaling)
def normalize_data(df):
    # Применяем Min-Max нормализацию (переводим все значения в диапазон от 0 до 1)
    return (df - df.min()) / (df.max() - df.min())

# Визуализация метрик
def visualize_metrics(df):
    # Определяем метрики для каждой задачи
    task_metrics = {
        "titanic": ["accuracy", "precision", "recall", "f1_score"],
        "iris": ["accuracy", "precision", "recall", "f1_score"],
        "house": ["mean_squared_error", "mean_absolute_error", "r2_score"],  # Исправление здесь
    }

    for task in df["task"].unique():
        print(f"\n=== Задача: {task} ===")
        task_data = df[df["task"] == task]
        print("Данные перед фильтрацией:")
        print(task_data)

        task_data.set_index("library", inplace=True)
        print("Колонки перед фильтрацией:", task_data.columns.tolist())

        # Диагностика соответствия метрик
        expected_columns = task_metrics.get(task, [])
        print("Ожидаемые метрики:", expected_columns)
        print("Несовпадающие метрики:", [col for col in expected_columns if col not in task_data.columns])

        # Фильтруем колонки
        relevant_columns = [
            col.strip() for col in task_data.columns if col.strip() in expected_columns
        ]
        print("Реальные пересечения:", relevant_columns)

        task_data = task_data[relevant_columns]
        print("После фильтрации колонок:")
        print(task_data)

        # Убираем строки с NaN
        task_data = task_data.dropna(how="all")
        print("После удаления строк с NaN:")
        print(task_data)

        # Убираем строки с нулями
        task_data = task_data.loc[(task_data != 0).any(axis=1)]
        print("После удаления строк с нулями:")
        print(task_data)

        if task_data.empty:
            print(f"Для задачи {task} нет данных для построения графика.")
            continue

        # Нормализуем только для задачи "house"
        if task == "house":
            task_data = normalize_data(task_data)
            print("Данные после нормализации:")
            print(task_data)

        # Построение графика
        task_data.plot(kind="bar", figsize=(10, 6), title=f"Метрики для {task}")
        plt.ylabel("Значение метрики" if task != "house" else "Нормализованное значение")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"results/{task}_metrics_comparison.png")
        plt.close()
        print(f"График для {task} сохранён.")

# Основной скрипт
if __name__ == "__main__":
    # Сбор данных
    metrics_df = collect_metrics(RESULTS_DIR)

    # Сохранение сводной таблицы
    metrics_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Сводная таблица сохранена в {OUTPUT_FILE}")

    # Визуализация метрик
    visualize_metrics(metrics_df)
    print("Графики метрик сохранены в папке results/")
