import os
import json
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Путь к результатам
RESULTS_DIR = "results/"

# Карта файлов JSON для каждой задачи
RESULTS_FILES = {
    "titanic": [
        "titanic_sklearn_metrics.json",
        "titanic_tensorflow_metrics.json",
        "titanic_pytorch_metrics.json",
    ],
    "iris": [
        "iris_sklearn_metrics.json",
        "iris_tensorflow_metrics.json",
        "iris_pytorch_metrics.json",
    ],
    "house": [
        "house_price_sklearn_metrics.json",
        "house_price_tensorflow_metrics.json",
        "house_price_pytorch_metrics.json",
    ],
}

# Загрузка метрик из JSON
def load_all_metrics(task_name):
    """
    Загружает метрики для всех библиотек по указанной задаче.

    Параметры:
        task_name (str): Название задачи (titanic, iris, house).

    Возвращает:
        str: Форматированный текст с метриками для всех библиотек.
    """
    if task_name not in RESULTS_FILES:
        return f"Задача '{task_name}' не поддерживается."

    response = f"Результаты для задачи '{task_name}':\n\n"
    for file_name in RESULTS_FILES[task_name]:
        file_path = os.path.join(RESULTS_DIR, file_name)
        if not os.path.exists(file_path):
            response += f"Файл {file_name} не найден.\n"
            continue

        with open(file_path, "r") as file:
            metrics = json.load(file)
            library_name = file_name.split("_")[1]  # Извлекаем название библиотеки из имени файла
            response += f"Библиотека: {library_name.capitalize()}\n"
            for key, value in metrics.items():
                response += f"  {key}: {value}\n"
            response += "\n"

    return response

# Команда для отправки метрик и графика
async def send_results(update: Update, context: ContextTypes.DEFAULT_TYPE, task_name: str):
    """
    Отправляет метрики и график для указанной задачи.

    Параметры:
        update (Update): Объект обновления Telegram.
        context (ContextTypes.DEFAULT_TYPE): Контекст команды.
        task_name (str): Название задачи.
    """
    # Отправка метрик
    response = load_all_metrics(task_name)
    await update.message.reply_text(response)

    # Путь к графику
    graph_path = os.path.join(RESULTS_DIR, f"{task_name}_metrics_comparison.png")

    # Отправка графика
    if os.path.exists(graph_path):
        await update.message.reply_photo(photo=open(graph_path, "rb"))
    else:
        await update.message.reply_text(f"График для задачи '{task_name}' не найден.")

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает команду /start.

    Параметры:
        update (Update): Объект обновления Telegram.
        context (ContextTypes.DEFAULT_TYPE): Контекст команды.
    """
    welcome_message = (
        "Привет! Я Телеграм-бот для анализа метрик моделей машинного обучения.\n\n"
        "Команды:\n"
        "/results_titanic — Метрики и график для задачи Titanic\n"
        "/results_iris — Метрики и график для задачи Iris\n"
        "/results_house — Метрики и график для задачи House Prices"
    )
    await update.message.reply_text(welcome_message)

# Обработчики команд для задач
async def results_titanic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_results(update, context, "titanic")

async def results_iris(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_results(update, context, "iris")

async def results_house(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_results(update, context, "house")

# Основной блок
if __name__ == "__main__":
    BOT_TOKEN = "<YOUR_BOT_TOKEN>"

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("results_titanic", results_titanic))
    app.add_handler(CommandHandler("results_iris", results_iris))
    app.add_handler(CommandHandler("results_house", results_house))

    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling()
