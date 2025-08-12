import os
import re
import requests
import tempfile
import subprocess
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import torch
import whisper

# ========= НАСТРОЙКИ =========
BOT_TOKEN = os.getenv("BOT_TOKEN")
ASR_MODEL = "small"        # Модель Whisper
LANGUAGE = "ru"            # Язык распознавания
COMPUTE_TYPE = "int8"      # int8 для экономии памяти

# ======== ЗАГРУЗКА МОДЕЛИ ========
print("Загружаю модель Whisper...")
model = whisper.load_model(ASR_MODEL, device="cpu")
print("✅ Модель загружена")

# ======== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========

def extract_gdrive_id(url):
    """Извлекает ID файла Google Drive из ссылки"""
    match = re.search(r"/d/([a-zA-Z0-9_-]{10,})", url)
    if match:
        return match.group(1)
    return None

def download_file(url):
    """Скачивает файл во временную папку и возвращает путь"""
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, "input_file")

    if "drive.google.com" in url:
        file_id = extract_gdrive_id(url)
        if not file_id:
            raise ValueError("Не удалось извлечь ID файла Google Drive. Убедитесь, что ссылка вида: https://drive.google.com/file/d/FILE_ID/view?usp=sharing")
        gdrive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(gdrive_url, stream=True)
        if r.status_code != 200:
            raise ValueError(f"Ошибка скачивания с Google Drive: {r.status_code}")
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path

    elif "zoom.us" in url:
        raise ValueError("Zoom поддерживается только с прямыми ссылками на скачивание и открытым доступом.")

    else:
        # Прямая ссылка
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise ValueError(f"Ошибка скачивания: {r.status_code}")
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path

async def transcribe_file(file_path):
    """Распознаёт речь из файла"""
    # Конвертация в wav для стабильной работы
    wav_path = file_path + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("▶ Начинаю распознавание...")
    result = model.transcribe(wav_path, language=LANGUAGE)
    text = result["text"].strip()
    print("✅ Распознавание завершено")
    return text

# ======== ОБРАБОТЧИК СООБЩЕНИЙ ========

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        message = update.message.text.strip()
        await update.message.reply_text("🌐 Скачиваю файл...")

        file_path = download_file(message)

        if os.path.getsize(file_path) < 100 * 1024:  # меньше 100 КБ
            await update.message.reply_text("❌ Файл слишком маленький. Нужно минимум 100 КБ.")
            return

        await update.message.reply_text("🎙 Распознаю речь, подождите...")
        text = await transcribe_file(file_path)

        if text:
            await update.message.reply_text("✅ Распознавание завершено:\n\n" + text)
        else:
            await update.message.reply_text("❌ Не удалось распознать речь. Попробуйте другой файл или источник.")

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

# ======== ЗАПУСК БОТА ========
if __name__ == "__main__":
    if not BOT_TOKEN:
        print("❌ Не найден BOT_TOKEN в переменных окружения!")
        exit(1)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Бот запущен. Ожидание сообщений...")
    app.run_polling()
