import os
import logging
import requests
import tempfile
import subprocess
from telegram.ext import Application, MessageHandler, filters
from telegram import Update
from telegram.ext import ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

FFMPEG_BIN = "ffmpeg"

def transcribe_deepgram(audio_path):
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    params = {
        "model": "nova-2",
        "language": "ru",
        "smart_format": "true",
        "tier": "enhanced",
        "diarize": "true",
        "utterances": "true",
        "profanity_filter": "false"
    }

    with open(audio_path, "rb") as f:
        r = requests.post(url, headers=headers, params=params, data=f)

    if r.status_code != 200:
        raise RuntimeError(f"Deepgram вернул ошибку {r.status_code}:\n{r.text}")

    data = r.json()
    try:
        text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        return text.strip() if text else ""
    except KeyError:
        return ""

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_id = None
    if update.message.audio:
        file_id = update.message.audio.file_id
    elif update.message.voice:
        file_id = update.message.voice.file_id
    elif update.message.video:
        file_id = update.message.video.file_id
    elif update.message.document:
        file_id = update.message.document.file_id

    if not file_id:
        await update.message.reply_text("Отправьте аудио, видео или документ с медиа.")
        return

    file = await context.bot.get_file(file_id)

    await update.message.reply_text("📥 Скачиваю файл...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        await file.download_to_drive(tmp_in.name)
        in_path = tmp_in.name

    if os.path.getsize(in_path) < 10_000:
        await update.message.reply_text("❌ Ошибка: Файл слишком маленький или не докачался (< 10 КБ).")
        os.remove(in_path)
        return

    wav_path = tempfile.mktemp(suffix=".wav")
    await update.message.reply_text("🎙 Конвертирую в WAV...")
    try:
        subprocess.run([FFMPEG_BIN, "-i", in_path, "-ar", "16000", "-ac", "1", wav_path], check=True)
    except subprocess.CalledProcessError:
        await update.message.reply_text("❌ Ошибка при конвертации файла.")
        os.remove(in_path)
        return

    await update.message.reply_text("🤖 Распознаю речь (Deepgram)...")
    try:
        text = transcribe_deepgram(wav_path)
    except RuntimeError as e:
        await update.message.reply_text(f"❌ Ошибка распознавания: {e}")
        os.remove(in_path)
        os.remove(wav_path)
        return

    os.remove(in_path)
    os.remove(wav_path)

    if text:
        await update.message.reply_text(f"✅ Распознанный текст:\n\n{text}")
    else:
        await update.message.reply_text("⚠️ Файл распознан, но текст пуст. Возможно, речь была слишком тихой или непонятной.")

if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise ValueError("❌ Не задан TELEGRAM_TOKEN")
    if not DEEPGRAM_API_KEY:
        raise ValueError("❌ Не задан DEEPGRAM_API_KEY")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, handle_message))
    logging.info("Бот запущен...")
    app.run_polling()
