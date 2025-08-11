import os
import requests
import tempfile
import subprocess
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Настройка логов
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Папка для временных файлов
TEMP_DIR = tempfile.gettempdir()

# Функция извлечения аудио из видео
def extract_audio(input_path, output_path):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path
    ]
    subprocess.run(cmd, check=True)

# Функция распознавания через Deepgram (без model/tier для бесплатного плана)
def transcribe_deepgram(file_path):
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    params = {
        "smart_format": "true",
        "punctuate": "true",
        "paragraphs": "true",
        "diarize": "true" if os.getenv("ASR_DIARIZE", "false").lower() in ("1","true","yes") else "false",
        "language": "ru"
    }

    with open(file_path, "rb") as f:
        r = requests.post(url, headers=headers, params=params, data=f)

    if r.status_code != 200:
        msg = r.text
        raise RuntimeError(f"Deepgram вернул ошибку {r.status_code}:\n{msg}")

    result = r.json()
    return result["results"]["channels"][0]["alternatives"][0]["transcript"]

# Обработчик медиа
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.effective_attachment:
        return

    file = await update.message.effective_attachment.get_file()
    temp_input = os.path.join(TEMP_DIR, file.file_id)
    temp_wav = temp_input + ".wav"

    await file.download_to_drive(temp_input)

    try:
        extract_audio(temp_input, temp_wav)
        text = transcribe_deepgram(temp_wav)
        await update.message.reply_text(text)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка распознавания: {e}")
    finally:
        for path in (temp_input, temp_wav):
            try:
                os.remove(path)
            except OSError:
                pass

# Запуск бота
if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        logging.error("Не указан TELEGRAM_TOKEN")
        exit(1)
    if not DEEPGRAM_API_KEY:
        logging.error("Не указан DEEPGRAM_API_KEY")
        exit(1)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_media))
    app.run_polling()
