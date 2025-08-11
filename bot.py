import os
import tempfile
import logging
import datetime
import requests
import subprocess

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ---------- Логирование ----------
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Конфиг ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")

ASR_PROVIDER = os.getenv("ASR_PROVIDER", "deepgram").lower()  # deepgram по умолчанию
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ASR_DIARIZE = os.getenv("ASR_DIARIZE", "false").lower() in ("1", "true", "yes")

if ASR_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
    raise RuntimeError("Установите DEEPGRAM_API_KEY в переменных окружения.")

# ---------- Утилиты ----------
def ffmpeg_extract_audio(input_path: str, audio_path: str) -> None:
    """Извлекает аудио в моно WAV 16 кГц через ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")[-1000:]
        raise RuntimeError(f"ffmpeg не смог извлечь аудио. Последние строки:\n{err}")

def ensure_min_size(path: str, min_bytes: int = 1_000_000) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError("Файл слишком маленький или не докачался (меньше 1 МБ).")

def transcribe_deepgram(wav_path: str) -> str:
    params = {
        "smart_format": "true",
        "diarize": "true" if ASR_DIARIZE else "false",
        "punctuate": "true",
        "paragraphs": "true",
        "language": "ru"
    }
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    with open(wav_path, "rb") as f:
        r = requests.post("https://api.deepgram.com/v1/listen",
                          params=params, headers=headers, data=f, timeout=3600)
    if r.status_code >= 300:
        raise RuntimeError(f"Deepgram ошибка {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        alts = data["results"]["channels"][0]["alternatives"]
        paragraphs = alts[0].get("paragraphs", {}).get("paragraphs")
        if paragraphs:
            parts = []
            for p in paragraphs:
                speaker = p.get("speaker", "")
                text = p.get("text", "").strip()
                if ASR_DIARIZE and speaker != "":
                    parts.append(f"Спикер {speaker}: {text}")
                else:
                    parts.append(text)
            return "\n\n".join(parts).strip()
        return alts[0]["transcript"].strip()
    except Exception:
        return ""

async def transcribe_and_reply(local_media_path: str, update: Update) -> None:
    await update.message.reply_text("⏳ Извлекаю аудио (ffmpeg)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        ffmpeg_extract_audio(local_media_path, audio_path)

        await update.message.reply_text("⏳ Распознаю (Deepgram)…")
        text = transcribe_deepgram(audio_path)
        if not text:
            await update.message.reply_text("Не удалось распознать речь.")
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        txt_path = os.path.join(tmpdir, f"transcript_{now}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        with open(txt_path, "rb") as f:
            await update.message.reply_document(document=InputFile(f, filename=os.path.basename(txt_path)))
        await update.message.reply_text("✅ Готово! Текст отправлен.")

def download_from_link(link: str, dest_path: str) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(link, headers=headers, stream=True, timeout=60) as r:
        content_type = (r.headers.get("Content-Type") or "").lower()
        if r.status_code != 200 or "html" in content_type:
            raise RuntimeError("Скачалась HTML-страница вместо файла. Дайте прямую ссылку на медиа.")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

# ---------- Обработчики ----------
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    media = update.message.video or update.message.voice or update.message.audio or update.message.document
    if not media:
        await update.message.reply_text("Это не поддерживаемый тип файла.")
        return

    file = await context.bot.get_file(media.file_id)
    await update.message.reply_text("⏳ Скачиваю файл...")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.bin")
        await file.download_to_drive(input_path)
        try:
            ensure_min_size(input_path)
        except Exception as e:
            await update.message.reply_text(f"❌ Проблема со скачиванием: {e}")
            return
        await transcribe_and_reply(input_path, update)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or update.message.caption or ""
    link = next((w for w in text.split() if w.startswith("http")), None)
    if not link:
        await update.message.reply_text("ℹ️ Отправьте ссылку на файл или загрузите видео/аудио напрямую.")
        return

    await update.message.reply_text("⏳ Загружаю файл по ссылке...")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "downloaded.bin")
        try:
            download_from_link(link, local_path)
            ensure_min_size(local_path)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при скачивании: {e}")
            return
        await transcribe_and_reply(local_path, update)

# ---------- Entrypoint ----------
def main() -> None:
    logger.info("📡 Бот запускается... (ASR_PROVIDER=%s, DIARIZE=%s)", ASR_PROVIDER, ASR_DIARIZE)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.VIDEO | filters.VOICE | filters.AUDIO | filters.Document.VIDEO | filters.Document.AUDIO, handle_media))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("✅ Бот запущен. Ожидание сообщений...")
    app.run_polling()

if __name__ == "__main__":
    main()
