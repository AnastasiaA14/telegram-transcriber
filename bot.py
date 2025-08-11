import os
import tempfile
import logging
import datetime
import requests
import subprocess

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters


# ---------- Логирование ----------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("tg-transcriber")


# ---------- Конфиг из переменных окружения ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "deepgram").lower()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ASR_DIARIZE = os.getenv("ASR_DIARIZE", "false").lower() in ("1", "true", "yes", "y")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")
if ASR_PROVIDER == "deepgram" and not DEEPGRAM_API_KEY:
    raise RuntimeError("Для Deepgram требуется переменная DEEPGRAM_API_KEY.")


# ---------- Небольшие утилиты ----------
def ensure_min_size(path: str, min_bytes: int = 1_000_000) -> None:
    """Проверяем, что файл не пустой/HTML и докачался хотя бы до 1 МБ."""
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError("Файл слишком маленький или не докачался (< 1 МБ).")


def ffmpeg_extract_audio(input_path: str, audio_path: str) -> None:
    """Извлекаем аудио в WAV моно 16 кГц через ffmpeg."""
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
        tail = proc.stderr.decode(errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg не смог извлечь аудио.\n{tail}")


def download_from_link(link: str, dest_path: str) -> None:
    """Качаем по прямой ссылке. Если пришла HTML-страница — бросаем ошибку."""
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(link, headers=headers, stream=True, timeout=120) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if r.status_code != 200 or "html" in ctype:
            raise RuntimeError("Похоже, это страница с предпросмотром. Нужна прямая ссылка на файл.")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def transcribe_deepgram(wav_path: str) -> str:
    """Отправляем аудио в Deepgram и возвращаем текст (с диаризацией по желанию)."""
    params = {
        "smart_format": "true",
        "punctuate": "true",
        "paragraphs": "true",
        "diarize": "true" if ASR_DIARIZE else "false",
        "language": "ru"
    }
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    with open(wav_path, "rb") as f:
        r = requests.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            data=f,
            timeout=3600
        )
    if r.status_code >= 300:
        msg = r.text[:600]
        raise RuntimeError(f"Deepgram вернул ошибку {r.status_code}:\n{msg}")

    data = r.json()
    try:
        alts = data["results"]["channels"][0]["alternatives"]
        # если Deepgram дал параграфы — склеиваем
        paragraphs = alts[0].get("paragraphs", {}).get("paragraphs")
        if paragraphs:
            parts = []
            for p in paragraphs:
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                if ASR_DIARIZE:
                    spk = p.get("speaker")
                    if spk is not None:
                        parts.append(f"Спикер {spk}: {text}")
                    else:
                        parts.append(text)
                else:
                    parts.append(text)
            return "\n\n".join(parts).strip()

        # иначе обычная расшифровка
        return (alts[0].get("transcript") or "").strip()
    except Exception:
        return ""


# ---------- Основная логика ----------
async def process_local_media(local_path: str, update: Update) -> None:
    await update.message.reply_text("⏳ Извлекаю аудио (ffmpeg)…")
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        ffmpeg_extract_audio(local_path, wav_path)

        await update.message.reply_text("⏳ Распознаю (Deepgram)…")
        text = transcribe_deepgram(wav_path)
        if not text:
            await update.message.reply_text("❌ Не удалось распознать речь.")
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_path = os.path.join(tmpdir, f"transcript_{now}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        with open(out_path, "rb") as f:
            await update.message.reply_document(
                document=InputFile(f, filename=os.path.basename(out_path))
            )
        await update.message.reply_text("✅ Готово! Текст отправлен.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        msg = update.message
        if not msg:
            return

        # --- если прислали медиа (видео/аудио/документ с медиа) ---
        media = msg.video or msg.voice or msg.audio or msg.document
        if media:
            await msg.reply_text("⏳ Скачиваю файл…")
            tg_file = await context.bot.get_file(media.file_id)
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "input.bin")
                await tg_file.download_to_drive(in_path)
                ensure_min_size(in_path)
                await process_local_media(in_path, update)
            return

        # --- если прислали ссылку в тексте/подписи ---
        text = (msg.text or "") + " " + (msg.caption or "")
        link = next((w for w in text.split() if w.startswith("http")), None)
        if link:
            await msg.reply_text("⏳ Загружаю файл по ссылке…")
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = os.path.join(tmpdir, "downloaded.bin")
                download_from_link(link, in_path)
                ensure_min_size(in_path)
                await process_local_media(in_path, update)
            return

        # --- иначе подсказка ---
        await msg.reply_text("ℹ️ Отправьте видео/аудио или прямую ссылку на файл. Я пришлю текст в .txt.")
    except Exception as e:
        logger.exception("Ошибка обработки сообщения: %s", e)
        await update.message.reply_text(f"❌ Ошибка: {e}")


def main() -> None:
    logger.info("📡 Запуск бота… (ASR_PROVIDER=%s, DIARIZE=%s)", ASR_PROVIDER, ASR_DIARIZE)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    # Один универсальный обработчик: сам определяет, что пришло
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    logger.info("✅ Бот запущен. Ожидание сообщений…")
    app.run_polling()


if __name__ == "__main__":
    main()
