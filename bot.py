import os
import re
import logging
import tempfile
import subprocess
import requests

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# ===== ЛОГИ =====
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("asr-bot")

# ===== КОНФИГ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")

# Провайдер: 'local' по умолчанию. (Deepgram можно включить позже переменной ASR_PROVIDER=deepgram)
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "local").lower()

# Настройки локальной модели faster-whisper
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")   # варианты: tiny/base/small/medium
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))  # 1 — быстрее
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 | int8_float16 | float32
LANGUAGE = os.getenv("LANGUAGE", "ru")  # 'ru' для стабильности. Поставь 'auto' — для автоопределения.

# Deepgram (на будущее, можно не задавать)
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ASR_DIARIZE = os.getenv("ASR_DIARIZE", "false").lower() in ("1", "true", "yes")

# ===== УТИЛИТЫ =====
def ensure_min_size(path: str, min_bytes: int = 10_000) -> None:
    """Ранний отсев пустых/битых файлов (10 КБ по умолчанию)."""
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"Файл слишком маленький или не докачался (< {min_bytes // 1000} КБ).")

def run_ffmpeg(cmd: list) -> None:
    """Запускаем ffmpeg и поднимаем понятную ошибку при падении."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg ошибка:\n{tail}")

def extract_audio_to_wav16k_mono(src_path: str, dst_wav_path: str) -> None:
    """Вытаскиваем аудио в WAV 16кГц/моно. Добавляем лёгкую нормализацию громкости."""
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-filter:a", "loudnorm=I=-23:TP=-2:LRA=7",
        dst_wav_path
    ]
    run_ffmpeg(cmd)

def normalize_link(url: str) -> str:
    """Пытаемся превратить общие ссылки в 'прямые' для скачивания."""
    try:
        url = url.strip()

        # Nextcloud/ownCloud публичные ссылки вида .../s/<id> -> добавляем /download
        if "/s/" in url and "download" not in url:
            if not url.endswith("/download"):
                url = url.rstrip("/") + "/download"

        # Google Drive:
        # 1) /file/d/<ID>/view?... -> /uc?export=download&id=<ID>
        m = re.search(r"drive\.google\.com/.*/file/d/([^/]+)/", url)
        if m:
            fid = m.group(1)
            return f"https://drive.google.com/uc?export=download&id={fid}"
        # 2) open?id=<ID> -> /uc?export=download&id=<ID>
        m = re.search(r"drive\.google\.com/.*[?&]id=([^&]+)", url)
        if m:
            fid = m.group(1)
            return f"https://drive.google.com/uc?export=download&id={fid}"

        return url
    except Exception:
        return url

def download_from_link(link: str, dest_path: str) -> None:
    """Качаем по ссылке (stream). Отсекаем HTML-страницы (непрямая ссылка)."""
    link = normalize_link(link)
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(link, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        # Если явно HTML/текст — это почти всегда страница предпросмотра
        if "html" in ctype or "text/" in ctype:
            raise RuntimeError("Скачалась HTML-страница. Нужна ПРЯМАЯ ссылка на файл (или Nextcloud с /download).")
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        if total < 10_000:
            raise RuntimeError("По ссылке пришёл слишком маленький файл (<10 КБ). Проверьте прямую ссылку.")

# ===== ЛОКАЛЬНОЕ РАСПОЗНАВАНИЕ (faster-whisper) =====
_faster_model = None
def load_faster_whisper():
    global _faster_model
    if _faster_model is None:
        from faster_whisper import WhisperModel
        log.info("Загружаю faster-whisper: model=%s compute=%s", WHISPER_MODEL, WHISPER_COMPUTE_TYPE)
        language = None if LANGUAGE.lower() == "auto" else LANGUAGE
        _faster_model = (WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=max(1, os.cpu_count() // 2)
        ), language)
    return _faster_model

def transcribe_local(wav_path: str) -> str:
    (model, language) = load_faster_whisper()
    segments, info = model.transcribe(
        wav_path,
        language=language,                    # None -> автоопределение
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False
    )
    parts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()

# ===== ОБЛАЧНОЕ РАСПОЗНАВАНИЕ (Deepgram, если включено) =====
def transcribe_deepgram(wav_path: str) -> str:
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"}
    params = {
        "smart_format": "true",
        "punctuate": "true",
        "paragraphs": "true",
        "diarize": "true" if ASR_DIARIZE else "false",
        "language": LANGUAGE if LANGUAGE.lower() != "auto" else "ru"
    }
    with open(wav_path, "rb") as f:
        r = requests.post("https://api.deepgram.com/v1/listen", headers=headers, params=params, data=f, timeout=1800)
    if r.status_code >= 300:
        raise RuntimeError(f"Deepgram вернул ошибку {r.status_code}:\n{r.text[:600]}")
    data = r.json()
    try:
        alts = data["results"]["channels"][0]["alternatives"]
        paragraphs = alts[0].get("paragraphs", {}).get("paragraphs") or []
        if paragraphs:
            out = []
            for p in paragraphs:
                txt = (p.get("text") or "").strip()
                if not txt:
                    continue
                if ASR_DIARIZE and p.get("speaker") is not None:
                    out.append(f"Спикер {p['speaker']}: {txt}")
                else:
                    out.append(txt)
            if out:
                return "\n\n".join(out).strip()
        return (alts[0].get("transcript") or "").strip()
    except Exception:
        return ""

# ===== ХЭНДЛЕРЫ =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    # 1) Если прислали вложение (видео/аудио/документ)
    media = msg.video or msg.voice or msg.audio or msg.document
    if media:
        await msg.reply_text("📥 Скачиваю файл…")
        tg_file = await context.bot.get_file(media.file_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.bin")
            await tg_file.download_to_drive(src)
            try:
                ensure_min_size(src, 10_000)
            except Exception as e:
                await msg.reply_text(f"❌ {e}")
                return

            await msg.reply_text("🎙 Извлекаю аудио (ffmpeg)…")
            wav = os.path.join(tmpdir, "audio.wav")
            try:
                extract_audio_to_wav16k_mono(src, wav)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка при конвертации: {e}")
                return

            try:
                if ASR_PROVIDER == "deepgram" and DEEPGRAM_API_KEY:
                    await msg.reply_text("🤖 Распознаю (Deepgram)…")
                    text = transcribe_deepgram(wav)
                else:
                    await msg.reply_text("🤖 Распознаю (локально, faster-whisper)…")
                    text = transcribe_local(wav)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка распознавания: {e}")
                return

            if text:
                out_path = os.path.join(tmpdir, "transcript.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)
                await msg.reply_document(InputFile(out_path, filename="transcript.txt"))
                await msg.reply_text("✅ Готово.")
            else:
                await msg.reply_text("⚠️ Текст не получен. Попробуйте запись подлиннее/громче или чище источник.")
        return

    # 2) Если прислали ссылку (в тексте или подписи)
    text = (msg.text or "") + " " + (msg.caption or "")
    m = re.search(r"(https?://\S+)", text)
    if m:
        link = m.group(1)
        await msg.reply_text("🌐 Скачиваю файл по ссылке…")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "download.bin")
            try:
                download_from_link(link, src)
                ensure_min_size(src, 10_000)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка загрузки по ссылке: {e}")
                return

            await msg.reply_text("🎙 Извлекаю аудио (ffmpeg)…")
            wav = os.path.join(tmpdir, "audio.wav")
            try:
                extract_audio_to_wav16k_mono(src, wav)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка при конвертации: {e}")
                return

            try:
                if ASR_PROVIDER == "deepgram" and DEEPGRAM_API_KEY:
                    await msg.reply_text("🤖 Распознаю (Deepgram)…")
                    text_out = transcribe_deepgram(wav)
                else:
                    await msg.reply_text("🤖 Распознаю (локально, faster-whisper)…")
                    text_out = transcribe_local(wav)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка распознавания: {e}")
                return

            if text_out:
                out_path = os.path.join(tmpdir, "transcript.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text_out)
                await msg.reply_document(InputFile(out_path, filename="transcript.txt"))
                await msg.reply_text("✅ Готово.")
            else:
                await msg.reply_text("⚠️ Текст не получен. Убедитесь, что ссылка указывает на сам файл, не на страницу.")
        return

    # 3) Подсказка
    await msg.reply_text("ℹ️ Пришлите аудио/видео вложением ИЛИ прямую ссылку на файл (Google Drive/Nextcloud/прямая ссылка).")

# ===== ЗАПУСК =====
def main():
    log.info("Запуск бота… ASR_PROVIDER=%s, WHISPER_MODEL=%s, compute=%s, language=%s",
             ASR_PROVIDER, WHISPER_MODEL, WHISPER_COMPUTE_TYPE, LANGUAGE)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    log.info("✅ Бот запущен. Ожидание сообщений…")
    app.run_polling()

if __name__ == "__main__":
    main()
