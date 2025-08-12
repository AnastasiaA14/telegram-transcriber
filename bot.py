import os
import re
import logging
import tempfile
import subprocess
import requests
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from telegram.error import BadRequest

# ===== ЛОГИ =====
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("asr-bot")

# ===== КОНФИГ =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")

# Локальное распознавание (faster-whisper)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")          # tiny/base/small/medium
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1")) # 1 — быстрее
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 | int8_float16 | float32
LANGUAGE = os.getenv("LANGUAGE", "ru")                       # 'auto' для автоопределения

# Лимит вложения в ТГ (ориентир): если больше — просим ссылку
TELEGRAM_ATTACHMENT_LIMIT = 45 * 1024 * 1024  # 45 МБ

# ===== УТИЛИТЫ =====
def ensure_min_size(path: str, min_bytes: int = 10_000) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"Файл слишком маленький или не докачался (< {min_bytes // 1000} КБ).")

def run_ffmpeg(cmd: list) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg ошибка:\n{tail}")

def extract_audio_to_wav16k_mono(src_path: str, dst_wav_path: str) -> None:
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

# ---------- Google Drive: делаем прямую ссылку ----------
def normalize_google_drive(url: str) -> str | None:
    # /file/d/<ID>/view?...  -> uc?export=download&id=<ID>
    m = re.search(r"drive\.google\.com/.*/file/d/([^/]+)/", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    # open?id=<ID> -> uc?export=download&id=<ID>
    m = re.search(r"drive\.google\.com/.*[?&]id=([^&]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return None

# ---------- Zoom: share/play -> download, добавляем pwd ----------
def normalize_zoom(url: str, pwd_hint: str | None) -> str | None:
    if "zoom.us/rec/" not in url:
        return None
    p = urlparse(url)
    path = p.path
    q = parse_qs(p.query)
    pwd = (q.get("pwd", [None])[0]) or (pwd_hint or None)

    # заменим /play или /share на /download
    path = path.replace("/play", "/download").replace("/share", "/download")

    # соберём query обратно, добавим pwd если есть
    if pwd:
        q["pwd"] = [pwd]
    query = urlencode({k: v[0] for k, v in q.items() if v and v[0] is not None})
    new_url = urlunparse((p.scheme, p.netloc, path, "", query, ""))
    return new_url

def download_http(link: str, dest_path: str, referer: str | None = None) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    with requests.get(link, headers=headers, stream=True, timeout=1800, allow_redirects=True) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        # Zoom/Drive могут отдавать octet-stream или video/* — это ок; HTML — плохо
        if "html" in ctype or "text/" in ctype:
            raise RuntimeError("Скачалась HTML-страница. Нужна прямая ссылка на файл (или верный пароль для Zoom).")
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        if total < 10_000:
            raise RuntimeError("По ссылке пришёл слишком маленький файл (<10 КБ).")

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
    segments, _ = model.transcribe(
        wav_path,
        language=language,
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

# ===== ХЭНДЛЕРЫ =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    # 0) Если это ссылка в тексте/подписи — обработаем как ссылку (универсально для больших файлов)
    text_all = (msg.text or "") + " " + (msg.caption or "")
    m_url = re.search(r"(https?://\S+)", text_all)
    m_pwd = re.search(r"\bpwd\s*[:=]\s*([A-Za-z0-9]+)", text_all)  # можно прислать: "pwd: ABCD1234"
    pwd_hint = m_pwd.group(1) if m_pwd else None

    if m_url:
        raw_url = m_url.group(1)
        await msg.reply_text("🌐 Скачиваю файл по ссылке…")

        # Нормализуем ссылку, если это Drive или Zoom
        gdrive = normalize_google_drive(raw_url)
        if gdrive:
            norm_url, referer = gdrive, None
        else:
            z = normalize_zoom(raw_url, pwd_hint)
            if z:
                norm_url, referer = z, raw_url  # Zoom иногда требует Referer на share/play
            else:
                norm_url, referer = raw_url, None

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "download.bin")
            try:
                download_http(norm_url, src, referer=referer)
                ensure_min_size(src, 10_000)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка загрузки: {e}\n"
                                     f"Для Zoom: пришлите ссылку на запись + пароль в сообщении, например:\n"
                                     f"https://us02web.zoom.us/rec/share/...  pwd: ABCD1234")
                return
            await process_local_file(src, msg)
        return

    # 1) Вложение (если нет ссылки)
    media = msg.video or msg.voice or msg.audio or msg.document
    if media:
        size = getattr(media, "file_size", None)
        if size and size > TELEGRAM_ATTACHMENT_LIMIT:
            await msg.reply_text("❗️ Файл слишком большой для вложения. Пришлите ссылку на файл (Google Drive/Zoom/другой хост).")
            return

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
            await process_local_file(src, msg)
        return

    # 2) Подсказка
    await msg.reply_text("ℹ️ Пришлите аудио/видео как вложение (если не очень большое) ИЛИ ссылку на файл.\n"
                         "Google Drive — обычная ссылка; Zoom — ссылка на запись + добавьте в сообщении `pwd: ПАРОЛЬ`.")

async def process_local_file(src: str, msg):
    try:
        await msg.reply_text("🎙 Извлекаю аудио (ffmpeg)…")
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = os.path.join(tmpdir, "audio.wav")
            extract_audio_to_wav16k_mono(src, wav)

            await msg.reply_text("🤖 Распознаю (локально, faster-whisper)…")
            text_out = transcribe_local(wav)

            if text_out:
                out_path = os.path.join(tmpdir, "transcript.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text_out)
                try:
                    await msg.reply_document(InputFile(out_path, filename="transcript.txt"))
                except BadRequest:
                    await msg.reply_text("✅ Готово. Текст длинный — пришлю частями.")
                    with open(out_path, "r", encoding="utf-8") as f:
                        data = f.read()
                    for i in range(0, len(data), 3500):
                        await msg.reply_text(data[i:i+3500])
                await msg.reply_text("✅ Готово.")
            else:
                await msg.reply_text("⚠️ Текст не получен. Убедитесь, что источник не пустой/без звука.")
    except Exception as e:
        await msg.reply_text(f"❌ Ошибка: {e}")

# ===== ЗАПУСК =====
def main():
    log.info("Запуск бота… WHISPER_MODEL=%s, compute=%s, language=%s", WHISPER_MODEL, WHISPER_COMPUTE_TYPE, LANGUAGE)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    log.info("✅ Бот запущен. Ожидание сообщений…")
    app.run_polling()

if __name__ == "__main__":
    main()
