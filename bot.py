import os
import re
import logging
import tempfile
import subprocess
import requests
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode, quote

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

# Лимит вложения в ТГ (ориентир)
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
    m = re.search(r"drive\.google\.com/.*/file/d/([^/]+)/", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    m = re.search(r"drive\.google\.com/.*[?&]id=([^&]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return None

# ---------- Вытаскиваем пароль (любые языки/форматы) ----------
PWD_PATTERNS = [
    r"\bpwd\s*[:=]\s*(\S+)",                         # pwd: XXXXX
    r"(?i)\bpasscode\s*[:=]\s*(\S+)",                # passcode: XXXXX
    r"(?i)\bpassword\s*[:=]\s*(\S+)",                # password: XXXXX
    r"(?i)\bсекретн[ыйаяе]\s*код\s*[:=]\s*(\S+)",    # Секретный код: XXXXX
    r"(?i)\bпарол[ья]\s*[:=]\s*(\S+)",               # Пароль: XXXXX
]
def extract_pwd_hint(text: str) -> str | None:
    for pat in PWD_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None

# ---------- Zoom: двухшаговая загрузка (share/play -> cookies -> download) ----------
def build_zoom_urls(raw_url: str, pwd_hint: str | None):
    """Возвращает (share_url_with_pwd, download_url, referer). Пароль кодируем."""
    p = urlparse(raw_url)
    if "zoom.us" not in p.netloc or "/rec/" not in p.path:
        return None

    q = parse_qs(p.query)
    pwd = q.get("pwd", [None])[0] or pwd_hint
    path_share = p.path
    # share/play -> оставим как есть для шага 1 (cookie)
    share_q = dict((k, v[0]) for k, v in q.items() if v and v[0] is not None)

    # Сконструируем download путь
    path_download = path_share.replace("/play", "/download").replace("/share", "/download")
    dl_q = dict(share_q)

    if pwd:
        # важно кодировать полностью
        dl_q["pwd"] = quote(pwd, safe="")
        # и в share тоже добавим (часто помогает избежать промежуточной формы)
        share_q["pwd"] = dl_q["pwd"]

    # полезно явно просить скачивание
    dl_q["download"] = "1"

    share_url = urlunparse((p.scheme, p.netloc, path_share, "", urlencode(share_q), ""))
    download_url = urlunparse((p.scheme, p.netloc, path_download, "", urlencode(dl_q), ""))

    return share_url, download_url, raw_url  # referer = исходная share/play ссылка

def download_zoom_2step(raw_url: str, pwd_hint: str | None, dest_path: str) -> None:
    built = build_zoom_urls(raw_url, pwd_hint)
    if not built:
        raise RuntimeError("Ссылка не похожа на Zoom запись.")
    share_url, download_url, referer = built

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": referer
    }
    s = requests.Session()

    # Шаг 1: зайти на share/play, чтобы получить cookies (и передать pwd через URL)
    r1 = s.get(share_url, headers=headers, allow_redirects=True, timeout=1800)
    # Иногда Zoom всё равно возвращает HTML-страницу, это нормально — главное, что cookies поставились

    # Шаг 2: скачать download с теми же cookies
    with s.get(download_url, headers=headers, stream=True, allow_redirects=True, timeout=1800) as r2:
        ctype = (r2.headers.get("Content-Type") or "").lower()
        if "html" in ctype or "text/" in ctype:
            raise RuntimeError("Zoom не выдал файл. Проверьте пароль и включено ли скачивание в настройках записи.")
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r2.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        if total < 10_000:
            raise RuntimeError("Zoom выдал слишком маленький файл (<10 КБ).")

# ---------- Обычное скачивание по прямому URL ----------
def download_http(link: str, dest_path: str, referer: str | None = None) -> None:
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer:
        headers["Referer"] = referer
    with requests.get(link, headers=headers, stream=True, timeout=1800, allow_redirects=True) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" in ctype or "text/" in ctype:
            raise RuntimeError("Скачалась HTML-страница. Нужна прямая ссылка на файл.")
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

    # 0) Ссылка в тексте/подписи
    text_all = (msg.text or "") + " " + (msg.caption or "")
    m_url = re.search(r"(https?://\S+)", text_all)
    pwd_hint = extract_pwd_hint(text_all)

    if m_url:
        raw_url = m_url.group(1)
        await msg.reply_text("🌐 Скачиваю файл по ссылке…")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "download.bin")
            try:
                if "zoom.us/rec/" in raw_url:
                    # Zoom: двухшаговая загрузка с паролем/куки
                    download_zoom_2step(raw_url, pwd_hint, src)
                else:
                    # Google Drive/другие — нормализуем Drive, иначе качаем как есть
                    gdrive = normalize_google_drive(raw_url)
                    norm_url = gdrive or raw_url
                    download_http(norm_url, src)
                ensure_min_size(src, 10_000)
            except Exception as e:
                await msg.reply_text(
                    "❌ Ошибка загрузки: {err}\n\n"
                    "Подсказки:\n"
                    "• Для Zoom пришлите ссылку на запись и пароль в сообщении, например:\n"
                    "  pwd: ABCD1234  или  Секретный код: ABCD1234\n"
                    "• Убедитесь, что у записи включено разрешение «Allow viewers to download»."
                    .format(err=e)
                )
                return
            await process_local_file(src, msg)
        return

    # 1) Вложение (если нет ссылки)
    media = msg.video or msg.voice or msg.audio or msg.document
    if media:
        size = getattr(media, "file_size", None)
        if size and size > TELEGRAM_ATTACHMENT_LIMIT:
            await msg.reply_text("❗️ Файл слишком большой для вложения. Пришлите ссылку (Zoom/Google Drive/другой хост).")
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
    await msg.reply_text("ℹ️ Пришлите аудио/видео вложением (если не очень большое) ИЛИ ссылку на файл.\n"
                         "Поддержка: Google Drive (обычная ссылка), Zoom (добавьте пароль в сообщении).")

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
