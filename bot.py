import os
import re
import html
import logging
import tempfile
import subprocess
import urllib.parse
import requests
from typing import Optional, Tuple, List

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# ================= ЛОГИ =================
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger("asr-bot")

# ================= КОНФИГ =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Переменная окружения TELEGRAM_TOKEN не установлена.")

# Провайдер: только локально (бесплатно)
ASR_PROVIDER = "local"

# Настройки faster-whisper
# По умолчанию — medium (лучшее качество на CPU). Можно поменять через переменную окружения.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")   # tiny/base/small/medium
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 | int8_float16 | float32
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))      # 1 — быстрее
LANGUAGE = os.getenv("LANGUAGE", "ru")  # 'ru' стабильно для русской речи. Поставь 'auto' — автоопределение.

# Обработка очень длинных файлов: режем WAV на куски (в секундах)
# 900с = 15 минут. Можно увеличить/уменьшить через переменную.
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "900"))

# Минимальный размер загруженного файла
MIN_BYTES = int(os.getenv("MIN_BYTES", "10000"))  # 10 КБ — отсев пустышек/обрывов

# ================= УТИЛИТЫ =================
def ensure_min_size(path: str, min_bytes: int = MIN_BYTES) -> None:
    """Ранний отсев пустых/битых файлов."""
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"Файл слишком маленький или не докачался (< {min_bytes // 1000} КБ).")

def run_ffmpeg(cmd: List[str]) -> None:
    """Запуск ffmpeg с понятной ошибкой."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore")[-2000:]
        raise RuntimeError(f"ffmpeg ошибка:\n{tail}")

def run_ffprobe_duration(path: str) -> Optional[float]:
    """Получить длительность файла (сек) через ffprobe. Вернуть None при ошибке."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            return None
        val = proc.stdout.strip()
        return float(val) if val else None
    except Exception:
        return None

def extract_audio_to_wav16k_mono(src_path: str, dst_wav_path: str) -> None:
    """
    Вытаскиваем аудио в WAV 16 кГц моно + лёгкая нормализация громкости.
    """
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
    """
    Нормализуем некоторые ссылки (Nextcloud -> /download).
    Google Drive обрабатываем отдельной логикой в download_from_link().
    """
    url = (url or "").strip()
    # Nextcloud / ownCloud: добавим /download для публичных ссылок с /s/
    if "/s/" in url and "download" not in url:
        if not url.endswith("/download"):
            url = url.rstrip("/") + "/download"
    return url

# =============== ZOOM ===============
ZOOM_HOST_RE = re.compile(r"https?://([\w\-]+\.)?zoom\.us/rec/", re.IGNORECASE)
PASS_PATTERNS = [
    r"(?:\b|^)(?:pwd|passcode|пароль|секретный\s*код)\s*[:：]\s*([A-Za-z0-9_\-\.\$\^\@\!\&\*]+)",
    r"Секретный\s*код\W*([A-Za-z0-9_\-\.\$\^\@\!\&\*]+)",
]

def extract_passcode(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in PASS_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def download_zoom_recording(share_url: str, passcode: Optional[str], dest_path: str) -> None:
    """
    Качаем Zoom Cloud Recording, если у записи включено «Allow viewers to download».
    Если отключено — Zoom не отдаст прямую ссылку (скачивание будет запрещено).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    url = share_url
    if passcode and "pwd=" not in url:
        q = "&" if ("?" in url) else "?"
        url = f"{url}{q}pwd={urllib.parse.quote(passcode)}"

    r = session.get(url, timeout=120)
    if r.status_code != 200:
        raise RuntimeError("Zoom не пустил на страницу записи. Проверьте ссылку/пароль.")

    html_text = r.text
    m = re.search(r'"downloadUrl"\s*:\s*"([^"]+)"', html_text)
    if not m:
        m = re.search(r'"downloadUrl"\s*:\s*"(https:\\/\\/[^"]+)"', html_text)
    if not m:
        raise RuntimeError("Zoom не выдал ссылку на скачивание. Включите «Allow viewers to download» у записи.")

    dl = m.group(1)
    dl = html.unescape(dl).replace("\\/", "/").replace("\\u0026", "&")

    with session.get(dl, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Zoom вернул статус {resp.status_code} при скачивании. Проверьте доступ.")
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk); total += len(chunk)
        if total < MIN_BYTES:
            raise RuntimeError("Zoom скачал слишком маленький файл. Возможно, неверный пароль/доступ.")

# =============== GOOGLE DRIVE ===============
# Поддерживаем максимальное число форматов ссылок и «большие файлы с confirm».
DRIVE_FILE_ID_RE_LIST = [
    re.compile(r"[?&]id=([^&]+)", re.I),
    re.compile(r"drive\.google\.com/(?:uc|open)\?.*?[?&]id=([^&]+)", re.I),
    re.compile(r"drive\.google\.com/.*/file/d/([^/]+)", re.I),
    re.compile(r"drive\.google\.com/uc\?export=download&confirm=[^&]+&id=([^&]+)", re.I),
    re.compile(r"drive\.usercontent\.google\.com/uc\?id=([^&]+)", re.I),
]

def drive_extract_id(url: str) -> Optional[str]:
    for rx in DRIVE_FILE_ID_RE_LIST:
        m = rx.search(url)
        if m:
            return m.group(1)
    return None

def drive_download_with_confirm(session: requests.Session, url: str, file_id: str, dest_path: str) -> None:
    """
    1) Запрашиваем uc?export=download&id=...
    2) Если пришёл HTML (предупреждение) — ищем confirm-токен (в ссылке или cookie download_warning)
    3) Повторяем запрос с confirm=... и стягиваем файл потоково.
    """
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(base, stream=True, timeout=300, allow_redirects=True)
    cdisp = (resp.headers.get("Content-Disposition") or "").lower()
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "attachment" in cdisp and not (ctype.startswith("text/") or "html" in ctype):
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        return

    # Ищем confirm токен
    token = None
    text = ""
    try:
        text = resp.text
    except Exception:
        pass

    m = re.search(r'href="[^"]*?confirm=([0-9A-Za-z_\-]+)[^"]*?&id=' + re.escape(file_id), text or "")
    if m:
        token = m.group(1)

    if not token:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

    if not token:
        raise RuntimeError("Google Drive требует подтверждение (confirm), токен не найден. Проверьте доступ «Любой по ссылке: Просмотр».")

    # Повторный запрос с confirm
    url2 = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
    resp2 = session.get(url2, stream=True, timeout=600, allow_redirects=True)
    ctype2 = (resp2.headers.get("Content-Type") or "").lower()
    if ctype2.startswith("text/") or "html" in ctype2:
        raise RuntimeError("Google Drive всё ещё отдаёт HTML. Скорее всего, у файла закрыт доступ или это не файл (папка/Google-док).")
    with open(dest_path, "wb") as f:
        for chunk in resp2.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def download_from_link(link: str, dest_path: str, maybe_passcode: Optional[str] = None) -> None:
    link = normalize_link(link)

    # Zoom
    if ZOOM_HOST_RE.search(link):
        if not maybe_passcode:
            raise RuntimeError(
                "Zoom требует пароль. Пришлите ссылку и пароль в одном сообщении, например:\n"
                "pwd: ABCD1234  или  Секретный код: ABCD1234"
            )
        download_zoom_recording(link, maybe_passcode, dest_path)
        return

    # Google Drive
    if "drive.google.com" in link or "drive.usercontent.google.com" in link:
        file_id = drive_extract_id(link)
        if not file_id:
            raise RuntimeError("Не удалось извлечь ID файла Google Drive. Пришлите ссылку «Поделиться» именно на ФАЙЛ (не на папку).")
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        drive_download_with_confirm(session, link, file_id, dest_path)
        ensure_min_size(dest_path, MIN_BYTES)
        return

    # Обычная прямая ссылка
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(link, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" in ctype or ctype.startswith("text/"):
            raise RuntimeError("Скачалась HTML-страница. Нужна ПРЯМАЯ ссылка на файл (или Nextcloud с /download).")
        total = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk); total += len(chunk)
        if total < MIN_BYTES:
            raise RuntimeError("По ссылке пришёл слишком маленький файл. Проверьте прямую ссылку и доступ.")

# =============== ЛОКАЛЬНОЕ РАСПОЗНАВАНИЕ (faster-whisper) ===============
_faster_model = None
_cached_language = None

def load_faster_whisper():
    """
    Ленивая загрузка модели (один раз на процесс).
    """
    global _faster_model, _cached_language
    if _faster_model is None:
        from faster_whisper import WhisperModel
        language = None if LANGUAGE.lower() == "auto" else LANGUAGE
        _cached_language = language
        log.info("Загружаю faster-whisper: model=%s compute=%s", WHISPER_MODEL, WHISPER_COMPUTE_TYPE)
        _faster_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=max(1, os.cpu_count() // 2)
        )
    return _faster_model, _cached_language

def transcribe_wav_chunked(wav_path: str) -> str:
    """
    Транскрибируем WAV целиком или по кускам (если длительность > CHUNK_SECONDS).
    Это защищает от OOM и ускоряет большие файлы на слабом CPU.
    """
    model, language = load_faster_whisper()

    duration = run_ffprobe_duration(wav_path) or 0.0
    if duration <= 0:
        # не удалось получить длительность — распознаем как есть
        segments, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False
        )
        parts = [ (seg.text or "").strip() for seg in segments if (seg.text or "").strip() ]
        return "\n".join(parts).strip()

    if duration <= CHUNK_SECONDS:
        segments, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=False
        )
        parts = [ (seg.text or "").strip() for seg in segments if (seg.text or "").strip() ]
        return "\n".join(parts).strip()

    # Режем по кускам
    out_texts: List[str] = []
    start = 0.0
    idx = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        while start < duration:
            t = min(CHUNK_SECONDS, duration - start)
            chunk_path = os.path.join(tmpdir, f"chunk_{idx:04d}.wav")
            # вырезаем кусок
            cmd = ["ffmpeg", "-y", "-i", wav_path, "-vn", "-ac", "1", "-ar", "16000",
                   "-ss", str(start), "-t", str(t), chunk_path]
            run_ffmpeg(cmd)

            segments, info = model.transcribe(
                chunk_path,
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False
            )
            chunk_parts = [ (seg.text or "").strip() for seg in segments if (seg.text or "").strip() ]
            if chunk_parts:
                out_texts.append("\n".join(chunk_parts).strip())

            start += t
            idx += 1

    return "\n\n".join(out_texts).strip()

# =============== ТЕЛЕГРАМ-ХЭНДЛЕРЫ ===============
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    # 1) Вложение (видео/аудио/документ)
    media = msg.video or msg.voice or msg.audio or msg.document
    if media:
        await msg.reply_text("📥 Скачиваю файл…")
        tg_file = await context.bot.get_file(media.file_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.bin")
            await tg_file.download_to_drive(src)
            try:
                ensure_min_size(src, MIN_BYTES)
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
                await msg.reply_text("🤖 Распознаю (локально, faster-whisper)…")
                text = transcribe_wav_chunked(wav)
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

    # 2) Ссылка (в тексте/подписи)
    text_in = (msg.text or "") + " " + (msg.caption or "")
    m = re.search(r"(https?://\S+)", text_in)
    if m:
        link = m.group(1)
        passcode = extract_passcode(text_in)  # для Zoom, если прислан
        await msg.reply_text("🌐 Скачиваю файл по ссылке…")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "download.bin")
            try:
                download_from_link(link, src, maybe_passcode=passcode)
                ensure_min_size(src, MIN_BYTES)
            except Exception as e:
                await msg.reply_text(
                    "❌ Ошибка загрузки: " + str(e) +
                    "\n\nПодсказки:\n• Для Zoom пришлите ссылку на запись и пароль в сообщении, например:\n"
                    "  pwd: ABCD1234  или  Секретный код: ABCD1234\n"
                    "• Убедитесь, что у записи включено «Allow viewers to download».\n"
                    "• Для Google Drive включите доступ: «Любой у кого есть ссылка: Просмотр»."
                )
                return

            await msg.reply_text("🎙 Извлекаю аудио (ffmpeg)…")
            wav = os.path.join(tmpdir, "audio.wav")
            try:
                extract_audio_to_wav16k_mono(src, wav)
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка при конвертации: {e}")
                return

            try:
                await msg.reply_text("🤖 Распознаю (локально, faster-whisper)…")
                text_out = transcribe_wav_chunked(wav)
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
    await msg.reply_text("ℹ️ Пришлите аудио/видео ВЛОЖЕНИЕМ или ссылку (Google Drive/Nextcloud/Zoom+пароль). Я верну .txt с текстом.")

# =============== ЗАПУСК ===============
def main():
    log.info("Запуск бота… ASR_PROVIDER=%s, WHISPER_MODEL=%s, compute=%s, language=%s, chunk=%ss",
             ASR_PROVIDER, WHISPER_MODEL, WHISPER_COMPUTE_TYPE, LANGUAGE, CHUNK_SECONDS)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    log.info("✅ Бот запущен. Ожидание сообщений…")
    app.run_polling()

if __name__ == "__main__":
    main()
