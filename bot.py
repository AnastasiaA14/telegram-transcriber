import os
import re
import html
import logging
import tempfile
import subprocess
import urllib.parse
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

ASR_PROVIDER = os.getenv("ASR_PROVIDER", "local").lower()  # 'local' по умолчанию
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")        # tiny/base/small/medium
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 | int8_float16 | float32
LANGUAGE = os.getenv("LANGUAGE", "ru")  # 'ru' стабильно для русской речи

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ASR_DIARIZE = os.getenv("ASR_DIARIZE", "false").lower() in ("1", "true", "yes")

# ===== УТИЛИТЫ =====
def ensure_min_size(path: str, min_bytes: int = 10_000) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        raise RuntimeError(f"Файл слишком маленький или не докачался (< {min_bytes // 1000} КБ).")

def run_ffmpeg(cmd: list) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="ignore")[-1500:]
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

def normalize_link(url: str) -> str:
    url = url.strip()
    # Nextcloud: добавить /download
    if "/s/" in url and "download" not in url:
        if not url.endswith("/download"):
            url = url.rstrip("/") + "/download"
    # Google Drive нормализуем НО настоящую загрузку сделаем в отдельной функции
    m = re.search(r"drive\.google\.com/.*/file/d/([^/]+)/", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    m = re.search(r"drive\.google\.com/.*[?&]id=([^&]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return url

# ====== ZOOM ======
ZOOM_HOST_RE = re.compile(r"https?://([\w\-]+\.)?zoom\.us/rec/", re.IGNORECASE)

PASS_PATTERNS = [
    r"(?:\b|^)(?:pwd|passcode|пароль|секретный\s*код)\s*[:：]\s*([A-Za-z0-9_\-\.\$\^\@\!\&\*]+)",
    r"Секретный\s*код\W*([A-Za-z0-9_\-\.\$\^\@\!\&\*]+)",
]

def extract_passcode(text: str) -> str | None:
    if not text:
        return None
    for pat in PASS_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def download_zoom_recording(share_url: str, passcode: str | None, dest_path: str) -> None:
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
                    f.write(chunk)
                    total += len(chunk)
        if total < 10_000:
            raise RuntimeError("Zoom скачал слишком маленький файл (<10 КБ). Возможно, неверный пароль/доступ.")

# ===== Google Drive большие файлы =====
DRIVE_UC_RE = re.compile(r"^https?://drive\.google\.com/uc\?", re.I)

def _drive_extract_id(url: str) -> str | None:
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"drive\.google\.com/.*/file/d/([^/]+)/", url)
    if m:
        return m.group(1)
    return None

def _drive_download_with_confirm(session: requests.Session, url: str, file_id: str, dest_path: str) -> None:
    """
    1) Первый запрос к uc?export=download&id=... — может вернуть HTML с предупреждением.
    2) Ищем confirm-токен (в ссылке или cookies).
    3) Повторяем запрос с confirm=... и стягиваем файл потоково.
    """
    # Первый запрос
    resp = session.get(url, stream=True, timeout=300, allow_redirects=True)
    cdisp = resp.headers.get("Content-Disposition")
    ctype = (resp.headers.get("Content-Type") or "").lower()

    # Если сразу отдали файл (есть Content-Disposition и не text/html) — качаем
    if (cdisp and "attachment" in cdisp.lower()) and ("html" not in ctype and not ctype.startswith("text/")):
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return

    # Иначе это HTML-страница предупреждения. Ищем confirm
    text = resp.text
    # 1) confirm в ссылке
    m = re.search(r'href="[^"]*?confirm=([0-9A-Za-z_\-]+)[^"]*?&id=' + re.escape(file_id), text)
    token = m.group(1) if m else None
    # 2) или в cookies
    if not token:
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
    if not token:
        raise RuntimeError("Google Drive требует подтверждение (confirm), токен не найден. Проверьте доступ и попробуйте снова.")

    # Повторный запрос с confirm
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    qs["confirm"] = [token]
    new_qs = urllib.parse.urlencode({k: v[0] for k, v in qs.items()})
    url2 = urllib.parse.urlunparse(parsed._replace(query=new_qs))

    resp2 = session.get(url2, stream=True, timeout=600, allow_redirects=True)
    ctype2 = (resp2.headers.get("Content-Type") or "").lower()
    if "html" in ctype2 or ctype2.startswith("text/"):
        raise RuntimeError("Google Drive снова вернул HTML. Проверьте, открыт ли доступ по ссылке («Любой у кого есть ссылка»).")
    with open(dest_path, "wb") as f:
        for chunk in resp2.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def download_from_link(link: str, dest_path: str, maybe_passcode: str | None = None) -> None:
    link = normalize_link(link)

    # Ветвь Zoom
    if ZOOM_HOST_RE.search(link):
        if not maybe_passcode:
            raise RuntimeError(
                "Zoom требует пароль. Пришлите ссылку и пароль в одном сообщении, например:\n"
                "pwd: ABCD1234  или  Секретный код: ABCD1234"
            )
        download_zoom_recording(link, maybe_passcode, dest_path)
        return

    # Google Drive (uc?export=download&id=...)
    if "drive.google.com" in link:
        file_id = _drive_extract_id(link)
        if not file_id:
            raise RuntimeError("Не удалось извлечь ID файла Google Drive. Пришлите обычную ссылку «Поделиться» на файл.")
        uc_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        _drive_download_with_confirm(session, uc_url, file_id, dest_path)
        if os.path.getsize(dest_path) < 10_000:
            raise RuntimeError("Google Drive отдал слишком маленький файл (<10 КБ). Проверьте доступ по ссылке.")
        return

    # Обычные прямые ссылки
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(link, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" in ctype or ctype.startswith("text/"):
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

    # 1) Вложение (видео/аудио/документ)
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

    # 2) Ссылка (в тексте/подписи)
    text = (msg.text or "") + " " + (msg.caption or "")
    m = re.search(r"(https?://\S+)", text)
    if m:
        link = m.group(1)
        passcode = extract_passcode(text)  # пароль для Zoom, если прислан
        await msg.reply_text("🌐 Скачиваю файл по ссылке…")
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "download.bin")
            try:
                download_from_link(link, src, maybe_passcode=passcode)
                ensure_min_size(src, 10_000)
            except Exception as e:
                await msg.reply_text(
                    "❌ Ошибка загрузки: " + str(e) +
                    "\n\nПодсказки:\n• Для Zoom пришлите ссылку на запись и пароль в сообщении, например:\n"
                    "  pwd: ABCD1234  или  Секретный код: ABCD1234\n"
                    "• Убедитесь, что у записи включено «Allow viewers to download»."
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
    await msg.reply_text("ℹ️ Пришлите аудио/видео вложением ИЛИ ссылку (Google Drive/Nextcloud/Zoom+пароль).")

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
