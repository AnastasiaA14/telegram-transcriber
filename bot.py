import os 
import tempfile
import logging
import datetime
import requests
import time
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import whisper

TELEGRAM_TOKEN = "7557009279:AAH9htYr2GVzCH9u5f9kGxgaoqrOSN0xkNQ"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

model = whisper.load_model("medium")

def extract_direct_download_link(link: str) -> str:
    if "drive.google.com" in link and "/file/d/" in link:
        file_id = link.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return link

def is_audio_file(file_path):
    return any(file_path.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'])

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    media = update.message.video or update.message.voice or update.message.audio or update.message.document
    if not media:
        await update.message.reply_text("Это не поддерживаемый тип файла.")
        return

    file = await context.bot.get_file(media.file_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input")
        audio_path = os.path.join(tmpdir, "audio.wav")

        await file.download_to_drive(input_path)
        await update.message.reply_text("⏳ Файл скачан. Обрабатываем...")

        try:
            if is_audio_file(input_path):
                audio = AudioSegment.from_file(input_path)
                audio.export(audio_path, format="wav")
            else:
                for _ in range(10):
                    try:
                        clip = VideoFileClip(input_path)
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise Exception("Файл занят другим процессом или повреждён.")
                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при извлечении аудио: {e}")
            return

        await update.message.reply_text("⏳ Распознаем...")

        try:
            result = model.transcribe(audio_path, fp16=False, verbose=False, language='ru')
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при распознавании: {e}")
            return

        text = result.get("text", "")
        if not text.strip():
            await update.message.reply_text("Не удалось распознать речь.")
            return

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        txt_path = os.path.join(tmpdir, f"transcript_{now}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        with open(txt_path, "rb") as f:
            await update.message.reply_document(document=InputFile(f, filename=os.path.basename(txt_path)))
        await update.message.reply_text("✅ Готово! Текст отправлен.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or update.message.caption or ""
    link = next((word for word in text.split() if word.startswith("http")), None)

    if not link:
        await update.message.reply_text("ℹ️ Отправьте ссылку на файл или загрузите видео/аудио напрямую.")
        return

    await update.message.reply_text("⏳ Загружаю файл по ссылке...")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "downloaded.mp4")
            audio_path = os.path.join(tmpdir, "audio.wav")

            link = extract_direct_download_link(link)
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(link, headers=headers, stream=True)

            content_type = response.headers.get("Content-Type", "")
            if "html" in content_type or response.status_code != 200:
                raise Exception("Скачался HTML, а не видео. Возможно, доступ к файлу ограничен.")

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(local_path) < 1_000_000:
                raise Exception("Файл слишком маленький. Возможно, он не был полностью скачан.")

            if is_audio_file(local_path):
                audio = AudioSegment.from_file(local_path)
                audio.export(audio_path, format="wav")
            else:
                for _ in range(10):
                    try:
                        clip = VideoFileClip(local_path)
                        break
                    except OSError:
                        time.sleep(1)
                else:
                    raise Exception("MoviePy не может открыть файл. Он занят другим процессом или повреждён.")
                clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

            await update.message.reply_text("⏳ Распознаем...")

            result = model.transcribe(audio_path, fp16=False, verbose=False, language='ru')
            text = result.get("text", "")

            if not text.strip():
                await update.message.reply_text("Не удалось распознать речь.")
                return

            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            txt_path = os.path.join(tmpdir, f"transcript_{now}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            with open(txt_path, "rb") as f:
                await update.message.reply_document(document=InputFile(f, filename=os.path.basename(txt_path)))
            await update.message.reply_text("✅ Готово! Текст отправлен.")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при обработке ссылки: {e}")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.VIDEO | filters.VOICE | filters.AUDIO | filters.Document.VIDEO | filters.Document.AUDIO, handle_media))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("📡 Бот запущен")
    app.run_polling()





