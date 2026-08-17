# ==============================================================================
# AGSoft_Load_Video.py
# ==============================================================================
# Нода: 🎬AGSoft Load Video
# Описание / Description:
# Загружает видеофайл с гибкими вариантами ввода и возвращает объект VIDEO,
# абсолютный путь к файлу и метаданные: ширину, высоту и длительность.
# Есть встроенный превью-плеер и кнопка загрузки (через флаг video_upload +
# JS-расширение из web/).
#
# Порядок приоритета:
# 1. внешний вход input_video
# 2. кастомный путь custom_path
# 3. файл из папки input
#
# При внешнем входе путь извлекается из VIDEO-объекта несколькими fallback-
# способами (VideoFromFile, атрибуты пути, get_path, get_stream_source).
#
# Загрузка файлов — ОДНИМ запросом /agsoft/upload: сервер пишет multipart-
# поток на диск чанками (без буферизации в память), прогресс на кнопке через
# XHR upload progress.
#
# MP4/MOV/WEBM отдаются как есть через /agsoft/stream (Range, перемотка).
# Для MKV/AVI/TS — живой поток /agsoft/preview БЕЗ КЭША:
# - старт с 0: видео copy + звук AAC (минимум CPU);
# - перемотка (start>0): точный seek с транскодом видео на лету
#   (libx264 ultrafast) — видео и звук стартуют с одной точки, разсинхрона
#   нет, перемотка точная; -async 1 страхует от дрейфа на длинных файлах;
# - убитые при перемотке ffmpeg-процессы корректно дожидаются (kill + wait),
#   чтобы в консоли не сыпались "I/O operation on closed pipe" (Windows).
#
# Loads a video file with flexible input options and returns the VIDEO
# object, its absolute file path and metadata: width, height and duration.
# Includes a player preview and upload button (via video_upload flag +
# web/ JS extension).
#
# Priority order:
# 1. external input_video
# 2. custom_path
# 3. file from input directory
#
# With an external input, the path is extracted from the VIDEO object using
# multiple fallback approaches.
#
# Uploads go through a SINGLE request /agsoft/upload: the server writes the
# multipart stream to disk in chunks (no memory buffering), progress shown on
# the button via XHR upload progress.
#
# MP4/MOV/WEBM are served as-is via /agsoft/stream (Range, seeking).
# For MKV/AVI/TS a live stream /agsoft/preview is used with NO CACHE:
# - start at 0: video copy + AAC audio (minimal CPU);
# - seeking (start>0): accurate seek with on-the-fly video transcode
#   (libx264 ultrafast) — video and audio start from the same point, no
#   desync, frame-accurate seeking; -async 1 prevents drift on long files;
# - ffmpeg processes killed on re-seek are properly awaited (kill + wait) so
#   "I/O operation on closed pipe" (Windows proactor) no longer spam.
#
# Возможности / Features:
# ⚡ Выходы: video / video_path / width / height / duration.
#   Outputs: video / video_path / width / height / duration.
# ⚡ Три источника с приоритетом: input_video / custom_path / video.
#   Three sources with priority: input_video / custom_path / video.
# ⚡ Превью-плеер и кнопка загрузки (video_upload=True + web/JS).
#   Player preview and upload button (video_upload=True + web/ JS).
# ⚡ Загрузка одним запросом (streaming multipart) с прогрессом.
#   Single-request streaming upload with progress.
# ⚡ Drag&drop видеофайла из проводника прямо на ноду (в JS-расширении).
#   Drag&drop a video file from the OS explorer onto the node (in JS ext).
# ⚡ МГНОВЕННАЯ и ТОЧНАЯ перемотка для MKV/AVI/TS БЕЗ кэша, без разсинхрона.
#   INSTANT and ACCURATE seeking for MKV/AVI/TS with NO cache, no desync.
# ⚡ Метаданные через ffmpeg (без декодирования), кэш только в RAM на сервере.
#   Metadata via ffmpeg (no decoding), cached in RAM server-side only.
# ⚡ Любой видео-формат: MP4, AVI, MOV, WEBM, MKV и др.
#   Any video format: MP4, AVI, MOV, WEBM, MKV and more.
# ⚡ IS_CHANGED для корректного кэширования, VALIDATE_INPUTS для валидации.
#   IS_CHANGED for proper caching, VALIDATE_INPUTS for validation.
#
# Автор / Author: AGSoft
# Дата / Date: 16.08.2026
# ==============================================================================

import os
import re
import asyncio
import logging
import mimetypes
import subprocess

import folder_paths
from aiohttp import web
from server import PromptServer

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

try:
    from comfy_api.input_impl import VideoFromFile
    from comfy.comfy_types import IO
    HAS_NEW_API = True
except ImportError:
    VideoFromFile = None

    class IO:
        VIDEO = "VIDEO"

    HAS_NEW_API = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Маркер версии: если этой строки нет в консоли после старта — файл не применился.
# Version marker: if this line is not in console after startup — file was not applied.
# print("[AGSoft Load Video] v16.14 loaded (single upload + stream + instant ACCURATE seek NO cache + synced audio + clean subprocess kill + metadata outputs + drag&drop + resizable player)")

# Кэш метаданных видео (width/height/duration/codec) — только в RAM.
# Video metadata cache (width/height/duration/codec) — RAM only.
_VIDEO_INFO_CACHE = {}

# Пресет транскода видео при перемотке (только для MKV/AVI/TS, start>0).
# "ultrafast" = минимум CPU (больше битрейт, но это локальная труба).
# Если качество превью после прыжка важнее CPU — поставьте "veryfast".
#
# Video transcode preset used when seeking (MKV/AVI/TS, start>0 only).
# "ultrafast" = minimal CPU (higher bitrate, but it's a local pipe).
# If post-seek preview quality matters more than CPU — use "veryfast".
SEEK_PRESET = "ultrafast"


# ------------------------------------------------------------------------------
# Загрузка файла ОДНИМ запросом: сервер пишет multipart-поток на диск чанками,
# без буферизации в память и без чанковой логики на стороне JS.
# Single-request upload: the server writes the multipart stream to disk in
# chunks, no memory buffering and no chunk logic on the JS side.
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.post("/agsoft/upload")
async def agsoft_upload(request):
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "No multipart body"}, status=400)

        saved_name = None

        field = await reader.next()
        while field is not None:
            if field.name == "file" and saved_name is None:
                safe_filename = os.path.basename(field.filename or "")

                if not safe_filename:
                    return web.json_response(
                        {"error": "No filename provided"}, status=400
                    )

                file_path = os.path.join(
                    folder_paths.get_input_directory(), safe_filename
                )

                # Пишем поток на диск кусками по 1 МБ — память не растёт.
                # Write the stream to disk in 1 MB pieces — memory stays flat.
                with open(file_path, "wb") as f:
                    while True:
                        chunk = await field.read_chunk(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

                saved_name = safe_filename

            field = await reader.next()

        if saved_name is None:
            return web.json_response({"error": "No file field in upload"}, status=400)

        return web.json_response({"status": "success", "name": saved_name})

    except Exception as e:
        print(f"[AGSoft Load Video] upload error: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ------------------------------------------------------------------------------
# Универсальная потоковая отдача файла с поддержкой Range.
# Используется для оригинальных файлов (MP4/MOV/WEBM — с перемоткой).
#
# Universal file streaming helper with Range support.
# Used for original files (MP4/MOV/WEBM — with seeking).
# ------------------------------------------------------------------------------
async def _agsoft_stream_file(request, path, mime=None):
    if not os.path.isfile(path):
        return web.Response(status=404)

    file_size = os.path.getsize(path)

    if not mime:
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "application/octet-stream"

    if file_size == 0:
        return web.Response(status=200, content_type=mime)

    start = 0
    end = file_size - 1
    status = 200

    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not m:
            return web.Response(
                status=416,
                headers={"Content-Range": f"bytes */{file_size}"}
            )

        first, last = m.group(1), m.group(2)

        if first == "" and last == "":
            return web.Response(
                status=416,
                headers={"Content-Range": f"bytes */{file_size}"}
            )

        # suffix range, например bytes=-500
        # suffix range, e.g. bytes=-500
        if first == "" and last != "":
            suffix = int(last)
            start = max(0, file_size - suffix)
            end = file_size - 1
        else:
            start = int(first)
            end = int(last) if last != "" else file_size - 1

        if start >= file_size or start > end:
            return web.Response(
                status=416,
                headers={"Content-Range": f"bytes */{file_size}"}
            )

        end = min(end, file_size - 1)
        status = 206

    count = end - start + 1

    resp = web.StreamResponse(status=status)
    resp.content_type = mime
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(count)

    if status == 206:
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    await resp.prepare(request)

    CHUNK = 1024 * 1024

    try:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = count
            while remaining > 0:
                chunk = f.read(min(CHUNK, remaining))
                if not chunk:
                    break
                await resp.write(chunk)
                remaining -= len(chunk)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError, OSError):
        pass

    try:
        await resp.write_eof()
    except Exception:
        pass

    return resp


# ------------------------------------------------------------------------------
# Потоковая отдача оригинальных видеофайлов из папки input.
# Streaming of original video files from input directory.
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.get("/agsoft/stream")
async def agsoft_stream(request):
    filename = request.query.get("filename", "")
    if not filename:
        return web.Response(status=400)

    safe = os.path.basename(filename)
    path = os.path.join(folder_paths.get_input_directory(), safe)

    return await _agsoft_stream_file(request, path)


# ------------------------------------------------------------------------------
# Корректная остановка ffmpeg-процесса.
# ВАЖНО для Windows: после kill() ОБЯЗАТЕЛЬНО await wait(), иначе транспорт
# subprocess'а уничтожается сборщиком мусора и в консоль сыпется:
#   Exception ignored in: <function _ProactorBasePipeTransport.__del__ ...>
#   ValueError: I/O operation on closed pipe
#
# Proper ffmpeg process shutdown.
# IMPORTANT on Windows: after kill() you MUST await wait(), otherwise the
# subprocess transport is destroyed by GC and the console gets spammed with:
#   Exception ignored in: <function _ProactorBasePipeTransport.__del__ ...>
#   ValueError: I/O operation on closed pipe
# ------------------------------------------------------------------------------
async def _stop_ffmpeg(proc):
    try:
        if proc.returncode is None:
            proc.kill()
    except Exception:
        pass

    try:
        await proc.wait()
    except Exception:
        pass


# ------------------------------------------------------------------------------
# ЖИВОЙ поток для контейнеров с "несъедобным" для браузера звуком (MKV/AVI/TS)
# + МГНОВЕННАЯ и ТОЧНАЯ перемотка БЕЗ кэша и БЕЗ разсинхрона.
#
# start = 0  : видео copy + звук AAC (минимум CPU).
# start > 0  : точный seek; видео транскодится на лету (libx264 ultrafast),
#              чтобы видео и звук стартовали с ОДНОЙ точки. При -c:v copy
#              видео начиналось бы с предыдущего ключевого кадра, а звук —
#              точно с запрошенной секунды → разсинхрон на величину GOP.
# -async 1   : страховка от дрейфа звука на длинных файлах.
#
# LIVE stream for containers with browser-incompatible audio (MKV/AVI/TS)
# + INSTANT and ACCURATE seeking with NO cache and NO desync.
#
# start = 0  : video copy + AAC audio (minimal CPU).
# start > 0  : accurate seek; video is transcoded on the fly (libx264
#              ultrafast) so video and audio start from the SAME point.
#              With -c:v copy the video would start at the previous keyframe
#              while audio starts exactly at the requested second → desync
#              by the GOP size.
# -async 1   : safety net against audio drift on long files.
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.get("/agsoft/preview")
async def agsoft_preview(request):
    filename = request.query.get("filename", "")
    if not filename:
        return web.Response(status=400)

    safe = os.path.basename(filename)
    path = os.path.join(folder_paths.get_input_directory(), safe)

    if not os.path.isfile(path):
        return web.Response(status=404)

    try:
        start = float(request.query.get("start", "0") or 0)
    except ValueError:
        start = 0.0

    if start < 0:
        start = 0.0

    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error"]

    # -ss ДО -i = быстрый seek по индексу контейнера.
    # -ss BEFORE -i = fast index-based seek.
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]

    cmd += ["-i", path, "-map", "0:v:0", "-map", "0:a:0?"]

    if start > 0:
        # Точная перемотка: транскод видео, синхронно со звуком.
        # Accurate seeking: video transcode, in sync with audio.
        cmd += [
            "-c:v", "libx264",
            "-preset", SEEK_PRESET,
            "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
    else:
        # Старт с нуля: видео копируем без перекодирования.
        # Start from zero: copy video without re-encoding.
        cmd += ["-c:v", "copy"]

    cmd += [
        "-c:a", "aac", "-b:a", "128k",
        "-async", "1",
        "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-"
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    resp = web.StreamResponse(status=200)
    resp.content_type = "video/mp4"
    resp.headers["Accept-Ranges"] = "none"

    await resp.prepare(request)

    try:
        while True:
            chunk = await proc.stdout.read(256 * 1024)
            if not chunk:
                break
            await resp.write(chunk)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError, OSError):
        pass
    finally:
        # Браузер закрыл соединение (новая перемотка/смена файла) —
        # убиваем ffmpeg и ОБЯЗАТЕЛЬНО дожидаемся (см. _stop_ffmpeg).
        # Browser closed the connection (new seek / file change) —
        # kill ffmpeg and ALWAYS wait for it (see _stop_ffmpeg).
        await _stop_ffmpeg(proc)

    try:
        await resp.write_eof()
    except Exception:
        pass

    return resp


# ------------------------------------------------------------------------------
# Метаданные видео (width/height/duration/codec) из stderr ffmpeg.
# БЫСТРО: без декодирования, ffmpeg только печатает информацию о файле.
# Результат кэшируется в RAM по (путь, mtime, размер). На диск НЕ пишется.
#
# ВАЖНО про парсинг: строка видеопотока выглядит так:
#   Stream #0:0(eng): Video: h264 (High), yuv420p(tv, bt709), 1280x720 [SAR...
# Разрешение идёт ПОСЛЕ нескольких запятых, поэтому ищем NxM по ВСЕЙ строке
# видеопотока, а не сразу после "Video:".
#
# Video metadata (width/height/duration/codec) from ffmpeg stderr.
# FAST: no decoding, ffmpeg only prints file info.
# Result cached in RAM by (path, mtime, size). NOTHING written to disk.
#
# IMPORTANT about parsing: the video stream line looks like:
#   Stream #0:0(eng): Video: h264 (High), yuv420p(tv, bt709), 1280x720 [SAR...
# The resolution comes AFTER several commas, so we search NxM over the WHOLE
# video stream line, not right after "Video:".
# ------------------------------------------------------------------------------
def _get_video_info(path):
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        key = (path, 0, 0)

    cached = _VIDEO_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    info = {"width": 0, "height": 0, "duration": 0.0, "codec": ""}

    try:
        # Без выхода ffmpeg завершится с ошибкой, но информацию напечатает.
        # Without an output ffmpeg exits with an error but still prints info.
        proc = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-i", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20,
        )
        err = proc.stderr or ""

        # Длительность / Duration: "Duration: 01:23:45.67, ..."
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if m:
            info["duration"] = (
                int(m.group(1)) * 3600 +
                int(m.group(2)) * 60 +
                float(m.group(3))
            )

        # Видеопоток: берём ВСЮ строку "Stream #...: Video: ...".
        # Предпочитаем первую строку, где есть разрешение NxM
        # (чтобы пропустить всякие attached pic / обложки без размера).
        #
        # Video stream: take the WHOLE "Stream #...: Video: ..." line.
        # Prefer the first line containing an NxM resolution
        # (to skip attached pics / covers without a size).
        video_line = None

        for lm in re.finditer(r"Stream\s*#\d+:\d+[^\n]*Video:[^\n]*", err):
            if re.search(r"\d{2,5}x\d{2,5}", lm.group(0)):
                video_line = lm.group(0)
                break

        if video_line is None:
            lm = re.search(r"Stream\s*#\d+:\d+[^\n]*Video:[^\n]*", err)
            if lm:
                video_line = lm.group(0)

        if video_line:
            # Кодек — сразу после "Video:" и до первой запятой.
            # Codec — right after "Video:" up to the first comma.
            cm = re.search(r"Video:\s*([^,\n]+)", video_line)
            if cm:
                info["codec"] = cm.group(1).strip()

            # Разрешение — по ВСЕЙ строке видеопотока.
            # Resolution — over the WHOLE video stream line.
            gm = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
            if gm:
                info["width"] = int(gm.group(1))
                info["height"] = int(gm.group(2))

    except Exception as e:
        logger.warning(f"[AGSoft Load Video] metadata probe failed: {e}")

    if len(_VIDEO_INFO_CACHE) > 128:
        _VIDEO_INFO_CACHE.clear()

    _VIDEO_INFO_CACHE[key] = info

    return info


@PromptServer.instance.routes.get("/agsoft/video_info")
async def agsoft_video_info(request):
    """
    Endpoint для JS: метаданные файла (длительность нужна полосе перемотки).
    JS endpoint: file metadata (duration is needed by the seek bar).
    """
    filename = request.query.get("filename", "")

    if not filename:
        return web.json_response({"error": "filename required"}, status=400)

    safe = os.path.basename(filename)
    path = os.path.join(folder_paths.get_input_directory(), safe)

    if not os.path.isfile(path):
        return web.json_response({"error": "file not found"}, status=404)

    # ffmpeg run — в отдельный поток, чтобы не блокировать event loop.
    # ffmpeg run — in a separate thread to avoid blocking the event loop.
    info = await asyncio.to_thread(_get_video_info, path)

    return web.json_response(info)


def _list_video_files():
    """
    Список видеофайлов из папки input (безопасно на любой версии ComfyUI).
    List of video files from input directory (safe on any ComfyUI version).
    """
    try:
        input_dir = folder_paths.get_input_directory()
        files = []

        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                full_path = os.path.join(input_dir, f)
                if os.path.isfile(full_path):
                    files.append(f)

        return folder_paths.filter_files_content_types(files, ["video"])
    except Exception:
        return []


def _extract_video_path(video_obj):
    """
    Извлекает путь из VIDEO-объекта, учитывая особенности ComfyUI.
    Extracts path from VIDEO object, accounting for ComfyUI specifics.
    """
    if video_obj is None:
        return ""

    # Способ 1: VideoFromFile с name mangling (приватный атрибут __file).
    # Method 1: VideoFromFile with name mangling (private __file attribute).
    try:
        if VideoFromFile is not None and isinstance(video_obj, VideoFromFile):
            path = getattr(video_obj, "_VideoFromFile__file", None)
            if path and isinstance(path, str) and os.path.exists(path):
                return os.path.abspath(path)
    except Exception:
        pass

    # Способ 2: стандартные атрибуты пути.
    # Method 2: standard path attributes.
    path_attrs = [
        "path",
        "_path",
        "filepath",
        "file_path",
        "source",
        "filename",
        "video_path",
        "_VideoFromFile__file",
    ]

    for attr in path_attrs:
        try:
            if hasattr(video_obj, attr):
                value = getattr(video_obj, attr)
                if isinstance(value, str) and os.path.exists(value):
                    return os.path.abspath(value)
        except Exception:
            continue

    # Способ 3: метод get_path().
    # Method 3: get_path() method.
    try:
        if hasattr(video_obj, "get_path") and callable(video_obj.get_path):
            path = video_obj.get_path()
            if isinstance(path, str) and os.path.exists(path):
                return os.path.abspath(path)
    except Exception:
        pass

    # Способ 4: метод get_stream_source().
    # Method 4: get_stream_source() method.
    try:
        if hasattr(video_obj, "get_stream_source") and callable(video_obj.get_stream_source):
            source = video_obj.get_stream_source()
            if isinstance(source, str) and os.path.exists(source):
                return os.path.abspath(source)
    except Exception:
        pass

    return ""


class AGSoftLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        video_files = _list_video_files()

        # Гарантируем непустой список — иначе фронтенд не создаст combo-виджет.
        # Guarantee non-empty list — otherwise frontend may not create combo widget.
        if not video_files:
            video_files = [" "]

        return {
            "required": {},
            "optional": {
                "input_video": (
                    IO.VIDEO,
                    {
                        "tooltip": (
                            "Optional: accept video from another node (highest priority).  "
                            "If connected, other inputs are ignored. The file path is extracted  "
                            "automatically from the VIDEO object.\n"
                            "---\n"
                            "Опционально: принимает видео от другой ноды (наивысший приоритет).  "
                            "Если подключён, другие входы игнорируются. Путь к файлу извлекается  "
                            "автоматически из объекта VIDEO."
                        )
                    }
                ),
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional: absolute path to a video file. Overrides the dropdown  "
                            "if input_video is not connected. Always use absolute paths.\n"
                            "Example (Windows): C:/videos/my_video.mp4\n"
                            "Example (Linux/Mac): /home/user/videos/my_video.mov\n"
                            "---\n"
                            "Опционально: абсолютный путь к видеофайлу. Переопределяет список,  "
                            "если не подключён внешний вход. Всегда используйте абсолютные пути.\n"
                            "Пример (Windows): C:/videos/my_video.mp4\n"
                            "Пример (Linux/Mac): /home/user/videos/my_video.mov"
                        )
                    }
                ),
                # Комбо с video_upload=True: фронтенд добавляет превью-плеер
                # и кнопку загрузки (как image_upload в Image Resize Plus).
                # Файл также можно просто перетащить из проводника на ноду.
                #
                # Combo with video_upload=True: frontend adds preview player
                # and upload button (like image_upload in Image Resize Plus).
                # A file can also simply be dragged from the explorer onto the node.
                "video": (
                    video_files,
                    {
                        "video_upload": True,
                        "tooltip": (
                            "Select / upload a video file from the input directory.  "
                            "Used when no external video and no custom path are set.  "
                            "Files are uploaded in a single streaming request with progress.  "
                            "You can also drag&drop a video file from the OS explorer onto the node.\n"
                            "Supported formats: MP4, AVI, MOV, WEBM, MKV and other common formats.\n"
                            "---\n"
                            "Выберите / загрузите видеофайл из папки input.  "
                            "Используется, когда не подключён внешний вход и не указан кастомный путь.  "
                            "Файлы загружаются одним потоковым запросом с прогрессом.  "
                            "Также можно просто перетащить видеофайл из проводника прямо на ноду.\n"
                            "Поддерживаемые форматы: MP4, AVI, MOV, WEBM, MKV и другие распространённые."
                        )
                    }
                ),
            },
        }

    # video / video_path / width / height / duration
    RETURN_TYPES = (IO.VIDEO, "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("video", "video_path", "width", "height", "duration")
    FUNCTION = "load_video"
    CATEGORY = "AGSoft/Video"

    # JS из web/ дорисовывает превью-плеер, свою полосу перемотки, кнопку
    # загрузки, строку информации и drag&drop.
    # JS from web/ draws preview player, custom seek bar, upload button,
    # info line and drag&drop.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬 AGSoft Load Video.\n"
        "Loads a video file with flexible input options and returns the VIDEO object, its "
        "absolute file path and metadata: width, height and duration.\n"
        "Includes a built-in player preview and an upload button.\n"
        "Priority order: external video input > custom path > file from input directory.\n"
        "With an external VIDEO input, the file path is extracted automatically from the VIDEO "
        "object using multiple fallback approaches (VideoFromFile, path attributes, get_path, "
        "get_stream_source).\n"
        "Files are uploaded in a single streaming request with progress.\n"
        "MP4/MOV/WEBM are served as-is with native seeking. MKV/AVI/TS use a live stream with "
        "AAC audio and INSTANT, ACCURATE seeking WITHOUT any cache and WITHOUT desync: start at 0 "
        "copies the video stream, every seek restarts ffmpeg with -ss and on-the-fly ultrafast "
        "transcode so audio and video stay in sync. Nothing is stored on disk or kept in memory.\n"
        "Metadata (width/height/duration) is probed via ffmpeg without decoding.\n"
        "Supported formats: MP4, AVI, MOV, WEBM, MKV and other common video formats.\n"
        "---\n"
        "🎬 AGSoft Load Video.\n"
        "Загружает видеофайл с гибкими вариантами ввода и возвращает объект VIDEO, абсолютный "
        "путь к файлу и метаданные: ширину, высоту и длительность.\n"
        "Есть встроенный превью-плеер и кнопка загрузки.\n"
        "Порядок приоритета: внешний вход > кастомный путь > файл из папки input.\n"
        "При внешнем VIDEO-входе путь извлекается автоматически из объекта VIDEO через несколько "
        "fallback-способов (VideoFromFile, атрибуты пути, get_path, get_stream_source).\n"
        "Файлы загружаются одним потоковым запросом с прогрессом.\n"
        "MP4/MOV/WEBM отдаются как есть с нативной перемоткой. MKV/AVI/TS идут живым потоком с "
        "AAC-звуком и МГНОВЕННОЙ и ТОЧНОЙ перемоткой БЕЗ кэша и БЕЗ разсинхрона: старт с 0 "
        "копирует видеопоток, каждая перемотка перезапускает ffmpeg с -ss и транскодом ultrafast "
        "на лету, чтобы звук и видео оставались синхронно. Ничего не хранится на диске и в памяти.\n"
        "Метаданные (ширина/высота/длительность) берутся через ffmpeg без декодирования.\n"
        "Поддерживаемые форматы: MP4, AVI, MOV, WEBM, MKV и другие распространённые видеоформаты."
    )

    def load_video(self, video=None, input_video=None, custom_path=""):
        try:
            # Приоритет 1: внешний вход. Извлекаем путь из VIDEO-объекта.
            # Priority 1: external input. Extract path from VIDEO object.
            if input_video is not None:
                logger.info("Using external video input")
                video_path = _extract_video_path(input_video)

                if not video_path:
                    logger.warning(
                        "Could not extract file path from external video object; "
                        "passing object through as-is."
                    )
                    return (input_video, video_path, 0, 0, 0.0)

                info = _get_video_info(video_path)
                return (
                    input_video,
                    video_path,
                    info["width"],
                    info["height"],
                    info["duration"],
                )

            # Приоритет 2: кастомный путь.
            # Priority 2: custom path.
            if custom_path and os.path.exists(custom_path):
                p = os.path.abspath(custom_path)
                logger.info(f"Using custom video path: {p}")

                video_obj = VideoFromFile(p) if VideoFromFile is not None else p
                info = _get_video_info(p)

                return (
                    video_obj,
                    p,
                    info["width"],
                    info["height"],
                    info["duration"],
                )

            # Приоритет 3: комбо / загруженный файл.
            # Priority 3: combo / uploaded file.
            if video and str(video).strip():
                p = os.path.abspath(folder_paths.get_annotated_filepath(video))
                logger.info(f"Loading video from: {p}")

                # Проверка MIME-типа (предупреждение, не ошибка).
                # MIME type check (warning, not error).
                mime_type, _ = mimetypes.guess_type(p)
                if mime_type and not mime_type.startswith("video"):
                    logger.warning(f"Selected file may not be a video (MIME: {mime_type}): {p}")

                video_obj = VideoFromFile(p) if VideoFromFile is not None else p
                info = _get_video_info(p)

                return (
                    video_obj,
                    p,
                    info["width"],
                    info["height"],
                    info["duration"],
                )

            raise ValueError(
                "[AGSoft Load Video] Не задан источник: подключите input_video, "
                "укажите custom_path или выберите файл в video."
            )

        except Exception as e:
            error_msg = f"[AGSoft Load Video] Ошибка загрузки видео: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def IS_CHANGED(cls, video=None, input_video=None, custom_path=""):
        if input_video is not None:
            try:
                path = _extract_video_path(input_video)
                if path and os.path.exists(path):
                    return os.path.getmtime(path)
            except Exception:
                pass
            return float("nan")

        try:
            if custom_path and os.path.exists(custom_path):
                return os.path.getmtime(custom_path)

            if video and str(video).strip():
                p = folder_paths.get_annotated_filepath(video)
                if os.path.exists(p):
                    return os.path.getmtime(p)
        except Exception:
            pass

        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video=None, input_video=None, custom_path="", **kwargs):
        # ВАЖНО: подключённые входы (линки) приходят сюда как None/отсутствуют —
        # реальное значение станет известно только при выполнении. Поэтому
        # здесь проверяем только РЕАЛЬНЫЕ строковые значения, а окончательная
        # проверка источника делается в load_video() при выполнении.
        #
        # IMPORTANT: connected inputs (links) arrive here as None/missing —
        # the real value is only known at execution time. So here we only
        # check REAL string values; the final source check happens in
        # load_video() at runtime.
        if custom_path:
            if not os.path.exists(custom_path):
                return f"Custom path does not exist: {custom_path}"

            if not os.path.isfile(custom_path):
                return f"Custom path is not a file: {custom_path}"

            mime_type, _ = mimetypes.guess_type(custom_path)
            if mime_type and not mime_type.startswith("video"):
                return f"Custom path is not a video file: {custom_path}"

            return True

        if video and str(video).strip():
            if not folder_paths.exists_annotated_filepath(video):
                return f"Video file not found: {video}"

        return True

NODE_CLASS_MAPPINGS = {
    "AGSoftLoadVideo": AGSoftLoadVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadVideo": "🎬AGSoft Load Video"
}