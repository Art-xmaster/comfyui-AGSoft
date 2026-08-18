# ==============================================================================
# AGSoft_Video_Save.py
# ==============================================================================
# Нода: 🎬AGSoft Video Save
# Описание / Description:
# Сохраняет видео из последовательности изображений ИЛИ конвертирует существующее
# видео (небольшой видео-конвертор).
# Сохраняет ТОЛЬКО то, что включено: видео без звука, видео со звуком, звук
# отдельным файлом, картинку — любые комбинации, без лишних файлов.
# После выполнения показывает превью результата прямо в ноде; превью
# сериализуется в воркфлоу и переживает перезагрузку страницы.
#
# РЕЖИМ NO-OP + ПРЕВЬЮ (СО ЗВУКОМ): если ВСЕ опции сохранения выключены, нода
# НЕ пишет файлов в output и НЕ падает с ошибкой — генерация выше по графу
# выполняется, а превью источника показывается, включая звук со входа audio:
# - файл + audio: быстрый ремукс (видео copy + AAC) во временный файл;
# - файл без audio: стрим напрямую по абсолютному пути (без записи);
# - тензор + audio: энкод кадров с аудио во временный файл;
# - тензор без audio: энкод кадров во временный файл.
# NO-OP MODE + PREVIEW (WITH SOUND): all save options off = nothing written to
# output, no error, and the source preview is shown INCLUDING the audio input:
# - file + audio: fast remux (video copy + AAC) to a temp file;
# - file without audio: direct stream by absolute path (no writing);
# - tensor + audio: frame encode with audio to a temp file;
# - tensor without audio: frame encode to a temp file.
#
# Входы с приоритетом:
# 1. video (объект VIDEO) — конвертация существующего видео
# 2. video_path (путь к файлу) — конвертация видео по пути
# 3. images (тензор кадров) — сохранение из последовательности кадров
#
# Звук для save_video_with_audio / save_audio берётся из:
# 1. входа audio (если подключён) — наивысший приоритет;
# 2. звуковой дорожки САМОГО исходного видео (если дорожка есть).
#
# Опции сохранения (можно включить несколько):
# - save_video: видео без звука
# - save_video_with_audio: видео со звуком
# - save_audio: звуковая дорожка отдельным файлом M4A
# - save_image: первый кадр как PNG
# - save_metadata: вшить воркфлоу (ЧИСТЫЙ граф) в метаданные файла;
#   перетаскивание сохранённого видео/PNG на канвас ВОССТАНАВЛИВАЕТ воркфлоу.
# - save_output: True = папка output, False = temp (превью без мусора в output)
#
# Путь сохранения: output (или temp) + subfolder + filename_prefix_<timestamp>.
# Пример: subfolder="MiniMax" → output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4
#
# Saves video from an image sequence OR converts an existing video (a small
# video converter). It saves ONLY what is
# enabled: video without audio, video with audio, audio as a separate file,
# first frame — any combination, no extra files. After execution the result
# preview is shown inside the node; the preview is serialized into the
# workflow and survives page reloads.
#
# NO-OP MODE + PREVIEW (WITH SOUND): all save options off = nothing written to
# output, no error, and the source preview is shown INCLUDING the audio input.
#
# Inputs with priority:
# 1. video (VIDEO object) — convert existing video
# 2. video_path (file path) — convert video by path
# 3. images (frame tensor) — save from frame sequence
#
# Audio for save_video_with_audio / save_audio is taken from:
# 1. the audio input (if connected) — highest priority;
# 2. the source video's OWN audio track (if present).
#
# Save options (can enable multiple):
# - save_video: video without audio
# - save_video_with_audio: video with audio
# - save_audio: audio track as a separate M4A file
# - save_image: first frame as PNG
# - save_metadata: embed the workflow (PURE graph) into file metadata;
#   dragging the saved video/PNG onto the canvas restores the workflow.
# - save_output: True = output dir, False = temp (preview without clutter)
#
# Save path: output (or temp) + subfolder + filename_prefix_<timestamp>.
# Example: subfolder="MiniMax" → output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4
#
# Выбор кодеков: h264/h265 (CPU), h264/h265 NVENC, VP9/AV1 (WebM),
# FFV1 (MKV), ProRes (MOV), GIF, WebP. Список строится динамически по
# доступным энкодерам ffmpeg (ffmpeg -encoders).
# Codec selection: h264/h265, NVENC, VP9/AV1 WebM, FFV1 MKV,
# ProRes MOV, GIF, WebP (list built from available ffmpeg encoders).
#
# Возможности / Features:
# ⚡ Три источника с приоритетом: video / video_path / images.
#   Three sources with priority: video / video_path / images.
# ⚡ Гибкие опции: видео / видео+звук / звук отдельно / картинка / комбинации.
#   Flexible options: video / video+audio / audio only / image / combinations.
# ⚡ NO-OP с превью СО ЗВУКОМ: вход audio подмешивается в превью
#   (ремукс/энкод в temp), output не трогается, без ошибки.
#   NO-OP preview WITH SOUND: the audio input is muxed into the preview
#   (remux/encode to temp), output untouched, no error.
# ⚡ Звук берётся из исходного видео автоматически (или со входа audio).
#   Audio is taken from the source video automatically (or the audio input).
# ⚡ save_metadata: воркфлоу вшивается ЧИСТЫМ графом и восстанавливается drag&drop.
#   save_metadata: workflow stored as PURE graph and restored by drag&drop.
# ⚡ Подпапки: subfolder="MiniMax" → output/MiniMax/... (санация ".." и слэшей).
#   Subfolders: subfolder="MiniMax" → output/MiniMax/... (".." and slashes sanitized).
# ⚡ Выбор кодеков (список по доступным энкодерам ffmpeg).
#   Codec selection (list from available ffmpeg encoders).
# ⚡ Превью результата в ноде, сериализуется в воркфлоу, ресайз как в Load Video.
#   Result preview in the node, serialized into the workflow, resize like Load Video.
# ⚡ Конвертация видео напрямую через ffmpeg (без промежуточных PNG).
#   Direct ffmpeg video conversion (no intermediate PNGs).
# ⚡ save_output: output или temp — контроль, куда пишутся файлы.
#   save_output: output or temp — control where files are written.
# ⚡ РАЗРЕШАЮЩАЯ валидация: подключённые линки не ломают VALIDATE_INPUTS.
#   Permissive validation: connected links do not break VALIDATE_INPUTS.
# ⚡ OUTPUT_NODE=True, IS_CHANGED, VALIDATE_INPUTS.
#   OUTPUT_NODE=True, IS_CHANGED, VALIDATE_INPUTS.
#
# Автор / Author: AGSoft
# Дата / Date: 18.08.2026
# ==============================================================================

import os
import re
import json
import time
import zlib
import struct
import logging
import tempfile
import subprocess
import shutil
import wave
import asyncio

import folder_paths
import numpy as np
import torch
from PIL import Image

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
# print("[AGSoft Video Save] v30.08 loaded (no-op preview WITH sound + subfolder + reordered widgets + PURE-graph embed + codecs + serialized preview + LoadVideo-style resize)")

# ------------------------------------------------------------------------------
# Пресеты форматов/кодеков.
# Список в комбо строится динамически по доступным энкодерам ffmpeg.
# Format/codec presets.
# The combo list is built dynamically from available ffmpeg encoders.
# ------------------------------------------------------------------------------
FORMAT_PRESETS = {
    "video/h264-mp4":       {"ext": "mp4",  "vcodec": "libx264",    "acodec": "aac",      "pix_fmt": "yuv420p",     "mime": "video/mp4"},
    "video/h265-mp4":       {"ext": "mp4",  "vcodec": "libx265",    "acodec": "aac",      "pix_fmt": "yuv420p",     "mime": "video/mp4"},
    "video/h264-nvenc-mp4": {"ext": "mp4",  "vcodec": "h264_nvenc", "acodec": "aac",      "pix_fmt": "yuv420p",     "mime": "video/mp4"},
    "video/h265-nvenc-mp4": {"ext": "mp4",  "vcodec": "hevc_nvenc", "acodec": "aac",      "pix_fmt": "yuv420p",     "mime": "video/mp4"},
    "video/webm":           {"ext": "webm", "vcodec": "libvpx-vp9", "acodec": "libopus",  "pix_fmt": "yuv420p",     "mime": "video/webm"},
    "video/av1-webm":       {"ext": "webm", "vcodec": "libsvtav1",  "acodec": "libopus",  "pix_fmt": "yuv420p",     "mime": "video/webm"},
    "video/ffv1-mkv":       {"ext": "mkv",  "vcodec": "ffv1",       "acodec": "pcm_s16le","pix_fmt": "yuv420p",     "mime": "video/x-matroska"},
    "video/prores-mov":     {"ext": "mov",  "vcodec": "prores_ks",  "acodec": "pcm_s16le","pix_fmt": "yuv422p10le", "mime": "video/quicktime"},
    "image/gif":            {"ext": "gif",  "vcodec": "gif",        "acodec": None,       "pix_fmt": None,         "mime": "image/gif"},
    "image/webp":           {"ext": "webp", "vcodec": "libwebp",    "acodec": None,       "pix_fmt": None,         "mime": "image/webp"},
}

# Контейнеры, куда можно вшить метаданные (воркфлоу).
# Containers that can hold embedded metadata (workflow).
META_EXTS = {"mp4", "mkv", "mov", "webm"}

_ENCODERS_CACHE = None


def _get_encoders():
    """
    Множество доступных энкодеров ffmpeg (кэш на всю сессию).
    Set of available ffmpeg encoders (cached for the session).
    """
    global _ENCODERS_CACHE
    if _ENCODERS_CACHE is not None:
        return _ENCODERS_CACHE

    enc = set()
    try:
        proc = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=20,
        )
        for m in re.finditer(r"^\sV[\w.]{5}\s+([\w-]+)", proc.stdout or "", re.M):
            enc.add(m.group(1))
    except Exception as e:
        logger.warning(f"[AGSoft Video Save] encoder probe failed: {e}")

    _ENCODERS_CACHE = enc
    return enc


def _available_formats():
    """
    Список форматов для комбо: только те, чей энкодер доступен.
    Если проб не удался — отдаём весь список.
    Combo format list: only presets whose encoder is available.
    If the probe failed — return the whole list.
    """
    enc = _get_encoders()
    if not enc:
        return list(FORMAT_PRESETS.keys())

    avail = [k for k, f in FORMAT_PRESETS.items() if f["vcodec"] in enc]
    if "video/h264-mp4" not in avail:
        avail.insert(0, "video/h264-mp4")
    return avail if avail else list(FORMAT_PRESETS.keys())


def _unwrap_workflow(obj):
    """
    КЛЮЧЕВАЯ причина «пустого воркфлоу»: ComfyUI (новый фронтенд) передаёт
    extra_pnginfo как {"workflow": <граф>}, а читатели (loadGraphData /
    handleFile / чанк-парсеры) ждут ЧИСТЫЙ граф с "nodes" на верхнем уровне.
    Поэтому обёртку разворачиваем; уже чистый граф оставляем как есть.
    The KEY reason for the "empty workflow": ComfyUI (new frontend) passes
    extra_pnginfo as {"workflow": <graph>}, while readers (loadGraphData /
    handleFile / chunk parsers) expect the PURE graph with "nodes" at the top
    level. So the wrapper is unwrapped; an already-pure graph is kept as is.
    """
    if isinstance(obj, dict) and "workflow" in obj and "nodes" not in obj:
        return obj["workflow"]
    return obj


def _norm_subfolder(s):
    """
    Санация подпапки: обратные слэши → прямые, убираем пустые сегменты,
    "." и ".." (защита от выхода за output/temp). Поддерживает вложенность:
    "MiniMax" или "MiniMax/2026".
    Subfolder sanitization: backslashes → slashes, drop empty segments,
    "." and ".." (no escape outside output/temp). Supports nesting:
    "MiniMax" or "MiniMax/2026".
    """
    if not s:
        return ""
    parts = []
    for p in str(s).replace("\\", "/").split("/"):
        p = p.strip()
        if p and p not in (".", ".."):
            parts.append(p)
    return "/".join(parts)


def _vcodec_args(f, crf):
    """Аргументы видеокодера + rate control (crf или cq для NVENC)."""
    v = f["vcodec"]
    args = ["-c:v", v]

    if f.get("pix_fmt"):
        args += ["-pix_fmt", f["pix_fmt"]]

    if v in ("libx264", "libx265", "libvpx-vp9", "libsvtav1"):
        args += ["-crf", str(crf)]
    elif v in ("h264_nvenc", "hevc_nvenc"):
        args += ["-rc", "vbr", "-cq", str(crf)]

    return args


def _audio_args(f):
    """Аргументы аудиокодера контейнера (None = контейнер без звука)."""
    a = f.get("acodec")
    if a is None:
        return None
    if a in ("pcm_s16le",):
        return ["-c:a", a]
    return ["-c:a", a, "-b:a", "192k"]


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
        "path", "_path", "filepath", "file_path",
        "source", "filename", "video_path", "_VideoFromFile__file",
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


def _run_ffmpeg(cmd):
    """
    Запускает ffmpeg и кидает исключение с stderr при ошибке.
    Runs ffmpeg and raises an exception with stderr on failure.
    """
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


# Кэш метаданных видео — только в RAM.
# Video metadata cache — RAM only.
_VIDEO_INFO_CACHE = {}


def _get_video_info(path):
    """
    Метаданные видео (width/height/duration/codec/has_audio) из stderr ffmpeg.
    БЫСТРО: без декодирования. Кэш в RAM по (путь, mtime, размер).
    Video metadata (width/height/duration/codec/has_audio) from ffmpeg stderr.
    FAST: no decoding. Cached in RAM by (path, mtime, size).
    """
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        key = (path, 0, 0)

    cached = _VIDEO_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    info = {
        "width": 0, "height": 0, "duration": 0.0,
        "codec": "", "has_audio": False, "audio_codec": "",
    }

    try:
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

        # Длительность / Duration.
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if m:
            info["duration"] = (
                int(m.group(1)) * 3600 +
                int(m.group(2)) * 60 +
                float(m.group(3))
            )

        # Видеопоток: ВСЯ строка, NxM ищем по всей строке (после запятых).
        # Video stream: WHOLE line, NxM searched over the whole line.
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
            cm = re.search(r"Video:\s*([^,\n]+)", video_line)
            if cm:
                info["codec"] = cm.group(1).strip()

            gm = re.search(r"(\d{2,5})x(\d{2,5})", video_line)
            if gm:
                info["width"] = int(gm.group(1))
                info["height"] = int(gm.group(2))

        # Аудиопоток / Audio stream.
        am = re.search(r"Stream\s*#\d+:\d+[^\n]*Audio:\s*([^,\n]+)", err)
        if am:
            info["has_audio"] = True
            info["audio_codec"] = am.group(1).strip()

    except Exception as e:
        logger.warning(f"[AGSoft Video Save] metadata probe failed: {e}")

    if len(_VIDEO_INFO_CACHE) > 128:
        _VIDEO_INFO_CACHE.clear()

    _VIDEO_INFO_CACHE[key] = info

    return info


# ------------------------------------------------------------------------------
# NO-OP превью: стриминг файла по АБСОЛЮТНОМУ пути с Range (без записи).
# Локальный инструмент; проверяем только существование файла.
# NO-OP preview: Range streaming of a file by ABSOLUTE path (no writing).
# Local tool; we only check the file exists.
# ------------------------------------------------------------------------------
async def _stream_file_range(request, path):
    if not os.path.isfile(path):
        return web.Response(status=404)

    file_size = os.path.getsize(path)

    import mimetypes
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
        if m:
            first, last = m.group(1), m.group(2)
            if first == "" and last == "":
                pass
            elif first == "" and last != "":
                suffix = int(last)
                start = max(0, file_size - suffix)
                end = file_size - 1
                status = 206
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


@PromptServer.instance.routes.get("/agsoft/stream_path")
async def agsoft_stream_path(request):
    """
    Отдача файла по абсолютному пути (для no-op превью источника-файла).
    Serve a file by absolute path (for no-op preview of a file source).
    """
    path = request.query.get("path", "")
    if not path:
        return web.Response(status=400)
    return await _stream_file_range(request, os.path.abspath(path))


async def _stop_ffmpeg(proc):
    """
    Корректная остановка ffmpeg: после kill() ОБЯЗАТЕЛЬНО await wait(),
    иначе на Windows сыпется "I/O operation on closed pipe".
    Proper ffmpeg shutdown: after kill() you MUST await wait(), otherwise
    Windows spams "I/O operation on closed pipe".
    """
    try:
        if proc.returncode is None:
            proc.kill()
    except Exception:
        pass
    try:
        await proc.wait()
    except Exception:
        pass


@PromptServer.instance.routes.get("/agsoft/preview_path")
async def agsoft_preview_path(request):
    """
    Живой транскод файла по абсолютному пути (MKV/AVI/TS → AAC на лету) для
    no-op превью. start>0 — точная перемотка с транскодом ultrafast.
    Live transcode of a file by absolute path (MKV/AVI/TS → AAC on the fly)
    for no-op preview. start>0 — accurate seek with ultrafast transcode.
    """
    path = request.query.get("path", "")
    if not path:
        return web.Response(status=400)
    path = os.path.abspath(path)

    if not os.path.isfile(path):
        return web.Response(status=404)

    try:
        start = float(request.query.get("start", "0") or 0)
    except ValueError:
        start = 0.0
    if start < 0:
        start = 0.0

    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]

    cmd += ["-i", path, "-map", "0:v:0", "-map", "0:a:0?"]

    if start > 0:
        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p",
        ]
    else:
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
        await _stop_ffmpeg(proc)

    try:
        await resp.write_eof()
    except Exception:
        pass

    return resp


def _convert_video(src, out_file, fps, crf, f):
    """Прямая конвертация видео БЕЗ звука."""
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:v:0",
        *_vcodec_args(f, crf),
        "-r", str(fps),
        out_file,
    ])


def _convert_video_with_audio(src, audio_wav, out_file, fps, crf, f):
    """Конвертация видео + звук из WAV за один проход."""
    a = _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        *_vcodec_args(f, crf),
        *a,
        "-r", str(fps),
        "-shortest",
        out_file,
    ])


def _convert_video_with_src_audio(src, out_file, fps, crf, f):
    """Конвертация видео СО ЗВУКОМ САМОГО ИСХОДНИКА за один проход."""
    a = _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:v:0", "-map", "0:a:0",
        *_vcodec_args(f, crf),
        *a,
        "-r", str(fps),
        "-shortest",
        out_file,
    ])


def _remux_video_with_audio(src, audio_wav, out_file):
    """
    БЫСТРЫЙ ремукс для no-op превью: видео копируется (без перекодирования),
    звук со входа audio кодируется в AAC. Для больших файлов — быстро.
    FAST remux for no-op preview: video is copied (no re-encode), the audio
    input is encoded to AAC. Fast even for large files.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-i", audio_wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_file,
    ])


def _save_frames_video(frames, out_file, fps, crf, f):
    """Видео из списка PIL Image через промежуточные PNG (без звука)."""
    tmp_dir = tempfile.mkdtemp(prefix="agsoft_video_save_")
    try:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmp_dir, f"frame_{i:06d}.png"), "PNG")

        _run_ffmpeg([
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
            *_vcodec_args(f, crf),
            out_file,
        ])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_frames_video_with_audio(frames, audio_wav, out_file, fps, crf, f):
    """Видео из списка PIL Image + звук со входа audio за один проход."""
    tmp_dir = tempfile.mkdtemp(prefix="agsoft_video_save_")
    try:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmp_dir, f"frame_{i:06d}.png"), "PNG")

        _run_ffmpeg([
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
            "-i", audio_wav,
            "-map", "0:v:0", "-map", "1:a:0",
            *_vcodec_args(f, crf),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            out_file,
        ])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _add_audio_to_video(video_file, audio_wav, out_file, f):
    """Добавляет аудиодорожку к готовому видео (копирование видеопотока)."""
    a = _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_file, "-i", audio_wav,
        "-c:v", "copy", *a,
        "-shortest",
        out_file,
    ])


def _extract_audio(src, out_file):
    """Извлечь звуковую дорожку исходника в отдельный M4A."""
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:a:0",
        "-c:a", "aac", "-b:a", "192k",
        out_file,
    ])


def _encode_audio(wav, out_file):
    """Перекодировать временный WAV в M4A (AAC)."""
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", wav,
        "-c:a", "aac", "-b:a", "192k",
        out_file,
    ])


def _extract_first_frame(src, out_png):
    """Первый кадр видео в PNG."""
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src, "-frames:v", "1", out_png,
    ])


def _save_audio_to_temp(audio):
    """Сохраняет объект AUDIO во временный WAV и возвращает путь."""
    wf = audio.get("waveform")
    sr = int(audio.get("sample_rate", 44100))

    arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2").T

    tmp_dir = folder_paths.get_temp_directory()
    os.makedirs(tmp_dir, exist_ok=True)

    fd, path = tempfile.mkstemp(prefix="agsoft_audio_", suffix=".wav", dir=tmp_dir)
    os.close(fd)

    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    return os.path.abspath(path)


def _make_png_text_chunk(keyword, text):
    """
    Собирает PNG-чанк tEXt: [len][type][keyword][0x00][text][crc].
    Builds a PNG tEXt chunk: [len][type][keyword][0x00][text][crc].
    """
    body = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    crc = zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", crc)


def _embed_png_metadata(png_path, prompt, extra_pnginfo):
    """
    Вшивает workflow/prompt в PNG ручной вставкой tEXt-чанков сразу после IHDR —
    байт-в-байт тот же метод, что в AGSoft_Save_workflowImage.js. ВАЖНО: в чанк
    "workflow" пишется ЧИСТЫЙ граф (_unwrap_workflow), иначе loadGraphData
    получит обёртку {"workflow": ...} и откроет ПУСТОЙ воркфлоу.
    Существующие tEXt workflow/prompt удаляются — вшивка идемпотентна.
    Embeds workflow/prompt into the PNG by manually inserting tEXt chunks right
    after IHDR — byte-for-byte the same method as in AGSoft_Save_workflowImage.js.
    IMPORTANT: the "workflow" chunk stores the PURE graph (_unwrap_workflow),
    otherwise loadGraphData receives the {"workflow": ...} wrapper and opens an
    EMPTY workflow. Existing workflow/prompt tEXt chunks are removed first —
    embedding is idempotent.
    """
    try:
        workflow_json = None
        if extra_pnginfo is not None:
            workflow_json = json.dumps(
                _unwrap_workflow(extra_pnginfo), ensure_ascii=True
            )

        prompt_json = None
        if prompt is not None:
            prompt_json = json.dumps(prompt, ensure_ascii=True)

        with open(png_path, "rb") as fh:
            data = fh.read()

        if data[:8] != b"\x89PNG\r\n\x1a\n":
            logger.warning(f"Not a PNG, skip metadata embed: {png_path}")
            return

        pos = 8
        out = [data[:8]]
        inserted = False

        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            ctype = data[pos + 4:pos + 8]
            chunk_end = pos + 12 + length

            if chunk_end > len(data):
                out.append(data[pos:])
                break

            chunk = data[pos:chunk_end]

            # Старые tEXt workflow/prompt удаляем (идемпотентность).
            if ctype == b"tEXt":
                body = data[pos + 8:pos + 8 + length]
                sep = body.find(b"\x00")
                kw = body[:sep] if sep != -1 else body
                if kw in (b"workflow", b"prompt"):
                    pos = chunk_end
                    continue

            out.append(chunk)

            # Сразу после IHDR вставляем свежие чанки.
            if ctype == b"IHDR" and not inserted:
                if workflow_json is not None:
                    out.append(_make_png_text_chunk("workflow", workflow_json))
                if prompt_json is not None:
                    out.append(_make_png_text_chunk("prompt", prompt_json))
                inserted = True

            pos = chunk_end

        if not inserted:
            logger.warning(f"No IHDR found, skip metadata embed: {png_path}")
            return

        with open(png_path, "wb") as fh:
            fh.write(b"".join(out))

        logger.info(
            f"Embedded workflow metadata into PNG: {png_path} "
            f"(workflow={len(workflow_json or '')} bytes, PURE graph)"
        )

    except Exception as e:
        logger.warning(f"PNG metadata embed failed for {png_path}: {e}")


def _embed_metadata(video_file, meta_dict):
    """
    Вшивает воркфлоу в метаданные видеофайла:
    быстрый remux-проход с ffmetadata-файлом (-map_metadata 1, -c copy).
    Для mp4/mov ОБЯЗАТЕЛЬНО -movflags use_metadata_tags — без него тег
    comment тихо отваливается и воркфлоу не переживает remux.
    Embeds the workflow into video file metadata:
    a fast remux pass with an ffmetadata file (-map_metadata 1, -c copy).
    For mp4/mov, -movflags use_metadata_tags is REQUIRED — without it the
    comment tag is silently dropped and the workflow does not survive.
    """
    ext = os.path.splitext(video_file)[1].lstrip(".")

    fd, meta_path = tempfile.mkstemp(prefix="agsoft_meta_", suffix=".ffmeta")
    os.close(fd)
    fd, tmp_out = tempfile.mkstemp(prefix="agsoft_meta_", suffix=f".{ext}")
    os.close(fd)

    try:
        raw = json.dumps(meta_dict, ensure_ascii=False)

        # Экранирование по спецификации ffmetadata.
        # Escaping per the ffmetadata spec.
        def esc(s):
            return (s.replace("\\", "\\\\")
                     .replace("=", "\\=")
                     .replace(";", "\\;")
                     .replace("#", "\\#")
                     .replace("\n", "\\n"))

        with open(meta_path, "w", encoding="utf-8") as fh:
            fh.write(";FFMETADATA1\n")
            fh.write("comment=" + esc(raw) + "\n")

        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_file, "-i", meta_path,
            "-map", "0", "-c", "copy",
            "-map_metadata", "1",
        ]

        # Для mp4/mov — разрешаем произвольные теги метаданных.
        # For mp4/mov — allow arbitrary metadata tags.
        if ext in ("mp4", "mov"):
            cmd += ["-movflags", "use_metadata_tags"]

        cmd += [tmp_out]

        _run_ffmpeg(cmd)

        os.replace(tmp_out, video_file)
        logger.info(f"Embedded workflow metadata: {video_file} ({len(raw)} bytes)")

    finally:
        for p in (meta_path, tmp_out):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _unescape_ffmetadata(s):
    """Обратное экранирование ffmetadata: \\= \\; \\# \\n \\\\."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            out.append("\n" if nxt == "n" else nxt)
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _parse_ffmetadata_comment(text):
    """
    Достаёт comment=<json> из дампа ffmetadata и возвращает ЧИСТЫЙ воркфлоу
    (обёртка {"workflow": ...} разворачивается — см. _unwrap_workflow).
    Extracts comment=<json> from an ffmetadata dump and returns the PURE
    workflow (the {"workflow": ...} wrapper is unwrapped — see _unwrap_workflow).
    """
    m = re.search(r"^comment=(.+)$", text, re.M)
    if not m:
        return None

    try:
        data = json.loads(_unescape_ffmetadata(m.group(1)))
    except Exception:
        return None

    if isinstance(data, dict):
        return _unwrap_workflow(data.get("workflow"))
    return None


def _path_allowed(path):
    """
    Разрешаем вшивку только в output/temp — защита от записи куда попало.
    Allow embedding only into output/temp — no writing to arbitrary paths.
    """
    p = os.path.abspath(path)
    for base in (folder_paths.get_output_directory(), folder_paths.get_temp_directory()):
        try:
            if os.path.commonpath([p, os.path.abspath(base)]) == os.path.abspath(base):
                return True
        except Exception:
            continue
    return False


@PromptServer.instance.routes.post("/agsoft/embed_workflow")
async def agsoft_embed_workflow(request):
    """
    Endpoint для JS: вшивает воркфлоу (app.graph.serialize() из браузера —
    как в AGSoft_Save_workflowImage.js) в сохранённый файл:
    * PNG  — tEXt-чанк "workflow" (ЧИСТЫЙ граф) после IHDR;
    * видео (mp4/mkv/mov/webm) — remux с ffmetadata + use_metadata_tags.
    Вызывается из onExecuted, поэтому работает даже если extra_pnginfo
    не передаётся сборкой ComfyUI. Обёртка {"workflow": ...} разворачивается.
    JS endpoint: embeds the workflow (app.graph.serialize() from the browser —
    like in AGSoft_Save_workflowImage.js) into the saved file:
    * PNG  — "workflow" tEXt chunk (PURE graph) after IHDR;
    * video (mp4/mkv/mov/webm) — ffmetadata remux + use_metadata_tags.
    Called from onExecuted, so it works even if extra_pnginfo never reaches
    the node. The {"workflow": ...} wrapper is unwrapped.
    """
    try:
        data = await request.json()
        path = data.get("path") or ""
        workflow = _unwrap_workflow(data.get("workflow"))
        prompt = data.get("prompt")

        if not path or not os.path.isfile(path):
            return web.json_response({"ok": False, "error": "file not found"})

        if not _path_allowed(path):
            return web.json_response({"ok": False, "error": "path not allowed"})

        ext = os.path.splitext(path)[1].lstrip(".").lower()

        if ext == "png":
            _embed_png_metadata(path, prompt, workflow)
        elif ext in META_EXTS:
            meta = {}
            if prompt is not None:
                meta["prompt"] = prompt
            if workflow is not None:
                meta["workflow"] = workflow
            if meta:
                _embed_metadata(path, meta)
        else:
            return web.json_response({"ok": False, "error": f"unsupported format: {ext}"})

        return web.json_response({"ok": True})

    except Exception as e:
        logger.warning(f"[AGSoft Video Save] embed_workflow failed: {e}")
        return web.json_response({"ok": False, "error": str(e)})


@PromptServer.instance.routes.post("/agsoft/extract_workflow")
async def agsoft_extract_workflow(request):
    """
    Endpoint для JS: принимает перетащенный видеофайл и возвращает вшитый
    в него ЧИСТЫЙ воркфлоу (если есть). Используется для восстановления
    воркфлоу перетаскиванием сохранённого видео на канвас.
    JS endpoint: accepts a dropped video file and returns the embedded PURE
    workflow (if any). Used to restore the workflow by dragging a saved
    video onto the canvas.
    """
    tmp = None
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"workflow": None})

        field = None
        f = await reader.next()
        while f is not None:
            if f.name == "file":
                field = f
                break
            f = await reader.next()

        if field is None:
            return web.json_response({"workflow": None})

        fd, tmp = tempfile.mkstemp(
            prefix="agsoft_drop_",
            suffix=os.path.splitext(field.filename or "")[1] or ".mp4",
        )
        os.close(fd)

        with open(tmp, "wb") as out:
            while True:
                chunk = await field.read_chunk(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        # Дамп метаданных контейнера через ffmpeg (читает только заголовок).
        # Dump the container metadata via ffmpeg (reads the header only).
        proc = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-i", tmp, "-f", "ffmetadata", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30,
        )

        workflow = _parse_ffmetadata_comment(proc.stdout or "")

        return web.json_response({"workflow": workflow})

    except Exception as e:
        logger.warning(f"[AGSoft Video Save] extract_workflow failed: {e}")
        return web.json_response({"workflow": None})
    finally:
        if tmp is not None:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


class AGSoftVideoSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Верхняя группа: имя/путь — логичнее видеть сразу.
                # Top group: naming/path — more logical to see first.
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "AGSoft_Video",
                        "tooltip": (
                            "Prefix for the output filename. A timestamp is added automatically.\n"
                            "---\n"
                            "Префикс имени файла. Временная метка добавляется автоматически."
                        )
                    }
                ),
                "subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Subfolder inside the output (or temp) directory. Nested paths are "
                            "allowed: 'MiniMax' or 'MiniMax/2026'. '..' and drive letters are "
                            "sanitized for safety.\n"
                            "Example: MiniMax → output/MiniMax/AGSoft_Video_<timestamp>.mp4\n"
                            "---\n"
                            "Подпапка внутри папки output (или temp). Разрешена вложенность: "
                            "'MiniMax' или 'MiniMax/2026'. '..' и диски санациируются для безопасности.\n"
                            "Пример: MiniMax → output/MiniMax/AGSoft_Video_<timestamp>.mp4"
                        )
                    }
                ),
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional: absolute path to a video file to convert.\n"
                            "---\n"
                            "Опционально: абсолютный путь к видеофайлу для конвертации."
                        )
                    }
                ),
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": (
                            "Frames per second for the output video.\n"
                            "---\n"
                            "Кадров в секунду для выходного видео."
                        )
                    }
                ),
                "format": (
                    _available_formats(),
                    {
                        "default": "video/h264-mp4",
                        "tooltip": (
                            "Output format / codec. The list shows only encoders "
                            "available in your ffmpeg build.\n"
                            "---\n"
                            "Формат / кодек выхода. В списке только энкодеры, "
                            "доступные в вашей сборке ffmpeg."
                        )
                    }
                ),
                "crf": (
                    "INT",
                    {
                        "default": 19,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": (
                            "Constant Rate Factor: 0=lossless, 51=worst. Ignored for "
                            "GIF/WebP/FFV1/ProRes.\n"
                            "---\n"
                            "Constant Rate Factor: 0=lossless, 51=худшее. "
                            "Игнорируется для GIF/WebP/FFV1/ProRes."
                        )
                    }
                ),
                "save_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Save video WITHOUT audio. All save options off = no-op with preview "
                            "(nothing saved, no error).\n"
                            "---\n"
                            "Сохранить видео БЕЗ звука. Все опции выключены = no-op с превью "
                            "(ничего не сохраняется, без ошибки)."
                        )
                    }
                ),
                "save_video_with_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save video WITH audio. Audio is taken from the audio input if "
                            "connected, otherwise from the source video's own audio track.\n"
                            "---\n"
                            "Сохранить видео СО звуком. Звук берётся со входа audio, если он "
                            "подключён, иначе — из звуковой дорожки самого исходного видео."
                        )
                    }
                ),
                "save_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save the audio track as a SEPARATE M4A file.\n"
                            "---\n"
                            "Сохранить звуковую дорожку ОТДЕЛЬНЫМ файлом M4A."
                        )
                    }
                ),
                "save_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save the first frame as a PNG image. With save_metadata enabled the "
                            "PNG also carries the workflow (tEXt chunk after IHDR) and opens it "
                            "on drag&drop.\n"
                            "---\n"
                            "Сохранить первый кадр как PNG. При включённом save_metadata в PNG "
                            "вшивается воркфлоу (tEXt-чанк после IHDR), и он открывается "
                            "перетаскиванием."
                        )
                    }
                ),
                "save_metadata": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Embed the workflow (PURE graph, no wrapper) into the saved file's "
                            "metadata. Dragging the saved PNG onto the canvas opens it natively;"
                            "dragging the saved video opens it via our "
                            "handler.\n"
                            "---\n"
                            "Вшить воркфлоу (ЧИСТЫЙ граф, без обёртки) в метаданные сохранённого "
                            "файла. Перетаскивание сохранённого PNG на канвас открывает его штатно;"
                            "сохранённого видео — через наш обработчик."
                        )
                    }
                ),
                "save_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True = save to the output folder; False = save to temp (preview only, "
                            "no clutter in output).\n"
                            "---\n"
                            "True = сохранять в папку output; False = во временную папку (только "
                            "превью, без мусора в output)."
                        )
                    }
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Frame sequence to save as video. Used when no video / video_path "
                            "inputs are connected.\n"
                            "---\n"
                            "Последовательность кадров для сохранения. Используется, когда не "
                            "подключены video / video_path."
                        )
                    }
                ),
                "video": (
                    IO.VIDEO,
                    {
                        "tooltip": (
                            "Optional: VIDEO object to convert (highest priority). Its own audio "
                            "track is used when the audio input is not connected.\n"
                            "---\n"
                            "Опционально: объект VIDEO для конвертации (наивысший приоритет). Его "
                            "собственная звуковая дорожка используется, когда вход audio не подключён."
                        )
                    }
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional: AUDIO to mux into the video / save as M4A. Highest priority "
                            "over the source video's own audio track. Also muxed into the no-op "
                            "preview so you HEAR the result before saving.\n"
                            "---\n"
                            "Опционально: AUDIO для подмешивания в видео / сохранения в M4A. "
                            "Наивысший приоритет перед собственной дорожкой исходного видео. "
                            "Также подмешивается в no-op превью — результат СЛЫШНО до сохранения."
                        )
                    }
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"
    CATEGORY = "AGSoft/Video"

    # Отдаём ui-словарь с превью, фронтенд вызывает onExecuted.
    # Return a ui dict with the preview, frontend calls onExecuted.
    OUTPUT_NODE = True

    # JS из web/ рисует превью (сериализуется в воркфлоу), ресайзит ноду как
    # Load Video, дошивает воркфлоу из браузера и восстанавливает его при
    # перетаскивании сохранённого ВИДЕО (PNG открывает сам ComfyUI).
    # JS from web/ draws the preview (serialized into the workflow), resizes
    # the node like Load Video, embeds the workflow from the browser and
    # restores it when a saved VIDEO is dragged in (PNGs open natively).
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬 AGSoft Video Save.\n"
        "Saves video from an image sequence OR converts an existing video (a small video "
        "converter). Saves ONLY what you enable: video without audio, video with audio, audio as "
        "a separate M4A, first frame — any combination, no extra files.\n"
        "NO-OP MODE WITH PREVIEW (WITH SOUND): all save options off = nothing written to output, "
        "no error, and the source preview is shown INCLUDING the audio input (fast remux/encode "
        "to temp).\n"
        "Priority: video > video_path > images.\n"
        "Audio is taken from the audio input if connected, otherwise from the source video's own "
        "track.\n"
        "Save path: output (or temp) + subfolder + filename_prefix_<timestamp> "
        "(e.g. output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4).\n"
        "save_metadata embeds the workflow as a PURE graph (no wrapper) into the saved files; "
        "dragging the saved PNG/video onto the canvas restores the workflow.\n"
        "Codec selection: h264/h265, NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV, GIF, WebP "
        "(list built from available ffmpeg encoders).\n"
        "The result preview is shown inside the node, serialized into the workflow, and the node "
        "resizes exactly like Load Video.\n"
        "---\n"
        "🎬 AGSoft Video Save.\n"
        "Сохраняет видео из последовательности кадров ИЛИ конвертирует существующее видео "
        "(небольшой видео-конвертор). Сохраняет ТОЛЬКО включённое: видео без звука, видео со "
        "звуком, звук отдельным M4A, первый кадр — любые комбинации, без лишних файлов.\n"
        "РЕЖИМ NO-OP С ПРЕВЬЮ (СО ЗВУКОМ): все опции сохранения выключены = output не "
        "трогается, без ошибки, а превью источника показывается СО звуком со входа audio "
        "(быстрый ремукс/энкод в temp).\n"
        "Приоритет: video > video_path > images.\n"
        "Звук берётся со входа audio, если он подключён, иначе — из дорожки самого исходного видео.\n"
        "Путь сохранения: output (или temp) + subfolder + filename_prefix_<timestamp> "
        "(например output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4).\n"
        "save_metadata вшивает воркфлоу ЧИСТЫМ графом (без обёртки); перетаскивание "
        "сохранённого PNG/видео на канвас восстанавливает воркфлоу.\n"
        "Выбор кодеков: h264/h265, NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV, GIF, WebP "
        "(список строится по доступным энкодерам ffmpeg).\n"
        "Превью результата показывается в ноде, сериализуется в воркфлоу, ресайз ноды — один-в-один "
        "как в Load Video."
    )

    def save_video(
        self,
        filename_prefix,
        subfolder,
        video_path,
        frame_rate,
        format,
        crf,
        save_video,
        save_video_with_audio,
        save_audio,
        save_image,
        save_metadata,
        save_output,
        images=None,
        video=None,
        audio=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        audio_wav = None
        try:
            # ------------------------------------------------------------------
            # NO-OP + ПРЕВЬЮ СО ЗВУКОМ: все опции выключены = output не
            # трогается, ошибки нет, но превью источника показывается, включая
            # звук со входа audio.
            # NO-OP + PREVIEW WITH SOUND: all options off = output untouched,
            # no error, and the source preview is shown INCLUDING the audio
            # input.
            # ------------------------------------------------------------------
            if not (save_video or save_video_with_audio or save_audio or save_image):
                logger.info(
                    "[AGSoft Video Save] All save options disabled — no-op with preview (with sound)."
                )

                if audio is not None:
                    audio_wav = _save_audio_to_temp(audio)

                src_preview = None
                if video is not None:
                    src_preview = _extract_video_path(video) or None
                if src_preview is None and video_path and os.path.exists(video_path):
                    src_preview = os.path.abspath(video_path)

                preview = None
                tmp_dir = folder_paths.get_temp_directory()
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_file = os.path.join(
                    tmp_dir,
                    f"agsoft_preview_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                )

                try:
                    if src_preview:
                        if audio_wav is not None:
                            # Файл + вход audio: быстрый ремукс (видео copy + AAC).
                            # File + audio input: fast remux (video copy + AAC).
                            try:
                                _remux_video_with_audio(src_preview, audio_wav, tmp_file)
                                preview = {
                                    "kind": "file",
                                    "filename": os.path.basename(tmp_file),
                                    "subfolder": "",
                                    "type": "temp",
                                    "format": "video/mp4",
                                    "fullpath": tmp_file,
                                }
                            except Exception:
                                # Ремукс не удался (кодек не в mp4) — стрим как есть.
                                # Remux failed (codec not mp4-safe) — stream as is.
                                preview = None
                        if preview is None and audio_wav is None:
                            # Файл без входа audio: стрим напрямую, без записи.
                            # File without audio input: direct stream, no writing.
                            ext = os.path.splitext(src_preview)[1].lstrip(".").lower()
                            preview = {
                                "kind": "path",
                                "path": src_preview,
                                "ext": ext,
                                "filename": os.path.basename(src_preview),
                                "subfolder": "",
                                "type": "path",
                                "format": "",
                                "fullpath": src_preview,
                            }
                    elif images is not None and len(images) > 0:
                        # Тензор: энкод кадров (+ аудио, если есть вход audio).
                        # Tensor: frame encode (+ audio if the audio input is set).
                        frames = [
                            Image.fromarray(
                                (t.cpu().numpy() * 255).astype(np.uint8), "RGB"
                            )
                            for t in images
                        ]
                        if audio_wav is not None:
                            _save_frames_video_with_audio(
                                frames, audio_wav, tmp_file, frame_rate, 23,
                                FORMAT_PRESETS["video/h264-mp4"],
                            )
                        else:
                            _save_frames_video(
                                frames, tmp_file, frame_rate, 23,
                                FORMAT_PRESETS["video/h264-mp4"],
                            )
                        preview = {
                            "kind": "file",
                            "filename": os.path.basename(tmp_file),
                            "subfolder": "",
                            "type": "temp",
                            "format": "video/mp4",
                            "fullpath": tmp_file,
                        }
                except Exception as e:
                    logger.warning(f"[AGSoft Video Save] no-op preview failed: {e}")

                return {
                    "ui": {"gifs": [preview] if preview else []},
                    "result": ("",),
                }

            f = FORMAT_PRESETS.get(format, FORMAT_PRESETS["video/h264-mp4"])

            # Контейнер без звука (GIF/WebP) — отключаем звуковые опции.
            # Audio-less container (GIF/WebP) — disable audio options.
            if f.get("acodec") is None:
                if save_video_with_audio:
                    logger.warning(f"{format} cannot hold audio — save_video_with_audio skipped.")
                    save_video_with_audio = False
                if save_audio:
                    logger.warning(f"{format} cannot hold audio — save_audio skipped.")
                    save_audio = False

            # ------------------------------------------------------------------
            # Воркфлоу: разворачиваем обёртку {"workflow": ...} ОДИН раз здесь.
            # Workflow: unwrap the {"workflow": ...} wrapper ONCE here.
            # ------------------------------------------------------------------
            wf_clean = None
            if save_metadata and extra_pnginfo is not None:
                wf_clean = _unwrap_workflow(extra_pnginfo)

            # ------------------------------------------------------------------
            # Источник: видео-файл или тензор кадров.
            # Source: a video file or a frame tensor.
            # ------------------------------------------------------------------
            src_video = None

            if video is not None:
                src_video = _extract_video_path(video)
                if not src_video:
                    raise ValueError(
                        "[AGSoft Video Save] Could not extract path from VIDEO object."
                    )
                logger.info(f"Converting VIDEO object: {src_video}")

            elif video_path and os.path.exists(video_path):
                src_video = os.path.abspath(video_path)
                logger.info(f"Converting video path: {src_video}")

            frames = None
            if src_video is None:
                if images is None or len(images) == 0:
                    raise ValueError(
                        "[AGSoft Video Save] No source: connect video, video_path or images."
                    )
                frames = []
                for img_tensor in images:
                    arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    frames.append(Image.fromarray(arr, "RGB"))
                logger.info(f"Saving {len(frames)} frames from IMAGE tensor")

            # ------------------------------------------------------------------
            # Источник ЗВУКА: вход audio > дорожка исходного видео.
            # AUDIO source: audio input > source video's track.
            # ------------------------------------------------------------------
            has_src_audio = False
            if src_video is not None:
                has_src_audio = bool(_get_video_info(src_video).get("has_audio"))

            if audio is not None:
                audio_wav = _save_audio_to_temp(audio)
                logger.info("Audio source: audio input")
            elif has_src_audio:
                logger.info("Audio source: source video's own audio track")

            have_audio = (audio_wav is not None) or has_src_audio

            if save_video_with_audio and not have_audio:
                logger.warning(
                    "save_video_with_audio enabled but no audio available "
                    "(no audio input and no audio track in source) — skipping."
                )
                save_video_with_audio = False

            if save_audio and not have_audio:
                logger.warning("save_audio enabled but no audio available — skipping.")
                save_audio = False

            # ------------------------------------------------------------------
            # Куда сохраняем: output или temp + подпапка (санация).
            # Where to save: output or temp + subfolder (sanitized).
            # ------------------------------------------------------------------
            if save_output:
                base_dir = folder_paths.get_output_directory()
                out_type = "output"
            else:
                base_dir = folder_paths.get_temp_directory()
                out_type = "temp"

            sub_clean = _norm_subfolder(subfolder)
            output_dir = os.path.join(base_dir, sub_clean) if sub_clean else base_dir
            os.makedirs(output_dir, exist_ok=True)

            if sub_clean:
                logger.info(f"Saving into subfolder: {sub_clean}")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{filename_prefix}_{timestamp}"

            video_file = None
            video_with_audio = None
            audio_file = None
            saved_image = None

            # ------------------------------------------------------------------
            # Источник — ВИДЕО: прямая конвертация через ffmpeg.
            # Source is VIDEO: direct ffmpeg conversion.
            # ------------------------------------------------------------------
            if src_video is not None:
                if save_video:
                    out_file = os.path.join(output_dir, f"{base_name}.{f['ext']}")
                    _convert_video(src_video, out_file, frame_rate, crf, f)
                    video_file = out_file
                    logger.info(f"Saved video (no audio): {out_file}")

                if save_video_with_audio:
                    out_file = os.path.join(
                        output_dir, f"{base_name}_audio.{f['ext']}"
                    )
                    if audio_wav is not None:
                        _convert_video_with_audio(
                            src_video, audio_wav, out_file, frame_rate, crf, f
                        )
                    else:
                        _convert_video_with_src_audio(
                            src_video, out_file, frame_rate, crf, f
                        )
                    video_with_audio = out_file
                    logger.info(f"Saved video with audio: {out_file}")

                if save_audio:
                    out_file = os.path.join(output_dir, f"{base_name}_sound.m4a")
                    if audio_wav is not None:
                        _encode_audio(audio_wav, out_file)
                    else:
                        _extract_audio(src_video, out_file)
                    audio_file = out_file
                    logger.info(f"Saved audio track: {out_file}")

                if save_image:
                    saved_image = os.path.join(output_dir, f"{base_name}.png")
                    _extract_first_frame(src_video, saved_image)
                    logger.info(f"Saved first frame: {saved_image}")

            # ------------------------------------------------------------------
            # Источник — КАДРЫ: сборка видео из PNG-последовательности.
            # Source is FRAMES: build video from a PNG sequence.
            # ------------------------------------------------------------------
            else:
                if save_video:
                    out_file = os.path.join(output_dir, f"{base_name}.{f['ext']}")
                    _save_frames_video(frames, out_file, frame_rate, crf, f)
                    video_file = out_file
                    logger.info(f"Saved video (no audio): {out_file}")

                if save_video_with_audio and audio_wav is not None:
                    silent = os.path.join(output_dir, f"{base_name}_silent.{f['ext']}")
                    _save_frames_video(frames, silent, frame_rate, crf, f)
                    try:
                        out_file = os.path.join(
                            output_dir, f"{base_name}_audio.{f['ext']}"
                        )
                        _add_audio_to_video(silent, audio_wav, out_file, f)
                        video_with_audio = out_file
                        logger.info(f"Saved video with audio: {out_file}")
                    finally:
                        try:
                            os.remove(silent)
                        except Exception:
                            pass

                if save_audio and audio_wav is not None:
                    out_file = os.path.join(output_dir, f"{base_name}_sound.m4a")
                    _encode_audio(audio_wav, out_file)
                    audio_file = out_file
                    logger.info(f"Saved audio track: {out_file}")

                if save_image:
                    saved_image = os.path.join(output_dir, f"{base_name}.png")
                    frames[0].save(saved_image, "PNG")
                    logger.info(f"Saved first frame: {saved_image}")

            # ------------------------------------------------------------------
            # save_metadata: ЧИСТЫЙ граф в PNG (tEXt после IHDR) и в метаданные
            # видео (remux с ffmetadata + use_metadata_tags).
            # save_metadata: PURE graph into PNG (tEXt after IHDR) and into
            # video metadata (ffmetadata remux + use_metadata_tags).
            # ------------------------------------------------------------------
            if save_metadata and (wf_clean is not None or prompt is not None):
                if saved_image is not None:
                    _embed_png_metadata(saved_image, prompt, wf_clean)

                meta_dict = {}
                if prompt is not None:
                    meta_dict["prompt"] = prompt
                if wf_clean is not None:
                    meta_dict["workflow"] = wf_clean

                for vf in (video_with_audio, video_file):
                    if vf is not None and os.path.splitext(vf)[1].lstrip(".") in META_EXTS:
                        try:
                            _embed_metadata(vf, meta_dict)
                        except Exception as e:
                            logger.warning(f"Metadata embed failed for {vf}: {e}")

            # ------------------------------------------------------------------
            # Превью и результат: видео со звуком > тихое видео > картинка > звук.
            # Preview and result: video with audio > silent video > image > audio.
            # ------------------------------------------------------------------
            preview_file = (
                video_with_audio or video_file or saved_image or audio_file
            )

            if preview_file in (video_with_audio, video_file) and preview_file:
                preview_mime = f["mime"]
            elif preview_file == saved_image:
                preview_mime = "image/png"
            elif preview_file == audio_file:
                preview_mime = "audio/mp4"
            else:
                preview_mime = ""

            result_path = preview_file or ""

            preview = {
                "kind": "file",
                "filename": os.path.basename(preview_file) if preview_file else "",
                "subfolder": sub_clean,
                "type": out_type,
                "format": preview_mime,
                "frame_rate": frame_rate,
                "fullpath": preview_file or "",
            }

            return {"ui": {"gifs": [preview]}, "result": (result_path,)}

        except Exception as e:
            error_msg = f"[AGSoft Video Save] Error saving video: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        finally:
            if audio_wav is not None:
                try:
                    os.remove(audio_wav)
                except Exception:
                    pass

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Всегда пересохраняем: имя файла содержит временную метку.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # ВАЖНО: подключённые входы (линки) приходят сюда как None/отсутствуют —
        # реальное значение станет известно только при выполнении.
        # Выключенные опции сохранения — НЕ ошибка (no-op режим с превью).
        # IMPORTANT: connected inputs (links) arrive here None/missing —
        # the real value is only known at execution time.
        # Disabled save options are NOT an error (no-op mode with preview).
        video = kwargs.get("video")
        images = kwargs.get("images")
        video_path = kwargs.get("video_path", "") or ""

        if video is not None or images is not None:
            return True

        if video_path:
            if not os.path.exists(video_path):
                return f"Video path does not exist: {video_path}"
            if not os.path.isfile(video_path):
                return f"Video path is not a file: {video_path}"

        return True


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoSave": AGSoftVideoSave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoSave": "🎬AGSoft Video Save"
}