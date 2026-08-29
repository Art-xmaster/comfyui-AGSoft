# ==============================================================================
# AGSoft_Video_Save.py
# ==============================================================================
# Node: 🎬AGSoft Video Save
# Version: v3.01
#
# Saves video from an image sequence OR converts an existing video.
# It saves ONLY what is enabled: video without audio, video with audio,
# audio as a separate file, first frame — any combination, no extra files.
#
# After execution the result preview is shown inside the node; the preview
# is serialized into the workflow and survives page reloads.
#
# NO-OP MODE + PREVIEW:
# If ALL save options are disabled, the node does NOT write files into output
# and does NOT fail. The upstream graph is executed, and the source preview
# is shown, including audio from the audio input:
# - file + audio: fast remux (video copy + AAC) to a temp file;
# - file without audio: direct stream by absolute path (no writing);
# - tensor + audio: frame encode with audio to a temp file;
# - tensor without audio: frame encode to a temp file.
#
# Source priority:
# 1. video (VIDEO object) — convert existing video;
# 2. video_path (file path) — convert video by path;
# 3. images (frame tensor) — save from frame sequence.
#
# Audio source priority:
# 1. audio input (if connected);
# 2. the source video's own audio track (if present).
#
# Save options:
# - save_video: video without audio;
# - save_video_with_audio: video with audio;
# - save_audio: audio track as a separate M4A file;
# - save_image: first frame as PNG;
# - save_metadata: embed workflow as PURE graph into file metadata;
# - save_output: True = output folder, False = temp folder.
#
# Save path:
# output (or temp) + subfolder + filename_prefix_<timestamp>.
# Example:
# subfolder="MiniMax" -> output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4
#
# Codec selection:
# h264/h265 CPU, h264/h265 NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV,
# GIF, WebP, plus video/copy-mp4 for fast stream copy.
# The list is built dynamically from available ffmpeg encoders.
#
# ---
#
# Нода: 🎬AGSoft Video Save
# Версия: v3.01
#
# Сохраняет видео из последовательности изображений ИЛИ конвертирует
# существующее видео.
# Сохраняет ТОЛЬКО то, что включено: видео без звука, видео со звуком,
# звук отдельным файлом, картинку — любые комбинации, без лишних файлов.
#
# После выполнения показывает превью результата прямо в ноде; превью
# сериализуется в воркфлоу и переживает перезагрузку страницы.
#
# РЕЖИМ NO-OP + ПРЕВЬЮ:
# Если ВСЕ опции сохранения выключены, нода НЕ пишет файлы в output и
# НЕ падает с ошибкой. Генерация выше по графу выполняется, а превью
# источника показывается, включая звук со входа audio:
# - файл + audio: быстрый ремукс (видео copy + AAC) во временный файл;
# - файл без audio: стрим напрямую по абсолютному пути (без записи);
# - тензор + audio: энкод кадров с аудио во временный файл;
# - тензор без audio: энкод кадров во временный файл.
#
# Приоритет источников:
# 1. video (объект VIDEO) — конвертация существующего видео;
# 2. video_path (путь к файлу) — конвертация видео по пути;
# 3. images (тензор кадров) — сохранение из последовательности кадров.
#
# Приоритет источника звука:
# 1. вход audio (если подключён);
# 2. звуковая дорожка самого исходного видео (если она есть).
#
# Опции сохранения:
# - save_video: видео без звука;
# - save_video_with_audio: видео со звуком;
# - save_audio: звуковая дорожка отдельным файлом M4A;
# - save_image: первый кадр как PNG;
# - save_metadata: вшить воркфлоу ЧИСТЫМ графом в метаданные файла;
# - save_output: True = папка output, False = temp.
#
# Путь сохранения:
# output (или temp) + subfolder + filename_prefix_<timestamp>.
# Пример:
# subfolder="MiniMax" -> output/MiniMax/AGSoft_Video_20260817_192654_audio.mp4
#
# Выбор кодеков:
# h264/h265 CPU, h264/h265 NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV,
# GIF, WebP, плюс video/copy-mp4 для быстрого копирования потока.
# Список строится динамически по доступным энкодерам ffmpeg.
#
# Автор / Author: AGSoft
# Дата / Date: 30.08.2026
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
import mimetypes

import folder_paths
import numpy as np
import torch
from PIL import Image
from aiohttp import web
from server import PromptServer


def _find_ffmpeg():
    """
    Ищем ffmpeg: сначала imageio_ffmpeg, затем PATH.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG_PATH = _find_ffmpeg()


def _ffmpeg_available():
    """
    True, если ffmpeg реально доступен.
    """
    p = FFMPEG_PATH or ""
    if not p:
        return False
    if os.path.isabs(p) or os.sep in p:
        return os.path.isfile(p)
    return shutil.which(p) is not None


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

# print("[AGSoft Video Save] v3.01 loaded (fixed hangs, faster encoding, one-pass audio, no double embed, safe async endpoints)")


# ------------------------------------------------------------------------------
# Настройки безопасности / таймаутов / лимитов
# ------------------------------------------------------------------------------

MAX_FFMPEG_TIMEOUT = int(os.environ.get("AGSOFT_FFMPEG_TIMEOUT", "900"))
PROBE_TIMEOUT = int(os.environ.get("AGSOFT_FFPROBE_TIMEOUT", "15"))

# Максимальный размер файла для drag&drop restore, MB
DROP_MAX_BYTES = int(os.environ.get("AGSOFT_DROP_MAX_MB", "2048")) * 1024 * 1024

# Если 0 / false / no — чтение/превью разрешено только из input/output/temp.
# По умолчанию разрешены любые пути, чтобы не ломать локальные сценарии.
ALLOW_ANY_READ_PATH = os.environ.get("AGSOFT_ALLOW_ANY_PATH", "1").lower() not in ("0", "false", "no")

_PREVIEW_ACTIVE = 0
_PREVIEW_MAX = int(os.environ.get("AGSOFT_PREVIEW_MAX", "2"))


# ------------------------------------------------------------------------------
# Пресеты форматов/кодеков
# ------------------------------------------------------------------------------

FORMAT_PRESETS = {
    "video/h264-mp4": {
        "ext": "mp4",
        "vcodec": "libx264",
        "acodec": "aac",
        "pix_fmt": "yuv420p",
        "mime": "video/mp4",
    },
    "video/h265-mp4": {
        "ext": "mp4",
        "vcodec": "libx265",
        "acodec": "aac",
        "pix_fmt": "yuv420p",
        "mime": "video/mp4",
    },
    "video/h264-nvenc-mp4": {
        "ext": "mp4",
        "vcodec": "h264_nvenc",
        "acodec": "aac",
        "pix_fmt": "yuv420p",
        "mime": "video/mp4",
    },
    "video/h265-nvenc-mp4": {
        "ext": "mp4",
        "vcodec": "hevc_nvenc",
        "acodec": "aac",
        "pix_fmt": "yuv420p",
        "mime": "video/mp4",
    },
    "video/webm": {
        "ext": "webm",
        "vcodec": "libvpx-vp9",
        "acodec": "libopus",
        "pix_fmt": "yuv420p",
        "mime": "video/webm",
    },
    "video/av1-webm": {
        "ext": "webm",
        "vcodec": "libsvtav1",
        "acodec": "libopus",
        "pix_fmt": "yuv420p",
        "mime": "video/webm",
    },
    "video/ffv1-mkv": {
        "ext": "mkv",
        "vcodec": "ffv1",
        "acodec": "pcm_s16le",
        "pix_fmt": "yuv420p",
        "mime": "video/x-matroska",
    },
    "video/prores-mov": {
        "ext": "mov",
        "vcodec": "prores_ks",
        "acodec": "pcm_s16le",
        "pix_fmt": "yuv422p10le",
        "mime": "video/quicktime",
    },
    "image/gif": {
        "ext": "gif",
        "vcodec": "gif",
        "acodec": None,
        "pix_fmt": None,
        "mime": "image/gif",
    },
    "image/webp": {
        "ext": "webp",
        "vcodec": "libwebp",
        "acodec": None,
        "pix_fmt": None,
        "mime": "image/webp",
    },

    # Быстрый режим без перекодирования видеопотока.
    # Аудио при необходимости перекодируется в AAC для совместимости с MP4.
    "video/copy-mp4": {
        "ext": "mp4",
        "vcodec": "copy",
        "acodec": "aac",
        "pix_fmt": None,
        "mime": "video/mp4",
    },
}

META_EXTS = {"mp4", "mkv", "mov", "webm"}
_ENCODERS_CACHE = None


def _get_encoders():
    """
    Множество доступных видеоэнкодеров ffmpeg.
    """
    global _ENCODERS_CACHE

    if _ENCODERS_CACHE is not None:
        return _ENCODERS_CACHE

    if not _ffmpeg_available():
        _ENCODERS_CACHE = set()
        return _ENCODERS_CACHE

    enc = set()

    try:
        proc = subprocess.run(
            [FFMPEG_PATH, "-nostdin", "-hide_banner", "-encoders"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )

        for m in re.finditer(r"^\s*V[\w.]{5}\s+([\w-]+)", proc.stdout or "", re.M):
            enc.add(m.group(1))
    except Exception as e:
        logger.warning(f"[AGSoft Video Save] encoder probe failed: {e}")

    _ENCODERS_CACHE = enc
    return enc


def _available_formats():
    """
    Список форматов для combo.
    """
    enc = _get_encoders()
    all_keys = list(FORMAT_PRESETS.keys())

    if not enc:
        return all_keys

    avail = []

    for k, f in FORMAT_PRESETS.items():
        if f["vcodec"] == "copy" or f["vcodec"] in enc:
            avail.append(k)

    if "video/h264-mp4" not in avail:
        avail.insert(0, "video/h264-mp4")

    if "video/copy-mp4" not in avail:
        avail.append("video/copy-mp4")

    return avail or all_keys


def _unwrap_workflow(obj):
    """
    Разворачивает обёртку {"workflow": <graph>} в чистый граф.
    """
    if isinstance(obj, dict) and "workflow" in obj and "nodes" not in obj:
        return obj["workflow"]
    return obj


def _norm_subfolder(s):
    """
    Санация подпапки.
    """
    if not s:
        return ""

    parts = []

    for p in str(s).replace("\\", "/").split("/"):
        p = p.strip()
        if p and p not in (".", ".."):
            parts.append(p)

    return "/".join(parts)


def _vcodec_args(f, crf, preset="veryfast"):
    """
    Аргументы видеокодера.
    """
    v = f.get("vcodec")

    if v == "copy":
        return ["-c:v", "copy"]

    args = ["-c:v", v]

    if f.get("pix_fmt"):
        args += ["-pix_fmt", f["pix_fmt"]]

    nvenc_preset_map = {
        "ultrafast": "p1",
        "superfast": "p1",
        "veryfast": "p2",
        "faster": "p3",
        "fast": "p4",
        "medium": "p5",
        "slow": "p6",
    }

    if v in ("libx264", "libx265"):
        args += ["-preset", preset, "-crf", str(crf)]
    elif v in ("libvpx-vp9",):
        args += ["-crf", str(crf), "-deadline", "realtime", "-cpu-used", "5"]
    elif v in ("libsvtav1",):
        args += ["-crf", str(crf), "-preset", "8"]
    elif v in ("h264_nvenc", "hevc_nvenc"):
        args += ["-rc", "vbr", "-cq", str(crf), "-preset", nvenc_preset_map.get(preset, "p4")]

    return args


def _audio_args(f):
    """
    Аргументы аудиокодера.
    """
    a = f.get("acodec")

    if a is None:
        return None

    if a in ("pcm_s16le",):
        return ["-c:a", a]

    return ["-c:a", a, "-b:a", "192k"]


def _movflags_args(f):
    """
    faststart для MP4/MOV.
    """
    if f.get("ext") in ("mp4", "mov"):
        return ["-movflags", "+faststart"]
    return []


def _extract_video_path(video_obj):
    """
    Извлекает путь из VIDEO-объекта.
    """
    if video_obj is None:
        return ""

    try:
        if VideoFromFile is not None and isinstance(video_obj, VideoFromFile):
            path = getattr(video_obj, "_VideoFromFile__file", None)
            if path and isinstance(path, str) and os.path.exists(path):
                return os.path.abspath(path)
    except Exception:
        pass

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

    try:
        if hasattr(video_obj, "get_path") and callable(video_obj.get_path):
            path = video_obj.get_path()
            if isinstance(path, str) and os.path.exists(path):
                return os.path.abspath(path)
    except Exception:
        pass

    try:
        if hasattr(video_obj, "get_stream_source") and callable(video_obj.get_stream_source):
            source = video_obj.get_stream_source()
            if isinstance(source, str) and os.path.exists(source):
                return os.path.abspath(source)
    except Exception:
        pass

    return ""


def _run_ffmpeg(cmd, timeout=None):
    """
    Запуск ffmpeg с таймаутом, -nostdin и без наследования stdin.
    """
    if timeout is None:
        timeout = MAX_FFMPEG_TIMEOUT

    cmd = list(cmd)

    if cmd and cmd[0] == FFMPEG_PATH:
        cmd = [FFMPEG_PATH, "-nostdin"] + cmd[1:]

    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timeout after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")


_VIDEO_INFO_CACHE = {}


def _get_video_info(path):
    """
    Быстрый probe видео через ffmpeg -i.
    """
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        key = (path, 0, 0)

    cached = _VIDEO_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    info = {
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "codec": "",
        "has_audio": False,
        "audio_codec": "",
    }

    if not _ffmpeg_available():
        _VIDEO_INFO_CACHE[key] = info
        return info

    try:
        proc = subprocess.run(
            [FFMPEG_PATH, "-nostdin", "-hide_banner", "-i", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=PROBE_TIMEOUT,
        )

        err = proc.stderr or ""

        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if m:
            info["duration"] = (
                int(m.group(1)) * 3600 +
                int(m.group(2)) * 60 +
                float(m.group(3))
            )

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


def _base_read_dirs():
    dirs = []

    for fn in (
        getattr(folder_paths, "get_input_directory", None),
        getattr(folder_paths, "get_output_directory", None),
        getattr(folder_paths, "get_temp_directory", None),
    ):
        if fn is None:
            continue
        try:
            d = fn()
            if d:
                dirs.append(os.path.abspath(d))
        except Exception:
            pass

    return dirs


def _read_path_allowed(path):
    """
    Проверка пути для чтения/превью.
    По умолчанию разрешены любые пути, чтобы не ломать существующие сценарии.
    Для безопасности установите AGSOFT_ALLOW_ANY_PATH=0.
    """
    if ALLOW_ANY_READ_PATH:
        return True

    p = os.path.abspath(path)

    for base in _base_read_dirs():
        try:
            if os.path.commonpath([p, base]) == base:
                return True
        except Exception:
            continue

    return False


def _path_allowed(path):
    """
    Разрешаем вшивку метаданных только в output/temp.
    """
    p = os.path.abspath(path)

    for base in (
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
    ):
        try:
            if os.path.commonpath([p, os.path.abspath(base)]) == os.path.abspath(base):
                return True
        except Exception:
            continue

    return False


def _cleanup_temp_previews(tmp_dir, max_age=3600):
    """
    Удаляет старые временные no-op превью.
    """
    try:
        now = time.time()

        for name in os.listdir(tmp_dir):
            if name.startswith("agsoft_preview_") and name.endswith(".mp4"):
                p = os.path.join(tmp_dir, name)
                try:
                    if os.path.isfile(p) and now - os.path.getmtime(p) > max_age:
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass


async def _stream_file_range(request, path):
    """
    Отдача файла с Range.
    """
    if not os.path.isfile(path):
        return web.Response(status=404)

    if not _read_path_allowed(path):
        return web.Response(status=403)

    try:
        file_size = await asyncio.to_thread(os.path.getsize, path)
    except Exception:
        return web.Response(status=500)

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
    f = None

    try:
        f = await asyncio.to_thread(open, path, "rb")
        await asyncio.to_thread(f.seek, start)

        remaining = count

        while remaining > 0:
            chunk = await asyncio.to_thread(f.read, min(CHUNK, remaining))
            if not chunk:
                break

            await resp.write(chunk)
            remaining -= len(chunk)

    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError, OSError):
        pass
    finally:
        if f is not None:
            try:
                await asyncio.to_thread(f.close)
            except Exception:
                pass

    try:
        await resp.write_eof()
    except Exception:
        pass

    return resp


@PromptServer.instance.routes.get("/agsoft/stream_path")
async def agsoft_stream_path(request):
    """
    Отдача файла по абсолютному пути.
    """
    path = request.query.get("path", "")

    if not path:
        return web.Response(status=400)

    path = os.path.abspath(path)

    if not _read_path_allowed(path):
        return web.Response(status=403)

    return await _stream_file_range(request, path)


async def _stop_ffmpeg(proc):
    """
    Корректная остановка ffmpeg.
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
    Живой транскод файла для no-op превью.
    """
    global _PREVIEW_ACTIVE

    path = request.query.get("path", "")

    if not path:
        return web.Response(status=400)

    path = os.path.abspath(path)

    if not os.path.isfile(path):
        return web.Response(status=404)

    if not _read_path_allowed(path):
        return web.Response(status=403)

    if not _ffmpeg_available():
        return web.Response(status=503, text="ffmpeg not available")

    if _PREVIEW_ACTIVE >= _PREVIEW_MAX:
        return web.Response(status=503, text="preview transcode busy")

    try:
        start = float(request.query.get("start", "0") or 0)
    except ValueError:
        start = 0.0

    if start < 0:
        start = 0.0

    cmd = [FFMPEG_PATH, "-nostdin", "-hide_banner", "-loglevel", "error"]

    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]

    cmd += ["-i", path, "-map", "0:v:0", "-map", "0:a:0?"]

    if start > 0:
        cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += ["-c:v", "copy"]

    cmd += [
        "-c:a", "aac",
        "-b:a", "128k",
        "-async", "1",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov",
        "-",
    ]

    _PREVIEW_ACTIVE += 1
    proc = None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
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

        try:
            await resp.write_eof()
        except Exception:
            pass

        return resp

    finally:
        if proc is not None:
            await _stop_ffmpeg(proc)

        _PREVIEW_ACTIVE -= 1


def _convert_video_silent(src, out_file, fps, crf, f, preset):
    """
    Конвертация видео без звука.
    """
    if f.get("vcodec") == "copy":
        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-i", src,
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
        ]
        cmd += _movflags_args(f)
        cmd.append(out_file)
        _run_ffmpeg(cmd)
        return

    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:v:0",
    ]

    cmd += _vcodec_args(f, crf, preset)

    if fps:
        cmd += ["-r", str(fps)]

    cmd += _movflags_args(f)
    cmd.append(out_file)

    _run_ffmpeg(cmd)


def _convert_video_with_audio(src, audio_wav, out_file, fps, crf, f, preset):
    """
    Конвертация видео + внешний звук за один проход.
    """
    a = _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]

    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-i", audio_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]

    cmd += _vcodec_args(f, crf, preset)
    cmd += a

    if f.get("vcodec") != "copy" and fps:
        cmd += ["-r", str(fps)]

    cmd += ["-shortest"]
    cmd += _movflags_args(f)
    cmd.append(out_file)

    _run_ffmpeg(cmd)


def _convert_video_with_src_audio(src, out_file, fps, crf, f, preset):
    """
    Конвертация видео со звуком исходника за один проход.
    """
    a = _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]

    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:v:0",
        "-map", "0:a:0",
    ]

    cmd += _vcodec_args(f, crf, preset)
    cmd += a

    if f.get("vcodec") != "copy" and fps:
        cmd += ["-r", str(fps)]

    cmd += ["-shortest"]
    cmd += _movflags_args(f)
    cmd.append(out_file)

    _run_ffmpeg(cmd)


def _strip_audio_copy(src, out_file):
    """
    Быстро создаёт копию видео без звука.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        out_file,
    ])


def _remux_video_with_audio(src, audio_wav, out_file):
    """
    Быстрый ремукс для no-op превью.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-i", audio_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        out_file,
    ])


def _frames_pipe_supported(f):
    return f.get("vcodec") not in ("gif", "libwebp", "copy")


def _save_frames_video_png(frames, out_file, fps, crf, f, preset, audio_wav=None):
    """
    fallback для GIF/WebP через временные PNG.
    """
    if not frames:
        raise ValueError("No frames to save")

    tmp_dir = tempfile.mkdtemp(prefix="agsoft_video_save")

    try:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmp_dir, f"frame_{i:06d}.png"), "PNG")

        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        ]

        if audio_wav:
            cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        else:
            cmd += ["-map", "0:v:0"]

        cmd += _vcodec_args(f, crf, preset)

        if audio_wav:
            cmd += _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]

        cmd += _movflags_args(f)
        cmd.append(out_file)

        _run_ffmpeg(cmd)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_frames_video_pipe(frames, out_file, fps, crf, f, preset, audio_wav=None):
    """
    Быстрое сохранение кадров через rawvideo pipe без PNG.
    """
    if not frames:
        raise ValueError("No frames to save")

    if not _frames_pipe_supported(f):
        return _save_frames_video_png(frames, out_file, fps, crf, f, preset, audio_wav)

    w, h = frames[0].size

    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
    ]

    if audio_wav:
        cmd += ["-i", audio_wav, "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    else:
        cmd += ["-map", "0:v:0"]

    cmd += _vcodec_args(f, crf, preset)

    if audio_wav:
        cmd += _audio_args(f) or ["-c:a", "aac", "-b:a", "192k"]

    cmd += _movflags_args(f)
    cmd.append(out_file)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        try:
            for frame in frames:
                if frame.size != (w, h):
                    frame = frame.resize((w, h))

                if frame.mode != "RGB":
                    frame = frame.convert("RGB")

                proc.stdin.write(frame.tobytes())

            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        stderr_data = proc.communicate(timeout=MAX_FFMPEG_TIMEOUT)[1]

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr_data.decode('utf-8', 'ignore')[-800:]}")

    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

            try:
                proc.wait()
            except Exception:
                pass


def _save_frames_video(frames, out_file, fps, crf, f, preset, audio_wav=None):
    """
    Универсальный вход для кадров.
    """
    if _frames_pipe_supported(f):
        _save_frames_video_pipe(frames, out_file, fps, crf, f, preset, audio_wav)
    else:
        _save_frames_video_png(frames, out_file, fps, crf, f, preset, audio_wav)


def _extract_audio(src, out_file):
    """
    Извлечь звуковую дорожку исходника в M4A.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-map", "0:a:0",
        "-c:a", "aac",
        "-b:a", "192k",
        out_file,
    ])


def _encode_audio(wav, out_file):
    """
    Перекодировать WAV в M4A.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", wav,
        "-c:a", "aac",
        "-b:a", "192k",
        out_file,
    ])


def _extract_first_frame(src, out_png):
    """
    Первый кадр видео в PNG.
    """
    _run_ffmpeg([
        FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-frames:v", "1",
        out_png,
    ])


def _save_audio_to_temp(audio):
    """
    Сохраняет AUDIO во временный WAV.
    """
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
    PNG tEXt chunk.
    """
    body = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1")
    crc = zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", crc)


def _embed_png_metadata(png_path, prompt, extra_pnginfo):
    """
    Вшивает workflow/prompt в PNG.
    """
    try:
        workflow_json = None

        if extra_pnginfo is not None:
            workflow_json = json.dumps(_unwrap_workflow(extra_pnginfo), ensure_ascii=True)

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

            if ctype == b"tEXt":
                body = data[pos + 8:pos + 8 + length]
                sep = body.find(b"\x00")
                kw = body[:sep] if sep != -1 else body

                if kw in (b"workflow", b"prompt"):
                    pos = chunk_end
                    continue

            out.append(chunk)

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
            f"(workflow={len(workflow_json or '')} bytes)"
        )

    except Exception as e:
        logger.warning(f"PNG metadata embed failed for {png_path}: {e}")


def _embed_metadata(video_file, meta_dict):
    """
    Вшивает метаданные в видео через ffmetadata remux.
    """
    ext = os.path.splitext(video_file)[1].lstrip(".").lower()

    fd, meta_path = tempfile.mkstemp(prefix="agsoft_meta_", suffix=".ffmeta")
    os.close(fd)

    fd, tmp_out = tempfile.mkstemp(
        prefix="agsoft_meta_",
        suffix=f".{ext}",
        dir=os.path.dirname(os.path.abspath(video_file)),
    )
    os.close(fd)

    try:
        raw = json.dumps(meta_dict, ensure_ascii=False)

        def esc(s):
            return (
                s.replace("\\", "\\\\")
                .replace("=", "\\=")
                .replace(";", "\\;")
                .replace("#", "\\#")
                .replace("\n", "\\n")
            )

        with open(meta_path, "w", encoding="utf-8") as fh:
            fh.write(";FFMETADATA1\n")
            fh.write("comment=" + esc(raw) + "\n")

        cmd = [
            FFMPEG_PATH, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_file,
            "-i", meta_path,
            "-map", "0",
            "-c", "copy",
            "-map_metadata", "1",
        ]

        if ext in ("mp4", "mov"):
            cmd += ["-movflags", "use_metadata_tags+faststart"]

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


def _comment_to_workflow(raw):
    if not raw:
        return None

    try:
        data = json.loads(raw)
    except Exception:
        return None

    if isinstance(data, dict):
        return _unwrap_workflow(data.get("workflow"))

    return _unwrap_workflow(data)


def _iter_boxes(buf, start, end):
    p = start

    while p + 8 <= end:
        n = struct.unpack(">I", buf[p:p + 4])[0]
        t = buf[p + 4:p + 8]

        if n < 8:
            break

        yield t, p + 8, min(p + n, end)
        p += n


def _parse_data_box(item):
    q = 0

    while q + 8 <= len(item):
        n = struct.unpack(">I", item[q:q + 4])[0]

        if n < 8 or q + n > len(item):
            break

        if item[q + 4:q + 8] == b"data":
            return item[q + 16:q + n].decode("utf-8", "ignore")

        q += n

    return None


def _extract_mp4_comment(path):
    """
    Нативный парсер MP4/MOV.
    """
    with open(path, "rb") as f:
        f.seek(0, 2)
        total = f.tell()
        f.seek(0)

        moov = None

        while f.tell() + 8 <= total:
            hdr = f.read(8)

            if len(hdr) < 8:
                break

            n = struct.unpack(">I", hdr[:4])[0]
            t = hdr[4:8]
            h = 8

            if n == 1:
                n = struct.unpack(">Q", f.read(8))[0]
                h = 16
            elif n == 0:
                n = total - (f.tell() - 8)

            if n < h:
                break

            if t == b"moov":
                moov = f.read(n - h)
                break

            f.seek(n - h, 1)

        if not moov:
            return None

        udta = None

        for t, s, e in _iter_boxes(moov, 0, len(moov)):
            if t == b"udta":
                udta = moov[s:e]
                break

        if not udta:
            return None

        keys, ilst = {}, None

        for t, s, e in _iter_boxes(udta, 0, len(udta)):
            if t != b"meta":
                continue

            body = udta[s + 4:e]

            for t2, s2, e2 in _iter_boxes(body, 0, len(body)):
                if t2 == b"keys":
                    cnt = struct.unpack(">I", body[s2 + 4:s2 + 8])[0]
                    p = s2 + 8

                    for i in range(1, cnt + 1):
                        if p + 8 > e2:
                            break

                        n = struct.unpack(">I", body[p:p + 4])[0]

                        if n < 8 or p + n > e2:
                            break

                        if body[p + 4:p + 8] == b"key":
                            keys[i] = body[p + 12:p + n].decode("utf-8", "ignore")

                        p += n

                elif t2 == b"ilst":
                    ilst = body[s2:e2]

        if ilst is None:
            for t, s, e in _iter_boxes(udta, 0, len(udta)):
                if t == b"\xa9cmt":
                    return _parse_data_box(udta[s:e])

            return None

        p = 0
        end = len(ilst)

        while p + 8 <= end:
            n = struct.unpack(">I", ilst[p:p + 4])[0]
            idx = struct.unpack(">I", ilst[p + 4:p + 8])[0]

            if n < 8 or p + n > end:
                break

            if keys.get(idx) == "comment":
                return _parse_data_box(ilst[p + 8:p + n])

            p += n

        return None


def _extract_mkv_comment(path):
    """
    Нативный парсер EBML/MKV/WebM.
    """
    SEG, TAGS, TAG, SIMPLE, NAME, STR = (
        0x18538067,
        0x125456A4,
        0x7373,
        0x67C8,
        0x45A3,
        0x4487,
    )

    with open(path, "rb") as f:
        f.seek(0, 2)
        limit = f.tell()
        f.seek(0)

        def read_hdr():
            b = f.read(1)

            if not b:
                return None, None

            b0 = b[0]
            il = next((l for l in range(1, 9) if b0 & (0x80 >> (l - 1))), 0)

            if not il:
                return None, None

            eid = b0

            for _ in range(il - 1):
                nb = f.read(1)
                if not nb:
                    return None, None
                eid = (eid << 8) | nb[0]

            sb = f.read(1)

            if not sb:
                return None, None

            s0 = sb[0]
            sl = next((l for l in range(1, 9) if s0 & (0x80 >> (l - 1))), 0)

            if not sl:
                return None, None

            size = s0 & (0xFF >> sl)

            for _ in range(sl - 1):
                nb = f.read(1)
                if not nb:
                    return None, None
                size = (size << 8) | nb[0]

            return eid, size

        def scan(limit):
            name = None

            while f.tell() < limit:
                eid, size = read_hdr()

                if eid is None:
                    return None

                nxt = f.tell() + size

                if nxt > limit:
                    if eid in (SEG, TAGS, TAG, SIMPLE):
                        r = scan(limit)
                        if r:
                            return r
                        f.seek(limit)
                        return None

                    return None

                if eid == NAME:
                    name = f.read(size).decode("utf-8", "ignore")
                    continue

                if eid == STR:
                    val = f.read(size).decode("utf-8", "ignore")
                    if name == "comment":
                        return val
                    continue

                if eid in (SEG, TAGS, TAG, SIMPLE):
                    r = scan(nxt)
                    if r:
                        return r

                f.seek(nxt)

            return None

        return scan(limit)


def _extract_workflow_sync(tmp, ext):
    """
    Синхронное извлечение воркфлоу.
    """
    workflow = None

    try:
        if ext in ("mp4", "mov", "m4v"):
            workflow = _comment_to_workflow(_extract_mp4_comment(tmp))
        elif ext in ("mkv", "webm"):
            workflow = _comment_to_workflow(_extract_mkv_comment(tmp))
    except Exception as e:
        logger.warning(f"[AGSoft Video Save] native meta parse failed: {e}")

    if workflow is None and _ffmpeg_available():
        try:
            proc = subprocess.run(
                [
                    FFMPEG_PATH,
                    "-nostdin",
                    "-hide_banner",
                    "-i", tmp,
                    "-f", "ffmetadata",
                    "-",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=30,
            )

            workflow = _parse_ffmetadata_comment(proc.stdout or "")
        except Exception as e:
            logger.warning(f"[AGSoft Video Save] ffmpeg meta fallback failed: {e}")

    return workflow


@PromptServer.instance.routes.post("/agsoft/embed_workflow")
async def agsoft_embed_workflow(request):
    """
    JS endpoint для вшивки воркфлоу.
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
            if workflow is None and prompt is None:
                return web.json_response({"ok": True})

            await asyncio.to_thread(_embed_png_metadata, path, prompt, workflow)

        elif ext in META_EXTS:
            meta = {}

            if prompt is not None:
                meta["prompt"] = prompt

            if workflow is not None:
                meta["workflow"] = workflow

            if meta:
                await asyncio.to_thread(_embed_metadata, path, meta)

        else:
            return web.json_response({"ok": False, "error": f"unsupported format: {ext}"})

        return web.json_response({"ok": True})

    except Exception as e:
        logger.warning(f"[AGSoft Video Save] embed_workflow failed: {e}")
        return web.json_response({"ok": False, "error": str(e)})


@PromptServer.instance.routes.post("/agsoft/extract_workflow")
async def agsoft_extract_workflow(request):
    """
    JS endpoint для восстановления воркфлоу из перетащенного видео.
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

        suffix = os.path.splitext(field.filename or "")[1] or ".mp4"

        fd, tmp = tempfile.mkstemp(
            prefix="agsoft_drop_",
            suffix=suffix,
        )
        os.close(fd)

        written = 0
        too_large = False

        out = await asyncio.to_thread(open, tmp, "wb")

        try:
            while True:
                chunk = await field.read_chunk(1024 * 1024)

                if not chunk:
                    break

                written += len(chunk)

                if written > DROP_MAX_BYTES:
                    too_large = True
                    break

                await asyncio.to_thread(out.write, chunk)
        finally:
            try:
                await asyncio.to_thread(out.close)
            except Exception:
                pass

        if too_large:
            return web.json_response({"workflow": None, "error": "file too large"})

        ext = os.path.splitext(field.filename or "")[1].lstrip(".").lower()
        workflow = await asyncio.to_thread(_extract_workflow_sync, tmp, ext)

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
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "AGSoft_Video",
                        "tooltip": (
                            "Prefix for the output filename. A timestamp is added automatically.\n"
                            "---\n"
                            "Префикс имени файла. Временная метка добавляется автоматически."
                        ),
                    },
                ),

                "subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Subfolder inside the output (or temp) directory. Nested paths are allowed: "
                            "'MiniMax' or 'MiniMax/2026'. '..' and unsafe segments are sanitized.\n"
                            "Example: MiniMax -> output/MiniMax/AGSoft_Video_<timestamp>.mp4\n"
                            "---\n"
                            "Подпапка внутри папки output (или temp). Разрешена вложенность: "
                            "'MiniMax' или 'MiniMax/2026'. '..' и небезопасные сегменты санациируются.\n"
                            "Пример: MiniMax -> output/MiniMax/AGSoft_Video_<timestamp>.mp4"
                        ),
                    },
                ),

                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional: absolute path to a video file to convert.\n"
                            "---\n"
                            "Опционально: абсолютный путь к видеофайлу для конвертации."
                        ),
                    },
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
                        ),
                    },
                ),

                "format": (
                    _available_formats(),
                    {
                        "default": "video/h264-mp4",
                        "tooltip": (
                            "Output format / codec. The list shows only encoders available in your ffmpeg build.\n"
                            "video/copy-mp4 copies the video stream without re-encoding when possible.\n"
                            "---\n"
                            "Формат / кодек выхода. В списке показываются только энкодеры, доступные в вашей сборке ffmpeg.\n"
                            "video/copy-mp4 копирует видеопоток без перекодирования, если это возможно."
                        ),
                    },
                ),

                "preset": (
                    [
                        "ultrafast",
                        "superfast",
                        "veryfast",
                        "faster",
                        "fast",
                        "medium",
                        "slow",
                    ],
                    {
                        "default": "veryfast",
                        "tooltip": (
                            "Encoding speed / quality preset for CPU encoders. "
                            "Faster presets encode quicker but may produce larger files.\n"
                            "---\n"
                            "Пресет скорости / качества для CPU-энкодеров. "
                            "Быстрые пресеты кодируют быстрее, но файл может получиться больше."
                        ),
                    },
                ),

                "crf": (
                    "INT",
                    {
                        "default": 19,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": (
                            "Constant Rate Factor: 0 = lossless, 51 = worst. "
                            "Ignored for GIF/WebP/FFV1/ProRes/copy modes.\n"
                            "---\n"
                            "Constant Rate Factor: 0 = без потерь, 51 = худшее качество. "
                            "Игнорируется для GIF/WebP/FFV1/ProRes/copy-режимов."
                        ),
                    },
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
                        ),
                    },
                ),

                "save_video_with_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save video WITH audio. Audio is taken from the audio input if connected, "
                            "otherwise from the source video's own audio track.\n"
                            "---\n"
                            "Сохранить видео СО звуком. Звук берётся со входа audio, если он подключён, "
                            "иначе — из звуковой дорожки самого исходного видео."
                        ),
                    },
                ),

                "save_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save the audio track as a SEPARATE M4A file.\n"
                            "---\n"
                            "Сохранить звуковую дорожку ОТДЕЛЬНЫМ файлом M4A."
                        ),
                    },
                ),

                "save_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Save the first frame as a PNG image. With save_metadata enabled the PNG also "
                            "carries the workflow (tEXt chunk after IHDR) and opens it on drag & drop.\n"
                            "---\n"
                            "Сохранить первый кадр как PNG. При включённом save_metadata в PNG вшивается "
                            "воркфлоу (tEXt-чанк после IHDR), и он открывается перетаскиванием."
                        ),
                    },
                ),

                "save_metadata": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Embed the workflow (PURE graph, no wrapper) into the saved file's metadata. "
                            "Dragging the saved PNG onto the canvas opens it natively; dragging the saved "
                            "video opens it via this extension.\n"
                            "---\n"
                            "Вшить воркфлоу (ЧИСТЫЙ граф, без обёртки) в метаданные сохранённого файла. "
                            "Перетаскивание сохранённого PNG на канвас открывает его штатно; "
                            "сохранённого видео — через это расширение."
                        ),
                    },
                ),

                "save_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True = save to the output folder; False = save to temp (preview only, "
                            "no clutter in output).\n"
                            "---\n"
                            "True = сохранять в папку output; False = во временную папку (только превью, "
                            "без мусора в output)."
                        ),
                    },
                ),
            },

            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Frame sequence to save as video. Used when no video / video_path inputs "
                            "are connected.\n"
                            "---\n"
                            "Последовательность кадров для сохранения. Используется, когда не подключены "
                            "входы video / video_path."
                        ),
                    },
                ),

                "video": (
                    IO.VIDEO,
                    {
                        "tooltip": (
                            "Optional: VIDEO object to convert (highest priority). Its own audio track is "
                            "used when the audio input is not connected.\n"
                            "---\n"
                            "Опционально: объект VIDEO для конвертации (наивысший приоритет). Его собственная "
                            "звуковая дорожка используется, когда вход audio не подключён."
                        ),
                    },
                ),

                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional: AUDIO to mux into the video / save as M4A. Highest priority over the "
                            "source video's own audio track. Also muxed into the no-op preview so you HEAR "
                            "the result before saving.\n"
                            "---\n"
                            "Опционально: AUDIO для подмешивания в видео / сохранения в M4A. Наивысший приоритет "
                            "перед собственной дорожкой исходного видео. Также подмешивается в no-op превью — "
                            "результат СЛЫШНО до сохранения."
                        ),
                    },
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
    OUTPUT_NODE = True
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "🎬 AGSoft Video Save.\n"
        "Saves video from an image sequence OR converts an existing video. "
        "Saves ONLY what you enable: video without audio, video with audio, audio as a separate M4A, "
        "first frame — any combination, no extra files.\n"
        "NO-OP MODE WITH PREVIEW: all save options off = nothing written to output, no error, and the "
        "source preview is shown INCLUDING the audio input (fast remux/encode to temp).\n"
        "Priority: video > video_path > images.\n"
        "Audio is taken from the audio input if connected, otherwise from the source video's own track.\n"
        "Save path: output (or temp) + subfolder + filename_prefix_<timestamp>.\n"
        "save_metadata embeds the workflow as a PURE graph into the saved files; dragging the saved "
        "PNG/video onto the canvas restores the workflow.\n"
        "Codec selection: h264/h265, NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV, GIF, WebP, plus "
        "video/copy-mp4 for fast stream copy.\n"
        "The result preview is shown inside the node, serialized into the workflow, and the node resizes "
        "like Load Video.\n"
        "---\n"
        "🎬 AGSoft Video Save.\n"
        "Сохраняет видео из последовательности кадров ИЛИ конвертирует существующее видео. "
        "Сохраняет ТОЛЬКО включённое: видео без звука, видео со звуком, звук отдельным M4A, первый кадр — "
        "любые комбинации, без лишних файлов.\n"
        "РЕЖИМ NO-OP С ПРЕВЬЮ: все опции сохранения выключены = output не трогается, без ошибки, а превью "
        "источника показывается СО звуком со входа audio (быстрый ремукс/энкод в temp).\n"
        "Приоритет: video > video_path > images.\n"
        "Звук берётся со входа audio, если он подключён, иначе — из дорожки самого исходного видео.\n"
        "Путь сохранения: output (или temp) + subfolder + filename_prefix_<timestamp>.\n"
        "save_metadata вшивает воркфлоу ЧИСТЫМ графом в сохранённые файлы; перетаскивание сохранённого "
        "PNG/видео на канвас восстанавливает воркфлоу.\n"
        "Выбор кодеков: h264/h265, NVENC, VP9/AV1 WebM, FFV1 MKV, ProRes MOV, GIF, WebP, плюс "
        "video/copy-mp4 для быстрого копирования потока.\n"
        "Превью результата показывается в ноде, сериализуется в воркфлоу, ресайз ноды — как в Load Video."
    )

    def save_video(
        self,
        filename_prefix,
        subfolder,
        video_path,
        frame_rate,
        format,
        preset,
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
            # NO-OP preview
            # ------------------------------------------------------------------
            if not (save_video or save_video_with_audio or save_audio or save_image):
                logger.info("[AGSoft Video Save] All save options disabled — no-op preview.")

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

                _cleanup_temp_previews(tmp_dir)

                tmp_file = os.path.join(
                    tmp_dir,
                    f"agsoft_preview_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                )

                try:
                    if src_preview:
                        if audio_wav is not None:
                            try:
                                _remux_video_with_audio(src_preview, audio_wav, tmp_file)
                                preview = {
                                    "kind": "file",
                                    "filename": os.path.basename(tmp_file),
                                    "subfolder": "",
                                    "type": "temp",
                                    "format": "video/mp4",
                                    "fullpath": tmp_file,
                                    "noop": True,
                                    "workflow_embedded": False,
                                    "supports_workflow": False,
                                }
                            except Exception:
                                preview = None

                        if preview is None:
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
                                "noop": True,
                                "workflow_embedded": False,
                                "supports_workflow": False,
                            }

                    elif images is not None and len(images) > 0:
                        frames = []

                        for img_tensor in images:
                            arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                            frames.append(Image.fromarray(arr, "RGB"))

                        _save_frames_video(
                            frames,
                            tmp_file,
                            frame_rate,
                            23,
                            FORMAT_PRESETS["video/h264-mp4"],
                            "veryfast",
                            audio_wav,
                        )

                        preview = {
                            "kind": "file",
                            "filename": os.path.basename(tmp_file),
                            "subfolder": "",
                            "type": "temp",
                            "format": "video/mp4",
                            "fullpath": tmp_file,
                            "noop": True,
                            "workflow_embedded": False,
                            "supports_workflow": False,
                        }

                except Exception as e:
                    logger.warning(f"[AGSoft Video Save] no-op preview failed: {e}")

                return {
                    "ui": {"gifs": [preview] if preview else []},
                    "result": ("",),
                }

            # ------------------------------------------------------------------
            # Обычный режим сохранения
            # ------------------------------------------------------------------
            f = FORMAT_PRESETS.get(format, FORMAT_PRESETS["video/h264-mp4"])

            if f.get("acodec") is None:
                if save_video_with_audio:
                    logger.warning(f"{format} cannot hold audio — save_video_with_audio skipped.")
                    save_video_with_audio = False

                if save_audio:
                    logger.warning(f"{format} cannot hold audio — save_audio skipped.")
                    save_audio = False

            wf_clean = None

            if save_metadata and extra_pnginfo is not None:
                wf_clean = _unwrap_workflow(extra_pnginfo)

            src_video = None

            if video is not None:
                src_video = _extract_video_path(video)

                if not src_video:
                    raise ValueError("[AGSoft Video Save] Could not extract path from VIDEO object.")

                logger.info(f"Converting VIDEO object: {src_video}")

            elif video_path and os.path.exists(video_path):
                src_video = os.path.abspath(video_path)
                logger.info(f"Converting video path: {src_video}")

            frames = None

            if src_video is None:
                if images is None or len(images) == 0:
                    raise ValueError("[AGSoft Video Save] No source: connect video, video_path or images.")

                if f.get("vcodec") == "copy":
                    logger.warning("video/copy-mp4 is not supported for image frames. Falling back to video/h264-mp4.")
                    format = "video/h264-mp4"
                    f = FORMAT_PRESETS[format]

                frames = []

                for img_tensor in images:
                    arr = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    frames.append(Image.fromarray(arr, "RGB"))

                logger.info(f"Saving {len(frames)} frames from IMAGE tensor")

            # ------------------------------------------------------------------
            # Probe звука исходника делаем только если он реально нужен.
            # ------------------------------------------------------------------
            has_src_audio = False

            need_src_audio_probe = (
                src_video is not None and
                audio is None and
                (save_video_with_audio or save_audio)
            )

            if need_src_audio_probe:
                has_src_audio = bool(_get_video_info(src_video).get("has_audio"))

            if audio is not None:
                audio_wav = _save_audio_to_temp(audio)
                logger.info("Audio source: audio input")
            elif has_src_audio:
                logger.info("Audio source: source video audio track")

            have_audio = (audio_wav is not None) or has_src_audio

            if save_video_with_audio and not have_audio:
                logger.warning("save_video_with_audio enabled but no audio available — skipping.")
                save_video_with_audio = False

            if save_audio and not have_audio:
                logger.warning("save_audio enabled but no audio available — skipping.")
                save_audio = False

            # ------------------------------------------------------------------
            # Output / temp
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

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{filename_prefix}_{timestamp}"

            video_file = None
            video_with_audio = None
            audio_file = None
            saved_image = None

            silent_out = os.path.join(output_dir, f"{base_name}.{f['ext']}")
            audio_out = os.path.join(output_dir, f"{base_name}_audio.{f['ext']}")

            # ------------------------------------------------------------------
            # Источник: видео
            # ------------------------------------------------------------------
            if src_video is not None:
                if save_video and save_video_with_audio:
                    if audio_wav is not None:
                        _convert_video_with_audio(
                            src_video,
                            audio_wav,
                            audio_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                        )
                    else:
                        _convert_video_with_src_audio(
                            src_video,
                            audio_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                        )

                    video_with_audio = audio_out

                    try:
                        _strip_audio_copy(audio_out, silent_out)
                        video_file = silent_out
                    except Exception:
                        _convert_video_silent(
                            src_video,
                            silent_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                        )
                        video_file = silent_out

                else:
                    if save_video:
                        _convert_video_silent(
                            src_video,
                            silent_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                        )
                        video_file = silent_out

                    if save_video_with_audio:
                        if audio_wav is not None:
                            _convert_video_with_audio(
                                src_video,
                                audio_wav,
                                audio_out,
                                frame_rate,
                                crf,
                                f,
                                preset,
                            )
                        else:
                            _convert_video_with_src_audio(
                                src_video,
                                audio_out,
                                frame_rate,
                                crf,
                                f,
                                preset,
                            )

                        video_with_audio = audio_out

                if save_audio:
                    audio_file = os.path.join(output_dir, f"{base_name}_sound.m4a")

                    if audio_wav is not None:
                        _encode_audio(audio_wav, audio_file)
                    else:
                        _extract_audio(src_video, audio_file)

                if save_image:
                    saved_image = os.path.join(output_dir, f"{base_name}.png")
                    _extract_first_frame(src_video, saved_image)

            # ------------------------------------------------------------------
            # Источник: кадры
            # ------------------------------------------------------------------
            else:
                if save_video and save_video_with_audio and audio_wav is not None:
                    _save_frames_video(
                        frames,
                        audio_out,
                        frame_rate,
                        crf,
                        f,
                        preset,
                        audio_wav,
                    )

                    video_with_audio = audio_out

                    try:
                        _strip_audio_copy(audio_out, silent_out)
                        video_file = silent_out
                    except Exception:
                        _save_frames_video(
                            frames,
                            silent_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                            None,
                        )
                        video_file = silent_out

                else:
                    if save_video:
                        _save_frames_video(
                            frames,
                            silent_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                            None,
                        )
                        video_file = silent_out

                    if save_video_with_audio and audio_wav is not None:
                        _save_frames_video(
                            frames,
                            audio_out,
                            frame_rate,
                            crf,
                            f,
                            preset,
                            audio_wav,
                        )
                        video_with_audio = audio_out

                if save_audio and audio_wav is not None:
                    audio_file = os.path.join(output_dir, f"{base_name}_sound.m4a")
                    _encode_audio(audio_wav, audio_file)

                if save_image:
                    saved_image = os.path.join(output_dir, f"{base_name}.png")
                    frames[0].save(saved_image, "PNG")

            # ------------------------------------------------------------------
            # Metadata embed
            # ------------------------------------------------------------------
            embedded_paths = set()

            if save_metadata and (wf_clean is not None or prompt is not None):
                if saved_image is not None:
                    try:
                        _embed_png_metadata(saved_image, prompt, wf_clean)

                        if wf_clean is not None:
                            embedded_paths.add(saved_image)
                    except Exception as e:
                        logger.warning(f"PNG metadata embed failed: {e}")

                meta_dict = {}

                if prompt is not None:
                    meta_dict["prompt"] = prompt

                if wf_clean is not None:
                    meta_dict["workflow"] = wf_clean

                if meta_dict:
                    for vf in (video_with_audio, video_file):
                        if vf is not None and os.path.splitext(vf)[1].lstrip(".").lower() in META_EXTS:
                            try:
                                _embed_metadata(vf, meta_dict)

                                if wf_clean is not None:
                                    embedded_paths.add(vf)
                            except Exception as e:
                                logger.warning(f"Metadata embed failed for {vf}: {e}")

            # ------------------------------------------------------------------
            # Preview
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

            preview_ext = os.path.splitext(preview_file)[1].lstrip(".").lower() if preview_file else ""
            supports_workflow = preview_ext in META_EXTS or preview_ext == "png"

            preview_workflow_embedded = bool(
                save_metadata and
                wf_clean is not None and
                preview_file in embedded_paths
            )

            preview = {
                "kind": "file",
                "filename": os.path.basename(preview_file) if preview_file else "",
                "subfolder": sub_clean,
                "type": out_type,
                "format": preview_mime,
                "frame_rate": frame_rate,
                "fullpath": preview_file or "",
                "noop": False,
                "workflow_embedded": preview_workflow_embedded,
                "supports_workflow": supports_workflow,
            }

            return {
                "ui": {"gifs": [preview]},
                "result": (result_path,),
            }

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
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
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