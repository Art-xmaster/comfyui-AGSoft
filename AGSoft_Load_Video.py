"""
==============================================================================
AGSoft_Load_Video.py
==============================================================================
Нода: 🎬AGSoft Load Video
Описание / Description:
Загружает видеофайл и возвращает объект VIDEO, путь, геометрию, длительность,
fps, битность и цветовое пространство. Кадры (IMAGE) и звук (AUDIO)
извлекаются ЛЕНИВО: только если эти выходы реально подключены дальше.
Приоритет источника: input_video > custom_path > файл из папки input.
Превью в стиле Video Save: панель кнопок над плеером, инфо-футер под ним.
---
Loads a video file and returns the VIDEO object, path, geometry, duration,
fps, bit depth and color space. Frames (IMAGE) and audio (AUDIO) are
extracted LAZILY: only when those outputs are actually connected downstream.
Source priority: input_video > custom_path > file from the input directory.
Video Save-style preview: button row over the player, info footer under it.
Возможности / Features:
⚡ Ленивые выходы: список подключённых выходов читается из графа выполнения,
кадры и звук декодируются только по требованию.
Lazy outputs: connected outputs are read from the execution graph, frames
and audio are decoded on demand only.
⚡ Один проход get_components() для кадров+звука; звук без кадров — отдельным
PCM-потоком без декодирования видео.
One get_components() pass for frames+audio; audio without frames uses a
separate PCM stream with no video decoding.
⚡ Выходы: video, images, audio, video_path, width, height, total_frames,
duration_seconds, fps, bit_depth, color_space.
Outputs: video, images, audio, video_path, width, height, total_frames,
duration_seconds, fps, bit_depth, color_space.
⚡ Точные total_frames/duration_seconds при декодированных кадрах, иначе
оценка из быстрого probe.
Exact total_frames/duration_seconds when frames are decoded, otherwise an
estimate from the fast probe.
⚡ Метаданные через ffmpeg -i без декодирования, кэш только в RAM; ответ
содержит абсолютный путь источника для кнопки Open.
Metadata via ffmpeg -i without decoding, RAM-only cache; the response carries
the absolute source path for the Open button.
⚡ Загрузка одним потоковым multipart: запись чанками вне event loop во
временный .part, атомарное переименование, прогресс на кнопке.
Single streaming multipart upload: chunked writes off the event loop into a
temporary .part, atomic rename, progress on the button.
⚡ MP4/MOV/WEBM отдаются как есть с Range; MKV/AVI/TS — живой поток с точной
перемоткой без кэша и без рассинхрона.
MP4/MOV/WEBM served as-is with Range; MKV/AVI/TS use a live stream with
accurate seeking, no cache and no desync.
⚡ Кнопка Open открывает текущий источник в новой вкладке в играбельном виде.
The Open button opens the current source in a new tab in a playable form.
⚡ Превью показывает реальный источник по приоритету и переживает
перезагрузку страницы; нода держит пользовательский размер.
The preview shows the real priority source and survives page reload; the
node keeps the user's size.
⚡ Drag&drop видеофайла из проводника прямо на ноду.
Drag&drop a video file from the OS explorer onto the node.
Автор / Author: AGSoft
Дата / Date: 23.11.2026
==============================================================================
"""
import os
import re
import asyncio
import logging
import mimetypes

import numpy as np
import torch

import folder_paths
from aiohttp import web
from server import PromptServer

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"

try:
    from comfy_api.input_impl import VideoFromFile
    from comfy.comfy_types import IO
except Exception:
    VideoFromFile = None

    class IO:
        VIDEO = "VIDEO"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Load Video] v11.27 loaded (robust output-link scan, audio fallback)")

SEEK_PRESET = "ultrafast"
PROBE_TIMEOUT = 20
AUDIO_RATE = 44100
UPLOAD_CHUNK = 1024 * 1024
_VIDEO_INFO_CACHE = {}

OUT_VIDEO = 0
OUT_IMAGES = 1
OUT_AUDIO = 2
OUT_PATH = 3
OUT_WIDTH = 4
OUT_HEIGHT = 5
OUT_FRAMES = 6
OUT_DURATION = 7
OUT_FPS = 8
OUT_DEPTH = 9
OUT_SPACE = 10
OUTPUT_COUNT = 11


def _t(en, ru):
    return en + "\n---\n" + ru


def _single(value, default=None):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return value if value is not None else default


def _base_dirs():
    dirs = []
    for fn in (folder_paths.get_input_directory,
               folder_paths.get_output_directory,
               folder_paths.get_temp_directory):
        try:
            d = fn()
            if d:
                dirs.append(os.path.realpath(d))
        except Exception:
            continue
    return dirs


def _contained(path):
    p = os.path.normcase(os.path.realpath(path))
    for base in _base_dirs():
        b = os.path.normcase(base)
        if p == b or p.startswith(b + os.sep):
            return True
    return False


def _resolve_request_path(request):
    # Путь из запроса: либо абсолютный внутри папок ComfyUI, либо имя из input.
    # Request path: either absolute inside ComfyUI folders, or a name in input.
    raw = request.query.get("path", "")
    if raw:
        p = os.path.realpath(raw)
        return p if _contained(p) else None
    name = request.query.get("filename", "")
    if not name:
        return None
    p = os.path.realpath(os.path.join(
        folder_paths.get_input_directory(), os.path.basename(name)))
    return p if _contained(p) else None


async def _stop_ffmpeg(proc):
    # После kill обязателен wait, иначе транспорт закрывается сборщиком мусора.
    # wait is required after kill, otherwise the transport is closed by GC.
    try:
        if proc.returncode is None:
            proc.kill()
    except Exception:
        pass
    try:
        await proc.wait()
    except Exception:
        pass


def _empty_info():
    return {
        "width": 0, "height": 0, "duration": 0.0, "fps": 0.0, "codec": "",
        "has_audio": False, "audio_codec": "", "size_bytes": 0,
    }


def _parse_info(err, path):
    info = _empty_info()
    try:
        info["size_bytes"] = os.path.getsize(path)
    except Exception:
        pass
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        info["duration"] = round(
            int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)
    am = re.search(r"Stream\s*#\d+:\d+[^\n]*Audio:\s*([^,\n]+)", err)
    if am:
        info["has_audio"] = True
        info["audio_codec"] = am.group(1).strip()
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
        fps = 0.0
        fm = re.search(r"([\d.]+)\s*fps", video_line)
        if fm:
            fps = float(fm.group(1))
        if fps <= 0:
            tm = re.search(r"([\d.]+)\s*tbr", video_line)
            if tm:
                value = float(tm.group(1))
                if 0 < value < 1000:
                    fps = value
        info["fps"] = round(fps, 3)
    return info


async def _get_video_info(path):
    # Быстрый probe без декодирования; результат кэшируется в памяти.
    # Fast probe without decoding; the result is cached in memory.
    if not path or not os.path.isfile(path):
        return _empty_info()
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except Exception:
        key = (path, 0, 0)
    cached = _VIDEO_INFO_CACHE.get(key)
    if cached is not None:
        return cached
    info = _empty_info()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            FFMPEG_PATH, "-hide_banner", "-i", path,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=PROBE_TIMEOUT)
        except asyncio.TimeoutError:
            err = b""
            logger.warning(f"[AGSoft Load Video] probe timeout: {path}")
        info = _parse_info(err.decode("utf-8", "ignore"), path)
    except Exception as e:
        logger.warning(f"[AGSoft Load Video] metadata probe failed: {e}")
    finally:
        if proc is not None:
            await _stop_ffmpeg(proc)
    if len(_VIDEO_INFO_CACHE) > 128:
        _VIDEO_INFO_CACHE.clear()
    _VIDEO_INFO_CACHE[key] = info
    return info


def _extract_video_path(video_obj):
    # Путь из VIDEO-объекта: приватное поле, атрибуты, get_path, get_stream_source.
    # Path from a VIDEO object: private field, attributes, get_path, get_stream_source.
    if video_obj is None:
        return ""
    try:
        if VideoFromFile is not None and isinstance(video_obj, VideoFromFile):
            path = video_obj._VideoFromFile__file if hasattr(
                video_obj, "_VideoFromFile__file") else None
            if isinstance(path, str) and os.path.isfile(path):
                return os.path.realpath(path)
    except Exception:
        pass
    names = ["path", "_path", "filepath", "file_path", "source", "filename",
             "video_path", "_VideoFromFile__file"]
    for name in names:
        try:
            if hasattr(video_obj, name):
                value = getattr(video_obj, name)
                if isinstance(value, str) and os.path.isfile(value):
                    return os.path.realpath(value)
        except Exception:
            continue
    for name in ("get_path", "get_stream_source"):
        try:
            if hasattr(video_obj, name):
                fn = getattr(video_obj, name)
                if callable(fn):
                    value = fn()
                    if isinstance(value, str) and os.path.isfile(value):
                        return os.path.realpath(value)
        except Exception:
            continue
    return ""


def _used_output_indices(prompt, unique_id):
    # Какие выходы ноды реально потребляются дальше в графе выполнения.
    # Which node outputs are actually consumed downstream in the execution graph.
    if not isinstance(prompt, dict) or unique_id is None:
        return set(range(OUTPUT_COUNT))
    me = str(unique_id)
    used = set()
    for node_id, node in prompt.items():
        if str(node_id) == me or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                src, index = value[0], value[1]
                try:
                    index = int(index)
                except (TypeError, ValueError):
                    continue
                if str(src) == me:
                    used.add(index)
    return used


def _safe_call(fn, fallback, method):
    try:
        value = fn()
        return fallback if value is None else value
    except Exception as e:
        logger.warning(f"[AGSoft Load Video] {method} failed: {e}")
        return fallback


def _components(video_obj):
    # Один проход по контейнеру: кадры, звук и частота кадров сразу.
    # One pass over the container: frames, audio and frame rate at once.
    images, audio, fps = None, None, 0.0
    if hasattr(video_obj, "get_components") and callable(video_obj.get_components):
        try:
            comp = video_obj.get_components()
            if comp is not None:
                images = comp.images if hasattr(comp, "images") else None
                audio = comp.audio if hasattr(comp, "audio") else None
                rate = comp.frame_rate if hasattr(comp, "frame_rate") else None
                if rate is not None:
                    fps = float(rate)
        except Exception as e:
            logger.warning(f"[AGSoft Load Video] get_components failed: {e}")
    if images is None and hasattr(video_obj, "get_pixels"):
        try:
            images = video_obj.get_pixels()
        except Exception as e:
            logger.warning(f"[AGSoft Load Video] get_pixels failed: {e}")
    if images is not None and isinstance(images, torch.Tensor):
        if images.dim() == 5:
            images = images.reshape(images.shape[0] * images.shape[1], *images.shape[2:])
        images = images.detach()
        if images.dtype != torch.float32:
            images = images.float()
        if images.numel() > 0 and float(images.max()) > 1.5:
            images = images / 255.0
    return images, audio, fps


async def _decode_audio_only(path):
    # Звук без кадров: сырой PCM-поток, видеопоток не декодируется.
    # Audio without frames: raw PCM stream, the video stream is not decoded.
    if not path or not os.path.isfile(path):
        return None
    cmd = [
        FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-i", path,
        "-vn", "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(AUDIO_RATE), "-",
    ]
    proc = None
    data = b""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        data, _ = await proc.communicate()
    except Exception as e:
        logger.warning(f"[AGSoft Load Video] audio decode failed: {e}")
    finally:
        if proc is not None:
            await _stop_ffmpeg(proc)
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 2).copy()
    wave = torch.from_numpy(arr).transpose(0, 1).unsqueeze(0)
    return {"waveform": wave, "sample_rate": AUDIO_RATE}


def _list_video_files():
    try:
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for name in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, name)):
                    files.append(name)
        return folder_paths.filter_files_content_types(files, ["video"])
    except Exception:
        return []


def _preview_dict(path, info):
    mime, _ = mimetypes.guess_type(path)
    return {
        "kind": "path",
        "path": path,
        "ext": os.path.splitext(path)[1].lstrip(".").lower(),
        "filename": os.path.basename(path),
        "subfolder": "",
        "type": "path",
        "format": mime or "video/mp4",
        "fullpath": path,
        "width": info["width"],
        "height": info["height"],
        "duration": info["duration"],
        "frame_rate": info["fps"],
        "codec": info["codec"],
        "has_audio": info["has_audio"],
        "audio_codec": info["audio_codec"],
        "size_bytes": info["size_bytes"],
    }


@PromptServer.instance.routes.post("/agsoft/upload")
async def agsoft_upload(request):
    # Один потоковый multipart: чанки пишутся вне event loop во временный .part,
    # готовый файл появляется атомарно; при обрыве мусор не остаётся в input.
    # One streaming multipart: chunks are written off the event loop into a
    # temporary .part, the finished file appears atomically; nothing is left behind.
    remote = str(request.remote or "")
    if remote.startswith("::ffff:"):
        remote = remote[7:]
    if remote not in ("127.0.0.1", "::1", "localhost"):
        return web.json_response({"error": "loopback only"}, status=403)
    tmp_path = ""
    saved_name = ""
    try:
        reader = await request.multipart()
        if reader is None:
            return web.json_response({"error": "No multipart body"}, status=400)
        field = await reader.next()
        while field is not None:
            if field.name != "file":
                field = await reader.next()
                continue
            safe_name = os.path.basename(field.filename or "")
            if not safe_name:
                return web.json_response({"error": "No filename provided"}, status=400)
            target = os.path.join(folder_paths.get_input_directory(), safe_name)
            tmp_path = target + ".part"
            handle = await asyncio.to_thread(open, tmp_path, "wb")
            try:
                while True:
                    chunk = await field.read_chunk(UPLOAD_CHUNK)
                    if not chunk:
                        break
                    await asyncio.to_thread(handle.write, chunk)
                await asyncio.to_thread(handle.flush)
            finally:
                await asyncio.to_thread(handle.close)
            saved_name = safe_name
            break
        if not saved_name:
            return web.json_response({"error": "No file field in upload"}, status=400)
        await asyncio.to_thread(os.replace, tmp_path, target)
        tmp_path = ""
        logger.info(f"[AGSoft Load Video] uploaded: {saved_name}")
        return web.json_response({"status": "success", "name": saved_name})
    except Exception as e:
        logger.error(f"[AGSoft Load Video] upload error: {e}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                await asyncio.to_thread(os.remove, tmp_path)
            except Exception:
                pass


@PromptServer.instance.routes.get("/agsoft/stream")
async def agsoft_stream(request):
    # Отдача оригинала с поддержкой Range (нативная перемотка браузера).
    # Original file serving with Range support (native browser seeking).
    path = _resolve_request_path(request)
    if not path:
        return web.Response(status=403)
    if not os.path.isfile(path):
        return web.Response(status=404)
    file_size = os.path.getsize(path)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    if file_size == 0:
        return web.Response(status=200, content_type=mime)
    start, end, status = 0, file_size - 1, 200
    range_header = request.headers.get("Range")
    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not m:
            return web.Response(status=416,
                                headers={"Content-Range": f"bytes */{file_size}"})
        first, last = m.group(1), m.group(2)
        if first == "" and last == "":
            return web.Response(status=416,
                                headers={"Content-Range": f"bytes */{file_size}"})
        if first == "":
            start = max(0, file_size - int(last))
        else:
            start = int(first)
            end = int(last) if last != "" else file_size - 1
        if start >= file_size or start > end:
            return web.Response(status=416,
                                headers={"Content-Range": f"bytes */{file_size}"})
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
    try:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = count
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
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


@PromptServer.instance.routes.get("/agsoft/preview")
async def agsoft_preview(request):
    # Живой поток для MKV/AVI/TS: старт с 0 — copy, перемотка — точный транскод.
    # Live stream for MKV/AVI/TS: start 0 uses copy, seeking transcodes accurately.
    path = _resolve_request_path(request)
    if not path:
        return web.Response(status=403)
    if not os.path.isfile(path):
        return web.Response(status=404)
    try:
        start = max(0.0, float(request.query.get("start", "0") or 0))
    except ValueError:
        start = 0.0
    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path, "-map", "0:v:0", "-map", "0:a:0?"]
    if start > 0:
        cmd += ["-c:v", "libx264", "-preset", SEEK_PRESET, "-crf", "23",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-async", "1",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "-"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
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
    finally:
        await _stop_ffmpeg(proc)
    try:
        await resp.write_eof()
    except Exception:
        pass
    return resp


@PromptServer.instance.routes.get("/agsoft/video_info")
async def agsoft_video_info(request):
    # Абсолютный путь в ответе нужен кнопке Open: она открывает источник
    # одинаково для input и output.
    # The absolute path in the response is used by Open: it opens the source
    # the same way for input and output.
    path = _resolve_request_path(request)
    if not path:
        return web.json_response({"error": "path not allowed"}, status=403)
    if not os.path.isfile(path):
        return web.json_response({"error": "file not found"}, status=404)
    info = await _get_video_info(path)
    return web.json_response({**info, "path": path})


class AGSoftLoadVideo:
    OUTPUT_NODE = True
    CATEGORY = "AGSoft/Video"
    FUNCTION = "load_video"
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = (IO.VIDEO, "IMAGE", "AUDIO", "STRING", "INT", "INT", "INT",
                    "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("video", "images", "audio", "video_path", "width", "height",
                    "total_frames", "duration_seconds", "fps", "bit_depth",
                    "color_space")
    OUTPUT_TOOLTIPS = (
        _t("VIDEO object of the source. Feed it into any VIDEO input.",
           "VIDEO-объект источника. Подаётся в любой VIDEO-вход."),
        _t("All frames as an IMAGE batch (N, H, W, 3), float32 0..1. Decoded ONLY "
           "when this output is connected; otherwise None.",
           "Все кадры батчем IMAGE (N, H, W, 3), float32 0..1. Декодируется ТОЛЬКО "
           "при подключённом выходе; иначе None."),
        _t("Audio track as AUDIO (waveform + sample_rate). Extracted ONLY when this "
           "output is connected; otherwise None.",
           "Звуковая дорожка как AUDIO (waveform + sample_rate). Извлекается ТОЛЬКО "
           "при подключённом выходе; иначе None."),
        _t("Absolute path of the resolved source file (empty for tensor-only VIDEO).",
           "Абсолютный путь разрешённого источника (пусто для тензорного VIDEO)."),
        _t("Frame width in pixels.", "Ширина кадра в пикселях."),
        _t("Frame height in pixels.", "Высота кадра в пикселях."),
        _t("Total frames: exact when frames are decoded, otherwise duration x fps.",
           "Всего кадров: точно при декодированных кадрах, иначе duration x fps."),
        _t("Duration in seconds: total_frames / fps when decoded, otherwise probe.",
           "Длительность в секундах: total_frames / fps при декоде, иначе probe."),
        _t("Frame rate in frames per second.", "Частота кадров в секунду."),
        _t("Bit depth per color component, fallback 8. Read only when connected.",
           "Битность на компоненту цвета, фоллбэк 8. Читается только при подключении."),
        _t("Color space / transfer: sRGB, HDR or HDR PQ. Read only when connected.",
           "Цветовое пространство / transfer: sRGB, HDR или HDR PQ. Только при подключении."),
    )
    DESCRIPTION = _t(
        "🎬 AGSoft Load Video. Loads a video file and returns the VIDEO object, path, "
        "geometry, duration, fps, bit depth and color space. Frames (IMAGE) and audio "
        "(AUDIO) are decoded LAZILY — only when those outputs are connected downstream. "
        "Source priority: input_video > custom_path > file from the input directory. "
        "Video Save-style preview: Open/Upload buttons over the player, info footer under it.",
        "🎬 AGSoft Load Video. Загружает видеофайл и возвращает объект VIDEO, путь, "
        "геометрию, длительность, fps, битность и цветовое пространство. Кадры (IMAGE) и "
        "звук (AUDIO) декодируются ЛЕНИВО — только если эти выходы подключены дальше. "
        "Приоритет: input_video > custom_path > файл из папки input. "
        "Превью в стиле Video Save: кнопки Open/Upload над плеером, инфо-футер под ним.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_video_files()
        if not files:
            files = [" "]
        return {
            "required": {},
            "optional": {
                "input_video": (IO.VIDEO, {
                    "tooltip": _t(
                        "Video from another node (highest priority). The file path is "
                        "extracted from the VIDEO object automatically.",
                        "Видео от другой ноды (наивысший приоритет). Путь к файлу "
                        "извлекается из объекта VIDEO автоматически."),
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": _t(
                        "Path to a video file inside ComfyUI input/output/temp. "
                        "Overrides the dropdown when input_video is not connected.",
                        "Путь к видеофайлу внутри папок ComfyUI input/output/temp. "
                        "Переопределяет список, если input_video не подключён."),
                }),
                "video": (files, {
                    "video_upload": True,
                    "tooltip": _t(
                        "Select or upload a video from the input directory. Used when no "
                        "external video and no custom path are set. Drag&drop works too.",
                        "Выберите или загрузите видео из папки input. Используется, когда "
                        "нет внешнего видео и кастомного пути. Drag&drop тоже работает."),
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Список подключённых выходов входит в сигнатуру: иначе кэш отдал бы
        # старые None после нового подключения images/audio.
        # Connected outputs are part of the signature: otherwise the cache would
        # serve stale None after images/audio get connected.
        prompt = _single(kwargs.get("prompt"))
        unique_id = _single(kwargs.get("unique_id"))
        used = sorted(_used_output_indices(prompt, unique_id))
        signature = ",".join(str(i) for i in used)
        path = ""
        external = _single(kwargs.get("input_video"))
        if external is not None:
            path = _extract_video_path(external)
        if not path:
            custom = _single(kwargs.get("custom_path"), "")
            if custom and os.path.isfile(str(custom)):
                path = os.path.realpath(str(custom))
        if not path:
            name = _single(kwargs.get("video"), "")
            if name and str(name).strip():
                try:
                    path = folder_paths.get_annotated_filepath(str(name))
                except Exception:
                    path = ""
        try:
            if path and os.path.isfile(path):
                return f"{os.path.getmtime(path)}:{os.path.getsize(path)}:{signature}"
        except Exception:
            pass
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        custom = _single(kwargs.get("custom_path"), "")
        if custom and str(custom).strip():
            p = os.path.realpath(str(custom).strip())
            if not os.path.isfile(p):
                return f"File not found / Файл не найден: {custom}"
            if not _contained(p):
                return ("Path must be inside ComfyUI input/output/temp / "
                        f"Путь должен быть внутри input/output/temp: {custom}")
            return True
        name = _single(kwargs.get("video"), "")
        if name and str(name).strip():
            if not folder_paths.exists_annotated_filepath(str(name)):
                return f"Video file not found / Видеофайл не найден: {name}"
        return True

    async def load_video(self, video=None, input_video=None, custom_path="",
                         unique_id=None, prompt=None):
        video = _single(video, "")
        custom_path = _single(custom_path, "")
        video_obj = None
        path = ""
        if input_video is not None:
            video_obj = input_video
            path = _extract_video_path(input_video)
            if not path:
                logger.warning("[AGSoft Load Video] External VIDEO has no file path; "
                               "metadata comes from the object only.")
        elif custom_path and str(custom_path).strip():
            path = os.path.realpath(str(custom_path).strip())
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[AGSoft Load Video] File not found / Файл не найден: {path}")
            if not _contained(path):
                raise ValueError(
                    "[AGSoft Load Video] Path must be inside ComfyUI input/output/temp.\n"
                    "---\nПуть должен быть внутри папок ComfyUI input/output/temp.")
            video_obj = VideoFromFile(path) if VideoFromFile is not None else path
        elif video and str(video).strip():
            path = os.path.realpath(folder_paths.get_annotated_filepath(str(video)))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"[AGSoft Load Video] File not found / Файл не найден: {path}")
            mime, _ = mimetypes.guess_type(path)
            if mime and not mime.startswith("video"):
                logger.warning(f"[AGSoft Load Video] Selected file may not be a video: {path}")
            video_obj = VideoFromFile(path) if VideoFromFile is not None else path
        else:
            raise ValueError(
                "[AGSoft Load Video] No source: connect input_video, set custom_path "
                "or pick a file.\n---\nНет источника: подключите input_video, укажите "
                "custom_path или выберите файл.")

        info = await _get_video_info(path) if path else _empty_info()
        used = _used_output_indices(prompt, unique_id)
        need_images = OUT_IMAGES in used
        need_audio = OUT_AUDIO in used

        images, audio, comp_fps = None, None, 0.0
        if need_images:
            images, comp_audio, comp_fps = _components(video_obj)
            if need_audio:
                audio = comp_audio
            if images is None:
                images = torch.zeros((0, 0, 0, 3), dtype=torch.float32)
        elif need_audio:
            audio = await _decode_audio_only(path)
        if need_audio and audio is None and need_images and path:
            audio = await _decode_audio_only(path)
        if need_audio and audio is None:
            audio = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": AUDIO_RATE}

        decoded = (images is not None and isinstance(images, torch.Tensor)
                   and images.dim() == 4 and images.shape[0] > 0)
        fps_out = float(comp_fps) if comp_fps > 0 else float(info["fps"])
        if decoded:
            total_frames = int(images.shape[0])
            height = int(images.shape[1])
            width = int(images.shape[2])
            duration = float(total_frames) / fps_out if fps_out > 0 else float(info["duration"])
        else:
            width = int(info["width"])
            height = int(info["height"])
            duration = float(info["duration"])
            total_frames = int(round(duration * fps_out)) if fps_out > 0 else 0

        bit_depth = 8
        if OUT_DEPTH in used:
            bit_depth = int(_safe_call(lambda: video_obj.get_bit_depth(), 8, "get_bit_depth"))
        color_space = "sRGB"
        if OUT_SPACE in used:
            color_space = str(_safe_call(
                lambda: video_obj.get_color_space(), "sRGB", "get_color_space"))

        preview = _preview_dict(path, info) if path else None
        if preview is not None:
            preview["width"] = width or info["width"]
            preview["height"] = height or info["height"]
            preview["duration"] = round(duration, 3)
            preview["frame_rate"] = fps_out
        logger.info(f"[AGSoft Load Video] {path or 'tensor VIDEO'} | {width}x{height} | "
                    f"{round(duration, 3)}s | {fps_out} fps | "
                    f"decode: images={need_images}, audio={need_audio}")
        return {
            "ui": {"gifs": [preview] if preview else []},
            "result": (video_obj, images, audio, path, width, height, total_frames,
                       round(duration, 3), fps_out, bit_depth, color_space),
        }


NODE_CLASS_MAPPINGS = {
    "AGSoftLoadVideo": AGSoftLoadVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadVideo": "🎬AGSoft Load Video",
}