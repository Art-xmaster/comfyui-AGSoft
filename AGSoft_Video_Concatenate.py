"""
==============================================================================
AGSoft_Video_Concatenate.py
==============================================================================
Ноды / Nodes:
🎬AGSoft Video Concatenate       — мгновенная склейка (stream copy).
🎬AGSoft Video Concatenate Plus — склейка с переходами xfade и постобработкой.
Обе ноды живут в одном модуле и делят общий слой хелперов (probe, resolve
клипов, temp, асинхронный запуск FFmpeg, роут сохранения).
---
🎬AGSoft Video Concatenate       — instant concatenation (stream copy).
🎬🪡AGSoft Video Concatenate Plus — concatenation with xfade transitions.
Both nodes share one module and a common helper layer (probe, clip resolve,
temp, async FFmpeg runner, save route).

Описание / Description:
Склейка 2–10 видеоклипов в один ролик. Клипы подаются объектами VIDEO или
IMAGE-тензорами (например из 🎬AGSoft Load Video), строками путей или через
drag&drop. Тензорные клипы материализуются во временные файлы ComfyUI temp,
файловые источники используются напрямую без перекодирования. В режиме
preview результат показывается внутри ноды и сохраняется кнопкой Save в
output; режим direct пишет в output сразу.
---
Concatenates 2–10 video clips into one. Clips arrive as VIDEO objects or
IMAGE tensors (e.g. from 🎬AGSoft Load Video), path strings or drag&drop.
Tensor clips are materialized into ComfyUI temp files; file-backed sources
are used directly without re-encoding. In preview mode the result is shown
inside the node and saved by the Save button into output; direct mode
writes into output immediately.

Возможности / Features:
⚡ Фиксированные опциональные входы video_1..video_10 типа "*": VIDEO
(VideoFromFile / VideoFromComponents), IMAGE-тензор [N,H,W,C], STRING-путь.
Приоритет: сокет > скрытый clip_N_path (заполняется drag&drop).
Fixed optional inputs video_1..video_10 of type "*": VIDEO
(VideoFromFile / VideoFromComponents), IMAGE tensor [N,H,W,C], STRING path.
Priority: socket > hidden clip_N_path (filled via drag&drop).
⚡ inputs_count задаёт число видимых сокетов; неподключённые входы
пропускаются; требуется минимум 2 клипа и совпадение с inputs_count.
inputs_count sets the number of visible sockets; unconnected inputs are
skipped; at least 2 clips and a match with inputs_count are required.
⚡ Единый probe через ffmpeg -i (без ffprobe): длительность, разрешение,
fps, кодек, pix_fmt, звуковая дорожка, размер файла.
Single probe via ffmpeg -i (no ffprobe): duration, resolution, fps, codec,
pix_fmt, audio track, file size.
⚡ Temp-namespace ноды agsoft_concat_<UNIQUE_ID> очищается в начале каждого
прогона; результат живёт до следующего прогона ноды.
Per-node temp namespace agsoft_concat_<UNIQUE_ID> is cleared at every run
start; the result lives until the node's next run.
⚡ Тензоры кодируются в libx264 crf16 veryfast yuv420p (+AAC из встроенного
звука); BytesIO-источники дампятся как есть с сохранением кодека.
Tensors are encoded to libx264 crf16 veryfast yuv420p (+AAC from embedded
audio); BytesIO sources are dumped as-is preserving the codec.
⚡ Fast: проверка совместимости (кодек, pix_fmt, размер, fps, наличие звука)
и склейка concat demuxer + -c copy; +faststart для mp4/mov.
Fast: compatibility check (codec, pix_fmt, size, fps, audio presence) and
concat demuxer + -c copy; +faststart for mp4/mov.
⚡ Fast auto_normalize=on: несовместимые клипы перекодируются во temp под
параметры первого (letterbox); если первый не h264/yuv420p — приводятся все.
off = двуязычная ошибка со списком несовпадений по клипам.
Fast auto_normalize=on: incompatible clips are re-encoded into temp to the
first clip's params (letterbox); if the first is not h264/yuv420p all are
normalized. off = bilingual error listing mismatches per clip.
⚡ Plus: 44 нативных перехода xfade + cut + random(seed) (воспроизводимо);
длительность перехода ограничивается длиной клипов на каждом стыке.
Plus: 44 native xfade transitions + cut + random(seed) (reproducible);
transition duration is clamped to clip lengths at every junction.
⚡ Plus: защита от рассинхрона звука (aresample + apad + atrim),
отсутствующий звук заменяется тишиной; audio_mode auto/mute.
Plus: audio desync protection (aresample + apad + atrim), missing audio is
replaced with silence; audio_mode auto/mute.
⚡ Plus: нормализация разрешения (first_clip/max_size/min_size/custom),
fps (auto/список), SAR и yuv420p; вписывание letterbox/crop/stretch.
Plus: resolution normalization (first_clip/max_size/min_size/custom),
fps (auto/list), SAR and yuv420p; fitting letterbox/crop/stretch.
⚡ Plus: глобальные fade in/out видео и звука, loudnorm EBU R128,
экспериментальная авто-нормализация цвета.
Plus: global video/audio fade in/out, EBU R128 loudnorm, experimental
auto color normalization.
⚡ Plus: энкодер auto/nvenc/cpu с fallback NVENC→libx264; пресеты качества
fast/balanced/quality.
Plus: encoder auto/nvenc/cpu with NVENC→libx264 fallback; quality presets
fast/balanced/quality.
⚡ save_mode preview (по умолчанию): результат во temp + превью в ноде +
кнопка Save копирует в output без перекодирования; direct: сразу в output.
save_mode preview (default): result in temp + in-node preview + Save button
copies into output without re-encoding; direct: straight into output.
⚡ Только папки ComfyUI: output_path = подпапка внутри output (абсолютные
пути отклоняются); чтение источников ограничено input/output/temp.
ComfyUI folders only: output_path = subfolder inside output (absolute paths
rejected); source reading is confined to input/output/temp.
⚡ Роут /agsoft/concat_save: loopback-only, containment temp→output, имя
через basename, санация подпапки, авто-суффикс при коллизии имён.
Route /agsoft/concat_save: loopback-only, temp→output containment, basename
for names, subfolder sanitization, auto-suffix on name collisions.
⚡ Выходы: VIDEO (VideoFromFile) первым, затем video_path, duration_seconds,
duration_timecode, file_size_mb, width, height, fps, frames_est.
Outputs: VIDEO (VideoFromFile) first, then video_path, duration_seconds,
duration_timecode, file_size_mb, width, height, fps, frames_est.
⚡ Превью для UI отдаётся через ui.gifs с реальным probe (кодек, звук,
байты, fps) для инфо-футера в стиле Video Save.
UI preview is delivered via ui.gifs with a real probe (codec, audio, bytes,
fps) for the Video Save-style info footer.
⚡ Асинхронный пайплайн: только asyncio.create_subprocess_exec; прогресс из
stderr (time=) бакетами по 5% → WS agsoft_concat_progress; проверка отмены;
kill + wait в finally.
Async pipeline: asyncio.create_subprocess_exec only; progress from stderr
(time=) in 5% buckets → WS agsoft_concat_progress; interrupt checks;
kill + wait in finally.
⚡ IS_CHANGED: sha256 виджетов и сигнатур клипов (mtime путей, форма+сумма
тензоров); VALIDATE_INPUTS проверяет существование файлов до выполнения.
IS_CHANGED: sha256 of widgets and clip signatures (path mtime, tensor
shape+sum); VALIDATE_INPUTS checks file existence before execution.

Автор / Author: AGSoft
Дата / Date: 23.11.2026
==============================================================================
"""

import os
import re
import random
import asyncio
import hashlib
import logging
import shutil

import numpy as np

import folder_paths
from server import PromptServer
from aiohttp import web

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"

try:
    from comfy_api.input_impl import VideoFromFile
except Exception:
    VideoFromFile = None

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None

try:
    import soundfile
except Exception:
    soundfile = None

# print("[AGSoft Video Concatenate] v11.23 loaded ")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsyncioConnectionLostFilter(logging.Filter):
    def filter(self, record):
        return "_call_connection_lost" not in record.getMessage()


logging.getLogger("asyncio").addFilter(AsyncioConnectionLostFilter())

MAX_CLIPS = 10
TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
NATIVE_XFADE_TRANSITIONS = [
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite",
    "radial", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose",
    "horzopen", "horzclose", "dissolve", "pixelize",
    "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice",
    "hblur", "fadegrays", "wipetl", "wipetr", "wipebl", "wipebr",
    "squeezeh", "squeezev", "zoomin",
]
_ENCODER_CACHE = {}


def _single(value, default=""):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return value if value is not None else default


def safe_float(value, default=0.0):
    try:
        return float(default) if value is None else float(value)
    except Exception:
        return float(default)


def safe_int(value, default=0):
    try:
        return int(default) if value is None else int(float(value))
    except Exception:
        return int(default)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _even(value):
    return max(2, int(round(float(value) / 2.0) * 2))


def format_timecode(seconds):
    try:
        seconds = max(0.0, float(seconds))
    except Exception:
        return "00:00:00.000"
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    if ms >= 1000:
        total += 1
        ms = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _roots():
    return [
        folder_paths.get_input_directory(),
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
    ]


def _contained(path, roots):
    rp = os.path.normcase(os.path.realpath(path))
    for root in roots:
        rr = os.path.normcase(os.path.realpath(root))
        if rp == rr or rp.startswith(rr + os.sep):
            return True
    return False


def safe_subfolder(value):
    s = str(value or "").strip().replace("\\", "/")
    if ":" in s or s.startswith("/"):
        logger.warning("[AGSoft Concat] Absolute paths are not allowed; using output root.")
        return ""
    parts = [p for p in s.split("/") if p and p not in (".", "..")]
    return "/".join(parts)


def unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(1, 1000):
        cand = f"{base}_{i:03d}{ext}"
        if not os.path.exists(cand):
            return cand
    return path


def _interrupt():
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


async def ffmpeg_run(cmd, total=0.0, on_progress=None, stdin_data=None):
    # Единый асинхронный запуск FFmpeg с прогрессом и гарантированной остановкой.
    # Single async FFmpeg runner with progress and guaranteed shutdown.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    chunks = []
    feeder = None

    async def feed():
        try:
            view = memoryview(stdin_data)
            step = 4 * 1024 * 1024
            for i in range(0, len(view), step):
                proc.stdin.write(view[i:i + step])
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    async def read_err():
        last = -1
        buf = b""
        while True:
            data = await proc.stderr.read(512)
            if not data:
                break
            buf += data
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                text = raw.decode("utf-8", "ignore")
                chunks.append(text + "\n")
                if total > 0 and on_progress:
                    m = TIME_RE.search(text)
                    if m:
                        cur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                               + float(m.group(3)))
                        pct = clamp(int(cur / total * 100), 0, 100)
                        bucket = pct // 5
                        if bucket > last:
                            last = bucket
                            _interrupt()
                            on_progress(pct)

    try:
        if stdin_data is not None:
            feeder = asyncio.create_task(feed())
        reader = asyncio.create_task(read_err())
        if feeder is not None:
            await feeder
        await reader
        await proc.wait()
        return proc.returncode, "".join(chunks)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def parse_media_info(path):
    info = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "has_audio": False, "size_mb": 0.0, "frames": 0,
        "codec": "", "codec_long": "", "pix_fmt": "", "audio_codec": "",
    }
    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
    except Exception:
        pass
    try:
        _, data = await ffmpeg_run([FFMPEG_PATH, "-hide_banner", "-i", path])
    except Exception as e:
        logger.warning(f"[AGSoft Concat] Cannot read media: {path} ({e})")
        return info
    m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", data)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        info["duration"] = round(dur, 3)
    am = re.search(r"Audio:\s*([^,\n]+)", data)
    if am:
        info["has_audio"] = True
        info["audio_codec"] = am.group(1).strip()
    for line in data.splitlines():
        if "Video:" not in line:
            continue
        lm = re.search(r"Video:\s*([^,\n]+)", line)
        if lm:
            info["codec_long"] = lm.group(1).strip()
        cm = re.search(r"Video:\s*([a-z0-9_]+)", line)
        if cm:
            info["codec"] = cm.group(1)
        pm = re.search(
            r",\s*(yuv[a-z0-9]+|rgb[a-z0-9]+|bgr[a-z0-9]+|nv12|p010[a-z0-9]*)\b", line)
        if pm:
            info["pix_fmt"] = pm.group(1)
        rm = re.search(r",\s*(\d{2,5})x(\d{2,5})\b", line)
        if rm:
            info["width"] = int(rm.group(1))
            info["height"] = int(rm.group(2))
        fps_value = None
        fm = re.search(r"([\d.]+)\s*fps", line)
        if fm:
            fps_value = safe_float(fm.group(1), 0.0)
        if not fps_value:
            tm = re.search(r"([\d.]+)\s*tbr", line)
            if tm:
                tbr = safe_float(tm.group(1), 0.0)
                if 0 < tbr < 1000:
                    fps_value = tbr
        if fps_value:
            info["fps"] = round(fps_value, 3)
        break
    if info["duration"] > 0 and info["fps"] > 0:
        info["frames"] = int(round(info["duration"] * info["fps"]))
    return info


def _temp_dir(node_id):
    root = os.path.join(folder_paths.get_temp_directory(), f"agsoft_concat_{node_id}")
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return root


def _write_audio_wav(audio, path):
    if soundfile is None or audio is None:
        return False
    try:
        wave = audio["waveform"] if isinstance(audio, dict) else audio
        rate = safe_int(audio.get("sample_rate", 44100), 44100) if isinstance(audio, dict) else 44100
        arr = wave[0].detach().cpu().numpy().transpose(1, 0)
        soundfile.write(path, arr, rate)
        return True
    except Exception as e:
        logger.warning(f"[AGSoft Concat] Audio export failed: {e}")
        return False


async def _encode_frames(images, fps, audio, out_path, target=None, total_cb=None):
    arr = images.detach().cpu().numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    n, h, w, _ = arr.shape
    cmd = [
        FFMPEG_PATH, "-y", "-hide_banner",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0",
    ]
    wav = out_path + ".wav"
    has_wav = _write_audio_wav(audio, wav)
    if has_wav:
        cmd += ["-i", wav]
    vf = []
    if target:
        tw, th = target
        vf.append(
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    vf.append("format=yuv420p")
    cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "16", "-pix_fmt", "yuv420p"]
    if has_wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd.append(out_path)
    total = n / max(fps, 0.01)
    rc, err = await ffmpeg_run(cmd, total=total, on_progress=total_cb,
                               stdin_data=arr.tobytes())
    if os.path.isfile(wav):
        os.remove(wav)
    if rc != 0 or not os.path.isfile(out_path):
        raise RuntimeError(f"[AGSoft Concat] Tensor encode failed: {err[-800:]}")
    return out_path


async def resolve_clip(value, index, temp_dir, tensor_fps, on_progress=None, target=None):
    # Приоритет: файловый источник без перекодирования, затем тензор -> temp.
    # Priority: file-backed source without re-encode, then tensor -> temp.
    value = _single(value, None)
    if value is None:
        return None
    if isinstance(value, str):
        p = value.strip()
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(folder_paths.get_input_directory(), p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[AGSoft Concat] Clip {index}: file not found: {p}")
        if not _contained(p, _roots()):
            raise ValueError(f"[AGSoft Concat] Clip {index}: path outside ComfyUI folders.")
        return p
    if hasattr(value, "get_stream_source"):
        src = value.get_stream_source()
        if isinstance(src, str) and os.path.isfile(src):
            if not _contained(src, _roots()):
                raise ValueError(f"[AGSoft Concat] Clip {index}: path outside ComfyUI folders.")
            return src
        if hasattr(src, "read"):
            out = os.path.join(temp_dir, f"clip_{index:02d}_src.mp4")
            with open(out, "wb") as f:
                if hasattr(src, "seek"):
                    src.seek(0)
                shutil.copyfileobj(src, f)
            return out
    images, fps, audio = None, tensor_fps, None
    if hasattr(value, "get_components"):
        comp = value.get_components()
        images = comp.images if hasattr(comp, "images") else None
        audio = comp.audio if hasattr(comp, "audio") else None
        if hasattr(comp, "frame_rate") and comp.frame_rate:
            fps = safe_float(comp.frame_rate, tensor_fps)
    elif hasattr(value, "shape") and getattr(value, "ndim", 0) == 4:
        images = value
    if images is None:
        raise ValueError(f"[AGSoft Concat] Clip {index}: unsupported input type.")
    out = os.path.join(temp_dir, f"clip_{index:02d}.mp4")
    return await _encode_frames(images, fps, audio, out, target=target, total_cb=on_progress)


async def collect_clips(kwargs, tensor_fps, temp_dir, on_progress=None):
    video_map = {}
    for key, val in kwargs.items():
        m = re.fullmatch(r"video_(\d+)", key)
        if m:
            video_map[int(m.group(1))] = val
    paths = []
    for idx in sorted(video_map.keys()):
        val = video_map[idx]
        if val is None or (isinstance(val, str) and not val.strip()):
            fallback = kwargs.get(f"clip_{idx}_path", "")
            val = fallback if str(fallback).strip() else None
        if val is None:
            continue
        p = await resolve_clip(val, idx, temp_dir, tensor_fps, on_progress)
        if p:
            paths.append(p)
    return paths


def _mismatch(info, base):
    diffs = []
    if info["codec"] != base["codec"]:
        diffs.append(f"codec {info['codec']} != {base['codec']}")
    if info["pix_fmt"] != base["pix_fmt"]:
        diffs.append(f"pix_fmt {info['pix_fmt']} != {base['pix_fmt']}")
    if (info["width"], info["height"]) != (base["width"], base["height"]):
        diffs.append(f"size {info['width']}x{info['height']} != "
                     f"{base['width']}x{base['height']}")
    if round(info["fps"], 2) != round(base["fps"], 2):
        diffs.append(f"fps {info['fps']} != {base['fps']}")
    if info["has_audio"] != base["has_audio"]:
        diffs.append("audio presence differs")
    return diffs


async def normalize_clip(path, info, base, temp_dir, index, on_progress=None):
    tw, th = _even(base["width"]), _even(base["height"])
    fps = base["fps"] or 24.0
    cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-i", path]
    if not info["has_audio"]:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    cmd += [
        "-vf",
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
        f"format=yuv420p,fps={fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "44100", "-ac", "2",
    ]
    if not info["has_audio"]:
        cmd += ["-shortest"]
    out = os.path.join(temp_dir, f"norm_{index:02d}.mp4")
    cmd.append(out)
    rc, err = await ffmpeg_run(cmd, total=info["duration"], on_progress=on_progress)
    if rc != 0 or not os.path.isfile(out):
        raise RuntimeError(f"[AGSoft Concat] Normalize failed: {err[-800:]}")
    return out


def _progress_sender(node_id):
    def send(pct, stage="render"):
        try:
            PromptServer.instance.send_sync(
                "agsoft_concat_progress", {"node": node_id, "percent": pct, "stage": stage})
        except Exception:
            pass
    return send


def _final_destination(output_name, subfolder, save_mode, temp_dir, default_name):
    safe_name = os.path.basename(str(output_name or default_name))
    stem, ext = os.path.splitext(safe_name)
    safe_name = f"{stem or 'joined_video'}{ext or '.mp4'}"
    sub = safe_subfolder(subfolder)
    if save_mode == "preview":
        return os.path.join(temp_dir, safe_name), sub, "temp", safe_name
    target = os.path.join(folder_paths.get_output_directory(), sub)
    os.makedirs(target, exist_ok=True)
    return unique_path(os.path.join(target, safe_name)), sub, "output", safe_name


def _preview_dict(path, kind, subfolder, info):
    # Превью для ui.gifs в формате Video Save.
    # Preview for ui.gifs in the Video Save format.
    size_bytes = 0
    try:
        size_bytes = os.path.getsize(path)
    except Exception:
        pass
    return {
        "kind": "file",
        "filename": os.path.basename(path),
        "subfolder": subfolder,
        "type": kind,
        "format": "video/mp4",
        "fullpath": path,
        "width": info["width"],
        "height": info["height"],
        "duration": info["duration"],
        "frame_rate": info["fps"],
        "codec": info.get("codec_long") or info["codec"],
        "has_audio": info["has_audio"],
        "audio_codec": info.get("audio_codec", ""),
        "size_bytes": size_bytes,
    }


def _clip_signature(val):
    val = _single(val, None)
    if val is None:
        return "-"
    if isinstance(val, str):
        p = val if os.path.isabs(val) else os.path.join(
            folder_paths.get_input_directory(), val)
        try:
            return f"{p}:{os.path.getmtime(p)}"
        except Exception:
            return p
    if hasattr(val, "shape"):
        try:
            return f"{tuple(val.shape)}:{float(val[0].sum()):.3f}"
        except Exception:
            return str(tuple(val.shape))
    return type(val).__name__


def _widgets_hash(kwargs, keys):
    h = hashlib.sha256()
    for k in keys:
        h.update(repr(kwargs.get(k)).encode("utf-8"))
    for i in range(1, MAX_CLIPS + 1):
        h.update(_clip_signature(kwargs.get(f"video_{i}")).encode("utf-8"))
        h.update(str(kwargs.get(f"clip_{i}_path", "")).encode("utf-8"))
    return h.hexdigest()


def _validate_paths(kwargs):
    for i in range(1, MAX_CLIPS + 1):
        for key in (f"video_{i}", f"clip_{i}_path"):
            val = _single(kwargs.get(key), "")
            if isinstance(val, str) and val.strip():
                p = val.strip()
                if not os.path.isabs(p):
                    p = os.path.join(folder_paths.get_input_directory(), p)
                if not os.path.isfile(p):
                    return f"File not found / Файл не найден: {val}"
    return True


def _video_output(path):
    if VideoFromFile is not None:
        try:
            return VideoFromFile(path)
        except Exception:
            return None
    return None


def _clip_inputs():
    inputs = {}
    for i in range(1, MAX_CLIPS + 1):
        inputs[f"video_{i}"] = ("*", {
            "tooltip": (
                "Clip input: VIDEO, IMAGE tensor or path string.\n"
                "---\n"
                "Вход клипа: VIDEO, IMAGE-тензор или строка пути.")})
    return inputs


def _clip_path_widgets():
    widgets = {}
    for i in range(1, MAX_CLIPS + 1):
        widgets[f"clip_{i}_path"] = ("STRING", {
            "default": "",
            "tooltip": (
                "Hidden fallback path inside ComfyUI input (set via drag&drop).\n"
                "---\n"
                "Скрытый резервный путь внутри input ComfyUI (ставится drag&drop).")})
    return widgets


OUTPUT_TOOLTIPS_COMMON = (
    "VIDEO object for chaining.\n---\nОбъект VIDEO для цепочек.",
    "Path of the resulting file.\n---\nПуть итогового файла.",
    "Duration in seconds.\n---\nДлительность в секундах.",
    "Duration timecode.\n---\nТаймкод длительности.",
    "File size in MB.\n---\nРазмер файла в МБ.",
    "Width.\n---\nШирина.",
    "Height.\n---\nВысота.",
    "Frames per second.\n---\nКадров в секунду.",
    "Estimated frame count.\n---\nОценка числа кадров.",
)


class AGSoftVideoConcatenate:
    OUTPUT_NODE = True
    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT",
                    "FLOAT", "INT")
    RETURN_NAMES = ("video", "video_path", "duration_seconds", "duration_timecode",
                    "file_size_mb", "width", "height", "fps", "frames_est")
    OUTPUT_TOOLTIPS = OUTPUT_TOOLTIPS_COMMON
    FUNCTION = "concat_videos"
    CATEGORY = "AGSoft/Video"
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "⚡ Fast concatenation via FFmpeg stream copy; incompatible clips can be "
        "auto-normalized into temp. Accepts VIDEO, IMAGE tensors and paths.\n"
        "---\n"
        "⚡ Мгновенная склейка через stream copy FFmpeg; несовместимые клипы "
        "авто-нормализуются во temp. Принимает VIDEO, IMAGE-тензоры и пути."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "output_name": ("STRING", {
                "default": "output_video.mp4",
                "tooltip": (
                    "Output filename (.mp4 added if missing).\n"
                    "---\n"
                    "Имя файла (.mp4 добавится автоматически).")}),
            "inputs_count": ([str(i) for i in range(2, MAX_CLIPS + 1)], {
                "default": "2",
                "tooltip": (
                    "Number of visible clip sockets.\n---\nЧисло видимых сокетов клипов.")}),
        }
        optional = {
            "output_path": ("STRING", {
                "default": "",
                "tooltip": (
                    "Subfolder inside ComfyUI output (absolute paths are rejected).\n"
                    "---\n"
                    "Подпапка внутри output ComfyUI (абсолютные пути отклоняются).")}),
            "save_mode": (["preview", "direct"], {
                "default": "preview",
                "tooltip": (
                    "preview = render into temp, then Save button copies to output. "
                    "direct = write to output immediately.\n"
                    "---\n"
                    "preview = рендер во temp, кнопка Save копирует в output. "
                    "direct = сразу в output.")}),
            "auto_normalize": (["on", "off"], {
                "default": "on",
                "tooltip": (
                    "Re-encode incompatible clips into temp instead of raising an error.\n"
                    "---\n"
                    "Перекодировать несовместимые клипы во temp вместо ошибки.")}),
            "tensor_fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                "tooltip": (
                    "FPS for IMAGE tensor clips without embedded rate.\n"
                    "---\n"
                    "FPS для IMAGE-тензоров без встроенной частоты.")}),
        }
        optional.update(_clip_path_widgets())
        optional.update(_clip_inputs())
        return {"required": required, "optional": optional,
                "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        keys = ["output_name", "inputs_count", "output_path", "save_mode",
                "auto_normalize", "tensor_fps"]
        return _widgets_hash(kwargs, keys)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return _validate_paths(kwargs)

    async def concat_videos(self, output_name, inputs_count, output_path="",
                            save_mode="preview", auto_normalize="on", tensor_fps=24.0,
                            unique_id=None, **kwargs):
        node_id = str(unique_id or "0")
        temp_dir = _temp_dir(node_id)
        send_p = _progress_sender(node_id)
        tensor_fps = safe_float(tensor_fps, 24.0)
        send_p(2, "resolve")
        video_list = await collect_clips(kwargs, tensor_fps, temp_dir,
                                         lambda p: send_p(2 + p // 20, "resolve"))
        expected = safe_int(inputs_count, 2)
        if len(video_list) < 2:
            raise ValueError("[AGSoft Video Concat] Need at least 2 clips! / Нужно минимум 2 клипа!")
        if len(video_list) < expected:
            raise ValueError(
                f"[AGSoft Video Concat] Connected {len(video_list)} of {expected}. / "
                f"Подключено {len(video_list)} из {expected}.")
        infos = [await parse_media_info(p) for p in video_list]
        base = infos[0]
        problems = []
        for i, info in enumerate(infos):
            diffs = _mismatch(info, base)
            if diffs:
                problems.append((i, diffs))
        if problems:
            if str(auto_normalize) != "on":
                detail = "; ".join(
                    f"clip {i + 1}: {', '.join(d)}" for i, d in problems)
                raise ValueError(
                    f"[AGSoft Video Concat] Incompatible clips / Несовместимые клипы: {detail}")
            # Если первый клип не h264/yuv420p, приводим все к общему h264.
            # If the first clip is not h264/yuv420p, normalize everything to h264.
            if base["codec"] != "h264" or not base["pix_fmt"].startswith("yuv420"):
                base = dict(base, codec="h264", pix_fmt="yuv420p")
                targets = list(range(len(video_list)))
            else:
                targets = [i for i, _ in problems]
            for n, i in enumerate(targets):
                send_p(10 + n * 40 // max(len(targets), 1), "normalize")
                video_list[i] = await normalize_clip(
                    video_list[i], infos[i], base, temp_dir, i,
                    lambda p: send_p(10 + p // 3, "normalize"))
            infos = [await parse_media_info(p) for p in video_list]
        final_path, sub, kind, safe_name = _final_destination(
            output_name, output_path, save_mode, temp_dir, "joined_video.mp4")
        if kind == "output":
            final_abs = os.path.normcase(os.path.realpath(final_path))
            for p in video_list:
                if os.path.normcase(os.path.realpath(p)) == final_abs:
                    raise ValueError("[AGSoft Video Concat] Output equals a source clip.")
        list_file = os.path.join(temp_dir, "concat_list.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in video_list:
                abs_p = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{abs_p}'\n")
        cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-f", "concat", "-safe", "0",
               "-i", list_file, "-c", "copy"]
        if safe_name.lower().endswith((".mp4", ".mov")):
            cmd += ["-movflags", "+faststart"]
        cmd.append(final_path)
        send_p(60, "concat")
        rc, err = await ffmpeg_run(cmd)
        if rc != 0 or not os.path.isfile(final_path):
            logger.error(f"[AGSoft Video Concat] FFmpeg log:\n{err[-4000:]}")
            last_line = next((l.strip() for l in reversed(err.splitlines())
                              if l.strip()), "FFmpeg error")
            raise RuntimeError(f"[AGSoft Video Concat] FFmpeg failed: {last_line}")
        info = await parse_media_info(final_path)
        timecode = format_timecode(info["duration"])
        send_p(100, "done")
        preview_sub = sub if kind == "output" else f"agsoft_concat_{node_id}"
        preview = _preview_dict(final_path, kind, preview_sub, info)
        logger.info(f"[AGSoft Video Concat] Saved: {final_path} | "
                    f"{timecode} | {info['size_mb']} MB")
        return {
            "ui": {"gifs": [preview]},
            "result": (_video_output(final_path), final_path, info["duration"], timecode,
                       info["size_mb"], info["width"], info["height"], info["fps"],
                       info["frames"]),
        }


def choose_output_size(infos, resize_mode, custom_width, custom_height):
    resize_mode = str(resize_mode or "first_clip").strip().lower()
    if resize_mode == "custom" and custom_width > 0 and custom_height > 0:
        return _even(custom_width), _even(custom_height)
    sizes = [(i["width"], i["height"]) for i in infos
             if i["width"] > 0 and i["height"] > 0]
    if not sizes:
        return 1280, 720
    if resize_mode == "max_size":
        return _even(max(w for w, h in sizes)), _even(max(h for w, h in sizes))
    if resize_mode == "min_size":
        return _even(min(w for w, h in sizes)), _even(min(h for w, h in sizes))
    return _even(sizes[0][0]), _even(sizes[0][1])


def choose_output_fps(infos, output_fps):
    output_fps = str(output_fps or "auto").strip().lower()
    if output_fps not in ("auto", "", "none"):
        v = safe_float(output_fps, 0.0)
        if v > 0:
            return v
    for info in infos:
        if info["fps"] > 0:
            return info["fps"]
    return 30.0


async def supports_encoder(encoder):
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    ok = False
    try:
        _, stdout = await ffmpeg_run([FFMPEG_PATH, "-hide_banner", "-encoders"])
        ok = bool(re.search(rf"\b{re.escape(encoder)}\b", stdout))
    except Exception:
        ok = False
    _ENCODER_CACHE[encoder] = ok
    return ok


async def choose_encoder(encoder_mode):
    mode = str(encoder_mode or "auto").strip().lower()
    if mode == "cpu":
        return "libx264"
    if mode == "nvenc":
        if await supports_encoder("h264_nvenc"):
            return "h264_nvenc"
        raise RuntimeError("[AGSoft Concat Plus] h264_nvenc unavailable / недоступен.")
    return "h264_nvenc" if await supports_encoder("h264_nvenc") else "libx264"


def build_encoder_args(encoder, quality_preset):
    quality = str(quality_preset or "balanced").strip().lower()
    if encoder == "h264_nvenc":
        if quality == "fast":
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "26",
                    "-pix_fmt", "yuv420p"]
        if quality == "quality":
            return ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "20",
                    "-pix_fmt", "yuv420p"]
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
                "-pix_fmt", "yuv420p"]
    if quality == "fast":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-pix_fmt", "yuv420p"]
    if quality == "quality":
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "20",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p"]


def build_filter_complex(infos, transition_type, transition_duration,
                         output_width, output_height, output_fps,
                         audio_mode, fit_mode, auto_color,
                         video_fade_in, video_fade_out,
                         audio_fade_in, audio_fade_out, audio_loudnorm, rng):
    parts, logs, durations = [], [], []
    n = len(infos)
    fit_mode = str(fit_mode or "letterbox").strip().lower()
    audio_mode = str(audio_mode or "auto").strip().lower()
    global_td = max(0.0, safe_float(transition_duration, 0.0))
    use_concat = str(transition_type or "fade").strip().lower() == "cut" or global_td <= 0.0
    for i, info in enumerate(infos):
        duration = float(info.get("duration") or 0.0)
        if duration <= 0.05:
            duration = 5.0
            logs.append(f"Clip {i + 1}: duration unknown, fallback 5.0s / "
                        f"длительность не определена, fallback 5.0 сек.")
        durations.append(duration)
        v = f"[{i}:v]"
        if str(auto_color) == "on":
            v += "normalize=blackpt=black:whitept=white:smoothing=0,"
        if fit_mode == "stretch":
            scale_part = f"scale={output_width}:{output_height}"
        elif fit_mode == "crop":
            scale_part = (
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
                f"crop={output_width}:{output_height}")
        else:
            scale_part = (
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
                f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black")
        # fps после setpts гарантирует CFR, иначе xfade падает.
        # fps after setpts guarantees CFR, otherwise xfade fails.
        v += (f"{scale_part},setsar=1,format=yuv420p,setpts=PTS-STARTPTS,"
              f"fps={output_fps:.6f}[v{i}]")
        parts.append(v)
        if audio_mode == "mute" or not info.get("has_audio", False):
            parts.append(f"anullsrc=r=44100:cl=stereo[sil{i}]")
            parts.append(f"[sil{i}]atrim=0:{duration:.6f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            parts.append(
                f"[{i}:a]aresample=44100:async=1,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad,atrim=0:{duration:.6f},asetpts=PTS-STARTPTS[a{i}]")
    if use_concat:
        streams = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{streams}concat=n={n}:v=1:a=1[vraw][araw]")
        total_duration = sum(durations)
        logs.append("Cut mode / режим cut.")
    else:
        last_v, last_a = "[v0]", "[a0]"
        prev_mix = durations[0]
        for i in range(n - 1):
            tt = str(transition_type).strip().lower()
            current = rng.choice(NATIVE_XFADE_TRANSITIONS) if tt == "random" else tt
            if current not in NATIVE_XFADE_TRANSITIONS:
                current = "fade"
            nxt = durations[i + 1]
            max_possible = min(prev_mix, nxt)
            cur_d = min(global_td, max_possible)
            if cur_d < 0.01:
                cur_d = max(0.01, max_possible)
            offset = max(0.0, prev_mix - cur_d)
            next_v = f"[vmix{i}]" if i < n - 2 else "[vraw]"
            next_a = f"[amix{i}]" if i < n - 2 else "[araw]"
            parts.append(
                f"{last_v}[v{i + 1}]xfade=transition={current}:"
                f"duration={cur_d:.6f}:offset={offset:.6f}{next_v}")
            parts.append(
                f"{last_a}[a{i + 1}]acrossfade=d={cur_d:.6f}:curve1=tri:curve2=tri{next_a}")
            logs.append(f"Junction {i + 1} / стык: '{current}', d={cur_d:.3f}, off={offset:.3f}")
            last_v, last_a = next_v, next_a
            prev_mix = prev_mix + nxt - cur_d
        total_duration = prev_mix
    if total_duration <= 0:
        total_duration = sum(durations)
    max_fade = total_duration / 2.0 if total_duration > 0 else 0.0
    vfi = clamp(safe_float(video_fade_in, 0.0), 0.0, max_fade)
    vfo = clamp(safe_float(video_fade_out, 0.0), 0.0, max_fade)
    afi = clamp(safe_float(audio_fade_in, 0.0), 0.0, max_fade)
    afo = clamp(safe_float(audio_fade_out, 0.0), 0.0, max_fade)
    v_label = "[vraw]"
    if vfi > 0:
        nxt = "[vfadein]" if vfo > 0 else "[outv]"
        parts.append(f"{v_label}fade=t=in:st=0:d={vfi:.6f}{nxt}")
        v_label = nxt
    if vfo > 0:
        fade_out_start = max(0.0, total_duration - vfo)
        parts.append(f"{v_label}fade=t=out:st={fade_out_start:.6f}:d={vfo:.6f}[outv]")
        v_label = "[outv]"
    if v_label != "[outv]":
        parts.append(f"{v_label}null[outv]")
    a_label = "[araw]"
    if str(audio_loudnorm) == "on":
        parts.append(f"{a_label}loudnorm=I=-16:TP=-1.5:LRA=11[aloud]")
        a_label = "[aloud]"
    if afi > 0:
        nxt = "[afadein]" if afo > 0 else "[outa]"
        parts.append(f"{a_label}afade=t=in:st=0:d={afi:.6f}{nxt}")
        a_label = nxt
    if afo > 0:
        fade_out_start = max(0.0, total_duration - afo)
        parts.append(f"{a_label}afade=t=out:st={fade_out_start:.6f}:d={afo:.6f}[outa]")
        a_label = "[outa]"
    if a_label != "[outa]":
        parts.append(f"{a_label}anull[outa]")
    return ";".join(parts), logs, total_duration


class AGSoftVideoConcatenatePlus:
    OUTPUT_NODE = True
    RETURN_TYPES = ("VIDEO", "STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT",
                    "FLOAT", "INT")
    RETURN_NAMES = ("video", "video_path", "duration_seconds", "duration_timecode",
                    "file_size_mb", "width", "height", "fps", "frames_est")
    OUTPUT_TOOLTIPS = OUTPUT_TOOLTIPS_COMMON
    FUNCTION = "concat_videos_plus"
    CATEGORY = "AGSoft/Video"
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "🎬 Advanced concatenation with xfade transitions, desync protection, "
        "normalization, fades, loudnorm, NVENC/CPU. Accepts VIDEO, IMAGE tensors, paths.\n"
        "---\n"
        "🎬 Продвинутая склейка с xfade-переходами, защитой от рассинхрона, "
        "нормализацией, фейдами, loudnorm, NVENC/CPU. Принимает VIDEO, IMAGE-тензоры, пути."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "output_name": ("STRING", {
                "default": "output_video_plus.mp4",
                "tooltip": ("Output filename.\n---\nИмя итогового файла.")}),
            "output_path": ("STRING", {
                "default": "",
                "tooltip": (
                    "Subfolder inside ComfyUI output (absolute paths are rejected).\n"
                    "---\n"
                    "Подпапка внутри output ComfyUI (абсолютные пути отклоняются).")}),
            "inputs_count": ([str(i) for i in range(2, MAX_CLIPS + 1)], {
                "default": "2",
                "tooltip": ("Number of visible clip sockets.\n---\nЧисло видимых сокетов.")}),
            "transition_type": (["cut", "random"] + NATIVE_XFADE_TRANSITIONS, {
                "default": "fade",
                "tooltip": (
                    "Transition type. cut = hard cut, random = per-junction random.\n"
                    "---\n"
                    "Тип перехода. cut = жёсткая склейка, random = случайный на стыке.")}),
            "transition_duration": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                "tooltip": (
                    "Transition duration, 0 = cut. Clamped to clip length.\n"
                    "---\n"
                    "Длительность перехода, 0 = cut. Ограничивается длиной клипа.")}),
            "audio_mode": (["auto", "mute"], {
                "default": "auto",
                "tooltip": (
                    "auto = keep audio, missing replaced with silence. mute = silent.\n"
                    "---\n"
                    "auto = сохранять звук, отсутствующий = тишина. mute = без звука.")}),
            "encoder_mode": (["auto", "nvenc", "cpu"], {
                "default": "auto",
                "tooltip": (
                    "auto = NVENC if available else CPU.\n"
                    "---\n"
                    "auto = NVENC при наличии, иначе CPU.")}),
            "quality_preset": (["fast", "balanced", "quality"], {
                "default": "balanced",
                "tooltip": ("Speed/quality trade-off.\n---\nБаланс скорости и качества.")}),
            "output_fps": (["auto", "24", "25", "29.97", "30", "50", "59.94", "60"], {
                "default": "auto",
                "tooltip": ("auto = first clip FPS.\n---\nauto = FPS первого клипа.")}),
            "resize_mode": (["first_clip", "max_size", "min_size", "custom"], {
                "default": "first_clip",
                "tooltip": (
                    "Output resolution source.\n---\nИсточник разрешения результата.")}),
            "fit_mode": (["letterbox", "crop", "stretch"], {
                "default": "letterbox",
                "tooltip": (
                    "letterbox = bars, crop = fill+crop, stretch = stretch.\n"
                    "---\n"
                    "letterbox = полосы, crop = заполнить+обрезать, stretch = растянуть.")}),
            "custom_width": ("INT", {
                "default": 0, "min": 0, "max": 8192, "step": 2,
                "tooltip": ("Custom width (resize_mode=custom).\n---\nКастомная ширина.")}),
            "custom_height": ("INT", {
                "default": 0, "min": 0, "max": 8192, "step": 2,
                "tooltip": ("Custom height (resize_mode=custom).\n---\nКастомная высота.")}),
            "auto_color": (["off", "on"], {
                "default": "off",
                "tooltip": (
                    "Experimental per-clip color normalization.\n"
                    "---\n"
                    "Экспериментальная авто-нормализация цвета.")}),
            "video_fade_in": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                "tooltip": ("Global video fade-in, s.\n---\nГлобальное появление видео, сек.")}),
            "video_fade_out": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                "tooltip": ("Global video fade-out, s.\n---\nГлобальное затухание видео, сек.")}),
            "audio_fade_in": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                "tooltip": ("Global audio fade-in, s.\n---\nГлобальное появление звука, сек.")}),
            "audio_fade_out": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                "tooltip": ("Global audio fade-out, s.\n---\nГлобальное затухание звука, сек.")}),
            "audio_loudnorm": (["off", "on"], {
                "default": "off",
                "tooltip": (
                    "EBU R128 loudness normalization.\n"
                    "---\n"
                    "Нормализация громкости EBU R128.")}),
            "save_mode": (["preview", "direct"], {
                "default": "preview",
                "tooltip": (
                    "preview = temp + Save button. direct = write to output.\n"
                    "---\n"
                    "preview = temp + кнопка Save. direct = сразу в output.")}),
            "tensor_fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                "tooltip": (
                    "FPS for IMAGE tensor clips.\n---\nFPS для IMAGE-тензоров.")}),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffff, "step": 1,
                "tooltip": (
                    "Seed for random transitions (reproducible).\n"
                    "---\n"
                    "Seed для случайных переходов (воспроизводимо).")}),
        }
        # Клиповые входы опциональны: подключаем только нужное число клипов.
        # Clip inputs are optional: only the needed number of clips is connected.
        optional = {}
        optional.update(_clip_path_widgets())
        optional.update(_clip_inputs())
        return {"required": required, "optional": optional,
                "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        keys = ["output_name", "output_path", "inputs_count", "transition_type",
                "transition_duration", "audio_mode", "encoder_mode", "quality_preset",
                "output_fps", "resize_mode", "fit_mode", "custom_width", "custom_height",
                "auto_color", "video_fade_in", "video_fade_out", "audio_fade_in",
                "audio_fade_out", "audio_loudnorm", "save_mode", "tensor_fps", "seed"]
        return _widgets_hash(kwargs, keys)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return _validate_paths(kwargs)

    async def concat_videos_plus(self, output_name, output_path, inputs_count,
                                 transition_type, transition_duration, audio_mode,
                                 encoder_mode, quality_preset, output_fps, resize_mode,
                                 fit_mode, custom_width, custom_height, auto_color,
                                 video_fade_in, video_fade_out, audio_fade_in,
                                 audio_fade_out, audio_loudnorm, save_mode="preview",
                                 tensor_fps=24.0, seed=0, unique_id=None, **kwargs):
        node_id = str(unique_id or "0")
        temp_dir = _temp_dir(node_id)
        send_p = _progress_sender(node_id)
        rng = random.Random(safe_int(seed, 0))
        send_p(2, "resolve")
        video_list = await collect_clips(kwargs, safe_float(tensor_fps, 24.0), temp_dir,
                                         lambda p: send_p(2 + p // 20, "resolve"))
        expected = safe_int(inputs_count, 2)
        if len(video_list) < 2:
            raise ValueError("[AGSoft Concat Plus] Need at least 2 clips! / Минимум 2 клипа!")
        if len(video_list) < expected:
            raise ValueError(
                f"[AGSoft Concat Plus] Connected {len(video_list)} of {expected}. / "
                f"Подключено {len(video_list)} из {expected}.")
        infos = [await parse_media_info(p) for p in video_list]
        no_video = [p for p, i in zip(video_list, infos) if i["width"] <= 0]
        if no_video:
            raise ValueError(f"[AGSoft Concat Plus] No video stream / нет видеопотока: {no_video}")
        ow, oh = choose_output_size(infos, resize_mode, custom_width, custom_height)
        ofps = choose_output_fps(infos, output_fps)
        filter_complex, logs, total_duration = build_filter_complex(
            infos, transition_type, transition_duration, ow, oh, ofps,
            audio_mode, fit_mode, auto_color, video_fade_in, video_fade_out,
            audio_fade_in, audio_fade_out, audio_loudnorm, rng)
        for line in logs:
            logger.info(f"[AGSoft Concat Plus] {line}")
        final_path, sub, kind, safe_name = _final_destination(
            output_name, output_path, save_mode, temp_dir, "output_video_plus.mp4")
        encoder = await choose_encoder(encoder_mode)
        input_args = []
        for p in video_list:
            input_args += ["-i", p]
        base_cmd = [FFMPEG_PATH, "-y", "-hide_banner", *input_args,
                    "-filter_complex", filter_complex,
                    "-map", "[outv]", "-map", "[outa]"]
        audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
        mux_args = ["-movflags", "+faststart"] if safe_name.lower().endswith(
            (".mp4", ".mov")) else []

        def on_pct(p):
            send_p(10 + int(p * 0.9), "render")
            if p % 20 == 0:
                logger.info(f"[AGSoft Concat Plus] Render progress: {p}%")

        cmd = base_cmd + build_encoder_args(encoder, quality_preset) + audio_args \
            + mux_args + [final_path]
        rc, err = await ffmpeg_run(cmd, total=total_duration, on_progress=on_pct)
        if rc != 0 and encoder == "h264_nvenc" and str(encoder_mode) == "auto":
            logger.warning("[AGSoft Concat Plus] NVENC failed, fallback to libx264.")
            cmd = base_cmd + build_encoder_args("libx264", quality_preset) + audio_args \
                + mux_args + [final_path]
            rc, err = await ffmpeg_run(cmd, total=total_duration, on_progress=on_pct)
        if rc != 0 or not os.path.isfile(final_path):
            logger.error(f"[AGSoft Concat Plus] FFmpeg log:\n{err[-4000:]}")
            last_line = next((l.strip() for l in reversed(err.splitlines())
                              if l.strip()), "FFmpeg error")
            raise RuntimeError(f"[AGSoft Concat Plus] FFmpeg failed: {last_line}")
        info = await parse_media_info(final_path)
        timecode = format_timecode(info["duration"])
        send_p(100, "done")
        preview_sub = sub if kind == "output" else f"agsoft_concat_{node_id}"
        preview = _preview_dict(final_path, kind, preview_sub, info)
        logger.info(f"[AGSoft Concat Plus] Saved: {final_path} | {timecode} | "
                    f"{info['size_mb']} MB | {info['width']}x{info['height']} | {info['fps']}")
        return {
            "ui": {"gifs": [preview]},
            "result": (_video_output(final_path), final_path, info["duration"], timecode,
                       info["size_mb"], info["width"], info["height"], info["fps"],
                       info["frames"]),
        }


@PromptServer.instance.routes.post("/agsoft/concat_save")
async def agsoft_concat_save(request):
    # Копия temp-результата в output; только loopback и containment.
    # Copies temp result into output; loopback-only with containment.
    if request.remote not in ("127.0.0.1", "::1"):
        return web.json_response({"error": "loopback only"}, status=403)
    try:
        data = await request.json()
        temp_path = os.path.realpath(str(data.get("temp_path", "")))
        if not _contained(temp_path, [folder_paths.get_temp_directory()]):
            return web.json_response({"error": "temp containment"}, status=400)
        if not os.path.isfile(temp_path) or os.path.getsize(temp_path) <= 0:
            return web.json_response({"error": "source missing"}, status=400)
        name = os.path.basename(str(data.get("name", "result.mp4")))
        stem, ext = os.path.splitext(name)
        name = f"{stem or 'result'}{ext or '.mp4'}"
        sub = safe_subfolder(data.get("subfolder", ""))
        target_dir = os.path.join(folder_paths.get_output_directory(), sub)
        os.makedirs(target_dir, exist_ok=True)
        dest = unique_path(os.path.join(target_dir, name))
        shutil.copyfile(temp_path, dest)
        return web.json_response({
            "path": dest, "name": os.path.basename(dest), "subfolder": sub})
    except Exception as e:
        logger.error(f"[AGSoft Concat] Save route error: {e}")
        return web.json_response({"error": str(e)}, status=500)


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoConcatenate": AGSoftVideoConcatenate,
    "AGSoftVideoConcatenatePlus": AGSoftVideoConcatenatePlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoConcatenate": "🎬AGSoft Video Concatenate",
    "AGSoftVideoConcatenatePlus": "🎬🪡AGSoft Video Concatenate Plus",
}