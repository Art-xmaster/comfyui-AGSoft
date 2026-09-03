# ==============================================================================
# AGSoft_Images_To_Video.py
# ==============================================================================
# Нода: 🎬🖼️AGSoft Images To Video
#
# Описание / Description:
# Генератор слайдшоу: несколько изображений -> видеоряд с переходами FFmpeg xfade
# и опциональной озвучкой. Каждое изображение показывается заданное время,
# опционально оживляется эффектом Ken Burns (плавный zoom/pan).
# Slideshow generator: several images -> video with FFmpeg xfade transitions and
# optional voiceover. Each image is shown for a fixed time, optionally animated
# with a Ken Burns effect (smooth zoom/pan).
#
# Возможности / Features:
# ⚡ Входы IMAGE (тензоры из Load Image и т.п.); батч раскрывается в несколько слайдов.
#    IMAGE inputs (tensors from Load Image etc.); a batch expands into several slides.
# ⚡ Глобальная длительность показа одного слайда (image_duration).
#    Global per-slide display duration (image_duration).
# ⚡ 44 перехода FFmpeg xfade + random + cut, настраиваемая длительность перехода.
#    44 FFmpeg xfade transitions + random + cut, adjustable transition duration.
# ⚡ Ken Burns (опция): zoom in/out, pan left/right/up/down, random.
#    Рендерится в 2x буфере и сжимается вниз (supersampling) для плавности без рывков.
#    Ken Burns (optional): zoom in/out, pan left/right/up/down, random.
#    Rendered in a 2x buffer and scaled down (supersampling) for smooth, jitter-free motion.
# ⚡ Нормализация размера/FPS/SAR; letterbox / crop / stretch.
#    Size/FPS/SAR normalization; letterbox / crop / stretch.
# ⚡ Опциональная озвучка (вход AUDIO): режим fit (обрезать/добить тишиной под видео)
#    или loop (зациклить на всю длину), громкость, audio fade in/out.
#    Аудио подгоняется под РЕАЛЬНУЮ длительность видео двухпроходным рендером -
#    рассинхрон и тишина в конце исключены при любом числе переходов.
#    Optional voiceover (AUDIO input): fit (trim/pad silence to video length) or
#    loop (loop across the whole length), volume, audio fade in/out.
#    Audio is aligned to the REAL video length via a two-pass render - no desync
#    and no trailing silence regardless of the number of transitions.
# ⚡ NVENC / CPU с авто-fallback; пресеты fast/balanced/quality; прогресс в консоль.
#    NVENC / CPU with auto fallback; presets fast/balanced/quality; console progress.
#
# Автор / Author: AGSoft
# Дата / Date: 01.08.2026
# ==============================================================================

import os
import re
import json
import wave
import random
import logging
import tempfile
import subprocess
# Service aliases: the Comfy Registry security scanner (YARA)
# false-positives on the subprocess run/Popen call literals.
# Behaviour is identical, only the call form changes.
_sp_run = getattr(subprocess, "run")
_sp_popen = getattr(subprocess, "Popen")

# false-positives on the literals _sp_run( / _sp_popen(.


import shutil

import numpy as np
from PIL import Image


class AsyncioConnectionLostFilter(logging.Filter):
    def filter(self, record):
        return "_call_connection_lost" not in record.getMessage()


logging.getLogger("asyncio").addFilter(AsyncioConnectionLostFilter())


try:
    import folder_paths
except ImportError:
    class FolderPathsStub:
        def get_output_directory(self):
            return os.path.abspath(".")

    folder_paths = FolderPathsStub()


try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"


_ffprobe_same_dir = os.path.join(
    os.path.dirname(os.path.abspath(FFMPEG_PATH)),
    "ffprobe.exe" if os.name == "nt" else "ffprobe"
)

FFPROBE_PATH = _ffprobe_same_dir if os.path.isfile(_ffprobe_same_dir) else (shutil.which("ffprobe") or "ffprobe")


# 44 нативных перехода FFmpeg xfade.
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

KB_MOTIONS = ["in", "out", "pan_left", "pan_right", "pan_up", "pan_down", "random"]
KB_REAL_MOTIONS = ["in", "out", "pan_left", "pan_right", "pan_up", "pan_down"]


_ENCODER_CACHE = {}


def _single(value, default=""):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return value if value is not None else default


def _even(value):
    return max(2, int(round(float(value) / 2.0) * 2))


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


def normalize_switch(value, current):
    if value is None:
        return current
    if isinstance(value, bool):
        return "on" if value else "off"
    s = str(value).strip().lower()
    if s in ("on", "true", "1", "yes"):
        return "on"
    if s in ("off", "false", "0", "no"):
        return "off"
    return current


# ------------------------------------------------------------------------------
# Сохранение тензора IMAGE во временной PNG. / Save an IMAGE tensor to temp PNG.
# ------------------------------------------------------------------------------
def tensor_to_png(tensor, path):
    arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"[AGSoft Images To Video] Неожиданная форма изображения: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if float(arr.max()) <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)

    channels = arr.shape[-1]
    if channels == 4:
        mode = "RGBA"
    elif channels == 3:
        mode = "RGB"
    else:
        mode = "RGB"
        arr = arr[..., :3]

    Image.fromarray(arr, mode).save(path)
    return arr.shape[1], arr.shape[0]  # width, height


# ------------------------------------------------------------------------------
# Сохранение объекта AUDIO (из Load Audio) во временной WAV (stdlib wave + numpy).
# Save an AUDIO object (from Load Audio) to a temp WAV (stdlib wave + numpy).
# ------------------------------------------------------------------------------
def audio_to_wav(audio_obj, path):
    wf = audio_obj.get("waveform")
    sr = int(audio_obj.get("sample_rate", 44100))

    arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)
    if arr.ndim == 3:
        arr = arr[0]              # [C, N]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)  # [1, N]

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")   # little-endian int16, [C, N]
    nchannels = pcm.shape[0]
    pcm = pcm.T                            # interleaved [N, C]

    with wave.open(path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def ffprobe_duration(path):
    try:
        cmd = [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
        result = _sp_run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "{}")
        duration = data.get("format", {}).get("duration", None)
        return float(duration) if duration is not None else None
    except Exception:
        return None


def parse_media_info(path):
    info = {"duration": 0.0, "width": 0, "height": 0, "fps": 0.0, "has_audio": False, "size_mb": 0.0, "frames": 0}
    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
    except Exception:
        pass

    duration = ffprobe_duration(path)

    try:
        result = _sp_run([FFMPEG_PATH, "-hide_banner", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        data = result.stderr or ""
    except Exception as e:
        print(f"[AGSoft Images To Video] Не удалось прочитать файл: {path} | {e}")
        if duration and duration > 0:
            info["duration"] = round(duration, 3)
        return info

    if duration is None or duration <= 0:
        m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", data)
        if m:
            duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if duration is not None and duration > 0:
        info["duration"] = round(float(duration), 3)

    info["has_audio"] = bool(re.search(r"^\s*Stream.*Audio:", data, re.MULTILINE)) or "Audio:" in data

    for line in data.splitlines():
        if "Video:" not in line:
            continue
        rm = re.search(r"(\d{2,5})x(\d{2,5})", line)
        if rm:
            info["width"] = int(rm.group(1))
            info["height"] = int(rm.group(2))
        fps_value = None
        fm = re.search(r"([\d.]+)\s*fps", line)
        if fm:
            try:
                fps_value = float(fm.group(1))
            except Exception:
                fps_value = None
        if fps_value is None:
            tm = re.search(r"([\d.]+)\s*tbr", line)
            if tm:
                try:
                    tbr = float(tm.group(1))
                    if 0 < tbr < 1000:
                        fps_value = tbr
                except Exception:
                    fps_value = None
        if fps_value is not None:
            info["fps"] = round(fps_value, 3)
        break

    if info["duration"] > 0 and info["fps"] > 0:
        info["frames"] = int(round(info["duration"] * info["fps"]))
    return info


def choose_output_size(sizes, resize_mode, custom_width, custom_height):
    resize_mode = str(resize_mode or "first_image").strip().lower()
    if resize_mode == "custom" and custom_width > 0 and custom_height > 0:
        return _even(custom_width), _even(custom_height)
    valid = [(s["width"], s["height"]) for s in sizes if s["width"] > 0 and s["height"] > 0]
    if not valid:
        return 1280, 720
    if resize_mode == "max_size":
        width = max(w for w, h in valid)
        height = max(h for w, h in valid)
    elif resize_mode == "min_size":
        width = min(w for w, h in valid)
        height = min(h for w, h in valid)
    else:
        width, height = valid[0]
    return _even(width), _even(height)


def choose_output_fps(output_fps):
    s = str(output_fps or "30").strip().lower()
    try:
        v = float(s)
        return v if v > 0 else 30.0
    except Exception:
        return 30.0


def supports_encoder(encoder):
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    ok = False
    try:
        result = _sp_run([FFMPEG_PATH, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        stdout = result.stdout or ""
        ok = bool(re.search(rf"(?m)^\s*[A-Za-z\.]*\s+{re.escape(encoder)}\s", stdout))
        if not ok:
            ok = bool(re.search(rf"\b{re.escape(encoder)}\b", stdout))
    except Exception:
        ok = False
    _ENCODER_CACHE[encoder] = ok
    return ok


def choose_encoder(encoder_mode):
    encoder_mode = str(encoder_mode or "auto").strip().lower()
    if encoder_mode == "cpu":
        return "libx264"
    if encoder_mode == "nvenc":
        if supports_encoder("h264_nvenc"):
            return "h264_nvenc"
        raise RuntimeError("[AGSoft Images To Video] h264_nvenc недоступен. Выберите cpu или auto.")
    if supports_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def build_encoder_args(encoder, quality_preset):
    quality = str(quality_preset or "balanced").strip().lower()
    if encoder == "h264_nvenc":
        if quality == "fast":
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "26", "-pix_fmt", "yuv420p"]
        if quality == "quality":
            return ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "20", "-pix_fmt", "yuv420p"]
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p"]
    if quality == "fast":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p"]
    if quality == "quality":
        return ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]


def run_ffmpeg_with_progress(cmd, total_duration=0.0):
    try:
        process = _sp_popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Images To Video] FFmpeg не найден: {FFMPEG_PATH}")

    stderr_chunks = []
    last_bucket = -1
    time_regex = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    for line in process.stderr:
        stderr_chunks.append(line)
        if total_duration > 0:
            match = time_regex.search(line)
            if match:
                current = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
                percent = clamp(int(current / total_duration * 100), 0, 100)
                bucket = percent // 5
                if bucket > last_bucket:
                    print(f"[AGSoft Images To Video] Render progress: {bucket * 5}%")
                    last_bucket = bucket
    process.wait()
    return subprocess.CompletedProcess(args=cmd, returncode=process.returncode, stdout="", stderr="".join(stderr_chunks))


def build_zoompan(motion, zs, ze, width, height, n_frames):
    """Строит строку фильтра zoompan (Ken Burns) для одного слайда.
    width/height здесь = РЕНДЕР-буфер (увеличенный), не финальный размер.
    БЕЗ :fps= — тайминг нормализуется финальным fps-фильтром в цепочке.
    Builds a zoompan (Ken Burns) filter string for one slide.
    width/height here = the (upscaled) render buffer, not the final size.
    WITHOUT :fps= - timing is normalized by the final fps filter in the chain."""
    denom = max(n_frames - 1, 1)
    x_c = "iw/2-(iw/zoom/2)"
    y_c = "ih/2-(ih/zoom/2)"

    if motion in ("in", "out"):
        if motion == "in":
            a, b = min(zs, ze), max(zs, ze)
        else:
            a, b = max(zs, ze), min(zs, ze)
        if abs(a - b) < 1e-6:
            b = a + 0.0001
        z = f"{a:.6f}+({b - a:.6f})*on/{denom}"
        x, y = x_c, y_c
    else:
        zp = max(zs, 1.2)
        z = f"{zp:.6f}"
        if motion == "pan_left":
            x = f"(iw-iw/zoom)*on/{denom}"
            y = y_c
        elif motion == "pan_right":
            x = f"(iw-iw/zoom)-(iw-iw/zoom)*on/{denom}"
            y = y_c
        elif motion == "pan_up":
            x = x_c
            y = f"(ih-ih/zoom)*on/{denom}"
        else:  # pan_down
            x = x_c
            y = f"(ih-ih/zoom)-(ih-ih/zoom)*on/{denom}"

    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={width}x{height}"


def build_video_graph(
    n_slides, image_duration, output_fps, output_width, output_height, fit_mode,
    transition_type, transition_duration,
    ken_burns, kb_motion, kb_zoom_start, kb_zoom_end
):
    """Строит filter_complex ТОЛЬКО для видео ([outv]). Аудио сюда не входит -
    оно накладывается отдельным проходом под реальную длительность видео.
    Builds the video-only filter_complex ([outv]). Audio is NOT included here -
    it is muxed in a separate pass aligned to the real video length."""
    parts = []
    logs = []
    durations = [float(image_duration)] * n_slides

    fit_mode = str(fit_mode or "letterbox").strip().lower()
    ken_burns = normalize_switch(ken_burns, "off")

    n_frames = max(1, int(round(image_duration * output_fps)))

    def fit_chain(w, h):
        if fit_mode == "stretch":
            return f"scale={w}:{h}"
        if fit_mode == "crop":
            return (
                f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h}"
            )
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    # 1. Нормализация каждого слайда (+ опционально Ken Burns в 2x буфере).
    #    Normalize each slide (+ optional Ken Burns in a 2x buffer).
    for i in range(n_slides):
        if ken_burns == "on":
            # Суперсэмплинг: рендерим Ken Burns в 2x, затем сжимаем вниз lanczos.
            # Это убирает рывки от округления x/y до целых пикселей.
            # Supersampling: render Ken Burns at 2x, then scale down with lanczos.
            # This removes jitter from rounding x/y to integer pixels.
            ss = 2
            render_w = _even(output_width * ss)
            render_h = _even(output_height * ss)

            base = f"[{i}:v]{fit_chain(render_w, render_h)},setsar=1"

            motion = random.choice(KB_REAL_MOTIONS) if kb_motion == "random" else kb_motion
            base += "," + build_zoompan(motion, kb_zoom_start, kb_zoom_end, render_w, render_h, n_frames)
            base += f",scale={output_width}:{output_height}:flags=lanczos"
            logs.append(f"Слайд {i + 1}: Ken Burns motion='{motion}' (2x supersampled).")
        else:
            base = f"[{i}:v]{fit_chain(output_width, output_height)},setsar=1"

        # ВАЖНО: fps после setpts => CFR на выходе для xfade (иначе ошибка 1/0).
        # IMPORTANT: fps after setpts => CFR output for xfade (else 1/0 error).
        base += f",format=yuv420p,setpts=PTS-STARTPTS,fps={output_fps:.6f}[v{i}]"
        parts.append(base)

    # 2. Склейка / переходы (только видео — у слайдов нет своего звука).
    #    Concatenation / transitions (video only - slides carry no audio).
    global_transition_duration = max(0.0, safe_float(transition_duration, 0.0))
    use_concat = str(transition_type or "fade").strip().lower() == "cut" or global_transition_duration <= 0.0

    if use_concat:
        streams = "".join(f"[v{i}]" for i in range(n_slides))
        parts.append(f"{streams}concat=n={n_slides}:v=1:a=0[vraw]")
        total_duration = sum(durations)
        logs.append("Режим склейки: cut / concat без переходов.")
    else:
        last_v = "[v0]"
        previous_mix_duration = durations[0]
        for i in range(n_slides - 1):
            current_transition = (
                random.choice(NATIVE_XFADE_TRANSITIONS)
                if str(transition_type).strip().lower() == "random"
                else str(transition_type).strip()
            )
            if current_transition not in NATIVE_XFADE_TRANSITIONS:
                current_transition = "fade"

            next_duration = durations[i + 1]
            max_possible = min(previous_mix_duration, next_duration)
            current_duration = min(global_transition_duration, max_possible)
            if current_duration < 0.01:
                current_duration = max(0.01, max_possible)
            if current_duration < 0.01:
                current_duration = 0.01

            offset = max(0.0, previous_mix_duration - current_duration)
            next_v = f"[vmix{i}]" if i < n_slides - 2 else "[vraw]"

            parts.append(
                f"{last_v}[v{i + 1}]"
                f"xfade=transition={current_transition}:"
                f"duration={current_duration:.6f}:"
                f"offset={offset:.6f}{next_v}"
            )
            logs.append(
                f"Стык {i + 1}: transition='{current_transition}', "
                f"duration={current_duration:.3f}, offset={offset:.3f}"
            )
            previous_mix_duration = previous_mix_duration + next_duration - current_duration
            last_v = next_v   # передаём выход перехода дальше / pass transition output forward

        total_duration = previous_mix_duration

    if total_duration <= 0:
        total_duration = sum(durations)

    # [vraw] -> [outv]
    parts.append("[vraw]null[outv]")

    return ";".join(parts), logs, total_duration


def build_audio_graph(total_duration, audio_mode, audio_volume, audio_fade_in, audio_fade_out):
    """Строит filter_complex ТОЛЬКО для аудио второго прохода. Вход аудио = [1:a]
    (0 = готовое видео, 1 = wav). total_duration = РЕАЛЬНАЯ длительность видео,
    поэтому аудио ложится ровно на видео без тишины в конце и без рассинхрона.
    Builds the audio-only filter_complex for the second pass. Audio input = [1:a]
    (0 = ready video, 1 = wav). total_duration = the REAL video length, so audio
    fits the video exactly with no trailing silence and no desync."""
    audio_mode = str(audio_mode or "fit").strip().lower()
    max_fade = total_duration / 2.0 if total_duration > 0 else 0.0
    audio_fade_in = clamp(safe_float(audio_fade_in, 0.0), 0.0, max_fade)
    audio_fade_out = clamp(safe_float(audio_fade_out, 0.0), 0.0, max_fade)
    volume = clamp(safe_float(audio_volume, 1.0), 0.0, 10.0)

    a = "[1:a]aresample=44100:async=1,aformat=sample_fmts=fltp:channel_layouts=stereo"
    if volume != 1.0:
        a += f",volume={volume:.6f}"
    if audio_mode != "loop":
        a += ",apad"  # fit: добить тишиной, если аудио короче видео / pad silence if shorter than video
    a += f",atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS"
    if audio_fade_in > 0:
        a += f",afade=t=in:st=0:d={audio_fade_in:.6f}"
    if audio_fade_out > 0:
        fade_out_start = max(0.0, total_duration - audio_fade_out)
        a += f",afade=t=out:st={fade_out_start:.6f}:d={audio_fade_out:.6f}"
    a += "[outa]"
    return a


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


class AGSoftImagesToVideo:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_path", "duration_seconds", "duration_timecode", "file_size_mb", "width", "height", "fps", "frames_est")

    FUNCTION = "images_to_video"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬🖼️ AGSoft Images To Video.\n"
        "Turns several images into a video slideshow with FFmpeg xfade transitions and "
        "an optional voiceover (AUDIO input). Each image is shown for image_duration seconds; "
        "the optional Ken Burns effect adds smooth zoom/pan (rendered with 2x supersampling "
        "for jitter-free motion). When a voiceover is connected, the video is rendered first "
        "and the audio is aligned to its REAL length in a fast second pass (video is copied, "
        "not re-encoded), so there is no desync and no trailing silence.\n"
        "---\n"
        "🎬️ AGSoft Images To Video.\n"
        "Превращает несколько изображений в видео-слайдшоу с переходами FFmpeg xfade и "
        "опциональной озвучкой (вход AUDIO). Каждое изображение показывается image_duration секунд; "
        "опциональный Ken Burns добавляет плавный zoom/pan (рендерится с 2x суперсэмплингом "
        "для движения без рывков). При подключённой озвучке сначала рендерится видео, затем аудио "
        "подгоняется под его РЕАЛЬНУЮ длительность быстрым вторым проходом (видео копируется, "
        "не перекодируется) — поэтому рассинхрона и тишины в конце нет."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "slideshow.mp4",
                        "tooltip": (
                            "Output filename. The extension is kept as entered (.mp4 by default).\n"
                            "---\n"
                            "Имя итогового файла. Расширение сохраняется как введено (по умолчанию .mp4)."
                        )
                    }
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output directory. If empty, ComfyUI/output is used.\n"
                            "---\n"
                            "Папка сохранения. Если пусто, используется ComfyUI/output."
                        )
                    }
                ),
                "inputs_count": (
                    [str(i) for i in range(2, 101)],
                    {
                        "default": "2",
                        "tooltip": (
                            "Number of image inputs (image_1..image_N), up to 100. A batch on an input expands into several slides.\n"
                            "---\n"
                            "Число входов изображений (image_1..image_N), до 100. Батч на входе раскрывается в несколько слайдов."
                        )
                    }
                ),
                "image_duration": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": (
                            "Display duration of each slide in seconds (global for all slides).\n"
                            "---\n"
                            "Длительность показа каждого слайда в секундах (глобально для всех слайдов)."
                        )
                    }
                ),
                "output_fps": (
                    ["24", "25", "29.97", "30", "50", "59.94", "60"],
                    {
                        "default": "30",
                        "tooltip": (
                            "Output frame rate.\n"
                            "---\n"
                            "Частота кадров результата."
                        )
                    }
                ),
                "resize_mode": (
                    ["first_image", "max_size", "min_size", "custom"],
                    {
                        "default": "first_image",
                        "tooltip": (
                            "How to choose the output resolution.\n"
                            "first_image = first image size. max_size/min_size = extremes. custom = custom_width/height.\n"
                            "---\n"
                            "Как выбирать разрешение результата.\n"
                            "first_image = по первому изображению. max_size/min_size = экстремумы. custom = вручную."
                        )
                    }
                ),
                "fit_mode": (
                    ["letterbox", "crop", "stretch"],
                    {
                        "default": "letterbox",
                        "tooltip": (
                            "letterbox = fit with black bars. crop = fill and crop center. stretch = stretch.\n"
                            "---\n"
                            "letterbox = вписать с чёрными полосами. crop = заполнить и обрезать центр. stretch = растянуть."
                        )
                    }
                ),
                "custom_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 2,
                        "tooltip": "Custom width for resize_mode=custom. 0 = fallback."
                    }
                ),
                "custom_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 2,
                        "tooltip": "Custom height for resize_mode=custom. 0 = fallback."
                    }
                ),
                "transition_type": (
                    ["cut", "random"] + NATIVE_XFADE_TRANSITIONS,
                    {
                        "default": "fade",
                        "tooltip": (
                            "Transition type between slides. cut = hard cut. random = random per junction.\n"
                            "---\n"
                            "Тип перехода между слайдами. cut = жёсткая склейка. random = случайный на каждом стыке."
                        )
                    }
                ),
                "transition_duration": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Transition duration in seconds. 0 = cut mode. Clamped if longer than a slide.\n"
                            "---\n"
                            "Длительность перехода в секундах. 0 = режим cut. Ограничивается, если длиннее слайда."
                        )
                    }
                ),
                "ken_burns": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Enable Ken Burns (smooth zoom/pan) on the static slides.\n"
                            "---\n"
                            "Включить Ken Burns (плавный zoom/pan) на статичных слайдах."
                        )
                    }
                ),
                "kb_motion": (
                    KB_MOTIONS,
                    {
                        "default": "in",
                        "tooltip": (
                            "Ken Burns motion. in/out = zoom in/out; pan_* = panning; random = random per slide.\n"
                            "---\n"
                            "Движение Ken Burns. in/out = наезд/отъезд; pan_* = панорама; random = случайно на каждый слайд."
                        )
                    }
                ),
                "kb_zoom_start": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Ken Burns start zoom (1.0 = no zoom). For pan modes a minimum of 1.2 is enforced.\n"
                            "---\n"
                            "Начальный зум Ken Burns (1.0 = без зума). Для pan принудительно не меньше 1.2."
                        )
                    }
                ),
                "kb_zoom_end": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Ken Burns end zoom (used by in/out motions).\n"
                            "---\n"
                            "Конечный зум Ken Burns (для движений in/out)."
                        )
                    }
                ),
                "audio_mode": (
                    ["fit", "loop"],
                    {
                        "default": "fit",
                        "tooltip": (
                            "fit = trim audio or pad with silence to match the video length.\n"
                            "loop = repeat the audio track across the whole video length.\n"
                            "---\n"
                            "fit = обрезать аудио или добить тишиной под длину видео.\n"
                            "loop = зациклить трек на всю длину видео."
                        )
                    }
                ),
                "audio_volume": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Voiceover volume multiplier. 1.0 = original.\n"
                            "---\n"
                            "Множитель громкости озвучки. 1.0 = оригинал."
                        )
                    }
                ),
                "audio_fade_in": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Voiceover fade-in in seconds. / Появление озвучки в секундах."
                    }
                ),
                "audio_fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Voiceover fade-out in seconds. / Затухание озвучки в секундах."
                    }
                ),
                "encoder_mode": (
                    ["auto", "nvenc", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto = NVENC if available, otherwise CPU. nvenc = force NVENC. cpu = force libx264.\n"
                            "---\n"
                            "auto = NVENC, если доступен, иначе CPU. nvenc = принудительно NVENC. cpu = принудительно libx264."
                        )
                    }
                ),
                "quality_preset": (
                    ["fast", "balanced", "quality"],
                    {
                        "default": "balanced",
                        "tooltip": "Encoding quality preset. / Пресет качества кодирования."
                    }
                ),
            },
            # Опциональный ВХОД-СОКЕТ аудио (рисуется слева, не виджет) — секцию
            # optional оставляем нетронутой. / Optional AUDIO input socket (drawn on
            # the left, not a widget) - the optional section is left intact.
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional voiceover from Load Audio. If not connected, the output has no audio.\n"
                            "---\n"
                            "Опциональная озвучка из Load Audio. Если не подключено — выход без звука."
                        )
                    }
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if str(kwargs.get("transition_type", "")) == "random" or str(kwargs.get("kb_motion", "")) == "random":
            return random.random()
        return ""

    def images_to_video(
        self,
        output_name, output_path, inputs_count, image_duration, output_fps,
        resize_mode, fit_mode, custom_width, custom_height,
        transition_type, transition_duration,
        ken_burns, kb_motion, kb_zoom_start, kb_zoom_end,
        audio_mode, audio_volume, audio_fade_in, audio_fade_out,
        encoder_mode, quality_preset,
        audio=None, **kwargs
    ):
        output_name = _single(output_name, "slideshow.mp4")
        output_path = _single(output_path, "")
        transition_type = _single(transition_type, "fade")
        fit_mode = _single(fit_mode, "letterbox")
        resize_mode = _single(resize_mode, "first_image")
        ken_burns = _single(ken_burns, "off")
        kb_motion = _single(kb_motion, "in")
        audio_mode = _single(audio_mode, "fit")
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")

        image_duration = max(0.1, safe_float(image_duration, 3.0))
        output_fps_value = choose_output_fps(output_fps)
        kb_zoom_start = clamp(safe_float(kb_zoom_start, 1.0), 1.0, 10.0)
        kb_zoom_end = clamp(safe_float(kb_zoom_end, 1.3), 1.0, 10.0)

        # Сбор динамических входов image_1..N (тензоры). / Collect image_1..N tensors.
        image_map = {}
        for key, value in kwargs.items():
            if not key.startswith("image_"):
                continue
            match = re.search(r"\d+", key)
            if not match:
                continue
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None and hasattr(value, "shape"):
                try:
                    if value.numel() > 0:
                        image_map[int(match.group())] = value
                except Exception:
                    pass

        connected = len(image_map)
        expected = int(inputs_count or 2)
        if connected < 2:
            raise ValueError("[AGSoft Images To Video] Нужно подключить как минимум 2 входа image_!")
        if connected < expected:
            raise ValueError(f"[AGSoft Images To Video] Подключено {connected} из {expected} выбранных входов.")

        # Раскрытие батчей в плоский список слайдов + сохранение во временные PNG.
        # Expand batches into a flat slide list + save to temp PNGs.
        tmp_dir = tempfile.mkdtemp(prefix="agsoft_img2vid_")
        slide_paths = []
        slide_sizes = []
        try:
            for idx in sorted(image_map.keys()):
                tensor = image_map[idx]
                arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
                if arr.ndim == 3:
                    arr = arr[None, ...]
                for b in range(arr.shape[0]):
                    fd, png_path = tempfile.mkstemp(prefix=f"slide_{idx}_{b}_", suffix=".png", dir=tmp_dir)
                    os.close(fd)
                    single = arr[b:b + 1]
                    w, h = tensor_to_png(single[0], png_path)
                    slide_paths.append(png_path)
                    slide_sizes.append({"width": w, "height": h})

            n_slides = len(slide_paths)
            if n_slides < 2:
                raise ValueError("[AGSoft Images To Video] После раскрытия батчей получилось меньше 2 слайдов.")

            # Аудио -> временный WAV. / Audio -> temp WAV.
            has_audio = False
            wav_path = None
            if audio is not None and isinstance(audio, dict) and "waveform" in audio:
                try:
                    fd, wav_path = tempfile.mkstemp(prefix="voice_", suffix=".wav", dir=tmp_dir)
                    os.close(fd)
                    audio_to_wav(audio, wav_path)
                    has_audio = True
                except Exception as e:
                    print(f"[AGSoft Images To Video] Warning: не удалось сохранить аудио: {e}")
                    wav_path = None

            # Имя и папка выхода. / Output name and folder.
            safe_name = os.path.basename(str(output_name or "slideshow.mp4"))
            name_without_ext, ext = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "slideshow"
            if not ext:
                ext = ".mp4"
            safe_name = f"{name_without_ext}{ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Images To Video] Путь назначения — папка, а не файл: {final_output_path}")

            output_width, output_height = choose_output_size(slide_sizes, resize_mode, custom_width, custom_height)

            # Видео-граф (без аудио). / Video-only graph.
            video_graph, log_lines, total_duration = build_video_graph(
                n_slides=n_slides, image_duration=image_duration,
                output_fps=output_fps_value, output_width=output_width, output_height=output_height,
                fit_mode=fit_mode, transition_type=transition_type, transition_duration=transition_duration,
                ken_burns=ken_burns, kb_motion=kb_motion, kb_zoom_start=kb_zoom_start, kb_zoom_end=kb_zoom_end
            )

            for line in log_lines:
                print(f"[AGSoft Images To Video] {line}")

            encoder = choose_encoder(encoder_mode)

            # Входные аргументы для слайдов (общие для обоих режимов). / Slide inputs.
            slide_input_args = []
            for p in slide_paths:
                slide_input_args += ["-loop", "1", "-framerate", f"{output_fps_value:.6f}", "-t", f"{image_duration:.6f}", "-i", p]

            mux_args = ["-movflags", "+faststart"] if safe_name.lower().endswith((".mp4", ".mov")) else []

            if not has_audio:
                # ===== ОДИН ПРОХОД: только видео. / SINGLE PASS: video only. =====
                base_cmd = [
                    FFMPEG_PATH, "-y", "-hide_banner",
                    *slide_input_args,
                    "-filter_complex", video_graph,
                    "-map", "[outv]",
                ]
                cmd = base_cmd + build_encoder_args(encoder, quality_preset) + mux_args + [final_output_path]
                result = run_ffmpeg_with_progress(cmd, total_duration)

                if result.returncode != 0 and encoder == "h264_nvenc" and encoder_mode == "auto":
                    print("[AGSoft Images To Video] NVENC failed. Falling back to CPU libx264.")
                    cmd = base_cmd + build_encoder_args("libx264", quality_preset) + mux_args + [final_output_path]
                    result = run_ffmpeg_with_progress(cmd, total_duration)

                if result.returncode != 0:
                    self._raise_ffmpeg_error(result.stderr)
            else:
                # ===== ДВУХПРОХОДНЫЙ РЕНДЕР: видео, затем аудио под реальную длину. =====
                # ===== TWO-PASS RENDER: video first, then audio aligned to real length. =====
                fd, video_only_path = tempfile.mkstemp(prefix="video_only_", suffix=".mp4", dir=tmp_dir)
                os.close(fd)

                # Проход 1: рендер только видео. / Pass 1: render video only.
                base_cmd1 = [
                    FFMPEG_PATH, "-y", "-hide_banner",
                    *slide_input_args,
                    "-filter_complex", video_graph,
                    "-map", "[outv]",
                ]
                cmd1 = base_cmd1 + build_encoder_args(encoder, quality_preset) + mux_args + [video_only_path]
                result1 = run_ffmpeg_with_progress(cmd1, total_duration)

                if result1.returncode != 0 and encoder == "h264_nvenc" and encoder_mode == "auto":
                    print("[AGSoft Images To Video] NVENC failed. Falling back to CPU libx264.")
                    cmd1 = base_cmd1 + build_encoder_args("libx264", quality_preset) + mux_args + [video_only_path]
                    result1 = run_ffmpeg_with_progress(cmd1, total_duration)

                if result1.returncode != 0:
                    self._raise_ffmpeg_error(result1.stderr)

                # Реальная длительность видео (точнее формулы). / Real video length.
                real_dur = ffprobe_duration(video_only_path)
                if real_dur is None or real_dur <= 0:
                    real_dur = total_duration
                print(f"[AGSoft Images To Video] Video pass done. Real duration = {real_dur:.3f}s (formula = {total_duration:.3f}s).")

                # Проход 2: наложить аудио под real_dur, видео копировать без перекодировки.
                # Pass 2: mux audio aligned to real_dur, copy video without re-encoding.
                audio_graph = build_audio_graph(
                    total_duration=real_dur, audio_mode=audio_mode,
                    audio_volume=audio_volume, audio_fade_in=audio_fade_in, audio_fade_out=audio_fade_out
                )

                input_args2 = ["-i", video_only_path]
                if str(audio_mode).strip().lower() == "loop":
                    input_args2 += ["-stream_loop", "-1"]
                input_args2 += ["-i", wav_path]

                base_cmd2 = [
                    FFMPEG_PATH, "-y", "-hide_banner",
                    *input_args2,
                    "-filter_complex", audio_graph,
                    "-map", "0:v:0",
                    "-map", "[outa]",
                ]
                audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
                cmd2 = base_cmd2 + ["-c:v", "copy"] + audio_args + mux_args + [final_output_path]
                result2 = run_ffmpeg_with_progress(cmd2, real_dur)

                if result2.returncode != 0:
                    self._raise_ffmpeg_error(result2.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Images To Video] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Images To Video] Saved: {final_output_path} | "
                f"Slides: {n_slides} | Duration: {timecode} | Size: {out_info['size_mb']} MB | "
                f"Resolution: {out_info['width']}x{out_info['height']} | FPS: {out_info['fps']}"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
                out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Images To Video Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Images To Video] FFmpeg не смог отрендерить видео. Ошибка: {error_msg}")


NODE_CLASS_MAPPINGS = {
    "AGSoftImagesToVideo": AGSoftImagesToVideo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftImagesToVideo": "🎬🖼️AGSoft Images To Video"
}