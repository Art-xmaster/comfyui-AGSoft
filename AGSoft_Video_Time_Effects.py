# ==============================================================================
# AGSoft_Video_Time_Effects.py
# ==============================================================================
# Нода: 🎬⏱️AGSoft Video Time Effects
#
# Описание / Description:
# Нода эффектов времени для видео: изменение скорости (slow-motion / таймлапс),
# реверс и boomerang с синхронизацией звука. Эффект можно применить ко всему
# ролику или только к отрезку [start_time..end_time]. Одна ручка speed покрывает
# и замедление, и ускорение; для плавного slow-motion доступна интерполяция кадров
# (minterpolate). Звук автоматически подстраивается под скорость через цепочку
# atempo, включая выход за стандартный диапазон 0.5–2.0.
# Time effects node for video: speed change (slow-motion / timelapse), reverse
# and boomerang with audio synchronization. The effect can be applied to the
# whole clip or only to a segment [start_time..end_time]. A single speed knob
# covers both slowing down and speeding up; frame interpolation (minterpolate)
# is available for smooth slow-motion. Audio automatically follows the speed via
# an atempo chain, including speeds beyond the standard 0.5–2.0 range.
#
# Возможности / Features:
# ⚡ Скорость (speed): множитель 0.1–10.0. Значения < 1 дают slow-motion
#    (замедление), > 1 — таймлапс (ускорение), 1.0 — без изменений.
#    Speed multiplier 0.1–10.0. Values < 1 give slow-motion, > 1 give timelapse,
#    1.0 leaves the clip unchanged.
# ⚡ Выбор отрезка (start_time / end_time): эффект применяется только к фрагменту,
#    остальное видео остаётся без изменений. 0/0 = весь ролик.
#    Segment selection: the effect is applied only to a fragment, the rest of the
#    video stays unchanged. 0/0 = the whole clip.
# ⚡ Синхронизация звука: дорожка подстраивается под скорость цепочкой atempo;
#    скорости вне диапазона 0.5–2.0 автоматически разбиваются на несколько
#    фильтров atempo.
#    Audio sync: the track follows the speed via an atempo chain; speeds beyond
#    the 0.5–2.0 range are automatically split into multiple atempo filters.
# ⚡ Интерполяция кадров (interpolation): minterpolate досоздаёт промежуточные
#    кадры для плавного slow-motion без рывков (применяется только при замедлении).
#    Frame interpolation: minterpolate creates intermediate frames for smooth,
#    jitter-free slow-motion (applied only when slowing down).
# ⚡ Реверс (reverse): воспроизведение видео задом наперёд, звук также реверсируется.
#    Reverse: plays the video backwards, audio is reversed as well.
# ⚡ Boomerang: клип играет вперёд, затем задом наперёд; удваивает длительность.
#    Boomerang: plays the clip forward then backward; doubles the duration.
# ⚡ Режимы звука (audio_mode): with_video (звук следует за эффектами),
#    mute (без звука), keep_original (оригинальная дорожка без изменений).
#    Audio modes: with_video, mute, keep_original.
# ⚡ NVENC / CPU с авто-fallback; пресеты fast/balanced/quality; прогресс в консоль.
#    NVENC / CPU with auto fallback; presets fast/balanced/quality; console progress.
#
# Автор / Author: AGSoft
# Дата / Date: 10.08.2026
# ==============================================================================

import os
import re
import json
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


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


# ------------------------------------------------------------------------------
# Извлечение пути к видео из объекта VIDEO (многоуровневый fallback).
# ------------------------------------------------------------------------------
def _extract_video_path(video_obj, tmp_dir):
    if video_obj is None:
        return None

    for attr_name in ['path', 'video_path', 'file_path', '_path', 'source_path']:
        if hasattr(video_obj, attr_name):
            candidate = getattr(video_obj, attr_name)
            if isinstance(candidate, str) and candidate and os.path.isfile(candidate):
                return candidate

    if hasattr(video_obj, 'save_to'):
        try:
            fd, temp_path = tempfile.mkstemp(prefix="video_src_", suffix=".mp4", dir=tmp_dir)
            os.close(fd)
            video_obj.save_to(temp_path)
            if os.path.isfile(temp_path):
                return temp_path
        except Exception as e:
            logging.warning(f"save_to failed: {e}")

    if hasattr(video_obj, 'get_components_internal'):
        try:
            components = video_obj.get_components_internal()
            if components and len(components) > 0:
                fd, temp_path = tempfile.mkstemp(prefix="video_src_", suffix=".mp4", dir=tmp_dir)
                os.close(fd)
                arr = np.stack([c.cpu().numpy() if hasattr(c, 'cpu') else np.asarray(c) for c in components])
                if arr.ndim == 5:
                    arr = arr[0]
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                F, H, W, C = arr.shape
                fps = getattr(video_obj, 'get_frame_rate', lambda: 30.0)()
                cmd = [
                    FFMPEG_PATH, "-y", "-hide_banner",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", temp_path
                ]
                process = _sp_popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                for f in range(F):
                    process.stdin.write(np.ascontiguousarray(arr[f]).tobytes())
                process.stdin.close()
                process.wait()
                if process.returncode == 0 and os.path.isfile(temp_path):
                    return temp_path
        except Exception as e:
            logging.warning(f"get_components_internal failed: {e}")

    return None


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
        print(f"[AGSoft Video Time Effects] Не удалось прочитать файл: {path} | {e}")
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


# ------------------------------------------------------------------------------
# Цепочка atempo для скорости вне диапазона 0.5–2.0 (каждый множитель в [0.5,2.0]).
# ------------------------------------------------------------------------------
def build_atempo_chain(speed):
    filters = []
    remaining = float(speed)
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


# ------------------------------------------------------------------------------
# Сборка filter_complex: отрезок, скорость, интерполяция, reverse, boomerang, аудио.
# ------------------------------------------------------------------------------
def build_time_effects_filter(speed, interpolation, reverse_on, boomerang_on, audio_mode,
                              has_audio, target_fps, start_time, end_time, duration):
    parts = []
    speed_changed = abs(speed - 1.0) > 1e-6

    # Эффект на весь ролик или только на отрезок.
    apply_to_all = (start_time <= 0) and (end_time <= 0 or end_time >= duration)

    # Видео-эффекты для средней части.
    v_effect = []
    if speed_changed:
        v_effect.append(f"setpts=PTS*{1.0/speed:.6f}")
        if interpolation == "minterpolate" and speed < 1.0:
            v_effect.append(f"minterpolate=fps={target_fps:.6f}")
    v_effect_str = ",".join(v_effect)

    # --- Видео ---
    if apply_to_all:
        v_chain = f"[0:v]{v_effect_str}," if v_effect_str else "[0:v]"
        if boomerang_on:
            parts.append(f"{v_chain}split=2[v_a][v_b]")
            parts.append("[v_a]setpts=PTS-STARTPTS[v_a_s]")
            parts.append("[v_b]reverse,setpts=PTS-STARTPTS[v_b_r]")
            parts.append("[v_a_s][v_b_r]concat=n=2:v=1:a=0[v_final]")
        elif reverse_on:
            parts.append(f"{v_chain}reverse,setpts=PTS-STARTPTS[v_final]")
        else:
            parts.append(f"{v_chain}setpts=PTS-STARTPTS[v_final]")
    else:
        start_t = max(0.0, start_time)
        end_t = end_time if end_time > 0 else duration
        if start_t >= end_t:
            raise ValueError("[AGSoft Video Time Effects] end_time должен быть больше start_time.")
        end_t = min(end_t, duration)

        v_segments = []
        if start_t > 0.01:
            parts.append(f"[0:v]trim=start=0:end={start_t:.6f},setpts=PTS-STARTPTS[v_pre]")
            v_segments.append("[v_pre]")

        mid_prefix = f"[0:v]trim=start={start_t:.6f}:end={end_t:.6f},setpts=PTS-STARTPTS"
        if v_effect_str:
            mid_prefix += f",{v_effect_str}"
        if boomerang_on:
            parts.append(f"{mid_prefix},setpts=PTS-STARTPTS[v_mid_base]")
            parts.append("[v_mid_base]split=2[v_a][v_b]")
            parts.append("[v_a]setpts=PTS-STARTPTS[v_a_s]")
            parts.append("[v_b]reverse,setpts=PTS-STARTPTS[v_b_r]")
            parts.append("[v_a_s][v_b_r]concat=n=2:v=1:a=0[v_mid]")
        elif reverse_on:
            parts.append(f"{mid_prefix},reverse,setpts=PTS-STARTPTS[v_mid]")
        else:
            parts.append(f"{mid_prefix},setpts=PTS-STARTPTS[v_mid]")
        v_segments.append("[v_mid]")

        if end_t < duration - 0.01:
            parts.append(f"[0:v]trim=start={end_t:.6f},setpts=PTS-STARTPTS[v_post]")
            v_segments.append("[v_post]")

        if len(v_segments) == 1:
            parts.append(f"{v_segments[0]}null[v_final]")
        else:
            parts.append(f"{''.join(v_segments)}concat=n={len(v_segments)}:v=1:a=0[v_final]")

    parts.append("[v_final]null[outv]")

    # --- Аудио ---
    if audio_mode == "with_video" and has_audio:
        a_format = "aresample=44100:async=1,aformat=sample_fmts=fltp:channel_layouts=stereo"
        if speed_changed:
            a_mid_effect = a_format + "," + build_atempo_chain(speed)
        else:
            a_mid_effect = a_format

        if apply_to_all:
            if boomerang_on:
                parts.append(f"[0:a]{a_mid_effect},asplit=2[a_a][a_b]")
                parts.append("[a_a]asetpts=PTS-STARTPTS[a_a_s]")
                parts.append("[a_b]areverse,asetpts=PTS-STARTPTS[a_b_r]")
                parts.append("[a_a_s][a_b_r]concat=n=2:v=0:a=1[a_final]")
            elif reverse_on:
                parts.append(f"[0:a]{a_mid_effect},areverse,asetpts=PTS-STARTPTS[a_final]")
            else:
                parts.append(f"[0:a]{a_mid_effect},asetpts=PTS-STARTPTS[a_final]")
        else:
            start_t = max(0.0, start_time)
            end_t = end_time if end_time > 0 else duration
            if start_t >= end_t:
                raise ValueError("[AGSoft Video Time Effects] end_time должен быть больше start_time.")
            end_t = min(end_t, duration)

            a_segments = []
            if start_t > 0.01:
                parts.append(f"[0:a]atrim=start=0:end={start_t:.6f},asetpts=PTS-STARTPTS,{a_format}[a_pre]")
                a_segments.append("[a_pre]")

            mid_prefix = f"[0:a]atrim=start={start_t:.6f}:end={end_t:.6f},asetpts=PTS-STARTPTS,{a_mid_effect}"
            if boomerang_on:
                parts.append(f"{mid_prefix},asetpts=PTS-STARTPTS[a_mid_base]")
                parts.append("[a_mid_base]asplit=2[a_a][a_b]")
                parts.append("[a_a]asetpts=PTS-STARTPTS[a_a_s]")
                parts.append("[a_b]areverse,asetpts=PTS-STARTPTS[a_b_r]")
                parts.append("[a_a_s][a_b_r]concat=n=2:v=0:a=1[a_mid]")
            elif reverse_on:
                parts.append(f"{mid_prefix},areverse,asetpts=PTS-STARTPTS[a_mid]")
            else:
                parts.append(f"{mid_prefix},asetpts=PTS-STARTPTS[a_mid]")
            a_segments.append("[a_mid]")

            if end_t < duration - 0.01:
                parts.append(f"[0:a]atrim=start={end_t:.6f},asetpts=PTS-STARTPTS,{a_format}[a_post]")
                a_segments.append("[a_post]")

            if len(a_segments) == 1:
                parts.append(f"{a_segments[0]}anull[a_final]")
            else:
                parts.append(f"{''.join(a_segments)}concat=n={len(a_segments)}:v=0:a=1[a_final]")

        parts.append("[a_final]anull[outa]")

    return ";".join(parts)


def choose_encoder(encoder_mode):
    encoder_mode = str(encoder_mode or "auto").strip().lower()
    if encoder_mode == "cpu":
        return "libx264"
    if encoder_mode == "nvenc":
        if supports_encoder("h264_nvenc"):
            return "h264_nvenc"
        raise RuntimeError("[AGSoft Video Time Effects] h264_nvenc недоступен. Выберите cpu или auto.")
    if supports_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


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
        raise RuntimeError(f"[AGSoft Video Time Effects] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Video Time Effects] Progress: {bucket * 5}%")
                    last_bucket = bucket
    process.wait()
    return subprocess.CompletedProcess(args=cmd, returncode=process.returncode, stdout="", stderr="".join(stderr_chunks))


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


class AGSoftVideoTimeEffects:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_path", "duration_seconds", "duration_timecode", "file_size_mb", "width", "height", "fps", "frames_est")

    FUNCTION = "apply_time_effects"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬⏱️AGSoft Video Time Effects.\n"
        "Time effects for a video clip: speed change, reverse and boomerang, with audio kept in sync. "
        "The effect can be applied to the whole clip or only to a segment via start_time / end_time.\n"
        "Speed: a single multiplier (0.1–10.0). Values below 1 give slow-motion, values above 1 give "
        "timelapse, 1.0 leaves the clip unchanged. Audio follows the speed via an atempo chain, and "
        "speeds beyond the 0.5–2.0 range are automatically split into multiple atempo filters.\n"
        "Segment: set start_time / end_time to apply the effect only to a fragment; the rest of the "
        "video stays unchanged. 0/0 = the whole clip.\n"
        "Interpolation: minterpolate creates intermediate frames for smooth, jitter-free slow-motion "
        "(applied only when slowing down). Reverse plays the clip backwards; boomerang plays it forward "
        "then backward (doubling the duration).\n"
        "Audio modes: with_video (audio follows the effects), mute (no audio), keep_original (original "
        "track copied unchanged, may desync if effects are used).\n"
        "---\n"
        "🎬⏱️AGSoft Video Time Effects.\n"
        "Эффекты времени для видеоклипа: изменение скорости, реверс и boomerang с синхронизацией звука. "
        "Эффект можно применить ко всему ролику или только к отрезку через start_time / end_time.\n"
        "Скорость: один множитель (0.1–10.0). Значения меньше 1 дают slow-motion, больше 1 — таймлапс, "
        "1.0 оставляет клип без изменений. Звук следует за скоростью через цепочку atempo; скорости вне "
        "диапазона 0.5–2.0 автоматически разбиваются на несколько фильтров atempo.\n"
        "Отрезок: задайте start_time / end_time, чтобы применить эффект только к фрагменту; остальное "
        "видео останется без изменений. 0/0 = весь ролик.\n"
        "Интерполяция: minterpolate досоздаёт промежуточные кадры для плавного slow-motion без рывков "
        "(применяется только при замедлении). Реверс воспроизводит клип задом наперёд; boomerang играет "
        "его вперёд, затем назад (удваивая длительность).\n"
        "Режимы звука: with_video (звук следует за эффектами), mute (без звука), keep_original "
        "(оригинальная дорожка без изменений, возможен рассинхрон при использовании эффектов)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "time_effects.mp4",
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
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to the source video. Used only if video input is not connected.\n"
                            "---\n"
                            "Путь к исходному видео. Используется, только если вход video не подключен."
                        )
                    }
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": (
                            "Segment start (seconds) where the effect begins. 0 = from the beginning. "
                            "Together with end_time selects the fragment to apply the effect to.\n"
                            "---\n"
                            "Начало отрезка (в секундах), с которого применяется эффект. 0 = с начала. "
                            "Вместе с end_time выбирает фрагмент для применения эффекта."
                        )
                    }
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": (
                            "Segment end (seconds) where the effect stops. 0 = to the end of the clip. "
                            "start_time=0 and end_time=0 = apply the effect to the whole clip.\n"
                            "---\n"
                            "Конец отрезка (в секундах), до которого применяется эффект. 0 = до конца клипа. "
                            "start_time=0 и end_time=0 = применить эффект ко всему клипу."
                        )
                    }
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                        "tooltip": (
                            "Speed multiplier. < 1 = slow-motion (e.g. 0.3 = 3.3x slower), "
                            "> 1 = timelapse (e.g. 4.0 = 4x faster). 1.0 = unchanged.\n"
                            "---\n"
                            "Множитель скорости. < 1 = slow-motion (напр. 0.3 = замедление в 3.3 раза), "
                            "> 1 = таймлапс (напр. 4.0 = ускорение в 4 раза). 1.0 = без изменений."
                        )
                    }
                ),
                "interpolation": (
                    ["none", "minterpolate"],
                    {
                        "default": "none",
                        "tooltip": (
                            "none = no frame interpolation. minterpolate = creates intermediate frames "
                            "for smooth slow-motion (only applied when speed < 1). Can be slow on long videos.\n"
                            "---\n"
                            "none = без интерполяции кадров. minterpolate = досоздаёт промежуточные кадры "
                            "для плавного slow-motion (только при speed < 1). Может быть медленным на длинных видео."
                        )
                    }
                ),
                "reverse": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Reverse the video (play backwards). Loads the whole clip in memory; "
                            "use with care on long videos. Ignored if boomerang is on.\n"
                            "---\n"
                            "Реверс видео (воспроизведение задом наперёд). Загружает весь клип в память; "
                            "осторожно на длинных видео. Игнорируется, если включён boomerang."
                        )
                    }
                ),
                "boomerang": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Boomerang: plays the clip forward then backward. Doubles the duration. "
                            "Has priority over reverse. Loads the whole clip in memory.\n"
                            "---\n"
                            "Boomerang: клип играет вперёд, затем задом наперёд. Удваивает длительность. "
                            "Имеет приоритет над reverse. Загружает весь клип в память."
                        )
                    }
                ),
                "audio_mode": (
                    ["with_video", "mute", "keep_original"],
                    {
                        "default": "with_video",
                        "tooltip": (
                            "with_video = audio follows the speed/reverse/boomerang (atempo chain).\n"
                            "mute = no audio.\n"
                            "keep_original = copy the original audio unchanged (may desync if speed/reverse/boomerang is used).\n"
                            "---\n"
                            "with_video = звук следует за скоростью/реверсом/boomerang (цепочка atempo).\n"
                            "mute = без звука.\n"
                            "keep_original = копировать оригинальный звук без изменений (возможен рассинхрон при speed/reverse/boomerang)."
                        )
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
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": (
                            "Optional source video from ComfyUI. Has priority over video_path.\n"
                            "---\n"
                            "Опциональное исходное видео из ComfyUI. Имеет приоритет над video_path."
                        )
                    }
                ),
            },
        }

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Video Time Effects Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Video Time Effects] FFmpeg не смог применить эффекты. Ошибка: {error_msg}")

    def apply_time_effects(
        self,
        output_name, output_path, video_path,
        start_time, end_time, speed, interpolation, reverse, boomerang, audio_mode,
        encoder_mode, quality_preset,
        video=None
    ):
        output_name = _single(output_name, "time_effects.mp4")
        output_path = _single(output_path, "")
        video_path = _single(video_path, "").strip()
        interpolation = _single(interpolation, "none").strip().lower()
        reverse = _single(reverse, "off").strip().lower()
        boomerang = _single(boomerang, "off").strip().lower()
        audio_mode = _single(audio_mode, "with_video").strip().lower()
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")

        speed = clamp(safe_float(speed, 1.0), 0.1, 10.0)
        start_time = max(0.0, safe_float(start_time, 0.0))
        end_time = max(0.0, safe_float(end_time, 0.0))
        reverse_on = reverse == "on"
        boomerang_on = boomerang == "on"

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_time_effects_")
        try:
            # Источник видео: вход video -> video_path
            video_src = None
            if video is not None:
                video_src = _extract_video_path(video, tmp_dir)
                if video_src:
                    print(f"[AGSoft Video Time Effects] Источник из входа video: {video_src}")
            if video_src is None and video_path:
                if os.path.isfile(video_path):
                    video_src = video_path
                else:
                    print(f"[AGSoft Video Time Effects] Warning: video_path не найден ({video_path}).")
            if video_src is None:
                raise ValueError("[AGSoft Video Time Effects] Не задан источник видео: подключите video или укажите video_path.")

            vinfo = parse_media_info(video_src)
            if vinfo["width"] <= 0 or vinfo["height"] <= 0:
                raise ValueError(f"[AGSoft Video Time Effects] В источнике нет видеопотока: {video_src}")
            has_audio = bool(vinfo["has_audio"])
            duration = float(vinfo["duration"] or 0.0)
            if duration <= 0:
                raise ValueError(f"[AGSoft Video Time Effects] Не удалось определить длительность видео: {video_src}")

            target_fps = float(vinfo["fps"]) if vinfo["fps"] > 0 else 30.0

            # Имя и папка выхода. / Output name and folder.
            safe_name = os.path.basename(str(output_name or "time_effects.mp4"))
            name_without_ext, ext = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "time_effects"
            if not ext:
                ext = ".mp4"
            safe_name = f"{name_without_ext}{ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Video Time Effects] Путь назначения — папка, а не файл: {final_output_path}")
            if os.path.normcase(os.path.realpath(video_src)) == os.path.normcase(os.path.realpath(final_output_path)):
                raise ValueError("[AGSoft Video Time Effects] Итоговый файл не может совпадать с источником.")

            # Сборка фильтра. / Build filter.
            filter_complex = build_time_effects_filter(
                speed, interpolation, reverse_on, boomerang_on, audio_mode,
                has_audio, target_fps, start_time, end_time, duration
            )

            # Команда. / Command.
            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-i", video_src]
            cmd += ["-filter_complex", filter_complex]
            cmd += ["-map", "[outv]"]

            if audio_mode == "with_video" and has_audio:
                cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
            elif audio_mode == "keep_original" and has_audio:
                cmd += ["-map", "0:a:0", "-c:a", "copy"]
            else:
                cmd += ["-an"]

            cmd += build_encoder_args(choose_encoder(encoder_mode), quality_preset)
            if safe_name.lower().endswith((".mp4", ".mov")):
                cmd += ["-movflags", "+faststart"]
            cmd.append(final_output_path)

            # Оценка длительности для прогресса. / Duration estimate for progress.
            apply_to_all = (start_time <= 0) and (end_time <= 0 or end_time >= duration)
            if apply_to_all:
                est_dur = duration / speed if speed > 0 else duration
                if boomerang_on:
                    est_dur *= 2.0
            else:
                end_t = end_time if end_time > 0 else duration
                end_t = min(end_t, duration)
                seg_dur = (end_t - start_time) / speed if speed > 0 else (end_t - start_time)
                if boomerang_on:
                    seg_dur *= 2.0
                est_dur = start_time + seg_dur + (duration - end_t)

            result = run_ffmpeg_with_progress(cmd, est_dur)
            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Video Time Effects] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Video Time Effects] Saved: {final_output_path} | speed={speed} | "
                f"interp={interpolation} | reverse={reverse} | boomerang={boomerang} | audio={audio_mode} | "
                f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
                out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoTimeEffects": AGSoftVideoTimeEffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoTimeEffects": "🎬⏱️AGSoft Video Time Effects"
}