# ==============================================================================
# AGSoft_Video_Speed_Ramp.py
# ==============================================================================
# Нода: 🎬⏱️AGSoft Video Speed Ramp
#
# Описание / Description:
# Плавное изменение скорости внутри одного клипа (разгон или замедление), а не
# ступенькой. Скорость меняется от speed_start до speed_end за ramp_time секунд
# (линейная кривая через setpts), после чего остаётся постоянной speed_end.
# Это даёт эффект плавного разгона в таймлапс или плавного входа в slow-motion.
# Smoothly changes the speed within a single clip (acceleration or deceleration),
# not as a step. The speed ramps from speed_start to speed_end over ramp_time
# seconds (linear curve via setpts), then stays constant at speed_end. This gives
# a smooth acceleration into timelapse or a smooth entry into slow-motion.
#
# Возможности / Features:
# ⚡ Начальная скорость (speed_start): скорость в начале ramp. < 1 = замедление,
#    > 1 = ускорение.
#    Start speed: speed at the beginning of the ramp. < 1 = slow-motion,
#    > 1 = timelapse.
# ⚡ Конечная скорость (speed_end): скорость, которой клип достигает через
#    ramp_time секунд.
#    End speed: the speed the clip reaches after ramp_time seconds.
# ⚡ Время разгона (ramp_time): секунды для изменения скорости от speed_start до
#    speed_end. 0 = ramp на весь клип. После ramp_time скорость постоянна.
#    Ramp time: seconds to change speed from speed_start to speed_end. 0 = ramp
#    across the whole clip. After ramp_time the speed is constant.
# ⚡ Режимы звука (audio_mode): keep_original (копировать оригинал, возможен
#    рассинхрон с переменным ramp), mute (без звука). Точная синхронизация аудио
#    с переменной скоростью невозможна.
#    Audio modes: keep_original (copy the original, possible desync with a variable
#    ramp), mute (no audio). Perfect audio sync with a variable speed is not possible.
# ⚡ Входы: video (сокет VIDEO, приоритет) или video_path (путь к файлу).
#    Inputs: video (VIDEO socket, priority) or video_path (file path).
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
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
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
        result = subprocess.run([FFMPEG_PATH, "-hide_banner", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        data = result.stderr or ""
    except Exception as e:
        print(f"[AGSoft Video Speed Ramp] Не удалось прочитать файл: {path} | {e}")
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
# ВАЖНО: в setpts переменная PTS — это единицы timebase (большие числа), а T —
# время кадра в секундах. Все расчёты скорости ведутся в секундах через T, а
# итог делится на TB, чтобы получить новые PTS в единицах timebase. Иначе
# сравнение if(lt(PTS, ramp_time)) всегда ложно и ramp не работает.
# ------------------------------------------------------------------------------
def build_speed_ramp_setpts_expr(speed_start, speed_end, ramp_time, duration):
    """Возвращает setpts-выражение для линейного speed ramp (в единицах PTS)."""
    s0 = float(speed_start)
    s1 = float(speed_end)

    if abs(s1 - s0) < 0.01:
        return f"PTS/{s0:.6f}"

    T_ramp = ramp_time if ramp_time > 0 else duration
    if T_ramp <= 0:
        T_ramp = duration
    ramp_to_whole = T_ramp >= duration

    coeff = T_ramp / (s1 - s0)
    # Новое время (в секундах) как функция текущего времени T (секунды):
    # f(T) = coeff * ln((s0 + (s1-s0)*T/T_ramp)/s0)
    ramp_sec = f"({coeff:.6f})*log(({s0:.6f}+({s1-s0:.6f})*T/{T_ramp:.6f})/{s0:.6f})"

    if ramp_to_whole:
        return f"({ramp_sec})/TB"

    c_val = coeff * np.log(s1 / s0)
    # После ramp_time скорость постоянна s1: f(T) = c_val + (T - T_ramp)/s1
    phase2_sec = f"({c_val:.6f})+(T-{T_ramp:.6f})/{s1:.6f}"

    return f"if(lt(T,{T_ramp:.6f}),{ramp_sec},{phase2_sec})/TB"


def build_speed_ramp_filter(speed_start, speed_end, ramp_time, duration, audio_mode, has_audio):
    parts = []
    ramp_expr = build_speed_ramp_setpts_expr(speed_start, speed_end, ramp_time, duration)
    # Оборачиваем в одинарные кавычки: запятые внутри if()/log() тогда не ломают
    # парсинг filter_complex.
    parts.append(f"[0:v]setpts='{ramp_expr}',setpts=PTS-STARTPTS[v_final]")
    parts.append("[v_final]null[outv]")

    if audio_mode == "keep_original" and has_audio:
        parts.append("[0:a]anull[outa]")

    return ";".join(parts)


def choose_encoder(encoder_mode):
    encoder_mode = str(encoder_mode or "auto").strip().lower()
    if encoder_mode == "cpu":
        return "libx264"
    if encoder_mode == "nvenc":
        if supports_encoder("h264_nvenc"):
            return "h264_nvenc"
        raise RuntimeError("[AGSoft Video Speed Ramp] h264_nvenc недоступен. Выберите cpu или auto.")
    if supports_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def supports_encoder(encoder):
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    ok = False
    try:
        result = subprocess.run([FFMPEG_PATH, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
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
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Video Speed Ramp] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Video Speed Ramp] Progress: {bucket * 5}%")
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


class AGSoftVideoSpeedRamp:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_path", "duration_seconds", "duration_timecode", "file_size_mb", "width", "height", "fps", "frames_est")

    FUNCTION = "apply_speed_ramp"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬⏱️AGSoft Video Speed Ramp.\n"
        "Smoothly changes the speed within a single clip (acceleration or deceleration), not as "
        "a step. The speed ramps from speed_start to speed_end over ramp_time seconds, then stays "
        "constant at speed_end.\n"
        "Speeds: speed_start is the speed at the beginning of the ramp, speed_end is the speed "
        "reached after ramp_time. Both < 1 give slow-motion, > 1 give timelapse.\n"
        "Ramp time: seconds to change speed from speed_start to speed_end. 0 = ramp across the "
        "whole clip. After ramp_time the speed stays constant at speed_end.\n"
        "Audio: a variable ramp cannot be perfectly synced with audio. keep_original copies the "
        "original track (possible desync), mute removes audio.\n"
        "Inputs: video (VIDEO socket, priority) or video_path (file path).\n"
        "Encoding: NVENC / CPU with auto fallback; presets fast / balanced / quality.\n"
        "---\n"
        "🎬⏱️AGSoft Video Speed Ramp.\n"
        "Плавно изменяет скорость внутри одного клипа (разгон или замедление), а не ступенькой. "
        "Скорость меняется от speed_start до speed_end за ramp_time секунд, после чего остаётся "
        "постоянной speed_end.\n"
        "Скорости: speed_start — скорость в начале ramp, speed_end — скорость, достигаемая через "
        "ramp_time. Значения < 1 дают замедление, > 1 — ускорение.\n"
        "Время разгона: секунды для изменения скорости от speed_start до speed_end. 0 = ramp на "
        "весь клип. После ramp_time скорость остаётся постоянной speed_end.\n"
        "Аудио: переменный ramp не может быть точно синхронизирован со звуком. keep_original "
        "копирует оригинальную дорожку (возможен рассинхрон), mute убирает звук.\n"
        "Входы: video (сокет VIDEO, приоритет) или video_path (путь к файлу).\n"
        "Кодирование: NVENC / CPU с авто-fallback; пресеты fast / balanced / quality."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "speed_ramp.mp4",
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
                "speed_start": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                        "tooltip": (
                            "Speed at the beginning of the ramp. < 1 = slow-motion, > 1 = timelapse.\n"
                            "---\n"
                            "Скорость в начале ramp. < 1 = замедление, > 1 = ускорение."
                        )
                    }
                ),
                "speed_end": (
                    "FLOAT",
                    {
                        "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.05,
                        "tooltip": (
                            "Speed at the end of the ramp. The clip reaches this speed after ramp_time.\n"
                            "---\n"
                            "Скорость в конце ramp. Клип достигает этой скорости через ramp_time."
                        )
                    }
                ),
                "ramp_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": (
                            "Seconds to ramp from speed_start to speed_end. 0 = ramp across the whole clip. "
                            "After ramp_time the speed stays constant at speed_end.\n"
                            "---\n"
                            "Секунды для разгона от speed_start до speed_end. 0 = ramp на весь клип. "
                            "После ramp_time скорость остаётся постоянной speed_end."
                        )
                    }
                ),
                "audio_mode": (
                    ["keep_original", "mute"],
                    {
                        "default": "keep_original",
                        "tooltip": (
                            "keep_original = copy the original audio unchanged (possible desync with a variable ramp).\n"
                            "mute = no audio.\n"
                            "---\n"
                            "keep_original = копировать оригинальное аудио без изменений (возможен рассинхрон с переменным ramp).\n"
                            "mute = без звука."
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
        print(f"\n[AGSoft Video Speed Ramp Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Video Speed Ramp] FFmpeg не смог применить speed ramp. Ошибка: {error_msg}")

    def apply_speed_ramp(
        self,
        output_name, output_path, video_path,
        speed_start, speed_end, ramp_time, audio_mode,
        encoder_mode, quality_preset,
        video=None
    ):
        output_name = _single(output_name, "speed_ramp.mp4")
        output_path = _single(output_path, "")
        video_path = _single(video_path, "").strip()
        audio_mode = _single(audio_mode, "keep_original").strip().lower()
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")

        speed_start = clamp(safe_float(speed_start, 1.0), 0.1, 10.0)
        speed_end = clamp(safe_float(speed_end, 2.0), 0.1, 10.0)
        ramp_time = max(0.0, safe_float(ramp_time, 0.0))

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_speed_ramp_")
        try:
            video_src = None
            if video is not None:
                video_src = _extract_video_path(video, tmp_dir)
                if video_src:
                    print(f"[AGSoft Video Speed Ramp] Источник из входа video: {video_src}")
            if video_src is None and video_path:
                if os.path.isfile(video_path):
                    video_src = video_path
                else:
                    print(f"[AGSoft Video Speed Ramp] Warning: video_path не найден ({video_path}).")
            if video_src is None:
                raise ValueError("[AGSoft Video Speed Ramp] Не задан источник видео: подключите video или укажите video_path.")

            vinfo = parse_media_info(video_src)
            if vinfo["width"] <= 0 or vinfo["height"] <= 0:
                raise ValueError(f"[AGSoft Video Speed Ramp] В источнике нет видеопотока: {video_src}")
            has_audio = bool(vinfo["has_audio"])
            duration = float(vinfo["duration"] or 0.0)
            if duration <= 0:
                raise ValueError(f"[AGSoft Video Speed Ramp] Не удалось определить длительность видео: {video_src}")

            safe_name = os.path.basename(str(output_name or "speed_ramp.mp4"))
            name_without_ext, ext = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "speed_ramp"
            if not ext:
                ext = ".mp4"
            safe_name = f"{name_without_ext}{ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Video Speed Ramp] Путь назначения — папка, а не файл: {final_output_path}")
            if os.path.normcase(os.path.realpath(video_src)) == os.path.normcase(os.path.realpath(final_output_path)):
                raise ValueError(f"[AGSoft Video Speed Ramp] Итоговый файл не может совпадать с источником.")

            filter_complex = build_speed_ramp_filter(speed_start, speed_end, ramp_time, duration, audio_mode, has_audio)

            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-i", video_src]
            cmd += ["-filter_complex", filter_complex]
            cmd += ["-map", "[outv]"]

            if audio_mode == "keep_original" and has_audio:
                cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
            else:
                cmd += ["-an"]

            cmd += build_encoder_args(choose_encoder(encoder_mode), quality_preset)
            if safe_name.lower().endswith((".mp4", ".mov")):
                cmd += ["-movflags", "+faststart"]
            cmd.append(final_output_path)

            avg_speed = (speed_start + speed_end) / 2.0
            est_dur = duration / avg_speed if avg_speed > 0 else duration

            result = run_ffmpeg_with_progress(cmd, est_dur)
            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Video Speed Ramp] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Video Speed Ramp] Saved: {final_output_path} | "
                f"speed {speed_start} -> {speed_end} over {ramp_time}s | "
                f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
                out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoSpeedRamp": AGSoftVideoSpeedRamp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoSpeedRamp": "🎬⏱️AGSoft Video Speed Ramp"
}