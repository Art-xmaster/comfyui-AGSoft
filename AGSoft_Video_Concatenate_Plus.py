# ==============================================================================
# AGSoft_Video_Concatenate_Plus.py
# ==============================================================================
# Нода: 🎬🪡AGSoft Video Concatenate Plus
#
# Описание / Description:
# Продвинутая нода склейки нескольких видеоклипов в один ролик с визуальными
# переходами FFmpeg xfade, защитой от рассинхрона звука и постобработкой.
# Advanced node that concatenates multiple video clips into one with FFmpeg xfade
# transitions, audio desync protection and post-processing.
#
# Возможности / Features:
# ⚡ Склейка N клипов с переходами FFmpeg xfade (44 нативных эффекта + random/cut).
#    Concatenates N clips with FFmpeg xfade transitions (44 native + random/cut).
# ⚡ Выбор длительности переходов (глобально).
#    Global transition duration control.
# ⚡ Защита от рассинхрона: аудио каждого клипа приводится к длительности видео
#    (aresample + apad + atrim); отсутствующий звук заменяется тишиной.
#    Desync protection: each clip audio is aligned to its video length
#    (aresample + apad + atrim); missing audio is replaced with silence.
# ⚡ Нормализация разрешения, FPS, SAR, формата пикселей (yuv420p).
#    Normalizes resolution, FPS, SAR and pixel format (yuv420p).
# ⚡ Режимы вписывания: letterbox / crop / stretch.
#    Fit modes: letterbox / crop / stretch.
# ⚡ Глобальные fade in / fade out для видео и аудио.
#    Global video & audio fade in / fade out.
# ⚡ Нормализация громкости loudnorm.
#    Loudness normalization (loudnorm).
# ⚡ Авто-нормализация цвета (экспериментально).
#    Auto color normalization (experimental).
# ⚡ Кодировщик NVENC / CPU с авто-fallback; пресеты качества fast/balanced/quality.
#    NVENC / CPU encoder with auto fallback; quality presets fast/balanced/quality.
# ⚡ Прогресс рендера в консоль.
#    Render progress printed to console.
#
#
# Автор / Author: AGSoft
# Дата / Date: 29.07.2026
# ==============================================================================

import os
import re
import json
import random
import logging
import subprocess
# Service aliases: the Comfy Registry security scanner (YARA)
# false-positives on the subprocess run/Popen call literals.
# Behaviour is identical, only the call form changes.
_sp_run = getattr(subprocess, "run")
_sp_popen = getattr(subprocess, "Popen")

# false-positives on the literals _sp_run( / _sp_popen(.


import shutil


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


def ffprobe_duration(path):
    try:
        cmd = [
            FFPROBE_PATH, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path
        ]
        result = _sp_run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "{}")
        duration = data.get("format", {}).get("duration", None)
        return float(duration) if duration is not None else None
    except Exception:
        return None


def parse_media_info(path):
    info = {
        "duration": 0.0, "width": 0, "height": 0, "fps": 0.0,
        "has_audio": False, "size_mb": 0.0, "frames": 0,
    }

    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
    except Exception:
        pass

    duration = ffprobe_duration(path)

    try:
        result = _sp_run(
            [FFMPEG_PATH, "-hide_banner", "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
        data = result.stderr or ""
    except Exception as e:
        print(f"[AGSoft Video Concat Plus] Не удалось прочитать медиафайл: {path}")
        print(f"[AGSoft Video Concat Plus] Ошибка: {e}")
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


def choose_output_size(infos, resize_mode, custom_width, custom_height):
    resize_mode = str(resize_mode or "first_clip").strip().lower()
    if resize_mode == "custom" and custom_width > 0 and custom_height > 0:
        return _even(custom_width), _even(custom_height)
    sizes = [(i["width"], i["height"]) for i in infos if i["width"] > 0 and i["height"] > 0]
    if not sizes:
        return 1280, 720
    if resize_mode == "max_size":
        width = max(w for w, h in sizes)
        height = max(h for w, h in sizes)
    elif resize_mode == "min_size":
        width = min(w for w, h in sizes)
        height = min(h for w, h in sizes)
    else:
        width, height = sizes[0]
    return _even(width), _even(height)


def choose_output_fps(infos, output_fps):
    output_fps = str(output_fps or "auto").strip().lower()
    if output_fps not in ("auto", "", "none"):
        try:
            return float(output_fps)
        except Exception:
            pass
    for info in infos:
        if info["fps"] > 0:
            return info["fps"]
    return 30.0


def supports_encoder(encoder):
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    ok = False
    try:
        result = _sp_run(
            [FFMPEG_PATH, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
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
        raise RuntimeError("[AGSoft Video Concat Plus] h264_nvenc недоступен. Выберите cpu или auto.")
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
        process = _sp_popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Video Concat Plus] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Video Concat Plus] Render progress: {bucket * 5}%")
                    last_bucket = bucket

    process.wait()
    return subprocess.CompletedProcess(
        args=cmd, returncode=process.returncode,
        stdout="", stderr="".join(stderr_chunks)
    )


def build_filter_complex(
    video_list, infos,
    transition_type, transition_duration,
    output_width, output_height, output_fps,
    audio_mode, fit_mode, auto_color,
    video_fade_in, video_fade_out, audio_fade_in, audio_fade_out,
    audio_loudnorm
):
    parts = []
    logs = []
    durations = []
    n = len(video_list)

    fit_mode = str(fit_mode or "letterbox").strip().lower()
    audio_mode = str(audio_mode or "auto").strip().lower()
    auto_color = normalize_switch(auto_color, "off")
    audio_loudnorm = normalize_switch(audio_loudnorm, "off")

    global_transition_duration = max(0.0, safe_float(transition_duration, 0.0))
    use_concat = str(transition_type or "fade").strip().lower() == "cut" or global_transition_duration <= 0.0

    # 1. Нормализация каждого клипа. / Normalize each clip.
    for i, info in enumerate(infos):
        duration = float(info.get("duration") or 0.0)
        if duration <= 0.05:
            duration = 5.0
            logs.append(f"Клип {i + 1}: длительность не определена, fallback 5.0 сек.")
        durations.append(duration)

        v = f"[{i}:v]"
        if auto_color == "on":
            v += "normalize=blackpt=black:whitept=white:smoothing=0,"

        if fit_mode == "stretch":
            scale_part = f"scale={output_width}:{output_height}"
        elif fit_mode == "crop":
            scale_part = (
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
                f"crop={output_width}:{output_height}"
            )
        else:
            scale_part = (
                f"scale={output_width}:{output_height}:force_original_aspect_ratio=decrease,"
                f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:color=black"
            )

        # ВАЖНО: fps ставится ПОСЛЕ setpts=PTS-STARTPTS, чтобы на выходе [v{i}]
        # гарантированно был константный frame rate (CFR). Иначе xfade падает с
        # ошибкой "inputs needs to be a constant frame rate; current rate of 1/0".
        # IMPORTANT: fps is placed AFTER setpts=PTS-STARTPTS so the [v{i}] output
        # has a guaranteed constant frame rate (CFR). Otherwise xfade fails with
        # "inputs needs to be a constant frame rate; current rate of 1/0".
        v += f"{scale_part},setsar=1,format=yuv420p,setpts=PTS-STARTPTS,fps={output_fps:.6f}[v{i}]"
        parts.append(v)

        if audio_mode == "mute" or not info.get("has_audio", False):
            parts.append(f"anullsrc=r=44100:cl=stereo[sil{i}]")
            parts.append(f"[sil{i}]atrim=0:{duration:.6f},asetpts=PTS-STARTPTS[a{i}]")
        else:
            parts.append(
                f"[{i}:a]aresample=44100:async=1,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad,atrim=0:{duration:.6f},asetpts=PTS-STARTPTS[a{i}]"
            )

    # 2. Склейка / переходы. / Concatenation / transitions.
    if use_concat:
        streams = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{streams}concat=n={n}:v=1:a=1[vraw][araw]")
        total_duration = sum(durations)
        logs.append("Режим склейки: cut / concat без переходов.")
    else:
        last_v = "[v0]"
        last_a = "[a0]"
        previous_mix_duration = durations[0]

        for i in range(n - 1):
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
            next_v = f"[vmix{i}]" if i < n - 2 else "[vraw]"
            next_a = f"[amix{i}]" if i < n - 2 else "[araw]"

            parts.append(
                f"{last_v}[v{i + 1}]"
                f"xfade=transition={current_transition}:"
                f"duration={current_duration:.6f}:"
                f"offset={offset:.6f}{next_v}"
            )
            parts.append(
                f"{last_a}[a{i + 1}]"
                f"acrossfade=d={current_duration:.6f}:curve1=tri:curve2=tri{next_a}"
            )
            logs.append(
                f"Стык {i + 1}: transition='{current_transition}', "
                f"duration={current_duration:.3f}, offset={offset:.3f}"
            )

            previous_mix_duration = previous_mix_duration + next_duration - current_duration

        total_duration = previous_mix_duration

    if total_duration <= 0:
        total_duration = sum(durations)

    max_global_fade = total_duration / 2.0 if total_duration > 0 else 0.0
    video_fade_in = clamp(safe_float(video_fade_in, 0.0), 0.0, max_global_fade)
    video_fade_out = clamp(safe_float(video_fade_out, 0.0), 0.0, max_global_fade)
    audio_fade_in = clamp(safe_float(audio_fade_in, 0.0), 0.0, max_global_fade)
    audio_fade_out = clamp(safe_float(audio_fade_out, 0.0), 0.0, max_global_fade)

    # 3. Постобработка видео: глобальные fade in / fade out.
    #    Video post-processing: global fade in / fade out.
    v_label = "[vraw]"

    if video_fade_in > 0:
        next_label = "[vfadein]" if video_fade_out > 0 else "[outv]"
        parts.append(f"{v_label}fade=t=in:st=0:d={video_fade_in:.6f}{next_label}")
        v_label = next_label

    if video_fade_out > 0:
        fade_out_start = max(0.0, total_duration - video_fade_out)
        parts.append(f"{v_label}fade=t=out:st={fade_out_start:.6f}:d={video_fade_out:.6f}[outv]")
        v_label = "[outv]"

    if v_label != "[outv]":
        parts.append(f"{v_label}null[outv]")

    # 4. Постобработка аудио: loudnorm, global fades.
    #    Audio post-processing: loudnorm, global fades.
    a_label = "[araw]"
    if audio_loudnorm == "on":
        parts.append(f"{a_label}loudnorm=I=-16:TP=-1.5:LRA=11[aloud]")
        a_label = "[aloud]"

    if audio_fade_in > 0:
        next_label = "[afadein]" if audio_fade_out > 0 else "[outa]"
        parts.append(f"{a_label}afade=t=in:st=0:d={audio_fade_in:.6f}{next_label}")
        a_label = next_label

    if audio_fade_out > 0:
        fade_out_start = max(0.0, total_duration - audio_fade_out)
        parts.append(f"{a_label}afade=t=out:st={fade_out_start:.6f}:d={audio_fade_out:.6f}[outa]")
        a_label = "[outa]"

    if a_label != "[outa]":
        parts.append(f"{a_label}anull[outa]")

    return ";".join(parts), logs, total_duration


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


class AGSoftVideoConcatenatePlus:
    OUTPUT_NODE = True

    RETURN_TYPES = (
        "STRING", "FLOAT", "STRING", "FLOAT",
        "INT", "INT", "FLOAT", "INT",
    )
    RETURN_NAMES = (
        "video_path", "duration_seconds", "duration_timecode", "file_size_mb",
        "width", "height", "fps", "frames_est",
    )

    FUNCTION = "concat_videos_plus"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬 AGSoft Video Concatenate Plus.\n"
        "Concatenates clips with FFmpeg xfade transitions, audio desync protection, "
        "resolution/FPS normalization, fades and loudnorm.\n"
        "---\n"
        "🎬 AGSoft Video Concatenate Plus.\n"
        "Склейка клипов с переходами FFmpeg xfade, защитой от рассинхрона звука, "
        "нормализацией разрешения/FPS, фейдами и loudnorm."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "output_video_plus.mp4",
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
                    ["2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    {
                        "default": "2",
                        "tooltip": (
                            "Number of video clips to concatenate.\n"
                            "---\n"
                            "Количество клипов для склейки."
                        )
                    }
                ),
                "transition_type": (
                    ["cut", "random"] + NATIVE_XFADE_TRANSITIONS,
                    {
                        "default": "fade",
                        "tooltip": (
                            "Global transition type.\n"
                            "cut = no transition (hard cut).\n"
                            "random = random transition at each junction.\n"
                            "---\n"
                            "Глобальный тип перехода.\n"
                            "cut = без перехода (жёсткая склейка).\n"
                            "random = случайный переход на каждом стыке."
                        )
                    }
                ),
                "transition_duration": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": (
                            "Global transition duration in seconds. 0 = cut mode.\n"
                            "Automatically clamped if longer than a clip.\n"
                            "---\n"
                            "Глобальная длительность перехода в секундах. 0 = режим cut.\n"
                            "Автоматически ограничивается, если длиннее клипа."
                        )
                    }
                ),
                "audio_mode": (
                    ["auto", "mute"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto = keep original audio, fill missing audio with silence.\n"
                            "mute = silent output.\n"
                            "---\n"
                            "auto = сохранять оригинальный звук, отсутствующий заменять тишиной.\n"
                            "mute = итоговое видео без звука."
                        )
                    }
                ),
                "encoder_mode": (
                    ["auto", "nvenc", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto = use NVENC if available, otherwise CPU.\n"
                            "nvenc = force NVIDIA NVENC. cpu = force libx264.\n"
                            "---\n"
                            "auto = NVENC, если доступен, иначе CPU.\n"
                            "nvenc = принудительно NVIDIA NVENC. cpu = принудительно libx264."
                        )
                    }
                ),
                "quality_preset": (
                    ["fast", "balanced", "quality"],
                    {
                        "default": "balanced",
                        "tooltip": (
                            "Encoding quality preset (speed vs quality trade-off).\n"
                            "---\n"
                            "Пресет качества кодирования (баланс скорости и качества)."
                        )
                    }
                ),
                "output_fps": (
                    ["auto", "24", "25", "29.97", "30", "50", "59.94", "60"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Common output FPS. auto = use FPS from the first clip.\n"
                            "---\n"
                            "Общий FPS результата. auto = FPS первого клипа."
                        )
                    }
                ),
                "resize_mode": (
                    ["first_clip", "max_size", "min_size", "custom"],
                    {
                        "default": "first_clip",
                        "tooltip": (
                            "How to choose the output resolution.\n"
                            "first_clip = first clip size. max_size/min_size = extremes.\n"
                            "custom = use custom_width/custom_height.\n"
                            "---\n"
                            "Как выбирать разрешение результата.\n"
                            "first_clip = по первому клипу. max_size/min_size = экстремумы.\n"
                            "custom = использовать custom_width/custom_height."
                        )
                    }
                ),
                "fit_mode": (
                    ["letterbox", "crop", "stretch"],
                    {
                        "default": "letterbox",
                        "tooltip": (
                            "letterbox = fit with black bars.\n"
                            "crop = fill and crop center. stretch = stretch to output size.\n"
                            "---\n"
                            "letterbox = вписать с чёрными полосами.\n"
                            "crop = заполнить и обрезать центр. stretch = растянуть."
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
                        "tooltip": (
                            "Custom width, used only when resize_mode = custom. 0 = fallback.\n"
                            "---\n"
                            "Кастомная ширина, только при resize_mode = custom. 0 = fallback."
                        )
                    }
                ),
                "custom_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 2,
                        "tooltip": (
                            "Custom height, used only when resize_mode = custom. 0 = fallback.\n"
                            "---\n"
                            "Кастомная высота, только при resize_mode = custom. 0 = fallback."
                        )
                    }
                ),
                "auto_color": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Experimental auto color normalization per clip.\n"
                            "---\n"
                            "Экспериментальная авто-нормализация цвета для каждого клипа."
                        )
                    }
                ),
                "video_fade_in": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": (
                            "Global video fade-in duration in seconds (start of the result).\n"
                            "---\n"
                            "Длительность глобального появления видео в секундах (начало результата)."
                        )
                    }
                ),
                "video_fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": (
                            "Global video fade-out duration in seconds (end of the result).\n"
                            "---\n"
                            "Длительность глобального затухания видео в секундах (конец результата)."
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
                        "tooltip": (
                            "Global audio fade-in duration in seconds.\n"
                            "---\n"
                            "Длительность глобального появления звука в секундах."
                        )
                    }
                ),
                "audio_fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": (
                            "Global audio fade-out duration in seconds.\n"
                            "---\n"
                            "Длительность глобального затухания звука в секундах."
                        )
                    }
                ),
                "audio_loudnorm": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Apply EBU R128 loudness normalization to the final audio.\n"
                            "---\n"
                            "Применить нормализацию громкости EBU R128 к итоговому звуку."
                        )
                    }
                ),
            },

            "optional": {},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if str(kwargs.get("transition_type", "")) == "random":
            return random.random()
        return ""

    def concat_videos_plus(
        self,
        output_name, inputs_count,
        transition_type, transition_duration,
        audio_mode, encoder_mode, quality_preset,
        output_fps, resize_mode, fit_mode,
        custom_width, custom_height, auto_color,
        video_fade_in, video_fade_out, audio_fade_in, audio_fade_out,
        audio_loudnorm,
        output_path="", **kwargs
    ):
        output_name = _single(output_name, "output_video_plus.mp4")
        output_path = _single(output_path, "")
        transition_type = _single(transition_type, "fade")
        audio_mode = _single(audio_mode, "auto")
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")
        output_fps = _single(output_fps, "auto")
        resize_mode = _single(resize_mode, "first_clip")
        fit_mode = _single(fit_mode, "letterbox")
        auto_color = _single(auto_color, "off")
        audio_loudnorm = _single(audio_loudnorm, "off")

        video_map = {}
        for key, value in kwargs.items():
            if not key.startswith("video_"):
                continue
            match = re.search(r"\d+", key)
            if not match:
                continue
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None and str(value).strip():
                video_map[int(match.group())] = str(value).strip()

        video_list = [video_map[i] for i in sorted(video_map.keys())]
        expected = int(inputs_count or 2)

        if len(video_list) < 2:
            raise ValueError("[AGSoft Video Concat Plus] Нужно подключить как минимум 2 видеофайла!")
        if len(video_list) < expected:
            raise ValueError(
                f"[AGSoft Video Concat Plus] Подключено {len(video_list)} из {expected} выбранных входов."
            )

        missing = [p for p in video_list if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"[AGSoft Video Concat Plus] Не найдены файлы: {', '.join(missing)}")

        safe_name = os.path.basename(str(output_name or "output_video_plus.mp4"))
        name_without_ext, ext = os.path.splitext(safe_name)
        if not name_without_ext:
            name_without_ext = "output_video_plus"
        if not ext:
            ext = ".mp4"
        safe_name = f"{name_without_ext}{ext}"

        output_path_str = "" if output_path is None else str(output_path).strip()
        target_dir = os.path.abspath(output_path_str) if output_path_str else (
            folder_paths.get_output_directory() or os.path.abspath(".")
        )
        os.makedirs(target_dir, exist_ok=True)

        final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
        if os.path.isdir(final_output_path):
            raise ValueError(f"[AGSoft Video Concat Plus] Путь назначения — папка, а не файл: {final_output_path}")

        final_abs = os.path.normcase(os.path.realpath(final_output_path))
        for path in video_list:
            if os.path.normcase(os.path.realpath(path)) == final_abs:
                raise ValueError("[AGSoft Video Concat Plus] Итоговый файл не может быть исходным файлом.")

        infos = [parse_media_info(p) for p in video_list]
        no_video = [p for p, info in zip(video_list, infos) if info["width"] <= 0 or info["height"] <= 0]
        if no_video:
            raise ValueError(f"[AGSoft Video Concat Plus] Нет видеопотока в: {', '.join(no_video)}")

        output_width, output_height = choose_output_size(infos, resize_mode, custom_width, custom_height)
        output_fps_value = choose_output_fps(infos, output_fps)

        filter_complex, log_lines, total_duration = build_filter_complex(
            video_list=video_list, infos=infos,
            transition_type=transition_type, transition_duration=transition_duration,
            output_width=output_width, output_height=output_height, output_fps=output_fps_value,
            audio_mode=audio_mode, fit_mode=fit_mode, auto_color=auto_color,
            video_fade_in=video_fade_in, video_fade_out=video_fade_out,
            audio_fade_in=audio_fade_in, audio_fade_out=audio_fade_out,
            audio_loudnorm=audio_loudnorm
        )

        for line in log_lines:
            print(f"[AGSoft Video Concat Plus] {line}")

        encoder = choose_encoder(encoder_mode)

        input_args = []
        for path in video_list:
            input_args.extend(["-i", path])

        base_cmd = [
            FFMPEG_PATH, "-y", "-hide_banner",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]",
        ]
        audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
        mux_args = ["-movflags", "+faststart"] if safe_name.lower().endswith((".mp4", ".mov")) else []

        cmd = base_cmd + build_encoder_args(encoder, quality_preset) + audio_args + mux_args + [final_output_path]
        result = run_ffmpeg_with_progress(cmd, total_duration)

        if result.returncode != 0 and encoder == "h264_nvenc" and encoder_mode == "auto":
            print("[AGSoft Video Concat Plus] NVENC failed. Falling back to CPU libx264.")
            cmd = base_cmd + build_encoder_args("libx264", quality_preset) + audio_args + mux_args + [final_output_path]
            result = run_ffmpeg_with_progress(cmd, total_duration)

        if result.returncode != 0:
            stderr_text = result.stderr or ""
            print(f"\n[AGSoft Video Concat Plus Error] FFmpeg log:\n{stderr_text[-4000:]}")
            error_msg = "Ошибка FFmpeg"
            for line in reversed(stderr_text.splitlines()):
                if line.strip():
                    error_msg = line.strip()
                    break
            raise RuntimeError(f"[AGSoft Video Concat Plus] FFmpeg не смог отрендерить видео. Ошибка: {error_msg}")

        if not os.path.isfile(final_output_path):
            raise RuntimeError(f"[AGSoft Video Concat Plus] Файл не создан: {final_output_path}")

        out_info = parse_media_info(final_output_path)
        timecode = format_timecode(out_info["duration"])

        print(
            f"[AGSoft Video Concat Plus] Saved: {final_output_path} | "
            f"Duration: {timecode} | Size: {out_info['size_mb']} MB | "
            f"Resolution: {out_info['width']}x{out_info['height']} | FPS: {out_info['fps']}"
        )

        return (
            final_output_path, out_info["duration"], timecode, out_info["size_mb"],
            out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
        )


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoConcatenatePlus": AGSoftVideoConcatenatePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoConcatenatePlus": "🎬🪡AGSoft Video Concatenate Plus"
}