# ==============================================================================
# AGSoft_Audio_Concatenate_Plus.py
# Нода: 🔊🪡AGSoft Audio Concatenate Plus
#
# Описание / Description:
# Склейка нескольких аудиофайлов в один с эффектами. Режимы: concat (встык) и
# crossfade (плавный переход между треками с настраиваемой длительностью и
# кривой). Дополнительно: автовыравнивание громкости всех треков (loudnorm,
# EBU R128), fade in/out всего результата, выбор формата и битрейта выхода.
# Количество входов задаётся динамически (inputs_count) через JS-сокеты audio_N.
# Concatenates several audio files into one with effects. Modes: concat (back-to-
# back) and crossfade (smooth transition with adjustable duration and curve).
# Additionally: automatic loudness normalization of all tracks (loudnorm,
# EBU R128), fade in/out of the whole result, output format and bitrate choice.
# The number of inputs is set dynamically (inputs_count) via JS audio_N sockets.
#
# Возможности / Features:
# ⚡ Динамическое число входов (inputs_count, 2-10) через JS-сокеты audio_N.
#    Dynamic number of inputs (inputs_count, 2-10) via JS audio_N sockets.
# ⚡ Режим concat (встык) и crossfade (плавный переход).
#    Concat (back-to-back) and crossfade (smooth transition) modes.
# ⚡ Длительность и кривая кроссфейда (tri/exp/log/qsin/hsin/dese/desi/nofade).
#    Crossfade duration and curve.
# ⚡ Нормализация громкости всех треков (loudnorm, EBU R128) перед склейкой.
#    Loudness normalization of all tracks (loudnorm, EBU R128) before joining.
# ⚡ Fade in / fade out всего результата.
#    Fade in / fade out of the whole result.
# ⚡ Формат выхода mp3/wav/flac/ogg/m4a + битрейт для lossy.
#    Output format mp3/wav/flac/ogg/m4a + bitrate for lossy.
#
# Автор / Author: AGSoft
# Дата / Date: 17.08.2026
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


MODES = ["concat", "crossfade"]
CURVES = ["tri", "exp", "log", "qsin", "hsin", "dese", "desi", "nofade"]
FORMAT_CHOICES = ["mp3", "wav", "flac", "ogg", "m4a"]
FORMAT_CODECS = {
    ".mp3": "libmp3lame",
    ".wav": "pcm_s16le",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".m4a": "aac",
}
LOSSY_EXTS = {".mp3", ".ogg", ".m4a"}

LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"


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


def codec_args_for(ext, bitrate):
    codec = FORMAT_CODECS.get(ext, "pcm_s16le")
    args = ["-c:a", codec]
    if ext in LOSSY_EXTS:
        args += ["-b:a", bitrate]
    return args


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


def parse_audio_info(path):
    info = {"duration": 0.0, "sample_rate": 0, "channels": 0, "size_mb": 0.0}
    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
    except Exception:
        pass

    dur = ffprobe_duration(path)
    if dur and dur > 0:
        info["duration"] = round(dur, 3)

    try:
        result = subprocess.run([FFMPEG_PATH, "-hide_banner", "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        data = result.stderr or ""
    except Exception:
        return info

    if info["duration"] <= 0:
        m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", data)
        if m:
            info["duration"] = round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)

    audio_line = ""
    for line in data.splitlines():
        if "Audio:" in line:
            audio_line = line
            break
    if audio_line:
        sm = re.search(r"(\d+)\s*Hz", audio_line)
        if sm:
            info["sample_rate"] = int(sm.group(1))
        cm = re.search(r"Hz,\s*(mono|stereo|(\d+)\s*channels)", audio_line)
        if cm:
            if cm.group(1) == "mono":
                info["channels"] = 1
            elif cm.group(1) == "stereo":
                info["channels"] = 2
            elif cm.group(2):
                info["channels"] = int(cm.group(2))
    return info


def build_concat_filter(sources, mode, crossfade_dur, curve, loudnorm_on, fade_in, fade_out, total_dur):
    """Сборка filter_complex для склейки аудио."""
    parts = []
    n = len(sources)

    # Нормализация каждого входа (sample rate + формат + опционально loudnorm).
    for i in range(n):
        chain = f"[{i}:a]aresample=44100:async=1,aformat=sample_fmts=fltp:channel_layouts=stereo"
        if loudnorm_on:
            chain += f",{LOUDNORM}"
        chain += f"[a{i}]"
        parts.append(chain)

    # Склейка.
    if n == 1:
        parts.append("[a0]anull[a_joined]")
    elif mode == "crossfade" and crossfade_dur > 0:
        # Последовательный acrossfade.
        prev = "a0"
        for i in range(1, n):
            out_label = "a_joined" if i == n - 1 else f"xf{i}"
            parts.append(
                f"[{prev}][a{i}]acrossfade=d={crossfade_dur:.6f}:c1={curve}:c2={curve}[{out_label}]"
            )
            prev = out_label
    else:
        # Простая склейка concat.
        inputs = "".join(f"[a{i}]" for i in range(n))
        parts.append(f"{inputs}concat=n={n}:v=0:a=1[a_joined]")

    # Fade in / fade out всего результата.
    fades = []
    if fade_in > 0:
        fades.append(f"afade=t=in:st=0:d={fade_in:.6f}")
    if fade_out > 0:
        st = max(0.0, total_dur - fade_out)
        fades.append(f"afade=t=out:st={st:.6f}:d={fade_out:.6f}")
    if fades:
        parts.append(f"[a_joined]{','.join(fades)}[a_final]")
    else:
        parts.append("[a_joined]anull[a_final]")

    return ";".join(parts)


def run_ffmpeg_with_progress(cmd, total_duration=0.0, tag="[AGSoft Audio Concat Plus]"):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"{tag} FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"{tag} Render progress: {bucket * 5}%")
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


class AGSoftAudioConcatenatePlus:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("audio_path", "duration_seconds", "duration_timecode", "file_size_mb")

    FUNCTION = "concat_audios_plus"
    CATEGORY = "AGSoft/Audio"

    # JS для динамических входов audio_N.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🔊🪡AGSoft Audio Concatenate Plus.\n"
        "Concatenates several audio files into one with effects. Modes: concat (back-to-back) and "
        "crossfade (smooth transition with adjustable duration and curve: tri/exp/log/qsin/hsin/"
        "dese/desi/nofade). Automatic loudness normalization (loudnorm, EBU R128) levels all tracks "
        "before joining. Fade in/out of the whole result. Output format mp3/wav/flac/ogg/m4a and "
        "bitrate. Number of inputs is set dynamically via inputs_count (JS adds audio_N sockets).\n"
        "---\n"
        "🔊🪡AGSoft Audio Concatenate Plus.\n"
        "Склейка нескольких аудиофайлов в один с эффектами. Режимы: concat (встык) и crossfade "
        "(плавный переход с настраиваемой длительностью и кривой: tri/exp/log/qsin/hsin/dese/desi/"
        "nofade). Автовыравнивание громкости (loudnorm, EBU R128) приводит все треки к единому "
        "уровню перед склейкой. Fade in/out всего результата. Формат выхода mp3/wav/flac/ogg/m4a и "
        "битрейт. Количество входов задаётся динамически через inputs_count (JS добавляет сокеты audio_N)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "output_audio_plus.mp3",
                        "tooltip": (
                            "Output filename. The extension is set by the chosen format.\n"
                            "---\n"
                            "Имя итогового файла. Расширение задаётся выбранным форматом."
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
                            "Number of audio tracks to concatenate.\n"
                            "---\n"
                            "Количество аудио-дорожек для склейки."
                        )
                    }
                ),
                "mode": (
                    MODES,
                    {
                        "default": "concat",
                        "tooltip": (
                            "concat = join tracks back-to-back (no transition).\n"
                            "crossfade = smooth transition between tracks.\n"
                            "---\n"
                            "concat = склейка встык (без перехода).\n"
                            "crossfade = плавный переход между треками."
                        )
                    }
                ),
                "crossfade_duration": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1,
                        "tooltip": (
                            "Crossfade duration (seconds) between adjacent tracks. Used only in crossfade mode.\n"
                            "---\n"
                            "Длительность кроссфейда (в секундах) между соседними треками. Только в режиме crossfade."
                        )
                    }
                ),
                "crossfade_curve": (
                    CURVES,
                    {
                        "default": "tri",
                        "tooltip": (
                            "Crossfade curve. tri = linear, exp = exponential, log = logarithmic, "
                            "qsin/hsin = sine-based, dese/desi = sigmoid, nofade = hard cut.\n"
                            "---\n"
                            "Кривая кроссфейда. tri = линейная, exp = экспоненциальная, log = логарифмическая, "
                            "qsin/hsin = синусоидные, dese/desi = сигмоидные, nofade = жёсткий рез."
                        )
                    }
                ),
                "loudnorm": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "Normalize loudness of all tracks to EBU R128 (I=-16 LUFS) before joining, "
                            "so no track is louder than another.\n"
                            "---\n"
                            "Нормализовать громкость всех треков по EBU R128 (I=-16 LUFS) перед склейкой, "
                            "чтобы ни один трек не был громче другого."
                        )
                    }
                ),
                "fade_in": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.05,
                        "tooltip": (
                            "Fade-in of the whole result (seconds). 0 = off.\n"
                            "---\n"
                            "Нарастание всего результата (в секундах). 0 = выкл."
                        )
                    }
                ),
                "fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.05,
                        "tooltip": (
                            "Fade-out of the whole result (seconds). 0 = off.\n"
                            "---\n"
                            "Затухание всего результата (в секундах). 0 = выкл."
                        )
                    }
                ),
                "format": (
                    FORMAT_CHOICES,
                    {
                        "default": "mp3",
                        "tooltip": (
                            "Output format. mp3/wav/flac/ogg/m4a.\n"
                            "---\n"
                            "Формат выхода. mp3/wav/flac/ogg/m4a."
                        )
                    }
                ),
                "bitrate": (
                    ["96k", "128k", "192k", "256k", "320k"],
                    {
                        "default": "192k",
                        "tooltip": (
                            "Bitrate for lossy formats (mp3/ogg/m4a). Ignored for wav/flac.\n"
                            "---\n"
                            "Битрейт для lossy-форматов (mp3/ogg/m4a). Игнорируется для wav/flac."
                        )
                    }
                ),
            },
            "optional": {},
        }

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Audio Concat Plus Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Audio Concat Plus] FFmpeg не смог склеить аудио. Ошибка: {error_msg}")

    def concat_audios_plus(
        self,
        output_name, inputs_count, mode,
        crossfade_duration, crossfade_curve, loudnorm,
        fade_in, fade_out, format, bitrate,
        output_path="", **kwargs
    ):
        output_name = _single(output_name, "output_audio_plus.mp3")
        output_path = _single(output_path, "")
        mode = _single(mode, "concat").strip().lower()
        crossfade_curve = _single(crossfade_curve, "tri").strip().lower()
        loudnorm_on = _single(loudnorm, "off").strip().lower() == "on"
        format_choice = _single(format, "mp3").strip().lower()
        bitrate = _single(bitrate, "192k").strip().lower()

        crossfade_duration = max(0.0, safe_float(crossfade_duration, 1.0))
        fade_in = max(0.0, safe_float(fade_in, 0.0))
        fade_out = max(0.0, safe_float(fade_out, 0.0))

        # Собрать пути из динамических входов-сокетов audio_N (как video_N в Video Concat Plus).
        audio_map = {}
        for key, value in kwargs.items():
            if not key.startswith("audio_"):
                continue
            match = re.search(r"\d+", key)
            if not match:
                continue
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None and str(value).strip():
                audio_map[int(match.group())] = str(value).strip()

        audio_list = [audio_map[i] for i in sorted(audio_map.keys())]

        expected = int(inputs_count or 2)
        if len(audio_list) < 2:
            raise ValueError("[AGSoft Audio Concat Plus] Нужно подключить как минимум 2 аудиофайла!")
        if len(audio_list) < expected:
            raise ValueError(
                f"[AGSoft Audio Concat Plus] Подключено {len(audio_list)} из {expected} выбранных входов."
            )

        missing = [p for p in audio_list if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"[AGSoft Audio Concat Plus] Не найдены файлы: {', '.join(missing)}")

        # Имя и папка выхода.
        target_ext = "." + format_choice
        safe_name = os.path.basename(str(output_name or "output_audio_plus"))
        name_no_ext, _ = os.path.splitext(safe_name)
        if not name_no_ext:
            name_no_ext = "output_audio_plus"
        safe_name = f"{name_no_ext}{target_ext}"

        output_path_str = "" if output_path is None else str(output_path).strip()
        target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
        os.makedirs(target_dir, exist_ok=True)
        final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
        if os.path.isdir(final_output_path):
            raise ValueError(f"[AGSoft Audio Concat Plus] Путь назначения — папка, а не файл: {final_output_path}")

        final_abs = os.path.normcase(os.path.realpath(final_output_path))
        for path in audio_list:
            if os.path.normcase(os.path.realpath(path)) == final_abs:
                raise ValueError("[AGSoft Audio Concat Plus] Итоговый файл не может быть исходным файлом.")

        # Длительности всех треков (для fade_out и общей длительности).
        durations = []
        for p in audio_list:
            info = parse_audio_info(p)
            d = info["duration"]
            if d <= 0.05:
                d = 5.0
                print(f"[AGSoft Audio Concat Plus] Длительность не определена, fallback 5.0 сек: {p}")
            durations.append(d)

        # Ограничить кроссфейд, чтобы он не превышал самый короткий трек.
        effective_crossfade = crossfade_duration
        if mode == "crossfade" and effective_crossfade > 0:
            min_dur = min(durations) if durations else 0.0
            if min_dur > 0 and effective_crossfade >= min_dur:
                effective_crossfade = max(0.1, min_dur * 0.5)
                print(f"[AGSoft Audio Concat Plus] Warning: кроссфейд уменьшен до {effective_crossfade:.2f}s (короткий трек).")

        # Общая длительность результата.
        total_dur = sum(durations)
        if mode == "crossfade" and effective_crossfade > 0:
            total_dur -= (len(audio_list) - 1) * effective_crossfade
        total_dur = max(0.0, total_dur)

        # Сборка фильтра.
        filter_complex = build_concat_filter(
            audio_list, mode, effective_crossfade, crossfade_curve,
            loudnorm_on, fade_in, fade_out, total_dur
        )

        # Команда: все входы + filter_complex.
        input_args = []
        for path in audio_list:
            input_args.extend(["-i", path])

        cmd = [FFMPEG_PATH, "-y", "-hide_banner", *input_args]
        cmd += ["-filter_complex", filter_complex]
        cmd += ["-map", "[a_final]"]
        cmd += codec_args_for(target_ext, bitrate)
        if safe_name.lower().endswith((".mp3",)):
            pass  # mp3 не нуждается в movflags
        cmd.append(final_output_path)

        result = run_ffmpeg_with_progress(cmd, total_dur, "[AGSoft Audio Concat Plus]")
        if result.returncode != 0:
            self._raise_ffmpeg_error(result.stderr)
        if not os.path.isfile(final_output_path):
            raise RuntimeError(f"[AGSoft Audio Concat Plus] Файл не создан: {final_output_path}")

        out_info = parse_audio_info(final_output_path)
        timecode = format_timecode(out_info["duration"])

        print(
            f"[AGSoft Audio Concat Plus] Saved: {final_output_path} | mode={mode} | "
            f"tracks={len(audio_list)} | crossfade={effective_crossfade:.2f}s | loudnorm={loudnorm_on} | "
            f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
        )

        return (
            final_output_path, out_info["duration"], timecode, out_info["size_mb"],
        )


NODE_CLASS_MAPPINGS = {
    "AGSoftAudioConcatenatePlus": AGSoftAudioConcatenatePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAudioConcatenatePlus": "🔊🪡AGSoft Audio Concatenate Plus"
}