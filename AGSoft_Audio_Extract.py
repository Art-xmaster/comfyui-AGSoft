# ==============================================================================
# AGSoft_Audio_Extract.py
# ==============================================================================
# Нода: 🎬🔊AGSoft Extract Audio From Video
#
# Описание / Description:
# Извлекает аудиодорожку из видео в отдельный файл (mp3/wav/m4a/flac/ogg),
# опционально фрагмент.
# 
# Автор / Author: AGSoft
# Дата / Date: 04.08.2026
# ==============================================================================

import os
import re
import json
import wave
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
    """
    Многоуровневое извлечение пути к файлу из объекта VIDEO.
    Приоритет: атрибуты -> save_to() -> get_components_internal().
    """
    if video_obj is None:
        return None

    # Способ 1: Прямой доступ через атрибуты
    for attr_name in ['path', 'video_path', 'file_path', '_path', 'source_path']:
        if hasattr(video_obj, attr_name):
            candidate = getattr(video_obj, attr_name)
            if isinstance(candidate, str) and candidate and os.path.isfile(candidate):
                return candidate

    # Способ 2: save_to() для сохранения во временный файл
    if hasattr(video_obj, 'save_to'):
        try:
            fd, temp_path = tempfile.mkstemp(prefix="video_src_", suffix=".mp4", dir=tmp_dir)
            os.close(fd)
            video_obj.save_to(temp_path)
            if os.path.isfile(temp_path):
                return temp_path
        except Exception as e:
            logging.warning(f"save_to failed: {e}")

    # Способ 3: get_components_internal() -> сохранить кадры во временный mp4
    if hasattr(video_obj, 'get_components_internal'):
        try:
            components = video_obj.get_components_internal()
            if components and len(components) > 0:
                fd, temp_path = tempfile.mkstemp(prefix="video_src_", suffix=".mp4", dir=tmp_dir)
                os.close(fd)
                
                # Конвертируем кадры в mp4 через FFmpeg
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


def audio_to_wav(audio_obj, path):
    wf = None
    sr = 44100
    if isinstance(audio_obj, dict):
        wf = audio_obj.get("waveform")
        sr = int(audio_obj.get("sample_rate", 44100))
    else:
        gw = getattr(audio_obj, "get_waveform", None) or getattr(audio_obj, "waveform", None)
        gs = getattr(audio_obj, "get_sample_rate", None) or getattr(audio_obj, "sample_rate", None)
        wf = gw() if callable(gw) else gw
        try:
            sr = int(gs() if callable(gs) else gs)
        except Exception:
            sr = 44100
    if wf is None:
        raise ValueError("[AGSoft Extract Audio From Video] В аудио-объекте нет waveform.")

    arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    nchannels = pcm.shape[0]
    pcm = pcm.T

    with wave.open(path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _extract_audio_from_video(video_obj):
    """Извлекает аудио-объект из VIDEO (dict/tuple/атрибут)."""
    obj = video_obj
    ga = getattr(obj, "get_audio", None)
    if callable(ga):
        try:
            a = ga()
            if a is not None:
                return a
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj.get("audio")
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict) and "waveform" in item:
                return item
    else:
        if hasattr(obj, "audio"):
            return getattr(obj, "audio")
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
        print(f"[AGSoft Extract Audio From Video] Не удалось прочитать файл: {path} | {e}")
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
        break

    return info


def run_ffmpeg_with_progress(cmd, total_duration=0.0):
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Extract Audio From Video] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Extract Audio From Video] Progress: {bucket * 5}%")
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


class AGSoftAudioExtract:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT")
    RETURN_NAMES = ("audio_path", "duration_seconds", "duration_timecode", "file_size_mb")

    FUNCTION = "extract_audio"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬🔊AGSoft Extract Audio From Video.\n"
        "Extracts the audio track from a video into a separate file (mp3 / wav / m4a / flac / ogg), "
        "optionally a fragment via start_time / end_time (0 = to the end). Source: video input "
        "(priority) or video_path.\n"
        "---\n"
        "🎬🔊AGSoft Extract Audio From Video.\n"
        "Извлекает аудиодорожку из видео в отдельный файл (mp3 / wav / m4a / flac / ogg), "
        "опционально - фрагмент через start_time / end_time (0 = до конца). Источник: вход "
        "video (приоритет) или video_path."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "extracted_audio",
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
                "format": (
                    ["mp3", "wav", "m4a", "flac", "ogg"],
                    {
                        "default": "mp3",
                        "tooltip": "Output audio format. / Формат выходного аудио."
                    }
                ),
                "audio_bitrate": (
                    ["96k", "128k", "192k", "256k", "320k"],
                    {
                        "default": "192k",
                        "tooltip": (
                            "Bitrate for lossy formats (mp3/m4a/ogg). Ignored for wav/flac.\n"
                            "---\n"
                            "Битрейт для lossy-форматов. Для wav/flac игнорируется."
                        )
                    }
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": "Fragment start in seconds. 0 = from the beginning. / Начало фрагмента. 0 = с начала."
                    }
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": "Fragment end in seconds. 0 = to the end. / Конец фрагмента. 0 = до конца."
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
        print(f"\n[AGSoft Extract Audio From Video Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Extract Audio From Video] FFmpeg не смог извлечь аудио. Ошибка: {error_msg}")

    def extract_audio(
        self,
        output_name, output_path, video_path, format, audio_bitrate, start_time, end_time,
        video=None
    ):
        output_name = _single(output_name, "extracted_audio")
        output_path = _single(output_path, "")
        video_path = _single(video_path, "").strip()
        fmt = _single(format, "mp3").strip().lower()
        bitrate = _single(audio_bitrate, "192k").strip().lower()

        start = max(0.0, safe_float(start_time, 0.0))
        end = max(0.0, safe_float(end_time, 0.0))
        if end > 0 and end <= start:
            raise ValueError("[AGSoft Extract Audio From Video] end_time должен быть больше start_time.")

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_extract_audio_")
        try:
            # Источник: вход video -> video_path
            source = None
            if video is not None:
                # Сначала пробуем получить путь к файлу
                source = _extract_video_path(video, tmp_dir)
                if source:
                    print(f"[AGSoft Extract Audio From Video] Источник из пути входа video: {source}")
                else:
                    # Если пути нет — берём встроенное аудио
                    audio_obj = _extract_audio_from_video(video)
                    if audio_obj is None:
                        raise ValueError(
                            f"[AGSoft Extract Audio From Video] Во входе video нет ни пути, ни аудиодорожки. type={type(video)}."
                        )
                    fd, tmp_wav = tempfile.mkstemp(prefix="src_audio_", suffix=".wav", dir=tmp_dir)
                    os.close(fd)
                    audio_to_wav(audio_obj, tmp_wav)
                    source = tmp_wav
                    print("[AGSoft Extract Audio From Video] Источник из встроенного аудио входа video (временный WAV).")
            if source is None and video_path:
                if os.path.isfile(video_path):
                    source = video_path
                else:
                    print(f"[AGSoft Extract Audio From Video] Warning: video_path не найден ({video_path}).")
            if source is None:
                raise ValueError("[AGSoft Extract Audio From Video] Не задан источник: подключите video или укажите video_path.")

            sinfo = parse_media_info(source)
            if not sinfo["has_audio"]:
                raise ValueError(f"[AGSoft Extract Audio From Video] В источнике нет аудиодорожки: {source}")

            safe_name = os.path.basename(str(output_name or "extracted_audio"))
            name_without_ext, _ = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "extracted_audio"
            safe_name = f"{name_without_ext}.{fmt}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Extract Audio From Video] Путь назначения — папка, а не файл: {final_output_path}")

            if os.path.normcase(os.path.realpath(source)) == os.path.normcase(os.path.realpath(final_output_path)):
                raise ValueError("[AGSoft Extract Audio From Video] Итоговый файл не может совпадать с источником.")

            if fmt == "mp3":
                codec_args = ["-c:a", "libmp3lame", "-b:a", bitrate]
            elif fmt == "wav":
                codec_args = ["-c:a", "pcm_s16le"]
            elif fmt == "m4a":
                codec_args = ["-c:a", "aac", "-b:a", bitrate]
            elif fmt == "flac":
                codec_args = ["-c:a", "flac"]
            else:  # ogg
                codec_args = ["-c:a", "libvorbis", "-b:a", bitrate]

            input_args = []
            if start > 0:
                input_args += ["-ss", f"{start:.6f}"]
            input_args += ["-i", source]

            t_arg = ["-t", f"{end - start:.6f}"] if end > 0 else []

            cmd = [
                FFMPEG_PATH, "-y", "-hide_banner",
                *input_args,
                "-vn",
                "-map", "0:a:0",
                *codec_args,
                *t_arg,
                final_output_path,
            ]

            expected_dur = (end - start) if end > 0 else float(sinfo["duration"] or 0.0)
            result = run_ffmpeg_with_progress(cmd, expected_dur)

            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Extract Audio From Video] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Extract Audio From Video] Saved: {final_output_path} | format={fmt} | "
                f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftAudioExtract": AGSoftAudioExtract
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAudioExtract": "🎬🔊AGSoft Extract Audio From Video"
}