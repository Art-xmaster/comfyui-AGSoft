# ==============================================================================
# AGSoft_Video_Split.py
# ==============================================================================
# Нода: 🎬✂️AGSoft Video Split
#
# Описание / Description:
# Вырезание фрагмента видео по старт/финишу в двух режимах:
#   fast    = stream copy, по ключевым кадрам (быстро, без перекодирования);
#   precise = перекод, покадрово (точно).
# Аудио: keep (родная дорожка) / replace (наложить новую) / mute (без звука).
#
# Автор / Author: AGSoft
# Дата / Date: 06.08.2026
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
        raise ValueError("[AGSoft Video Split] В аудио-объекте нет waveform.")

    arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    pcm = pcm.T

    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


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
        print(f"[AGSoft Video Split] Не удалось прочитать файл: {path} | {e}")
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


def choose_encoder(encoder_mode):
    encoder_mode = str(encoder_mode or "auto").strip().lower()
    if encoder_mode == "cpu":
        return "libx264"
    if encoder_mode == "nvenc":
        if supports_encoder("h264_nvenc"):
            return "h264_nvenc"
        raise RuntimeError("[AGSoft Video Split] h264_nvenc недоступен. Выберите cpu или auto.")
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
        raise RuntimeError(f"[AGSoft Video Split] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Video Split] Progress: {bucket * 5}%")
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


class AGSoftVideoSplit:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_path", "duration_seconds", "duration_timecode", "file_size_mb", "width", "height", "fps", "frames_est")

    FUNCTION = "split_video"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬✂️AGSoft Video Split.\n"
        "Cuts a fragment [start_time..end_time] out of a video in two modes: "
        "fast (stream copy, by keyframes - quick, no re-encode) or precise (re-encode, frame-accurate). "
        "Audio: keep (original track) / replace (overlay a new one from audio_path or audio input) / mute. "
        "Sources: video input (priority) or video_path.\n"
        "---\n"
        "🎬✂️AGSoft Video Split.\n"
        "Вырезает фрагмент [start_time..end_time] из видео в двух режимах: "
        "fast (stream copy, по ключевым кадрам - быстро, без перекодирования) или precise (перекод, покадрово). "
        "Аудио: keep (родная дорожка) / replace (наложить новую из audio_path или входа audio) / mute. "
        "Источники: вход video (приоритет) или video_path."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "split_video.mp4",
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
                "cut_mode": (
                    ["fast", "precise"],
                    {
                        "default": "fast",
                        "tooltip": (
                            "fast = stream copy, cut by keyframes (quick, no re-encode, start snaps to a keyframe).\n"
                            "precise = re-encode, frame-accurate cut.\n"
                            "---\n"
                            "fast = stream copy, резка по ключевым кадрам (быстро, старт привязан к ключевому кадру).\n"
                            "precise = перекод, покадрово точная резка."
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
                "audio_source": (
                    ["keep", "replace", "mute"],
                    {
                        "default": "keep",
                        "tooltip": (
                            "keep = keep the original audio track (cut together with the video).\n"
                            "replace = overlay a new track from audio_path / audio input.\n"
                            "mute = no audio.\n"
                            "---\n"
                            "keep = оставить родную дорожку (режется вместе с видео).\n"
                            "replace = наложить новую дорожку из audio_path / входа audio.\n"
                            "mute = без звука."
                        )
                    }
                ),
                "audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to the new audio (used only when audio_source=replace). Takes priority over the audio input.\n"
                            "---\n"
                            "Путь к новому аудио (только при audio_source=replace). Приоритет над входом audio."
                        )
                    }
                ),
                "audio_mode": (
                    ["fit", "loop"],
                    {
                        "default": "fit",
                        "tooltip": (
                            "Only when audio_source=replace. fit = trim/pad silence to the fragment length. "
                            "loop = loop the new track across the fragment.\n"
                            "---\n"
                            "Только при audio_source=replace. fit = обрезать/добить тишиной под длину фрагмента. "
                            "loop = зациклить новую дорожку под фрагмент."
                        )
                    }
                ),
                "audio_volume": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "New track volume (replace only). 1.0 = original. / Громкость новой дорожки (только replace). 1.0 = оригинал."
                    }
                ),
                "audio_fade_in": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                        "tooltip": "New track fade-in (replace only). / Появление новой дорожки (только replace)."
                    }
                ),
                "audio_fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                        "tooltip": "New track fade-out (replace only). / Затухание новой дорожки (только replace)."
                    }
                ),
                "encoder_mode": (
                    ["auto", "nvenc", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Used only in precise mode. auto = NVENC if available, otherwise CPU.\n"
                            "---\n"
                            "Только в режиме precise. auto = NVENC, если доступен, иначе CPU."
                        )
                    }
                ),
                "quality_preset": (
                    ["fast", "balanced", "quality"],
                    {
                        "default": "balanced",
                        "tooltip": "Encoding quality (precise mode only). / Качество кодирования (только precise)."
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
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional new voiceover (used only when audio_source=replace and audio_path is empty).\n"
                            "---\n"
                            "Опциональная новая озвучка (только при audio_source=replace и пустом audio_path)."
                        )
                    }
                ),
            },
        }

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Video Split Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Video Split] FFmpeg не смог вырезать фрагмент. Ошибка: {error_msg}")

    def split_video(
        self,
        output_name, output_path, video_path,
        cut_mode, start_time, end_time,
        audio_source, audio_path, audio_mode, audio_volume, audio_fade_in, audio_fade_out,
        encoder_mode, quality_preset,
        video=None, audio=None
    ):
        output_name = _single(output_name, "split_video.mp4")
        output_path = _single(output_path, "")
        video_path = _single(video_path, "").strip()
        cut_mode = _single(cut_mode, "fast").strip().lower()
        audio_source = _single(audio_source, "keep").strip().lower()
        audio_path = _single(audio_path, "").strip()
        audio_mode = _single(audio_mode, "fit").strip().lower()
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_video_split_")
        try:
            # Источник видео: вход video -> video_path
            video_src = None
            if video is not None:
                video_src = _extract_video_path(video, tmp_dir)
                if video_src:
                    print(f"[AGSoft Video Split] Источник из входа video: {video_src}")
            if video_src is None and video_path:
                if os.path.isfile(video_path):
                    video_src = video_path
                else:
                    print(f"[AGSoft Video Split] Warning: video_path не найден ({video_path}).")
            if video_src is None:
                raise ValueError("[AGSoft Video Split] Не задан источник видео: подключите video или укажите video_path.")

            vinfo = parse_media_info(video_src)
            if vinfo["width"] <= 0 or vinfo["height"] <= 0:
                raise ValueError(f"[AGSoft Video Split] В источнике нет видеопотока: {video_src}")
            duration = float(vinfo["duration"] or 0.0)
            if duration <= 0:
                raise ValueError(f"[AGSoft Video Split] Не удалось определить длительность видео: {video_src}")

            # Границы фрагмента. / Fragment bounds.
            start = clamp(safe_float(start_time, 0.0), 0.0, duration)
            if safe_float(end_time, 0.0) > 0:
                end = clamp(safe_float(end_time, 0.0), start, duration)
            else:
                end = duration
            frag_dur = end - start
            if frag_dur <= 0:
                raise ValueError("[AGSoft Video Split] end_time должен быть больше start_time.")

            # Новая озвучка (только replace). / New voiceover (replace only).
            wav_path = None
            if audio_source == "replace":
                if audio_path:
                    if os.path.isfile(audio_path):
                        wav_path = audio_path
                    else:
                        print(f"[AGSoft Video Split] Warning: audio_path не найден ({audio_path}), пробуем вход audio.")
                if wav_path is None and audio is not None:
                    try:
                        fd, tmp_wav = tempfile.mkstemp(prefix="voice_", suffix=".wav", dir=tmp_dir)
                        os.close(fd)
                        audio_to_wav(audio, tmp_wav)
                        wav_path = tmp_wav
                    except Exception as e:
                        print(f"[AGSoft Video Split] Warning: не удалось сохранить audio: {e}")
                        wav_path = None
                if wav_path is None:
                    raise ValueError("[AGSoft Video Split] audio_source=replace, но не задан источник озвучки.")

            # Имя и папка выхода. / Output name and folder.
            safe_name = os.path.basename(str(output_name or "split_video.mp4"))
            name_without_ext, ext = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "split_video"
            if not ext:
                ext = ".mp4"
            safe_name = f"{name_without_ext}{ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Video Split] Путь назначения — папка, а не файл: {final_output_path}")
            if os.path.normcase(os.path.realpath(video_src)) == os.path.normcase(os.path.realpath(final_output_path)):
                raise ValueError("[AGSoft Video Split] Итоговый файл не может совпадать с источником.")

            # Сборка команды. / Build command.
            input_args = ["-ss", f"{start:.6f}", "-i", video_src]
            filter_complex = ""

            if audio_source == "replace":
                if audio_mode == "loop":
                    input_args += ["-stream_loop", "-1"]
                input_args += ["-i", wav_path]
                a = "[1:a]aresample=44100:async=1,aformat=sample_fmts=fltp:channel_layouts=stereo"
                volume = clamp(safe_float(audio_volume, 1.0), 0.0, 10.0)
                if volume != 1.0:
                    a += f",volume={volume:.6f}"
                if audio_mode == "fit":
                    a += ",apad"
                a += f",atrim=0:{frag_dur:.6f},asetpts=PTS-STARTPTS"
                if audio_fade_in > 0:
                    a += f",afade=t=in:st=0:d={audio_fade_in:.6f}"
                if audio_fade_out > 0:
                    a += f",afade=t=out:st={max(0.0, frag_dur - audio_fade_out):.6f}:d={audio_fade_out:.6f}"
                a += "[outa]"
                filter_complex = a

            cmd = [FFMPEG_PATH, "-y", "-hide_banner", *input_args]
            if filter_complex:
                cmd += ["-filter_complex", filter_complex]

            cmd += ["-map", "0:v:0"]
            if audio_source == "keep":
                cmd += ["-map", "0:a:0"]
            elif audio_source == "replace":
                cmd += ["-map", "[outa]"]
            else:  # mute
                cmd += ["-an"]

            # Видеокодек: fast = copy, precise = перекод. / Video codec.
            if cut_mode == "fast":
                cmd += ["-c:v", "copy"]
                if audio_source == "keep":
                    cmd += ["-c:a", "copy"]
                elif audio_source == "replace":
                    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
            else:
                cmd += build_encoder_args(choose_encoder(encoder_mode), quality_preset)
                if audio_source != "mute":
                    cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]

            cmd += ["-t", f"{frag_dur:.6f}"]
            if safe_name.lower().endswith((".mp4", ".mov")):
                cmd += ["-movflags", "+faststart"]
            cmd.append(final_output_path)

            result = run_ffmpeg_with_progress(cmd, frag_dur)
            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Video Split] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Video Split] Saved: {final_output_path} | mode={cut_mode} | audio={audio_source} | "
                f"[{format_timecode(start)} .. {format_timecode(end)}] | Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
                out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoSplit": AGSoftVideoSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoSplit": "🎬✂️AGSoft Video Split"
}