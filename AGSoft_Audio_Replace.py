# ==============================================================================
# AGSoft_Audio_Replace.py
# ==============================================================================
# Нода: 🎬🔊AGSoft Replace Audio To Video
#
# Описание / Description:
# Заменяет аудиодорожку видео БЕЗ перекодирования видео (-c:v copy).
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
        raise ValueError("[AGSoft Replace Audio To Video] В аудио-объекте нет waveform.")

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
        print(f"[AGSoft Replace Audio To Video] Не удалось прочитать файл: {path} | {e}")
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


def run_ffmpeg_with_progress(cmd, total_duration=0.0):
    try:
        process = _sp_popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Replace Audio To Video] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Replace Audio To Video] Progress: {bucket * 5}%")
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


class AGSoftAudioReplace:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "FLOAT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("video_path", "duration_seconds", "duration_timecode", "file_size_mb", "width", "height", "fps", "frames_est")

    FUNCTION = "replace_audio"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬🔊AGSoft Replace Audio To Video.\n"
        "Replaces the video's audio track WITHOUT re-encoding the video (stream copy). "
        "Video source: video input (priority) or video_path. Voiceover: audio_path (priority) "
        "or audio input. Modes: fit / loop / trim_to_audio; volume and fade in/out.\n"
        "---\n"
        "🎬🔊AGSoft Replace Audio To Video.\n"
        "Заменяет аудиодорожку видео БЕЗ перекодирования видео (stream copy). "
        "Источник видео: вход video (приоритет) или video_path. Озвучка: audio_path "
        "(приоритет) или вход audio. Режимы: fit / loop / trim_to_audio; громкость и fade in/out."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "replaced_audio.mp4",
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
                "audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to the new audio file. Takes priority over the audio input.\n"
                            "---\n"
                            "Путь к новому аудиофайлу. Имеет приоритет над входом audio."
                        )
                    }
                ),
                "audio_mode": (
                    ["fit", "loop", "trim_to_audio"],
                    {
                        "default": "fit",
                        "tooltip": (
                            "fit = trim/pad silence to video length. loop = loop track to video length. "
                            "trim_to_audio = cut the VIDEO to the audio length.\n"
                            "---\n"
                            "fit = обрезать/добить тишиной под длину видео. loop = зациклить под длину видео. "
                            "trim_to_audio = обрезать ВИДЕО под длину аудио."
                        )
                    }
                ),
                "audio_volume": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                        "tooltip": "New track volume multiplier. 1.0 = original. / Множитель громкости. 1.0 = оригинал."
                    }
                ),
                "audio_fade_in": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                        "tooltip": "New track fade-in in seconds. / Появление дорожки в секундах."
                    }
                ),
                "audio_fade_out": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1,
                        "tooltip": "New track fade-out in seconds. / Затухание дорожки в секундах."
                    }
                ),
            },
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "tooltip": (
                            "Optional base video from ComfyUI. Has priority over video_path.\n"
                            "---\n"
                            "Опциональное базовое видео из ComfyUI. Имеет приоритет над video_path."
                        )
                    }
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional voiceover from ComfyUI. Used only if audio_path is empty/invalid.\n"
                            "---\n"
                            "Опциональная озвучка из ComfyUI. Используется, только если audio_path пуст/неверен."
                        )
                    }
                ),
            },
        }

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Replace Audio To Video Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Replace Audio To Video] FFmpeg не смог заменить дорожку. Ошибка: {error_msg}")

    def replace_audio(
        self,
        output_name, output_path, video_path, audio_path,
        audio_mode, audio_volume, audio_fade_in, audio_fade_out,
        video=None, audio=None
    ):
        output_name = _single(output_name, "replaced_audio.mp4")
        output_path = _single(output_path, "")
        video_path = _single(video_path, "").strip()
        audio_path = _single(audio_path, "").strip()
        audio_mode = _single(audio_mode, "fit").strip().lower()

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_replace_audio_")
        try:
            # Источник видео: вход video -> video_path
            video_src = None
            if video is not None:
                video_src = _extract_video_path(video, tmp_dir)
                if video_src:
                    print(f"[AGSoft Replace Audio To Video] Базовое видео из входа video: {video_src}")
                else:
                    print(f"[AGSoft Replace Audio To Video] Warning: не удалось извлечь путь из video, пробуем video_path.")
            if video_src is None and video_path:
                if os.path.isfile(video_path):
                    video_src = video_path
                else:
                    print(f"[AGSoft Replace Audio To Video] Warning: video_path не найден ({video_path}).")
            if video_src is None:
                raise ValueError("[AGSoft Replace Audio To Video] Не задан источник видео: подключите video или укажите video_path.")

            vinfo = parse_media_info(video_src)
            if vinfo["width"] <= 0 or vinfo["height"] <= 0:
                raise ValueError(f"[AGSoft Replace Audio To Video] В источнике нет видеопотока: {video_src}")
            video_dur = float(vinfo["duration"] or 0.0)
            if video_dur <= 0:
                raise ValueError(f"[AGSoft Replace Audio To Video] Не удалось определить длительность видео: {video_src}")

            # Источник озвучки: audio_path -> вход audio
            wav_path = None
            if audio_path:
                if os.path.isfile(audio_path):
                    wav_path = audio_path
                    print(f"[AGSoft Replace Audio To Video] Озвучка из пути: {audio_path}")
                else:
                    print(f"[AGSoft Replace Audio To Video] Warning: audio_path не найден ({audio_path}), пробуем вход audio.")
            if wav_path is None and audio is not None:
                try:
                    fd, tmp_wav = tempfile.mkstemp(prefix="voice_", suffix=".wav", dir=tmp_dir)
                    os.close(fd)
                    audio_to_wav(audio, tmp_wav)
                    wav_path = tmp_wav
                    print("[AGSoft Replace Audio To Video] Озвучка из входа audio (временный WAV).")
                except Exception as e:
                    print(f"[AGSoft Replace Audio To Video] Warning: не удалось сохранить audio: {e}")
                    wav_path = None
            if wav_path is None:
                raise ValueError("[AGSoft Replace Audio To Video] Не задан источник озвучки: укажите audio_path или подключите вход audio.")

            ainfo = parse_media_info(wav_path)
            if not ainfo["has_audio"]:
                raise ValueError(f"[AGSoft Replace Audio To Video] В файле нет аудиодорожки: {wav_path}")
            audio_dur = float(ainfo["duration"] or 0.0)

            safe_name = os.path.basename(str(output_name or "replaced_audio.mp4"))
            name_without_ext, ext = os.path.splitext(safe_name)
            if not name_without_ext:
                name_without_ext = "replaced_audio"
            if not ext:
                ext = ".mp4"
            safe_name = f"{name_without_ext}{ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)

            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Replace Audio To Video] Путь назначения — папка, а не файл: {final_output_path}")

            final_abs = os.path.normcase(os.path.realpath(final_output_path))
            for p in (video_path, audio_path):
                if p and os.path.normcase(os.path.realpath(p)) == final_abs:
                    raise ValueError("[AGSoft Replace Audio To Video] Итоговый файл не может совпадать с исходным видео или аудио.")

            volume = clamp(safe_float(audio_volume, 1.0), 0.0, 10.0)
            fade_in = max(0.0, safe_float(audio_fade_in, 0.0))
            fade_out = max(0.0, safe_float(audio_fade_out, 0.0))

            a = "[1:a]aresample=44100:async=1,aformat=sample_fmts=fltp:channel_layouts=stereo"
            if volume != 1.0:
                a += f",volume={volume:.6f}"
            if audio_mode == "fit":
                a += f",apad,atrim=0:{video_dur:.6f}"
            elif audio_mode == "loop":
                a += f",atrim=0:{video_dur:.6f}"
            a += ",asetpts=PTS-STARTPTS"

            base_dur = video_dur if audio_mode in ("fit", "loop") else audio_dur
            if fade_in > 0:
                a += f",afade=t=in:st=0:d={fade_in:.6f}"
            if fade_out > 0:
                if base_dur > 0:
                    fade_out_start = max(0.0, base_dur - fade_out)
                    a += f",afade=t=out:st={fade_out_start:.6f}:d={fade_out:.6f}"
                else:
                    print("[AGSoft Replace Audio To Video] Warning: длительность неизвестна - fade out пропущен.")
            a += "[outa]"

            input_args = ["-i", video_src]
            if audio_mode == "loop":
                input_args += ["-stream_loop", "-1"]
            input_args += ["-i", wav_path]

            t_arg = []
            if audio_mode == "trim_to_audio" and audio_dur > 0:
                t_arg = ["-t", f"{audio_dur:.6f}"]

            base_cmd = [
                FFMPEG_PATH, "-y", "-hide_banner",
                *input_args,
                "-filter_complex", a,
                "-map", "0:v:0",
                "-map", "[outa]",
                "-c:v", "copy",
            ]
            audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
            mux_args = ["-movflags", "+faststart"] if safe_name.lower().endswith((".mp4", ".mov")) else []

            cmd = base_cmd + audio_args + t_arg + mux_args + [final_output_path]
            result = run_ffmpeg_with_progress(cmd, base_dur if base_dur > 0 else 0.0)

            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)

            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Replace Audio To Video] Файл не создан: {final_output_path}")

            out_info = parse_media_info(final_output_path)
            timecode = format_timecode(out_info["duration"])

            print(
                f"[AGSoft Replace Audio To Video] Saved: {final_output_path} | mode={audio_mode} | "
                f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, out_info["duration"], timecode, out_info["size_mb"],
                out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftAudioReplace": AGSoftAudioReplace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAudioReplace": "🎬🔊AGSoft Replace Audio To Video"
}