# ==============================================================================
# AGSoft_Audio_Split.py
# Нода: 🔊✂️AGSoft Audio Split
#
# Описание / Description:
# Вырезает фрагмент из аудиофайла по заданным временным меткам start_time и
# end_time. Два режима: fast (stream copy, без перекодировки, быстро) и precise
# (перекодирование, точность до сэмпла). Формат выхода по умолчанию как у
# источника (без перекодировки); можно выбрать mp3/wav/flac/ogg/m4a. Возвращает
# путь к файлу, тип AUDIO для цепочек и метаданные.
# ВАЖНО про источник: stream copy (без перекодировки) возможен только когда
# файл подключён напрямую через audio_path. Если звук приходит через сокет
# AUDIO (из другой ноды), он уже декодирован в PCM, поэтому форматом "source"
# будет WAV, а не исходный MP3 и т.п.
# Cuts a fragment from an audio file by start_time and end_time. Two modes:
# fast (stream copy, no re-encode, quick) and precise (re-encode, sample-accurate).
# Output format defaults to the source format (no re-encode); mp3/wav/flac/ogg/m4a
# can be selected. Returns the file path, an AUDIO type for chaining, and metadata.
# IMPORTANT about the source: stream copy (no re-encode) is only possible when
# the file is connected directly via audio_path. If audio comes through the AUDIO
# socket (from another node), it is already decoded to PCM, so the "source" format
# becomes WAV, not the original MP3 etc.
#
# Возможности / Features:
# ⚡ Вырезание фрагмента по start_time / end_time (0 = до конца).
#    Fragment cutting by start_time / end_time (0 = to the end).
# ⚡ Режим fast (stream copy, без перекодировки) и precise (точно до сэмпла).
#    Fast mode (stream copy, no re-encode) and precise (sample-accurate).
# ⚡ Формат выхода: source (как у источника) / mp3 / wav / flac / ogg / m4a.
#    Output format: source (same as input) / mp3 / wav / flac / ogg / m4a.
# ⚡ Источник: вход audio (сокет AUDIO, приоритет) или audio_path (путь к файлу).
#    Source: audio input (AUDIO socket, priority) or audio_path (file path).
# ⚡ ВАЖНО: сокет AUDIO передаёт уже декодированный PCM (WAV), поэтому
#    format=source через сокет даст WAV. Для настоящего stream copy
#    подключайте файл через audio_path.
#    IMPORTANT: the AUDIO socket delivers already-decoded PCM (WAV), so
#    format=source via the socket gives WAV. For a true stream copy connect
#    the file via audio_path.
# ⚡ Выход типа AUDIO для цепочек + метаданные (длительность, sample rate, каналы).
#    AUDIO-type output for chaining + metadata (duration, sample rate, channels).
#
# Автор / Author: AGSoft
# Дата / Date: 15.08.2026
# ==============================================================================

import os
import re
import json
import wave
import logging
import tempfile
import subprocess
import shutil

import torch
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


# Форматы и кодеки. / Formats and codecs.
FORMAT_CHOICES = ["source", "mp3", "wav", "flac", "ogg", "m4a"]
FORMAT_CODECS = {
    ".mp3": "libmp3lame",
    ".wav": "pcm_s16le",
    ".flac": "flac",
    ".ogg": "libvorbis",
    ".m4a": "aac",
}
LOSSY_EXTS = {".mp3", ".ogg", ".m4a"}


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


def resolve_target_ext(format_choice, source_ext):
    """Определяет расширение выхода. source = расширение источника (или .wav)."""
    if format_choice == "source":
        return source_ext if source_ext in FORMAT_CODECS else ".wav"
    return "." + format_choice


def codec_args_for(ext, bitrate):
    """Аргументы кодека для расширения; битрейт только для lossy."""
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


def audio_to_wav(audio_obj, path):
    """Сохраняет объект AUDIO (сокет) в WAV."""
    wf = audio_obj.get("waveform")
    sr = int(audio_obj.get("sample_rate", 44100))

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


def audio_file_to_comfy_audio(path, tmp_dir):
    """Декодирует аудиофайл в объект AUDIO (для цепочек). Форма: (1, channels, samples)."""
    wav_path = os.path.join(tmp_dir, "_comfy_audio_out.wav")
    cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-i", path, "-vn", "-c:a", "pcm_s16le", wav_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
        with wave.open(wav_path, "rb") as w:
            nchannels = w.getnchannels()
            sr = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        arr = arr.reshape(-1, nchannels)   # (samples, channels)
        arr = arr.T                        # (channels, samples)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, channels, samples)
        return {"waveform": tensor, "sample_rate": sr}
    except Exception as e:
        print(f"[AGSoft Audio Split] Warning: не удалось создать AUDIO-выход: {e}")
        return {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}


def run_ffmpeg_with_progress(cmd, total_duration=0.0, tag="[AGSoft Audio Split]"):
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
                    print(f"{tag} Progress: {bucket * 5}%")
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


class AGSoftAudioSplit:
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "AUDIO", "FLOAT", "STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("audio_path", "audio", "duration_seconds", "duration_timecode", "file_size_mb", "sample_rate", "channels")

    FUNCTION = "split_audio"
    CATEGORY = "AGSoft/Audio"

    DESCRIPTION = (
        "🔊✂️AGSoft Audio Split.\n"
        "Cuts a fragment from an audio file by start_time / end_time. Mode fast = stream copy "
        "(no re-encode, quick, cut at codec-frame boundary), mode precise = re-encode (sample-accurate). "
        "Output format defaults to the source format (no re-encode); mp3/wav/flac/ogg/m4a can be chosen. "
        "Returns the file path, an AUDIO type for chaining, and metadata.\n"
        "IMPORTANT about the source: stream copy is only possible when the file is connected directly "
        "via audio_path. If audio comes through the AUDIO socket (from another node), it is already "
        "decoded to PCM, so the 'source' format becomes WAV, not the original MP3. To get MP3 etc. in "
        "that case, choose it explicitly (re-encode) or connect the file via audio_path.\n"
        "---\n"
        "🔊✂️AGSoft Audio Split.\n"
        "Вырезает фрагмент из аудиофайла по start_time / end_time. Режим fast = stream copy "
        "(без перекодировки, быстро, рез по границе кадра кодека), режим precise = перекодирование "
        "(точность до сэмпла). Формат выхода по умолчанию как у источника (без перекодировки); можно "
        "выбрать mp3/wav/flac/ogg/m4a. Возвращает путь к файлу, тип AUDIO для цепочек и метаданные.\n"
        "ВАЖНО про источник: stream copy возможен только когда файл подключён напрямую через audio_path. "
        "Если звук приходит через сокет AUDIO (из другой ноды), он уже декодирован в PCM, поэтому "
        "форматом 'source' будет WAV, а не исходный MP3. Чтобы получить MP3 и т.д. в этом случае, "
        "выберите его явно (перекодирование) либо подключите файл через audio_path."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "audio_split.mp3",
                        "tooltip": (
                            "Output filename. The extension is set by the effective output format.\n"
                            "---\n"
                            "Имя итогового файла. Расширение задаётся итоговым форматом."
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
                "audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to the source audio file. Used only if audio input is not connected.\n"
                            "TIP: for a true lossless stream copy (fast mode, format=source), connect the "
                            "file here via audio_path rather than the AUDIO socket - the socket always "
                            "delivers already-decoded PCM (WAV).\n"
                            "---\n"
                            "Путь к исходному аудиофайлу. Используется, только если вход audio не подключен.\n"
                            "СОВЕТ: для настоящего stream copy без потерь (режим fast, format=source) "
                            "подключайте файл сюда через audio_path, а не через сокет AUDIO - сокет всегда "
                            "передаёт уже декодированный PCM (WAV)."
                        )
                    }
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": (
                            "Fragment start (seconds). 0 = from the beginning.\n"
                            "---\n"
                            "Начало фрагмента (в секундах). 0 = с начала."
                        )
                    }
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                        "tooltip": (
                            "Fragment end (seconds). 0 = to the end of the audio.\n"
                            "---\n"
                            "Конец фрагмента (в секундах). 0 = до конца аудио."
                        )
                    }
                ),
                "cut_mode": (
                    ["fast", "precise"],
                    {
                        "default": "fast",
                        "tooltip": (
                            "fast = stream copy (no re-encode, quick, cut at codec-frame boundary).\n"
                            "precise = re-encode (sample-accurate cut).\n"
                            "---\n"
                            "fast = stream copy (без перекодировки, быстро, рез по границе кадра кодека).\n"
                            "precise = перекодирование (точность до сэмпла)."
                        )
                    }
                ),
                "format": (
                    FORMAT_CHOICES,
                    {
                        "default": "source",
                        "tooltip": (
                            "Output format. source = keep the source format (no re-encode in fast mode). "
                            "mp3/wav/flac/ogg/m4a = convert to that format.\n"
                            "NOTE: if the source is connected via the AUDIO socket (not audio_path), the audio "
                            "is already decoded to PCM, so the 'source' format is WAV. To get MP3 etc., choose "
                            "it explicitly (re-encode) or connect the file via audio_path for a true stream copy.\n"
                            "---\n"
                            "Формат выхода. source = сохранить формат источника (без перекодировки в режиме fast). "
                            "mp3/wav/flac/ogg/m4a = конвертировать в этот формат.\n"
                            "ВАЖНО: если источник подключён через сокет AUDIO (а не audio_path), аудио уже "
                            "декодировано в PCM, поэтому форматом 'source' будет WAV. Чтобы получить MP3 и т.д., "
                            "выберите его явно (перекодирование) либо подключите файл через audio_path для "
                            "настоящего stream copy."
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
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional source audio from ComfyUI. Has priority over audio_path.\n"
                            "NOTE: audio via this socket is already decoded to PCM (WAV), so format=source "
                            "will produce WAV. For a true lossless stream copy use audio_path instead.\n"
                            "---\n"
                            "Опциональный исходный звук из ComfyUI. Имеет приоритет над audio_path.\n"
                            "ВАЖНО: аудио через этот сокет уже декодировано в PCM (WAV), поэтому "
                            "format=source даст WAV. Для настоящего stream copy без потерь используйте audio_path."
                        )
                    }
                ),
            },
        }

    @staticmethod
    def _raise_ffmpeg_error(stderr_text):
        stderr_text = stderr_text or ""
        print(f"\n[AGSoft Audio Split Error] FFmpeg log:\n{stderr_text[-4000:]}")
        error_msg = "Ошибка FFmpeg"
        for line in reversed(stderr_text.splitlines()):
            if line.strip():
                error_msg = line.strip()
                break
        raise RuntimeError(f"[AGSoft Audio Split] FFmpeg не смог вырезать фрагмент. Ошибка: {error_msg}")

    def split_audio(
        self,
        output_name, output_path, audio_path,
        start_time, end_time, cut_mode, format, bitrate,
        audio=None
    ):
        output_name = _single(output_name, "audio_split.mp3")
        output_path = _single(output_path, "")
        audio_path = _single(audio_path, "").strip()
        cut_mode = _single(cut_mode, "fast").strip().lower()
        format_choice = _single(format, "source").strip().lower()
        bitrate = _single(bitrate, "192k").strip().lower()

        start_time = max(0.0, safe_float(start_time, 0.0))
        end_time = max(0.0, safe_float(end_time, 0.0))

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_audio_split_")
        try:
            # Источник: вход audio -> audio_path.
            source = None
            source_is_temp = False
            if audio is not None:
                fd, tmp_wav = tempfile.mkstemp(prefix="src_audio_", suffix=".wav", dir=tmp_dir)
                os.close(fd)
                audio_to_wav(audio, tmp_wav)
                source = tmp_wav
                source_is_temp = True
                print(f"[AGSoft Audio Split] Источник из входа audio: {source}")
            if source is None and audio_path:
                if os.path.isfile(audio_path):
                    source = audio_path
                else:
                    print(f"[AGSoft Audio Split] Warning: audio_path не найден ({audio_path}).")
            if source is None:
                raise ValueError("[AGSoft Audio Split] Не задан источник: подключите audio или укажите audio_path.")

            sinfo = parse_audio_info(source)
            duration = float(sinfo["duration"] or 0.0)
            if duration <= 0:
                raise ValueError(f"[AGSoft Audio Split] Не удалось определить длительность: {source}")

            start = start_time
            end = end_time if end_time > 0 else duration
            if start >= end:
                raise ValueError("[AGSoft Audio Split] end_time должен быть больше start_time.")
            end = min(end, duration)
            cut_duration = end - start

            source_ext = os.path.splitext(source)[1].lower()
            target_ext = resolve_target_ext(format_choice, source_ext)
            use_copy = (cut_mode == "fast") and (target_ext == source_ext)

            # Имя и папка выхода.
            safe_name = os.path.basename(str(output_name or "audio_split"))
            name_no_ext, _ = os.path.splitext(safe_name)
            if not name_no_ext:
                name_no_ext = "audio_split"
            safe_name = f"{name_no_ext}{target_ext}"

            output_path_str = "" if output_path is None else str(output_path).strip()
            target_dir = os.path.abspath(output_path_str) if output_path_str else (folder_paths.get_output_directory() or os.path.abspath("."))
            os.makedirs(target_dir, exist_ok=True)
            final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))
            if os.path.isdir(final_output_path):
                raise ValueError(f"[AGSoft Audio Split] Путь назначения — папка, а не файл: {final_output_path}")
            if not source_is_temp and os.path.normcase(os.path.realpath(source)) == os.path.normcase(os.path.realpath(final_output_path)):
                raise ValueError("[AGSoft Audio Split] Итоговый файл не может совпадать с источником.")

            # Команда.
            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-ss", f"{start:.6f}", "-i", source, "-t", f"{cut_duration:.6f}"]
            if use_copy:
                cmd += ["-c:a", "copy"]
            else:
                cmd += codec_args_for(target_ext, bitrate)
            cmd.append(final_output_path)

            result = run_ffmpeg_with_progress(cmd, cut_duration, "[AGSoft Audio Split]")
            if result.returncode != 0:
                self._raise_ffmpeg_error(result.stderr)
            if not os.path.isfile(final_output_path):
                raise RuntimeError(f"[AGSoft Audio Split] Файл не создан: {final_output_path}")

            out_info = parse_audio_info(final_output_path)
            timecode = format_timecode(out_info["duration"])
            comfy_audio = audio_file_to_comfy_audio(final_output_path, tmp_dir)

            print(
                f"[AGSoft Audio Split] Saved: {final_output_path} | mode={cut_mode} | format={target_ext} | "
                f"Duration: {timecode} | Size: {out_info['size_mb']} MB"
            )

            return (
                final_output_path, comfy_audio, out_info["duration"], timecode,
                out_info["size_mb"], out_info["sample_rate"], out_info["channels"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "AGSoftAudioSplit": AGSoftAudioSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAudioSplit": "🔊✂️AGSoft Audio Split"
}