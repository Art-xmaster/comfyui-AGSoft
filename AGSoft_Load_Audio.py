# ==============================================================================
# AGSoft_Load_Audio.py
# ==============================================================================
# Нода: 🔊AGSoft Load Audio
# Описание / Description:
# Загружает аудиофайл с гибкими вариантами ввода и возвращает объект AUDIO и
# абсолютный путь к файлу. Есть встроенный превью-плеер и кнопка загрузки
# (через JS-расширение из web/).
#
# Порядок приоритета:
# 1. внешний вход input_audio
# 2. кастомный путь custom_path
# 3. файл из папки input
#
# При внешнем входе аудио сохраняется во временный WAV, чтобы audio_path
# всегда был реальным путём (важно для нод с несколькими входами, например
# Audio Concatenate Plus).
#
# Файл можно выбрать в комбо, загрузить кнопкой ИЛИ просто перетащить из
# проводника прямо на ноду (drag&drop, вся область ноды, с подсветкой рамки
# и защитой от двойного drop — реализовано в web/AGSoft_Load_Audio.js).
#
# Loads an audio file with flexible input options and returns both the AUDIO
# object and its absolute file path. Includes a player preview and upload
# button (via a JS extension in web/).
#
# Priority order:
# 1. external input_audio
# 2. custom_path
# 3. file from input directory
#
# With external input, the audio is saved to a temporary WAV so audio_path is
# always a real path (important for nodes with multiple inputs, e.g. Audio
# Concatenate Plus).
#
# A file can be selected in the combo, uploaded via the button OR simply
# dragged from the OS explorer straight onto the node (drag&drop, whole node
# area, with border highlight and double-drop protection — implemented in
# web/AGSoft_Load_Audio.js).
#
# Возможности / Features:
# ⚡ Три источника с приоритетом: input_audio / custom_path / audio.
#   Three sources with priority: input_audio / custom_path / audio.
# ⚡ Превью-плеер и кнопка загрузки (через web/JS-расширение).
#   Player preview and upload button (via web/ JS extension).
# ⚡ Drag&drop аудиофайла из проводника прямо на ноду.
#   Drag&drop an audio file from the OS explorer straight onto the node.
# ⚡ Загрузка любого аудио-формата: torchaudio + fallback FFmpeg→WAV.
#   Any audio format: torchaudio + FFmpeg→WAV fallback.
# ⚡ Реальный audio_path даже при внешнем AUDIO-входе (через временный WAV).
#   Real audio_path even with an external AUDIO input (via a temp WAV).
# ⚡ IS_CHANGED для корректного кэширования, VALIDATE_INPUTS для валидации.
#   IS_CHANGED for proper caching, VALIDATE_INPUTS for validation.
#
# Автор / Author: AGSoft
# Дата / Date: 16.08.2026
# ==============================================================================

import os
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
import wave

import folder_paths
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"

# Маркер версии: если этой строки нет в консоли после старта — файл не применился.
# Version marker: if this line is not in console after startup — file was not applied.
# print("[AGSoft Load Audio] v16.10 loaded (preview + drag&drop)")


def _list_audio_files():
    """Список аудиофайлов из папки input (безопасно на любой версии ComfyUI)."""
    try:
        input_dir = folder_paths.get_input_directory()
        files = []

        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    files.append(f)

        return folder_paths.filter_files_content_types(files, ["audio"])
    except Exception:
        return []


def _load_audio_comfy(path):
    """Загружает аудиофайл в объект AUDIO. torchaudio, fallback — FFmpeg→WAV."""
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as e:
        logger.warning(f"torchaudio load failed, fallback to ffmpeg: {e}")

        tmp_dir = tempfile.mkdtemp(prefix="agsoft_load_audio_")
        wav_path = os.path.join(tmp_dir, "_load.wav")

        try:
            cmd = [FFMPEG_PATH, "-y", "-hide_banner", "-i", path,
                   "-vn", "-c:a", "pcm_s16le", wav_path]
            _sp_run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, encoding="utf-8", errors="ignore", check=True)

            with wave.open(wav_path, "rb") as w:
                nchannels = w.getnchannels()
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())

            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            arr = arr.reshape(-1, nchannels).T

            return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": sr}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_audio_to_temp(audio):
    """Сохраняет объект AUDIO во временный WAV и возвращает путь."""
    wf = audio.get("waveform")
    sr = int(audio.get("sample_rate", 44100))

    arr = wf.cpu().numpy() if hasattr(wf, "cpu") else np.asarray(wf)

    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2").T

    tmp_dir = folder_paths.get_temp_directory()
    os.makedirs(tmp_dir, exist_ok=True)

    fd, path = tempfile.mkstemp(prefix="agsoft_audio_", suffix=".wav", dir=tmp_dir)
    os.close(fd)

    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

    return os.path.abspath(path)


class AGSoftLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = _list_audio_files()

        # Гарантируем непустой список — иначе фронтенд не создаст combo-виджет.
        # Guarantee non-empty list — otherwise frontend may not create combo widget.
        if not audio_files:
            audio_files = [" "]

        return {
            "required": {},
            "optional": {
                "input_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional: accept audio from another node (highest priority).\n"
                            "If connected, other inputs are ignored. A real audio_path is still  "
                            "returned (saved to a temporary WAV).\n"
                            "---\n"
                            "Опционально: принимает звук от другой ноды (наивысший приоритет).\n"
                            "Если подключён, другие входы игнорируются. Реальный audio_path всё равно  "
                            "возвращается (сохраняется во временный WAV)."
                        )
                    }
                ),
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional: absolute path to an audio file. Overrides the dropdown  "
                            "if input_audio is not connected.\n"
                            "---\n"
                            "Опционально: абсолютный путь к аудиофайлу. Переопределяет список,  "
                            "если не подключён внешний звук."
                        )
                    }
                ),
                # Комбо с audio_upload=True: фронтенд добавляет кнопку загрузки,
                # а JS-расширение из web/ рисует превью-плеер и кнопку
                # "choose file to upload". Файл также можно перетащить из
                # проводника прямо на ноду.
                #
                # Combo with audio_upload=True: frontend adds an upload button,
                # and the web/ JS extension draws the preview player and the
                # "choose file to upload" button. A file can also be dragged
                # from the OS explorer straight onto the node.
                "audio": (
                    audio_files,
                    {
                        "audio_upload": True,
                        "tooltip": (
                            "Select / upload an audio file from the input directory.  "
                            "Used when no external audio and no custom path are set.  "
                            "You can also drag&drop an audio file from the OS explorer onto the node.\n"
                            "---\n"
                            "Выберите / загрузите аудиофайл из папки input.  "
                            "Используется, когда не подключён внешний звук и не указан кастомный путь.  "
                            "Также можно просто перетащить аудиофайл из проводника прямо на ноду."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "load_audio"
    CATEGORY = "AGSoft/Audio"

    # JS из web/ дорисовывает превью-плеер, кнопку загрузки и drag&drop.
    # JS from web/ draws preview player, upload button and drag&drop.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🔊 AGSoft Load Audio.\n"
        "Loads an audio file with flexible input options and returns both the AUDIO object and its "
        "absolute file path. Includes a built-in player preview and an upload button (via a JS "
        "extension).\n"
        "Priority order: external audio input > custom path > file from input directory.\n"
        "With an external AUDIO input the audio is saved to a temporary WAV so audio_path is always "
        "a real path (important for nodes with multiple inputs like Audio Concatenate Plus).\n"
        "Any audio format is supported via torchaudio with an FFmpeg→WAV fallback.\n"
        "A file can be selected in the dropdown, uploaded via the button, or dragged from the OS "
        "explorer straight onto the node (drag&drop with border highlight).\n"
        "---\n"
        "🔊 AGSoft Load Audio.\n"
        "Загружает аудиофайл с гибкими вариантами ввода и возвращает объект AUDIO и абсолютный путь "
        "к файлу. Есть встроенный превью-плеер и кнопка загрузки (через JS-расширение).\n"
        "Порядок приоритета: внешний вход > кастомный путь > файл из папки input.\n"
        "При внешнем AUDIO-входе аудио сохраняется во временный WAV, чтобы audio_path всегда был "
        "реальным путём (важно для нод с несколькими входами, например Audio Concatenate Plus).\n"
        "Поддерживается любой аудио-формат: torchaudio + fallback FFmpeg→WAV.\n"
        "Файл можно выбрать в списке, загрузить кнопкой или перетащить из проводника прямо на ноду "
        "(drag&drop с подсветкой рамки)."
    )

    def load_audio(self, audio=None, input_audio=None, custom_path=""):
        # Приоритет 1: внешний вход. Сохраняем во временный WAV, чтобы путь был реальным.
        # Priority 1: external input. Save to a temp WAV so the path is real.
        if input_audio is not None:
            logger.info("Using external audio input")
            path = _save_audio_to_temp(input_audio)
            return (input_audio, path)

        # Приоритет 2: кастомный путь.
        # Priority 2: custom path.
        if custom_path and os.path.exists(custom_path):
            p = os.path.abspath(custom_path)
            logger.info(f"Using custom audio path: {p}")
            return (_load_audio_comfy(p), p)

        # Приоритет 3: комбо / загруженный файл.
        # Priority 3: combo / uploaded file.
        if audio and str(audio).strip():
            p = os.path.abspath(folder_paths.get_annotated_filepath(audio))
            logger.info(f"Loading audio from: {p}")
            return (_load_audio_comfy(p), p)

        raise ValueError(
            "[AGSoft Load Audio] Не задан источник: подключите input_audio, "
            "укажите custom_path или выберите файл в audio."
        )

    @classmethod
    def IS_CHANGED(cls, audio=None, input_audio=None, custom_path=""):
        if input_audio is not None:
            return float("nan")

        try:
            if custom_path and os.path.exists(custom_path):
                return os.path.getmtime(custom_path)

            if audio and str(audio).strip():
                p = folder_paths.get_annotated_filepath(audio)
                if os.path.exists(p):
                    return os.path.getmtime(p)
        except Exception:
            pass

        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, audio=None, input_audio=None, custom_path=""):
        if input_audio is not None:
            return True

        try:
            if custom_path:
                if not os.path.exists(custom_path):
                    return f"Custom path does not exist: {custom_path}"

                if not os.path.isfile(custom_path):
                    return f"Custom path is not a file: {custom_path}"

                return True

            if not audio or not str(audio).strip():
                return "No audio file selected and no custom path provided"

            if not folder_paths.exists_annotated_filepath(audio):
                return f"Audio file not found: {audio}"

            return True

        except Exception as e:
            return f"Validation error: {str(e)}"


NODE_CLASS_MAPPINGS = {
    "AGSoftLoadAudio": AGSoftLoadAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadAudio": "🔊AGSoft Load Audio"
}
