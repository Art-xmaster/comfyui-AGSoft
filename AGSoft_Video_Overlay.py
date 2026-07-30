# ==============================================================================
# AGSoft_Video_Overlay.py
# ==============================================================================
# Нода: 🎬AGSoft Video Overlay
#
# Описание / Description:
# Нода наложения текста и watermark (логотипа) поверх ОДНОГО видеоролика.
# Построена на базе AGSoft Video Concatenate Plus, но без склейки и переходов:
# один вход, перекод с оверлеем, те же 8 выходов.
# Overlay node that draws text and a watermark (logo) over a SINGLE video clip.
# Built from AGSoft Video Concatenate Plus but without concatenation/transitions:
# one input, re-encode with overlay, the same 8 outputs.
#
# Возможности / Features:
# ⚡ Один вход video (подключается загрузчик пути или вводится путь вручную).
#    Single video input (connect a path loader or type a path manually).
# ⚡ Текст поверх видео (drawtext): шрифты из папки fonts/ ноды (dropdown),
#    палитра из 24 цветов (dropdown), прозрачность текста (alpha), размер, позиция.
#    Text overlay (drawtext): fonts from the node's fonts/ folder (dropdown),
#    24-color palette (dropdown), text transparency (alpha), size, position.
# ⚡ Watermark (PNG) с позицией, двусторонним масштабом (1.0 = оригинал,
#    <1 уменьшает, >1 увеличивает) и прозрачностью (alpha).
#    Watermark (PNG) with position, two-way scale (1.0 = original,
#    <1 shrinks, >1 enlarges) and transparency (alpha).
# ⚡ Кодировщик NVENC / CPU с авто-fallback; пресеты качества fast/balanced/quality.
#    NVENC / CPU encoder with auto fallback; quality presets fast/balanced/quality.
# ⚡ Аудио не трогается фильтрами (просто перекод aac) — рассинхрон невозможен.
#    Audio is not touched by filters (plain aac re-encode) - desync is impossible.
# ⚡ Прогресс рендера в консоль.
#    Render progress printed to console.
#
# Автор / Author: AGSoft
# Дата / Date: 30.07.2026
# ==============================================================================

import os
import re
import json
import logging
import subprocess
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


FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def scan_fonts():
    fonts = []
    if not os.path.isdir(FONTS_DIR):
        return fonts
    for fname in sorted(os.listdir(FONTS_DIR)):
        if fname.lower().endswith((".ttf", ".otf", ".ttc")):
            fonts.append(fname)
    return fonts


def get_font_path(font_name):
    if not font_name or font_name == "(none)":
        return None
    path = os.path.join(FONTS_DIR, font_name)
    return path if os.path.isfile(path) else None


FONT_COLORS = [
    "white", "black", "red", "green", "blue", "yellow",
    "cyan", "magenta", "orange", "purple", "pink", "brown",
    "gray", "lime", "navy", "teal", "maroon", "olive",
    "silver", "gold", "coral", "crimson", "turquoise", "violet",
]


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


def safe_int(value, default=0):
    try:
        return int(default) if value is None else int(float(value))
    except Exception:
        return int(default)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _escape_level(value, specials):
    out = []
    for ch in str(value):
        if ch == "\\" or ch in specials:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def ff_escape_value(value):
    s = _escape_level(value, set(":='"))
    s = _escape_level(s, set("'[],; "))
    return s


def sanitize_color(value, default="white"):
    s = re.sub(r"[^a-zA-Z0-9#._@]", "", str(value or default))
    return s or default


def ffprobe_duration(path):
    try:
        cmd = [
            FFPROBE_PATH, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path
        ]
        result = subprocess.run(
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
        result = subprocess.run(
            [FFMPEG_PATH, "-hide_banner", "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
        data = result.stderr or ""
    except Exception as e:
        print(f"[AGSoft Video Overlay] Не удалось прочитать медиафайл: {path}")
        print(f"[AGSoft Video Overlay] Ошибка: {e}")
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


def supports_encoder(encoder):
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    ok = False
    try:
        result = subprocess.run(
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
        raise RuntimeError("[AGSoft Video Overlay] h264_nvenc недоступен. Выберите cpu или auto.")
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
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore"
        )
    except FileNotFoundError:
        raise RuntimeError(f"[AGSoft Video Overlay] FFmpeg не найден: {FFMPEG_PATH}")

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
                    print(f"[AGSoft Video Overlay] Render progress: {bucket * 5}%")
                    last_bucket = bucket

    process.wait()
    return subprocess.CompletedProcess(
        args=cmd, returncode=process.returncode,
        stdout="", stderr="".join(stderr_chunks)
    )


def build_overlay_filter(
    overlay_text, font_path, font_color, font_alpha, font_size, text_x, text_y,
    watermark_index, watermark_x, watermark_y, watermark_scale, watermark_alpha
):
    parts = []
    logs = []

    overlay_text = str(overlay_text or "").replace("\r", " ").replace("\n", " ").strip()
    has_text = bool(overlay_text and font_path)
    has_wm = watermark_index is not None

    stages = []
    if has_text:
        stages.append("text")
    if has_wm:
        stages.append("wm")

    if not stages:
        logs.append("Ни текст, ни watermark не заданы — выполнен проход без оверлея.")

    cur = "[0:v]"
    base_out = "[vbase]" if stages else "[outv]"
    parts.append(f"{cur}setsar=1,format=yuv420p{base_out}")
    cur = base_out

    for idx, st in enumerate(stages):
        is_last = (idx == len(stages) - 1)
        out = "[outv]" if is_last else f"[vs{idx}]"

        if st == "text":
            escaped_font = ff_escape_value(str(font_path).replace("\\", "/"))
            escaped_text = ff_escape_value(" ".join(overlay_text.split()))

            if font_alpha < 0.999:
                color_spec = f"{font_color}@{font_alpha:.2f}"
            else:
                color_spec = font_color

            drawtext = (
                f"drawtext=fontfile={escaped_font}:text={escaped_text}:"
                f"fontcolor={ff_escape_value(color_spec)}:fontsize={font_size}:"
                f"x={ff_escape_value(text_x)}:y={ff_escape_value(text_y)}"
            )
            parts.append(f"{cur}{drawtext}{out}")
            logs.append(f"Текст наложен: шрифт='{os.path.basename(font_path)}', цвет={color_spec}, размер={font_size}.")

        else:
            wm_alpha_part = f",colorchannelmixer=aa={watermark_alpha:.2f}" if watermark_alpha < 0.999 else ""

            parts.append(
                f"[{watermark_index}:v]scale=iw*{watermark_scale:.6f}:-2:flags=lanczos,"
                f"format=rgba{wm_alpha_part}[wmimg]"
            )
            parts.append(
                f"{cur}[wmimg]overlay=x={ff_escape_value(watermark_x)}:"
                f"y={ff_escape_value(watermark_y)}:eof_action=repeat{out}"
            )
            logs.append(f"Watermark наложен: масштаб={watermark_scale:.2f}, alpha={watermark_alpha:.2f}.")

        cur = out

    return ";".join(parts), logs


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


class AGSoftVideoOverlay:
    OUTPUT_NODE = True

    RETURN_TYPES = (
        "STRING", "FLOAT", "STRING", "FLOAT",
        "INT", "INT", "FLOAT", "INT",
    )
    RETURN_NAMES = (
        "video_path", "duration_seconds", "duration_timecode", "file_size_mb",
        "width", "height", "fps", "frames_est",
    )

    FUNCTION = "overlay_video"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "🎬 AGSoft Video Overlay.\n"
        "Draws text and/or a watermark (logo) over a SINGLE video clip and re-encodes it.\n"
        "Text uses fonts from the node's fonts/ folder (dropdown), a 24-color palette "
        "and a transparency slider; the watermark has a two-way size multiplier "
        "(1.0 = original, <1 shrinks, >1 enlarges) and a transparency slider.\n"
        "Audio is passed through (aac re-encode, no filters) so desync is impossible.\n"
        "---\n"
        "🎬 AGSoft Video Overlay.\n"
        "Наложение текста и/или watermark (логотипа) поверх ОДНОГО видеоклипа с перекодом.\n"
        "Текст использует шрифты из папки fonts/ ноды (список), палитру из 24 цветов "
        "и слайдер прозрачности; у watermark есть двусторонний множитель размера "
        "(1.0 = оригинал, <1 уменьшает, >1 увеличивает) и слайдер прозрачности.\n"
        "Звук проходит насквозь (перекод aac без фильтров) — рассинхрон невозможен."
    )

    @classmethod
    def INPUT_TYPES(cls):
        font_list = ["(none)"] + scan_fonts()

        return {
            "required": {
                "video": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Input video path. Connect a path loader (the widget auto-converts "
                            "to an input socket on link drop) or type an absolute path manually.\n"
                            "---\n"
                            "Путь входного видео. Подключите загрузчик пути (виджет сам станет "
                            "сокетом при дропе линка) или введите абсолютный путь вручную."
                        )
                    }
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": "overlay_video.mp4",
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
                "overlay_text": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Text to draw over the video. Requires a font other than (none).\n"
                            "---\n"
                            "Текст поверх видео. Нужен шрифт, отличный от (none)."
                        )
                    }
                ),
                "font_name": (
                    font_list,
                    {
                        "default": font_list[0],
                        "tooltip": (
                            "Font for the text overlay, taken from the node's fonts/ folder.\n"
                            "(none) = no text is drawn. Add a .ttf/.otf to fonts/ and recreate the node to refresh.\n"
                            "---\n"
                            "Шрифт для текста из папки fonts/ ноды.\n"
                            "(none) = текст не рисуется. Добавьте .ttf/.otf в fonts/ и пересоздайте ноду для обновления."
                        )
                    }
                ),
                "font_color": (
                    FONT_COLORS,
                    {
                        "default": "white",
                        "tooltip": (
                            "Text color from a preset palette (FFmpeg/X11 color names).\n"
                            "---\n"
                            "Цвет текста из готовой палитры (имена цветов FFmpeg/X11)."
                        )
                    }
                ),
                "font_alpha": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Text transparency. 1.0 = fully opaque, 0.0 = fully transparent.\n"
                            "---\n"
                            "Прозрачность текста. 1.0 = полностью непрозрачно, 0.0 = полностью прозрачно."
                        )
                    }
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 48,
                        "min": 4,
                        "max": 500,
                        "step": 1,
                        "tooltip": (
                            "Text font size in pixels.\n"
                            "---\n"
                            "Размер шрифта текста в пикселях."
                        )
                    }
                ),
                "text_x": (
                    "STRING",
                    {
                        "default": "20",
                        "tooltip": (
                            "Drawtext X expression. Examples: 20, (W-tw)/2, W-tw-20.\n"
                            "---\n"
                            "Выражение координаты X текста. Примеры: 20, (W-tw)/2, W-tw-20."
                        )
                    }
                ),
                "text_y": (
                    "STRING",
                    {
                        "default": "H-th-20",
                        "tooltip": (
                            "Drawtext Y expression. Examples: 20, (H-th)/2, H-th-20.\n"
                            "---\n"
                            "Выражение координаты Y текста. Примеры: 20, (H-th)/2, H-th-20."
                        )
                    }
                ),
                "watermark_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional path to a watermark image (PNG with transparency recommended).\n"
                            "---\n"
                            "Опционально: путь к изображению watermark (рекомендуется PNG с прозрачностью)."
                        )
                    }
                ),
                "watermark_x": (
                    "STRING",
                    {
                        "default": "W-w-20",
                        "tooltip": (
                            "Overlay X expression. Example: W-w-20 (top-right with margin).\n"
                            "---\n"
                            "Выражение координаты X. Пример: W-w-20 (справа вверху с отступом)."
                        )
                    }
                ),
                "watermark_y": (
                    "STRING",
                    {
                        "default": "20",
                        "tooltip": (
                            "Overlay Y expression. Example: 20.\n"
                            "---\n"
                            "Выражение координаты Y. Пример: 20."
                        )
                    }
                ),
                "watermark_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.05,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": (
                            "Watermark size multiplier. 1.0 = original size (100%).\n"
                            "Below 1.0 shrinks it (e.g. 0.5 = 50%), above 1.0 enlarges it (e.g. 2.0 = 200%).\n"
                            "---\n"
                            "Множитель размера watermark. 1.0 = исходный размер (100%).\n"
                            "Меньше 1.0 — уменьшение (например 0.5 = 50%), больше 1.0 — увеличение (например 2.0 = 200%)."
                        )
                    }
                ),
                "watermark_alpha": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Watermark transparency. 1.0 = fully opaque, 0.0 = fully transparent.\n"
                            "---\n"
                            "Прозрачность watermark. 1.0 = полностью непрозрачно, 0.0 = полностью прозрачно."
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
            },

            "optional": {},
        }

    def overlay_video(
        self,
        video,
        output_name,
        overlay_text, font_name, font_color, font_alpha, font_size, text_x, text_y,
        watermark_path, watermark_x, watermark_y, watermark_scale, watermark_alpha,
        encoder_mode, quality_preset,
        output_path="",
        **kwargs
    ):
        video = _single(video, "")
        output_name = _single(output_name, "overlay_video.mp4")
        output_path = _single(output_path, "")
        overlay_text = _single(overlay_text, "")
        font_name = _single(font_name, "(none)")
        font_color = _single(font_color, "white")
        text_x = _single(text_x, "20")
        text_y = _single(text_y, "H-th-20")
        watermark_path = _single(watermark_path, "")
        watermark_x = _single(watermark_x, "W-w-20")
        watermark_y = _single(watermark_y, "20")
        encoder_mode = _single(encoder_mode, "auto")
        quality_preset = _single(quality_preset, "balanced")

        font_color = sanitize_color(font_color, "white")
        font_alpha = clamp(safe_float(font_alpha, 1.0), 0.0, 1.0)
        watermark_alpha = clamp(safe_float(watermark_alpha, 1.0), 0.0, 1.0)
        font_size = clamp(safe_int(font_size, 48), 4, 500)
        watermark_scale = clamp(safe_float(watermark_scale, 1.0), 0.05, 10.0)

        video_path = str(video or "").strip()
        if not video_path:
            raise ValueError("[AGSoft Video Overlay] Не подключён входной видеофайл (виджет video пуст).")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"[AGSoft Video Overlay] Входной файл не найден: {video_path}")

        safe_name = os.path.basename(str(output_name or "overlay_video.mp4"))
        name_without_ext, ext = os.path.splitext(safe_name)
        if not name_without_ext:
            name_without_ext = "overlay_video"
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
            raise ValueError(f"[AGSoft Video Overlay] Путь назначения — папка, а не файл: {final_output_path}")

        final_abs = os.path.normcase(os.path.realpath(final_output_path))
        if os.path.normcase(os.path.realpath(video_path)) == final_abs:
            raise ValueError("[AGSoft Video Overlay] Итоговый файл не может быть исходным файлом.")

        in_info = parse_media_info(video_path)
        if in_info["width"] <= 0 or in_info["height"] <= 0:
            raise ValueError(f"[AGSoft Video Overlay] Не удалось найти видеопоток в: {video_path}")

        watermark_index = None
        if watermark_path and os.path.isfile(watermark_path):
            watermark_index = 1
        elif watermark_path:
            print(f"[AGSoft Video Overlay] Warning: watermark не найден: {watermark_path}")

        font_path = get_font_path(font_name)
        overlay_text = str(overlay_text or "").strip()
        if overlay_text and not font_path and font_name != "(none)":
            print(f"[AGSoft Video Overlay] Warning: шрифт '{font_name}' не найден в папке fonts/.")
        if overlay_text and not font_path:
            print("[AGSoft Video Overlay] Warning: текст задан, но шрифт не выбран/не найден — текст пропущен.")

        filter_complex, log_lines = build_overlay_filter(
            overlay_text=overlay_text, font_path=font_path,
            font_color=font_color, font_alpha=font_alpha,
            font_size=font_size, text_x=text_x, text_y=text_y,
            watermark_index=watermark_index,
            watermark_x=watermark_x, watermark_y=watermark_y,
            watermark_scale=watermark_scale, watermark_alpha=watermark_alpha
        )

        for line in log_lines:
            print(f"[AGSoft Video Overlay] {line}")

        encoder = choose_encoder(encoder_mode)

        input_args = ["-i", video_path]
        if watermark_index is not None:
            input_args.extend(["-i", watermark_path])

        base_cmd = [
            FFMPEG_PATH, "-y", "-hide_banner",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
        ]

        audio_args = ["-c:a", "aac", "-b:a", "192k"]
        mux_args = ["-movflags", "+faststart"] if safe_name.lower().endswith((".mp4", ".mov")) else []

        cmd = base_cmd + build_encoder_args(encoder, quality_preset) + audio_args + mux_args + [final_output_path]
        result = run_ffmpeg_with_progress(cmd, in_info["duration"])

        if result.returncode != 0 and encoder == "h264_nvenc" and str(encoder_mode).strip().lower() == "auto":
            print("[AGSoft Video Overlay] NVENC failed. Falling back to CPU libx264.")
            cmd = base_cmd + build_encoder_args("libx264", quality_preset) + audio_args + mux_args + [final_output_path]
            result = run_ffmpeg_with_progress(cmd, in_info["duration"])

        if result.returncode != 0:
            stderr_text = result.stderr or ""
            print(f"\n[AGSoft Video Overlay Error] FFmpeg log:\n{stderr_text[-4000:]}")
            error_msg = "Ошибка FFmpeg"
            for line in reversed(stderr_text.splitlines()):
                if line.strip():
                    error_msg = line.strip()
                    break
            raise RuntimeError(f"[AGSoft Video Overlay] FFmpeg не смог отрендерить видео. Ошибка: {error_msg}")

        if not os.path.isfile(final_output_path):
            raise RuntimeError(f"[AGSoft Video Overlay] Файл не создан: {final_output_path}")

        out_info = parse_media_info(final_output_path)
        timecode = format_timecode(out_info["duration"])

        print(
            f"[AGSoft Video Overlay] Saved: {final_output_path} | "
            f"Duration: {timecode} | Size: {out_info['size_mb']} MB | "
            f"Resolution: {out_info['width']}x{out_info['height']} | FPS: {out_info['fps']}"
        )

        return (
            final_output_path, out_info["duration"], timecode, out_info["size_mb"],
            out_info["width"], out_info["height"], out_info["fps"], out_info["frames"],
        )


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoOverlay": AGSoftVideoOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoOverlay": "🎬AGSoft Video Overlay"
}