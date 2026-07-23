# ==============================================================================
# AGSoft_Video_Concatenate.py
# ==============================================================================
# Нода:🎬🪡AGSoft Video Concatenate
#
# ОСОБЕННОСТИ:
# ⚡Мгновенное склеивание нескольких видео и аудио роликов без перекодирования (через stream copy в FFmpeg).
# Исходные ролики должны иметь одинаковый формат, кодек и разрешение.
# 
# Автор: AGSoft
# Дата: 22.07.2026
# ==============================================================================

import os
import subprocess
import logging
import re
import tempfile


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


class AGSoftVideoConcatenate:
    OUTPUT_NODE = True

    RETURN_TYPES = (
        "STRING",  # video_path
        "FLOAT",   # duration_seconds
        "STRING",  # duration_timecode
        "FLOAT",   # file_size_mb
        "INT",     # width
        "INT",     # height
        "FLOAT",   # fps
        "INT",     # frames_est
    )

    RETURN_NAMES = (
        "video_path",
        "duration_seconds",
        "duration_timecode",
        "file_size_mb",
        "width",
        "height",
        "fps",
        "frames_est",
    )

    FUNCTION = "concat_videos"
    CATEGORY = "AGSoft/Video"

    DESCRIPTION = (
        "⚡ Fast concatenation of multiple video and audio clips without re-encoding (via FFmpeg stream copy).\n"
        "Source clips must have matching formats, codecs, and resolutions.\n"
        "---\n"
        "⚡ Мгновенное склеивание нескольких видео и аудио роликов без перекодирования (через stream copy в FFmpeg).\n"
        "Исходные ролики должны иметь одинаковый формат, кодек и разрешение."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_name": (
                    "STRING",
                    {
                        "default": "output_video.mp4",
                        "tooltip": (
                            "The filename for the concatenated output video (e.g., result.mp4).\n"
                            "The .mp4 extension will be added automatically if missing.\n"
                            "---\n"
                            "Имя итогового видеофайла (например, result.mp4).\n"
                            "Расширение .mp4 применится автоматически, если оно не указано."
                        )
                    }
                ),
                "inputs_count": (
                    ["2", "3", "4", "5", "6", "7", "8", "9", "10"],
                    {
                        "default": "2",
                        "tooltip": (
                            "Select the total number of video clips you want to concatenate.\n"
                            "---\n"
                            "Выберите общее количество видеороликов, которые необходимо объединить."
                        )
                    }
                ),
            },
            "optional": {
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional: Absolute directory path to save the file (e.g., D:/Videos).\n"
                            "If left empty, the file will be saved in ComfyUI/output directory.\n"
                            "---\n"
                            "Опционально: абсолютный путь к папке сохранения (например, D:/Videos).\n"
                            "Если оставить пустым, сохранится в стандартную папку ComfyUI/output."
                        )
                    }
                ),
            }
        }

    def concat_videos(self, output_name, inputs_count, output_path="", **kwargs):
        if isinstance(output_name, (list, tuple)):
            output_name = output_name[0] if output_name else ""

        if isinstance(output_path, (list, tuple)):
            output_path = output_path[0] if output_path else ""

        video_map = {}

        for key, val in kwargs.items():
            if key.startswith("video_"):
                match = re.search(r"\d+", key)
                if not match:
                    continue

                if isinstance(val, (list, tuple)):
                    val = val[0] if val else None

                if val is not None and str(val).strip():
                    index = int(match.group())
                    video_map[index] = str(val).strip()

        sorted_indices = sorted(video_map.keys())
        video_list = [video_map[idx] for idx in sorted_indices]

        expected = int(inputs_count or 2)

        if len(video_list) < 2:
            raise ValueError("[AGSoft Video Concat] Необходимо подключить как минимум 2 видеофайла!")

        if len(video_list) < expected:
            raise ValueError(
                f"[AGSoft Video Concat] Подключено {len(video_list)} из {expected} выбранных входов."
            )

        missing = [p for p in video_list if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"[AGSoft Video Concat] Не найдены файлы: {', '.join(missing)}"
            )

        safe_name = os.path.basename(str(output_name or "joined_video.mp4"))
        name_without_ext, ext = os.path.splitext(safe_name)

        if not name_without_ext:
            name_without_ext = "joined_video"

        if not ext:
            ext = ".mp4"

        safe_name = f"{name_without_ext}{ext}"

        output_path_str = "" if output_path is None else str(output_path).strip()

        if output_path_str:
            target_dir = os.path.abspath(output_path_str)
        else:
            target_dir = folder_paths.get_output_directory() or os.path.abspath(".")

        os.makedirs(target_dir, exist_ok=True)

        final_output_path = os.path.normpath(os.path.join(target_dir, safe_name))

        if os.path.isdir(final_output_path):
            raise ValueError(
                f"[AGSoft Video Concat] Путь назначения является папкой, а не файлом: {final_output_path}"
            )

        final_abs = os.path.normcase(os.path.realpath(final_output_path))

        for path in video_list:
            if os.path.normcase(os.path.realpath(path)) == final_abs:
                raise ValueError(
                    "[AGSoft Video Concat] Итоговый файл не может быть одним из исходных файлов."
                )

        fd, list_file_path = tempfile.mkstemp(prefix="agsoft_concat_", suffix=".txt")

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for path in video_list:
                    abs_path = os.path.abspath(path).replace("\\", "/")
                    abs_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{abs_path}'\n")

            cmd = [
                FFMPEG_PATH,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
            ]

            if safe_name.lower().endswith((".mp4", ".mov")):
                cmd += ["-movflags", "+faststart"]

            cmd.append(final_output_path)

            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="ignore"
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"[AGSoft Video Concat] FFmpeg не найден: {FFMPEG_PATH}"
                )

            if result.returncode != 0:
                print(f"\n[AGSoft Video Concat Error] Лог FFmpeg:\n{result.stderr}")

                error_msg = "Ошибка FFmpeg"
                for line in reversed((result.stderr or "").splitlines()):
                    if line.strip():
                        error_msg = line.strip()
                        break

                raise RuntimeError(f"FFmpeg не смог склеить ролики. Ошибка: {error_msg}")

        finally:
            if os.path.exists(list_file_path):
                try:
                    os.remove(list_file_path)
                except Exception:
                    pass

        if not os.path.isfile(final_output_path):
            raise RuntimeError(
                f"[AGSoft Video Concat] Файл не был создан: {final_output_path}"
            )

        info = self.get_media_info(final_output_path)

        print(
            f"[AGSoft Video Concat] Saved: {final_output_path} | "
            f"Duration: {info['timecode']} | "
            f"Size: {info['size_mb']} MB"
        )

        return (
            final_output_path,
            info["duration"],
            info["timecode"],
            info["size_mb"],
            info["width"],
            info["height"],
            info["fps"],
            info["frames"],
        )

    @staticmethod
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

    @classmethod
    def get_media_info(cls, path):
        info = {
            "duration": 0.0,
            "timecode": "00:00:00.000",
            "size_mb": 0.0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frames": 0,
        }

        try:
            info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
        except Exception:
            pass

        try:
            p = subprocess.run(
                [FFMPEG_PATH, "-hide_banner", "-i", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            data = p.stderr or ""
        except Exception as e:
            print(f"[AGSoft Video Concat] Не удалось получить метаданные: {e}")
            return info

        duration_match = re.search(
            r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)",
            data
        )

        if duration_match:
            duration = (
                int(duration_match.group(1)) * 3600
                + int(duration_match.group(2)) * 60
                + float(duration_match.group(3))
            )
            info["duration"] = round(duration, 3)
            info["timecode"] = cls.format_timecode(duration)

        for line in data.splitlines():
            if "Video:" not in line:
                continue

            res_match = re.search(r",\s*(\d{2,5})x(\d{2,5})\b", line)
            if res_match:
                info["width"] = int(res_match.group(1))
                info["height"] = int(res_match.group(2))

            fps_value = None

            fps_match = re.search(r"([\d.]+)\s*fps", line)
            if fps_match:
                try:
                    fps_value = float(fps_match.group(1))
                except Exception:
                    fps_value = None

            if fps_value is None:
                tbr_match = re.search(r"([\d.]+)\s*tbr", line)
                if tbr_match:
                    try:
                        tbr_value = float(tbr_match.group(1))
                        if 0 < tbr_value < 1000:
                            fps_value = tbr_value
                    except Exception:
                        fps_value = None

            if fps_value is not None:
                info["fps"] = round(fps_value, 3)

            break

        if info["duration"] > 0 and info["fps"] > 0:
            info["frames"] = int(round(info["duration"] * info["fps"]))

        return info


NODE_CLASS_MAPPINGS = {
    "AGSoftVideoConcatenate": AGSoftVideoConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoConcatenate": "🎬🪡AGSoft Video Concatenate"
}