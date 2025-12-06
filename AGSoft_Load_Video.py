# AGSoft Load Video
# Автор: AGSoft
# Дата: 06.12.2025 г.

import os
import logging
from typing import Dict, Any, Tuple, Optional
import folder_paths
import mimetypes
from comfy_api.input_impl import VideoFromFile
from comfy.comfy_types import IO

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AGSoftLoadVideo:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                full_path = os.path.join(input_dir, f)
                if os.path.isfile(full_path):
                    files.append(f)
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {},
            "optional": {
                "input_video": (IO.VIDEO, {
                    "tooltip":
"""
Optional: Accept video from another node. Has highest priority - if connected, other inputs are ignored.
Example: Connect output from "Load Video" node or "Create Video" node.
Note: Path extraction works automatically for most video sources.

Опционально: Принимает видео от другой ноды. Имеет наивысший приоритет - если подключен, другие входы игнорируются.
Пример: Подключите выход от ноды "Load Video" или "Create Video".
Примечание: Извлечение пути работает автоматически для большинства источников видео.
""" 
                }),
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Enter a custom absolute path to a video file."
"""
Optional: Enter a custom absolute path to a video file. This overrides the file selection if input_video is not connected.
Example (Windows): C:/videos/my_video.mp4
Example (Mac/Linux): /home/user/videos/my_video.mov
Note: Path must be accessible to the ComfyUI process and point to a valid video file.
Warning: Using relative paths may cause errors - always use absolute paths.

Опционально: Введите кастомный абсолютный путь к видео-файлу. Это переопределяет выбор файла, если не подключено внешнее видео.
Пример (Windows): C:/videos/my_video.mp4
Пример (Mac/Linux): /home/user/videos/my_video.mov
Примечание: Путь должен быть доступен для процесса ComfyUI и указывать на корректный видео-файл.
Предупреждение: Использование относительных путей может вызвать ошибки - всегда используйте абсолютные пути.
"""
                }),
                "video_file": (sorted(files), {
                    "tooltip": "Select a video file from your input directory."
"""
Select a video file from your ComfyUI input directory. This option is used when no external video is connected and no custom path is specified.
Supported formats: MP4, AVI, MOV, WEBM, MKV and other common video formats.
Tip: Files must be placed in your ComfyUI/input folder first.

Выберите видео-файл из директории ComfyUI input. Этот вариант используется, когда не подключено внешнее видео и не указан кастомный путь.
Поддерживаемые форматы: MP4, AVI, MOV, WEBM, MKV и другие распространенные видео форматы.
Совет: Файлы должны быть сначала размещены в папке ComfyUI/input.
"""
                }),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "load_video"
    CATEGORY = "AGSoft/Video"
    DESCRIPTION = """
Loads a video file with flexible input options and returns both the video object and its absolute file path.
Priority order:
1. External video input (highest priority)
2. Custom path
3. File from input directory
Returns both the video object for processing and the absolute file path for reference or metadata.

Загружает видео-файл с гибкими вариантами ввода и возвращает как объект видео, так и абсолютный путь к файлу.
Порядок приоритета:
1. Внешний видео-вход (наивысший приоритет)
2. Кастомный путь
3. Файл из директории input
Возвращает как объект видео для дальнейшей обработки, так и абсолютный путь к файлу для справочной информации или метаданных.
"""

    def extract_path_from_video(self, video_obj):
        """Извлекает путь из видео-объекта, учитывая особенности ComfyUI"""
        # Способ 1: Попытка получить путь через приватный атрибут __file (с name mangling)
        try:
            # Используем name mangling для доступа к приватному атрибуту
            if isinstance(video_obj, VideoFromFile):
                path = getattr(video_obj, '_VideoFromFile__file', None)
                if path and isinstance(path, str) and os.path.exists(path):
                    return os.path.abspath(path)
        except Exception:
            pass

        # Способ 2: Проверяем другие возможные атрибуты
        path_attrs = ['path', '_path', 'filepath', 'file_path', 'source', 'filename']
        for attr in path_attrs:
            if hasattr(video_obj, attr):
                value = getattr(video_obj, attr)
                if isinstance(value, str) and os.path.exists(value):
                    return os.path.abspath(value)
        
        # Способ 3: Проверяем метод get_path
        if hasattr(video_obj, 'get_path') and callable(video_obj.get_path):
            path = video_obj.get_path()
            if isinstance(path, str) and os.path.exists(path):
                return os.path.abspath(path)
                
        # Способ 4: Если объект имеет метод get_stream_source
        if hasattr(video_obj, 'get_stream_source') and callable(video_obj.get_stream_source):
            source = video_obj.get_stream_source()
            if isinstance(source, str) and os.path.exists(source):
                return os.path.abspath(source)
                
        return ""

    def load_video(
        self,
        input_video: Optional[object] = None,
        video_file: Optional[str] = None,
        custom_path: Optional[str] = ""
    ) -> Tuple[object, str]:
        try:
            # Priority 1: External video input
            if input_video is not None:
                logger.info("Using external video input")
                video_path = self.extract_path_from_video(input_video)
                if not video_path:
                    # Если не смогли извлечь путь, используем стандартный подход
                    logger.warning("Could not extract file path from external video object")
                    # Пытаемся создать VideoFromFile из внешнего объекта
                    try:
                        video_path = input_video.get_stream_source()
                        if isinstance(video_path, str) and os.path.exists(video_path):
                            video_path = os.path.abspath(video_path)
                        else:
                            video_path = ""
                    except:
                        video_path = ""
                return (input_video, video_path)
            
            # Priority 2 & 3: Custom path or file from input directory
            video_path = ""
            
            if custom_path and os.path.exists(custom_path):
                video_path = os.path.abspath(custom_path)
                logger.info(f"Using custom video path: {video_path}")
            elif video_file:
                base_dir = folder_paths.get_input_directory()
                annotated_path = folder_paths.get_annotated_filepath(video_file)
                if os.path.exists(annotated_path):
                    video_path = annotated_path
                else:
                    video_path = os.path.join(base_dir, video_file)
                video_path = os.path.abspath(video_path)
                logger.info(f"Loading video from: {video_path}")
            else:
                raise ValueError("No valid video source provided. Connect input_video, specify custom_path, or select a video_file.")

            # Проверка существования файла
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Проверка типа файла
            mime_type, _ = mimetypes.guess_type(video_path)
            if mime_type and not mime_type.startswith('video'):
                logger.warning(f"Selected file may not be a video file (MIME: {mime_type}): {video_path}")

            # Создаем видео объект
            video_obj = VideoFromFile(video_path)
            return (video_obj, video_path)
            
        except Exception as e:
            error_msg = f"Error loading video: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @classmethod
    def IS_CHANGED(cls, input_video=None, video_file=None, custom_path=""):
        if input_video is not None:
            # Пытаемся получить путь для отслеживания изменений
            video_path = ""
            try:
                if isinstance(input_video, VideoFromFile):
                    video_path = getattr(input_video, '_VideoFromFile__file', "")
                elif hasattr(input_video, 'get_stream_source'):
                    video_path = input_video.get_stream_source()
                if isinstance(video_path, str) and os.path.exists(video_path):
                    return os.path.getmtime(video_path)
            except:
                pass
            return float("NaN")
            
        try:
            if custom_path and os.path.exists(custom_path):
                return os.path.getmtime(custom_path)
                
            if video_file:
                base_dir = folder_paths.get_input_directory()
                annotated_path = folder_paths.get_annotated_filepath(video_file)
                if os.path.exists(annotated_path):
                    return os.path.getmtime(annotated_path)
                file_path = os.path.join(base_dir, video_file)
                if os.path.exists(file_path):
                    return os.path.getmtime(file_path)
        except Exception as e:
            logger.warning(f"Error checking file change: {str(e)}")
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, input_video=None, video_file=None, custom_path=""):
        if input_video is not None:
            return True
            
        try:
            if custom_path:
                if not os.path.exists(custom_path):
                    return f"Custom path does not exist: {custom_path}"
                if not os.path.isfile(custom_path):
                    return f"Custom path is not a file: {custom_path}"
                mime_type, _ = mimetypes.guess_type(custom_path)
                if mime_type and not mime_type.startswith('video'):
                    return f"Custom path is not a video file: {custom_path}"
                return True
                
            if not video_file:
                return "No video file selected and no custom path provided"
                
            base_dir = folder_paths.get_input_directory()
            file_path = os.path.join(base_dir, video_file)
            annotated_path = folder_paths.get_annotated_filepath(video_file)
            
            # Проверяем оба возможных пути
            path_exists = os.path.exists(file_path) or os.path.exists(annotated_path)
            if not path_exists:
                return f"Video file not found: {video_file}"
                
            # Выбираем существующий путь для проверки
            check_path = annotated_path if os.path.exists(annotated_path) else file_path
            if not os.path.isfile(check_path):
                return f"Path is not a file: {check_path}"
                
            mime_type, _ = mimetypes.guess_type(check_path)
            if mime_type and not mime_type.startswith('video'):
                return f"File is not a video: {video_file}"
                
            return True
        except Exception as e:
            return f"Validation error: {str(e)}"

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "AGSoftLoadVideo": AGSoftLoadVideo
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadVideo": "🎬AGSoft Load Video"
}