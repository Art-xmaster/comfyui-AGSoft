"""
# AGSoft VideoFrameExtractor
# Автор: AGSoft
# Дата: 10.05.2026 г.
Извлекает кадры из видеофайла с гибкими настройками выбора.
Поддерживает различные режимы выборки кадров и предоставляет подробную информацию о видео.
"""

# Импорты стандартных библиотек Python
import os
import json
import logging
from typing import Tuple, List, Dict, Any, Optional

# Импорты из ComfyUI
import folder_paths
import torch
import numpy as np

# Импорт OpenCV для работы с видео
try:
    import cv2
except ImportError:
    raise ImportError("OpenCV (cv2) is required for this node. Please install it with 'pip install opencv-python'.")

# Настройка логгера
logger = logging.getLogger(__name__)


class AGSoftVideoFrameExtractor:
    """
    Основной класс ноды для извлечения кадров из видео.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Определяет входные параметры ноды.
        """
        return {
            "required": {
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Absolute or relative path to the video file. "
                            "Has the lowest priority when video or video_frames are connected.\n"
                            "Абсолютный или относительный путь к видеофайлу. "
                            "Имеет наименьший приоритет, если подключены video или video_frames."
                        )
                    },
                ),
                "frame_selection_mode": (
                    ["first", "last", "exact_frame", "range", "sample", "all"],
                    {
                        "default": "first",
                        "tooltip": """
first: Extracts the very first frame.
last: Extracts the very last frame (robust method).
exact_frame: Extracts a single frame by its number.
range: Extracts frames from 'start_frame' to 'end_frame' with 'step'.
sample: Extracts 'num_frames' evenly spaced frames.
all: Extracts all frames with the given 'step'.

first: Извлекает самый первый кадр.
last: Извлекает самый последний кадр (надежным методом).
exact_frame: Извлекает один кадр по его номеру.
range: Извлекает кадры от 'start_frame' до 'end_frame' с шагом 'step'.
sample: Извлекает 'num_frames' равномерно распределенных кадров.
all: Извлекает все кадры с заданным шагом 'step'.
"""
                    },
                ),
            },
            "optional": {
                "video": (
                    "VIDEO",
                    {
                        "default": None,
                        "tooltip": """
Video object from Video Helper Suite (VHS) Load Video node or standard ComfyUI Load Video.
Has the highest priority. If connected, video_frames and custom_path are ignored.
Видеообъект от ноды Load Video из Video Helper Suite (VHS) или стандартного ComfyUI.
Имеет наивысший приоритет. При подключении video_frames и custom_path игнорируются.
"""
                    },
                ),
                "video_frames": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": """
Pre-loaded video frames as tensor (e.g., from VHS or LoadVideo node). 
Has medium priority. Used only if 'video' is not connected.
Предварительно загруженные кадры видео в виде тензора (например, из VHS или LoadVideo).
Имеет средний приоритет. Используется только если 'video' не подключен.
"""
                    },
                ),
                "exact_frame": (
                    "INT",
                    {
                        "default": 1, "min": 1, "max": 10000000,
                        "tooltip": """
Frame number to extract (1-based index). Used only in 'exact_frame' mode.
Номер кадра для извлечения (нумерация с 1). Используется только в режиме 'exact_frame'.
"""
                    },
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 1, "min": 1, "max": 10000000,
                        "tooltip": """
Starting frame number for 'range' mode (1-based index). Used only in 'range' mode.
Начальный номер кадра для режима 'range' (нумерация с 1). Используется только в режиме 'range'.
"""
                    },
                ),
                "end_frame": (
                    "INT",
                    {
                        "default": 100, "min": 1, "max": 10000000,
                        "tooltip": """
Ending frame number for 'range' mode (1-based index). Used only in 'range' mode.
Конечный номер кадра для режима 'range' (нумерация с 1). Используется только в режиме 'range'.
"""
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 10, "min": 1, "max": 10000000,
                        "tooltip": """
Number of frames to extract in 'sample' mode. Used only in 'sample' mode to determine how many frames to evenly distribute across the video.
Количество кадров для извлечения в режиме 'sample'. Используется только в режиме 'sample' для определения количества кадров, равномерно распределенных по видео.
"""
                    },
                ),
                "step": (
                    "INT",
                    {
                        "default": 10, "min": 1, "max": 1000000,
                        "tooltip": """
Extract every Nth frame. For example, step=2 will extract every second frame, reducing the total number of frames by half. 
Used only in 'range' and 'all' modes.
Извлекать каждый N-й кадр. Например, step=2 будет извлекать каждый второй кадр, уменьшая общее количество кадров вдвое. 
Используется только в режимах 'range' и 'all'.
"""
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame", "width", "height", "total_frames", "video_metadata_json")
    FUNCTION = "extract_frames"
    CATEGORY = "AGSoft/Video"
    DESCRIPTION = """
Extracts frames from a video file with flexible selection options. Supports various modes including extracting the first/last frame, specific frames, frame ranges, or all frames with customizable step and maximum frame limits. Returns the extracted frames as images along with video metadata.

Извлекает кадры из видеофайла с гибкими опциями выбора. Поддерживает различные режимы, включая извлечение первого/последнего кадра, конкретных кадров, диапазонов кадров или всех кадров с настраиваемым шагом и ограничением максимального количества кадров. Возвращает извлеченные кадры как изображения вместе с метаданными видео.
"""

    def extract_frames(
        self,
        custom_path: str,
        frame_selection_mode: str,
        video: Optional[Any] = None,
        video_frames: Optional[torch.Tensor] = None,
        exact_frame: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        num_frames: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """
        Основной метод извлечения кадров.
        Приоритет: video > video_frames > custom_path
        """
        
        # Приоритет 1: video (объект от VHS или стандартного Load Video)
        if video is not None:
            logger.info("Using video object from 'video' input")
            return self._extract_from_video_object(
                video_obj=video,
                frame_selection_mode=frame_selection_mode,
                exact_frame=exact_frame,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames,
                step=step,
            )
        
        # Приоритет 2: video_frames (уже загруженные кадры)
        if video_frames is not None:
            logger.info("Using pre-loaded video_frames tensor")
            return self._extract_from_tensor(
                video_frames=video_frames,
                frame_selection_mode=frame_selection_mode,
                exact_frame=exact_frame,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames,
                step=step,
            )
        
        # Приоритет 3: custom_path (путь к файлу)
        if not custom_path or not custom_path.strip():
            raise ValueError("No video source provided. Please connect either 'video', 'video_frames', or provide a 'custom_path'.")
        
        video_path = os.path.abspath(custom_path.strip())
        logger.info(f"Using video path from 'custom_path' input: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")
        
        return self._extract_from_file(
            video_path=video_path,
            frame_selection_mode=frame_selection_mode,
            exact_frame=exact_frame,
            start_frame=start_frame,
            end_frame=end_frame,
            num_frames=num_frames,
            step=step,
        )
    
    def _extract_from_video_object(
        self,
        video_obj: Any,
        frame_selection_mode: str,
        exact_frame: Optional[int],
        start_frame: Optional[int],
        end_frame: Optional[int],
        num_frames: Optional[int],
        step: Optional[int],
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """
        Извлекает кадры из видеообъекта, совместимого с Video Helper Suite.
        Поддерживает как VHS объекты, так и стандартные VideoFromFile.
        """
        
        # --- Способ 1: VHS объект (имеет атрибуты и методы VHS) ---
        # VHS Load Video возвращает кортеж: (images, frame_count, ...)
        # Но если подключить VIDEO выход, то получаем объект с методами
        
        # Проверяем, является ли объект словарём/кортежем от VHS
        if isinstance(video_obj, (tuple, list)) and len(video_obj) >= 2:
            # VHS Load Video возвращает (images, frame_count, width, height, ...)
            if isinstance(video_obj[0], torch.Tensor):
                logger.info("Detected VHS tuple output format")
                images_tensor = video_obj[0]  # уже тензор изображений
                frame_count = video_obj[1] if len(video_obj) > 1 else images_tensor.shape[0]
                height, width = images_tensor.shape[1], images_tensor.shape[2]
                
                # Расчёт индексов
                indices = self._calculate_frame_indices(
                    frame_selection_mode, frame_count,
                    exact_frame, start_frame, end_frame, num_frames, step
                )
                
                # Извлекаем нужные кадры
                selected_frames = images_tensor[indices]
                
                metadata = self._create_metadata_json(
                    "[from VHS object]", width, height, frame_count, 0,
                    frame_selection_mode, indices,
                    {"exact_frame": exact_frame, "start_frame": start_frame,
                     "end_frame": end_frame, "num_frames": num_frames, "step": step,
                     "source": "VHS_tuple"}
                )
                return (selected_frames, width, height, frame_count, metadata)
        
        # --- Способ 2: Прямой доступ к кадрам через VHS методы ---
        if hasattr(video_obj, '__class__') and 'VHS' in str(video_obj.__class__):
            logger.info("Detected VHS object, attempting direct frame access")
            try:
                # У VHS объектов обычно есть метод get_frame или атрибут frames
                if hasattr(video_obj, 'get_frame'):
                    # Получаем информацию о видео
                    frame_count = getattr(video_obj, 'frame_count', 0)
                    width = getattr(video_obj, 'width', 0)
                    height = getattr(video_obj, 'height', 0)
                    fps = getattr(video_obj, 'fps', 0)
                    
                    if frame_count == 0:
                        frame_count = getattr(video_obj, 'total_frames', 0)
                    
                    indices = self._calculate_frame_indices(
                        frame_selection_mode, frame_count,
                        exact_frame, start_frame, end_frame, num_frames, step
                    )
                    
                    frames = []
                    for idx in indices:
                        frame = video_obj.get_frame(idx)
                        if frame is not None:
                            frames.append(frame)
                    
                    if frames:
                        frames_tensor = torch.stack(frames) if isinstance(frames[0], torch.Tensor) else torch.from_numpy(np.array(frames))
                        metadata = self._create_metadata_json(
                            "[from VHS object]", width, height, frame_count, fps,
                            frame_selection_mode, indices,
                            {"exact_frame": exact_frame, "start_frame": start_frame,
                             "end_frame": end_frame, "num_frames": num_frames, "step": step,
                             "source": "VHS_direct"}
                        )
                        return (frames_tensor, width, height, frame_count, metadata)
            except Exception as e:
                logger.warning(f"VHS direct access failed: {e}")
        
        # --- Способ 3: Стандартный ComfyUI VideoFromFile ---
        # Получаем путь через доступные атрибуты
        video_path = None
        
        # Пробуем найти путь в различных атрибутах
        for attr_name in ['path', 'video_path', 'file_path', '_path', 'source_path']:
            if hasattr(video_obj, attr_name):
                candidate = getattr(video_obj, attr_name)
                if isinstance(candidate, str) and candidate:
                    video_path = candidate
                    logger.info(f"Found video path in attribute '{attr_name}': {video_path}")
                    break
        
        # Если пути нет, пробуем использовать save_to для сохранения во временный файл
        if not video_path and hasattr(video_obj, 'save_to'):
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                video_obj.save_to(temp_path)
                video_path = temp_path
                logger.info(f"Saved video to temporary file: {temp_path}")
                # Временный файл будет удалён после использования в _extract_from_file
            except Exception as e:
                logger.warning(f"Failed to save video to temp file: {e}")
        
        if video_path and os.path.exists(video_path):
            try:
                return self._extract_from_file(
                    video_path, frame_selection_mode,
                    exact_frame, start_frame, end_frame, num_frames, step
                )
            finally:
                # Если использовали временный файл - удаляем его
                if video_path != getattr(video_obj, 'path', None) and video_path != getattr(video_obj, 'video_path', None):
                    try:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                    except:
                        pass
        
        # --- Способ 4: Пробуем получить через get_components_internal ---
        if hasattr(video_obj, 'get_components_internal'):
            try:
                logger.info("Attempting to get frames via get_components_internal")
                components = video_obj.get_components_internal()
                if components and len(components) > 0:
                    # Получаем информацию о размерах
                    dimensions = video_obj.get_dimensions() if hasattr(video_obj, 'get_dimensions') else (0, 0)
                    width, height = dimensions if dimensions else (components[0].shape[2], components[0].shape[1])
                    frame_count = len(components)
                    fps = video_obj.get_frame_rate() if hasattr(video_obj, 'get_frame_rate') else 0
                    
                    indices = self._calculate_frame_indices(
                        frame_selection_mode, frame_count,
                        exact_frame, start_frame, end_frame, num_frames, step
                    )
                    
                    frames = []
                    for idx in indices:
                        if idx < len(components):
                            frame_tensor = components[idx]
                            # Конвертируем тензор в формат IMAGE (BGR -> RGB уже в _extract_from_tensor)
                            if isinstance(frame_tensor, torch.Tensor):
                                frames.append(frame_tensor)
                    
                    if frames:
                        frames_tensor = torch.stack(frames)
                        metadata = self._create_metadata_json(
                            "[from video object]", width, height, frame_count, fps,
                            frame_selection_mode, indices,
                            {"exact_frame": exact_frame, "start_frame": start_frame,
                             "end_frame": end_frame, "num_frames": num_frames, "step": step,
                             "source": "get_components_internal"}
                        )
                        return (frames_tensor, width, height, frame_count, metadata)
            except Exception as e:
                logger.warning(f"get_components_internal failed: {e}")
        
        raise TypeError(f"Unsupported video object type: {type(video_obj)}. Cannot extract frames. "
                        f"The node supports VHS (Video Helper Suite) output, standard ComfyUI Load Video, "
                        f"or providing a direct path via custom_path.")
    
    def _extract_from_file(
        self,
        video_path: str,
        frame_selection_mode: str,
        exact_frame: Optional[int],
        start_frame: Optional[int],
        end_frame: Optional[int],
        num_frames: Optional[int],
        step: Optional[int],
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """Извлекает кадры из видеофайла через OpenCV."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Уточнение total_frames
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, _ = cap.read()
            if not ret:
                logger.warning("OpenCV CAP_PROP_FRAME_COUNT is inaccurate. Trying to find last valid frame...")
                found = False
                for i in range(1, 21):
                    test_idx = max(0, total_frames - 1 - i)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, test_idx)
                    ret, _ = cap.read()
                    if ret:
                        total_frames = test_idx + 1
                        found = True
                        break
                if not found:
                    logger.warning("Could not verify any frame near the end. Using CAP_PROP_FRAME_COUNT as-is.")
        else:
            total_frames = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_indices = self._calculate_frame_indices(
            frame_selection_mode, total_frames,
            exact_frame, start_frame, end_frame, num_frames, step
        )

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logger.warning(f"Could not read frame at index {idx}. Skipping.")

        cap.release()

        if not frames:
            raise RuntimeError("No frames were successfully extracted.")

        # BGR → RGB и конвертация в тензор
        frames_np = np.array(frames)
        frames_rgb = np.ascontiguousarray(frames_np[..., ::-1])
        frames_tensor = torch.from_numpy(frames_rgb).float() / 255.0

        metadata_json = self._create_metadata_json(
            video_path, width, height, total_frames, fps,
            frame_selection_mode, frame_indices,
            {"exact_frame": exact_frame, "start_frame": start_frame,
             "end_frame": end_frame, "num_frames": num_frames, "step": step}
        )

        return (frames_tensor, width, height, total_frames, metadata_json)
    
    def _extract_from_tensor(
        self,
        video_frames: torch.Tensor,
        frame_selection_mode: str,
        exact_frame: Optional[int],
        start_frame: Optional[int],
        end_frame: Optional[int],
        num_frames: Optional[int],
        step: Optional[int],
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """Извлекает кадры из предварительно загруженного тензора."""
        
        if video_frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor (frames, height, width, channels), got {video_frames.dim()}D")
        
        total_frames = video_frames.shape[0]
        height = video_frames.shape[1]
        width = video_frames.shape[2]
        channels = video_frames.shape[3]
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {channels}")
        
        frame_indices = self._calculate_frame_indices(
            frame_selection_mode, total_frames,
            exact_frame, start_frame, end_frame, num_frames, step
        )
        
        selected_frames = video_frames[frame_indices]
        
        if selected_frames.max() > 1.0:
            logger.warning("Tensor values exceed 1.0, clamping to [0, 1]")
            selected_frames = selected_frames.clamp(0, 1)
        
        metadata_json = self._create_metadata_json(
            "[from video_frames tensor]", width, height, total_frames, 0,
            frame_selection_mode, frame_indices,
            {"exact_frame": exact_frame, "start_frame": start_frame,
             "end_frame": end_frame, "num_frames": num_frames, "step": step,
             "source": "video_frames_tensor"}
        )
        
        return (selected_frames, width, height, total_frames, metadata_json)

    def _calculate_frame_indices(
        self,
        mode: str,
        total_frames: int,
        exact_frame: Optional[int],
        start_frame: Optional[int],
        end_frame: Optional[int],
        num_frames: Optional[int],
        step: Optional[int],
    ) -> List[int]:
        """Вычисляет список индексов кадров (0-based)."""
        if total_frames == 0:
            return []

        if mode == "first":
            return [0]
        elif mode == "last":
            return [total_frames - 1]
        elif mode == "exact_frame":
            idx = max(0, min(exact_frame - 1, total_frames - 1))
            return [idx]
        elif mode == "range":
            start_0b = max(0, (start_frame or 1) - 1)
            end_0b = min((end_frame or total_frames) - 1, total_frames - 1)
            step_val = step if step is not None else 1
            return list(range(start_0b, end_0b + 1, step_val))
        elif mode == "sample":
            if num_frames is None or num_frames <= 0:
                num_frames = 10
            if num_frames >= total_frames:
                return list(range(total_frames))
            indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int)
            return indices.tolist()
        elif mode == "all":
            step_val = step if step is not None else 1
            return list(range(0, total_frames, step_val))
        else:
            raise ValueError(f"Unknown frame selection mode: {mode}")

    def _create_metadata_json(
        self,
        video_path: str,
        width: int,
        height: int,
        total_frames: int,
        fps: float,
        frame_selection_mode: str,
        frame_indices: List[int],
        node_params: Dict[str, Any],
    ) -> str:
        """Создаёт JSON-строку с метаданными."""
        metadata = {
            "video_info": {
                "path": video_path,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "fps": round(fps, 2) if fps > 0 else 0,
                "duration_seconds": round(total_frames / fps, 2) if fps > 0 else 0,
                "orientation": "landscape" if width >= height else "portrait",
            },
            "node_execution_params": {
                "frame_selection_mode": frame_selection_mode,
                **{k: v for k, v in node_params.items() if v is not None},
            },
            "extracted_frame_indices_1_based": [idx + 1 for idx in frame_indices],
            "extracted_frame_count": len(frame_indices),
        }
        return json.dumps(metadata, indent=4, ensure_ascii=False)


# --- Регистрация ноды ---
NODE_CLASS_MAPPINGS = {
    "AGSoftVideoFrameExtractor": AGSoftVideoFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftVideoFrameExtractor": "🎬AGSoft VideoFrameExtractor"
}
