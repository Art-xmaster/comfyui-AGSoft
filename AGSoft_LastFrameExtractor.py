"""
# AGSoft LastFrameExtractor
# Автор: AGSoft
# Дата: 10.05.2026 г.
Извлекает один кадр из конца видеофайла. Позволяет выбрать конкретную позицию от конца.
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


class AGSoftLastFrameExtractor:
    """
    Основной класс ноды для извлечения одного кадра из конца видео.
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
                "frame_offset_from_end": (
                    "INT",
                    {
                        "default": 1, "min": 1, "max": 10000,
                        "tooltip": """
Position from the end of the video (1-based).
frame_offset_from_end=1 - extracts the very last frame.
frame_offset_from_end=2 - extracts the second-to-last frame.
frame_offset_from_end=3 - extracts the third-to-last frame, etc.

Позиция от конца видео (нумерация с 1).
frame_offset_from_end=1 - извлекает самый последний кадр.
frame_offset_from_end=2 - извлекает предпоследний кадр.
frame_offset_from_end=3 - извлекает третий с конца кадр и т.д.
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
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame", "width", "height", "total_frames", "video_metadata_json")
    FUNCTION = "extract_last_frame"
    CATEGORY = "AGSoft/Video"
    DESCRIPTION = """
Extracts a single frame from the end of a video file. Allows you to choose a specific position from the end (e.g., last frame, second-to-last, third-to-last, etc.). Returns the extracted frame as an image along with video metadata.

Извлекает один кадр из конца видеофайла. Позволяет выбрать конкретную позицию от конца (например, последний кадр, предпоследний, третий с конца и т.д.). Возвращает извлеченный кадр как изображение вместе с метаданными видео.
"""

    def extract_last_frame(
        self,
        custom_path: str,
        frame_offset_from_end: int,
        video: Optional[Any] = None,
        video_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """
        Основной метод извлечения одного кадра с конца видео.
        Приоритет: video > video_frames > custom_path
        """
        
        # Приоритет 1: video (объект от VHS или стандартного Load Video)
        if video is not None:
            logger.info("Using video object from 'video' input")
            return self._extract_from_video_object(
                video_obj=video,
                frame_offset_from_end=frame_offset_from_end,
            )
        
        # Приоритет 2: video_frames (уже загруженные кадры)
        if video_frames is not None:
            logger.info("Using pre-loaded video_frames tensor")
            return self._extract_from_tensor(
                video_frames=video_frames,
                frame_offset_from_end=frame_offset_from_end,
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
            frame_offset_from_end=frame_offset_from_end,
        )
    
    def _extract_from_video_object(
        self,
        video_obj: Any,
        frame_offset_from_end: int,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """
        Извлекает кадр из видеообъекта, совместимого с Video Helper Suite.
        Поддерживает как VHS объекты, так и стандартные VideoFromFile.
        """
        
        # --- Способ 1: VHS объект (имеет атрибуты и методы VHS) ---
        if isinstance(video_obj, (tuple, list)) and len(video_obj) >= 2:
            if isinstance(video_obj[0], torch.Tensor):
                logger.info("Detected VHS tuple output format")
                images_tensor = video_obj[0]
                frame_count = video_obj[1] if len(video_obj) > 1 else images_tensor.shape[0]
                height, width = images_tensor.shape[1], images_tensor.shape[2]
                
                # Расчёт индекса с конца
                target_index = self._calculate_end_index(frame_count, frame_offset_from_end)
                
                # Извлекаем нужный кадр
                selected_frame = images_tensor[target_index:target_index+1]  # Сохраняем размерность (1, H, W, C)
                
                metadata = self._create_metadata_json(
                    "[from VHS object]", width, height, frame_count, 0,
                    frame_offset_from_end, target_index,
                    {"source": "VHS_tuple"}
                )
                return (selected_frame, width, height, frame_count, metadata)
        
        # --- Способ 2: Прямой доступ к кадрам через VHS методы ---
        if hasattr(video_obj, '__class__') and 'VHS' in str(video_obj.__class__):
            logger.info("Detected VHS object, attempting direct frame access")
            try:
                if hasattr(video_obj, 'get_frame'):
                    frame_count = getattr(video_obj, 'frame_count', 0)
                    width = getattr(video_obj, 'width', 0)
                    height = getattr(video_obj, 'height', 0)
                    fps = getattr(video_obj, 'fps', 0)
                    
                    if frame_count == 0:
                        frame_count = getattr(video_obj, 'total_frames', 0)
                    
                    target_index = self._calculate_end_index(frame_count, frame_offset_from_end)
                    
                    frame = video_obj.get_frame(target_index)
                    
                    if frame is not None:
                        frame_tensor = frame.unsqueeze(0) if isinstance(frame, torch.Tensor) else torch.from_numpy(np.array([frame]))
                        metadata = self._create_metadata_json(
                            "[from VHS object]", width, height, frame_count, fps,
                            frame_offset_from_end, target_index,
                            {"source": "VHS_direct"}
                        )
                        return (frame_tensor, width, height, frame_count, metadata)
            except Exception as e:
                logger.warning(f"VHS direct access failed: {e}")
        
        # --- Способ 3: Стандартный ComfyUI VideoFromFile ---
        video_path = None
        
        for attr_name in ['path', 'video_path', 'file_path', '_path', 'source_path']:
            if hasattr(video_obj, attr_name):
                candidate = getattr(video_obj, attr_name)
                if isinstance(candidate, str) and candidate:
                    video_path = candidate
                    logger.info(f"Found video path in attribute '{attr_name}': {video_path}")
                    break
        
        if not video_path and hasattr(video_obj, 'save_to'):
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                video_obj.save_to(temp_path)
                video_path = temp_path
                logger.info(f"Saved video to temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to save video to temp file: {e}")
        
        if video_path and os.path.exists(video_path):
            try:
                return self._extract_from_file(
                    video_path, frame_offset_from_end
                )
            finally:
                if video_path != getattr(video_obj, 'path', None) and video_path != getattr(video_obj, 'video_path', None):
                    try:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
                    except:
                        pass
        
        # --- Способ 4: Пробуем получить через get_components_internal ---
        if hasattr(video_obj, 'get_components_internal'):
            try:
                logger.info("Attempting to get frame via get_components_internal")
                components = video_obj.get_components_internal()
                if components and len(components) > 0:
                    dimensions = video_obj.get_dimensions() if hasattr(video_obj, 'get_dimensions') else (0, 0)
                    width, height = dimensions if dimensions else (components[0].shape[2], components[0].shape[1])
                    frame_count = len(components)
                    fps = video_obj.get_frame_rate() if hasattr(video_obj, 'get_frame_rate') else 0
                    
                    target_index = self._calculate_end_index(frame_count, frame_offset_from_end)
                    
                    if target_index < len(components):
                        frame_tensor = components[target_index]
                        if isinstance(frame_tensor, torch.Tensor):
                            frame_tensor = frame_tensor.unsqueeze(0)  # Добавляем batch dimension
                            metadata = self._create_metadata_json(
                                "[from video object]", width, height, frame_count, fps,
                                frame_offset_from_end, target_index,
                                {"source": "get_components_internal"}
                            )
                            return (frame_tensor, width, height, frame_count, metadata)
            except Exception as e:
                logger.warning(f"get_components_internal failed: {e}")
        
        raise TypeError(f"Unsupported video object type: {type(video_obj)}. Cannot extract frame. "
                        f"The node supports VHS (Video Helper Suite) output, standard ComfyUI Load Video, "
                        f"or providing a direct path via custom_path.")
    
    def _extract_from_file(
        self,
        video_path: str,
        frame_offset_from_end: int,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """Извлекает кадр из видеофайла через OpenCV."""
        
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

        # Расчёт индекса с конца
        target_index = self._calculate_end_index(total_frames, frame_offset_from_end)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Could not extract frame at index {target_index} (offset {frame_offset_from_end} from end)")

        # BGR → RGB и конвертация в тензор
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)  # Добавляем batch dimension: (H, W, C) -> (1, H, W, C)

        metadata_json = self._create_metadata_json(
            video_path, width, height, total_frames, fps,
            frame_offset_from_end, target_index,
            {"source": "file"}
        )

        return (frame_tensor, width, height, total_frames, metadata_json)
    
    def _extract_from_tensor(
        self,
        video_frames: torch.Tensor,
        frame_offset_from_end: int,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """Извлекает кадр из предварительно загруженного тензора."""
        
        if video_frames.dim() != 4:
            raise ValueError(f"Expected 4D tensor (frames, height, width, channels), got {video_frames.dim()}D")
        
        total_frames = video_frames.shape[0]
        height = video_frames.shape[1]
        width = video_frames.shape[2]
        channels = video_frames.shape[3]
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {channels}")
        
        target_index = self._calculate_end_index(total_frames, frame_offset_from_end)
        
        selected_frame = video_frames[target_index:target_index+1]  # Сохраняем размерность (1, H, W, C)
        
        if selected_frame.max() > 1.0:
            logger.warning("Tensor values exceed 1.0, clamping to [0, 1]")
            selected_frame = selected_frame.clamp(0, 1)
        
        metadata_json = self._create_metadata_json(
            "[from video_frames tensor]", width, height, total_frames, 0,
            frame_offset_from_end, target_index,
            {"source": "video_frames_tensor"}
        )
        
        return (selected_frame, width, height, total_frames, metadata_json)

    def _calculate_end_index(
        self,
        total_frames: int,
        offset_from_end: int,
    ) -> int:
        """
        Вычисляет индекс кадра с конца видео (0-based).
        offset_from_end=1 - последний кадр (total_frames - 1)
        offset_from_end=2 - предпоследний кадр (total_frames - 2)
        и т.д.
        """
        if total_frames == 0:
            raise ValueError("Video has no frames to extract")
        
        if offset_from_end > total_frames:
            logger.warning(f"Offset {offset_from_end} exceeds total frames {total_frames}. Using first frame (index 0)")
            return 0
        
        return total_frames - offset_from_end

    def _create_metadata_json(
        self,
        video_path: str,
        width: int,
        height: int,
        total_frames: int,
        fps: float,
        frame_offset_from_end: int,
        frame_index: int,
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
                "frame_offset_from_end": frame_offset_from_end,
                **{k: v for k, v in node_params.items() if v is not None},
            },
            "extracted_frame_info": {
                "frame_index_0_based": frame_index,
                "frame_number_1_based": frame_index + 1,
                "frames_from_end": frame_offset_from_end,
            },
            "extracted_frame_count": 1,
            "extraction_type": "single_from_end",
        }
        return json.dumps(metadata, indent=4, ensure_ascii=False)


# --- Регистрация ноды ---
NODE_CLASS_MAPPINGS = {
    "AGSoftLastFrameExtractor": AGSoftLastFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLastFrameExtractor": "🎬AGSoft LastFrameExtractor"
}