"""
# AGSoft VideoFrameExtractor
# Автор: AGSoft
# Дата: 06.12.2025 г.
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
                            "This has the highest priority.\n"
                            "Абсолютный или относительный путь к видеофайлу. "
                            "Имеет наивысший приоритет."
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
        exact_frame: Optional[int] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        num_frames: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, int, str]:
        """
        Основной метод извлечения кадров.
        """
        # --- Шаг 1: Путь к видео ---
        if not custom_path:
            raise ValueError("custom_path is required and cannot be empty.")
        video_path = os.path.abspath(custom_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")

        # --- Шаг 2: Открытие видео и получение метаданных ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Уточнение total_frames без полного чтения видео
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, _ = cap.read()
            if not ret:
                logger.warning("OpenCV CAP_PROP_FRAME_COUNT is inaccurate. Trying to find last valid frame...")
                found = False
                for i in range(1, 21):  # максимум 20 попыток
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

        # --- Шаг 3: Расчёт индексов кадров ---
        frame_indices = self._calculate_frame_indices(
            frame_selection_mode,
            total_frames,
            exact_frame,
            start_frame,
            end_frame,
            num_frames,
            step
        )

        # --- Шаг 4: Извлечение кадров ---
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

        # --- Шаг 5: Преобразование в тензор ---
        frames_np = np.array(frames)  # Shape: (N, H, W, 3) in BGR
        # Исправление negative strides
        frames_rgb = np.ascontiguousarray(frames_np[..., ::-1])  # BGR → RGB + contiguous
        frames_tensor = torch.from_numpy(frames_rgb).float() / 255.0

        # --- Шаг 6: JSON с метаданными ---
        metadata_json = self._create_metadata_json(
            video_path=video_path,
            width=width,
            height=height,
            total_frames=total_frames,
            fps=fps,
            frame_selection_mode=frame_selection_mode,
            frame_indices=frame_indices,
            node_params={
                "exact_frame": exact_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "num_frames": num_frames,
                "step": step,
            }
        )

        return (frames_tensor, width, height, total_frames, metadata_json)

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
            return [max(0, min(exact_frame - 1, total_frames - 1))]
        elif mode == "range":
            start_0b = max(0, start_frame - 1)
            end_0b = min(end_frame - 1, total_frames - 1)
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