"""
==============================================================================
AGSoft_Crop_Face.py
==============================================================================
Нода: 🧑✂️AGSoft Crop Face
Описание / Description:
Высокоточное обнаружение и кадрирование лиц через RetinaFace (MobileNet0.25).
Нода находит лица, фильтрует их по уверенности, минимальному размеру и
покадровой маске, сортирует выбранным способом, отбирает нужное количество
начиная с заданного индекса (циклически) и кадрирует каждое с настраиваемым
масштабом, пропорциями и независимым сдвигом по горизонтали и вертикали.
Окно кадрирования у края кадра либо сдвигается внутрь, сохраняя точный размер
и пропорции (shift_inside), либо урезается по границе (clamp).
Сырые детекции кэшируются покадрово: смена уверенности, маски, минимального
размера, сортировки или геометрии кропа не перезапускает нейросеть.
Режим on_error управляет поведением при настоящем сбое: raise останавливает
очередь и показывает ошибку, passthrough пропускает исходный кадр дальше.
Отсутствие лиц сбоем не считается и всегда пропускает исходник.
Первый выход crop_data содержит оригинальный кадр и точный бокс ПЕРВОГО
возвращённого лица — этого достаточно, чтобы обработать лицо любыми нодами и
вернуть его на место через 🖼️AGSoft Crop Stitch одним проводом. Остальные
боксы отдаются списком bboxes, а овалы лиц — батчем face_mask, совпадающим по
порядку и размерам с батчем face.
Каналы каждого кропа выравниваются под раскладку исходного тензора, поэтому
батч собирается без конфликтов RGB/RGBA и остаётся совместимым со Stitch.
Детектор кэшируется в экземпляре ноды и переинициализируется только при смене
лимита обнаружения. Встроенный пакет Pytorch_Retinaface подключается
автоматически, при отсутствии служебного файла пакета он создаётся.
---
High-accuracy face detection and cropping via RetinaFace (MobileNet0.25).
The node detects faces, filters them by confidence, minimum size and a
per-frame mask, sorts them by the chosen rule, selects the requested count
starting from a given index (cyclically) and crops each one with configurable
scale, aspect ratio and independent horizontal/vertical shift.
Near frame borders the crop window is either moved inside, keeping the exact
size and ratio (shift_inside), or cut at the border (clamp).
Raw detections are cached per frame: changing confidence, mask, minimum size,
sorting or crop geometry does not re-run the network.
The on_error mode controls real failures: raise stops the queue and surfaces
the error, passthrough passes the source frame through.
"No faces found" is not a failure and always passes the source through.
The FIRST output crop_data carries the original frame and the exact box of
the FIRST returned face — enough to process the face with any nodes and put
it back via 🖼️AGSoft Crop Stitch with a single link. The remaining boxes are
exposed as bboxes, and the face ovals as a face_mask batch matching the order
and sizes of the face batch.
Every crop has its channel layout aligned to the source tensor, so the batch
stacks without RGB/RGBA conflicts and stays compatible with Stitch.
The detector is cached per node instance and re-created only when the
detection limit changes. The bundled Pytorch_Retinaface package is imported
automatically; a missing package init file is created on the fly.
Возможности / Features:
⚡ crop_data — ПЕРВЫЙ выход: оригинал + бокс лица для Crop Stitch.
  crop_data is the FIRST output: original + face box for Crop Stitch.
⚡ face_mask — овал каждого лица с размягчением края (mask_softness).
  face_mask — per-face oval with a soft edge (mask_softness).
⚡ bboxes — список (x, y, w, h) всех возвращённых лиц в пикселях исходника.
  bboxes — (x, y, w, h) list of every returned face in source pixels.
⚡ Кэш сырых детекций по кадру: фильтры и сортировка без перезапуска сети.
  Raw per-frame detection cache: filters and sorting without re-running.
⚡ on_error: raise (остановить очередь) / passthrough (пропустить исходник).
  on_error: raise (stop the queue) / passthrough (pass the source through).
⚡ sort_by: size / confidence / left-to-right + min_face_size в пикселях.
  sort_by: size / confidence / left-to-right + min_face_size in pixels.
⚡ edge_mode: shift_inside сохраняет размер и пропорции у края кадра.
  edge_mode: shift_inside keeps size and ratio near frame borders.
⚡ Покадровая маска: mask[i] фильтрует кадр i, одиночная маска — весь батч.
  Per-frame mask: mask[i] filters frame i, a single mask covers the batch.
⚡ Выравнивание каналов перед ресайзом и стеком (RGB/RGBA не смешиваются).
  Channel alignment before resize and stack (no RGB/RGBA mixing).
⚡ RetinaFace MobileNet0.25 на устройстве ComfyUI (CUDA/MPS/XPU/CPU).
  RetinaFace MobileNet0.25 on the ComfyUI device (CUDA/MPS/XPU/CPU).
⚡ scale_factor, независимый сдвиг по X/Y, пресеты пропорций, divisible_by.
  scale_factor, independent X/Y shift, aspect presets, divisible_by.
Автор / Author: AGSoft
Дата / Date: 21.11.2026
==============================================================================
"""

import hashlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

import comfy.utils
from comfy.model_management import get_torch_device

# print("[AGSoft Crop Face] v11.21 loaded (detection cache + face mask + sorting + edge modes)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_DIR = os.path.join(NODE_DIR, "Pytorch_Retinaface")
DETECTOR_TOP_K = 50
DETECTION_CACHE_LIMIT = 32
CROP_DATA_VERSION = 1
FACE_MASK_INSET = 0.96


def _load_retinaface() -> Tuple[bool, Any]:
    """Import the bundled detector package. / Импорт встроенного пакета детектора."""
    if not os.path.isdir(RF_DIR):
        logger.error("[AGSoft Crop Face] package folder not found / папка не найдена: %s", RF_DIR)
        return False, None
    init_file = os.path.join(RF_DIR, "__init__.py")
    if not os.path.exists(init_file):
        # The bundled sources use relative imports and need a package marker.
        # Исходники используют относительные импорты и требуют маркер пакета.
        try:
            with open(init_file, "w", encoding="utf-8") as f:
                f.write("# Auto-generated by AGSoft to fix relative imports\n")
        except OSError as e:
            logger.warning("[AGSoft Crop Face] cannot create package init / "
                           "не удалось создать файл пакета: %s", e)
    if NODE_DIR not in sys.path:
        sys.path.insert(0, NODE_DIR)
    try:
        import Pytorch_Retinaface.pytorch_retinaface as rf_module
        return True, rf_module.Pytorch_RetinaFace
    except Exception as e:
        logger.exception("[AGSoft Crop Face] detector import failed / "
                         "ошибка импорта детектора: %s", e)
        return False, None


RETINAFACE_AVAILABLE, Pytorch_RetinaFace = _load_retinaface()
if RETINAFACE_AVAILABLE:
    logger.info("[AGSoft Crop Face] RetinaFace ready / RetinaFace готов")


class AGSoftCropFace:
    """Face detection and cropping with a Stitch-compatible crop_data output.
    Обнаружение и кадрирование лиц с выходом crop_data для Stitch."""

    DESCRIPTION = (
        "Detects faces with RetinaFace and crops them with configurable scale, aspect "
        "ratio and independent X/Y shift. Raw detections are cached per frame, so "
        "confidence, mask, sorting and crop geometry changes do not re-run the network. "
        "Supports batches, per-frame masks, minimum face size, sorting rules, border "
        "handling (shift_inside / clamp) and an explicit error mode. Outputs crop_data "
        "for AGSoft Crop Stitch, the face batch, all boxes and a face oval mask batch.\n"
        "---\n"
        "Обнаруживает лица через RetinaFace и кадрирует их с настраиваемым масштабом, "
        "пропорциями и независимым сдвигом по X/Y. Сырые детекции кэшируются покадрово, "
        "поэтому смена уверенности, маски, сортировки и геометрии кропа не перезапускает "
        "сеть. Поддерживает батчи, покадровые маски, минимальный размер лица, правила "
        "сортировки, обработку краёв (shift_inside / clamp) и явный режим ошибок. "
        "Выдаёт crop_data для AGSoft Crop Stitch, батч лиц, все боксы и батч масок-овалов."
    )
    CATEGORY = "AGSoft/Crop"

    RETURN_TYPES = ("CROP_DATA", "IMAGE", "BBOX", "MASK")
    RETURN_NAMES = ("crop_data", "face", "bboxes", "face_mask")
    OUTPUT_TOOLTIPS = (
        "Original frame + exact box of the FIRST returned face, ready for Crop Stitch.\n---\n"
        "Оригинальный кадр + точный бокс ПЕРВОГО возвращённого лица для Crop Stitch.",
        "Batch of cropped faces, all resized to a single size.\n---\n"
        "Батч кадрированных лиц, приведённых к одному размеру.",
        "List of (x, y, w, h) boxes for every returned face, in source pixels.\n---\n"
        "Список боксов (x, y, w, h) всех возвращённых лиц в пикселях исходника.",
        "Face oval mask per returned face, same order and size as the face batch.\n---\n"
        "Маска-овал на каждое возвращённое лицо, тот же порядок и размер, что у батча лиц.",
    )
    FUNCTION = "crop_face"

    def __init__(self) -> None:
        self.face_detector: Any = None
        self.current_max_faces: Optional[int] = None
        self._detection_cache: Dict[str, np.ndarray] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Input image containing one or more faces to detect and crop. "
                        "Supports batch processing; the crop keeps the source channel "
                        "layout (RGB or RGBA).\n---\n"
                        "📷 Входное изображение с одним или несколькими лицами. "
                        "Поддерживает пакетную обработку; кроп сохраняет раскладку "
                        "каналов исходника (RGB или RGBA)."
                    ),
                }),
                "number_of_faces": ("INT", {
                    "default": 5, "min": 1, "max": 100, "step": 1, "display": "number",
                    "tooltip": (
                        "🔢 How many faces to return after filtering and sorting. "
                        "More faces found → only this number is returned; fewer found → "
                        "all found are returned, without duplicates.\n---\n"
                        "🔢 Сколько лиц вернуть после фильтрации и сортировки. "
                        "Найдено больше → вернётся только это количество; найдено меньше → "
                        "вернутся все найденные, без дубликатов."
                    ),
                }),
                "scale_factor": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "📏 How much to enlarge the crop around the face. 1.0 = tight crop, "
                        "2.0 = face plus some background (recommended), 3.0+ = wide crop with "
                        "lots of context.\n---\n"
                        "📏 Насколько увеличить область кадрирования вокруг лица. "
                        "1.0 = плотный кроп, 2.0 = лицо + немного фона (рекомендуется), "
                        "3.0+ = широкий кроп с большим контекстом."
                    ),
                }),
                "shift_factor_x": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "↔️ Horizontal position of the face inside the crop. 0.00 = face at "
                        "the far left, 0.50 = centered (default), 1.00 = face at the far "
                        "right.\n---\n"
                        "↔️ Горизонтальная позиция лица в кадре. 0.00 = лицо слева, "
                        "0.50 = по центру (по умолчанию), 1.00 = лицо справа."
                    ),
                }),
                "shift_factor_y": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "⬆️⬇️ Vertical position of the face inside the crop. 0.00 = face at "
                        "the very top, 0.50 = centered (default), 1.00 = face at the very "
                        "bottom.\n---\n"
                        "⬆️⬇️ Вертикальная позиция лица в кадре. 0.00 = лицо вверху, "
                        "0.50 = по центру (по умолчанию), 1.00 = лицо внизу."
                    ),
                }),
                "start_index": ("INT", {
                    "default": 0, "min": 0, "max": 999, "step": 1, "display": "number",
                    "tooltip": (
                        "🔀 Which face to start from in the sorted list (0 = first). "
                        "Selection wraps cyclically when more faces are requested than "
                        "found.\n---\n"
                        "🔀 С какого лица начать в отсортированном списке (0 = первое). "
                        "Выбор циклический, если запрошено больше лиц, чем найдено."
                    ),
                }),
                "max_faces_per_image": ("INT", {
                    "default": 50, "min": 1, "max": 1000, "step": 1,
                    "tooltip": (
                        "🔍 Maximum faces the detector will look for (technical limit). "
                        "Changing this value invalidates the detection cache; "
                        "number_of_faces controls how many faces are returned.\n---\n"
                        "🔍 Максимум лиц, которые детектор будет искать (технический "
                        "предел). Смена этого значения сбрасывает кэш детекций; "
                        "number_of_faces задаёт сколько лиц вернуть."
                    ),
                }),
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "5:4", "4:5"], {
                        "default": "1:1",
                        "tooltip": (
                            "📐 Width-to-height ratio of the final crop. 1:1 = square "
                            "(avatars, face swaps), 3:4 = portrait, 16:9 = widescreen, "
                            "9:16 = vertical (Reels/Stories).\n---\n"
                            "📐 Соотношение ширины к высоте финального кропа. "
                            "1:1 = квадрат (аватарки, замена лиц), 3:4 = портрет, "
                            "16:9 = широкий формат, 9:16 = вертикальный (Reels/Stories)."
                        ),
                    }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "🎯 Minimum confidence to accept a face. 0.90-1.0 = very strict, "
                        "0.70-0.85 = balanced (recommended), 0.50-0.65 = lenient. "
                        "Applied on top of the cache, so changing it is instant.\n---\n"
                        "🎯 Минимальная уверенность для принятия лица. 0.90-1.0 = очень "
                        "строго, 0.70-0.85 = сбалансировано (рекомендуется), "
                        "0.50-0.65 = мягко. Применяется поверх кэша, поэтому меняется "
                        "мгновенно."
                    ),
                }),
                "divisible_by": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1, "display": "number",
                    "tooltip": (
                        "🔢 Round output dimensions to multiples of N. Important for "
                        "VAE/KSampler compatibility: 8 = standard, 16 = stricter, "
                        "1 = no rounding.\n---\n"
                        "🔢 Округлить размеры выхода до кратных N. Важно для "
                        "совместимости с VAE/KSampler: 8 = стандарт, 16 = строже, "
                        "1 = без округления."
                    ),
                }),
                "sort_by": (["size", "confidence", "left-to-right"], {
                    "default": "size",
                    "tooltip": (
                        "↕️ Face ordering before selection. size = largest first, "
                        "confidence = most certain first, left-to-right = by X position. "
                        "Uses the cache, no re-detection.\n---\n"
                        "↕️ Порядок лиц перед выбором. size = большие первыми, "
                        "confidence = уверенные первыми, left-to-right = по координате X. "
                        "Работает поверх кэша, без повторного детекта."
                    ),
                }),
                "min_face_size": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 1, "display": "number",
                    "tooltip": (
                        "📏 Skip faces whose shorter box side is below this many pixels. "
                        "0 = accept any size; 64 is a good filter for background faces. "
                        "Applied on top of the cache.\n---\n"
                        "📏 Пропускать лица, у которых короткая сторона бокса меньше "
                        "этого числа пикселей. 0 = любой размер; 64 хорошо отсекает лица "
                        "на фоне. Применяется поверх кэша."
                    ),
                }),
                "edge_mode": (["shift_inside", "clamp"], {
                    "default": "shift_inside",
                    "tooltip": (
                        "🖼 Crop window behaviour near frame borders. shift_inside moves "
                        "the window inside the frame, keeping the exact size and ratio "
                        "(better for Stitch). clamp cuts it at the border (legacy).\n---\n"
                        "🖼 Поведение окна кропа у краёв кадра. shift_inside сдвигает окно "
                        "внутрь, сохраняя точный размер и пропорции (лучше для Stitch). "
                        "clamp урезает по границе (старое поведение)."
                    ),
                }),
                "mask_softness": ("INT", {
                    "default": 8, "min": 0, "max": 128, "step": 1, "display": "number",
                    "tooltip": (
                        "🌫 Soft edge width of the face oval in output pixels. "
                        "0 = hard edge; 8-24 gives a natural blend when stitching the "
                        "face back.\n---\n"
                        "🌫 Ширина размягчения края овала лица в пикселях выхода. "
                        "0 = жёсткий край; 8-24 даёт естественный шов при вставке лица "
                        "обратно."
                    ),
                }),
                "on_error": (["raise", "passthrough"], {
                    "default": "raise",
                    "tooltip": (
                        "⚠️ Behaviour on a real failure (detector unavailable, internal "
                        "error). raise = stop the queue and show the error; "
                        "passthrough = return the source frame and continue. "
                        "'No faces found' is not a failure and always passes through.\n---\n"
                        "⚠️ Поведение при настоящем сбое (детектор недоступен, внутренняя "
                        "ошибка). raise = остановить очередь и показать ошибку; "
                        "passthrough = вернуть исходный кадр и продолжить. "
                        "«Лица не найдены» сбоем не считается и всегда пропускается."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "🎭 Optional mask limiting where faces are detected. Applied per "
                        "frame: mask[i] filters frame i; a single mask covers the whole "
                        "batch. Only faces whose center falls into a bright area are "
                        "accepted: white (1.0) = search here, black (0.0) = ignore.\n---\n"
                        "🎭 Опциональная маска для ограничения поиска лиц. Применяется "
                        "покадрово: mask[i] фильтрует кадр i; одиночная маска работает на "
                        "весь батч. Принимаются только лица, чей центр попал в яркую "
                        "область: белый (1.0) = искать здесь, чёрный (0.0) = игнорировать."
                    ),
                }),
            },
        }

    def _parse_aspect_ratio(self, aspect_ratio_str: str) -> float:
        """Parse 'W:H' into a float ratio. / Парсит 'W:H' в числовое отношение."""
        try:
            w, h = map(float, aspect_ratio_str.split(":"))
            if h <= 0:
                return 1.0
            return w / h
        except (ValueError, AttributeError, ZeroDivisionError):
            logger.warning("[AGSoft Crop Face] bad aspect ratio, fallback 1:1 / "
                           "некорректные пропорции, откат 1:1: %s", aspect_ratio_str)
            return 1.0

    def _ensure_detector(self, max_faces_per_image: int) -> None:
        """Reuse the detector until the limit changes. / Переиспользует детектор."""
        if self.face_detector is not None and self.current_max_faces == max_faces_per_image:
            return
        logger.info("[AGSoft Crop Face] loading RetinaFace (MobileNet0.25, max_faces=%s)",
                    max_faces_per_image)
        self.face_detector = Pytorch_RetinaFace(
            top_k=DETECTOR_TOP_K,
            keep_top_k=max_faces_per_image,
            device=str(get_torch_device()),
        )
        self.current_max_faces = max_faces_per_image
        logger.info("[AGSoft Crop Face] detector ready / детектор готов")

    def _frame_hash(self, tensor: torch.Tensor) -> str:
        """Shape/dtype plus evenly spread samples identify a frame cheaply.
        Форма/dtype и равномерная выборка дёшево опознают кадр."""
        flat = tensor.reshape(-1)
        total = flat.numel()
        step = max(1, total // 4096)
        sample = flat[::step][:4096]
        meta = f"{tuple(tensor.shape)}|{tensor.dtype}|".encode()
        return hashlib.sha256(meta + sample.cpu().numpy().tobytes()).hexdigest()

    def _detect_frame(self, img_np: np.ndarray, cache_key: str) -> np.ndarray:
        """Raw detections with a per-frame cache; filters are applied later.
        Сырые детекции с покадровым кэшем; фильтры применяются позже."""
        cached = self._detection_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        detections = self.face_detector.detect_faces(img_np)
        if detections is None or len(detections) == 0:
            array = np.zeros((0, 5), dtype=np.float32)
        else:
            array = np.asarray(detections, dtype=np.float32)
            if array.ndim != 2:
                array = np.zeros((0, 5), dtype=np.float32)
        if len(self._detection_cache) >= DETECTION_CACHE_LIMIT:
            del self._detection_cache[next(iter(self._detection_cache))]
        self._detection_cache[cache_key] = array
        return array

    def _match_channels(self, tensor: torch.Tensor, channels: int) -> torch.Tensor:
        """Align channel layout before resize/stack. / Выравнивает каналы до ресайза и стека."""
        current = int(tensor.shape[-1])
        if current == channels or current < 1:
            return tensor
        if current > channels:
            return tensor[..., :channels]
        if channels == 4 and current == 3:
            alpha = torch.ones_like(tensor[..., :1])
            return torch.cat([tensor, alpha], dim=-1)
        pad = tensor[..., :1].repeat(1, 1, channels - current)
        return torch.cat([tensor, pad], dim=-1)

    def _frame_mask(self, mask: Optional[torch.Tensor],
                    frame_index: int) -> Optional[np.ndarray]:
        """Mask of this frame; a single mask is reused for the batch.
        Маска этого кадра; одиночная маска переиспользуется на батч."""
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask.cpu().numpy()
        idx = min(frame_index, int(mask.shape[0]) - 1)
        return mask[idx].cpu().numpy()

    def _place_crop_window(self, cx: float, cy: float, crop_w: int, crop_h: int,
                           img_w: int, img_h: int,
                           edge_mode: str) -> Tuple[int, int, int, int]:
        """Keep the exact window size by shifting it inside, or cut at borders.
        Сохраняет точный размер окна сдвигом внутрь либо режет по границам."""
        if edge_mode == "shift_inside" and crop_w <= img_w and crop_h <= img_h:
            x1 = int(round(cx - crop_w / 2))
            y1 = int(round(cy - crop_h / 2))
            x1 = max(0, min(x1, img_w - crop_w))
            y1 = max(0, min(y1, img_h - crop_h))
            return x1, y1, x1 + crop_w, y1 + crop_h
        x1 = max(0, int(cx - crop_w / 2))
        y1 = max(0, int(cy - crop_h / 2))
        x2 = min(img_w, int(cx + crop_w / 2))
        y2 = min(img_h, int(cy + crop_h / 2))
        return x1, y1, x2, y2

    def _face_oval(self, crop_h: int, crop_w: int, face_box: Tuple[int, int, int, int],
                   origin: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Ellipse inscribed into the face box, in crop coordinates.
        Эллипс, вписанный в бокс лица, в координатах кропа."""
        x1, y1, x2, y2 = face_box
        ox, oy = origin
        cx = (x1 + x2) / 2.0 - ox
        cy = (y1 + y2) / 2.0 - oy
        rx = max(1.0, (x2 - x1) / 2.0 * FACE_MASK_INSET)
        ry = max(1.0, (y2 - y1) / 2.0 * FACE_MASK_INSET)
        yy = torch.arange(crop_h, device=device, dtype=torch.float32).unsqueeze(1)
        xx = torch.arange(crop_w, device=device, dtype=torch.float32).unsqueeze(0)
        dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        return (dist <= 1.0).to(torch.float32)

    def _soften_mask(self, mask: torch.Tensor, softness: int) -> torch.Tensor:
        """Blur the oval edge in output pixels. / Размывает край овала в пикселях выхода."""
        if softness <= 0 or not CV2_AVAILABLE:
            return mask
        h, w = int(mask.shape[0]), int(mask.shape[1])
        kernel = min(2 * int(softness) + 1, (min(h, w) // 2) | 1)
        if kernel < 3:
            return mask
        array = mask.cpu().numpy()
        array = cv2.GaussianBlur(array, (kernel, kernel), 0)
        return torch.from_numpy(np.clip(array, 0.0, 1.0)).to(device=mask.device,
                                                             dtype=torch.float32)

    def _make_crop_data(self, image: torch.Tensor, frame_index: int,
                        box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Stitch-ready package; the frame is shared, Stitch clones it before writing.
        Пакет для Stitch; кадр разделяется, Stitch клонирует его перед записью."""
        frame = image[frame_index:frame_index + 1]
        orig_h, orig_w = frame.shape[1], frame.shape[2]
        x = max(0, min(int(box[0]), orig_w - 1))
        y = max(0, min(int(box[1]), orig_h - 1))
        w = max(1, min(int(box[2]), orig_w - x))
        h = max(1, min(int(box[3]), orig_h - y))
        return {"version": CROP_DATA_VERSION, "original_image": frame,
                "x": x, "y": y, "w": w, "h": h}

    def _fallback(self, image: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor,
                                                      List[Tuple[int, int, int, int]],
                                                      torch.Tensor]:
        """Pass the source through when nothing was cropped.
        Пропускает исходник, если кроп не состоялся."""
        batch, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        box = (0, 0, w, h)
        full_mask = torch.ones((batch, h, w), dtype=torch.float32, device=image.device)
        return (self._make_crop_data(image, 0, box), image, [box], full_mask)

    def crop_face(self, image: torch.Tensor, number_of_faces: int,
                  scale_factor: float, shift_factor_x: float, shift_factor_y: float,
                  start_index: int, max_faces_per_image: int, aspect_ratio: str,
                  confidence_threshold: float = 0.7, divisible_by: int = 8,
                  sort_by: str = "size", min_face_size: int = 0,
                  edge_mode: str = "shift_inside", mask_softness: int = 8,
                  on_error: str = "raise", mask: Optional[torch.Tensor] = None
                  ) -> Tuple[Dict[str, Any], torch.Tensor,
                             List[Tuple[int, int, int, int]], torch.Tensor]:
        if image is None or image.dim() != 4:
            raise ValueError("Expected IMAGE tensor [B,H,W,C] / "
                             "Ожидается тензор IMAGE [B,H,W,C]")
        try:
            if not RETINAFACE_AVAILABLE or Pytorch_RetinaFace is None:
                raise ImportError("Pytorch_RetinaFace is not loaded / "
                                  "Модуль Pytorch_RetinaFace не загружен.")
            if not CV2_AVAILABLE:
                raise ImportError("opencv-python is required / требуется opencv-python.")
            self._ensure_detector(max_faces_per_image)
            self._cache_hits = 0
            self._cache_misses = 0
            aspect_ratio_float = self._parse_aspect_ratio(aspect_ratio)
            batch_size = int(image.shape[0])
            crop_channels = int(image.shape[-1])
            all_cropped_faces: List[torch.Tensor] = []
            all_face_masks: List[torch.Tensor] = []
            all_base_sizes: List[float] = []
            bboxes: List[Tuple[int, int, int, int]] = []
            first_frame = 0
            logger.info("[AGSoft Crop Face] processing %s image(s) / обработка кадров",
                        batch_size)
            if mask is not None:
                logger.info("[AGSoft Crop Face] per-frame mask filter active / "
                            "покадровый фильтр по маске включён")

            for img_idx in range(batch_size):
                single_image = image[img_idx]
                mask_np = self._frame_mask(mask, img_idx)
                img_np = (single_image.cpu().numpy() * 255.0).astype(np.float32)
                if img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cache_key = f"{self._frame_hash(single_image)}|{max_faces_per_image}"
                detections = self._detect_frame(img_np, cache_key)
                if len(detections) == 0:
                    logger.info("[AGSoft Crop Face] frame %s: no faces / лиц нет", img_idx)
                    continue

                valid_detections = []
                for det in detections:
                    x1, y1, x2, y2 = det[:4]
                    confidence = float(det[4]) if len(det) > 4 else 1.0
                    if confidence < confidence_threshold:
                        continue
                    face_w, face_h = x2 - x1, y2 - y1
                    if min_face_size > 0 and min(face_w, face_h) < min_face_size:
                        continue
                    if mask_np is not None:
                        mask_h, mask_w = mask_np.shape
                        cx_check = max(0, min(int((x1 + x2) / 2), mask_w - 1))
                        cy_check = max(0, min(int((y1 + y2) / 2), mask_h - 1))
                        if mask_np[cy_check, cx_check] < 0.5:
                            continue
                    valid_detections.append((x1, y1, x2, y2, confidence, face_w * face_h))

                if not valid_detections:
                    logger.info("[AGSoft Crop Face] frame %s: no faces after filters / "
                                "нет лиц после фильтров", img_idx)
                    continue
                if sort_by == "confidence":
                    valid_detections.sort(key=lambda d: d[4], reverse=True)
                elif sort_by == "left-to-right":
                    valid_detections.sort(key=lambda d: d[0])
                else:
                    valid_detections.sort(key=lambda d: d[5], reverse=True)
                total_faces = len(valid_detections)
                actual_number = min(number_of_faces, total_faces)
                selected = [valid_detections[(start_index + i) % total_faces]
                            for i in range(actual_number)]
                logger.info("[AGSoft Crop Face] frame %s: %s valid, %s selected / "
                            "валидных, выбрано", img_idx, total_faces, len(selected))

                img_h, img_w = img_np.shape[:2]
                for x1, y1, x2, y2, conf, size in selected:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cx, cy = x1 + w / 2, y1 + h / 2
                    base_size = max(w, h) * scale_factor

                    if aspect_ratio_float > 1.0:
                        crop_w = int(base_size * aspect_ratio_float)
                        crop_h = int(base_size)
                    elif aspect_ratio_float < 1.0:
                        crop_w = int(base_size)
                        crop_h = int(base_size / aspect_ratio_float)
                    else:
                        crop_w = int(base_size)
                        crop_h = int(base_size)
                    if divisible_by > 1:
                        crop_w = max(1, (crop_w // divisible_by) * divisible_by)
                        crop_h = max(1, (crop_h // divisible_by) * divisible_by)

                    # Shift moves the face inside the crop window.
                    # Сдвиг перемещает лицо внутри окна кадрирования.
                    cx_shifted = cx - (crop_w * (shift_factor_x - 0.5))
                    cy_shifted = cy - (crop_h * (shift_factor_y - 0.5))
                    nx1, ny1, nx2, ny2 = self._place_crop_window(
                        cx_shifted, cy_shifted, crop_w, crop_h, img_w, img_h, edge_mode)

                    cropped = single_image[ny1:ny2, nx1:nx2]
                    if cropped.numel() == 0:
                        continue
                    cropped = self._match_channels(cropped, crop_channels)
                    oval = self._face_oval(ny2 - ny1, nx2 - nx1, (x1, y1, x2, y2),
                                           (nx1, ny1), cropped.device)
                    if not all_cropped_faces:
                        first_frame = img_idx
                    all_cropped_faces.append(cropped)
                    all_face_masks.append(oval)
                    all_base_sizes.append(base_size)
                    bboxes.append((nx1, ny1, nx2 - nx1, ny2 - ny1))
                    logger.debug("[AGSoft Crop Face] conf=%.2f box=%s", conf, bboxes[-1])

            logger.info("[AGSoft Crop Face] detection cache: %s hit / %s miss",
                        self._cache_hits, self._cache_misses)
            if not all_cropped_faces:
                logger.warning("[AGSoft Crop Face] no faces cropped, source passed through / "
                               "лица не кадрированы, исходник пропущен дальше")
                return self._fallback(image)

            # Long side follows the average base size, aspect defines the short side.
            # Длинная сторона по среднему base_size, пропорции задают короткую.
            avg_base_size = sum(all_base_sizes) / len(all_base_sizes)
            if aspect_ratio_float >= 1.0:
                final_w = int(avg_base_size)
                final_h = int(avg_base_size / aspect_ratio_float)
            else:
                final_w = int(avg_base_size * aspect_ratio_float)
                final_h = int(avg_base_size)
            if divisible_by > 1:
                final_w = max(1, (final_w // divisible_by) * divisible_by)
                final_h = max(1, (final_h // divisible_by) * divisible_by)
            logger.info("[AGSoft Crop Face] %s face(s), final size %sx%s / лиц, размер",
                        len(all_cropped_faces), final_w, final_h)

            resized: List[torch.Tensor] = []
            resized_masks: List[torch.Tensor] = []
            for face, oval in zip(all_cropped_faces, all_face_masks):
                if face.shape[0] != final_h or face.shape[1] != final_w:
                    face = comfy.utils.common_upscale(
                        face.unsqueeze(0).movedim(-1, 1),
                        final_w, final_h, "lanczos", "center",
                    ).movedim(1, -1).squeeze(0)
                if oval.shape[0] != final_h or oval.shape[1] != final_w:
                    oval = comfy.utils.common_upscale(
                        oval.unsqueeze(0).unsqueeze(0),
                        final_w, final_h, "bilinear", "center",
                    ).squeeze(0).squeeze(0)
                resized.append(self._match_channels(face, crop_channels))
                resized_masks.append(self._soften_mask(oval, mask_softness))

            crop_data = self._make_crop_data(image, first_frame, bboxes[0])
            return (crop_data, torch.stack(resized, dim=0), bboxes,
                    torch.stack(resized_masks, dim=0))
        except Exception as e:
            logger.exception("[AGSoft Crop Face] failed / сбой: %s", e)
            if on_error == "raise":
                raise RuntimeError(f"[AGSoft Crop Face] {e}") from e
            logger.warning("[AGSoft Crop Face] passthrough: source returned / "
                           "исходник пропущен дальше")
            return self._fallback(image)


NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "AGSoftCropFace": AGSoftCropFace,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "AGSoftCropFace": "🧑✂️AGSoft Crop Face",
}