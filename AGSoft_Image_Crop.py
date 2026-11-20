"""
==============================================================================
AGSoft_Image_Crop.py
==============================================================================
Нода: 🖼️✂️AGSoft Image Crop
Описание / Description:
Пакетная обрезка изображений: 9 предустановленных позиций + custom,
процентная обрезка, сохранение пропорций, инверсная обрезка краёв,
поддержка маски, автоматический кэш, выбор устройства (CPU/GPU/Auto
через ComfyUI: CUDA/MPS/XPU/DirectML). Первый выход crop_data
(оригинал + точный бокс с учётом inverse-обрезки) совместим с
🖼️AGSoft Crop Stitch для вставки обработанного кадра обратно.
---
Batch image cropping: 9 preset positions + custom, percentage cropping,
aspect-ratio preservation, inverse edge trimming, mask support, automatic
caching, device selection (CPU/GPU/Auto via ComfyUI: CUDA/MPS/XPU/DirectML).
The FIRST output crop_data (original + exact box including the inverse trim)
is compatible with 🖼️AGSoft Crop Stitch for stitching the processed frame back.
Возможности / Features:
⚡ crop_data — ПЕРВЫЙ выход: оригинал + бокс для Crop Stitch.
  crop_data is the FIRST output: original + box for Crop Stitch.
⚡ 9 позиций + пользовательские координаты. / 9 positions + custom coords.
⚡ Проценты от размера кадра. / Percentage of the frame size.
⚡ Сохранение пропорций оригинала. / Maintain original aspect ratio.
⚡ Инверсная обрезка краёв (0–49%). / Inverse edge trimming (0–49%).
⚡ Батчи изображений и масок. / Image and mask batches.
⚡ Авто-кэш: повтор только при смене изображения или координат.
  Auto cache: recompute only when the image or coordinates change.
⚡ Устройство через ComfyUI (CUDA/MPS/XPU/DirectML). / Device via ComfyUI.
⚡ Проверка custom-координат до выполнения. / Custom coords validated upfront.
Автор / Author: AGSoft
Дата / Date: 21.11.2026
==============================================================================
"""

import hashlib
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch

try:
    from comfy import model_management
except ImportError:
    model_management = None

# print("[AGSoft Image Crop] v11.21 loaded (crop + crop_data + auto device/cache)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AGSoft_Image_Crop:
    """Batch crop node with a Stitch-compatible crop_data output.
    Нода пакетной обрезки с выходом crop_data, совместимым со Stitch."""

    DESCRIPTION = (
        "Crop image with advanced features: 9 preset positions + custom coordinates, "
        "percentage cropping, maintain aspect ratio, inverse edge trimming, batch "
        "processing, automatic caching, ComfyUI device selection. The first output "
        "crop_data feeds AGSoft Crop Stitch.\n"
        "---\n"
        "Обрезка изображения: 9 позиций + custom, проценты, сохранение пропорций, "
        "инверсная обрезка краёв, батчи, авто-кэш, выбор устройства через ComfyUI. "
        "Первый выход crop_data подаётся в AGSoft Crop Stitch.\n"
        "Examples / Примеры: center 512x512; use_percentage=True width=50 height=50; "
        "inverse_crop_percent=5; maintain_aspect_ratio=True; mask input supported."
    )
    CATEGORY = "AGSoft/Crop"

    RETURN_TYPES = ("CROP_DATA", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("crop_data", "cropped_image", "cropped_mask",
                    "cropped_width", "cropped_height")
    OUTPUT_TOOLTIPS = (
        "Original image + exact crop box (inverse trim included) for Crop Stitch.\n---\n"
        "Оригинал + точный бокс кропа (с учётом inverse-обрезки) для Crop Stitch.",
        "Cropped image batch.\n---\nОбрезанный батч изображения.",
        "Cropped mask (None if no mask input).\n---\nОбрезанная маска (None, если маски не было).",
        "Crop width in pixels.\n---\nШирина кропа в пикселях.",
        "Crop height in pixels.\n---\nВысота кропа в пикселях.",
    )
    FUNCTION = "crop_image"

    def __init__(self) -> None:
        # Cache stores crops + box only; the original is shared, not cloned.
        # Кэш хранит кропы + бокс; оригинал разделяется, не клонируется.
        self._cache: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor],
                                     int, int, int, int]] = {}
        self._max_cache_size: int = 100

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Input image tensor [BATCH, HEIGHT, WIDTH, CHANNELS].\n---\n"
                        "Входной тензор изображения [BATCH, HEIGHT, WIDTH, CHANNELS]."
                    ),
                }),
                "width": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1, "display": "number",
                    "tooltip": (
                        "Crop width in pixels or percent (depends on use_percentage).\n---\n"
                        "Ширина обрезки в пикселях или процентах (зависит от use_percentage)."
                    ),
                }),
                "height": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1, "display": "number",
                    "tooltip": (
                        "Crop height in pixels or percent (depends on use_percentage).\n---\n"
                        "Высота обрезки в пикселях или процентах (зависит от use_percentage)."
                    ),
                }),
                "position": ([
                    "top-left", "top-center", "top-right",
                    "center-left", "center", "center-right",
                    "bottom-left", "bottom-center", "bottom-right",
                    "custom",
                ], {
                    "default": "custom",
                    "tooltip": (
                        "Crop area position: top-left / top-center / top-right / "
                        "center-left / center / center-right / bottom-left / "
                        "bottom-center / bottom-right; custom uses start_x, start_y.\n---\n"
                        "Позиция области обрезки: верх/центр/низ × лево/центр/право; "
                        "custom использует start_x, start_y."
                    ),
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": (
                        "Optional mask, must match image dimensions.\n---\n"
                        "Опциональная маска, должна совпадать по размеру с изображением."
                    ),
                }),
                "use_percentage": ("BOOLEAN", {
                    "default": False, "label_on": "Yes / Да", "label_off": "No / Нет",
                    "tooltip": (
                        "Interpret width/height as percent of the image size.\n---\n"
                        "Интерпретировать width/height как проценты от размера изображения."
                    ),
                }),
                "maintain_aspect_ratio": ("BOOLEAN", {
                    "default": False, "label_on": "Yes / Да", "label_off": "No / Нет",
                    "tooltip": (
                        "Maintain the original image aspect ratio.\n---\n"
                        "Сохранять соотношение сторон исходного изображения."
                    ),
                }),
                "inverse_crop_percent": ("INT", {
                    "default": 0, "min": 0, "max": 49, "step": 1, "display": "slider",
                    "tooltip": (
                        "Trim this percent from every edge before cropping (0-49), "
                        "e.g. 10 removes 10% per side; useful against edge artifacts.\n---\n"
                        "Обрезка краёв на этот процент перед кропом (0-49), "
                        "например 10 убирает по 10% с каждой стороны; против артефактов по краям."
                    ),
                }),
                "start_x": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1, "display": "number",
                    "tooltip": (
                        "Start X for position='custom' (0-8192); values beyond the image "
                        "are clamped to its bounds.\n---\n"
                        "Начальный X для position='custom' (0-8192); значения за пределами "
                        "изображения ограничиваются его границами."
                    ),
                }),
                "start_y": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 1, "display": "number",
                    "tooltip": (
                        "Start Y for position='custom' (0-8192); values beyond the image "
                        "are clamped to its bounds.\n---\n"
                        "Начальный Y для position='custom' (0-8192); значения за пределами "
                        "изображения ограничиваются его границами."
                    ),
                }),
                "device": (["auto", "cpu", "gpu"], {
                    "default": "auto",
                    "tooltip": (
                        "auto: ComfyUI picks the best device (CUDA/MPS/XPU/DirectML/CPU); "
                        "cpu: force CPU; gpu: force accelerator (falls back to CPU).\n---\n"
                        "auto: ComfyUI выбирает лучшее устройство (CUDA/MPS/XPU/DirectML/CPU); "
                        "cpu: принудительно CPU; gpu: принудительно ускоритель (откат на CPU)."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, position: str = "center", start_x: int = 0,
                        start_y: int = 0,
                        input_types: Optional[Dict[str, str]] = None,
                        **kwargs) -> Union[bool, str]:
        """Fail fast on bad custom coordinates. / Ранняя ошибка при плохих custom-координатах."""
        if position != "custom":
            return True
        linked = input_types or {}
        for name, value in (("start_x", start_x), ("start_y", start_y)):
            if name in linked:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                return (f"{name} must be an integer when position='custom'.\n---\n"
                        f"{name} должен быть целым числом при position='custom'.")
            if value < 0:
                return (f"{name} must be >= 0 when position='custom'.\n---\n"
                        f"{name} должен быть >= 0 при position='custom'.")
            if value > 8192:
                return (f"{name} must be <= 8192 when position='custom'.\n---\n"
                        f"{name} должен быть <= 8192 при position='custom'.")
        return True

    def _get_device(self, device_choice: str) -> torch.device:
        """Resolve the compute device via ComfyUI. / Устройство через ComfyUI."""
        if device_choice == "cpu":
            return torch.device("cpu")
        if model_management is not None:
            dev = model_management.get_torch_device()
            if device_choice == "gpu" and dev.type == "cpu":
                logger.warning("[AGSoft Image Crop] accelerator not available, using CPU / "
                               "ускоритель не доступен, используется CPU")
            return dev
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_choice == "gpu":
            logger.warning("[AGSoft Image Crop] GPU not available, using CPU / "
                           "GPU не доступен, используется CPU")
        return torch.device("cpu")

    def _generate_cache_key(self, image_hash: str, mask_hash: Optional[str],
                            width: int, height: int, position: str,
                            use_percentage: bool, maintain_aspect_ratio: bool,
                            inverse_crop_percent: int, start_x: int,
                            start_y: int) -> str:
        """Key covers content + every crop param, so a hit means nothing changed.
        Ключ покрывает содержимое + все параметры: попадание = ничего не менялось."""
        key_string = (
            f"{image_hash}|{mask_hash}|{width}|{height}|{position}|{use_percentage}|"
            f"{maintain_aspect_ratio}|{inverse_crop_percent}|{start_x}|{start_y}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_image_hash(self, tensor: torch.Tensor) -> str:
        """Shape/dtype + evenly spread samples defeat identical-header collisions.
        Форма/dtype + равномерная выборка защищают от совпадения начала тензоров."""
        flat = tensor.reshape(-1)
        total = flat.numel()
        step = max(1, total // 4096)
        sample = flat[::step][:4096]
        meta = f"{tuple(tensor.shape)}|{tensor.dtype}|".encode()
        return hashlib.sha256(meta + sample.cpu().numpy().tobytes()).hexdigest()

    def _apply_inverse_crop(self, image: torch.Tensor, mask: Optional[torch.Tensor],
                            percent: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """Trim percent from each edge; returns offsets for crop_data.
        Обрезает процент с каждого края; возвращает смещения для crop_data."""
        if percent <= 0:
            return image, mask, 0, 0
        _, img_height, img_width, _ = image.shape
        crop_pixels_h = min(int(img_height * (percent / 100.0)), img_height // 2)
        crop_pixels_w = min(int(img_width * (percent / 100.0)), img_width // 2)
        if img_height - 2 * crop_pixels_h < 1 or img_width - 2 * crop_pixels_w < 1:
            return image, mask, 0, 0
        cropped_image = image[:, crop_pixels_h:img_height - crop_pixels_h,
                              crop_pixels_w:img_width - crop_pixels_w, :]
        cropped_mask = None
        if mask is not None:
            cropped_mask = mask[:, crop_pixels_h:img_height - crop_pixels_h,
                                crop_pixels_w:img_width - crop_pixels_w]
        return cropped_image, cropped_mask, crop_pixels_w, crop_pixels_h

    def _calculate_crop_size(self, img_width: int, img_height: int,
                             width: int, height: int, use_percentage: bool,
                             maintain_aspect_ratio: bool) -> Tuple[int, int]:
        """Final crop size (percent/aspect aware). / Итоговый размер кропа."""
        if use_percentage:
            crop_w = max(1, int(img_width * (width / 100.0)))
            crop_h = max(1, int(img_height * (height / 100.0)))
        else:
            crop_w, crop_h = width, height
        if maintain_aspect_ratio:
            original_aspect = img_width / img_height
            crop_aspect = crop_w / crop_h
            if crop_aspect > original_aspect:
                crop_h = int(crop_w / original_aspect)
            else:
                crop_w = int(crop_h * original_aspect)
        crop_w = max(1, min(crop_w, img_width))
        crop_h = max(1, min(crop_h, img_height))
        return crop_w, crop_h

    def _calculate_crop_position(self, img_width: int, img_height: int,
                                 crop_w: int, crop_h: int, position: str,
                                 start_x: int, start_y: int) -> Tuple[int, int]:
        """Crop origin by preset/custom, clamped. / Начало кропа по пресету/custom."""
        if position == "custom":
            x, y = start_x, start_y
        elif position == "top-left":
            x, y = 0, 0
        elif position == "top-center":
            x, y = (img_width - crop_w) // 2, 0
        elif position == "top-right":
            x, y = img_width - crop_w, 0
        elif position == "center-left":
            x, y = 0, (img_height - crop_h) // 2
        elif position == "center-right":
            x, y = img_width - crop_w, (img_height - crop_h) // 2
        elif position == "bottom-left":
            x, y = 0, img_height - crop_h
        elif position == "bottom-center":
            x, y = (img_width - crop_w) // 2, img_height - crop_h
        elif position == "bottom-right":
            x, y = img_width - crop_w, img_height - crop_h
        else:
            x, y = (img_width - crop_w) // 2, (img_height - crop_h) // 2
        x = max(0, min(x, img_width - crop_w))
        y = max(0, min(y, img_height - crop_h))
        return x, y

    def _make_crop_data(self, original: torch.Tensor, x: int, y: int,
                        w: int, h: int) -> Dict[str, Any]:
        """Stitch-compatible box clamped to the original; no clone (Stitch clones).
        Бокс для Stitch, ограниченный оригиналом; без клона (Stitch клонирует сам)."""
        orig_h, orig_w = original.shape[1], original.shape[2]
        bx = max(0, min(x, orig_w - 1))
        by = max(0, min(y, orig_h - 1))
        bw = max(1, min(w, orig_w - bx))
        bh = max(1, min(h, orig_h - by))
        return {"original_image": original, "x": bx, "y": by,
                "w": bw, "h": bh, "device": original.device}

    def crop_image(self, image: torch.Tensor, width: int, height: int,
                   position: str, mask: Optional[torch.Tensor] = None,
                   use_percentage: bool = False, maintain_aspect_ratio: bool = False,
                   inverse_crop_percent: int = 0, start_x: int = 0, start_y: int = 0,
                   device: str = "auto"
                   ) -> Tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor], int, int]:
        if image is None:
            raise ValueError("Image cannot be None / Изображение не может быть пустым")
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)} / "
                            f"Ожидается torch.Tensor, получен {type(image)}")
        if len(image.shape) != 4:
            raise ValueError(f"Expected 4D [B,H,W,C], got {image.shape} / "
                             f"Ожидается 4D [B,H,W,C], получена форма {image.shape}")
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Mask must be torch.Tensor, got {type(mask)} / "
                                f"Маска должна быть torch.Tensor, получен {type(mask)}")
            if len(mask.shape) != 3:
                raise ValueError(f"Expected 3D mask [B,H,W], got {mask.shape} / "
                                 f"Ожидается 3D маска [B,H,W], получена форма {mask.shape}")
            if mask.shape[:3] != image.shape[:3]:
                raise ValueError(f"Mask shape {mask.shape} doesn't match image {image.shape} / "
                                 f"Размеры маски {mask.shape} не совпадают с изображением {image.shape}")

        # Keep the untouched original for crop_data (Stitch pastes into it).
        # Сохраняем нетронутый оригинал для crop_data (Stitch вставляет в него).
        original_image = image
        original_device = image.device
        target_device = self._get_device(device)
        image = image.to(target_device)
        if mask is not None:
            mask = mask.to(target_device)

        image_hash = self._get_image_hash(image)
        mask_hash = self._get_image_hash(mask) if mask is not None else None
        cache_key = self._generate_cache_key(
            image_hash, mask_hash, width, height, position, use_percentage,
            maintain_aspect_ratio, inverse_crop_percent, start_x, start_y)
        if cache_key in self._cache:
            c_img, c_mask, c_w, c_h, c_x, c_y = self._cache[cache_key]
            crop_data = self._make_crop_data(original_image, c_x, c_y, c_w, c_h)
            return (crop_data, c_img.to(original_device),
                    c_mask.to(original_device) if c_mask is not None else None,
                    c_w, c_h)

        _, img_height, img_width, _ = image.shape
        if img_width < 1 or img_height < 1:
            raise ValueError(f"Invalid image size {img_width}x{img_height} / "
                             f"Некорректные размеры изображения {img_width}x{img_height}")

        # Inverse edge trim; offsets shift the crop_data box back to the original.
        # Инверсная обрезка краёв; смещения возвращают бокс crop_data к оригиналу.
        trim_x, trim_y = 0, 0
        if inverse_crop_percent > 0:
            image, mask, trim_x, trim_y = self._apply_inverse_crop(
                image, mask, inverse_crop_percent)
            _, img_height, img_width, _ = image.shape

        crop_w, crop_h = self._calculate_crop_size(
            img_width, img_height, width, height, use_percentage, maintain_aspect_ratio)
        x, y = self._calculate_crop_position(
            img_width, img_height, crop_w, crop_h, position, start_x, start_y)
        abs_x, abs_y = x + trim_x, y + trim_y

        cropped = image[:, y:y + crop_h, x:x + crop_w, :]
        cropped_mask = mask[:, y:y + crop_h, x:x + crop_w] if mask is not None else None

        if len(self._cache) >= self._max_cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[cache_key] = (
            cropped.clone(),
            cropped_mask.clone() if cropped_mask is not None else None,
            crop_w, crop_h, abs_x, abs_y)

        crop_data = self._make_crop_data(original_image, abs_x, abs_y, crop_w, crop_h)
        return (crop_data, cropped.to(original_device),
                cropped_mask.to(original_device) if cropped_mask is not None else None,
                crop_w, crop_h)


NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "AGSoft Image Crop": AGSoft_Image_Crop,
}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "AGSoft Image Crop": "🖼️✂️AGSoft Image Crop",
}