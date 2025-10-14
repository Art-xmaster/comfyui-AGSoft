# ComfyUI/custom_nodes/comfyui-AGSoft/AGSoft_Mask_Fix.py

import torch
import numpy as np
import cv2
from typing import Tuple

# ==============================
# Утилита: convert_mask_to_image (для preview)
# ==============================

def convert_mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    """
    Конвертирует MASK в IMAGE (B, H, W, 3) с диапазоном [0.0, 1.0]
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # (1, H, W)
    return mask.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B, H, W, 3)

# ==============================
# НОДА 1: AGSoft_Mask_Fix (с feather)
# ==============================

class AGSoft_Mask_Fix:
    """
    Расширенная обработка маски:
    - эрозия/дилатация,
    - заполнение дыр,
    - удаление шума,
    - сглаживание,
    - расширение маски (expand),
    - растушёвка краёв (feather),
    - бинаризация по порогу,
    - инверсия.

    Поддержка CPU/CUDA.
    Превью выводится через convert_mask_to_image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_dilate": ("INT", {
                    "default": 0,
                    "min": -256,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Положительное значение — расширение маски (дилатация), отрицательное — сужение (эрозия)."
                }),
                "fill_holes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Заполняет чёрные дыры внутри белых областей маски (например, глаза внутри лица)."
                }),
                "remove_isolated_pixels": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Удаляет мелкие изолированные области (шум). Чем выше — тем агрессивнее очистка."
                }),
                "smooth": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Многократное сглаживание с бинаризацией. Эффективно убирает 'лесенки'."
                }),
                "expand": ("INT", {
                    "default": 0,
                    "min": -256,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Дополнительное расширение/сужение маски. При 'tapered_corners=True' — плавное."
                }),
                "tapered_corners": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если включено — расширение/сужение происходит с плавными (скруглёнными) краями."
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Растушёвка краёв маски. Создаёт градиентный переход на указанное количество пикселей."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Порог бинаризации после всех операций. Значения >= порога → белые (1.0)."
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Инвертирует финальную маску: чёрное становится белым и наоборот."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений. 'cuda' ускоряет обработку на GPU, но требует совместимой видеокарты."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask_out", "convert_mask_to_image")  # ← изменено
    FUNCTION = "fix_mask"
    CATEGORY = "AGSoft/Mask"

    def fix_mask(
        self,
        mask: torch.Tensor,
        erode_dilate: int,
        fill_holes: bool,
        remove_isolated_pixels: int,
        smooth: int,
        expand: int,
        tapered_corners: bool,
        feather: int,
        threshold: float,
        invert: bool,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            mask_cpu = mask.to("cpu")
            mask_np = mask_cpu.numpy()

            if mask_np.ndim == 2:
                mask_np = mask_np[None, ...]

            batch_size = mask_np.shape[0]
            processed_masks = []

            for i in range(batch_size):
                m = mask_np[i].copy()
                m_uint8 = (np.clip(m, 0.0, 1.0) * 255).astype(np.uint8)

                # --- 1. Эрозия / Дилатация ---
                if erode_dilate != 0:
                    kernel_size = max(1, abs(erode_dilate) // 8 + 1)
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    if erode_dilate > 0:
                        m_uint8 = cv2.dilate(m_uint8, kernel, iterations=1)
                    else:
                        m_uint8 = cv2.erode(m_uint8, kernel, iterations=1)

                # --- 2. Заполнение дыр ---
                if fill_holes:
                    h, w = m_uint8.shape
                    im_flood = m_uint8.copy()
                    mask_flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
                    cv2.floodFill(im_flood, mask_flood, (0, 0), 255)
                    im_flood_inv = cv2.bitwise_not(im_flood)
                    m_uint8 = cv2.bitwise_or(m_uint8, im_flood_inv)

                # --- 3. Удаление изолированных пикселей ---
                if remove_isolated_pixels > 0:
                    k = min(2 * remove_isolated_pixels + 1, 15)
                    kernel = np.ones((k, k), dtype=np.uint8)
                    m_uint8 = cv2.morphologyEx(m_uint8, cv2.MORPH_OPEN, kernel)

                # --- 4. Сглаживание ---
                smooth_iter = min(smooth // 16, 8)
                for _ in range(smooth_iter):
                    m_uint8 = cv2.GaussianBlur(m_uint8, (3, 3), 0)
                    _, m_uint8 = cv2.threshold(m_uint8, 127, 255, cv2.THRESH_BINARY)

                # --- 5. Expand ---
                if expand != 0:
                    if tapered_corners:
                        sigma = max(0.5, abs(expand) * 0.1)
                        blurred = cv2.GaussianBlur(m_uint8, (0, 0), sigma)
                        thr = 127 - (64 if expand > 0 else -64)
                        thr = np.clip(thr, 0, 255)
                        _, m_uint8 = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)
                    else:
                        kernel_size = max(1, abs(expand) // 4 + 1)
                        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                        if expand > 0:
                            m_uint8 = cv2.dilate(m_uint8, kernel, iterations=1)
                        else:
                            m_uint8 = cv2.erode(m_uint8, kernel, iterations=1)

                # --- 6. Feather (растушёвка) ---
                if feather > 0:
                    dist = cv2.distanceTransform(m_uint8, cv2.DIST_L2, 5)
                    dist = np.clip(dist / feather, 0, 1)
                    m_float = dist.astype(np.float32)
                else:
                    m_float = m_uint8.astype(np.float32) / 255.0

                # --- 7. Бинаризация ---
                m_binary = (m_float >= threshold).astype(np.float32)

                # --- 8. Инверсия ---
                if invert:
                    m_binary = 1.0 - m_binary

                processed_masks.append(m_binary)

            result_mask = torch.from_numpy(np.stack(processed_masks, axis=0)).to(target_device)
            mask_preview = convert_mask_to_image(result_mask)

            return (result_mask, mask_preview)

        except Exception as e:
            raise RuntimeError(f"AGSoft_Mask_Fix error: {e}") from e


# ==============================
# НОДА 2: AGSoft_Mask_Blur
# ==============================

class AGSoft_Mask_Blur:
    """
    Чистое размытие маски с сохранением формы.
    Аналог стандартной ноды 'Mask Blur', но с поддержкой device и превью.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 0.5,
                    "tooltip": "Радиус гауссова размытия. Чем выше — тем мягче края."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений. 'cuda' ускоряет обработку на GPU, но требует совместимой видеокарты."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask_out", "convert_mask_to_image")  # ← изменено
    FUNCTION = "blur_mask"
    CATEGORY = "AGSoft/Mask"

    def blur_mask(
        self,
        mask: torch.Tensor,
        blur_radius: float,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if blur_radius <= 0:
                target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                mask_out = mask.to(target_device)
                mask_preview = convert_mask_to_image(mask_out)
                return (mask_out, mask_preview)

            mask_cpu = mask.to("cpu").numpy()
            if mask_cpu.ndim == 2:
                mask_cpu = mask_cpu[None, ...]

            batch_size = mask_cpu.shape[0]
            blurred_masks = []

            for i in range(batch_size):
                m = mask_cpu[i].copy()
                m_blurred = cv2.GaussianBlur(m, (0, 0), blur_radius, borderType=cv2.BORDER_REPLICATE)
                blurred_masks.append(m_blurred)

            result_mask = torch.from_numpy(np.stack(blurred_masks, axis=0))
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            result_mask = result_mask.to(target_device)
            mask_preview = convert_mask_to_image(result_mask)

            return (result_mask, mask_preview)

        except Exception as e:
            raise RuntimeError(f"AGSoft_Mask_Blur error: {e}") from e


# ==============================
# НОДА 3: AGSoft_Mask_Composite
# ==============================

class AGSoft_Mask_Composite:
    """
    Композитинг двух масок: сложение, вычитание, умножение, максимум, минимум.
    Полезно для объединения или вычитания областей.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "operation": (["add", "subtract", "multiply", "max", "min"], {
                    "default": "add",
                    "tooltip": "Операция над двумя масками: add — объединение, subtract — вычитание, multiply — пересечение, max/min — выбор наибольшего/наименьшего значения."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений. 'cuda' ускоряет обработку на GPU, но требует совместимой видеокарты."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask_out", "convert_mask_to_image")  # ← изменено
    FUNCTION = "composite_mask"
    CATEGORY = "AGSoft/Mask"

    def composite_mask(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        operation: str,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

            mask1 = mask1.to(target_device)
            mask2 = mask2.to(target_device)

            if mask1.shape != mask2.shape:
                h, w = mask1.shape[-2], mask1.shape[-1]
                mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)

            if operation == "add":
                result = torch.clamp(mask1 + mask2, 0.0, 1.0)
            elif operation == "subtract":
                result = torch.clamp(mask1 - mask2, 0.0, 1.0)
            elif operation == "multiply":
                result = mask1 * mask2
            elif operation == "max":
                result = torch.max(mask1, mask2)
            elif operation == "min":
                result = torch.min(mask1, mask2)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            mask_preview = convert_mask_to_image(result)

            return (result, mask_preview)

        except Exception as e:
            raise RuntimeError(f"AGSoft_Mask_Composite error: {e}") from e


# ==============================
# НОДА 4: AGSoft_Mask_From_Color
# ==============================

class AGSoft_Mask_From_Color:
    """
    Создаёт маску по цвету из изображения.
    Используется для выделения объектов по цвету (например, зелёный экран).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {
                    "default": "0,255,0",
                    "tooltip": "Целевой цвет в формате R,G,B (например, '0,255,0' для зелёного)."
                }),
                "tolerance": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Допустимое отклонение от целевого цвета. Чем выше — тем больше цветов попадает в маску."
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Инвертирует финальную маску: чёрное становится белым и наоборот."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений. 'cuda' ускоряет обработку на GPU, но требует совместимой видеокарты."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask_out", "convert_mask_to_image")  # ← изменено
    FUNCTION = "create_mask_from_color"
    CATEGORY = "AGSoft/Mask"

    def create_mask_from_color(
        self,
        image: torch.Tensor,
        target_color: str,
        tolerance: int,
        invert: bool,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            image_cpu = image.to("cpu").numpy()  # (B, H, W, 3)

            try:
                r, g, b = map(int, target_color.split(','))
            except Exception:
                raise ValueError("Target color must be in format 'R,G,B'")

            batch_size = image_cpu.shape[0]
            masks = []

            for i in range(batch_size):
                img = image_cpu[i]  # (H, W, 3), float32 [0.0, 1.0]
                img_uint8 = (img * 255).astype(np.uint8)
                target = np.array([b, g, r], dtype=np.uint8)  # OpenCV uses BGR
                diff = np.linalg.norm(img_uint8.astype(np.float32) - target, axis=2)
                mask = (diff <= tolerance).astype(np.float32)
                masks.append(mask)

            result_mask = torch.from_numpy(np.stack(masks, axis=0)).to(target_device)
            if invert:
                result_mask = 1.0 - result_mask

            mask_preview = convert_mask_to_image(result_mask)

            return (result_mask, mask_preview)

        except Exception as e:
            raise RuntimeError(f"AGSoft_Mask_From_Color error: {e}") from e


# ==============================
# РЕГИСТРАЦИЯ ВСЕХ НОД
# ==============================

NODE_CLASS_MAPPINGS = {
    "AGSoft_Mask_Fix": AGSoft_Mask_Fix,
    "AGSoft_Mask_Blur": AGSoft_Mask_Blur,
    "AGSoft_Mask_Composite": AGSoft_Mask_Composite,
    "AGSoft_Mask_From_Color": AGSoft_Mask_From_Color,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Mask_Fix": "AGSoft Mask Fix",
    "AGSoft_Mask_Blur": "AGSoft Mask Blur",
    "AGSoft_Mask_Composite": "AGSoft Mask Composite",
    "AGSoft_Mask_From_Color": "AGSoft Mask From Color",
}
