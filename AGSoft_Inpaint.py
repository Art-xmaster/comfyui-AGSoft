"""
Ноды для подготовки изображения к inpainting и последующей вставки результата обратно.

Основные возможности:
- Обрезка изображения вокруг активной области маски с учётом отступа (padding)
- Применение размытия краёв маски (feathering) для плавного перехода
- Выравнивание размеров обрезки до кратного числа (например, 64) — требуется многими моделями
- Опциональный апскейл изображения перед inpainting (маска не масштабируется!)
- Принудительное квадратное обрезание (force_square) для совместимости с моделями
- Поддержка нескольких режимов апскейла: точный размер, масштаб, вписывание по ширине/высоте
- Два режима сшивания: alpha-blend (плавное смешивание) и Poisson blending (сохранение освещения и текстуры)
- Автоматический даунскейл после inpainting и корректное сшивание в оригинальное изображение

Важно:
- Маска всегда обрабатывается в оригинальном разрешении обрезки — это гарантирует точное сшивание.
- Poisson blending использует абсолютные координаты центра маски в оригинальном изображении.
- Все операции ресайза поддерживают Lanczos (через PIL) и интерполяцию через PyTorch.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any
import torch.nn.functional as F
from PIL import Image

# Проверка наличия OpenCV для Poisson blending
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _apply_feathering(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Применяет гауссово размытие (feathering) к краям маски.
    
    Параметры:
        mask (torch.Tensor): Маска формы [H, W], значения в [0, 1]
        radius (int): Радиус размытия в пикселях. При radius <= 0 возвращается исходная маска.
    
    Возвращает:
        torch.Tensor: Размытая маска формы [H, W]
    """
    if radius <= 0:
        return mask

    # Подготавливаем маску для свёртки: [1, 1, H, W]
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    kernel_size = 2 * radius + 1
    sigma = float(radius / 3.0)
    
    # Создание 1D гауссианы
    coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Создание 2D ядра через внешнее произведение
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel = kernel_2d[None, None, :, :]  # [1, 1, K, K]

    # Применение свёртки с отражением на границах
    padding = radius
    padded_mask = F.pad(mask_4d, (padding, padding, padding, padding), mode='reflect')
    feathered = F.conv2d(padded_mask, kernel, padding=0)
    return feathered.squeeze(0).squeeze(0)


def _make_multiple(value: int, multiple: int) -> int:
    """
    Округляет значение ВВЕРХ до ближайшего числа, кратного заданному.
    
    Используется для выравнивания размеров под требования моделей (например, Stable Diffusion требует кратность 64).
    
    Параметры:
        value (int): Исходное значение
        multiple (int): Число, кратное которому нужно округлить
    
    Возвращает:
        int: Округлённое значение
    """
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _resize_tensor(tensor: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    """
    Изменяет размер тензора с поддержкой 2D (маски) и 3D (изображения).
    
    Для Lanczos используется PIL (высококачественный ресайз).
    Для остальных методов — F.interpolate из PyTorch.
    
    Параметры:
        tensor (torch.Tensor): Входной тензор [H, W] или [H, W, C]
        width (int): Целевая ширина
        height (int): Целевая высота
        method (str): Метод интерполяции: "lanczos", "bicubic", "bilinear", "nearest"
    
    Возвращает:
        torch.Tensor: Ресайзнутый тензор той же размерности
    """
    if tensor.ndim not in (2, 3):
        raise ValueError(f"Ожидался 2D или 3D тензор, получен {tensor.ndim}D")

    current_h, current_w = tensor.shape[:2]
    if current_h == height and current_w == width:
        return tensor

    if method == "lanczos":
        # Используем PIL для Lanczos — поддерживает grayscale и RGB
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        resized_pil = pil_img.resize((width, height), Image.LANCZOS)
        resized_np = np.array(resized_pil).astype(np.float32) / 255.0
        return torch.from_numpy(resized_np).to(tensor.device, dtype=tensor.dtype)
    else:
        # Используем F.interpolate — требуется 4D тензор [B, C, H, W]
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
        }
        mode = mode_map.get(method, "bilinear")
        
        if tensor.ndim == 2:
            # Маска: [H, W] → [1, 1, H, W]
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
            resized_4d = F.interpolate(tensor_4d, size=(height, width), mode=mode, align_corners=False)
            return resized_4d.squeeze(0).squeeze(0)  # [H, W]
        else:
            # Изображение: [H, W, C] → [1, C, H, W]
            tensor_4d = tensor.permute(2, 0, 1).unsqueeze(0)
            resized_4d = F.interpolate(tensor_4d, size=(height, width), mode=mode, align_corners=False)
            return resized_4d.squeeze(0).permute(1, 2, 0)  # [H, W, C]


class AGSoft_Inpaint_Crop:
    """
    Обрезает изображение и маску вокруг активной области с учётом отступа, выравнивания и feathering.
    
    Поддерживает гибкий апскейл изображения (маска остаётся в оригинальном разрешении).
    Результат можно использовать в любом inpainting-пайплайне.
    """
    DESCRIPTION = "Обрезает изображение вокруг маски с отступом, выравниванием и опциональным апскейлом."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Исходное изображение в формате [B, H, W, C]. Обычно подаётся от Load Image."
                }),
                "mask": ("MASK", {
                    "tooltip": "Маска inpainting в формате [B, H, W]. Белое (1.0) — область для редактирования."
                }),
                "padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Дополнительный отступ (в пикселях) вокруг маски для контекста. Увеличивает область обрезки."
                }),
                "feathering": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Радиус размытия краёв маски. Улучшает плавность перехода при сшивании. Применяется в оригинальном разрешении обрезки."
                }),
                "multiple_of": (["2", "4", "8", "16", "32", "64", "112", "128"], {
                    "default": "64",
                    "tooltip": "Размеры обрезки будут округлены вверх до кратного этому числу. Требуется некоторыми моделями (например, SDXL — 64, SD1.5 — 8)."
                }),
                "force_square": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если включено, обрезка будет принудительно квадратной (сторона = max(ширина, высота)). Рекомендуется для большинства inpainting-моделей."
                }),
                "upscale_mode": (["disabled", "to_size", "scale_by", "fit_to_width", "fit_to_height"], {
                    "default": "disabled",
                    "tooltip": (
                        "Режим апскейла изображения перед inpainting:\n"
                        "• disabled — без апскейла (рекомендуется для больших изображений)\n"
                        "• to_size — задать точные целевые размеры\n"
                        "• scale_by — масштабировать на коэффициент (сохраняя пропорции)\n"
                        "• fit_to_width — изменить ширину до заданной, высоту — пропорционально\n"
                        "• fit_to_height — изменить высоту до заданной, ширину — пропорционально"
                    )
                }),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Целевая ширина изображения после апскейла. Используется в режимах: to_size, fit_to_width."
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Целевая высота изображения после апскейла. Используется в режимах: to_size, fit_to_height."
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Коэффициент масштабирования. Например, 2.0 — увеличить в 2 раза. Используется только в режиме scale_by."
                }),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Метод интерполяции при апскейле изображения:\n• lanczos — наилучшее качество для изображений\n• bicubic — хорошее качество, быстрее Lanczos\n• bilinear — быстрее, но ниже качество\n• nearest — без сглаживания (для пиксель-арта)"
                }),
            },
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "AGSoft/Inpainting"

    def crop(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        padding: int,
        feathering: int,
        multiple_of: str,
        force_square: bool,
        upscale_mode: str,
        target_width: int,
        target_height: int,
        scale_factor: float,
        upscale_method: str,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, int, int]:
        """
        Выполняет обрезку изображения и маски с последующей подготовкой к inpainting.
        
        Возвращает данные для сшивания (stitcher), обрезанное изображение, маску и её размеры.
        """
        # Работаем с первым изображением в батче (ComfyUI обычно использует batch=1 для inpainting)
        orig_image = image[0]
        orig_mask = mask[0]
        multiple = int(multiple_of)

        # Находим активную область маски
        mask_np = orig_mask.cpu().numpy()
        coords = np.argwhere(mask_np > 0)

        # Если маска пустая — возвращаем всё как есть
        if coords.size == 0:
            stitcher = {
                "original_image": orig_image.clone(),
                "crop_offset": (0, 0),
                "cropped_shape": (orig_image.shape[0], orig_image.shape[1]),
                "feathered_mask": torch.zeros_like(orig_mask),
                "was_upscaled": False,
                "device": image.device,
            }
            H, W = orig_image.shape[0], orig_image.shape[1]
            return (stitcher, image, mask, W, H)

        # Определяем bounding box маски
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        H, W = orig_image.shape[0], orig_image.shape[1]

        # Добавляем отступ
        y1 = max(0, int(y_min) - padding)
        x1 = max(0, int(x_min) - padding)
        y2 = min(H, int(y_max) + padding)
        x2 = min(W, int(x_max) + padding)

        raw_crop_h = y2 - y1
        raw_crop_w = x2 - x1

        # Выравниваем до кратного
        aligned_h = _make_multiple(raw_crop_h, multiple)
        aligned_w = _make_multiple(raw_crop_w, multiple)

        # Центрируем обрезку
        dh = aligned_h - raw_crop_h
        dw = aligned_w - raw_crop_w
        y1_adj = max(0, y1 - dh // 2)
        x1_adj = max(0, x1 - dw // 2)

        # Коррекция, если вышли за границы изображения
        if y1_adj + aligned_h > H:
            y1_adj = H - aligned_h
        if x1_adj + aligned_w > W:
            x1_adj = W - aligned_w

        final_y2 = y1_adj + aligned_h
        final_x2 = x1_adj + aligned_w

        # Принудительное квадратное обрезание
        if force_square:
            current_h = final_y2 - y1_adj
            current_w = final_x2 - x1_adj
            max_side = max(current_h, current_w)
            max_side = _make_multiple(max_side, multiple)
            max_side = max(max_side, multiple)  # избегаем нулевого размера

            pad_h = max_side - current_h
            pad_w = max_side - current_w
            y1_new = y1_adj - pad_h // 2
            x1_new = x1_adj - pad_w // 2

            # Коррекция координат, чтобы не выйти за границы
            if y1_new < 0:
                y1_new = 0
            if x1_new < 0:
                x1_new = 0
            if y1_new + max_side > H:
                y1_new = H - max_side
            if x1_new + max_side > W:
                x1_new = W - max_side

            y1_adj = y1_new
            x1_adj = x1_new
            final_y2 = y1_adj + max_side
            final_x2 = x1_adj + max_side

        # Выполняем обрезку
        cropped_img = orig_image[y1_adj:final_y2, x1_adj:final_x2, :]
        cropped_msk = orig_mask[y1_adj:final_y2, x1_adj:final_x2]

        orig_crop_h, orig_crop_w = cropped_img.shape[0], cropped_img.shape[1]

        # Применяем feathering в оригинальном разрешении обрезки
        feathered_msk = _apply_feathering(cropped_msk, feathering) if feathering > 0 else cropped_msk

        # Опциональный апскейл изображения (маска не апскейлится!)
        new_w, new_h = orig_crop_w, orig_crop_h
        was_upscaled = False

        if upscale_mode != "disabled":
            if upscale_mode == "to_size":
                new_w, new_h = target_width, target_height
            elif upscale_mode == "scale_by":
                new_w = int(orig_crop_w * scale_factor)
                new_h = int(orig_crop_h * scale_factor)
            elif upscale_mode == "fit_to_width":
                new_w = target_width
                new_h = int(orig_crop_h * (target_width / orig_crop_w)) if orig_crop_w > 0 else orig_crop_h
            elif upscale_mode == "fit_to_height":
                new_h = target_height
                new_w = int(orig_crop_w * (target_height / orig_crop_h)) if orig_crop_h > 0 else orig_crop_w

            # Если включён force_square — делаем и апскейл квадратным
            if force_square:
                max_side_upscale = max(new_w, new_h)
                max_side_upscale = _make_multiple(max_side_upscale, multiple)
                max_side_upscale = max(max_side_upscale, multiple)
                new_w = new_h = max_side_upscale

            # Апскейлим, только если размер изменился
            if new_w != orig_crop_w or new_h != orig_crop_h:
                cropped_img = _resize_tensor(cropped_img, new_w, new_h, upscale_method)
                was_upscaled = True

        # Подготавливаем данные для сшивания
        stitcher = {
            "original_image": orig_image.clone(),
            "crop_offset": (y1_adj, x1_adj),
            "cropped_shape": (orig_crop_h, orig_crop_w),
            "feathered_mask": feathered_msk,
            "was_upscaled": was_upscaled,
            "device": image.device,
        }

        out_h, out_w = cropped_img.shape[0], cropped_img.shape[1]
        return (
            stitcher,
            cropped_img.unsqueeze(0),
            cropped_msk.unsqueeze(0),
            out_w,
            out_h,
        )


class AGSoft_Inpaint_Stitch:
    """
    Вставляет результат inpainting обратно в оригинальное изображение.
    
    Поддерживает два режима сшивания:
    - alpha_blend: плавное смешивание по маске (универсальный, всегда работает)
    - poisson_blend: продвинутое сшивание с сохранением градиентов (требует opencv-python)
    
    Автоматически выполняет даунскейл, если изображение было апскейлено в Crop-ноде.
    """
    DESCRIPTION = "Сшивает inpainted-изображение с оригиналом. Поддерживает alpha и Poisson blending."

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        blend_modes = ["alpha_blend"]
        if CV2_AVAILABLE:
            blend_modes.append("poisson_blend")
        else:
            blend_modes.append("poisson_blend (недоступен)")

        return {
            "required": {
                "stitcher": ("STITCHER", {
                    "tooltip": "Данные от ноды AGSoft Inpaint Crop. Содержит оригинал, координаты обрезки, маску и метаданные."
                }),
                "inpainted_image": ("IMAGE", {
                    "tooltip": "Результат inpainting-модели (например, от KSampler). Может быть в увеличенном разрешении."
                }),
                "blend_mode": (blend_modes, {
                    "default": "alpha_blend",
                    "tooltip": (
                        "Режим сшивания:\n"
                        "• alpha_blend — плавное смешивание по маске. Работает всегда.\n"
                        "• poisson_blend — продвинутое сшивание через OpenCV. Сохраняет освещение и текстуру. Требует установки opencv-python."
                    )
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "AGSoft/Inpainting"

    def stitch(self, stitcher: Dict[str, Any], inpainted_image: torch.Tensor, blend_mode: str) -> Tuple[torch.Tensor,]:
        """
        Выполняет сшивание inpainted-изображения с оригиналом.
        
        Обрабатывает как апскейленные, так и неапскейленные случаи.
        Для Poisson blending использует абсолютные координаты центра маски.
        """
        if blend_mode == "poisson_blend (недоступен)":
            blend_mode = "alpha_blend"

        original_image = stitcher["original_image"]
        offset_y, offset_x = stitcher["crop_offset"]
        orig_crop_h, orig_crop_w = stitcher["cropped_shape"]
        feathered_mask = stitcher["feathered_mask"]
        was_upscaled = stitcher.get("was_upscaled", False)
        device = stitcher["device"]

        inpainted = inpainted_image[0]

        # Гарантируем, что inpainted имеет размер оригинальной обрезки
        if was_upscaled:
            inpainted = _resize_tensor(inpainted, orig_crop_w, orig_crop_h, "lanczos")
        else:
            # Модель могла изменить размер — приводим к ожидаемому
            if inpainted.shape[0] != orig_crop_h or inpainted.shape[1] != orig_crop_w:
                inpainted = _resize_tensor(inpainted, orig_crop_w, orig_crop_h, "lanczos")

        result = original_image.clone()

        # Извлекаем соответствующие регионы
        orig_region = original_image[offset_y:offset_y+orig_crop_h, offset_x:offset_x+orig_crop_w, :].cpu().numpy()
        inp_region = inpainted[:orig_crop_h, :orig_crop_w, :].cpu().numpy()
        mask_region = feathered_mask[:orig_crop_h, :orig_crop_w].cpu().numpy()

        # Убедимся, что у изображений 3 канала (на случай grayscale)
        if orig_region.shape[2] == 1:
            orig_region = np.repeat(orig_region, 3, axis=2)
        if inp_region.shape[2] == 1:
            inp_region = np.repeat(inp_region, 3, axis=2)

        if blend_mode == "poisson_blend" and CV2_AVAILABLE:
            try:
                # Подготавливаем маску для OpenCV
                if mask_region.ndim == 2:
                    mask_for_poisson = mask_region
                else:
                    mask_for_poisson = mask_region.max(axis=2)

                # Проверка на пустую маску
                if mask_for_poisson.max() < 0.01:
                    mask_3ch = mask_region[..., None] if mask_region.ndim == 2 else mask_region
                    blended_np = orig_region * (1.0 - mask_3ch) + inp_region * mask_3ch
                else:
                    coords = np.argwhere(mask_for_poisson > 0.01)
                    if coords.size == 0:
                        mask_3ch = mask_region[..., None] if mask_region.ndim == 2 else mask_region
                        blended_np = orig_region * (1.0 - mask_3ch) + inp_region * mask_3ch
                    else:
                        # Центр маски ВНУТРИ региона обрезки
                        center_y_local = int(coords[:, 0].mean())
                        center_x_local = int(coords[:, 1].mean())

                        # АБСОЛЮТНЫЙ центр в координатах оригинального изображения
                        center_y_abs = offset_y + center_y_local
                        center_x_abs = offset_x + center_x_local
                        center = (center_x_abs, center_y_abs)

                        # Подготовка данных для OpenCV (uint8, [0,255])
                        src_u8 = (np.clip(inp_region, 0.0, 1.0) * 255).astype(np.uint8)
                        tgt_u8 = (np.clip(orig_region, 0.0, 1.0) * 255).astype(np.uint8)
                        mask_u8 = (np.clip(mask_for_poisson, 0.0, 1.0) * 255).astype(np.uint8)

                        # Выполнение Poisson blending
                        blended_u8 = cv2.seamlessClone(src_u8, tgt_u8, mask_u8, center, cv2.NORMAL_CLONE)
                        blended_np = blended_u8.astype(np.float32) / 255.0

            except Exception:
                # Fallback на alpha-blend при любой ошибке
                mask_3ch = mask_region[..., None] if mask_region.ndim == 2 else mask_region
                blended_np = orig_region * (1.0 - mask_3ch) + inp_region * mask_3ch
        else:
            # Alpha blending
            mask_3ch = mask_region[..., None] if mask_region.ndim == 2 else mask_region
            blended_np = orig_region * (1.0 - mask_3ch) + inp_region * mask_3ch

        # Конвертация обратно в тензор и вставка в результат
        blended_tensor = torch.from_numpy(blended_np).to(device=device, dtype=original_image.dtype)
        result[offset_y:offset_y+orig_crop_h, offset_x:offset_x+orig_crop_w, :] = blended_tensor

        return (result.unsqueeze(0),)


# === РЕГИСТРАЦИЯ НОД ===
NODE_CLASS_MAPPINGS = {
    "AGSoft_Inpaint_Crop": AGSoft_Inpaint_Crop,
    "AGSoft_Inpaint_Stitch": AGSoft_Inpaint_Stitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Inpaint_Crop": "AGSoft Inpaint Crop",
    "AGSoft_Inpaint_Stitch": "AGSoft Inpaint Stitch",
}