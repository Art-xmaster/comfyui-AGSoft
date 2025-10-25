# AGSoft_Img_Res_MP.py
# Автор: AGSoft
# Дата: 26 октября 2025 г.

"""
Нода для изменения размера изображения с сохранением пропорций.
Позволяет задавать целевой размер в мегапикселях (MP), выбрать метод интерполяции,
округлить размеры до кратного значения, определить стратегию обработки пропорций
(crop/pad), выбрать цвет паддинга и позицию изображения.
"""

import torch
import numpy as np
from PIL import Image, ImageOps

# Словарь методов интерполяции для PIL
INTERPOLATION_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "area": Image.BOX,
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS
}

# Стратегии обработки пропорций
FIT_MODES = ["crop", "pad"]

# Расширенный список цветов паддинга в HEX (как вы просили)
PADDING_COLORS_HEX = {
    "black": "#000000",
    "white": "#FFFFFF",
    "gray": "#808080",
    "silver": "#C0C0C0",
    "light_gray": "#D3D3D3",
    "dark_gray": "#A9A9A9",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "orange": "#FFA500",
    "pink": "#FFC0CB",
    "brown": "#A52A2A",
    "purple": "#800080",
    "violet": "#EE82EE",
    "indigo": "#4B0082",
    "teal": "#008080",
    "navy": "#000080",
    "olive": "#808000",
    "maroon": "#800000",
    "dark_blue": "#00008B",
    "light_blue": "#ADD8E6",
    "light_green": "#90EE90",
    "dark_green": "#006400",
    "transparent": "transparent"  # специальное значение
}

# Позиции для центрирования (centering в PIL)
POSITIONS = {
    "center": (0.5, 0.5),
    "top-left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top-right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom-right": (1.0, 1.0),
}


def hex_to_rgb_or_rgba(hex_color: str, has_alpha: bool = False):
    """Преобразует HEX-цвет в кортеж RGB или RGBA для PIL."""
    if hex_color == "transparent":
        return (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb + (255,) if has_alpha else rgb
    elif len(hex_color) == 8:
        rgba = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
        return rgba
    else:
        return (0, 0, 0)  # fallback


class AGSoft_Img_Res_MP:
    """
    Нода для изменения размера изображения с сохранением пропорций.

    Эта нода позволяет гибко контролировать размер выходного изображения.
    Вы задаёте желаемое количество мегапикселей (MP), а нода автоматически
    вычисляет новый размер, сохраняя соотношение сторон исходного изображения.

    Параметр 'multiple_of' позволяет округлить итоговую ширину и высоту
    до ближайшего числа, кратного указанному значению (например, 64 или 8).
    Это часто требуется для совместимости с некоторыми моделями генерации изображений.

    Ключевой параметр 'fit_mode' определяет, что делать, если после округления
    до кратности пропорции немного сбиваются:
    - 'crop': Изображение будет масштабировано так, чтобы заполнить весь целевой
      прямоугольник, и излишки будут обрезаны. Никаких пустых полей не будет,
      но часть изображения может быть потеряна.
    - 'pad': Изображение будет полностью помещено в целевой прямоугольник,
      а пустые области (по краям) будут заполнены выбранным цветом.

    Параметр 'padding_color' позволяет выбрать цвет фона при 'fit_mode=pad'.
    Параметр 'position' определяет, где будет располагаться изображение:
    в центре, по углам или по краям.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Входное изображение или батч изображений для изменения размера."
                }),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 50.0,
                    "step": 0.1,
                    "tooltip": "Целевое разрешение изображения в мегапикселях (MP).\n"
                               "Например, 1.0 MP = 1000000 пикселей (примерно 1024x1024)."
                }),
                "interpolation": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "lanczos",
                    "tooltip": "Метод интерполяции при изменении размера:\n"
                               "• nearest — самый быстрый, но низкое качество (пиксельный).\n"
                               "• bilinear — сглаженный, быстрый, подходит для preview.\n"
                               "• bicubic — более плавный, чем bilinear, хорош для фото.\n"
                               "• area (BOX) — оптимален для уменьшения изображений (сохраняет детали).\n"
                               "• lanczos — наивысшее качество, но медленнее; идеален для финального рендера.\n"
                               "• nearest-exact — как 'nearest', но с улучшённой точностью (редко используется)."
                }),
                "multiple_of": ([0, 1, 2, 4, 8, 16, 32, 64, 112, 128], {
                    "default": 0,
                    "tooltip": "Округляет итоговую ширину и высоту до кратного этому числу.\n"
                               "0 означает отключение. Полезно для совместимости с моделями (например, SDXL любит 1024, что кратно 64)."
                }),
                "fit_mode": (FIT_MODES, {
                    "default": "pad",
                    "tooltip": "Стратегия обработки пропорций после округления до кратности:\n"
                               "- 'crop': Обрезает изображение, чтобы заполнить всё пространство (может потерять края).\n"
                               "- 'pad': Добавляет поля, чтобы всё изображение поместилось (без потерь)."
                }),
                "padding_color": (list(PADDING_COLORS_HEX.keys()), {
                    "default": "black",
                    "tooltip": "Цвет полей при 'fit_mode=pad'.\n"
                               "Для прозрачности изображение должно быть в формате RGBA (4 канала)."
                }),
                "position": (list(POSITIONS.keys()), {
                    "default": "center",
                    "tooltip": "Позиция изображения при 'fit_mode=pad' или 'fit_mode=crop'.\n"
                               "Определяет, какая часть изображения будет видна после операции."
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "resize_image"
    CATEGORY = "AGSoft/Image"

    def resize_image(
        self,
        image: torch.Tensor,
        target_megapixels: float,
        interpolation: str,
        multiple_of: int,
        fit_mode: str,
        padding_color: str,
        position: str
    ):
        batch_size, orig_h, orig_w, channels = image.shape
        if batch_size == 0:
            return (image, orig_w, orig_h)

        img_np_batch = image.cpu().numpy()
        resized_images = []
        final_width, final_height = 0, 0

        centering_tuple = POSITIONS[position]
        hex_color = PADDING_COLORS_HEX[padding_color]
        is_transparent = (padding_color == "transparent")

        for i in range(batch_size):
            # Определяем режим PIL
            has_alpha = (channels == 4)
            pil_mode = 'RGBA' if has_alpha else 'RGB'

            # Конвертация в PIL
            img_array = (img_np_batch[i] * 255).astype(np.uint8)
            if has_alpha:
                pil_img = Image.fromarray(img_array, mode='RGBA')
            else:
                pil_img = Image.fromarray(img_array, mode='RGB')

            # --- Шаг 1: Вычисление целевого размера контейнера ---
            target_pixels = max(1, int(target_megapixels * 1_000_000))
            if orig_w <= 0 or orig_h <= 0:
                orig_w, orig_h = 1, 1

            ratio = (target_pixels / (orig_w * orig_h)) ** 0.5
            base_w = max(1, int(orig_w * ratio))
            base_h = max(1, int(orig_h * ratio))

            if multiple_of > 0:
                target_w = max(multiple_of, (base_w // multiple_of) * multiple_of)
                target_h = max(multiple_of, (base_h // multiple_of) * multiple_of)
            else:
                target_w, target_h = base_w, base_h

            final_width, final_height = target_w, target_h

            # --- Шаг 2: Промежуточный размер с сохранением пропорций ---
            if fit_mode == "crop":
                scale = max(target_w / orig_w, target_h / orig_h)
            else:
                scale = min(target_w / orig_w, target_h / orig_h)

            fit_w = max(1, int(orig_w * scale))
            fit_h = max(1, int(orig_h * scale))

            # --- Шаг 3: Ресайз и применение fit_mode ---
            resample = INTERPOLATION_METHODS[interpolation]
            resized_pil = pil_img.resize((fit_w, fit_h), resample=resample)

            if fit_mode == "crop":
                dx = max(0, fit_w - target_w)
                dy = max(0, fit_h - target_h)
                left = int(dx * centering_tuple[0])
                top = int(dy * centering_tuple[1])
                output_img = resized_pil.crop((left, top, left + target_w, top + target_h))
            else:  # pad
                # Преобразуем HEX в RGB/RGBA
                pad_color_value = hex_to_rgb_or_rgba(hex_color, has_alpha=has_alpha)
                output_img = ImageOps.pad(
                    resized_pil,
                    (target_w, target_h),
                    method=Image.Resampling.NEAREST,
                    color=pad_color_value,
                    centering=centering_tuple
                )

            # Обратно в numpy → тензор
            output_np = np.array(output_img).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(output_np)
            resized_images.append(output_tensor.unsqueeze(0))

        final_image = torch.cat(resized_images, dim=0)
        return (final_image, final_width, final_height)


# === Регистрация ноды ===
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Res_MP": AGSoft_Img_Res_MP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Res_MP": "AGSoft Image Resize MP"
}
