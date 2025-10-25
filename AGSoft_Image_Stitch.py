# AGSoft_Image_Stitch.py
# Автор: AGSoft
# Дата: 25 октября 2025 г.

import torch
from comfy.utils import common_upscale

# Предустановленные цвета фона
BACKGROUND_COLOR_PRESETS = {
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
}

class AGSoft_Image_Stitch:
    CATEGORY = "AGSoft/Image"
    FUNCTION = "stitch"
    OUTPUT_NODE = False
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")

    @classmethod
    def INPUT_TYPES(cls):
        # Формат: "название (#HEX)"
        color_options = [f"{name} ({hex_code})" for name, hex_code in BACKGROUND_COLOR_PRESETS.items()]
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "Обязательное первое изображение. От него зависит ориентация сшивания."}),
                "stitch_mode": (["right", "down", "left", "up", "2x2", "context_mode"], {
                    "default": "right",
                    "tooltip": (
                        "Режим сшивания:\n"
                        "• right — справа от первого\n"
                        "• down — под первым\n"
                        "• left/up — слева/сверху\n"
                        "• 2x2 — сетка 2×2 (до 4 изображений)\n"
                        "• context_mode — специальный режим:\n"
                        "  - При 3 изображениях: image1+image2 вертикально, image3 справа.\n"
                        "  - При 4 изображениях: image1+image2+image3 вертикально, image4 справа."
                    )
                }),
                "match_image_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Если включено — второе и последующие изображения масштабируются ПРОПОРЦИОНАЛЬНО, чтобы соответствовать первому (без искажения)."
                }),
                "megapixels": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01,
                    "tooltip": "Целевой размер финального изображения в мегапикселях (0 = без ограничений)."
                }),
                "max_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Максимальная ширина результата (0 = без ограничений). Игнорируется, если megapixels > 0."
                }),
                "max_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Максимальная высота результата (0 = без ограничений). Игнорируется, если megapixels > 0."
                }),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                    "default": "lanczos",
                    "tooltip": "Метод интерполяции при масштабировании. Lanczos даёт наилучшее качество."
                }),
                "spacing_width": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Расстояние (в пикселях) между изображениями. Заполняется цветом фона."
                }),
                "background_color_preset": (color_options, {
                    "default": "white (#FFFFFF)",
                    "tooltip": "Выберите предустановленный цвет фона. Используется, если custom_background_color пуст."
                }),
                # --- НОВЫЙ ПАРАМЕТР В СТИЛЕ AGSOFT ---
                "multiple_of": ([0, 1, 2, 4, 8, 16, 32, 64, 112, 128], {
                    "default": 0,
                    "tooltip": (
                        "Привести итоговые размеры изображения к кратности указанному числу.\n"
                        "• 0 = отключено\n"
                        "• 8 = стандарт для большинства моделей Stable Diffusion\n"
                        "• 64 = требуется для некоторых VAE или видео-моделей\n"
                        "• 112, 128 = для специфических архитектур\n\n"
                        "Если включено, изображение будет центрировано на холсте,\n"
                        "размеры которого кратны выбранному значению."
                    )
                }),
                # --- КОНЕЦ ---
            },
            "optional": {
                "image2": ("IMAGE", {"tooltip": "Второе изображение (опционально)."}),
                "image3": ("IMAGE", {"tooltip": "Третье изображение (опционально)."}),
                "image4": ("IMAGE", {"tooltip": "Четвёртое изображение (опционально)."}),
                "custom_background_color": ("STRING", {
                    "default": "",
                    "placeholder": "#RRGGBB",
                    "tooltip": "Произвольный цвет фона в формате #RRGGBB. Если указан — переопределяет background_color_preset."
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        bg_color = kwargs.get("custom_background_color", "").strip()
        if not bg_color:
            preset = kwargs.get("background_color_preset", "white (#FFFFFF)")
            # Извлекаем HEX из строки вида "white (#FFFFFF)" → "#FFFFFF"
            bg_color = preset.split(" (")[-1].rstrip(")")
        if not cls._is_valid_hex_color(bg_color):
            return f"Неверный формат цвета фона: {bg_color}. Используйте #RRGGBB."
        return True

    @staticmethod
    def _is_valid_hex_color(color_str):
        if not isinstance(color_str, str) or len(color_str) != 7 or color_str[0] != '#':
            return False
        try:
            int(color_str[1:], 16)
            return True
        except ValueError:
            return False

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def pad_with_color(self, image, padding, color_val):
        batch, height, width, channels = image.shape
        r, g, b = color_val
        pad_top, pad_bottom, pad_left, pad_right = padding
        new_height = height + pad_top + pad_bottom
        new_width = width + pad_left + pad_right
        result = torch.zeros((batch, new_height, new_width, channels), device=image.device)
        if channels >= 3:
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            if channels == 4:
                result[..., 3] = 1.0
        result[:, pad_top:pad_top+height, pad_left:pad_left+width, :] = image
        return result

    def match_dimensions(self, image1, image2, stitch_mode, color_val):
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        if stitch_mode in ["left", "right"]:
            if h1 != h2:
                target_h = max(h1, h2)
                if h1 < target_h:
                    pad_h = target_h - h1
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image1 = self.pad_with_color(image1, (pad_top, pad_bottom, 0, 0), color_val)
                if h2 < target_h:
                    pad_h = target_h - h2
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image2 = self.pad_with_color(image2, (pad_top, pad_bottom, 0, 0), color_val)
        else:
            if w1 != w2:
                target_w = max(w1, w2)
                if w1 < target_w:
                    pad_w = target_w - w1
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image1 = self.pad_with_color(image1, (0, 0, pad_left, pad_right), color_val)
                if w2 < target_w:
                    pad_w = target_w - w2
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image2 = self.pad_with_color(image2, (0, 0, pad_left, pad_right), color_val)
        return image1, image2

    def ensure_same_channels(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat([
                    image1,
                    torch.ones(*image1.shape[:-1], max_channels - image1.shape[-1], device=image1.device),
                ], dim=-1)
            if image2.shape[-1] < max_channels:
                image2 = torch.cat([
                    image2,
                    torch.ones(*image2.shape[:-1], max_channels - image2.shape[-1], device=image2.device),
                ], dim=-1)
        return image1, image2

    def create_spacing(self, image1, image2, spacing_width, stitch_mode, color_val):
        if spacing_width <= 0:
            return None
        if stitch_mode in ["left", "right"]:
            spacing_shape = (
                image1.shape[0],
                max(image1.shape[1], image2.shape[1]),
                spacing_width,
                image1.shape[-1],
            )
        else:
            spacing_shape = (
                image1.shape[0],
                spacing_width,
                max(image1.shape[2], image2.shape[2]),
                image1.shape[-1],
            )
        spacing = torch.zeros(spacing_shape, device=image1.device)
        r, g, b = color_val
        if spacing.shape[-1] >= 3:
            spacing[..., 0] = r
            spacing[..., 1] = g
            spacing[..., 2] = b
            if spacing.shape[-1] == 4:
                spacing[..., 3] = 1.0
        return spacing

    def stitch_two_images(self, image1, image2, stitch_mode, match_image_size, spacing_width, color_val, upscale_method):
        if image2 is None:
            return image1
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat([image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)])
            if image2.shape[0] < max_batch:
                image2 = torch.cat([image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)])
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2
            if stitch_mode in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:
                target_w, target_h = w1, int(w1 / aspect_ratio)
            image2 = common_upscale(
                image2.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        else:
            image1, image2 = self.match_dimensions(image1, image2, stitch_mode, color_val)
        image1, image2 = self.ensure_same_channels(image1, image2)
        spacing = self.create_spacing(image1, image2, spacing_width, stitch_mode, color_val)
        images = [image2, image1] if stitch_mode in ["left", "up"] else [image1, image2]
        if spacing is not None:
            images.insert(1, spacing)
        concat_dim = 2 if stitch_mode in ["left", "right"] else 1
        result = torch.cat(images, dim=concat_dim)
        return result

    def create_blank_like(self, reference_image, color_val):
        batch, height, width, channels = reference_image.shape
        result = torch.zeros((batch, height, width, channels), device=reference_image.device)
        r, g, b = color_val
        if channels >= 3:
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            if channels == 4:
                result[..., 3] = 1.0
        return result

    def stitch_multi_mode(self, image1, image2, image3, image4, stitch_mode, match_image_size, spacing_width, color_val, upscale_method):
        images = [img for img in [image1, image2, image3, image4] if img is not None]
        if len(images) == 0:
            return torch.zeros((1, 64, 64, 3))
        if len(images) == 1:
            return images[0]
        current = images[0]
        for next_img in images[1:]:
            current = self.stitch_two_images(current, next_img, stitch_mode, match_image_size, spacing_width, color_val, upscale_method)
        return current

    def stitch_grid_2x2(self, image1, image2, image3, image4, match_image_size, spacing_width, color_val, upscale_method):
        ref = image1
        img2 = image2 if image2 is not None else self.create_blank_like(ref, color_val)
        row1 = self.stitch_two_images(ref, img2, "right", match_image_size, spacing_width, color_val, upscale_method)
        img3 = image3 if image3 is not None else self.create_blank_like(ref, color_val)
        img4 = image4 if image4 is not None else self.create_blank_like(ref, color_val)
        row2 = self.stitch_two_images(img3, img4, "right", match_image_size, spacing_width, color_val, upscale_method)
        result = self.stitch_two_images(row1, row2, "down", match_image_size, spacing_width, color_val, upscale_method)
        return result

    def stitch_context_mode(self, image1, image2, image3, match_image_size, spacing_width, color_val, upscale_method, image4=None):
        """Режим context_mode:
        - Если image4 есть → слева: image1+image2+image3, справа: image4.
        - Если image4 нет, но есть image3 → слева: image1+image2, справа: image3.
        """
        if image1 is None:
            return torch.zeros((1, 64, 64, 3))

        has_image4 = image4 is not None

        if has_image4:
            left_images = [img for img in [image1, image2, image3] if img is not None]
            right_image = image4
        else:
            left_images = [img for img in [image1, image2] if img is not None]
            right_image = image3

        if right_image is None:
            if len(left_images) == 0:
                return torch.zeros((1, 64, 64, 3))
            elif len(left_images) == 1:
                return left_images[0]
            else:
                result = left_images[0]
                for img in left_images[1:]:
                    result = self.stitch_two_images(result, img, "down", match_image_size, spacing_width, color_val, upscale_method)
                return result

        # Приводим к одному батчу
        all_imgs = left_images + [right_image]
        max_batch = max(img.shape[0] for img in all_imgs if img is not None)
        for i, img in enumerate(all_imgs):
            if img is not None and img.shape[0] < max_batch:
                all_imgs[i] = torch.cat([img, img[-1:].repeat(max_batch - img.shape[0], 1, 1, 1)])
        left_images = all_imgs[:-1]
        right_image = all_imgs[-1]

        # Обрабатываем левую колонку
        if match_image_size and len(left_images) > 1:
            w1 = left_images[0].shape[2]
            for i in range(1, len(left_images)):
                h, w = left_images[i].shape[1:3]
                aspect_ratio = h / w
                target_w = w1
                target_h = int(w1 * aspect_ratio)
                left_images[i] = common_upscale(
                    left_images[i].movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
                ).movedim(1, -1)
        elif not match_image_size and len(left_images) > 1:
            for i in range(1, len(left_images)):
                left_images[0], left_images[i] = self.match_dimensions(left_images[0], left_images[i], "down", color_val)

        for i in range(1, len(left_images)):
            left_images[0], left_images[i] = self.ensure_same_channels(left_images[0], left_images[i])

        # Сшиваем левую колонку вертикально
        left_column = left_images[0]
        for i in range(1, len(left_images)):
            spacing = self.create_spacing(left_column, left_images[i], spacing_width, "down", color_val)
            parts = [left_column]
            if spacing is not None:
                parts.append(spacing)
            parts.append(left_images[i])
            left_column = torch.cat(parts, dim=1)

        # Обрабатываем правое изображение
        if match_image_size:
            h_left = left_column.shape[1]
            hr, wr = right_image.shape[1:3]
            aspect_ratio = wr / hr
            target_h = h_left
            target_w = int(h_left * aspect_ratio)
            right_image = common_upscale(
                right_image.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        else:
            left_column, right_image = self.match_dimensions(left_column, right_image, "right", color_val)

        left_column, right_image = self.ensure_same_channels(left_column, right_image)

        # Сшиваем горизонтально
        spacing = self.create_spacing(left_column, right_image, spacing_width, "right", color_val)
        parts = [left_column]
        if spacing is not None:
            parts.append(spacing)
        parts.append(right_image)
        result = torch.cat(parts, dim=2)

        return result

    def stitch(self, image1, stitch_mode, match_image_size, megapixels, max_width, max_height, upscale_method, spacing_width, background_color_preset, multiple_of, image2=None, image3=None, image4=None, custom_background_color=""):
        if image1 is None:
            raise RuntimeError("image1 is required")
        
        # Определяем итоговый цвет фона
        if custom_background_color and custom_background_color.strip():
            bg_hex = custom_background_color.strip()
        else:
            # Извлекаем HEX из строки вида "white (#FFFFFF)"
            bg_hex = background_color_preset.split(" (")[-1].rstrip(")")
        color_val = self.hex_to_rgb(bg_hex)

        if stitch_mode == "context_mode":
            result = self.stitch_context_mode(image1, image2, image3, match_image_size, spacing_width, color_val, upscale_method, image4=image4)
        elif stitch_mode == "2x2":
            result = self.stitch_grid_2x2(image1, image2, image3, image4, match_image_size, spacing_width, color_val, upscale_method)
        else:
            result = self.stitch_multi_mode(image1, image2, image3, image4, stitch_mode, match_image_size, spacing_width, color_val, upscale_method)
        h, w = result.shape[1:3]
        need_resize = False
        target_w, target_h = w, h
        if megapixels > 0:
            aspect_ratio = w / h
            target_pixels = int(megapixels * 1024 * 1024)
            target_h = int((target_pixels / aspect_ratio) ** 0.5)
            target_w = int(aspect_ratio * target_h)
            need_resize = True
        elif max_width > 0 or max_height > 0:
            if max_width > 0 and w > max_width:
                scale_factor = max_width / w
                target_w = max_width
                target_h = int(h * scale_factor)
                need_resize = True
            else:
                target_w, target_h = w, h
            if max_height > 0 and (target_h > max_height or (target_h == h and h > max_height)):
                scale_factor = max_height / target_h
                target_h = max_height
                target_w = int(target_w * scale_factor)
                need_resize = True
        if need_resize:
            result = common_upscale(
                result.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        
        # --- Приведение к кратности с ЦЕНТРИРОВАНИЕМ (в стиле AGSoft) ---
        if multiple_of > 0:
            current_h, current_w = result.shape[1:3]
            # Округляем ВВЕРХ до ближайшего кратного
            new_w = ((current_w + multiple_of - 1) // multiple_of) * multiple_of
            new_h = ((current_h + multiple_of - 1) // multiple_of) * multiple_of

            if new_w != current_w or new_h != current_h:
                # Вычисляем паддинг для центрирования
                pad_w_total = new_w - current_w
                pad_h_total = new_h - current_h
                pad_left = pad_w_total // 2
                pad_right = pad_w_total - pad_left
                pad_top = pad_h_total // 2
                pad_bottom = pad_h_total - pad_top

                result = self.pad_with_color(result, (pad_top, pad_bottom, pad_left, pad_right), color_val)
        # --- Конец ---

        final_height, final_width = result.shape[1:3]
        return (result, final_width, final_height)

# --- РЕГИСТРАЦИЯ ---
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Stitch": AGSoft_Image_Stitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Stitch": "AGSoft Image Stitch"
}