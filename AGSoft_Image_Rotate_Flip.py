"""
AGSoft_Image_Rotate_Flip.py
Нода для поворота и отражения изображений в ComfyUI.
Поддержка:
- Угол поворота (-360..360, шаг 1°)
- Предустановки: 90, 180, 270, 360
- Отражение: без, по горизонтали, по вертикали (в самом низу ноды)
- Интерполяция: nearest, bilinear
- Фон: transparent, black, white, gray, red, green, blue, cyan, yellow
- Expand: авто-расширение холста
- Выходы: IMAGE, MASK (альфа-канал или полная непрозрачность), ширина (INT), высота (INT)
- Batch processing (B, H, W, C)

Автор: AGSoft
Дата: 06.06.2026

"""
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Union, List

INTERPOLATION_MODES = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
}

BACKGROUND_COLORS = {
    "transparent": None,
    "black": [0.0, 0.0, 0.0],
    "white": [1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
}

class AGSoft_Image_Rotate_Flip:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "angle_degrees": (
                    "INT", {
                        "default": 0, "min": -360, "max": 360, "step": 1,
                        "tooltip": "Угол поворота в градусах. Положительное значение — против часовой стрелки.",
                    },
                ),
                "interpolation": (
                    ["nearest", "bilinear"], {
                        "default": "bilinear",
                        "tooltip": "Алгоритм сглаживания. Только эти два метода стабильно работают с тензорами.",
                    },
                ),
                "background_color": (
                    ["transparent", "black", "white", "gray", "red", "green", "blue", "cyan", "yellow"], {
                        "default": "transparent",
                        "tooltip": "Цвет заполнения пустых зон. 'transparent' добавляет альфа-канал при необходимости.",
                    },
                ),
                "expand": (
                    "BOOLEAN", {
                        "default": False,
                        "tooltip": "Расширяет холст, чтобы избежать обрезки краёв изображения при повороте.",
                    },
                ),
            },
            "optional": {
                "rotation_preset": (
                    ["none", "90", "180", "270", "360"], {
                        "default": "none",
                        "tooltip": "Быстрый выбор угла. Переопределяет значение angle_degrees.",
                    },
                ),
                "flip_mode": (
                    ["none", "horizontal", "vertical"], {
                        "default": "none",
                        "tooltip": "Зеркальное отражение. Применяется после поворота.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "transform_image"
    CATEGORY = "AGSoft/Image"
    DESCRIPTION = "Поворот и отражение изображения с выбором фона, интерполяции и расширением холста. Возвращает маску и новые размеры."

    def transform_image(
        self,
        image: torch.Tensor,
        angle_degrees: int,
        interpolation: str,
        background_color: str,
        expand: bool,
        rotation_preset: str = "none",
        flip_mode: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if rotation_preset != "none":
            angle_degrees = int(rotation_preset)

        interp_mode = INTERPOLATION_MODES.get(interpolation)
        if interp_mode is None:
            raise ValueError(f"Неподдерживаемый метод интерполяции: {interpolation}")

        # (B, H, W, C) → (B, C, H, W)
        image_chw = image.permute(0, 3, 1, 2)

        fill_value: Union[None, List[float]] = None
        if background_color == "transparent":
            if image_chw.shape[1] == 3:
                alpha = torch.ones(
                    (image_chw.shape[0], 1, image_chw.shape[2], image_chw.shape[3]),
                    dtype=image_chw.dtype, device=image_chw.device,
                )
                image_chw = torch.cat([image_chw, alpha], dim=1)
        else:
            base_color = BACKGROUND_COLORS[background_color]
            num_channels = image_chw.shape[1]
            fill_value = base_color + [1.0] if num_channels == 4 else base_color

        try:
            transformed_chw = TF.rotate(
                img=image_chw,
                angle=angle_degrees,
                interpolation=interp_mode,
                expand=expand,
                fill=fill_value,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при трансформации: {e}")

        if flip_mode == "horizontal":
            transformed_chw = TF.hflip(transformed_chw)
        elif flip_mode == "vertical":
            transformed_chw = TF.vflip(transformed_chw)

        B, C, H, W = transformed_chw.shape

        # Извлекаем маску: если 4 канала, берем альфа-канал, иначе создаем маску полной непрозрачности
        if C == 4:
            mask = transformed_chw[:, 3, :, :]
        else:
            mask = torch.ones((B, H, W), dtype=transformed_chw.dtype, device=transformed_chw.device)

        # Обратно в (B, H, W, C) для IMAGE
        final_image = transformed_chw.permute(0, 2, 3, 1)

        return (final_image, mask, W, H)

NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Rotate_Flip": AGSoft_Image_Rotate_Flip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Rotate_Flip": "🔄 AGSoft Image Rotate & Flip",
}