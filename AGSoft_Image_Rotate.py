"""
AGSoft_Image_Rotate.py

Нода для поворота изображений в ComfyUI.

Поддержка:
- Угол поворота (шаг 1°)
- Предустановки: 90, 180, 270, 360
- Методы интерполяции: nearest, bilinear (единственные, совместимые с torch.Tensor)
- Цвет фона: transparent, black, white, gray, red, green, blue, cyan, yellow
- Expand: увеличивать холст или нет
- Batch processing (B, H, W, C)

Совместимость: ComfyUI 0.3.62+, PyTorch/torchvision (любая стабильная версия)
"""

import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Union, List

# Только поддерживаемые режимы интерполяции для torch.Tensor
INTERPOLATION_MODES = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
}

# Цвета фона в нормализованном RGB [0.0, 1.0]
BACKGROUND_COLORS = {
    "transparent": None,
    "black": [0.0, 0.0, 0.0],
    "white": [1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5],
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "cyan": [0.0, 1.0, 1.0],      # голубой
    "yellow": [1.0, 1.0, 0.0],
}


class AGSoft_Image_Rotate:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "angle_degrees": (
                    "INT",
                    {
                        "default": 0,
                        "min": -360,
                        "max": 360,
                        "step": 1,
                        "tooltip": "Угол поворота в градусах. Положительное — против часовой стрелки.",
                    },
                ),
                "interpolation": (
                    ["nearest", "bilinear"],
                    {
                        "default": "bilinear",
                        "tooltip": "Только эти два метода поддерживаются для тензоров в ComfyUI.",
                    },
                ),
                "background_color": (
                    [
                        "transparent",
                        "black", "white", "gray",
                        "red", "green", "blue",
                        "cyan", "yellow"
                    ],
                    {
                        "default": "transparent",
                        "tooltip": "Цвет фона для новых пикселей. 'Прозрачный' добавляет альфа-канал при необходимости.",
                    },
                ),
                "expand": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Если включено — холст расширяется, чтобы вместить всё изображение без обрезки.",
                    },
                ),
            },
            "optional": {
                "rotation_preset": (
                    ["none", "90", "180", "270", "360"],
                    {
                        "default": "none",
                        "tooltip": "Переопределяет угол поворота.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_image"
    CATEGORY = "AGSoft/Image"
    DESCRIPTION = "Поворот изображения с выбором фона, интерполяции и expand. Поддержка батчей."

    def rotate_image(
        self,
        image: torch.Tensor,
        angle_degrees: int,
        interpolation: str,
        background_color: str,
        expand: bool,
        rotation_preset: str = "none",
    ) -> Tuple[torch.Tensor]:
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
                # Добавляем альфа-канал (1.0 = непрозрачный)
                alpha = torch.ones(
                    (image_chw.shape[0], 1, image_chw.shape[2], image_chw.shape[3]),
                    dtype=image_chw.dtype,
                    device=image_chw.device,
                )
                image_chw = torch.cat([image_chw, alpha], dim=1)
        else:
            base_color = BACKGROUND_COLORS[background_color]
            num_channels = image_chw.shape[1]
            if num_channels == 4:
                # RGBA: добавляем альфа = 1.0
                fill_value = base_color + [1.0]
            else:
                # RGB
                fill_value = base_color

        try:
            rotated_chw = TF.rotate(
                img=image_chw,
                angle=angle_degrees,
                interpolation=interp_mode,
                expand=expand,
                fill=fill_value,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка при повороте изображения: {e}")

        # Обратно в (B, H, W, C)
        rotated = rotated_chw.permute(0, 2, 3, 1)
        return (rotated,)


# === РЕГИСТРАЦИЯ НОДЫ ===
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Rotate": AGSoft_Image_Rotate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Rotate": "AGSoft Image Rotate 🔄",
}