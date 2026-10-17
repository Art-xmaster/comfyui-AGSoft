"""
==============================================================================
AGSoft_RGB_RGBA_Convert.py
==============================================================================
Ноды: 🖼️AGSoft RGBA to RGB + 🖼️AGSoft RGB to RGBA
Описание / Description:
Пара нод для взаимной конвертации канального состава изображений ComfyUI
IMAGE. Работают как со стандартным батчем [B,H,W,C], так и с одиночным
тензором [H,W,C], который автоматически разворачивается в батч из одного кадра.

🖼️AGSoft RGBA to RGB — приводит изображение к стандартному 3-канальному RGB:
• RGBA (4 кан.) — удаляет альфа-канал, сохраняя R,G,B;
• RGB (3 кан.) — передаёт без изменений;
• Grayscale (1 кан.) — дублирует единственный канал в R=G=B.

🖼️AGSoft RGB to RGBA — приводит изображение к 4-канальному RGBA:
• RGB (3 кан.) — добавляет непрозрачный альфа-канал (alpha = 1.0);
• Grayscale (1 кан.) — сначала дублируется в RGB, затем добавляется альфа;
• RGBA (4 кан.) — возвращается без изменений.
Важно: нода НЕ создаёт прозрачность, альфа всегда равна 1.0. Подключайте её
выход только к нодам, явно поддерживающим 4-канальный IMAGE (многие core-ноды
ComfyUI ожидают строго 3 канала).
---
A pair of nodes for converting the channel layout of ComfyUI IMAGE tensors
in both directions. Both accept a standard batch [B,H,W,C] as well as a
single [H,W,C] tensor, which is automatically expanded to a one-frame batch.

🖼️AGSoft RGBA to RGB — reduces an image to standard 3-channel RGB:
• RGBA (4ch) — removes the alpha channel, keeps R,G,B;
• RGB (3ch) — passes through unchanged;
• Grayscale (1ch) — duplicates the single channel into R=G=B.

🖼️AGSoft RGB to RGBA — converts an image to 4-channel RGBA:
• RGB (3ch) — adds an opaque alpha channel (alpha = 1.0);
• Grayscale (1ch) — first duplicated into RGB, then alpha is added;
• RGBA (4ch) — returned unchanged.
Important: this node does NOT create transparency, alpha is always 1.0.
Connect its output only to nodes that explicitly accept 4-channel IMAGE
(many core ComfyUI nodes expect exactly 3 channels).
Возможности / Features:
⚡ Строгая валидация входа: допускаются только 1 (grayscale), 3 (RGB) и 4 (RGBA) канала, иначе понятная двуязычная ошибка
Strict input validation: only 1 (grayscale), 3 (RGB) and 4 (RGBA) channels allowed, otherwise a clear bilingual error
⚡ uint8 [0..255] автоматически нормализуется в float [0..1]; float передаётся как есть; прочие целочисленные типы отклоняются
uint8 [0..255] automatically normalized to float [0..1]; float passed through; other integer types rejected
⚡ Проверка ранга и размеров тензора до конвертации: скаляры, 2D, 5D+ и пустые тензоры отклоняются
Rank and dimension checks before conversion: scalars, 2D, 5D+ and empty tensors are rejected
⚡ Чистые тензорные операции на GPU/CPU без копий в RAM; выход — непрерывный (contiguous) float-тензор
Pure tensor operations on GPU/CPU, no RAM round-trips; output is a contiguous float tensor
Автор / Author: AGSoft
Дата / Date: 17.10.2026
==============================================================================
"""

import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("[AGSoft RGBA to RGB / RGB to RGBA] v10.17 loaded (channel layout conversion)")


def _validate_image(image):
    # Validate and normalize input to B,H,W,C float tensor with allowed channel counts / Валидация и нормализация входа в B,H,W,C float тензор с допустимым числом каналов
    if not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor, got {type(image)} / "
            f"Ожидается torch.Tensor, получен {type(image)}"
        )

    original_ndim = image.ndim
    original_shape = tuple(image.shape)

    # Check rank first to reject scalars and 5D+ tensors immediately / Сначала проверяем ранг, чтобы сразу отклонять скаляры и 5D+ тензоры
    if original_ndim == 3:
        # Single image H,W,C -> batch 1,H,W,C / Одиночное изображение H,W,C -> батч 1,H,W,C
        image = image.unsqueeze(0)
    elif original_ndim != 4:
        raise ValueError(
            f"Expected ComfyUI IMAGE [B,H,W,C] or single image [H,W,C], got {original_ndim}D tensor with shape {original_shape} / "
            f"Ожидается ComfyUI IMAGE [B,H,W,C] или одиночное изображение [H,W,C], получен {original_ndim}D тензор с формой {original_shape}"
        )

    if any(s <= 0 for s in image.shape):
        raise ValueError(
            f"Invalid tensor dimensions {image.shape} / "
            f"Некорректные размеры тензора {image.shape}"
        )

    # Normalize only uint8; reject other integer types / Нормализуем только uint8; другие целочисленные типы отклоняем
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    elif not image.is_floating_point():
        raise ValueError(
            f"Unsupported dtype {image.dtype}; expected floating point or uint8 / "
            f"Неподдерживаемый тип {image.dtype}; ожидается плавающий тип или uint8"
        )

    # Validate channel count: only 1, 3, or 4 allowed / Валидация числа каналов: допускаются только 1, 3 или 4
    channels = image.shape[-1]
    if channels not in (1, 3, 4):
        raise ValueError(
            f"Expected 1, 3, or 4 channels, got {channels} / "
            f"Ожидается 1, 3 или 4 канала, получено {channels}"
        )

    return image


class AGSoftRGBAtoRGB:
    """
    Конвертирует тензор изображения в трёхканальный формат RGB.
    ---
    Converts an image tensor to 3-channel RGB format.
    """

    CATEGORY = "AGSoft/Image"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgb_image",)

    OUTPUT_TOOLTIPS = (
        "RGB image with shape B,H,W,3. Alpha channel is removed from 4-channel input.\n"
        "---\n"
        "RGB-изображение с формой B,H,W,3. Альфа-канал удаляется из 4-канального входа."
    )

    DESCRIPTION = (
        "Converts ComfyUI IMAGE tensors to 3-channel RGB. 4-channel RGBA loses alpha, 1-channel grayscale is duplicated to RGB. Only 1, 3, or 4 channel inputs are accepted.\n"
        "---\n"
        "Преобразует тензоры ComfyUI IMAGE в 3-канальный RGB. У 4-канального RGBA удаляется альфа, 1-канальные градации серого дублируются в RGB. Допускаются только входы с 1, 3 или 4 каналами."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {
                "tooltip": (
                    "ComfyUI IMAGE tensor [B,H,W,C]. A single image [H,W,C] is also accepted. Standard RGB has 3 channels; RGBA has 4 channels.\n"
                    "---\n"
                    "Тензор ComfyUI IMAGE [B,H,W,C]. Одиночное изображение [H,W,C] также принимается. Стандартный RGB имеет 3 канала; RGBA — 4 канала."
                ),
            }),
        }
        return {"required": required}

    def convert(self, image):
        image = _validate_image(image)
        channels = image.shape[-1]

        if channels == 4:
            # RGBA -> RGB: remove alpha / RGBA -> RGB: удаление альфа-канала
            out = image[..., :3]
        elif channels == 3:
            # Already RGB / Уже RGB
            out = image
        else:
            # Grayscale (1 channel) -> RGB: duplicate channel / Градации серого (1 канал) -> RGB: дублирование канала
            out = image.repeat(1, 1, 1, 3)

        return (out.contiguous().float(),)


class AGSoftRGBtoRGBA:
    """
    Добавляет непрозрачный альфа-канал к изображению, если его нет.
    ---
    Adds an opaque alpha channel to an image if it is missing.
    """

    CATEGORY = "AGSoft/Image"
    FUNCTION = "convert"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba_image",)

    OUTPUT_TOOLTIPS = (
        "RGBA image with shape B,H,W,4. If alpha was missing, it is added with value 1.0. Use this output only with nodes that accept 4-channel IMAGE.\n"
        "---\n"
        "RGBA-изображение с формой B,H,W,4. Если альфа отсутствовала, она добавляется со значением 1.0. Используйте этот выход только с нодами, принимающими 4-канальный IMAGE."
    )

    DESCRIPTION = (
        "Adds an opaque alpha channel (1.0) to 3-channel RGB. 1-channel grayscale is first duplicated to RGB, then alpha is added. 4-channel input is returned unchanged. This node does not create transparency.\n"
        "---\n"
        "Добавляет непрозрачный альфа-канал (1.0) к 3-канальному RGB. 1-канальные градации серого сначала дублируются в RGB, затем добавляется альфа. 4-канальный вход возвращается без изменений. Эта нода не создаёт прозрачность."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE", {
                "tooltip": (
                    "ComfyUI IMAGE tensor [B,H,W,C]. A single image [H,W,C] is also accepted. If alpha is missing, an opaque alpha (1.0) is added.\n"
                    "---\n"
                    "Тензор ComfyUI IMAGE [B,H,W,C]. Одиночное изображение [H,W,C] также принимается. Если альфа отсутствует, добавляется непрозрачная альфа (1.0)."
                ),
            }),
        }
        return {"required": required}

    def convert(self, image):
        image = _validate_image(image)
        channels = image.shape[-1]

        if channels == 4:
            # Already RGBA / Уже RGBA
            out = image
        elif channels == 3:
            # RGB -> RGBA: add opaque alpha / RGB -> RGBA: добавление непрозрачной альфы
            alpha = torch.ones_like(image[..., :1])
            out = torch.cat([image, alpha], dim=-1)
        else:
            # Grayscale (1 channel) -> RGBA: duplicate to RGB, then add alpha / Градации серого (1 канал) -> RGBA: дублирование в RGB, затем добавление альфы
            rgb = image.repeat(1, 1, 1, 3)
            alpha = torch.ones_like(rgb[..., :1])
            out = torch.cat([rgb, alpha], dim=-1)

        return (out.contiguous().float(),)


# Node registration / Регистрация нод
NODE_CLASS_MAPPINGS = {
    "AGSoftRGBAtoRGB": AGSoftRGBAtoRGB,
    "AGSoftRGBtoRGBA": AGSoftRGBtoRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftRGBAtoRGB": "🖼️AGSoft RGBA to RGB",
    "AGSoftRGBtoRGBA": "🖼️AGSoft RGB to RGBA",
}