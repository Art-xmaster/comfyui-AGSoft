"""
==============================================================================
AGSoft_Add_Images.py
==============================================================================
Нода: 🖼️AGSoft Add Images
Описание / Description:
Универсальный агрегатор изображений с динамическими опциональными входами
(до 50): подключение к последнему свободному слоту создаёт следующий вход,
отключение последнего подключённого убирает лишний пустой слот (логика в
JS из web/). Предназначен как общий источник изображений для других нод
AGSoft (Stitch Plus и будущих).
Виджет resize_mode выбирает обработку размеров:
- keep (по умолчанию) — размеры НЕ меняются: если все изображения одного
  размера, выход — обычный батч-тензор (совместим с core-нодами
  Preview/Save), иначе — список тензоров исходных размеров;
- stretch — каждое изображение приводится точно к размеру первого
  (пропорции могут исказиться);
- contain — пропорционально вписывается в размер первого, поля чёрным;
- cover — пропорционально масштабируется и центрально обрезается до
  размера первого.
Режимы stretch/contain/cover всегда отдают один батч-тензор; интерполяция —
upscale_method. Батч-тензоры на входах разбиваются на отдельные кадры;
каналы унифицируются до максимума (RGBA дополняется альфой=1). В режиме
keep размеры не изменяются никогда.
---
Universal image aggregator with dynamic optional inputs (up to 50):
connecting to the last free slot creates the next input, disconnecting the
last connected one removes the extra empty slot (logic in JS from web/).
Intended as a common image source for other AGSoft nodes (Stitch Plus and
future ones).
The resize_mode widget selects size handling:
- keep (default) — NO resizing: if all images share the same size the output
  is a regular batch tensor (compatible with core Preview/Save nodes),
  otherwise a list of tensors with original sizes;
- stretch — every image is brought exactly to the first image size (aspect
  may distort);
- contain — proportional fit inside the first image size, pads are black;
- cover — proportional scale and center-crop to the first image size.
stretch/contain/cover always output a single batch tensor; interpolation is
upscale_method. Batch tensors on inputs are split into separate frames;
channels are unified to the max (RGBA gets alpha=1). keep mode never changes
sizes.
Возможности / Features:
⚡ До 50 опциональных входов IMAGE с автосозданием/автоудалением слотов
   (JS: синхронный трим при создании, пересчёт высоты ноды).
   Up to 50 optional IMAGE inputs with auto-create/auto-remove slots
   (JS: synchronous trim on creation, node height recompute).
⚡ Виджет resize_mode: keep / stretch / contain / cover + upscale_method.
   resize_mode widget: keep / stretch / contain / cover + upscale_method.
⚡ Адаптивный выход: батч при одинаковых размерах (совместим с core-нодами),
   список при разных (размеры сохранены).
   Adaptive output: batch for uniform sizes (core-node compatible), list for
   different sizes (sizes preserved).
⚡ Батч-входы разбиваются на кадры, каналы унифицируются до максимума.
   Batch inputs split into frames, channels unified to max.
⚡ Компактная высота ноды: 50 объявленных слотов обрезаются до
   (подключено + 1) синхронно в nodeCreated.
   Compact node height: 50 declared slots trimmed to (connected + 1)
   synchronously in nodeCreated.
Автор / Author: AGSoft
Дата / Date: 25.09.2026
==============================================================================
"""
import logging
import torch
from comfy.utils import common_upscale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("[AGSoft Add Images] v09.25 loaded (universal aggregator, resize_mode widget, dynamic inputs, max 50)")

MAX_INPUTS = 50

def split_frames(tensor):
    # Split batch tensor into list of single-frame tensors / Разбить батч-тензор на список одиночных кадров
    if isinstance(tensor, (list, tuple)):
        out = []
        for t in tensor:
            out.extend(split_frames(t))
        return out
    if tensor.dim() == 4:
        return [tensor[i:i + 1] for i in range(tensor.shape[0])]
    if tensor.dim() == 3:
        return [tensor.unsqueeze(0)]
    return [tensor]

class AGSoft_Add_Images:
    CATEGORY = "AGSoft/Image"
    FUNCTION = "combine"
    OUTPUT_NODE = False
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = (
        "Images in connection order: batch tensor for uniform sizes or resize modes, list of tensors otherwise; keep mode preserves original sizes.\n"
        "---\nИзображения в порядке подключения: батч-тензор при одинаковых размерах или режимах ресайза, иначе список тензоров; режим keep сохраняет исходные размеры.",
    )
    DESCRIPTION = (
        "Aggregate up to 50 optional image inputs. resize_mode=keep (default) does not resize: batch if all sizes equal, else list. stretch/contain/cover resize every image to the first image size and output one batch.\n"
        "---\nОбъединение до 50 опциональных входов изображений. resize_mode=keep (по умолчанию) не меняет размеры: батч при одинаковых, иначе список. stretch/contain/cover приводят все изображения к размеру первого и дают один батч."
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "resize_mode": (["keep", "stretch", "contain", "cover"], {
                "default": "keep",
                "tooltip": (
                    "Size handling:\n"
                    "• keep — do not resize; output is a batch if all sizes equal, otherwise a list of tensors (default)\n"
                    "• stretch — resize every image exactly to the first image size (possible distortion)\n"
                    "• contain — proportional fit inside the first image size, pads are black\n"
                    "• cover — proportional scale and center-crop to the first image size\n"
                    "stretch/contain/cover always output a single batch tensor.\n"
                    "---\n"
                    "Обработка размеров:\n"
                    "• keep — не изменять размеры; выход — батч при одинаковых размерах, иначе список тензоров (по умолчанию)\n"
                    "• stretch — привести каждое изображение точно к размеру первого (возможно искажение)\n"
                    "• contain — пропорционально вписать в размер первого, поля чёрным\n"
                    "• cover — пропорционально масштабировать и центрально обрезать до размера первого\n"
                    "stretch/contain/cover всегда дают один батч-тензор."
                )
            }),
            "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                "default": "lanczos",
                "tooltip": "Interpolation used by stretch/contain/cover. Ignored in keep.\n---\nИнтерполяция для stretch/contain/cover. Игнорируется в keep."
            }),
        }
        for i in range(1, MAX_INPUTS + 1):
            optional["image_" + str(i)] = ("IMAGE", {
                "tooltip": (
                    "Optional image input " + str(i) + ". Connecting the last slot creates the next one.\n"
                    "---\nОпциональный вход изображения " + str(i) + ". Подключение к последнему слоту создаёт следующий."
                )
            })
        return {"required": {}, "optional": optional}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @staticmethod
    def _unify_channels(images):
        # Pad all tensors to max channel count / Дополнить все тензоры до максимального числа каналов
        max_c = max(t.shape[-1] for t in images)
        out = []
        for t in images:
            if t.shape[-1] < max_c:
                t = torch.cat([t, torch.ones(*t.shape[:-1], max_c - t.shape[-1], device=t.device)], dim=-1)
            out.append(t)
        return out

    @staticmethod
    def _pad_black(image, padding):
        # Pad with black (alpha=1 if present) / Поля чёрным цветом (альфа=1 при наличии)
        batch, height, width, channels = image.shape
        top, bottom, left, right = padding
        result = torch.zeros((batch, height + top + bottom, width + left + right, channels), device=image.device)
        if channels == 4:
            result[..., 3] = 1.0
        result[:, top:top + height, left:left + width, :] = image
        return result

    def _fit_to(self, image, target_w, target_h, mode, method):
        # Bring one image to exact target size per mode / Привести одно изображение точно к целевому размеру по режиму
        h, w = image.shape[1:3]
        if mode == "stretch":
            return common_upscale(image.movedim(-1, 1), target_w, target_h, method, "disabled").movedim(1, -1)
        scale = min(target_w / w, target_h / h) if mode == "contain" else max(target_w / w, target_h / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        image = common_upscale(image.movedim(-1, 1), nw, nh, method, "disabled").movedim(1, -1)
        if mode == "contain":
            pad_w, pad_h = target_w - nw, target_h - nh
            return self._pad_black(image, (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2))
        off_y, off_x = (nh - target_h) // 2, (nw - target_w) // 2
        return image[:, off_y:off_y + target_h, off_x:off_x + target_w, :]

    def combine(self, resize_mode="keep", upscale_method="lanczos", **kwargs):
        out = []
        for i in range(1, MAX_INPUTS + 1):
            v = kwargs.get("image_" + str(i))
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        out.extend(split_frames(t))
            elif isinstance(v, torch.Tensor):
                out.extend(split_frames(v))
        if not out:
            return ([],)
        out = self._unify_channels(out)
        if resize_mode == "keep":
            ref = out[0].shape[1:]
            if all(t.shape[1:] == ref for t in out):
                # Same sizes: regular batch for core-node compatibility / Одинаковые размеры: обычный батч для совместимости с core-нодами
                return (torch.cat(out, dim=0),)
            # Different sizes: list, sizes preserved / Разные размеры: список, размеры сохранены
            return (out,)
        ref_h, ref_w = out[0].shape[1:3]
        processed = [self._fit_to(t, ref_w, ref_h, resize_mode, upscale_method) for t in out]
        return (torch.cat(processed, dim=0),)

NODE_CLASS_MAPPINGS = {
    "AGSoft_Add_Images": AGSoft_Add_Images
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Add_Images": "🖼️AGSoft Add Images"
}