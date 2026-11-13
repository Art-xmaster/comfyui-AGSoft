"""
==============================================================================
AGSoft_Load_Image_Mask_Resize.py
==============================================================================
Нода: 🖼️AGSoft Load Image & Mask Resize
Описание / Description:
    Загружает изображение (комбо + upload + опциональный input_image +
    custom_path первой строкой) и ресайзит его синхронно с маской
    (из альфа-канала или входной mask).
    Порядок приоритета источников: input_image > custom_path > комбо image.
    Превью следует за реально использованным источником: тензор → первый
    кадр; custom_path → файл (живо при вводе, после выполнения при линке);
    комбо → выбранный файл. Реализовано связкой серверного состояния и
    JS-расширения web/AGSoft_Load_Image_Mask_Resize.js.
    Порядок трансформаций: rotate → resize → flip.
    Условия ресайза — ЧИСТЫЕ ТРИГГЕРЫ: resize_if_larger/smaller отвечают
    только за «ресайзить ли» (совпало → ресайз по fit_mode; нет → исходный
    размер). За «как» отвечает fit_mode: stretch / crop / pad / proportional
    (proportional = пропорционально вписать в цель, выход = получившийся
    размер, без полей и без обрезки).
    Сервер отдаёт в состоянии превью реальные размеры источника и выхода —
    JS рисует по ним статус-бар (МП, ширина, высота) и пересчитывает цель
    в реальном времени при правке виджетов.
    ---
    Loads an image (combo + upload + optional input_image + custom_path as
    the first row) and resizes it synchronously with a mask (from the alpha
    channel or the input mask).
    Source priority: input_image > custom_path > image combo.
    The preview follows the actually used source: tensor → first frame;
    custom_path → file (live when typed, after execution when linked);
    combo → the selected file. Implemented via server state + the JS
    extension web/AGSoft_Load_Image_Mask_Resize.js.
    Transform order: rotate → resize → flip.
    Resize conditions are PURE TRIGGERS: resize_if_larger/smaller only
    decide "whether to resize" (triggered → resize per fit_mode; otherwise
    → original size). "How" is fully owned by fit_mode: stretch / crop /
    pad / proportional (proportional = scale to fit inside the target,
    output = the resulting size, no bars, no crop).
    The server publishes real source/output sizes in the preview state —
    the JS status bar uses them (MP, width, height) and recomputes the
    target live while the user edits widgets.
Возможности / Features:
    ⚡ Источники: input_image (приоритет) > custom_path > комбо image (+upload).
       Sources: input_image (priority) > custom_path > image combo (+upload).
    ⚡ Превью следует за источником (тензор → первый кадр, custom_path → файл).
       Preview follows the source (tensor → first frame, custom_path → file).
    ⚡ Авто-скрытие виджетов по режиму размеров и стратегии (JS).
       Mode-aware widget auto-hiding by size mode and strategy (JS).
    ⚡ Статус-бар: реальные МП/Ш/В и пересчитанные МП/Ш/В в реальном времени.
       Status bar: real MP/W/H and recomputed MP/W/H in real time.
    ⚡ Полное сворачивание виджетов: входы/выходы + превью + статус-бар.
       Full widget collapse: inputs/outputs + preview + status bar.
    ⚡ Выходы: image, mask, width, height, filename, filepath.
       Outputs: image, mask, width, height, filename, filepath.
    ⚡ 7 режимов: мегапиксели / проценты / ширина / высота / оба / longest / shortest.
       7 modes: megapixels / percentage / width / height / both / longest / shortest.
    ⚡ Условия-триггеры: resize_if_larger/smaller (совпало → ресайз, нет → исходный).
       Trigger conditions: resize_if_larger/smaller (triggered → resize, else original).
    ⚡ 4 стратегии: stretch / crop / pad / proportional.
       4 strategies: stretch / crop / pad / proportional.
    ⚡ Поворот angle_degrees + пресеты; отражение flip_mode (после ресайза).
       Rotation angle_degrees + presets; flip_mode (after resize).
    ⚡ 9 позиций выравнивания для crop/pad.
       9 alignment positions for crop/pad.
    ⚡ Premultiplied alpha: lanczos+прозрачность без ореолов; маски не трогает.
       Premultiplied alpha: lanczos+transparency without halos; masks untouched.
    ⚡ Синхронная обработка изображения и маски (не рассинхронизируются).
       Synchronous image+mask processing (never desync).
    ⚡ Кратность (divisible_by), инверсия маски, CPU/CUDA.
       Divisibility (divisible_by), mask inversion, CPU/CUDA.
Требуется / Required:
    web/AGSoft_Load_Image_Mask_Resize.js — без него не работают авто-скрытие
    виджетов, статус-бар, сворачивание и превью.
    web/AGSoft_Load_Image_Mask_Resize.js — without it auto-hiding, the status
    bar, collapsing and the preview do not work.
Автор / Author: AGSoft
Дата / Date: 14.11.2026
==============================================================================
"""

import os
import time
import hashlib
import shutil
import asyncio
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Load Image & Mask Resize] v11.14 loaded (mode-aware widgets + live status bar + collapse)")

# =============================================================================
# Префикс временных превью-копий в папке input (общий с Load Image & Mask).
# Prefix for temporary preview copies inside the input folder.
# =============================================================================
AGSOFT_PREVIEW_PREFIX = "agsoft_preview"

_PIL_EXT_MAP = {
    "png": ".png",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "gif": ".gif",
    "webp": ".webp",
    "bmp": ".bmp",
    "tiff": ".tiff",
    "tif": ".tiff",
    "ico": ".ico",
    "mpo": ".jpg",
}

# =============================================================================
# КОНСТАНТЫ / CONSTANTS
# =============================================================================
INTERPOLATION_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "area": Image.BOX,
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS,
}

# 4 стратегии: stretch / crop / pad / proportional.
# 4 strategies: stretch / crop / pad / proportional.
FIT_MODES = ["stretch", "crop", "pad", "proportional"]
CONDITION_MODES = ["none", "width", "height", "both"]
FLIP_MODES = ["none", "horizontal", "vertical"]
ROTATION_PRESETS = ["none", "90", "180", "270", "360"]
RESIZE_MODES = [
    "target_megapixels",
    "target_percentage",
    "target_width",
    "target_height",
    "target_both",
    "target_longest",
    "target_shortest",
]

IMAGE_COLORS_HEX = {
    "black": "#000000", "white": "#FFFFFF", "gray": "#808080", "silver": "#C0C0C0",
    "light_gray": "#D3D3D3", "dark_gray": "#A9A9A9", "red": "#FF0000", "green": "#00FF00",
    "blue": "#0000FF", "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "orange": "#FFA500", "pink": "#FFC0CB", "brown": "#A52A2A", "purple": "#800080",
    "violet": "#EE82EE", "indigo": "#4B0082", "teal": "#008080", "navy": "#000080",
    "olive": "#808000", "maroon": "#800000", "dark_blue": "#00008B", "light_blue": "#ADD8E6",
    "light_green": "#90EE90", "dark_green": "#006400", "transparent": "transparent",
}

MASK_COLORS_HEX = {
    "black": "#000000", "white": "#FFFFFF", "gray": "#808080", "silver": "#C0C0C0",
    "light_gray": "#D3D3D3", "dark_gray": "#A9A9A9", "transparent": "transparent",
}

CROP_POSITIONS = {
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


# =============================================================================
# ХЕЛПЕРЫ / HELPERS
# =============================================================================
def hex_to_rgb_or_rgba(hex_color: str, has_alpha: bool = False) -> Tuple[int, ...]:
    """HEX → RGB/RGBA. / HEX → RGB/RGBA."""
    if hex_color == "transparent":
        return (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb + (255,) if has_alpha else rgb
    if len(hex_color) == 8:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4, 6))
    return (0, 0, 0)


def tensor_to_pil(tensor: torch.Tensor, is_mask: bool = False) -> Image.Image:
    """Тензор → PIL. / Tensor → PIL."""
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if is_mask:
        if array.ndim == 3:
            if array.shape[-1] == 1:
                array = array.squeeze(-1)
            elif array.shape[-1] == 3:
                array = np.dot(array[..., :3], [0.299, 0.587, 0.114]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[-1] == 1:
        array = np.concatenate([array] * 3, axis=-1)
    elif array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array, mode="RGB")


def pil_to_tensor(pil_image: Image.Image, is_mask: bool = False) -> torch.Tensor:
    """PIL → тензор. / PIL → tensor."""
    if is_mask:
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)
    if pil_image.mode == "RGBA":
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)
    pil_image = pil_image.convert("RGB")
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)


def _premultiply_rgba(pil_img: Image.Image) -> Image.Image:
    """Premultiply: RGB *= A (убирает ореолы). / Removes halos."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4]
    out = np.concatenate([rgb, arr[..., 3:4]], axis=-1)
    return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), mode="RGBA")


def _unpremultiply_rgba(pil_img: Image.Image) -> Image.Image:
    """Un-premultiply: RGB /= A (защита от A=0). / Guarded against A=0."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    alpha = arr[..., 3:4]
    safe = np.maximum(alpha, 1e-5)
    rgb = np.where(alpha > 1e-5, arr[..., :3] / safe, 0)
    out = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), mode="RGBA")


def _rotation_resample(resample: int) -> int:
    """PIL rotate принимает ТОЛЬКО NEAREST/BILINEAR/BICUBIC."""
    if resample == Image.NEAREST:
        return Image.NEAREST
    if resample == Image.BILINEAR:
        return Image.BILINEAR
    return Image.BICUBIC


def _rotate_image(
    pil_img: Image.Image,
    angle: int,
    resample: int,
    pad_color: Tuple[int, ...],
    has_alpha: bool,
) -> Image.Image:
    """Поворот изображения (auto-expand). / Rotate image (auto-expand)."""
    if has_alpha and pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    fill = pad_color if pil_img.mode == "RGBA" else pad_color[:3]
    return pil_img.rotate(angle, resample=_rotation_resample(resample), expand=True, fillcolor=fill)


def _rotate_mask(pil_mask: Image.Image, angle: int, fill_value: int) -> Image.Image:
    """Поворот маски (nearest, auto-expand). / Rotate mask (nearest, expand)."""
    return pil_mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=fill_value)


def _normalize_mask(mask: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    """Приводит маску к [B,H,W], согласует батч. / Normalize mask to [B,H,W]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            mask = 0.299 * mask[..., 0] + 0.587 * mask[..., 1] + 0.114 * mask[..., 2]
    if mask.shape[0] != batch_size and mask.shape[0] == 1:
        mask = mask.expand(batch_size, -1, -1)
    elif mask.shape[0] != batch_size:
        raise ValueError(f"Mask batch ({mask.shape[0]}) != image batch ({batch_size})")
    return mask.to(device)


def _apply_flip(
    pil_img: Image.Image,
    pil_mask: Image.Image,
    flip_mode: str,
) -> Tuple[Image.Image, Image.Image]:
    """Отражение ПОСЛЕ ресайза. / Flip AFTER resize."""
    if flip_mode == "horizontal":
        return ImageOps.mirror(pil_img), ImageOps.mirror(pil_mask)
    if flip_mode == "vertical":
        return ImageOps.flip(pil_img), ImageOps.flip(pil_mask)
    return pil_img, pil_mask


def _conditional_dims(
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    resize_if_larger: str,
    resize_if_smaller: str,
) -> Tuple[int, int]:
    """
    ЧИСТЫЕ ТРИГГЕРЫ (без аспекта — за «как» отвечает fit_mode):
    условие совпало → целевые размеры; не совпало → исходные.
    none/none = ресайз всегда.
    PURE TRIGGERS (no aspect logic — fit_mode owns the "how"):
    condition triggered → target sizes; otherwise → original ones.
    none/none = always resize.
    """
    if resize_if_larger != "none":
        trig = (
            (resize_if_larger == "width" and orig_w > target_w) or
            (resize_if_larger == "height" and orig_h > target_h) or
            (resize_if_larger == "both" and (orig_w > target_w or orig_h > target_h))
        )
        return (target_w, target_h) if trig else (orig_w, orig_h)
    if resize_if_smaller != "none":
        trig = (
            (resize_if_smaller == "width" and orig_w < target_w) or
            (resize_if_smaller == "height" and orig_h < target_h) or
            (resize_if_smaller == "both" and (orig_w < target_w or orig_h < target_h))
        )
        return (target_w, target_h) if trig else (orig_w, orig_h)
    return target_w, target_h


# =============================================================================
# ХЕЛПЕРЫ ПРЕВЬЮ / PREVIEW HELPERS
# =============================================================================
def _agsoft_probe_image(path: str) -> Tuple[str, int, int]:
    """
    Реальное расширение по формату PIL + размер (браузеру нужен понятный
    суффикс, статус-бару — реальные Ш/В).
    Real extension by PIL format + size (the browser needs a proper suffix,
    the status bar needs real W/H).
    """
    fallback = os.path.splitext(path)[1].lower() or ".png"
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").lower()
            width, height = im.size
        return _PIL_EXT_MAP.get(fmt, fallback), int(width), int(height)
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] probe failed for {path}: {e}")
        return fallback, 0, 0


def _agsoft_cleanup_previews(folder: str, prefix: str, keep: str) -> None:
    """Удаляет устаревшие превью-копии. / Removes stale preview copies."""
    try:
        for name in os.listdir(folder):
            if name.startswith(prefix) and name != keep:
                try:
                    os.remove(os.path.join(folder, name))
                except Exception as e:
                    logger.debug(f"[AGSoft Load Image & Mask Resize] cleanup skipped {name}: {e}")
    except Exception as e:
        logger.debug(f"[AGSoft Load Image & Mask Resize] cleanup failed: {e}")


def _agsoft_ensure_preview_sync(custom_path: str) -> Optional[Dict[str, Any]]:
    """
    БЛОКИРУЮЩАЯ функция: делает файл доступным для просмотра через input и
    возвращает {"image", "width", "height"}. Если файл уже в input — имя
    без копирования. Старые копии того же пути удаляются.
    BLOCKING function: makes the file viewable through input and returns
    {"image", "width", "height"}. If already inside input — the name without
    copying. Old copies of the same path are removed.
    """
    path = os.path.realpath(custom_path)
    if not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] not an image: {path} ({e})")
        return None

    ext, width, height = _agsoft_probe_image(path)
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[AGSoft Load Image & Mask Resize] input dir failed: {e}")
        return None

    base = os.path.basename(path)
    if os.path.dirname(path) == input_dir and not base.startswith(AGSOFT_PREVIEW_PREFIX):
        return {"image": base, "width": width, "height": height}

    try:
        st = os.stat(path)
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] stat failed: {e}")
        return None

    path_hash = hashlib.sha256(path.encode("utf-8", "ignore")).hexdigest()[:16]
    name = f"{AGSOFT_PREVIEW_PREFIX}{path_hash}_{int(st.st_mtime)}_{st.st_size}{ext}"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        tmp = dst + ".tmp"
        try:
            shutil.copy2(path, tmp)
            os.replace(tmp, dst)
        except Exception as e:
            logger.warning(f"[AGSoft Load Image & Mask Resize] preview copy failed: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as cleanup_error:
                logger.debug(f"[AGSoft Load Image & Mask Resize] temp cleanup: {cleanup_error}")
            return None

    _agsoft_cleanup_previews(input_dir, f"{AGSOFT_PREVIEW_PREFIX}{path_hash}_", name)
    return {"image": name, "width": width, "height": height}


def _agsoft_tensor_preview_png(tensor: torch.Tensor, unique_id: Any) -> Optional[Dict[str, Any]]:
    """
    Первый кадр тензора → превью-PNG в input, старые превью ноды удаляются.
    First tensor frame → preview PNG inside input, old node previews removed.
    """
    if tensor is None or not hasattr(tensor, "cpu"):
        return None
    frame = tensor[0] if tensor.dim() == 4 else tensor
    pil = tensor_to_pil(frame, is_mask=False)
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")
    width, height = pil.size

    input_dir = os.path.realpath(folder_paths.get_input_directory())
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[AGSoft Load Image & Mask Resize] input dir failed: {e}")
        return None

    raw_uid = str(unique_id if unique_id is not None else "x")
    uid = "".join(ch for ch in raw_uid if ch.isalnum() or ch in "-_")[:32] or "x"
    name = f"{AGSOFT_PREVIEW_PREFIX}resize_{uid}_{int(time.time() * 1000)}.png"
    dst = os.path.join(input_dir, name)
    try:
        pil.save(dst, "PNG")
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] preview save failed: {e}")
        return None

    _agsoft_cleanup_previews(input_dir, f"{AGSOFT_PREVIEW_PREFIX}resize_{uid}_", name)
    return {"image": name, "width": int(width), "height": int(height)}


def _agsoft_is_loopback(request: Any) -> bool:
    """Копирование файлов разрешено только локальным клиентам. / Local clients only."""
    remote = request.remote if hasattr(request, "remote") else ""
    return str(remote) in ("127.0.0.1", "::1", "localhost")


# =============================================================================
# Состояние превью по unique_id:
# {"image": имя в input, "custom": bool, "stamp": str,
#  "src_w"/"src_h": реальный размер источника, "out_w"/"out_h": размер выхода}.
# "custom" = источник переопределил комбо (тензор или custom_path).
# "stamp"  = строковая метка версии: JS применяет состояние только при новом
# штампе (защита от перезаписей при повторных опросах).
# Preview state by unique_id. "custom" = the source overrode the combo.
# "stamp" is a string version mark: JS applies the state only on a new stamp.
# =============================================================================
_LAST_PREVIEW: Dict[str, Dict[str, Any]] = {}

_server = None
if PromptServer is not None and hasattr(PromptServer, "instance"):
    _server = PromptServer.instance

if _server is not None and web is not None and not hasattr(_server, "_agsoft_image_mask_resize_route"):
    try:
        # Живой апдейт при вводе пути руками (до очереди).
        # Live update while typing a path (before queue).
        @_server.routes.post("/agsoft/image_mask_resize_ensure_preview")
        async def agsoft_image_mask_resize_ensure_preview(request):
            """Готовит превью внешнего файла. / Prepares an external file preview."""
            if not _agsoft_is_loopback(request):
                return web.json_response({"error": "local only"}, status=403)
            try:
                data = await request.json()
            except Exception as e:
                logger.warning(f"[AGSoft Load Image & Mask Resize] bad preview request: {e}")
                data = {}
            custom_path = str(data.get("custom_path", "")).strip()
            if not custom_path:
                return web.json_response({"error": "empty path"}, status=400)
            info = await asyncio.to_thread(_agsoft_ensure_preview_sync, custom_path)
            if not info:
                return web.json_response({"error": "not an image or not found"}, status=400)
            return web.json_response(info)

        # Состояние после выполнения: JS читает по событию executed.
        # Post-execution state: JS reads it on the executed event.
        @_server.routes.get("/agsoft/image_mask_resize_preview_state")
        async def agsoft_image_mask_resize_preview_state(request):
            """Отдаёт состояние превью ноды. / Returns the node preview state."""
            node_id = str(request.query.get("node_id", ""))
            state = _LAST_PREVIEW.get(node_id)
            if not state:
                return web.json_response({
                    "image": "", "custom": False, "stamp": "",
                    "src_w": 0, "src_h": 0, "out_w": 0, "out_h": 0,
                })
            return web.json_response(state)

        _server._agsoft_image_mask_resize_route = True
    except Exception as e:
        logger.error(f"[AGSoft Load Image & Mask Resize] route registration failed: {e}")


# =============================================================================
# ОБЩИЙ ПАЙПЛАЙН: rotate → resize → flip
# =============================================================================
def _resize_batch(
    loaded_image: torch.Tensor,
    loaded_mask: torch.Tensor,
    resize_mode: str,
    target_megapixels: float,
    target_percentage: float,
    target_width: int,
    target_height: int,
    fit_mode: str,
    interpolation: str,
    divisible_by: int,
    crop_position: str,
    pad_image_color: str,
    pad_mask_color: str,
    device: str,
    flip_mode: str,
    angle_degrees: int,
    rotation_preset: str,
    resize_if_larger: str,
    resize_if_smaller: str,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Синхронный пайплайн изображения и маски. / Sync image+mask pipeline."""
    if device == "cuda" and torch.cuda.is_available():
        target_device = "cuda"
    else:
        target_device = "cpu"
        if device == "cuda":
            warnings.warn("CUDA requested but not available. Falling back to CPU.")

    loaded_image = loaded_image.to(target_device)
    loaded_mask = loaded_mask.to(target_device)
    batch_size, orig_h, orig_w, channels = loaded_image.shape
    angle = int(rotation_preset) if rotation_preset != "none" else int(angle_degrees)

    if resize_mode == "target_megapixels":
        target_pixels = max(1, int(target_megapixels * 1_000_000))
        if orig_w <= 0 or orig_h <= 0:
            orig_w, orig_h = 1, 1
        ratio = (target_pixels / (orig_w * orig_h)) ** 0.5
        base_w, base_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))
    elif resize_mode == "target_percentage":
        ratio = abs(target_percentage) / 100.0
        base_w, base_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))
    elif resize_mode == "target_width":
        base_w = max(1, target_width)
        ratio = base_w / orig_w if orig_w > 0 else 1
        base_h = max(1, int(orig_h * ratio))
    elif resize_mode == "target_height":
        base_h = max(1, target_height)
        ratio = base_h / orig_h if orig_h > 0 else 1
        base_w = max(1, int(orig_w * ratio))
    elif resize_mode == "target_both":
        base_w, base_h = max(1, target_width), max(1, target_height)
    elif resize_mode == "target_longest":
        side = max(orig_w, orig_h)
        ratio = target_width / side if side > 0 else 1
        base_w, base_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))
    elif resize_mode == "target_shortest":
        side = min(orig_w, orig_h)
        ratio = target_width / side if side > 0 else 1
        base_w, base_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))
    else:
        raise ValueError(f"Неизвестный режим ресайза: {resize_mode}")

    # Условия — чистые триггеры: совпало → цель, нет → исходный размер.
    # Conditions are pure triggers: triggered → target, else → original size.
    target_w, target_h = _conditional_dims(
        orig_w, orig_h, base_w, base_h,
        resize_if_larger, resize_if_smaller,
    )
    if divisible_by > 0:
        target_w = max(divisible_by, (target_w // divisible_by) * divisible_by)
        target_h = max(divisible_by, (target_h // divisible_by) * divisible_by)

    resized_images, resized_masks = [], []
    has_alpha = (channels == 4)
    image_pad_color = hex_to_rgb_or_rgba(IMAGE_COLORS_HEX[pad_image_color], has_alpha)
    mask_pad_color = hex_to_rgb_or_rgba(MASK_COLORS_HEX[pad_mask_color], False)
    mask_fill = 0 if pad_mask_color == "transparent" else int(mask_pad_color[0])
    resample_method = INTERPOLATION_METHODS[interpolation]
    centering_tuple = CROP_POSITIONS[crop_position]

    for i in range(batch_size):
        pil_image = tensor_to_pil(loaded_image[i], is_mask=False)
        pil_mask = tensor_to_pil(loaded_mask[i], is_mask=True)
        if angle != 0:
            pil_image = _rotate_image(pil_image, angle, resample_method, image_pad_color, has_alpha)
            pil_mask = _rotate_mask(pil_mask, angle, mask_fill)

        use_premul = (pil_image.mode == "RGBA") and (resample_method != Image.NEAREST)
        if use_premul:
            pil_image = _premultiply_rgba(pil_image)

        if fit_mode == "stretch":
            resized_image = pil_image.resize((target_w, target_h), resample=resample_method)
            resized_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
        elif fit_mode == "crop":
            scale = max(target_w / pil_image.width, target_h / pil_image.height)
            fit_w = max(1, int(pil_image.width * scale))
            fit_h = max(1, int(pil_image.height * scale))
            scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
            scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
            dx, dy = max(0, fit_w - target_w), max(0, fit_h - target_h)
            left, top = int(dx * centering_tuple[0]), int(dy * centering_tuple[1])
            resized_image = scaled_image.crop((left, top, left + target_w, top + target_h))
            resized_mask = scaled_mask.crop((left, top, left + target_w, top + target_h))
        elif fit_mode == "pad":
            scale = min(target_w / pil_image.width, target_h / pil_image.height)
            fit_w = max(1, int(pil_image.width * scale))
            fit_h = max(1, int(pil_image.height * scale))
            scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
            scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
            if use_premul:
                bg_image = Image.new("RGBA", (target_w, target_h), image_pad_color)
            elif has_alpha and pad_image_color == "transparent":
                bg_image = Image.new("RGBA", (target_w, target_h), image_pad_color)
            else:
                bg_image = Image.new("RGB", (target_w, target_h), image_pad_color[:3])
            bg_mask = Image.new("L", (target_w, target_h), mask_fill)
            left = int((target_w - fit_w) * centering_tuple[0])
            top = int((target_h - fit_h) * centering_tuple[1])
            if use_premul:
                bg_image.paste(scaled_image, (left, top))
            elif scaled_image.mode == "RGBA" and bg_image.mode == "RGB":
                bg_image.paste(scaled_image.convert("RGB"), (left, top))
            else:
                bg_image.paste(scaled_image, (left, top))
            bg_mask.paste(scaled_mask, (left, top))
            resized_image, resized_mask = bg_image, bg_mask
        elif fit_mode == "proportional":
            # Пропорционально вписать в цель; выход = получившийся размер,
            # без полей и без обрезки.
            # Scale proportionally to fit inside the target; output = the
            # resulting size, no bars, no crop.
            scale = min(target_w / pil_image.width, target_h / pil_image.height)
            new_w = max(1, int(pil_image.width * scale))
            new_h = max(1, int(pil_image.height * scale))
            resized_image = pil_image.resize((new_w, new_h), resample=resample_method)
            resized_mask = pil_mask.resize((new_w, new_h), resample=Image.NEAREST)
        else:
            raise ValueError(f"Неизвестная стратегия: {fit_mode}")

        if use_premul:
            resized_image = _unpremultiply_rgba(resized_image)
        resized_image, resized_mask = _apply_flip(resized_image, resized_mask, flip_mode)
        resized_images.append(pil_to_tensor(resized_image, is_mask=False).unsqueeze(0))
        resized_masks.append(pil_to_tensor(resized_mask, is_mask=True).unsqueeze(0))

    final_image = torch.cat(resized_images, dim=0).to(target_device)
    final_mask = torch.cat(resized_masks, dim=0).to(target_device)
    # Реальный размер выхода (для proportional он может отличаться от цели).
    # Real output size (for proportional it may differ from the target).
    final_height, final_width = int(final_image.shape[1]), int(final_image.shape[2])
    return final_image, final_mask, final_width, final_height


# =============================================================================
# ТУЛТИПЫ И ПОРЯДОК ВИДЖЕТОВ
# Порядок неизменен: старые воркфлоу совместимы по widgets_values.
# The order is fixed: old workflows stay compatible by widgets_values.
# =============================================================================
def _resize_inputs() -> Dict[str, Any]:
    """Виджеты ресайза в заданном порядке. / Resize widgets in the given order."""
    return {
        "resize_mode": (RESIZE_MODES, {
            "default": "target_percentage",
            "tooltip": (
                "Size mode: percentage / megapixels / width / height / both / longest / shortest. "
                "Only the widgets used by this mode stay visible.\n"
                "---\n"
                "Режим размеров: проценты / мегапиксели / ширина / высота / оба / longest / shortest. "
                "Видимыми остаются только виджеты этого режима."
            ),
        }),
        "target_percentage": ("FLOAT", {
            "default": 100.0, "min": 0.0, "max": 1000.0, "step": 0.1,
            "tooltip": (
                "Target size as % of original (100=same, 50=half, 200=double).\n"
                "---\n"
                "Целевой размер в % от исходного (100=тот же, 50=половина, 200=двойной)."
            ),
        }),
        "target_megapixels": ("FLOAT", {
            "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
            "tooltip": (
                "Target megapixels (MP). Used in target_megapixels mode.\n"
                "---\n"
                "Целевые мегапиксели (MP). Используется в режиме target_megapixels."
            ),
        }),
        "target_width": ("INT", {
            "default": 1024, "min": 1, "max": 8192, "step": 1,
            "tooltip": (
                "Target width (px). Also = the side for longest/shortest modes.\n"
                "---\n"
                "Целевая ширина (px). Также = сторона для longest/shortest."
            ),
        }),
        "target_height": ("INT", {
            "default": 1024, "min": 1, "max": 8192, "step": 1,
            "tooltip": (
                "Target height (px).\n"
                "---\n"
                "Целевая высота (px)."
            ),
        }),
        "fit_mode": (FIT_MODES, {
            "default": "crop",
            "tooltip": (
                "Scaling strategy: stretch (distort to exact target) / crop (proportional cover + cut) / "
                "pad (proportional contain + bars) / proportional (fit inside the target, "
                "output = resulting size, no bars, no crop).\n"
                "---\n"
                "Стратегия: stretch (искажить в точную цель) / crop (пропорционально закрыть и обрезать) / "
                "pad (пропорционально вписать с полями) / proportional (вписать в цель, "
                "выход = получившийся размер, без полей и обрезки)."
            ),
        }),
        "crop_position": (list(CROP_POSITIONS.keys()), {
            "default": "center",
            "tooltip": (
                "Anchor for crop/pad (9 positions).\n"
                "---\n"
                "Якорь для crop/pad (9 позиций)."
            ),
        }),
        "interpolation": (list(INTERPOLATION_METHODS.keys()), {
            "default": "lanczos",
            "tooltip": (
                "Interpolation. RGBA auto-uses premultiplied alpha.\n"
                "---\n"
                "Интерполяция. RGBA автоматически с premultiplied alpha."
            ),
        }),
        "resize_if_larger": (CONDITION_MODES, {
            "default": "none",
            "tooltip": (
                "Pure trigger: resize ONLY if the image is LARGER than target; otherwise the original size "
                "is kept. Triggered → resize per fit_mode. none = always resize.\n"
                "---\n"
                "Чистый триггер: ресайз ТОЛЬКО если изображение БОЛЬШЕ цели, иначе исходный размер. "
                "Сработало → ресайз по fit_mode. none = ресайз всегда."
            ),
        }),
        "resize_if_smaller": (CONDITION_MODES, {
            "default": "none",
            "tooltip": (
                "Pure trigger: resize ONLY if the image is SMALLER than target; otherwise the original size "
                "is kept. Triggered → resize per fit_mode. none = off.\n"
                "---\n"
                "Чистый триггер: ресайз ТОЛЬКО если изображение МЕНЬШЕ цели, иначе исходный размер. "
                "Сработало → ресайз по fit_mode. none = выкл."
            ),
        }),
        "pad_image_color": (list(IMAGE_COLORS_HEX.keys()), {
            "default": "white",
            "tooltip": (
                "Padding color for the image (pad) and the rotation fill.\n"
                "---\n"
                "Цвет полей изображения (pad) и заливки при повороте."
            ),
        }),
        "pad_mask_color": (list(MASK_COLORS_HEX.keys()), {
            "default": "black",
            "tooltip": (
                "Padding color for the mask (pad) and the rotation fill.\n"
                "---\n"
                "Цвет полей маски (pad) и заливки при повороте."
            ),
        }),
        "invert_mask": ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Invert the mask before processing.\n"
                "---\n"
                "Инвертировать маску перед обработкой."
            ),
        }),
        "flip_mode": (FLIP_MODES, {
            "default": "none",
            "tooltip": (
                "Mirror AFTER resize: none / horizontal / vertical.\n"
                "---\n"
                "Отражение ПОСЛЕ ресайза: none / horizontal / vertical."
            ),
        }),
        "angle_degrees": ("INT", {
            "default": 0, "min": -360, "max": 360, "step": 1,
            "tooltip": (
                "Rotation angle (counter-clockwise), applied BEFORE resize.\n"
                "---\n"
                "Угол поворота (против часовой), до ресайза."
            ),
        }),
        "rotation_preset": (ROTATION_PRESETS, {
            "default": "none",
            "tooltip": (
                "Quick angle; overrides angle_degrees.\n"
                "---\n"
                "Быстрый угол; переопределяет angle_degrees."
            ),
        }),
        "divisible_by": ("INT", {
            "default": 0, "min": 0, "max": 128, "step": 1,
            "tooltip": (
                "Round sizes to a multiple (0=off).\n"
                "---\n"
                "Кратность размеров (0=выкл)."
            ),
        }),
        "device": (["cpu", "cuda"], {
            "default": "cpu",
            "tooltip": (
                "Compute device: cpu / cuda.\n"
                "---\n"
                "Устройство: cpu / cuda."
            ),
        }),
    }


# =============================================================================
# НОДА: 🖼️AGSoft Load Image & Mask Resize
# =============================================================================
class AGSoft_Load_Image_Mask_Resize:
    # JS лежит рядом с этим файлом в папке web/ (общая папка со всеми нодами).
    # Файл: web/AGSoft_Load_Image_Mask_Resize.js — БЕЗ него не работают
    # авто-скрытие виджетов, статус-бар, сворачивание и превью.
    # JS is located next to this file in the web/ folder (shared folder).
    # File: web/AGSoft_Load_Image_Mask_Resize.js — without it auto-hiding,
    # the status bar, collapsing and the preview do not work.
    WEB_DIRECTORY = "./web"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [
                f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                # Служебные превью-копии не показываем в списке.
                # Hide service preview copies from the list.
                and not f.startswith(AGSOFT_PREVIEW_PREFIX)
            ]
            files = folder_paths.filter_files_content_types(files, ["image"])
        if not files:
            files = [" "]

        # custom_path — ПЕРВАЯ строка ноды (по просьбе автора).
        # custom_path — the FIRST row of the node (author's request).
        required = {
            "custom_path": ("STRING", {
                "default": "",
                "tooltip": (
                    "Optional: absolute path to an image (or a linked string). "
                    "Priority: input_image > custom_path > image combo. The preview follows it "
                    "(live when typed, after execution when linked).\n"
                    "---\n"
                    "Опционально: абсолютный путь к изображению (или линк со строкой). "
                    "Приоритет: input_image > custom_path > комбо image. Превью следует за ним "
                    "(живо при вводе, после выполнения — если линком)."
                ),
            }),
            "image": (sorted(files), {
                "image_upload": True,
                "tooltip": (
                    "Select / upload an image from the input directory (used when input_image "
                    "is not connected and custom_path is empty).\n"
                    "---\n"
                    "Выберите / загрузите изображение из папки input (используется, если не подключён "
                    "input_image и пуст custom_path)."
                ),
            }),
        }
        required.update(_resize_inputs())
        optional = {
            "input_image": ("IMAGE", {
                "tooltip": (
                    "Optional image tensor (priority over the file combo).\n"
                    "---\n"
                    "Опциональный тензор изображения (приоритет над файловым комбо)."
                ),
            }),
            "mask": ("MASK", {
                "tooltip": (
                    "Optional input mask (priority over the alpha-derived mask).\n"
                    "---\n"
                    "Опциональная входная маска (приоритет над альфа-маской)."
                ),
            }),
        }
        hidden = {"unique_id": "UNIQUE_ID"}
        return {"required": required, "optional": optional, "hidden": hidden}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename", "filepath")
    OUTPUT_TOOLTIPS = (
        "Processed image tensor.\n---\nТензор обработанного изображения.",
        "Processed mask tensor.\n---\nТензор обработанной маски.",
        "Output width in pixels.\n---\nШирина результата в пикселях.",
        "Output height in pixels.\n---\nВысота результата в пикселях.",
        "Source file name (or tensor).\n---\nИмя исходного файла (или tensor).",
        "Full path to the source file.\n---\nПолный путь к исходному файлу.",
    )
    FUNCTION = "load_and_resize_image"
    CATEGORY = "AGSoft/Image"
    DESCRIPTION = (
        "🖼️ AGSoft Load Image & Mask Resize.\n"
        "Loads an image (input_image tensor > custom_path > combo/upload) and resizes it with a mask. "
        "rotate → resize → flip, sync image+mask, 7 size modes, pure trigger conditions "
        "(resize_if_larger/smaller), 4 strategies (stretch / crop / pad / proportional), 9 positions, "
        "divisibility, CPU/CUDA, premultiplied alpha. Mode-aware widget auto-hiding, live status bar "
        "(real MP/W/H → recomputed MP/W/H) and full widget collapse (JS from web/).\n"
        "---\n"
        "🖼️ AGSoft Load Image & Mask Resize.\n"
        "Загружает изображение (тензор input_image > custom_path > комбо/upload) и ресайзит с маской. "
        "rotate → resize → flip, синхронные image+mask, 7 режимов, условия-триггеры "
        "(resize_if_larger/smaller), 4 стратегии (stretch / crop / pad / proportional), 9 позиций, "
        "кратность, CPU/CUDA, premultiplied alpha. Авто-скрытие виджетов по режиму, живой статус-бар "
        "(реальные МП/Ш/В → пересчитанные МП/Ш/В) и полное сворачивание виджетов (JS из web/)."
    )

    @classmethod
    def IS_CHANGED(cls, image: str = "", custom_path: str = "", **kwargs) -> Any:
        try:
            if custom_path and os.path.exists(custom_path):
                return f"{os.path.getmtime(custom_path)}_{os.path.getsize(custom_path)}"
            if image and str(image).strip():
                path = folder_paths.get_annotated_filepath(image)
                if os.path.exists(path):
                    return f"{os.path.getmtime(path)}_{os.path.getsize(path)}"
        except Exception as e:
            logger.debug(f"[AGSoft Load Image & Mask Resize] IS_CHANGED: {e}")
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image: str = "", custom_path: str = "", **kwargs) -> Any:
        # РАЗРЕШАЮЩАЯ валидация: подключённые тензоры приходят сюда как
        # None/отсутствуют, поэтому пустое комбо — НЕ ошибка.
        # Permissive validation: connected tensors arrive as None/missing,
        # so an empty combo is NOT an error.
        if custom_path:
            if not os.path.exists(custom_path):
                return f"Custom path does not exist: {custom_path}"
            if not os.path.isfile(custom_path):
                return f"Custom path is not a file: {custom_path}"
            return True
        if image and str(image).strip():
            if not folder_paths.exists_annotated_filepath(image):
                return f"Invalid image file: {image}"
        return True

    def load_and_resize_image(
        self,
        image: str = "",
        resize_mode: str = "target_percentage",
        target_percentage: float = 100.0,
        target_megapixels: float = 1.0,
        target_width: int = 1024,
        target_height: int = 1024,
        fit_mode: str = "crop",
        crop_position: str = "center",
        interpolation: str = "lanczos",
        resize_if_larger: str = "none",
        resize_if_smaller: str = "none",
        pad_image_color: str = "white",
        pad_mask_color: str = "black",
        invert_mask: bool = False,
        flip_mode: str = "none",
        angle_degrees: int = 0,
        rotation_preset: str = "none",
        divisible_by: int = 0,
        device: str = "cpu",
        input_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        custom_path: str = "",
        unique_id: Any = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, str, str]:
        try:
            # =================================================================
            # БЛОК ЗАГРУЗКИ ИСТОЧНИКА.
            # Приоритет: input_image > custom_path > комбо image.
            # SOURCE LOADING BLOCK.
            # Priority: input_image > custom_path > image combo.
            # =================================================================
            preview_info = None
            used_override = False

            if input_image is not None:
                # ---------- Приоритет 1: тензор изображения ----------
                loaded_image = input_image
                batch_size, orig_h, orig_w, channels = loaded_image.shape
                if mask is not None:
                    loaded_mask = _normalize_mask(mask, batch_size, "cpu")
                elif channels == 4:
                    loaded_mask = 1.0 - loaded_image[..., 3]
                else:
                    loaded_mask = torch.zeros((batch_size, orig_h, orig_w), dtype=torch.float32)
                has_name = bool(image and str(image).strip())
                filename_with_ext = os.path.basename(image) if has_name else "tensor"
                full_path = folder_paths.get_annotated_filepath(image) if has_name else ""
                # Превью: первый кадр тензора. / Preview: first tensor frame.
                try:
                    preview_info = _agsoft_tensor_preview_png(loaded_image, unique_id)
                except Exception as e:
                    logger.warning(f"[AGSoft Load Image & Mask Resize] tensor preview: {e}")
                    preview_info = None
                used_override = True
            else:
                # ---------- Приоритет 2/3: custom_path > комбо image ----------
                used_custom = bool(custom_path) and os.path.exists(custom_path)
                if not used_custom and (not image or not str(image).strip()):
                    raise ValueError(
                        "[AGSoft Load Image & Mask Resize] Не задан источник: "
                        "подключите input_image, укажите custom_path или выберите файл в image."
                    )
                if used_custom:
                    image_path = os.path.realpath(custom_path)
                else:
                    image_path = folder_paths.get_annotated_filepath(image)
                filename_with_ext = os.path.basename(image_path)
                full_path = image_path

                output_images, output_masks = [], []
                first_w = first_h = None
                with node_helpers.pillow(Image.open, image_path) as img:
                    img_format = img.format
                    for frame in ImageSequence.Iterator(img):
                        frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
                        if frame.mode == "I":
                            frame = frame.point(lambda value: value * (1 / 255))
                        frame_rgb = frame.convert("RGB")
                        if not output_images:
                            first_w, first_h = frame_rgb.size
                        if frame_rgb.size[0] != first_w or frame_rgb.size[1] != first_h:
                            continue
                        array = np.array(frame_rgb).astype(np.float32) / 255.0
                        output_images.append(torch.from_numpy(array)[None, ])
                        if "A" in frame.getbands():
                            alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                            output_masks.append((1.0 - torch.from_numpy(alpha)).unsqueeze(0))
                        elif frame.mode == "P" and "transparency" in frame.info:
                            alpha = np.array(frame.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                            output_masks.append((1.0 - torch.from_numpy(alpha)).unsqueeze(0))
                        else:
                            output_masks.append(
                                torch.zeros((first_h, first_w), dtype=torch.float32).unsqueeze(0)
                            )

                if not output_images:
                    raise ValueError(f"No readable frames / Нет читаемых кадров: {image_path}")
                if len(output_images) > 1 and img_format not in ("MPO",):
                    loaded_image = torch.cat(output_images, dim=0)
                    loaded_mask = torch.cat(output_masks, dim=0)
                else:
                    loaded_image = output_images[0]
                    loaded_mask = output_masks[0]

                batch_size = loaded_image.shape[0]
                if mask is not None:
                    loaded_mask = _normalize_mask(mask, batch_size, "cpu")
                # Превью: файл (копия в input при необходимости).
                # Preview: the file (copied into input when needed).
                try:
                    preview_info = _agsoft_ensure_preview_sync(image_path)
                except Exception as e:
                    logger.warning(f"[AGSoft Load Image & Mask Resize] file preview: {e}")
                    preview_info = None
                used_override = used_custom

            # =================================================================
            # РЕСАЙЗ-ПАЙПЛАЙН / RESIZE PIPELINE
            # =================================================================
            if invert_mask:
                loaded_mask = 1.0 - loaded_mask
            final_image, final_mask, fw, fh = _resize_batch(
                loaded_image, loaded_mask,
                resize_mode, target_megapixels, target_percentage,
                target_width, target_height, fit_mode, interpolation,
                divisible_by, crop_position, pad_image_color, pad_mask_color,
                device, flip_mode, angle_degrees, rotation_preset,
                resize_if_larger, resize_if_smaller,
            )

            # Состояние для JS: превью + реальные размеры источника и выхода.
            # State for JS: preview + real source and output sizes.
            if unique_id is not None:
                info = preview_info or {}
                _LAST_PREVIEW[str(unique_id)] = {
                    "image": info.get("image", ""),
                    "custom": bool(used_override),
                    "stamp": str(time.time_ns()),
                    "src_w": int(loaded_image.shape[2]),
                    "src_h": int(loaded_image.shape[1]),
                    "out_w": int(fw),
                    "out_h": int(fh),
                }

            return (final_image, final_mask, fw, fh, filename_with_ext, full_path)
        except Exception as e:
            raise RuntimeError(
                f"Ошибка при загрузке и ресайзе: {e}\n"
                f"Файл: {custom_path or image}\nПараметры: resize_mode={resize_mode}, fit_mode={fit_mode}"
            )


NODE_CLASS_MAPPINGS = {
    "AGSoft_Load_Image_Mask_Resize": AGSoft_Load_Image_Mask_Resize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Load_Image_Mask_Resize": "🖼️AGSoft Load Image & Mask Resize",
}