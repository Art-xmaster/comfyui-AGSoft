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
    Превью следует за реально использованным источником и показывает тот же
    пайплайн, что и сервер. Состояние превью привязано к ключу источника
    (source): JS применяет его только если источник не менялся.
    Порядок трансформаций: rotate → resize → flip.
    Условия ресайза — ЧИСТЫЕ ТРИГГЕРЫ: resize_if_larger/smaller отвечают
    только за «ресайзить ли»; за «как» отвечает fit_mode:
    stretch / crop / pad / proportional.
    Сервер отдаёт списки изображений input и output для окна эскизов,
    генерирует уменьшенные WebP-эскизы с кэшем в temp и импортирует картинку
    из output/temp в input, чтобы выбранный файл стал обычным источником.
    ---
    Loads an image (combo + upload + optional input_image + custom_path as
    the first row) and resizes it synchronously with a mask (from the alpha
    channel or the input mask).
    Source priority: input_image > custom_path > image combo.
    The preview follows the actually used source and shows the same pipeline
    as the server. The preview state is bound to a source key, so JS applies
    it only while the source is unchanged.
    Transform order: rotate → resize → flip.
    Resize conditions are PURE TRIGGERS: resize_if_larger/smaller only decide
    "whether to resize"; fit_mode owns the "how": stretch / crop / pad /
    proportional.
    The server lists input and output images for the thumbnail picker,
    generates small cached WebP thumbnails in temp and imports a picture
    from output/temp into input so the chosen file becomes a regular source.
Возможности / Features:
    ⚡ Источники: input_image (приоритет) > custom_path > комбо image (+upload).
       Sources: input_image (priority) > custom_path > image combo (+upload).
    ⚡ Окно эскизов: папки input и output, подпапки, кэш уменьшенных WebP.
       Thumbnail picker: input and output folders, subfolders, cached WebP.
    ⚡ Выбор из output копирует файл в input и сразу делает его источником.
       Picking from output copies the file into input and selects it.
    ⚡ Состояние превью привязано к источнику (без подмены после вкладок).
       Preview state is bound to the source (no stale swap after tab switch).
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
    web/AGSoft_Load_Image_Mask_Resize.js — без него не работают превью,
    эскизы, авто-скрытие виджетов, статус-бар и сворачивание.
    web/AGSoft_Load_Image_Mask_Resize.js — without it the preview, thumbnails,
    auto-hiding, status bar and collapsing do not work.
Автор / Author: AGSoft
Дата / Date: 16.11.2026
==============================================================================
"""

import os
import time
import hashlib
import shutil
import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

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

# print("[AGSoft Load Image & Mask Resize] v11.16 loaded (input/output thumbnail picker + combo arrows)")

# =============================================================================
# Префикс временных превью-копий в папке input.
# Prefix for temporary preview copies inside the input folder.
# =============================================================================
AGSOFT_PREVIEW_PREFIX = "agsoft_preview"
AGSOFT_THUMBS_LIMIT = 6000
AGSOFT_THUMBS_SCAN_LIMIT = 20000
AGSOFT_THUMBS_DEPTH = 4
AGSOFT_THUMBS_TTL = 5.0
AGSOFT_THUMB_DIR = "agsoft_thumbs"
AGSOFT_THUMB_SIZES = (128, 256, 384)
AGSOFT_THUMB_TTL = 7 * 24 * 3600
AGSOFT_THUMB_CLEAN_EVERY = 600.0

_AGSOFT_IMAGE_EXTS = (
    ".png", ".jpg", ".jpeg", ".jfif", ".webp", ".gif",
    ".bmp", ".tif", ".tiff", ".ico", ".avif",
)

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
    PURE TRIGGERS (no aspect logic — fit_mode owns the "how"):
    condition triggered → target sizes; otherwise → original ones.
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
# ХЕЛПЕРЫ ПРЕВЬЮ И ФАЙЛОВ / PREVIEW AND FILE HELPERS
# =============================================================================
def _agsoft_probe_image(path: str) -> Tuple[str, int, int]:
    """Реальное расширение формата PIL + размер. / Real PIL format suffix + size."""
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


def _agsoft_safe_name(name: str) -> str:
    """Имя файла без путей и служебных символов. / File name without paths or unsafe chars."""
    base = os.path.basename(str(name).replace("\\", "/"))
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in " ._-()[]").strip()
    return (cleaned or "image.png")[:120]


def _agsoft_root_for_type(kind: str) -> str:
    """Корневая папка ComfyUI по типу. / ComfyUI root folder by type."""
    if kind == "output":
        return os.path.realpath(folder_paths.get_output_directory())
    if kind == "temp":
        return os.path.realpath(folder_paths.get_temp_directory())
    return os.path.realpath(folder_paths.get_input_directory())


def _agsoft_within(path: str, root: str) -> bool:
    """Проверка, что путь внутри корня. / Checks the path stays inside the root."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _agsoft_media_path(filename: str, subfolder: str, kind: str) -> Optional[str]:
    """Безопасный путь к файлу внутри input/output/temp. / Safe path inside a root."""
    root = _agsoft_root_for_type(kind)
    safe = _agsoft_safe_name(filename)
    sub = str(subfolder or "").replace("\\", "/").strip("/")
    parts = [part for part in sub.split("/") if part not in ("", ".", "..")]
    path = os.path.realpath(os.path.join(root, *parts, safe))
    if not _agsoft_within(path, root) or not os.path.isfile(path):
        return None
    return path


def _agsoft_ensure_preview_sync(custom_path: str) -> Optional[Dict[str, Any]]:
    """
    БЛОКИРУЮЩАЯ: делает файл доступным для просмотра через input и возвращает
    {"image", "width", "height"}. Старые копии того же пути удаляются.
    BLOCKING: makes the file viewable through input and returns
    {"image", "width", "height"}. Old copies of the same path are removed.
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


def _agsoft_import_sync(filename: str, subfolder: str, kind: str) -> Optional[Dict[str, Any]]:
    """
    Копирует картинку из input/output/temp в корень input под чистым именем,
    чтобы она стала обычным источником комбо.
    Copies a picture from input/output/temp into the input root under a clean
    name so it becomes a regular combo source.
    """
    src = _agsoft_media_path(filename, subfolder, kind)
    if src is None:
        return None
    try:
        with Image.open(src) as im:
            im.verify()
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] import rejected {src}: {e}")
        return None

    ext, width, height = _agsoft_probe_image(src)
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    try:
        os.makedirs(input_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[AGSoft Load Image & Mask Resize] input dir failed: {e}")
        return None

    if os.path.dirname(src) == input_dir:
        return {"image": os.path.basename(src), "width": width, "height": height}

    try:
        st = os.stat(src)
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] stat failed: {e}")
        return None

    stem, suffix = os.path.splitext(_agsoft_safe_name(filename))
    tag = hashlib.sha256(f"{src}|{int(st.st_mtime)}|{st.st_size}".encode("utf-8", "ignore")).hexdigest()[:8]
    name = f"{stem}_{tag}{suffix or ext}"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        tmp = dst + ".tmp"
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception as e:
            logger.warning(f"[AGSoft Load Image & Mask Resize] import copy failed: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as cleanup_error:
                logger.debug(f"[AGSoft Load Image & Mask Resize] temp cleanup: {cleanup_error}")
            return None
    return {"image": name, "width": width, "height": height}


def _agsoft_thumbs_scan(kind: str) -> Dict[str, Any]:
    """
    Список изображений папки (с подпапками) для окна эскизов, свежие сверху.
    Folder images (with subfolders) for the thumbnail picker, newest first.
    """
    files: List[Dict[str, Any]] = []
    folders: List[str] = []
    root = _agsoft_root_for_type(kind)
    if not os.path.isdir(root):
        return {"files": files, "folders": folders}

    base_depth = root.rstrip(os.sep).count(os.sep)
    # В output папки часто названы по дате: обход с конца даёт свежие файлы
    # даже если сработал лимит сканирования.
    # Output folders are often date-named: a reverse walk keeps the newest
    # files in the list even when the scan limit is reached.
    reverse = kind == "output"
    try:
        for current, dirs, names in os.walk(root):
            dirs.sort(reverse=reverse)
            depth = current.rstrip(os.sep).count(os.sep) - base_depth
            if depth >= AGSOFT_THUMBS_DEPTH:
                dirs[:] = []
            rel = os.path.relpath(current, root)
            sub = "" if rel == "." else rel.replace("\\", "/")
            if sub:
                folders.append(sub)
            for name in names:
                if kind == "input" and name.startswith(AGSOFT_PREVIEW_PREFIX):
                    continue
                if os.path.splitext(name)[1].lower() not in _AGSOFT_IMAGE_EXTS:
                    continue
                full = os.path.join(current, name)
                try:
                    st = os.stat(full)
                except Exception as e:
                    logger.debug(f"[AGSoft Load Image & Mask Resize] thumb stat {name}: {e}")
                    continue
                files.append({
                    "name": name,
                    "subfolder": sub,
                    "mtime": int(st.st_mtime),
                    "size": int(st.st_size),
                })
                if len(files) >= AGSOFT_THUMBS_SCAN_LIMIT:
                    break
            if len(files) >= AGSOFT_THUMBS_SCAN_LIMIT:
                break
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] thumbnail scan failed: {e}")

    files.sort(key=lambda item: item["mtime"], reverse=True)
    folders.sort()
    return {"files": files[:AGSOFT_THUMBS_LIMIT], "folders": folders}


# Кэш списков: обход дерева не должен повторяться при каждом открытии окна.
# List caches: the tree walk must not repeat on every picker opening.
_THUMBS_CACHE: Dict[str, Dict[str, Any]] = {
    "input": {"stamp": 0.0, "data": None},
    "output": {"stamp": 0.0, "data": None},
}
_THUMB_CLEAN_STAMP = 0.0


def _agsoft_thumbs_cached(kind: str, force: bool) -> Dict[str, Any]:
    """Список с коротким TTL. / Listing with a short TTL."""
    bucket = _THUMBS_CACHE["output"] if kind == "output" else _THUMBS_CACHE["input"]
    now = time.time()
    cached = bucket["data"]
    stamp = float(bucket["stamp"])
    if not force and cached is not None and now - stamp < AGSOFT_THUMBS_TTL:
        return cached
    data = _agsoft_thumbs_scan(kind)
    bucket["data"] = data
    bucket["stamp"] = now
    return data


def _agsoft_cleanup_thumbs(folder: str) -> None:
    """Редкая уборка старых эскизов. / Occasional cleanup of old thumbnails."""
    global _THUMB_CLEAN_STAMP
    now = time.time()
    if now - _THUMB_CLEAN_STAMP < AGSOFT_THUMB_CLEAN_EVERY:
        return
    _THUMB_CLEAN_STAMP = now
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if now - os.path.getmtime(path) > AGSOFT_THUMB_TTL:
                    os.remove(path)
            except Exception as e:
                logger.debug(f"[AGSoft Load Image & Mask Resize] thumb cleanup {name}: {e}")
    except Exception as e:
        logger.debug(f"[AGSoft Load Image & Mask Resize] thumb cleanup failed: {e}")


def _agsoft_thumb_sync(filename: str, subfolder: str, size: int, kind: str) -> Optional[str]:
    """
    Уменьшенный WebP-эскиз с кэшем в temp: браузер тянет десятки килобайт
    вместо полных кадров, повторные открытия окна мгновенные.
    Small cached WebP thumbnail in temp: the browser pulls kilobytes instead
    of full frames, and reopening the picker is instant.
    """
    src = _agsoft_media_path(filename, subfolder, kind)
    if src is None:
        return None
    try:
        st = os.stat(src)
    except Exception as e:
        logger.warning(f"[AGSoft Load Image & Mask Resize] thumb stat failed: {e}")
        return None

    cache_dir = os.path.join(os.path.realpath(folder_paths.get_temp_directory()), AGSOFT_THUMB_DIR)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"[AGSoft Load Image & Mask Resize] thumb dir failed: {e}")
        return None

    key_source = f"{src}|{int(st.st_mtime)}|{st.st_size}|{size}"
    key = hashlib.sha256(key_source.encode("utf-8", "ignore")).hexdigest()[:24]
    dst = os.path.join(cache_dir, f"{key}.webp")
    if not os.path.exists(dst):
        tmp = dst + ".tmp"
        try:
            with Image.open(src) as source:
                # Быстрое декодирование JPEG в уменьшенном масштабе.
                # Fast JPEG decode at a reduced scale.
                source.draft("RGB", (size * 2, size * 2))
                frame = ImageOps.exif_transpose(source)
                frame.thumbnail((size, size), Image.LANCZOS, reducing_gap=2.0)
                if frame.mode not in ("RGB", "RGBA"):
                    frame = frame.convert("RGBA" if "A" in frame.getbands() else "RGB")
                frame.save(tmp, "WEBP", quality=82, method=4)
            os.replace(tmp, dst)
        except Exception as e:
            logger.warning(f"[AGSoft Load Image & Mask Resize] thumbnail failed for {src}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as cleanup_error:
                logger.debug(f"[AGSoft Load Image & Mask Resize] temp cleanup: {cleanup_error}")
            return None

    _agsoft_cleanup_thumbs(cache_dir)
    return dst


def _agsoft_tensor_preview_png(tensor: torch.Tensor, unique_id: Any) -> Optional[Dict[str, Any]]:
    """Первый кадр тензора → превью-PNG в input. / First tensor frame → preview PNG."""
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
# Состояние превью по unique_id. "source" — ключ источника на момент запуска:
# JS применяет состояние только если ключ совпал с текущим.
# Preview state by unique_id. "source" is the source key at execution time:
# JS applies the state only when the key matches the current one.
# =============================================================================
_LAST_PREVIEW: Dict[str, Dict[str, Any]] = {}

_server = None
if PromptServer is not None and hasattr(PromptServer, "instance"):
    _server = PromptServer.instance

if _server is not None and web is not None and not hasattr(_server, "_agsoft_image_mask_resize_route"):
    try:
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

        @_server.routes.post("/agsoft/image_mask_resize_import")
        async def agsoft_image_mask_resize_import(request):
            """Импорт картинки в input для комбо. / Imports a picture into input."""
            if not _agsoft_is_loopback(request):
                return web.json_response({"error": "local only"}, status=403)
            try:
                data = await request.json()
            except Exception as e:
                logger.warning(f"[AGSoft Load Image & Mask Resize] bad import request: {e}")
                data = {}
            filename = str(data.get("filename", "")).strip()
            subfolder = str(data.get("subfolder", "")).strip()
            kind = str(data.get("type", "input")).strip().lower()
            if not filename:
                return web.json_response({"error": "empty filename"}, status=400)
            info = await asyncio.to_thread(_agsoft_import_sync, filename, subfolder, kind)
            if not info:
                return web.json_response({"error": "not an image or not found"}, status=400)
            return web.json_response(info)

        @_server.routes.get("/agsoft/image_mask_resize_thumbs")
        async def agsoft_image_mask_resize_thumbs(request):
            """Список изображений папки для эскизов. / Folder image list for thumbnails."""
            force = str(request.query.get("force", "")) == "1"
            kind = str(request.query.get("type", "input")).strip().lower()
            data = await asyncio.to_thread(_agsoft_thumbs_cached, kind, force)
            return web.json_response(data)

        @_server.routes.get("/agsoft/image_mask_resize_thumb")
        async def agsoft_image_mask_resize_thumb(request):
            """Отдаёт готовый уменьшенный эскиз. / Serves a ready small thumbnail."""
            filename = str(request.query.get("filename", ""))
            subfolder = str(request.query.get("subfolder", ""))
            kind = str(request.query.get("type", "input")).strip().lower()
            try:
                wanted = int(request.query.get("size", "256"))
            except Exception:
                wanted = 256
            size = min(AGSOFT_THUMB_SIZES, key=lambda item: abs(item - wanted))
            path = await asyncio.to_thread(_agsoft_thumb_sync, filename, subfolder, size, kind)
            if not path:
                return web.Response(status=404)
            try:
                with open(path, "rb") as handle:
                    payload = handle.read()
            except Exception as e:
                logger.warning(f"[AGSoft Load Image & Mask Resize] thumb read failed: {e}")
                return web.Response(status=500)
            return web.Response(
                body=payload,
                content_type="image/webp",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        @_server.routes.get("/agsoft/image_mask_resize_preview_state")
        async def agsoft_image_mask_resize_preview_state(request):
            """Отдаёт состояние превью ноды. / Returns the node preview state."""
            node_id = str(request.query.get("node_id", ""))
            state = _LAST_PREVIEW.get(node_id)
            if not state:
                return web.json_response({
                    "image": "", "custom": False, "stamp": "", "source": "",
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
            # Пропорционально вписать в цель; выход = получившийся размер.
            # Scale proportionally to fit inside the target; output = result.
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
    # Файл: web/AGSoft_Load_Image_Mask_Resize.js — БЕЗ него не работают превью,
    # эскизы, авто-скрытие виджетов, статус-бар и сворачивание.
    # File: web/AGSoft_Load_Image_Mask_Resize.js — without it the preview,
    # thumbnails, auto-hiding, status bar and collapsing do not work.
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
                    "Priority: input_image > custom_path > image combo.\n"
                    "---\n"
                    "Опционально: абсолютный путь к изображению (или линк со строкой). "
                    "Приоритет: input_image > custom_path > комбо image."
                ),
            }),
            "image": (sorted(files), {
                "image_upload": True,
                "tooltip": (
                    "Select / upload an image from the input directory. Click the edges of the widget "
                    "to step through files, click the middle to open the thumbnail picker "
                    "(input / output, search, subfolders).\n"
                    "---\n"
                    "Выберите / загрузите изображение из папки input. Клик по краям виджета листает "
                    "файлы, клик по центру открывает окно эскизов (input / output, поиск, подпапки)."
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
        "rotate → resize → flip, sync image+mask, 7 size modes, pure trigger conditions, "
        "4 strategies (stretch / crop / pad / proportional), 9 positions, divisibility, CPU/CUDA, "
        "premultiplied alpha. Input/output thumbnail picker, Assets-panel drag&drop, live pipeline "
        "preview, mode-aware widget hiding, status bar and full collapse (JS from web/).\n"
        "---\n"
        "🖼️ AGSoft Load Image & Mask Resize.\n"
        "Загружает изображение (тензор input_image > custom_path > комбо/upload) и ресайзит с маской. "
        "rotate → resize → flip, синхронные image+mask, 7 режимов, условия-триггеры, "
        "4 стратегии (stretch / crop / pad / proportional), 9 позиций, кратность, CPU/CUDA, "
        "premultiplied alpha. Эскизы input/output, drag&drop из панели Assets, живое превью пайплайна, "
        "авто-скрытие виджетов, статус-бар и сворачивание (JS из web/)."
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
            source_key = ""

            if input_image is not None:
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
                try:
                    preview_info = _agsoft_tensor_preview_png(loaded_image, unique_id)
                except Exception as e:
                    logger.warning(f"[AGSoft Load Image & Mask Resize] tensor preview: {e}")
                    preview_info = None
                used_override = True
                source_key = "tensor"
            else:
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
                try:
                    preview_info = _agsoft_ensure_preview_sync(image_path)
                except Exception as e:
                    logger.warning(f"[AGSoft Load Image & Mask Resize] file preview: {e}")
                    preview_info = None
                used_override = used_custom
                source_key = str(custom_path).strip() if used_custom else str(image)

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

            if unique_id is not None:
                info = preview_info or {}
                _LAST_PREVIEW[str(unique_id)] = {
                    "image": info.get("image", ""),
                    "custom": bool(used_override),
                    "stamp": str(time.time_ns()),
                    "source": source_key,
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