# ==============================================================================
# AGSoft_Load_Image_Mask.py
# ==============================================================================
# Нода: 🖼️AGSoft Load Image & Mask
#
# Простая нода загрузки изображения (и маски из альфа-канала) из папки input
# ИЛИ по кастомному пути, с превью (image_upload) как в Load Image.
# Входы/выходы как у 🖼️AGSoft Image & Mask Resize Plus.
# Simple node to load an image (and mask from alpha channel) from the input
# directory OR a custom path, with a preview (image_upload) like Load Image.
# Inputs/outputs match 🖼️AGSoft Image & Mask Resize Plus.
#
# Возможности / Features:
# ⚡ Выходы: image, mask, width, height, filename, filepath (как Resize Plus).
#    Outputs: image, mask, width, height, filename, filepath (like Resize Plus).
# ⚡ Два источника: custom_path (приоритет) > комбо image.
#    Two sources: custom_path (priority) > image combo.
# ⚡ Превью и кнопка загрузки (image_upload=True).
#    Preview and upload button (image_upload=True).
# ⚡ НАТИВНОЕ превью следует за custom_path:
#    - при вводе пути руками — живой апдейт (endpoint ensure_preview);
#    - при подключённом ЛИНКЕ custom_path — апдейт после выполнения
#      (сервер помнит состояние по unique_id, JS слушает событие executed).
#    NATIVE preview follows custom_path:
#    - typed path — live update (ensure_preview endpoint);
#    - LINKED custom_path — update after execution (server stores state by
#      unique_id, JS listens to the executed event).
# ⚡ Маска из альфа-канала (или чёрная, если альфы нет).
#    Mask from alpha channel (or black if no alpha).
# ⚡ Поддержка анимированных изображений и EXIF.
#    Animated images and EXIF support.
#
# Автор / Author: AGSoft
# Дата / Date: 22.08.2026
# ==============================================================================

import os
import asyncio
import hashlib
import shutil

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

from typing import Tuple

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

# Маркер версии: если этой строки нет в консоли после старта — файл не применился.
# Version marker: if this line is not in console after startup — file was not applied.
print("[AGSoft Load Image & Mask] v1.4 loaded (preview follows custom_path: widget + LINK + executed)")

# ==============================================================================
# Префикс временных превью-копий в папке input.
# Prefix for temporary preview copies inside the input folder.
# ==============================================================================
AGSOFT_PREVIEW_PREFIX = "__agsoft_preview__"

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


def _agsoft_real_ext(path: str) -> str:
    """
    Реальное расширение по формату PIL (чтобы браузер мог показать превью).
    Real extension by PIL format (so the browser can display the preview).
    """
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").lower()
        return _PIL_EXT_MAP.get(fmt, os.path.splitext(path)[1].lower() or ".png")
    except Exception:
        return os.path.splitext(path)[1].lower() or ".png"


def _agsoft_ensure_preview_sync(custom_path: str):
    """
    БЛОКИРУЮЩАЯ функция (вызывается через asyncio.to_thread):
    делает файл из custom_path доступным через /view?type=input и возвращает
    его имя в папке input. Если файл уже лежит в input — возвращает его имя
    без копирования. Старые копии того же пути удаляются.

    BLOCKING function (called via asyncio.to_thread):
    makes the custom_path file available via /view?type=input and returns
    its name inside the input folder. If the file already lives in input —
    returns its name without copying. Old copies of the same path are removed.
    """
    custom_path = os.path.abspath(custom_path)

    if not os.path.isfile(custom_path):
        return None

    # Проверка, что это читаемое изображение. / Check it is a readable image.
    try:
        with Image.open(custom_path) as im:
            im.verify()
    except Exception:
        return None

    input_dir = os.path.abspath(folder_paths.get_input_directory())
    os.makedirs(input_dir, exist_ok=True)

    base = os.path.basename(custom_path)

    # Файл уже в input (и не наша служебная копия) — копирование не нужно.
    # File already in input (and not our service copy) — no copy needed.
    if os.path.dirname(custom_path) == input_dir and not base.startswith(AGSOFT_PREVIEW_PREFIX):
        return base

    st = os.stat(custom_path)
    path_hash = hashlib.sha256(custom_path.encode("utf-8", "ignore")).hexdigest()[:16]
    ext = _agsoft_real_ext(custom_path)

    # Имя включает mtime+size → новый файл = новое имя (обход кэша браузера).
    # Name includes mtime+size → new file = new name (browser cache busting).
    name = f"{AGSOFT_PREVIEW_PREFIX}{path_hash}_{int(st.st_mtime)}_{st.st_size}{ext}"
    dst = os.path.join(input_dir, name)

    if not os.path.exists(dst):
        tmp = dst + ".tmp"
        shutil.copy2(custom_path, tmp)
        os.replace(tmp, dst)

    # Удаляем старые копии ТОЛЬКО этого же пути (другие ноды не трогаем).
    # Remove old copies of THIS path only (other nodes untouched).
    prefix = f"{AGSOFT_PREVIEW_PREFIX}{path_hash}_"
    try:
        for f in os.listdir(input_dir):
            if f.startswith(prefix) and f != name:
                try:
                    os.remove(os.path.join(input_dir, f))
                except Exception:
                    pass
    except Exception:
        pass

    return name


# ==============================================================================
# Состояние превью по unique_id ноды: {"image": имя в input, "custom": bool}.
# Заполняется ПРИ ВЫПОЛНЕНИИ ноды — это единственный способ узнать реальное
# значение custom_path, пришедшего по ЛИНКУ.
# Preview state by node unique_id: {"image": input name, "custom": bool}.
# Filled AT EXECUTION — the only way to know the real custom_path value
# that arrived through a LINK.
# ==============================================================================
_LAST_PREVIEW = {}

_server = getattr(PromptServer, "instance", None) if PromptServer is not None else None

if _server is not None and web is not None and not getattr(_server, "_agsoft_image_mask_route", False):
    try:
        # ----------------------------------------------------------------------
        # Живой апдейт при вводе пути руками (до очереди).
        # Live update while typing a path (before queue).
        # ----------------------------------------------------------------------
        @_server.routes.post("/agsoft/image_mask_ensure_preview")
        async def agsoft_image_mask_ensure_preview(request):
            try:
                data = await request.json()
            except Exception:
                data = {}

            custom_path = str(data.get("custom_path", "")).strip()
            if not custom_path:
                return web.json_response({"error": "empty path"}, status=400)

            # Копирование/проверка — в отдельном потоке, не блокируем event loop.
            # Copy/check — in a separate thread, don't block the event loop.
            name = await asyncio.to_thread(_agsoft_ensure_preview_sync, custom_path)

            if not name:
                return web.json_response({"error": "not an image or not found"}, status=400)

            return web.json_response({"image": name})

        # ----------------------------------------------------------------------
        # Состояние после выполнения: JS дёргает его по событию executed.
        # Post-execution state: JS polls it on the executed event.
        # ----------------------------------------------------------------------
        @_server.routes.get("/agsoft/image_mask_preview_state")
        async def agsoft_image_mask_preview_state(request):
            node_id = str(request.query.get("node_id", ""))
            st = _LAST_PREVIEW.get(node_id)
            if not st:
                return web.json_response({"image": "", "custom": False})
            return web.json_response(st)

        _server._agsoft_image_mask_route = True
    except Exception as e:
        print(f"[AGSoft Load Image & Mask] route registration failed: {e}")


# ==============================================================================
# Основная нода
# Main node
# ==============================================================================
class AGSoftLoadImageMask:

    # JS лежит рядом с этим файлом в папке web/.
    # JS is located next to this file in the web/ folder.
    WEB_DIRECTORY = "./web"

    @classmethod
    def INPUT_TYPES(cls):
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

        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True,
                    "tooltip": (
                        "Выберите / загрузите изображение из папки input. "
                        "Используется, если не задан custom_path.\n"
                        "---\n"
                        "Select / upload an image from the input directory. "
                        "Used when custom_path is not set."
                    ),
                }),
            },
            "optional": {
                "custom_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Опционально: абсолютный путь к изображению (или линк со строкой). "
                        "Переопределяет комбо. Превью автоматически переключается на этот "
                        "файл (после выполнения, если путь пришёл линком).\n"
                        "---\n"
                        "Optional: absolute path to an image (or a linked string). "
                        "Overrides the combo. The preview automatically switches to this "
                        "file (after execution when the path arrives via a link)."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename", "filepath")

    FUNCTION = "load_image_mask"

    CATEGORY = "AGSoft/Image"

    DESCRIPTION = (
        "🖼️ AGSoft Load Image & Mask.\n"
        "Loads an image (and mask from alpha channel) from input dir or a custom path, with "
        "preview. Outputs: image, mask, width, height, filename, filepath.\n"
        "The native preview follows custom_path: live when typed, after execution when linked.\n"
        "---\n"
        "🖼️ AGSoft Load Image & Mask.\n"
        "Загружает изображение (и маску из альфа-канала) из папки input или по кастомному пути, "
        "с превью. Выходы: image, mask, width, height, filename, filepath.\n"
        "Нативное превью следует за custom_path: живо при вводе руками, после выполнения — "
        "если путь пришёл линком."
    )

    @classmethod
    def IS_CHANGED(cls, image: str, custom_path: str = "", **kwargs):
        try:
            if custom_path and os.path.exists(custom_path):
                return f"{os.path.getmtime(custom_path)}_{os.path.getsize(custom_path)}"

            if image and str(image).strip():
                p = folder_paths.get_annotated_filepath(image)
                if os.path.exists(p):
                    return f"{os.path.getmtime(p)}_{os.path.getsize(p)}"
        except Exception:
            pass

        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, custom_path: str = "", **kwargs):
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

    def load_image_mask(self, image: str, custom_path: str = "", unique_id=None, **kwargs) -> Tuple:
        try:
            # Приоритет: custom_path > комбо image.
            # Priority: custom_path > image combo.
            used_custom = bool(custom_path) and os.path.exists(custom_path)

            if used_custom:
                image_path = os.path.abspath(custom_path)
            else:
                image_path = folder_paths.get_annotated_filepath(image)

            filename_with_ext = os.path.basename(image_path)
            full_path = image_path

            img = node_helpers.pillow(Image.open, image_path)

            output_images = []
            output_masks = []
            w = h = None

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda x: x * (1 / 255))

                image_rgb = i.convert("RGB")

                if not output_images:
                    w, h = image_rgb.size

                if image_rgb.size[0] != w or image_rgb.size[1] != h:
                    continue

                image_array = np.array(image_rgb).astype(np.float32) / 255.0
                output_images.append(torch.from_numpy(image_array)[None, ])

                if 'A' in i.getbands():
                    m = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    output_masks.append((1.0 - torch.from_numpy(m)).unsqueeze(0))
                elif i.mode == 'P' and 'transparency' in i.info:
                    m = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    output_masks.append((1.0 - torch.from_numpy(m)).unsqueeze(0))
                else:
                    output_masks.append(torch.zeros((h, w), dtype=torch.float32).unsqueeze(0))

            if len(output_images) > 1 and img.format not in ['MPO']:
                loaded_image = torch.cat(output_images, dim=0)
                loaded_mask = torch.cat(output_masks, dim=0)
            else:
                loaded_image = output_images[0]
                loaded_mask = output_masks[0]

            # ------------------------------------------------------------------
            # Превью: делаем файл доступным через /view и запоминаем состояние
            # ноды (для JS, по событию executed).
            # Preview: make the file available via /view and remember the node
            # state (for JS, on the executed event).
            # ------------------------------------------------------------------
            preview_name = None
            try:
                preview_name = _agsoft_ensure_preview_sync(image_path)
            except Exception:
                preview_name = None

            if unique_id is not None:
                _LAST_PREVIEW[str(unique_id)] = {
                    "image": preview_name or "",
                    "custom": bool(used_custom),
                }

            result = (loaded_image, loaded_mask, w, h, filename_with_ext, full_path)

            ui = {
                "agsoft_preview": (
                    [{"filename": preview_name, "subfolder": "", "type": "input"}]
                    if preview_name else []
                )
            }

            return {"result": result, "ui": ui}

        except Exception as e:
            raise RuntimeError(f"[AGSoft Load Image & Mask] Ошибка загрузки: {e}\nФайл: {image}")


NODE_CLASS_MAPPINGS = {
    "AGSoftLoadImageMask": AGSoftLoadImageMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadImageMask": "🖼️AGSoft Load Image & Mask",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']