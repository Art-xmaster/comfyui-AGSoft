# ==============================================================================
# AGSoft_Save_Image_Plus.py
# ==============================================================================
# Нода: 🖼️💾AGSoft Save Image Plus
# Описание / Description:
# Расширенная нода сохранения изображений с ПРЕВЬЮ ВСЕГО БАТЧА в ноде и
# кнопкой "Save now" НАД КАЖДЫМ изображением.
#
# Два режима работы:
# - save_image=True  → пишет все изображения батча в output сразу + превью;
# - save_image=False → пишет только временные превью (preview-only, без мусора
#   в output); кнопка "Save now" над каждым изображением сохраняет его в
#   output по требованию с текущими настройками виджетов (формат, качество,
#   путь, вшивание воркфлоу).
#
# Превью: для каждого изображения батча пишется temp-PNG и отдаётся в
# ui.agsoft_previews (фронтенд этот ключ не рисует — превью и кнопки строит
# JS). Сетка превью адаптивная и вписывается в текущий размер ноды.
#
# Выходы (как в AGSoft Image & Mask Resize Plus):
# IMAGE (проход), width, height (первого изображения), filename, saved_path
# (для батча — через запятую).
#
# Поддерживает PNG/JPG/WebP/BMP с раздельными настройками сжатия/качества,
# подпапки (output_path) и подпапку с датой (create_dated_subfolder),
# вшивание workflow в PNG (tEXt-чанки) и отдельный .json для JPG/WebP/BMP.
#
# Extended image saver with a PREVIEW OF THE WHOLE BATCH inside the node and
# a "Save now" button OVER EACH image.
#
# Two modes:
# - save_image=True  → writes all batch images to output immediately + preview;
# - save_image=False → writes temp previews only (no clutter in output); the
#   "Save now" button over each image saves it to output on demand with the
#   current widget settings (format, quality, path, workflow embedding).
#
# Preview: a temp PNG is written for each batch image and returned in
# ui.agsoft_previews (the frontend does not render this key — the JS builds
# the previews and buttons). The preview grid is adaptive and fits the
# current node size.
#
# Outputs (like AGSoft Image & Mask Resize Plus):
# IMAGE (passed through), width, height (of the first image), filename,
# saved_path (comma-joined for batches).
#
# Supports PNG/JPG/WebP/BMP with per-format compression/quality settings,
# subfolders (output_path) and a dated subfolder (create_dated_subfolder),
# workflow embedding into PNG (tEXt chunks) and a separate .json for
# JPG/WebP/BMP.
#
# Возможности / Features:
# ⚡ Превью всего батча в ноде + кнопка "Save now" над каждым изображением.
#   Whole-batch preview in the node + a "Save now" button over each image.
# ⚡ Два режима: save_image=True (весь батч в output) / False (только temp).
#   Two modes: save_image=True (whole batch to output) / False (temp only).
# ⚡ Выходы как в Image Resize Plus: images / width / height / filename / saved_path.
#   Outputs like Image Resize Plus: images / width / height / filename / saved_path.
# ⚡ PNG/JPG/WebP/BMP с раздельными настройками сжатия/качества.
#   PNG/JPG/WebP/BMP with per-format compression/quality settings.
# ⚡ Подпапки (output_path) + подпапка с датой (create_dated_subfolder).
#   Subfolders (output_path) + dated subfolder (create_dated_subfolder).
# ⚡ Вшивание workflow: PNG tEXt / отдельный .json для JPG/WebP/BMP.
#   Workflow embedding: PNG tEXt / separate .json for JPG/WebP/BMP.
# ⚡ Кнопка "Save now" шлёт воркфлоу из браузера (app.graph.serialize()).
#   The "Save now" button sends the workflow from the browser (app.graph.serialize()).
# ⚡ Адаптивная сетка превью на канвасе, вписывается в размер ноды (JS).
#   Adaptive canvas preview grid fitted into the node size (JS).
# ⚡ OUTPUT_NODE=True, IS_CHANGED, VALIDATE_INPUTS.
#   OUTPUT_NODE=True, IS_CHANGED, VALIDATE_INPUTS.
#
# Автор / Author: AGSoft
# Дата / Date: 20.08.2026
# ==============================================================================

import os
import re
import json
import time
import logging
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
import numpy as np
import torch

from aiohttp import web
from server import PromptServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Маркер версии: если этой строки нет в консоли после старта — файл не применился.
# Version marker: if this line is not in console after startup — file was not applied.
# print("[AGSoft Save Image Plus] v20.08 loaded (style header + bilingual tooltips + batch preview + Save now over each image)")

# ------------------------------------------------------------------------------
# Форматы и их параметры по умолчанию.
# Formats and their default parameters.
# ------------------------------------------------------------------------------
FORMAT_PRESETS = {
    "png":  {"ext": "png",  "default_q": 1,  "q_min": 0, "q_max": 9},
    "jpg":  {"ext": "jpg",  "default_q": 90, "q_min": 1, "q_max": 100},
    "webp": {"ext": "webp", "default_q": 90, "q_min": 1, "q_max": 100},
    "bmp":  {"ext": "bmp",  "default_q": 0,  "q_min": 0, "q_max": 0},
}


def _tensor_to_pil(img_tensor):
    """Конвертирует тензор IMAGE [H,W,C] (float 0..1) в PIL Image."""
    arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return Image.fromarray(arr)


def _sanitize_subfolder(s):
    """Санация подпапки: ".." и абсолютные пути отбрасываются."""
    if not s:
        return ""
    parts = []
    for p in str(s).replace("\\", "/").split("/"):
        p = p.strip()
        if p and p not in (".", ".."):
            parts.append(p)
    return "/".join(parts)


def _build_save_dir(output_path, create_dated_subfolder):
    """Строит путь сохранения: output + output_path + (опц.) YYYY-MM-DD."""
    base = folder_paths.get_output_directory()
    sub = _sanitize_subfolder(output_path)
    target = os.path.join(base, sub) if sub else base
    if create_dated_subfolder:
        target = os.path.join(target, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(target, exist_ok=True)
    return target


def _next_filename(directory, prefix, ext, overwrite=False, batch_index=None):
    """Генерирует имя файла с номером: prefix_001.ext, prefix_002.ext..."""
    try:
        if overwrite:
            num = 1 if batch_index is None else batch_index
            return os.path.join(directory, f"{prefix}_{num:03d}.{ext}")

        if batch_index is not None:
            base_with_idx = f"{prefix}_{batch_index:03d}"
            pattern = re.compile(
                rf"^{re.escape(base_with_idx)}_(\d+)\.{re.escape(ext)}$"
            )
        else:
            pattern = re.compile(
                rf"^{re.escape(prefix)}_(\d+)\.{re.escape(ext)}$"
            )

        max_num = 0
        if os.path.exists(directory):
            for f in os.listdir(directory):
                m = pattern.match(f)
                if m:
                    n = int(m.group(1))
                    if n > max_num:
                        max_num = n

        if batch_index is not None:
            name = f"{base_with_idx}_{max_num + 1:03d}.{ext}"
        else:
            name = f"{prefix}_{max_num + 1:03d}.{ext}"

        return os.path.join(directory, name)

    except Exception as e:
        logger.warning(f"[AGSoft Save Image Plus] next filename failed: {e}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(directory, f"{prefix}_{stamp}.{ext}")


def _save_image_to_path(pil_img, target_path, fmt, quality, embed_workflow,
                        prompt, extra_pnginfo):
    """Сохраняет PIL-изображение в файл с формат-специфичными параметрами."""
    kwargs = {}

    if fmt == "png":
        kwargs["compress_level"] = min(9, max(0, int(quality)))
        if embed_workflow:
            meta = PngInfo()
            if prompt is not None:
                meta.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    try:
                        meta.add_text(k, json.dumps(v))
                    except Exception:
                        pass
            kwargs["pnginfo"] = meta

    elif fmt == "jpg":
        kwargs["quality"] = min(100, max(1, int(quality)))
        kwargs["optimize"] = True
        kwargs["progressive"] = True

    elif fmt == "webp":
        kwargs["quality"] = min(100, max(1, int(quality)))
        kwargs["method"] = 4

    pil_img.save(target_path, **kwargs)
    return target_path


def _save_workflow_json(image_path, prompt, extra_pnginfo):
    """Сохраняет отдельный .json рядом с изображением (для JPG/WebP/BMP)."""
    try:
        workflow = None
        if extra_pnginfo and "workflow" in extra_pnginfo:
            workflow = extra_pnginfo["workflow"]
        else:
            workflow = {"prompt": prompt or {}, "extra_pnginfo": extra_pnginfo or {}}

        json_path = os.path.splitext(image_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(workflow, fh, ensure_ascii=False, indent=2)
        return json_path
    except Exception as e:
        logger.warning(f"[AGSoft Save Image Plus] JSON save failed: {e}")
        return ""


def _perform_save(pil_img, save_dir, prefix, fmt, quality,
                  embed_workflow, prompt, extra_pnginfo,
                  overwrite, batch_index=None):
    """
    Общая функция сохранения: генерит путь, пишет файл, (для JPG/WebP/BMP)
    пишет отдельный .json.
    """
    os.makedirs(save_dir, exist_ok=True)
    ext = FORMAT_PRESETS[fmt]["ext"]
    target_path = _next_filename(save_dir, prefix, ext, overwrite, batch_index)

    _save_image_to_path(
        pil_img, target_path, fmt, quality,
        embed_workflow, prompt, extra_pnginfo,
    )

    if embed_workflow and fmt in ("jpg", "webp", "bmp"):
        _save_workflow_json(target_path, prompt, extra_pnginfo)

    logger.info(f"[AGSoft Save Image Plus] saved: {target_path}")
    return target_path


# ------------------------------------------------------------------------------
# ENDPOINT: /agsoft/save_now — кнопка "Save now" из JS сохраняет конкретное
# превью (temp-файл) в output с текущими настройками виджетов.
# ВАЖНО: воркфлоу приходит ИЗ БРАУЗЕРА (app.graph.serialize()) в поле
# "workflow" — без него вшивать в изображение было бы нечего.
#
# ENDPOINT: /agsoft/save_now — the "Save now" button in JS saves a specific
# preview (temp file) to output with the current widget settings.
# IMPORTANT: the workflow comes FROM THE BROWSER (app.graph.serialize()) in
# the "workflow" field — without it there would be nothing to embed.
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.post("/agsoft/save_now")
async def agsoft_save_now(request):
    try:
        data = await request.json()

        temp_path = data.get("temp_path", "") or ""
        temp_filename = data.get("temp_filename", "") or ""

        if not temp_path and temp_filename:
            temp_path = os.path.join(folder_paths.get_temp_directory(), temp_filename)

        if not temp_path or not os.path.isfile(temp_path):
            return web.json_response({"ok": False, "error": "temp file missing"})

        # Защита: temp_path должен быть в temp.
        # Safety: temp_path must be inside temp.
        tmp_base = os.path.abspath(folder_paths.get_temp_directory())
        if not os.path.abspath(temp_path).startswith(tmp_base):
            return web.json_response({"ok": False, "error": "path not allowed"})

        params = data.get("params", {})
        prefix = params.get("filename_prefix", "image") or "image"
        output_path = params.get("output_path", "") or ""
        create_dated = bool(params.get("create_dated_subfolder", True))
        fmt = params.get("image_format", "png") or "png"
        if fmt not in FORMAT_PRESETS:
            fmt = "png"

        q_key = f"{fmt}_quality" if fmt != "png" else "png_compression"
        quality = params.get(q_key, FORMAT_PRESETS[fmt]["default_q"])
        try:
            quality = int(quality)
        except Exception:
            quality = FORMAT_PRESETS[fmt]["default_q"]

        overwrite = bool(params.get("overwrite_existing", False))
        embed_wf = bool(params.get("embed_workflow", True))

        # Воркфлоу приходит из браузера (app.graph.serialize()).
        # The workflow comes from the browser (app.graph.serialize()).
        workflow = data.get("workflow") or None
        prompt = data.get("prompt") or None
        extra_pnginfo = {"workflow": workflow} if workflow is not None else None

        save_dir = _build_save_dir(output_path, create_dated)

        try:
            pil_img = Image.open(temp_path).convert("RGB")
        except Exception as e:
            return web.json_response({"ok": False, "error": f"cannot open: {e}"})

        out_path = _perform_save(
            pil_img, save_dir, prefix, fmt, quality,
            embed_wf, prompt, extra_pnginfo,
            overwrite=overwrite,
        )

        return web.json_response({
            "ok": True,
            "saved_path": out_path,
            "filename": os.path.basename(out_path),
            "directory": os.path.dirname(out_path),
        })

    except Exception as e:
        logger.warning(f"[AGSoft Save Image Plus] save_now failed: {e}")
        return web.json_response({"ok": False, "error": str(e)})


class AGSoftSaveImagePlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_image": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True = save all batch images to output immediately.\n"
                            "False = preview only (temp previews); use the 'Save now' button "
                            "over each image to save later with current widget settings.\n"
                            "---\n"
                            "True = сохранять весь батч в output сразу.\n"
                            "False = только превью (temp); сохраняйте позже кнопкой 'Save now' "
                            "над каждым изображением с текущими настройками виджетов."
                        )
                    }
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "image",
                        "tooltip": (
                            "Base filename without extension. A number is added automatically "
                            "(001, 002...).\n"
                            "---\n"
                            "Базовое имя файла без расширения. Номер добавляется автоматически "
                            "(001, 002...)."
                        )
                    }
                ),
                "output_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Subfolder inside output (e.g. 'my_project'). Nested paths like "
                            "'project/sub' are allowed. '..' and absolute paths are stripped.\n"
                            "---\n"
                            "Подпапка внутри output (например 'my_project'). Разрешены вложенные "
                            "пути 'project/sub'. '..' и абсолютные пути отбрасываются."
                        )
                    }
                ),
                "create_dated_subfolder": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Add a YYYY-MM-DD subfolder inside output_path.\n"
                            "---\n"
                            "Добавить подпапку ГГГГ-ММ-ДД внутри output_path."
                        )
                    }
                ),
                "image_format": (
                    list(FORMAT_PRESETS.keys()),
                    {
                        "default": "png",
                        "tooltip": (
                            "Output image format.\n"
                            "---\n"
                            "Формат выходного изображения."
                        )
                    }
                ),
                "png_compression": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 9,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "PNG compression: 0=no compression (fast, large files), 6=balance, "
                            "9=max compression (slow, small files).\n"
                            "---\n"
                            "Сжатие PNG: 0=без сжатия (быстро, большие файлы), 6=баланс, "
                            "9=максимум (медленно, маленькие файлы)."
                        )
                    }
                ),
                "jpg_quality": (
                    "INT",
                    {
                        "default": 90,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "JPEG quality: 1=worst (smallest file), 70=balance, 90=high, "
                            "100=best (largest file).\n"
                            "---\n"
                            "Качество JPEG: 1=худшее (маленький файл), 70=баланс, 90=высокое, "
                            "100=лучшее (большой файл)."
                        )
                    }
                ),
                "webp_quality": (
                    "INT",
                    {
                        "default": 90,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "slider",
                        "tooltip": (
                            "WebP quality: 1=worst, 75=balance, 90=high, 100=best (lossless "
                            "above 90).\n"
                            "---\n"
                            "Качество WebP: 1=худшее, 75=баланс, 90=высокое, 100=лучшее "
                            "(без потерь выше 90)."
                        )
                    }
                ),
                "overwrite_existing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overwrite existing files with the same name instead of incrementing "
                            "the number.\n"
                            "---\n"
                            "Перезаписывать существующие файлы с тем же именем вместо увеличения "
                            "номера."
                        )
                    }
                ),
                "embed_workflow": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "PNG: embed prompt+workflow into the image (tEXt chunks, opens on "
                            "drag&drop).\n"
                            "JPG/WebP/BMP: save a separate .json next to the image.\n"
                            "---\n"
                            "PNG: вшить prompt+workflow в изображение (tEXt-чанки, открывается "
                            "перетаскиванием).\n"
                            "JPG/WebP/BMP: сохранить отдельный .json рядом с изображением."
                        )
                    }
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image(s) to save. Batch supported — every image gets its own "
                            "preview and 'Save now' button.\n"
                            "---\n"
                            "Изображение(я) для сохранения. Батч поддерживается — у каждого "
                            "изображения своё превью и кнопка 'Save now'."
                        )
                    }
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    # Пропускаем IMAGE дальше + width/height/filename/saved_path (как в Image Resize Plus).
    # Pass IMAGE through + width/height/filename/saved_path (like Image Resize Plus).
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "width", "height", "filename", "saved_path")
    FUNCTION = "save_image_plus"
    CATEGORY = "AGSoft/Image"

    OUTPUT_NODE = True

    # JS из web/ рисует превью батча и кнопки "Save now" над каждым изображением.
    # JS from web/ draws the batch preview and the "Save now" buttons over each image.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🖼️💾 AGSoft Save Image Plus.\n"
        "Advanced image saver with a PREVIEW OF THE WHOLE BATCH inside the node and a "
        "'Save now' button OVER EACH image.\n"
        "Two modes: save_image=True writes all batch images to output immediately; "
        "save_image=False writes temp previews only (no clutter) — save any image later with "
        "its 'Save now' button using the current widget settings.\n"
        "Outputs: IMAGE (passed through), width, height, filename, saved_path (comma-joined "
        "for batches) — like AGSoft Image & Mask Resize Plus.\n"
        "PNG/JPG/WebP/BMP with per-format compression/quality, subfolders, dated subfolder, "
        "workflow embedding (PNG tEXt / separate .json for others).\n"
        "---\n"
        "🖼️💾 AGSoft Save Image Plus.\n"
        "Расширенная нода сохранения с ПРЕВЬЮ ВСЕГО БАТЧА в ноде и кнопкой 'Save now' НАД "
        "КАЖДЫМ изображением.\n"
        "Два режима: save_image=True — весь батч в output сразу; save_image=False — только "
        "temp-превью (без мусора), любое изображение сохраняется своей кнопкой 'Save now' с "
        "текущими настройками виджетов.\n"
        "Выходы: IMAGE (прокидывается), width, height, filename, saved_path (для батча — через "
        "запятую) — как в AGSoft Image & Mask Resize Plus.\n"
        "PNG/JPG/WebP/BMP с раздельным сжатием/качеством, подпапки, подпапка с датой, вшивание "
        "workflow (PNG tEXt / отдельный .json для остальных)."
    )

    def save_image_plus(
        self,
        save_image,
        filename_prefix,
        output_path,
        create_dated_subfolder,
        image_format,
        png_compression,
        jpg_quality,
        webp_quality,
        overwrite_existing,
        embed_workflow,
        images=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        try:
            if images is None or len(images) == 0:
                # Ничего не подключено — тихий no-op, чтобы не ронять весь prompt.
                # No images connected — silent no-op so the whole prompt doesn't fail.
                return {
                    "ui": {"agsoft_previews": []},
                    "result": (torch.zeros((1, 8, 8, 3), dtype=torch.float32), 0, 0, "", ""),
                }

            fmt = image_format if image_format in FORMAT_PRESETS else "png"
            if fmt == "png":
                quality = png_compression
            elif fmt == "jpg":
                quality = jpg_quality
            elif fmt == "webp":
                quality = webp_quality
            else:
                quality = 0

            ts = time.strftime("%Y%m%d_%H%M%S")
            tmp_dir = folder_paths.get_temp_directory()
            os.makedirs(tmp_dir, exist_ok=True)

            batch_size = images.shape[0]
            is_batch = batch_size > 1

            previews = []
            saved_paths = []
            saved_names = []
            first_w = first_h = 0

            out_dir = None
            if save_image:
                out_dir = _build_save_dir(output_path, create_dated_subfolder)

            # ------------------------------------------------------------------
            # Каждое изображение батча: temp-превью (для JS) + (опц.) сохранение.
            # Each batch image: temp preview (for JS) + (optional) save.
            # ------------------------------------------------------------------
            for i in range(batch_size):
                pil = _tensor_to_pil(images[i])
                if i == 0:
                    first_w, first_h = pil.size

                temp_name = f"agsoft_imgplus_{ts}_{i:03d}.png"
                temp_path = os.path.join(tmp_dir, temp_name)
                pil.save(temp_path, "PNG")

                saved_path = ""
                saved_name = ""
                if save_image:
                    saved_path = _perform_save(
                        pil, out_dir, filename_prefix, fmt, quality,
                        embed_workflow, prompt, extra_pnginfo,
                        overwrite_existing,
                        batch_index=(i + 1) if is_batch else None,
                    )
                    saved_name = os.path.basename(saved_path)
                    saved_paths.append(saved_path)
                    saved_names.append(saved_name)

                previews.append({
                    "filename": temp_name,
                    "subfolder": "",
                    "type": "temp",
                    "index": i,
                    "saved_filename": saved_name,
                    "saved_path": saved_path,
                })

            return {
                # Свой ключ: фронтенд его не рисует — превью и кнопки строит JS.
                # Custom key: the frontend does not render it — JS builds the
                # previews and buttons.
                "ui": {"agsoft_previews": previews},
                "result": (
                    images,
                    first_w,
                    first_h,
                    ",".join(saved_names),
                    ",".join(saved_paths),
                ),
            }

        except Exception as e:
            err = f"[AGSoft Save Image Plus] error: {e}"
            logger.error(err)
            raise RuntimeError(err)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Всегда перегенериваем: имя файла содержит временную метку.
        # Always re-run: the filename contains a timestamp.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Подключённые входы (линки) приходят как None — реальное значение
        # станет известно только при выполнении.
        # Connected inputs (links) arrive as None — the real value is only
        # known at execution time.
        if kwargs.get("images") is not None:
            return True
        return True


NODE_CLASS_MAPPINGS = {
    "AGSoftSaveImagePlus": AGSoftSaveImagePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftSaveImagePlus": "🖼️💾AGSoft Save Image Plus"
}