"""
==============================================================================
AGSoft_Paint_Plus.py
==============================================================================
Нода: 🎨 AGSoft Paint Plus
Описание / Description:
Интерактивный холст для рисования эскизов (позы, сцены, дома, комиксы) с
выдачей рисунка тензором IMAGE и маски штрихов MASK. Рисование происходит
в JS-фронтенде (DOM-виджет), рисунок хранится PNG-файлом в temp-папке;
нода получает только ключ файла и собирарует из него тензоры.
---
Interactive canvas for drawing sketches (poses, scenes, houses, comics)
with the drawing emitted as an IMAGE tensor and a stroke MASK. Drawing
happens in the JS frontend (DOM widget) and is stored as a PNG file in the
temp folder; the node receives only the file key and builds tensors.

Возможности / Features:
⚡ Кисти: circle, square, spray, soft (аэрограф), flat (перо 45 градусов)
Brushes: circle, square, spray, soft (airbrush), flat (45 deg chisel)
⚡ Фигуры: line, circle, square, rect, star, triangle, diamond, ellipse,
arrow, polygon, text и 4 вида speech bubble для комиксов
Shapes: line, circle, square, rect, star, triangle, diamond, ellipse,
arrow, polygon, text and 4 speech bubble variants for comics
⚡ Свойства штриха: прозрачность 0.05..1.0, заливка замкнутых фигур
Stroke properties: opacity 0.05..1.0, fill for closed shapes
⚡ Загрузка референса: stretch, fit с полями, crop по центру (upload_mode)
Reference upload: stretch, fit with bars, center crop (upload_mode)
⚡ Выходы: IMAGE (рисунок поверх цвета фона), MASK (альфа штрихов),
ширина и высота холста
Outputs: IMAGE (drawing over background color), MASK (stroke alpha),
canvas width and height
⚡ Хранение: PNG в temp/agsoft_paint, в виджете и workflow только ключ;
кнопка Save и меню правого клика пишут PNG/маску в выбранную папку
или в output/agsoft_paint
Storage: PNG in temp/agsoft_paint, widget and workflow hold only the key;
Save button and right-click menu write PNG/mask to a chosen folder or
into output/agsoft_paint

Роуты / Routes:
POST /agsoft/paint_plus/save     - запись PNG рисунка в temp (loopback)
GET  /agsoft/paint_plus/load     - отдача PNG рисунка по ключу
POST /agsoft/paint_plus/save_png - запись изображения/маски в output

Автор / Author: AGSoft
Дата / Date: 29.10.2026
==============================================================================
"""

import os
import re
import time
import base64
import logging
import numpy as np
import torch
from PIL import Image
import folder_paths
from aiohttp import web
from server import PromptServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Paint Plus] v10.29 loaded (canvas to image+mask, save menu)")

PAINT_DIR = os.path.join(folder_paths.get_temp_directory(), "agsoft_paint")
os.makedirs(PAINT_DIR, exist_ok=True)

OUTPUT_PAINT_DIR = os.path.join(folder_paths.get_output_directory(), "agsoft_paint")

MAX_PAINT_BYTES = 32 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KEY_PATTERN = re.compile(r"agsoft_paint_[A-Za-z0-9_\-]+\.png")
OUT_PATTERN = re.compile(r"agsoft_paint_(image|mask)_[A-Za-z0-9_\-]+\.png")

LOOPBACK_HOSTS = ("127.0.0.1", "::1", "::ffff:127.0.0.1")


def _paint_path(key):
    """Путь файла рисунка по ключу с containment-проверкой внутри temp.
    Drawing file path by key with containment check inside temp dir."""
    name = os.path.basename(str(key or ""))
    if not KEY_PATTERN.fullmatch(name):
        return None
    root = os.path.realpath(PAINT_DIR)
    path = os.path.realpath(os.path.join(PAINT_DIR, name))
    if not path.startswith(root + os.sep):
        return None
    return path


def _hex_to_rgb(text):
    """HEX-цвет в float 0..1; фолбэк — белый. / HEX to 0..1 floats, fallback white."""
    s = str(text).strip().lstrip("#")
    if len(s) == 6:
        try:
            return (int(s[0:2], 16) / 255.0,
                    int(s[2:4], 16) / 255.0,
                    int(s[4:6], 16) / 255.0)
        except ValueError:
            return 1.0, 1.0, 1.0
    return 1.0, 1.0, 1.0


@PromptServer.instance.routes.post("/agsoft/paint_plus/save")
async def agsoft_paint_plus_save(request):
    if request.remote not in LOOPBACK_HOSTS:
        return web.json_response({"error": "loopback only"}, status=403)
    try:
        payload = await request.json()
        key = str(payload.get("key", ""))
        data = str(payload.get("data", ""))
        if not data.startswith("data:image/png;base64,"):
            return web.json_response({"error": "bad data"}, status=400)
        raw = base64.b64decode(data.split(",", 1)[1])
        if len(raw) > MAX_PAINT_BYTES or raw[:8] != PNG_SIGNATURE:
            return web.json_response({"error": "bad png"}, status=400)
        path = _paint_path(key)
        if path is None:
            return web.json_response({"error": "bad key"}, status=400)
        # Temp-папка очищается ComfyUI на старте: пересоздаём перед записью.
        # ComfyUI clears temp on startup: recreate the dir before writing.
        os.makedirs(PAINT_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
        return web.json_response({"ok": True, "key": os.path.basename(path)})
    except Exception as e:
        logger.warning("paint save failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.get("/agsoft/paint_plus/load")
async def agsoft_paint_plus_load(request):
    path = _paint_path(request.query.get("key", ""))
    if path is None or not os.path.isfile(path):
        return web.json_response({"error": "not found"}, status=404)
    chunks = []
    try:
        with open(path, "rb") as f:
            f.seek(0)
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except Exception as e:
        logger.warning("paint load failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)
    return web.Response(body=b"".join(chunks), content_type="image/png")


@PromptServer.instance.routes.post("/agsoft/paint_plus/save_png")
async def agsoft_paint_plus_save_png(request):
    """PNG изображения или маски в output/agsoft_paint для дальнейшего
    использования в ComfyUI. / Image or mask PNG into output/agsoft_paint
    for further use inside ComfyUI."""
    if request.remote not in LOOPBACK_HOSTS:
        return web.json_response({"error": "loopback only"}, status=403)
    try:
        payload = await request.json()
        data = str(payload.get("data", ""))
        kind = "mask" if str(payload.get("kind", "")) == "mask" else "image"
        name = os.path.basename(str(payload.get("name", "")))
        if not data.startswith("data:image/png;base64,"):
            return web.json_response({"error": "bad data"}, status=400)
        raw = base64.b64decode(data.split(",", 1)[1])
        if len(raw) > MAX_PAINT_BYTES or raw[:8] != PNG_SIGNATURE:
            return web.json_response({"error": "bad png"}, status=400)
        if not OUT_PATTERN.fullmatch(name):
            name = f"agsoft_paint_{kind}_{int(time.time() * 1000)}.png"
        os.makedirs(OUTPUT_PAINT_DIR, exist_ok=True)
        root = os.path.realpath(OUTPUT_PAINT_DIR)
        path = os.path.realpath(os.path.join(OUTPUT_PAINT_DIR, name))
        if not path.startswith(root + os.sep):
            return web.json_response({"error": "bad name"}, status=400)
        with open(path, "wb") as f:
            f.write(raw)
        return web.json_response({
            "ok": True, "name": name, "subfolder": "agsoft_paint"
        })
    except Exception as e:
        logger.warning("paint output save failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)


class AGSoftPaintPlus:
    DESCRIPTION = (
        "Interactive canvas: brushes (circle/square/spray/soft/flat), shapes "
        "(line, circle, square, rect, star, triangle, diamond, ellipse, arrow, "
        "polygon, text, 4 speech bubbles), opacity, fill, undo/redo.\n"
        "Upload modes: stretch, fit with bars, center crop.\n"
        "Save button and right-click menu: copy/save PNG or mask to a chosen "
        "folder or into output/agsoft_paint.\n"
        "Outputs: IMAGE tensor, MASK (stroke alpha), width, height.\n"
        "---\n"
        "Интерактивный холст: кисти (circle/square/spray/soft/flat), фигуры "
        "(line, circle, square, rect, star, triangle, diamond, ellipse, arrow, "
        "polygon, text, 4 вида speech bubble), прозрачность, заливка, undo/redo.\n"
        "Режимы загрузки: stretch, fit с полями, crop по центру.\n"
        "Кнопка Save и меню правого клика: копирование и сохранение PNG или "
        "маски в выбранную папку или в output/agsoft_paint.\n"
        "Выходы: тензор IMAGE, маска MASK (альфа штрихов), ширина, высота."
    )

    CATEGORY = "AGSoft/Image"
    FUNCTION = "process_canvas"
    WEB_DIRECTORY = "./web"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": (
                        "Canvas width in pixels.\n"
                        "---\n"
                        "Ширина холста в пикселях."
                    ),
                }),
                "height": ("INT", {
                    "default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": (
                        "Canvas height in pixels.\n"
                        "---\n"
                        "Высота холста в пикселях."
                    ),
                }),
                "canvas_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": (
                        "Background color (HEX), live under transparent strokes.\n"
                        "---\n"
                        "Цвет фона (HEX), живой под прозрачными штрихами."
                    ),
                }),
                "brush_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": (
                        "Brush and shape drawing color (HEX).\n"
                        "---\n"
                        "Цвет кисти и фигур (HEX)."
                    ),
                }),
                "brush_size": ("INT", {
                    "default": 10, "min": 1, "max": 200, "step": 1,
                    "tooltip": (
                        "Brush stroke width in pixels.\n"
                        "---\n"
                        "Толщина линии кисти в пикселях."
                    ),
                }),
                "brush_shape": (["circle", "square", "spray", "soft", "flat"], {
                    "default": "circle",
                    "tooltip": (
                        "Brush tip: round, square, spray dots, soft airbrush, "
                        "flat chisel at 45 degrees.\n"
                        "---\n"
                        "Кончик кисти: круг, квадрат, спрей, мягкий аэрограф, "
                        "плоское перо под 45 градусов."
                    ),
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stroke and shape opacity 0.05..1.0.\n"
                        "---\n"
                        "Прозрачность штрихов и фигур 0.05..1.0."
                    ),
                }),
                "fill_shapes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Fill closed shapes instead of outline only.\n"
                        "---\n"
                        "Заливать замкнутые фигуры вместо контура."
                    ),
                }),
                "polygon_sides": ("INT", {
                    "default": 6, "min": 3, "max": 12, "step": 1,
                    "tooltip": (
                        "Number of sides for the polygon shape.\n"
                        "---\n"
                        "Число сторон для фигуры polygon."
                    ),
                }),
                "text_value": ("STRING", {
                    "default": "AGSoft",
                    "tooltip": (
                        "Text placed by the text shape; drag height sets font size.\n"
                        "---\n"
                        "Текст для фигуры text; высота протяжки задаёт кегль."
                    ),
                }),
                "upload_mode": (["stretch", "fit", "crop_center"], {
                    "default": "stretch",
                    "tooltip": (
                        "How an uploaded image is placed: stretch to full "
                        "canvas, fit inside with bars, or center crop.\n"
                        "---\n"
                        "Как размещается загруженная картинка: растянуть на "
                        "весь холст, вписать с полями или кроп по центру."
                    ),
                }),
                "drawing_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "Hidden key of the PNG file in the temp folder.\n"
                        "---\n"
                        "Скрытый ключ PNG-файла в temp-папке."
                    ),
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_TOOLTIPS = (
        "Drawn image tensor over canvas color.\n---\nТензор рисунка поверх цвета холста.",
        "Mask of drawn strokes (alpha channel).\n---\nМаска нарисованных штрихов (альфа).",
        "Canvas width in pixels.\n---\nШирина холста в пикселях.",
        "Canvas height in pixels.\n---\nВысота холста в пикселях."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        key = kwargs.get("drawing_key", "")
        path = _paint_path(key) if key else None
        if path and os.path.isfile(path):
            return os.path.getmtime(path)
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        w = kwargs.get("width", 1024)
        h = kwargs.get("height", 1024)
        if w <= 0 or h <= 0:
            return "Width and height must be > 0"
        return True

    def process_canvas(self, width, height, canvas_color, brush_color, brush_size,
                       brush_shape, opacity, fill_shapes, polygon_sides,
                       text_value, upload_mode, drawing_key):
        rgba = None
        path = _paint_path(drawing_key) if drawing_key else None
        if path and os.path.isfile(path):
            try:
                img = Image.open(path).convert("RGBA")
                if img.size != (width, height):
                    try:
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        img = img.resize((width, height))
                rgba = np.array(img).astype(np.float32) / 255.0
            except Exception as e:
                logger.warning("paint key load failed: %s", e)
        if rgba is None:
            rgba = np.zeros((height, width, 4), dtype=np.float32)
        alpha = rgba[..., 3:4]
        cr, cg, cb = _hex_to_rgb(canvas_color)
        bg = np.empty((height, width, 3), dtype=np.float32)
        bg[..., 0] = cr
        bg[..., 1] = cg
        bg[..., 2] = cb
        image = bg * (1.0 - alpha) + rgba[..., :3] * alpha
        mask = rgba[..., 3].copy()
        t_img = torch.from_numpy(image).unsqueeze(0)
        t_mask = torch.from_numpy(mask).unsqueeze(0)
        return (t_img, t_mask, width, height)


NODE_CLASS_MAPPINGS = {
    "AGSoftPaintPlus": AGSoftPaintPlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftPaintPlus": "🎨AGSoft Paint Plus"
}