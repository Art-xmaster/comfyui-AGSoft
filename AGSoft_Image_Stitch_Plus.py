"""
==============================================================================
AGSoft_Image_Stitch_Plus.py
==============================================================================
Нода: 🖼️AGSoft Image Stitch Plus
Описание / Description:
Нода сшивания изображений в сетки и ленты для видео-генераций без
обязательных входов IMAGE. Изображения попадают в ноду двумя способами:
- из внутренней панели превью (drag&drop или выбор файла, до 50 шт.): файлы
  загружаются в input/agsoft_stitch/, отображаются мини-превью, упорядоченный
  список хранится в скрытом виджете image_list_json и восстанавливается после
  перезагрузки workflow;
- через внешний вход input_images (тензор, батч или список из AGSoft Add
  Images) — добавляется после изображений панели.
Режимы сшивания: линейные right/down/left/up, 2x2, context_mode (колонка из
всех кроме последнего слева, последний справа), ленты row/column, сетки
grid_auto (ближайший квадрат) и grid_custom (grid_cols x grid_rows) с
порядком заполнения fill_order и унифицированными ячейками: cell_fit
(contain/cover/stretch), cell_aspect, cell_width/cell_height.
Промежутки spacing_x/spacing_y, внешняя рамка outer_padding, цвет фона
(пресет или custom_background_color), финальный ресайз (megapixels/
max_width/max_height + upscale_method) и приведение к кратности multiple_of
с центрированием содержимого на холсте.
Нумерация выхода: виджет number_position (off + 6 позиций) рисует компактные
бейджи с порядковыми номерами (1..N) в стиле панели — полупрозрачная
скруглённая плашка, рамка и цифры #7fd4ff, антиалиасинг 4x; только на выходе,
превью панели не затрагиваются.
Выходы: IMAGE (результат), WIDTH, HEIGHT, ORDER_JSON (JSON-список
использованных источников в порядке сшивания).
---
Image stitcher node for video-generation grids and strips without mandatory
IMAGE inputs. Images enter the node in two ways:
- from the internal preview panel (drag&drop or file picker, up to 50 items):
  files upload to input/agsoft_stitch/, show as mini previews, the ordered
  list lives in the hidden widget image_list_json and restores after workflow
  reload;
- from the external input input_images (tensor, batch or list from AGSoft Add
  Images) — appended after panel images.
Stitch modes: linear right/down/left/up, 2x2, context_mode (column of all but
last on the left, last on the right), strips row/column, grids grid_auto
(nearest square) and grid_custom (grid_cols x grid_rows) with fill_order and
uniform cells: cell_fit (contain/cover/stretch), cell_aspect,
cell_width/cell_height.
Spacing spacing_x/spacing_y, outer frame outer_padding, background color
(preset or custom_background_color), final resize (megapixels/max_width/
max_height + upscale_method) and multiple_of rounding with content centering
on the canvas.
Output numbering: number_position widget (off + 6 positions) draws compact
panel-style badges with sequence numbers (1..N) — translucent rounded plate,
#7fd4ff border and digits, 4x anti-aliasing; output only, panel previews
untouched.
Outputs: IMAGE (result), WIDTH, HEIGHT, ORDER_JSON (JSON list of used sources
in stitch order).
Возможности / Features:
⚡ Панель превью: drag&drop, загрузка, reorder (drag + кнопки ◀▶), удаление,
   замена двойным кликом, enable/disable, счётчик N/50, прогресс-бар, тосты.
   Preview panel: drag&drop, upload, reorder (drag + ◀▶ buttons), delete,
   dblclick replace, enable/disable, N/50 counter, progress bar, toasts.
⚡ Нумерация выхода одним виджетом number_position: off + 6 позиций,
   компактные бейджи в стиле панели (#7fd4ff), антиалиасинг 4x.
   Output numbering via single number_position widget: off + 6 positions,
   compact panel-style badges (#7fd4ff), 4x anti-aliasing.
⚡ Внешний вход input_images (тензор/батч/список из AGSoft Add Images) после
   изображений панели; батч разбивается на кадры.
   External input_images (tensor/batch/list from AGSoft Add Images) appended
   after panel images; batches split into frames.
⚡ Сетки NxM / авто-сетка / ленты, порядок заполнения, унифицированные ячейки
   (contain/cover/stretch), пропорции и размер ячейки.
   NxM grids / auto grid / strips, fill order, uniform cells
   (contain/cover/stretch), cell aspect and size.
⚡ spacing_x/spacing_y, outer_padding, multiple_of с центрированием,
   megapixels/max_width/max_height, upscale_method.
   spacing_x/spacing_y, outer_padding, centered multiple_of,
   megapixels/max_width/max_height, upscale_method.
⚡ VALIDATE_INPUTS не блокирует ноду при пустой панели, если подключён
   input_images; IS_CHANGED по mtime файлов панели.
   VALIDATE_INPUTS does not block the node on empty panel when input_images
   is connected; IS_CHANGED by panel files mtime.
Автор / Author: AGSoft
Дата / Date: 25.09.2026
==============================================================================
"""
import os
import json
import math
import logging
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps
from comfy.utils import common_upscale

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("[AGSoft Image Stitch Plus] v09.25 loaded (preview panel grid stitcher, panel-style numbering badges, external input)")

MAX_IMAGES = 50
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
BADGE_CYAN = (0.4980392156862745, 0.8313725490196078, 1.0)  # #7fd4ff

BACKGROUND_COLOR_PRESETS = {
    "black": "#000000", "white": "#FFFFFF", "gray": "#808080", "silver": "#C0C0C0",
    "light_gray": "#D3D3D3", "dark_gray": "#A9A9A9", "red": "#FF0000", "green": "#00FF00",
    "blue": "#0000FF", "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "orange": "#FFA500", "pink": "#FFC0CB", "brown": "#A52A2A", "purple": "#800080",
    "violet": "#EE82EE", "indigo": "#4B0082", "teal": "#008080", "navy": "#000080",
    "olive": "#808000", "maroon": "#800000", "dark_blue": "#00008B", "light_blue": "#ADD8E6",
    "light_green": "#90EE90", "dark_green": "#006400",
}

DIGIT_FONT_5X7 = {
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
}

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

class AGSoft_Image_Stitch_Plus:
    CATEGORY = "AGSoft/Image"
    FUNCTION = "stitch"
    OUTPUT_NODE = False
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT", "ORDER_JSON")
    OUTPUT_TOOLTIPS = (
        "Stitched result image.\n---\nИтоговое сшитое изображение.",
        "Final width in pixels.\n---\nИтоговая ширина в пикселях.",
        "Final height in pixels.\n---\nИтоговая высота в пикселях.",
        "JSON list of used sources in stitch order.\n---\nJSON-список использованных источников в порядке сшивания.",
    )
    DESCRIPTION = (
        "Stitch up to 50 panel images plus external inputs into strips or grids for video generation, with compact output numbering.\n"
        "---\nСшивание до 50 изображений панели плюс внешние входы в ленты или сетки для видео-генераций, с компактной нумерацией выхода."
    )

    @classmethod
    def INPUT_TYPES(cls):
        color_options = [f"{name} ({hex_code})" for name, hex_code in BACKGROUND_COLOR_PRESETS.items()]
        return {
            "required": {
                "stitch_mode": (["right", "down", "left", "up", "2x2", "context_mode", "row", "column", "grid_auto", "grid_custom"], {
                    "default": "grid_auto",
                    "tooltip": (
                        "Stitch mode:\n"
                        "• right/down/left/up — linear chain from first image\n"
                        "• 2x2 — grid 2x2 (cell logic)\n"
                        "• context_mode — column of all but last on left, last on right\n"
                        "• row — strip 1xN, column — strip Nx1\n"
                        "• grid_auto — nearest square grid\n"
                        "• grid_custom — fixed grid_cols x grid_rows\n"
                        "---\n"
                        "Режим сшивания:\n"
                        "• right/down/left/up — линейная цепочка от первого изображения\n"
                        "• 2x2 — сетка 2x2 (логика ячеек)\n"
                        "• context_mode — колонка из всех кроме последнего слева, последний справа\n"
                        "• row — лента 1xN, column — лента Nx1\n"
                        "• grid_auto — ближайшая квадратная сетка\n"
                        "• grid_custom — фиксированные grid_cols x grid_rows"
                    )
                }),
                "number_position": (["off", "top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"], {
                    "default": "off",
                    "tooltip": (
                        "Single widget for output numbering: off or corner/center position of small panel-style badges with sequence numbers (1..N, stitch order). Output only, panel previews untouched.\n"
                        "---\nЕдиный виджет нумерации выхода: off или позиция в углу/центре маленьких бейджей в стиле панели с порядковыми номерами (1..N, порядок сшивания). Только выход, превью панели не затрагиваются."
                    )
                }),
                "grid_cols": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Columns for grid_custom (rows extend if images overflow).\n---\nКолонки для grid_custom (строки добавляются при переполнении)."}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1,
                    "tooltip": "Rows for grid_custom (columns extend if fill_order=column_first overflows).\n---\nСтроки для grid_custom (колонки добавляются при переполнении fill_order=column_first)."}),
                "fill_order": (["row_first", "column_first"], {
                    "default": "row_first",
                    "tooltip": "Grid fill order: by rows or by columns.\n---\nПорядок заполнения сетки: по строкам или по столбцам."
                }),
                "cell_fit": (["contain", "cover", "stretch"], {
                    "default": "contain",
                    "tooltip": "How image fits uniform cell: contain (pad with bg), cover (center crop), stretch (distort).\n---\nКак изображение вписывается в ячейку: contain (поля фоном), cover (центральный кроп), stretch (искажение)."
                }),
                "cell_aspect": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9"], {
                    "default": "auto",
                    "tooltip": "Cell aspect ratio: auto = aspect of first image.\n---\nПропорции ячейки: auto = пропорции первого изображения."
                }),
                "cell_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Cell width in px (0 = auto from first image).\n---\nШирина ячейки в px (0 = авто по первому изображению)."}),
                "cell_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Cell height in px (0 = auto from first image).\n---\nВысота ячейки в px (0 = авто по первому изображению)."}),
                "match_image_size": ("BOOLEAN", {"default": True,
                    "tooltip": "Linear/context modes only: scale next images proportionally to first.\n---\nТолько линейные/context режимы: пропорциональное масштабирование последующих к первому."}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01,
                    "tooltip": "Target final size in megapixels (0 = no limit).\n---\nЦелевой размер итога в мегапикселях (0 = без лимита)."}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Max final width (0 = no limit). Ignored if megapixels > 0.\n---\nМакс. ширина итога (0 = без лимита). Игнорируется при megapixels > 0."}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Max final height (0 = no limit). Ignored if megapixels > 0.\n---\nМакс. высота итога (0 = без лимита). Игнорируется при megapixels > 0."}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for scaling. Lanczos = best quality.\n---\nМетод интерполяции при масштабировании. Lanczos = лучшее качество."}),
                "spacing_x": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Horizontal gap between cells/images in px (bg color).\n---\nГоризонтальный промежуток между ячейками/изображениями в px (цвет фона)."}),
                "spacing_y": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Vertical gap between cells/images in px (bg color).\n---\nВертикальный промежуток между ячейками/изображениями в px (цвет фона)."}),
                "outer_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Outer frame around whole result in px (bg color).\n---\nВнешняя рамка вокруг всего результата в px (цвет фона)."}),
                "background_color_preset": (color_options, {
                    "default": "white (#FFFFFF)",
                    "tooltip": "Preset background color. Used if custom_background_color is empty.\n---\nПредустановленный цвет фона. Используется, если custom_background_color пуст."}),
                "multiple_of": ([0, 1, 2, 4, 8, 16, 32, 64, 112, 128], {
                    "default": 0,
                    "tooltip": (
                        "Round final sizes up to a multiple and center content on the padded canvas.\n"
                        "• 0 = off, 8 = SD standard, 64 = some VAE/video models, 112/128 = special architectures\n"
                        "---\n"
                        "Округлить итоговые размеры вверх до кратности и центрировать содержимое на холсте.\n"
                        "• 0 = выкл, 8 = стандарт SD, 64 = некоторые VAE/видео-модели, 112/128 = особые архитектуры"
                    )
                }),
            },
            "optional": {
                "input_images": ("IMAGE", {
                    "tooltip": (
                        "External images: single tensor, batch or list from AGSoft Add Images. Appended after panel images.\n"
                        "---\nВнешние изображения: тензор, батч или список из AGSoft Add Images. Добавляются после изображений панели."
                    )
                }),
                "custom_background_color": ("STRING", {"default": "", "placeholder": "#RRGGBB",
                    "tooltip": "Custom background color #RRGGBB. Overrides preset.\n---\nПроизвольный цвет фона #RRGGBB. Переопределяет пресет."}),
                "image_list_json": ("STRING", {"default": "[]",
                    "tooltip": "Service field: JSON list of panel images (managed by UI).\n---\nСлужебное поле: JSON-список изображений панели (управляется UI)."}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # NOTE: at validation stage linked inputs are not present in kwargs, so we never
        # reject on empty panel here; runtime stitch() raises if there are no sources.
        # ВАЖНО: на стадии валидации подключённые линки не приходят в kwargs, поэтому здесь
        # мы никогда не отклоняем ноду из-за пустой панели; пустоту источников ловит stitch().
        bg_color = kwargs.get("custom_background_color", "").strip()
        if not bg_color:
            bg_color = kwargs.get("background_color_preset", "white (#FFFFFF)").split(" (")[-1].rstrip(")")
        if not cls._is_valid_hex_color(bg_color):
            return f"Invalid background color format: {bg_color}. Use #RRGGBB. / Неверный формат цвета фона: {bg_color}. Используйте #RRGGBB."
        items = cls._parse_list(kwargs.get("image_list_json", "[]"))
        if len(items) > MAX_IMAGES:
            return f"Too many images: {len(items)} > {MAX_IMAGES}. / Слишком много изображений: {len(items)} > {MAX_IMAGES}."
        for it in items:
            if not it.get("enabled", True):
                continue
            p, err = cls._resolve_path(it)
            if err:
                return err
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        items = cls._parse_list(kwargs.get("image_list_json", "[]"))
        marks = []
        for it in items:
            if not it.get("enabled", True):
                continue
            p, err = cls._resolve_path(it)
            if err or not os.path.exists(p):
                return float("nan")
            marks.append(os.path.getmtime(p))
        return tuple(marks) + (kwargs.get("image_list_json", "[]"),)

    @staticmethod
    def _is_valid_hex_color(color_str):
        if not isinstance(color_str, str) or len(color_str) != 7 or color_str[0] != '#':
            return False
        try:
            int(color_str[1:], 16)
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_list(raw):
        try:
            data = json.loads(raw) if raw else []
            if isinstance(data, list):
                return [it for it in data if isinstance(it, dict) and it.get("name")]
        except Exception:
            pass
        return []

    @staticmethod
    def _resolve_path(item):
        base = os.path.abspath(folder_paths.get_input_directory())
        sub = str(item.get("subfolder", "") or "").replace("\\", "/").strip("/")
        name = os.path.basename(str(item.get("name", "")))
        if os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
            return None, f"Unsupported file type: {name}. / Неподдерживаемый тип файла: {name}."
        full = os.path.abspath(os.path.join(base, sub, name)) if sub else os.path.abspath(os.path.join(base, name))
        if not (full == base or full.startswith(base + os.sep)):
            return None, f"Path outside input folder: {name}. / Путь вне папки input: {name}."
        if not os.path.exists(full):
            return None, f"File not found: {full}. / Файл не найден: {full}."
        return full, None

    def _load_image(self, path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[0:2], 16) / 255.0, int(hex_color[2:4], 16) / 255.0, int(hex_color[4:6], 16) / 255.0)

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
        result[:, pad_top:pad_top + height, pad_left:pad_left + width, :] = image
        return result

    def _upscale(self, image, target_w, target_h, upscale_method):
        return common_upscale(image.movedim(-1, 1), target_w, target_h, upscale_method, "disabled").movedim(1, -1)

    def _blank(self, batch, height, width, channels, color_val, device):
        result = torch.zeros((batch, height, width, channels), device=device)
        r, g, b = color_val
        if channels >= 3:
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            if channels == 4:
                result[..., 3] = 1.0
        return result

    @staticmethod
    def _rr_mask(H, W, inset, radius):
        # Rounded-rect boolean mask at supersampled resolution / Булева маска скруглённого прямоугольника в суперсэмплированном разрешении
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        x0, x1 = inset, W - 1 - inset
        y0, y1 = inset, H - 1 - inset
        qx = torch.clamp(x0 + radius - xx, min=0.0) + torch.clamp(xx - (x1 - radius), min=0.0)
        qy = torch.clamp(y0 + radius - yy, min=0.0) + torch.clamp(yy - (y1 - radius), min=0.0)
        rect = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        return rect & (torch.sqrt(qx * qx + qy * qy) <= radius)

    def _draw_number(self, image, number, position):
        # Panel-style badge: translucent black rounded plate, #7fd4ff border and digits, 4x anti-aliasing
        # Бейдж в стиле панели: полупрозрачная чёрная скруглённая плашка, рамка и цифры #7fd4ff, антиалиасинг 4x
        b, h, w, c = image.shape
        s_str = str(int(number))
        glyph_units = 6 * len(s_str) - 1
        s = max(2, min(3, min(h, w) // 64))
        t = max(1, s // 2)
        pad = s
        m = 2 * s
        box_h, box_w = 7 * s + 2 * (t + pad), glyph_units * s + 2 * (t + pad)
        if box_h + 2 * m > h or box_w + 2 * m > w:
            return image
        ss = 4
        H, W = box_h * ss, box_w * ss
        t_ss, pad_ss = t * ss, pad * ss
        r_ss = float((t + pad) * ss)
        inner = self._rr_mask(H, W, float(t_ss), max(1.0, r_ss - t_ss))
        outer = self._rr_mask(H, W, 0.0, r_ss)
        border = outer & ~inner
        digits = torch.zeros(H, W, dtype=torch.bool)
        off = t_ss + pad_ss
        for di, ch in enumerate(s_str):
            rows = DIGIT_FONT_5X7[ch]
            for r in range(7):
                for bit in range(5):
                    if (rows[r] >> (4 - bit)) & 1:
                        y0 = off + r * s * ss
                        x0 = off + (di * 6 + bit) * s * ss
                        digits[y0:y0 + s * ss, x0:x0 + s * ss] = True
        def down(mask):
            return mask.float().view(H // ss, ss, W // ss, ss).mean(dim=(1, 3))
        a_bg = down(inner) * 0.65
        a_bd = down(border) * 0.9
        a_dg = down(digits)
        cyan = torch.tensor(BADGE_CYAN, dtype=torch.float32)
        y = m if position.startswith("top") else h - box_h - m
        tail = position.split("_")[-1]
        if tail == "left":
            x = m
        elif tail == "center":
            x = (w - box_w) // 2
        else:
            x = w - box_w - m
        img = image.clone()
        dev = image.device
        a_bg = a_bg[..., None].to(dev)
        a_bd = a_bd[..., None].to(dev)
        a_dg = a_dg[..., None].to(dev)
        cyan = cyan.to(dev)
        reg = img[:, y:y + box_h, x:x + box_w, :3]
        reg = reg * (1.0 - a_bg)
        reg = reg * (1.0 - a_bd) + cyan * a_bd
        reg = reg * (1.0 - a_dg) + cyan * a_dg
        img[:, y:y + box_h, x:x + box_w, :3] = reg
        if c == 4:
            img[:, y:y + box_h, x:x + box_w, 3] = 1.0
        return img

    def ensure_same_channels(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat([image1, torch.ones(*image1.shape[:-1], max_channels - image1.shape[-1], device=image1.device)], dim=-1)
            if image2.shape[-1] < max_channels:
                image2 = torch.cat([image2, torch.ones(*image2.shape[:-1], max_channels - image2.shape[-1], device=image2.device)], dim=-1)
        return image1, image2

    def _unify_list(self, images):
        base = images[0]
        unified = [base]
        for img in images[1:]:
            base, img = self.ensure_same_channels(base, img)
            unified.append(img)
        unified[0] = base
        max_batch = max(img.shape[0] for img in unified)
        fixed = []
        for img in unified:
            if img.shape[0] < max_batch:
                img = torch.cat([img, img[-1:].repeat(max_batch - img.shape[0], 1, 1, 1)])
            fixed.append(img)
        return fixed

    def match_dimensions(self, image1, image2, stitch_mode, color_val):
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        if stitch_mode in ["left", "right"]:
            if h1 != h2:
                target_h = max(h1, h2)
                if h1 < target_h:
                    pad_h = target_h - h1
                    image1 = self.pad_with_color(image1, (pad_h // 2, pad_h - pad_h // 2, 0, 0), color_val)
                if h2 < target_h:
                    pad_h = target_h - h2
                    image2 = self.pad_with_color(image2, (pad_h // 2, pad_h - pad_h // 2, 0, 0), color_val)
        else:
            if w1 != w2:
                target_w = max(w1, w2)
                if w1 < target_w:
                    pad_w = target_w - w1
                    image1 = self.pad_with_color(image1, (0, 0, pad_w // 2, pad_w - pad_w // 2), color_val)
                if w2 < target_w:
                    pad_w = target_w - w2
                    image2 = self.pad_with_color(image2, (0, 0, pad_w // 2, pad_w - pad_w // 2), color_val)
        return image1, image2

    def create_spacing(self, image1, image2, spacing_width, stitch_mode, color_val):
        if spacing_width <= 0:
            return None
        if stitch_mode in ["left", "right"]:
            spacing_shape = (image1.shape[0], max(image1.shape[1], image2.shape[1]), spacing_width, image1.shape[-1])
        else:
            spacing_shape = (image1.shape[0], spacing_width, max(image1.shape[2], image2.shape[2]), image1.shape[-1])
        return self._blank(*spacing_shape, color_val, image1.device)

    def stitch_two_images(self, image1, image2, stitch_mode, match_image_size, spacing_width, color_val, upscale_method, number2=0, number_position="off"):
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
                target_h, target_w = h1, max(1, int(h1 * aspect_ratio))
            else:
                target_w, target_h = w1, max(1, int(w1 / aspect_ratio))
            image2 = self._upscale(image2, target_w, target_h, upscale_method)
        else:
            image1, image2 = self.match_dimensions(image1, image2, stitch_mode, color_val)
        image1, image2 = self.ensure_same_channels(image1, image2)
        if number2:
            image2 = self._draw_number(image2, number2, number_position)
        spacing = self.create_spacing(image1, image2, spacing_width, stitch_mode, color_val)
        images = [image2, image1] if stitch_mode in ["left", "up"] else [image1, image2]
        if spacing is not None:
            images.insert(1, spacing)
        concat_dim = 2 if stitch_mode in ["left", "right"] else 1
        return torch.cat(images, dim=concat_dim)

    def stitch_multi_mode(self, images, stitch_mode, match_image_size, spacing_x, spacing_y, color_val, upscale_method, number_position):
        spacing = spacing_x if stitch_mode in ["left", "right"] else spacing_y
        num_on = number_position != "off"
        imgs = list(images)
        if num_on:
            imgs[0] = self._draw_number(imgs[0], 1, number_position)
        current = imgs[0]
        for pos, next_img in enumerate(imgs[1:], start=2):
            current = self.stitch_two_images(current, next_img, stitch_mode, match_image_size, spacing, color_val, upscale_method, number2=pos if num_on else 0, number_position=number_position)
        return current

    def stitch_context_mode(self, images, match_image_size, spacing_x, spacing_y, color_val, upscale_method, number_position):
        # Column of all-but-last on left, last image on right / Колонка из всех кроме последнего слева, последний справа
        num_on = number_position != "off"
        if len(images) == 1:
            return self._draw_number(images[0], 1, number_position) if num_on else images[0]
        if len(images) == 2:
            first = self._draw_number(images[0], 1, number_position) if num_on else images[0]
            return self.stitch_two_images(first, images[1], "right", match_image_size, spacing_x, color_val, upscale_method, number2=2 if num_on else 0, number_position=number_position)
        left_images = list(images[:-1])
        right_image = images[-1]
        left_images = self._unify_list(left_images)
        if match_image_size and len(left_images) > 1:
            w1 = left_images[0].shape[2]
            for i in range(1, len(left_images)):
                h, w = left_images[i].shape[1:3]
                target_w = w1
                target_h = max(1, int(w1 * (h / w)))
                left_images[i] = self._upscale(left_images[i], target_w, target_h, upscale_method)
        elif not match_image_size and len(left_images) > 1:
            for i in range(1, len(left_images)):
                left_images[0], left_images[i] = self.match_dimensions(left_images[0], left_images[i], "down", color_val)
        for i in range(1, len(left_images)):
            left_images[0], left_images[i] = self.ensure_same_channels(left_images[0], left_images[i])
        if num_on:
            left_images = [self._draw_number(img, i + 1, number_position) for i, img in enumerate(left_images)]
            right_image = self._draw_number(right_image, len(images), number_position)
        left_column = left_images[0]
        for i in range(1, len(left_images)):
            left_column, left_images[i] = self.ensure_same_channels(left_column, left_images[i])
            parts = [left_column]
            if spacing_y > 0:
                parts.append(self._blank(left_column.shape[0], spacing_y, max(left_column.shape[2], left_images[i].shape[2]), left_column.shape[-1], color_val, left_column.device))
            parts.append(left_images[i])
            left_column = torch.cat(parts, dim=1)
        if match_image_size:
            h_left = left_column.shape[1]
            hr, wr = right_image.shape[1:3]
            right_image = self._upscale(right_image, max(1, int(h_left * (wr / hr))), h_left, upscale_method)
        else:
            left_column, right_image = self.match_dimensions(left_column, right_image, "right", color_val)
        left_column, right_image = self.ensure_same_channels(left_column, right_image)
        parts = [left_column]
        if spacing_x > 0:
            parts.append(self._blank(left_column.shape[0], max(left_column.shape[1], right_image.shape[1]), spacing_x, left_column.shape[-1], color_val, left_column.device))
        parts.append(right_image)
        return torch.cat(parts, dim=2)

    def fit_cell(self, image, cell_w, cell_h, cell_fit, upscale_method, color_val):
        h, w = image.shape[1:3]
        if cell_fit == "stretch":
            return self._upscale(image, cell_w, cell_h, upscale_method)
        scale = min(cell_w / w, cell_h / h) if cell_fit == "contain" else max(cell_w / w, cell_h / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        image = self._upscale(image, nw, nh, upscale_method)
        if cell_fit == "contain":
            pad_w, pad_h = cell_w - nw, cell_h - nh
            return self.pad_with_color(image, (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2), color_val)
        off_y, off_x = (nh - cell_h) // 2, (nw - cell_w) // 2
        return image[:, off_y:off_y + cell_h, off_x:off_x + cell_w, :]

    def stitch_grid(self, images, cols, rows, fill_order, cell_fit, cell_w, cell_h, spacing_x, spacing_y, outer_padding, color_val, upscale_method, number_position):
        num_on = number_position != "off"
        unified = self._unify_list(images)
        channels = unified[0].shape[-1]
        batch = unified[0].shape[0]
        device = unified[0].device
        total = cols * rows
        cells = [None] * total
        for i, img in enumerate(unified):
            if i >= total:
                break
            if fill_order == "row_first":
                r, c = i // cols, i % cols
            else:
                r, c = i % rows, i // rows
            fitted = self.fit_cell(img, cell_w, cell_h, cell_fit, upscale_method, color_val)
            if num_on:
                fitted = self._draw_number(fitted, i + 1, number_position)
            cells[r * cols + c] = fitted
        for i in range(total):
            if cells[i] is None:
                cells[i] = self._blank(batch, cell_h, cell_w, channels, color_val, device)
        row_tensors = []
        for r in range(rows):
            parts = []
            for c in range(cols):
                if c > 0 and spacing_x > 0:
                    parts.append(self._blank(batch, cell_h, spacing_x, channels, color_val, device))
                parts.append(cells[r * cols + c])
            row_tensors.append(torch.cat(parts, dim=2))
        stacked = []
        for r, row_t in enumerate(row_tensors):
            if r > 0 and spacing_y > 0:
                stacked.append(self._blank(batch, spacing_y, row_t.shape[2], channels, color_val, device))
            stacked.append(row_t)
        result = torch.cat(stacked, dim=1)
        if outer_padding > 0:
            result = self.pad_with_color(result, (outer_padding, outer_padding, outer_padding, outer_padding), color_val)
        return result

    def _cell_size(self, first_image, cell_aspect, cell_width, cell_height):
        h1, w1 = first_image.shape[1:3]
        base_w, base_h = float(w1), float(h1)
        if cell_aspect != "auto":
            aw, ah = [float(x) for x in cell_aspect.split(":")]
            ratio = aw / ah
            if base_w / base_h > ratio:
                base_w = base_h * ratio
            else:
                base_h = base_w / ratio
        cw = cell_width if cell_width > 0 else int(base_w)
        ch = cell_height if cell_height > 0 else int(base_h)
        if cell_width > 0 and cell_height == 0:
            ch = max(1, int(cw * (base_h / base_w)))
        if cell_height > 0 and cell_width == 0:
            cw = max(1, int(ch * (base_w / base_h)))
        return max(8, cw), max(8, ch)

    def stitch(self, stitch_mode, number_position, grid_cols, grid_rows, fill_order, cell_fit, cell_aspect, cell_width, cell_height,
               match_image_size, megapixels, max_width, max_height, upscale_method, spacing_x, spacing_y,
               outer_padding, background_color_preset, multiple_of, input_images=None, custom_background_color="", image_list_json="[]"):
        items = self._parse_list(image_list_json)
        enabled = [it for it in items if it.get("enabled", True)]
        images, used_names = [], []
        for it in enabled:
            path, err = self._resolve_path(it)
            if err:
                raise RuntimeError(err)
            images.append(self._load_image(path))
            used_names.append(os.path.join(str(it.get("subfolder", "") or ""), it["name"]).replace("\\", "/"))
        if input_images is not None:
            ext = split_frames(input_images)
            for k, t in enumerate(ext, start=1):
                images.append(t)
                used_names.append("external_" + str(k))
        if not images:
            raise RuntimeError("No image sources: panel is empty and input_images not connected. / Нет источников изображений: панель пуста и input_images не подключён.")
        if custom_background_color and custom_background_color.strip():
            bg_hex = custom_background_color.strip()
        else:
            bg_hex = background_color_preset.split(" (")[-1].rstrip(")")
        color_val = self.hex_to_rgb(bg_hex)
        n = len(images)
        if stitch_mode in ("right", "down", "left", "up"):
            result = self.stitch_multi_mode(images, stitch_mode, match_image_size, spacing_x, spacing_y, color_val, upscale_method, number_position)
        elif stitch_mode == "context_mode":
            result = self.stitch_context_mode(images, match_image_size, spacing_x, spacing_y, color_val, upscale_method, number_position)
        else:
            if stitch_mode == "2x2":
                cols, rows = 2, 2
            elif stitch_mode == "row":
                cols, rows = n, 1
            elif stitch_mode == "column":
                cols, rows = 1, n
            elif stitch_mode == "grid_auto":
                cols = max(1, math.ceil(math.sqrt(n)))
                rows = max(1, math.ceil(n / cols))
            else:  # grid_custom
                cols, rows = max(1, grid_cols), max(1, grid_rows)
                if fill_order == "row_first":
                    rows = max(rows, math.ceil(n / cols))
                else:
                    cols = max(cols, math.ceil(n / rows))
            cell_w, cell_h = self._cell_size(images[0], cell_aspect, cell_width, cell_height)
            result = self.stitch_grid(images, cols, rows, fill_order, cell_fit, cell_w, cell_h,
                                      spacing_x, spacing_y, outer_padding, color_val, upscale_method, number_position)
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
                target_w, target_h = max_width, int(h * scale_factor)
                need_resize = True
            else:
                target_w, target_h = w, h
            if max_height > 0 and target_h > max_height:
                scale_factor = max_height / target_h
                target_h, target_w = max_height, int(target_w * scale_factor)
                need_resize = True
        if need_resize:
            result = self._upscale(result, max(1, target_w), max(1, target_h), upscale_method)
        if multiple_of > 0:
            current_h, current_w = result.shape[1:3]
            new_w = ((current_w + multiple_of - 1) // multiple_of) * multiple_of
            new_h = ((current_h + multiple_of - 1) // multiple_of) * multiple_of
            if new_w != current_w or new_h != current_h:
                pad_w_total, pad_h_total = new_w - current_w, new_h - current_h
                result = self.pad_with_color(result, (pad_h_total // 2, pad_h_total - pad_h_total // 2,
                                                      pad_w_total // 2, pad_w_total - pad_w_total // 2), color_val)
        if outer_padding > 0 and stitch_mode not in ("2x2", "row", "column", "grid_auto", "grid_custom"):
            result = self.pad_with_color(result, (outer_padding, outer_padding, outer_padding, outer_padding), color_val)
        final_height, final_width = result.shape[1:3]
        return (result, final_width, final_height, json.dumps(used_names, ensure_ascii=False))

NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Stitch_Plus": AGSoft_Image_Stitch_Plus
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Stitch_Plus": "🖼️AGSoft Image Stitch Plus"
}