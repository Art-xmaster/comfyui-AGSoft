# ==============================================================================
# AGSoft_Image_Crop_Plus.py
# ==============================================================================
# Ноды: 🖼️✂️AGSoft Image Crop Plus  +  🖼️🧵AGSoft Crop Stitch
# Описание / Description:
# Crop Plus — интерактивная обрезка изображения с живой рамкой прямо в ноде.
# Приоритет источника: input_image > custom_path > комбо image_name; превью
# следует за реально использованным источником. Маска обрезается синхронно с
# изображением, размеры выравниваются по кратности (multiple).
# Режимы рамки: Preset Ratio (по умолчанию) / Manual Size.
# ПАУЗА: при подключённом тензоре нода ставит выполнение на паузу — настройте
# кроп и нажмите "▶️ Продолжить" (живые координаты уходят на сервер, ✕ отменяет
# job, авто-продолжение через 300 с). Выход crop_data — ПЕРВЫЙ (оригинал +
# точный бокс обрезки).
# Crop Stitch — вставляет обработанный кадр обратно в оригинал по crop_data
# (авто-ресайз к размеру бокса; режимы paste / feather_blend / poisson_blend).
#
# Crop Plus — interactive image cropping with a live frame inside the node.
# Source priority: input_image > custom_path > image_name combo; the preview
# follows the actually used source. The mask is cropped synchronously, sizes
# are aligned to a multiple. Frame modes: Preset Ratio (default) / Manual Size.
# PAUSE: with a tensor connected the node pauses execution — adjust the crop and
# press "▶️ Resume" (live coords go to the server, ✕ cancels the job, auto-resume
# after 300 s). The crop_data output is FIRST (original + exact crop box).
# Crop Stitch — stitches the processed frame back into the original via
# crop_data (auto-resize to the box size; paste / feather_blend / poisson_blend).
#
# Возможности / Features:
# ⚡ Источники: input_image (приоритет) > custom_path > комбо image_name.
#   Sources: input_image (priority) > custom_path > image_name combo.
# ⚡ Пауза при тензоре: настрой кроп и нажми ▶️ (✕ отменяет, таймаут 300 с).
#   Pause on tensor: adjust the crop and press ▶️ (✕ cancels, 300 s timeout).
# ⚡ Живые координаты кропа во время паузы (обходят "впечённый" виджет).
#   Live crop coords during pause (bypass the baked-in widget).
# ⚡ crop_data — первый выход: оригинал + точный бокс для Stitch.
#   crop_data is the FIRST output: original + exact box for Stitch.
# ⚡ Маска обрезается синхронно с изображением. / Mask cropped in sync.
# ⚡ Выравнивание выхода по кратности (8/32/64). / Multiple alignment (8/32/64).
# ⚡ Рамка Preset Ratio держит пропорции жёстко; alignRect = математика Python
#   (рамка == crop_coords == реальный кроп, пиксель-в-пиксель).
#   Preset Ratio frame keeps proportions; alignRect mirrors Python math
#   (frame == crop_coords == real crop, pixel-perfect).
# ⚡ Stitch: paste / feather_blend (плавный шов) / poisson_blend (нужен opencv).
#   Stitch: paste / feather_blend (smooth seam) / poisson_blend (needs opencv).
# ⚡ Превью следует за источником (🖼️ image_name / 📂 custom_path / 🧠 input_image),
#   серверное состояние + поллинг; рамка пользователя сохраняется при смене источника.
#   Preview follows the source (🖼️ / 📂 / 🧠), server state + polling; the user's
#   frame is preserved across source changes.
#
# Автор / Author: AGSoft
# Дата / Date: 24.08.2026
# ==============================================================================

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import json
import os
import time
import hashlib
import shutil
import asyncio
import folder_paths
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import comfy.model_management as _mm  # для interrupt (кнопка ✕)
except Exception:
    _mm = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AGSoft_Image_Crop_Plus")

AGSOFT_CROP_PREFIX = "agsoft_crop_preview"
PAUSE_TIMEOUT = 300  # сек: авто-продолжение, если кнопку не нажали
_PIL_EXT_MAP = {
    "png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "gif": ".gif",
    "webp": ".webp", "bmp": ".bmp", "tiff": ".tiff", "tif": ".tiff",
    "ico": ".ico", "mpo": ".jpg",
}

def pil_to_tensor(pil_image):
    arr = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(tensor, is_mask=False):
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if is_mask:
        if array.ndim == 3:
            array = array[..., 0] if array.shape[-1] == 1 else np.dot(array[..., :3], [0.299, 0.587, 0.114])
        return Image.fromarray(array, mode='L')
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[-1] == 1:
        array = np.concatenate([array] * 3, axis=-1)
    elif array.shape[-1] == 4:
        return Image.fromarray(array, mode='RGBA')
    return Image.fromarray(array, mode='RGB')

def _resize_tensor(tensor, width, height, method="lanczos"):
    """Ресайз тензора [H,W,C] или [H,W] к (width,height)."""
    if tensor.ndim not in (2, 3):
        raise ValueError(f"Ожидался 2D/3D тензор, получен {tensor.ndim}D")
    ch, cw = tensor.shape[:2]
    if ch == height and cw == width:
        return tensor
    if method == "lanczos":
        np_img = (tensor.cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(np_img)
        rp = pil.resize((width, height), Image.LANCZOS)
        return torch.from_numpy(np.array(rp).astype(np.float32) / 255.0).to(tensor.device, dtype=tensor.dtype)
    mode = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic"}.get(method, "bilinear")
    if tensor.ndim == 2:
        t4 = tensor.unsqueeze(0).unsqueeze(0)
        r4 = F.interpolate(t4, size=(height, width), mode=mode, align_corners=False)
        return r4.squeeze(0).squeeze(0)
    t4 = tensor.permute(2, 0, 1).unsqueeze(0)
    r4 = F.interpolate(t4, size=(height, width), mode=mode, align_corners=False)
    return r4.squeeze(0).permute(1, 2, 0)

def _border_weight(h, w, feather):
    """Вес 0..1: 0 у границы, 1 в центре (плавный шов)."""
    if feather <= 0:
        return np.ones((h, w), dtype=np.float32)
    yy = np.arange(h)[:, None].astype(np.float32)
    xx = np.arange(w)[None, :].astype(np.float32)
    dist = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
    return np.clip(dist / feather, 0.0, 1.0)

# ==============================================================================
# ПРЕВЬЮ-ХЕЛПЕРЫ (копии в input для /view) + СОСТОЯНИЕ
# ==============================================================================
def _crop_real_ext(path):
    try:
        with Image.open(path) as im:
            return _PIL_EXT_MAP.get((im.format or "").lower(), os.path.splitext(path)[1].lower() or ".png")
    except Exception:
        return os.path.splitext(path)[1].lower() or ".png"

def _crop_ensure_preview_sync(custom_path):
    """Делает файл доступным через /view?type=input, возвращает имя в input."""
    custom_path = os.path.abspath(custom_path)
    if not os.path.isfile(custom_path):
        return None
    try:
        with Image.open(custom_path) as im:
            im.verify()
    except Exception:
        return None
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    os.makedirs(input_dir, exist_ok=True)
    base = os.path.basename(custom_path)
    if os.path.dirname(custom_path) == input_dir and not base.startswith(AGSOFT_CROP_PREFIX):
        return base
    st = os.stat(custom_path)
    ph = hashlib.sha256(custom_path.encode("utf-8", "ignore")).hexdigest()[:16]
    ext = _crop_real_ext(custom_path)
    name = f"{AGSOFT_CROP_PREFIX}{ph}_{int(st.st_mtime)}_{st.st_size}{ext}"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        tmp = dst + ".tmp"
        shutil.copy2(custom_path, tmp)
        os.replace(tmp, dst)
    prefix = f"{AGSOFT_CROP_PREFIX}{ph}_"
    try:
        for f in os.listdir(input_dir):
            if f.startswith(prefix) and f != name:
                try: os.remove(os.path.join(input_dir, f))
                except Exception: pass
    except Exception:
        pass
    return name

def _crop_tensor_preview_png(tensor, unique_id):
    """Первый кадр тензора → превью-PNG в input."""
    if tensor is None or not hasattr(tensor, "cpu"):
        return None
    t = tensor[0] if tensor.dim() == 4 else tensor
    pil = tensor_to_pil(t, is_mask=False)
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGB")
    input_dir = os.path.abspath(folder_paths.get_input_directory())
    os.makedirs(input_dir, exist_ok=True)
    uid = str(unique_id if unique_id is not None else "x")
    name = f"{AGSOFT_CROP_PREFIX}t_{uid}_{int(time.time() * 1000)}.png"
    pil.save(os.path.join(input_dir, name), "PNG")
    prefix = f"{AGSOFT_CROP_PREFIX}t_{uid}_"
    try:
        for f in os.listdir(input_dir):
            if f.startswith(prefix) and f != name:
                try: os.remove(os.path.join(input_dir, f))
                except Exception: pass
    except Exception:
        pass
    return name

_LAST_CROP = {}   # uid -> {image, custom, stamp, kind, waiting}
_CROP_LIVE = {}   # uid -> живые crop_coords из UI (во время паузы)
_CROP_GO = {}     # uid -> True, когда нажата кнопка "▶️ Продолжить"
_server = None
_web = None
try:
    from aiohttp import web as _web
    from server import PromptServer as _PS
    _server = getattr(_PS, "instance", None)
except Exception:
    _web = None
    _server = None

if _server is not None and _web is not None and not getattr(_server, "_agsoft_crop_route", False):
    try:
        @_server.routes.post("/agsoft/crop_ensure_preview")
        async def agsoft_crop_ensure_preview(request):
            try: data = await request.json()
            except Exception: data = {}
            p = str(data.get("custom_path", "")).strip()
            if not p:
                return _web.json_response({"error": "empty path"}, status=400)
            name = await asyncio.to_thread(_crop_ensure_preview_sync, p)
            if not name:
                return _web.json_response({"error": "not an image"}, status=400)
            return _web.json_response({"image": name})

        @_server.routes.get("/agsoft/crop_preview_state")
        async def agsoft_crop_preview_state(request):
            st = _LAST_CROP.get(str(request.query.get("node_id", "")))
            if not st:
                return _web.json_response({"image": "", "custom": False, "stamp": 0, "kind": "", "waiting": False})
            return _web.json_response(st)

        # Живые координаты кропа во время паузы (обходят "впечённый" в prompt виджет).
        @_server.routes.post("/agsoft/crop_live_coords")
        async def agsoft_crop_live_coords(request):
            try: data = await request.json()
            except Exception: data = {}
            _CROP_LIVE[str(data.get("node_id", ""))] = str(data.get("crop_coords", "[]"))
            return _web.json_response({"ok": True})

        # Сигнал "▶️ Продолжить" из UI.
        @_server.routes.post("/agsoft/crop_resume")
        async def agsoft_crop_resume(request):
            try: data = await request.json()
            except Exception: data = {}
            _CROP_GO[str(data.get("node_id", ""))] = True
            return _web.json_response({"ok": True})

        _server._agsoft_crop_route = True
    except Exception as e:
        print(f"[AGSoft Image Crop Plus] route registration failed: {e}")

# ==============================================================================
# НОДА: ✂️ AGSoft Image Crop Plus
# ==============================================================================
class AGSoft_Image_Crop_Plus:
    DESCRIPTION = """Interactive image cropping. Source priority: input_image > custom_path > image_name combo. The mask is cropped synchronously. Dimensions are aligned to the 'multiple'. Modes: Preset Ratio (default) / Manual Size. NEW: when a tensor is connected, execution pauses so you can adjust the crop, then press ▶️ Resume (✕ cancels the job).
Интерактивная обрезка. Приоритет источника: input_image > custom_path > комбо image_name. Маска обрезается синхронно. Размеры выравниваются по 'multiple'. Режимы: Preset Ratio (по умолчанию) / Manual Size. НОВОЕ: при подключённом тензоре выполнение ставится на паузу — настройте кроп и нажмите ▶️ Продолжить (✕ отменяет job)."""
    CATEGORY = "AGSoft/Image"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir)
                     if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith(AGSOFT_CROP_PREFIX)]
            files = folder_paths.filter_files_content_types(files, ["image"])
        except Exception:
            files = []
        return {
            "required": {
                # БЕЗ image_upload: True — ComfyUI иначе добавляет DOM-виджет
                # "choose file to upload", который ломал раскладку превью.
                # Загрузка идёт через свою кнопку "📁 Загрузить / Upload".
                "custom_path": ("STRING", {"default": "",
                    "tooltip": "Optional absolute path to an image. Priority: input_image > custom_path > image_name. Preview loads right after the path is entered (no Queue needed).\n---\nОпциональный абсолютный путь. Приоритет: input_image > custom_path > image_name. Превью подхватывается сразу после ввода пути (без Queue)."}),
                "image_name": ([""] + sorted(files), {
                    "tooltip": "Select an image from the input folder, or upload via the node button.\n---\nВыберите изображение из папки input или загрузите кнопкой."}),
                # Режим "Points (4 clicks)" удалён как избыточный.
                # Основной режим — Preset Ratio.
                "crop_mode": (["Preset Ratio", "Manual Size"], {"default": "Preset Ratio",
                    "tooltip": "Cropping method.\n---\nМетод обрезки."}),
                "aspect_ratio": (["1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"], {"default": "1:1",
                    "tooltip": "Target aspect ratio (Preset Ratio mode).\n---\nЦелевые пропорции (Preset Ratio)."}),
                "manual_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Crop width (Manual Size).\n---\nШирина обрезки (Manual Size)."}),
                "manual_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Crop height (Manual Size).\n---\nВысота обрезки (Manual Size)."}),
                "crop_coords": ("STRING", {"default": "[]", "multiline": False,
                    "tooltip": "AUTO-FILLED by the UI. Do not edit.\n---\nЗАПОЛНЯЕТСЯ интерфейсом. Не редактировать."}),
                "multiple": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Align output to a multiple (8/32/64).\n---\nВыравнивание выхода по кратности (8/32/64)."}),
                "pause_for_crop": ("BOOLEAN", {"default": True,
                    "tooltip": "Pause execution when a tensor is connected: adjust the crop in the node and press ▶️ Resume (auto-resume after 300 s, ✕ cancels).\n---\nПауза выполнения при подключённом тензоре: настройте кроп и нажмите ▶️ Продолжить (авто-продолжение через 300 с, ✕ отменяет)."}),
            },
            "optional": {
                "input_image": ("IMAGE", {"tooltip": "Optional image tensor (priority). With pause_for_crop the execution waits for your crop and ▶️ Resume.\n---\nОпциональный тензор (приоритет). При pause_for_crop выполнение ждёт ваш кроп и кнопку ▶️ Продолжить."}),
                "mask": ("MASK", {"tooltip": "Optional mask, cropped synchronously.\n---\nОпциональная маска, обрезается синхронно."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # crop_data — ПЕРВЫЙ выход (удобнее линковать в Stitch).
    RETURN_TYPES = ("CROP_DATA", "IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("crop_data", "cropped_image", "cropped_mask", "width", "height", "filename", "filepath")
    FUNCTION = "crop_image"

    def _load_source(self, input_image, mask, custom_path, image_name, unique_id):
        preview_name = None
        if input_image is not None:
            loaded = input_image
            b, oh, ow, c = loaded.shape
            if mask is not None:
                lm = mask
            elif c == 4:
                lm = 1.0 - loaded[..., 3]
            else:
                lm = torch.ones((b, oh, ow), dtype=torch.float32)
            fn = os.path.basename(image_name) if image_name else "tensor"
            fp = folder_paths.get_annotated_filepath(image_name) if image_name else ""
            try:
                preview_name = _crop_tensor_preview_png(loaded, unique_id)
            except Exception:
                logger.exception("[AGSoft Image Crop Plus] tensor preview failed")
                preview_name = None
            return loaded, lm, fn, fp, preview_name, True
        used_custom = bool(custom_path) and os.path.exists(custom_path)
        if used_custom:
            path = os.path.abspath(custom_path)
        else:
            if not image_name:
                raise ValueError("Не задан источник: input_image / custom_path / image_name.")
            path = folder_paths.get_annotated_filepath(image_name)
        pil = Image.open(path).convert("RGB")
        rgba = Image.open(path)
        if rgba.mode in ("RGBA", "LA") or (rgba.mode == "P" and "transparency" in rgba.info):
            a = np.array(rgba.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
            lm = (1.0 - torch.from_numpy(a)).unsqueeze(0)
        else:
            lm = torch.ones((1, pil.size[1], pil.size[0]))
        if mask is not None:
            lm = mask
        if used_custom:
            try:
                preview_name = _crop_ensure_preview_sync(path)
            except Exception:
                logger.exception("[AGSoft Image Crop Plus] custom preview failed")
                preview_name = None
        return pil_to_tensor(pil), lm, os.path.basename(path), path, preview_name, used_custom

    def crop_image(self, custom_path="", image_name="", crop_mode="Preset Ratio",
                   aspect_ratio="1:1", manual_width=512, manual_height=512,
                   crop_coords="[]", multiple=8, pause_for_crop=True,
                   input_image=None, mask=None, unique_id=None):
        try:
            loaded, lm, fn, fp, preview_name, used_override = self._load_source(
                input_image, mask, custom_path, image_name, unique_id)
            orig = loaded[0]
            img_w, img_h = orig.shape[1], orig.shape[0]
            uid = str(unique_id) if unique_id is not None else None
            if uid is not None:
                # kind — какой источник реально использован (для подписи в UI).
                kind = "tensor" if input_image is not None else ("custom" if (custom_path and os.path.exists(custom_path)) else "combo")
                _LAST_CROP[uid] = {"image": preview_name or "", "custom": bool(used_override), "stamp": int(time.time() * 1000), "kind": kind, "waiting": False}

                # === ПАУЗА ДЛЯ КРОПА: источник — тензор, превью уже в ноде.
                # Ждём кнопку "▶️ Продолжить" (или таймаут); живые координаты
                # кропа приходят через /agsoft/crop_live_coords.
                # Кнопка ✕ (interrupt) прерывает паузу и job — очередь разблокируется.
                if kind == "tensor" and pause_for_crop and _server is not None:
                    _LAST_CROP[uid]["waiting"] = True
                    try: _server.send_sync("agsoft_crop_waiting", {"node_id": uid})
                    except Exception: pass
                    t0 = time.time()
                    try:
                        while not _CROP_GO.get(uid) and (time.time() - t0) < PAUSE_TIMEOUT:
                            time.sleep(0.2)
                            if _mm is not None:
                                _mm.throw_exception_if_processing_interrupted()
                    except Exception:
                        _CROP_GO.pop(uid, None)
                        _CROP_LIVE.pop(uid, None)
                        _LAST_CROP[uid]["waiting"] = False
                        try: _server.send_sync("agsoft_crop_resumed", {"node_id": uid})
                        except Exception: pass
                        raise
                    _CROP_GO.pop(uid, None)
                    live = _CROP_LIVE.pop(uid, None)
                    if live:
                        crop_coords = live  # приоритет — координаты из паузы
                    _LAST_CROP[uid]["waiting"] = False
                    try: _server.send_sync("agsoft_crop_resumed", {"node_id": uid})
                    except Exception: pass

            if not crop_coords or crop_coords in ("[]", "{}"):
                crop_data = {"original_image": loaded.clone(), "x": 0, "y": 0, "w": img_w, "h": img_h, "device": loaded.device}
                return (crop_data, loaded, lm, img_w, img_h, fn, fp)
            data = json.loads(crop_coords)
            if isinstance(data, list) and len(data) == 4:
                # Обратная совместимость со старыми воркфлоу (4 точки).
                xs = [p['x'] for p in data]; ys = [p['y'] for p in data]
                min_x = max(0, int(min(xs))); min_y = max(0, int(min(ys)))
                w = int(max(xs)) - min_x; h = int(max(ys)) - min_y
            elif isinstance(data, dict) and 'x' in data and 'w' in data:
                min_x, min_y, w, h = int(data['x']), int(data['y']), int(data['w']), int(data['h'])
            else:
                crop_data = {"original_image": loaded.clone(), "x": 0, "y": 0, "w": img_w, "h": img_h, "device": loaded.device}
                return (crop_data, loaded, lm, img_w, img_h, fn, fp)
            mult = max(1, int(multiple))
            w = max(mult, (w // mult) * mult)
            h = max(mult, (h // mult) * mult)
            min_x = max(0, min(min_x, img_w - w))
            min_y = max(0, min(min_y, img_h - h))
            cropped = orig[min_y:min_y + h, min_x:min_x + w, :].unsqueeze(0)
            if lm.dim() == 2:
                lm = lm.unsqueeze(0)
            cropped_mask = lm[:, min_y:min_y + h, min_x:min_x + w]
            crop_data = {"original_image": loaded.clone(), "x": min_x, "y": min_y, "w": w, "h": h, "device": loaded.device}
            return (crop_data, cropped, cropped_mask, w, h, fn, fp)
        except Exception as e:
            raise RuntimeError(f"[AGSoft Image Crop Plus] Ошибка: {e}")

# ==============================================================================
# НОДА: 🧵 AGSoft Crop Stitch (на основе AGSoft Inpaint Stitch)
# ==============================================================================
class AGSoft_Crop_Stitch:
    DESCRIPTION = """Stitches a processed crop back into the original image using crop_data. Auto-resizes the crop to the box size. Modes: paste / feather_blend / poisson_blend.
Вставляет обработанный кадр обратно в оригинал по crop_data. Авто-ресайз к размеру бокса. Режимы: paste / feather_blend / poisson_blend."""
    CATEGORY = "AGSoft/Image"

    @classmethod
    def INPUT_TYPES(cls):
        blend_modes = ["paste", "feather_blend"]
        blend_modes.append("poisson_blend" if CV2_AVAILABLE else "poisson_blend (недоступен)")
        return {
            "required": {
                "crop_data": ("CROP_DATA", {"tooltip": "Данные от ✂️AGSoft Image Crop Plus (оригинал + бокс).\n---\nData from ✂️AGSoft Image Crop Plus (original + box)."}),
                "image": ("IMAGE", {"tooltip": "Обработанный кадр (может быть другого размера — будет ресайз).\n---\nProcessed frame (any size — will be resized)."}),
                "blend_mode": (blend_modes, {"default": "feather_blend",
                    "tooltip": "paste — жёсткая вставка; feather_blend — плавный шов; poisson_blend — сохранение освещения (нужен opencv).\n---\npaste — hard paste; feather_blend — smooth seam; poisson_blend — lighting-preserving (needs opencv)."}),
                "feathering": ("INT", {"default": 16, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Ширина плавного шва для feather_blend.\n---\nSeam width for feather_blend."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"

    def stitch(self, crop_data, image, blend_mode="feather_blend", feathering=16):
        if blend_mode == "poisson_blend (недоступен)":
            blend_mode = "paste"
        original = crop_data["original_image"]
        x, y, w, h = int(crop_data["x"]), int(crop_data["y"]), int(crop_data["w"]), int(crop_data["h"])
        device = crop_data.get("device", original.device)
        inp = image[0]
        # Ограничиваем координаты размерами оригинального изображения
        orig_h, orig_w = original.shape[1], original.shape[2]
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)
        # Авто-ресайз обработанного кадра к размеру бокса.
        if inp.shape[0] != h or inp.shape[1] != w:
            inp = _resize_tensor(inp, w, h, "lanczos")
        result = original.clone()
        orig_region = original[0, y:y + h, x:x + w, :].cpu().numpy()
        inp_region = inp[:h, :w, :].cpu().numpy()
        if orig_region.shape[2] == 1:
            orig_region = np.repeat(orig_region, 3, axis=2)
        if inp_region.shape[2] == 1:
            inp_region = np.repeat(inp_region, 3, axis=2)
        if blend_mode == "poisson_blend" and CV2_AVAILABLE:
            try:
                src_u8 = (np.clip(inp_region, 0, 1) * 255).astype(np.uint8)
                tgt_u8 = (np.clip(orig_region, 0, 1) * 255).astype(np.uint8)
                mask_u8 = np.ones((h, w), dtype=np.uint8) * 255
                center = (x + w // 2, y + h // 2)
                blended = cv2.seamlessClone(src_u8, tgt_u8, mask_u8, center, cv2.NORMAL_CLONE)
                blended_np = blended.astype(np.float32) / 255.0
            except Exception:
                blended_np = inp_region
        elif blend_mode == "feather_blend":
            weight = _border_weight(h, w, feathering)[..., None]
            blended_np = orig_region * (1.0 - weight) + inp_region * weight
        else:
            blended_np = inp_region
        blended_tensor = torch.from_numpy(blended_np).to(device=device, dtype=original.dtype)
        result[0, y:y + h, x:x + w, :] = blended_tensor
        return (result,)

NODE_CLASS_MAPPINGS = {
    "AGSoft Image Crop Plus": AGSoft_Image_Crop_Plus,
    "AGSoft_Crop_Stitch": AGSoft_Crop_Stitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft Image Crop Plus": "🖼️✂️AGSoft Image Crop Plus",
    "AGSoft_Crop_Stitch": "🖼️🧵AGSoft Crop Stitch",
}