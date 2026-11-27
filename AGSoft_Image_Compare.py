"""
==============================================================================
AGSoft_Image_Compare.py
==============================================================================
Нода / Node: 🖼️↔️AGSoft Image Compare
Версия / Version: v11.28
Автор / Author: AGSoft
Дата / Date: 28.11.2026
------------------------------------------------------------------------------
Описание / Description:

EN:
A server-side node for visual quality control of two images or IMAGE batches.
It takes image_a (reference) and image_b (candidate), normalizes each side to a
float32 tensor [N, H, W, 3] in the 0..1 range on CPU, aligns both sides and
compares them frame by frame.

Normalization. A lone [H, W, C] frame gains a batch axis, 5D batches are
flattened, uint8 ranges are divided by 255, single-channel data is repeated to
three channels and extra channels are dropped. swap_sides exchanges A and B
before any alignment, so the sides can be reversed without relinking inputs.

Alignment. resize_mode / resize_method bring both sides to a common resolution
(B resized to A, A resized to B, both resized to the smaller side, or a strict
bilingual error); length_mode reconciles different batch lengths (trim to the
shortest, pad the shorter side with black frames, or a strict bilingual error).

Outputs (fixed order, IMAGE only). wipe_preview — every frame split vertically
at wipe_position percent: the left part comes from B, the right part from A.
side_by_side_preview — full frames glued horizontally, B on the left, A on the
right. diff_frames — the amplified absolute difference |A-B| * diff_scale + 0.5
clamped to 0..1. view_mode, zoom, pan_x and pan_y affect only the browser
viewport and never change these outputs.

Preview pipeline. The node writes per-frame PNG previews for four sides: A
(reference), B (candidate), D (amplified difference) and S (horizontal concat of
full frames B|A for the side-by-side view). Files live in the
agsoft_image_compare subfolder of the ComfyUI temp folder; the folder is purged
on every ComfyUI start, lazily recreated before each write and trimmed to the
six most recent cache keys. At most 32 frames per key are cached, while the
IMAGE outputs always cover the whole aligned batch. The cache key is a hash of
both sides' content fingerprints plus the processing options, so previews are
rewritten only when the result would actually change; wipe_position, view_mode
and the view transform stay out of the key because the wipe composite and the
transform are applied in the browser.

Statistics. For every cached frame the node measures the mean and the maximum
absolute difference and the share of pixels whose difference exceeds 1/255, and
publishes them together with the frame size and the frame counts as per-node
metadata for the info line of the viewer.

Endpoints. Cached frames are served through a local endpoint with manual
chunked reads and no-store caching; client disconnects during transfer are
caught and logged at debug level only. A second endpoint returns per-node
metadata. A third, loopback-only endpoint accepts a raw PNG body from the
viewer, validates the PNG signature and the body size and writes the file into
the ComfyUI output folder as AGSoft_ImageCompare_<mode>_<timestamp>.png. Every
path is resolved and kept inside its allowed root.

Execution behaviour. The node is a graph terminal (OUTPUT_NODE): a workflow
containing only loaders and this node is valid and executes without any
downstream connection. IS_CHANGED always forces recomputation, VALIDATE_INPUTS
accepts everything, and empty or unsupported inputs raise a bilingual error.

---

RU:
Серверная нода визуального контроля качества двух изображений или батчей IMAGE.
Принимает image_a (эталон) и image_b (кандидат), приводит каждую сторону к
float32-тензору [N, H, W, 3] в диапазоне 0..1 на CPU, выравнивает обе стороны и
сравнивает их покадрово.

Нормализация. Одиночный кадр [H, W, C] получает ось батча, 5D-батчи схлопываются,
uint8-диапазон делится на 255, одноканальные данные повторяются до трёх каналов,
лишние каналы отбрасываются. swap_sides меняет A и B местами до любого
выравнивания, поэтому стороны можно поменять без переподключения входов.

Выравнивание. resize_mode / resize_method приводят стороны к общему разрешению
(B к размеру A, A к размеру B, обе к меньшей стороне либо строгая двуязычная
ошибка); length_mode согласует разную длину батчей (обрезка до короткого, добивка
короткой стороны чёрными кадрами либо строгая двуязычная ошибка).

Выходы (фиксированный порядок, только IMAGE). wipe_preview — каждый кадр разрезан
вертикально по wipe_position процентов: левая часть из B, правая из A.
side_by_side_preview — полные кадры, склеенные горизонтально: B слева, A справа.
diff_frames — усиленная абсолютная разница |A-B| * diff_scale + 0.5, зажатая в
0..1. view_mode, zoom, pan_x и pan_y влияют только на область просмотра в
браузере и никогда не меняют эти выходы.

Конвейер превью. Нода пишет покадровые PNG-превью для четырёх сторон: A (эталон),
B (кандидат), D (усиленная разница) и S (горизонтальная склейка полных кадров B|A
для режима side-by-side). Файлы лежат в подпапке agsoft_image_compare временной
папки ComfyUI; папка полностью очищается при каждом старте ComfyUI, лениво
пересоздаётся перед каждой записью и усекается до шести последних ключей кэша.
На один ключ кэшируется не более 32 кадров, тогда как выходы IMAGE всегда
покрывают весь выровненный батч. Ключ кэша — хеш контентных отпечатков обеих
сторон плюс параметры обработки, поэтому превью перезаписываются только когда
результат действительно меняется; wipe_position, view_mode и трансформация вида
в ключ не входят, потому что wipe-склейка и трансформация выполняются в браузере.

Статистика. Для каждого кэшированного кадра нода измеряет среднюю и максимальную
абсолютную разницу и долю пикселей, чья разница превышает 1/255, и публикует их
вместе с размером кадра и числом кадров как метаданные ноды для строки
информации вьювера.

Эндпоинты. Кэшированные кадры отдаются локальным эндпоинтом с ручным чанковым
чтением и no-store кэшированием; обрывы соединения клиентом во время передачи
перехватываются и логируются только на уровне debug. Второй эндпоинт отдаёт
метаданные ноды. Третий, loopback-only эндпоинт принимает сырое PNG-тело из
вьювера, валидирует PNG-сигнатуру и размер тела и пишет файл в папку output
ComfyUI под именем AGSoft_ImageCompare_<mode>_<timestamp>.png. Каждый путь
резолвится и удерживается внутри своего разрешённого корня.

Поведение при выполнении. Нода является терминалом графа (OUTPUT_NODE): workflow,
содержащий только загрузчики и эту ноду, валиден и выполняется без нижестоящих
подключений. IS_CHANGED всегда форсирует пересчёт, VALIDATE_INPUTS принимает всё,
а пустые или неподдерживаемые входы вызывают двуязычную ошибку.
------------------------------------------------------------------------------
Возможности / Features:

⚡ Both sides normalized to one float32 [N, H, W, 3] 0..1 tensor per side, with
batch-axis, 5D, uint8 and channel-count handling, plus swap_sides before
alignment.
Обе стороны нормализуются к одному float32-тензору [N, H, W, 3] 0..1 на сторону
с обработкой оси батча, 5D, uint8 и числа каналов, плюс swap_sides до
выравнивания.

⚡ Size alignment (to_a / to_b / min / none) with a resampling choice and batch
length alignment (trim / pad with black / require equal), both with strict
bilingual errors.
Выравнивание размеров (to_a / to_b / min / none) с выбором ресемплинга и
выравнивание длины батча (обрезка / добивка чёрным / требование равенства), оба
со строгими двуязычными ошибками.

⚡ Three fixed IMAGE outputs with exact formulas: wipe split at wipe_position
(B left / A right), horizontal B|A glue, amplified |A-B| * diff_scale + 0.5.
Три фиксированных выхода IMAGE с точными формулами: wipe-разрез по wipe_position
(B слева / A справа), горизонтальная склейка B|A, усиленная
|A-B| * diff_scale + 0.5.

⚡ Per-frame PNG cache A / B / D / S in the temp folder: purged on every start,
lazily recreated, LRU-limited to six keys, 32 preview frames per key, rewritten
only on a real content or option change.
Покадровый PNG-кэш A / B / D / S во временной папке: очистка при каждом старте,
ленивое пересоздание, LRU-лимит в шесть ключей, 32 превью-кадра на ключ,
перезапись только при реальном изменении данных или параметров.

⚡ Per-frame difference statistics (mean, maximum, share of changed pixels)
published as per-node metadata for the viewer info line.
Покадровая статистика разницы (средняя, максимальная, доля изменённых пикселей)
публикуется как метаданные ноды для строки информации вьювера.

⚡ Local endpoints: chunked frame serving with quiet client-disconnect handling,
no-store metadata, and a loopback-only PNG save with signature and size
validation into the output folder.
Локальные эндпоинты: чанковая отдача кадров с тихим перехватом отключений
клиента, no-store метаданные и loopback-only сохранение PNG с валидацией
сигнатуры и размера в папку output.

⚡ Viewport-only widgets (view_mode, zoom, pan_x, pan_y) never change the outputs
and never invalidate the preview cache; preview_width caps the cached PNG width.
Виджеты только для просмотра (view_mode, zoom, pan_x, pan_y) не меняют выходы и
не инвалидируют кэш превью; preview_width ограничивает ширину кэшируемых PNG.

⚡ Terminal node: executes standalone with nothing connected downstream, always
recomputes, bilingual runtime errors for empty or unsupported inputs.
Терминальная нода: выполняется автономно без нижестоящих подключений, всегда
пересчитывается, двуязычные ошибки времени выполнения для пустых или
неподдерживаемых входов.
==============================================================================
"""

import os
import re
import time
import asyncio
import hashlib
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

import folder_paths
import comfy.utils
from server import PromptServer
from aiohttp import web

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Image Compare] v11.28 loaded (PNG cache, batch stats, save-frame, swap sides)")

#==============================================================================
# Кэш во временной папке ComfyUI / Cache in the ComfyUI temp folder
#==============================================================================
CACHE_DIR = os.path.join(folder_paths.get_temp_directory(), "agsoft_image_compare")
KEY_BY_NODE: Dict[str, str] = {}
META_BY_KEY: Dict[str, Dict[str, Any]] = {}
LRU_MAX_KEYS = 6
PREVIEW_MAX_FRAMES = 32

# Обрывы клиента при отдаче файлов / Client disconnects while serving files
_CLIENT_GONE = (ConnectionError, ConnectionResetError, OSError, asyncio.CancelledError)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _purge_cache() -> None:
    """Полная очистка кэша при старте / Full cache purge on start."""
    try:
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"[AGSoft Image Compare] cache purge skipped: {e}")


_purge_cache()


def _ensure_cache_dir() -> bool:
    """Лениво пересоздаёт папку кэша / Lazily recreates the cache dir."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return True
    except Exception as e:
        logger.warning(f"[AGSoft Image Compare] cache dir create failed: {e}")
        return False


def _lru_trim() -> None:
    """Держит не более LRU_MAX_KEYS ключей / Keeps at most LRU_MAX_KEYS keys."""
    if not os.path.isdir(CACHE_DIR):
        return
    try:
        keys: Dict[str, float] = {}
        for fn in os.listdir(CACHE_DIR):
            m = re.match(r"cmp([0-9a-f]{16})_", fn)
            if m:
                p = os.path.join(CACHE_DIR, fn)
                keys[m.group(1)] = max(keys.get(m.group(1), 0.0), os.path.getmtime(p))
        order = sorted(keys.items(), key=lambda kv: kv[1])
        stale = order[:-LRU_MAX_KEYS] if len(order) > LRU_MAX_KEYS else []
        for key, _ in stale:
            for fn in os.listdir(CACHE_DIR):
                if key in fn:
                    os.remove(os.path.join(CACHE_DIR, fn))
            META_BY_KEY.pop(key, None)
    except Exception as e:
        logger.warning(f"[AGSoft Image Compare] lru trim skipped: {e}")


def _T(en: str, ru: str) -> str:
    """Двуязычный текст / Bilingual text."""
    return en + "\n---\n" + ru


#==============================================================================
# Утилиты / Utilities
#==============================================================================
def _to_images(value: Any) -> torch.Tensor:
    """IMAGE-тензор или dict → float32 [N, H, W, 3] 0..1 на CPU.
    IMAGE tensor or dict to float32 [N, H, W, 3] 0..1 on CPU."""
    t = value
    if isinstance(t, dict):
        t = t.get("images")
        if t is None:
            t = value.get("pixels")
    if not isinstance(t, torch.Tensor):
        raise TypeError(_T("Unsupported input type", "Неподдерживаемый тип входа")
                        + f": {type(value).__name__}")
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() == 5:
        t = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
    if t.dim() != 4:
        raise ValueError(_T("Expected IMAGE batch [N,H,W,C]", "Ожидается батч IMAGE [N,H,W,C]")
                         + f", got {tuple(t.shape)}")
    t = t.detach().cpu()
    if t.dtype != torch.float32:
        t = t.float()
    if t.numel() > 0 and t.max() > 1.5:
        t = t / 255.0
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    elif t.shape[-1] > 3:
        t = t[..., :3]
    return t


def _fp(t: torch.Tensor) -> str:
    """Дешёвый контентный отпечаток / Cheap content fingerprint."""
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    flat = t.flatten()
    step = max(1, flat.numel() // 262144)
    samp = flat[::step][:262144]
    h.update(samp.contiguous().numpy().tobytes())
    return h.hexdigest()[:8]


def _resize_frames(frames: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    """Ресемплинг батча через core-утилиту / Batch resample via core utility."""
    s = frames[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, w, h, method, "disabled")
    return s.movedim(1, -1)


def _write_png(path: str, fr: torch.Tensor, max_w: int) -> bool:
    """Один кадр [H,W,C] в PNG / One [H,W,C] frame to PNG."""
    try:
        fr = fr.clamp(0, 1)
        if max_w and fr.shape[1] > max_w:
            nh = max(2, int(round(fr.shape[0] * max_w / fr.shape[1])))
            fr = _resize_frames(fr.unsqueeze(0), max_w, nh, "lanczos")[0]
        arr = (fr * 255.0).round().to(torch.uint8).numpy()
        Image.fromarray(arr).save(path, format="PNG", compress_level=1)
        return True
    except Exception as e:
        logger.warning(f"[AGSoft Image Compare] png write failed: {e}")
        return False


def _ensure_pngs(key: str, a: torch.Tensor, b: torch.Tensor,
                 preview_width: int, diff_scale: float) -> int:
    """Кэш PNG A/B/D/S по кадрам; запись только при промахе.
    S — горизонтальная склейка полных кадров B|A.
    Per-frame PNG cache A/B/D/S; written only on miss.
    S is the horizontal concat of full frames B|A."""
    if not _ensure_cache_dir():
        return 0
    n = a.shape[0]
    pf = min(n, PREVIEW_MAX_FRAMES)
    for i in range(pf):
        pa = os.path.join(CACHE_DIR, f"cmp{key}_A_{i:04d}.png")
        pb = os.path.join(CACHE_DIR, f"cmp{key}_B_{i:04d}.png")
        pd = os.path.join(CACHE_DIR, f"cmp{key}_D_{i:04d}.png")
        ps = os.path.join(CACHE_DIR, f"cmp{key}_S_{i:04d}.png")
        if not os.path.isfile(pa):
            _write_png(pa, a[i], preview_width)
        if not os.path.isfile(pb):
            _write_png(pb, b[i], preview_width)
        if not os.path.isfile(pd):
            _write_png(pd, (a[i] - b[i]) * diff_scale + 0.5, preview_width)
        if not os.path.isfile(ps):
            _write_png(ps, torch.cat([b[i], a[i]], dim=1), preview_width)
    _lru_trim()
    return pf


def _frame_stats(a: torch.Tensor, b: torch.Tensor, pf: int) -> List[Dict[str, float]]:
    """Статистика разницы по кадрам для инфостроки / Per-frame diff stats for the bar."""
    out: List[Dict[str, float]] = []
    for i in range(pf):
        d = (a[i] - b[i]).abs()
        changed = (d.amax(dim=-1) > 1.0 / 255.0).float().mean() * 100.0
        out.append({"mean": round(float(d.mean()), 4),
                    "max": round(float(d.max()), 4),
                    "pct": round(float(changed), 2)})
    return out


#==============================================================================
# HTTP-эндпоинты / HTTP endpoints
#==============================================================================
def _safe_cache_path(name: str) -> Optional[str]:
    """Путь только внутри кэша / Path strictly inside the cache."""
    p = os.path.realpath(os.path.join(CACHE_DIR, name))
    if not p.startswith(os.path.realpath(CACHE_DIR) + os.sep):
        return None
    return p if os.path.isfile(p) else None


@PromptServer.instance.routes.get("/agsoft/image_compare/image")
async def agsoft_image_compare_image(request: web.Request) -> web.StreamResponse:
    """Отдача PNG-кадра кэша / Serve a cached PNG frame."""
    key = request.rel_url.query.get("key", "")
    side = request.rel_url.query.get("side", "A")
    frame_s = request.rel_url.query.get("frame", "0")
    if not re.fullmatch(r"[0-9a-f]{16}", key) or side not in ("A", "B", "D", "S"):
        return web.Response(status=400)
    try:
        frame = int(frame_s)
    except ValueError:
        return web.Response(status=400)
    if frame < 0 or frame >= PREVIEW_MAX_FRAMES:
        return web.Response(status=400)
    path = _safe_cache_path(f"cmp{key}_{side}_{frame:04d}.png")
    if path is None:
        return web.Response(status=404)
    size = os.path.getsize(path)
    headers = {"Content-Type": "image/png", "Content-Length": str(size),
               "Cache-Control": "no-store"}
    resp = web.StreamResponse(status=200, headers=headers)
    try:
        await resp.prepare(request)
    except _CLIENT_GONE:
        return resp
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                await resp.write(chunk)
        await resp.write_eof()
    except _CLIENT_GONE:
        logger.debug(f"[AGSoft Image Compare] image {side} dropped by client (key {key})")
    return resp


@PromptServer.instance.routes.get("/agsoft/image_compare/meta")
async def agsoft_image_compare_meta(request: web.Request) -> web.Response:
    """Метаданные кэша по ноде / Per-node cache metadata."""
    node = request.rel_url.query.get("node", "")
    key = KEY_BY_NODE.get(node)
    if not key:
        return web.json_response({"key": None}, headers={"Cache-Control": "no-store"})
    meta = dict(META_BY_KEY.get(key, {}))
    meta["key"] = key
    return web.json_response(meta, headers={"Cache-Control": "no-store"})


@PromptServer.instance.routes.post("/agsoft/image_compare/save_frame")
async def agsoft_image_compare_save_frame(request: web.Request) -> web.Response:
    """Сохранение кадра сравнения: loopback-only, PNG-валидация, запись в output.
    Save a comparison frame: loopback-only, PNG validation, write into output."""
    if request.remote not in ("127.0.0.1", "::1"):
        return web.Response(status=403)
    mode = request.rel_url.query.get("mode", "wipe")
    if mode not in ("wipe", "side_by_side", "difference"):
        return web.Response(status=400)
    data = await request.read()
    if not data or len(data) > 32 * 1024 * 1024:
        return web.Response(status=400)
    if data[:8] != _PNG_MAGIC:
        return web.Response(status=400)
    name = f"AGSoft_ImageCompare_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    out_dir = folder_paths.get_output_directory()
    path = os.path.realpath(os.path.join(out_dir, name))
    if not path.startswith(os.path.realpath(out_dir) + os.sep):
        return web.Response(status=400)
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.warning(f"[AGSoft Image Compare] save frame failed: {e}")
        return web.Response(status=500)
    logger.info(f"[AGSoft Image Compare] frame saved: {name}")
    return web.json_response({"ok": True, "name": name})


#==============================================================================
# Нода: 🖼️↔️ AGSoft Image Compare
#==============================================================================
class AGSoftImageCompare:
    """Визуальное сравнение двух изображений / Visual comparison of two images."""

    CATEGORY = "AGSoft/Image"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    OUTPUT_NODE = True

    DESCRIPTION = _T(
        "Visually compares two images or IMAGE batches: size alignment "
        "(resize_mode/resize_method), batch length alignment (length_mode), "
        "swap_sides, and three IMAGE outputs — wipe_preview (B left / A right "
        "at wipe_position), side_by_side_preview (B|A glue) and diff_frames "
        "(|A-B|*diff_scale+0.5). UI: DOM viewport with a draggable wipe line, "
        "frame slider for batches, wheel zoom + drag pan, Space to peek side A, "
        "per-frame difference stats in the info bar and a save button that "
        "stores the current comparison frame as PNG into the output folder. "
        "Terminal node: runs even with nothing connected.",
        "Визуально сравнивает два изображения или батчи IMAGE: выравнивание "
        "размеров (resize_mode/resize_method), выравнивание длины батча "
        "(length_mode), swap_sides и три выхода IMAGE — wipe_preview (B слева / "
        "A справа по wipe_position), side_by_side_preview (склейка B|A) и "
        "diff_frames (|A-B|*diff_scale+0.5). UI: DOM-область с перетаскиваемой "
        "wipe-линией, слайдер кадра для батчей, зум колесом + панорамирование "
        "драгом, Space для показа стороны A, статистика разницы по кадрам в "
        "инфостроке и кнопка сохранения текущего кадра сравнения в PNG в папку "
        "output. Терминальная нода: выполняется даже без подключённых выходов.")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": _T(
                    "Reference side A: IMAGE tensor or batch.",
                    "Эталонная сторона A: тензор IMAGE или батч.")}),
                "image_b": ("IMAGE", {"tooltip": _T(
                    "Candidate side B: IMAGE tensor or batch.",
                    "Сторона-кандидат B: тензор IMAGE или батч.")}),
                "view_mode": (["wipe", "side_by_side", "difference"],
                              {"default": "wipe", "tooltip": _T(
                    "UI view mode: wipe slider, server side-by-side image B|A, "
                    "or the amplified difference image.",
                    "Режим просмотра UI: wipe-слайдер, серверное изображение "
                    "B|A рядом или усиленная разница.")}),
                "wipe_position": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1,
                                          "display": "number", "tooltip": _T(
                    "Divider position in percent: left shows B, right shows A. "
                    "Also used for the wipe_preview output and the saved frame.",
                    "Позиция разделителя в процентах: слева B, справа A. "
                    "Используется для выхода wipe_preview и сохраняемого кадра.")}),
                "diff_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                         "display": "number", "tooltip": _T(
                    "Amplification of the visual difference: |A-B|*scale+0.5.",
                    "Усиление визуальной разницы: |A-B|*scale+0.5.")}),
                "resize_mode": (["to_a", "to_b", "min", "none"],
                                {"default": "to_a", "tooltip": _T(
                    "Size mismatch handling: resize B to A, A to B, both to the "
                    "smaller, or none (error on mismatch).",
                    "Обработка несовпадения размеров: B к A, A к B, оба к "
                    "меньшему, или none (ошибка при несовпадении).")}),
                "resize_method": (["lanczos", "bicubic", "nearest"],
                                  {"default": "lanczos", "tooltip": _T(
                    "Resampling algorithm used when sizes differ.",
                    "Алгоритм ресемплинга при различии размеров.")}),
                "length_mode": (["trim_shortest", "pad_with_black", "require_equal"],
                                {"default": "trim_shortest", "tooltip": _T(
                    "Batch length mismatch handling: trim to the shortest, pad "
                    "the shorter with black frames, or require equal lengths.",
                    "Обработка разной длины батча: обрезка до короткого, добивка "
                    "короткого чёрными кадрами или требование равенства.")}),
                "swap_sides": ("BOOLEAN", {"default": False, "tooltip": _T(
                    "Swap A and B before comparison without relinking inputs.",
                    "Поменять A и B местами перед сравнением без переподключения "
                    "входов.")}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                   "display": "number", "tooltip": _T(
                    "Viewport zoom (client-side only). Mouse wheel over the "
                    "preview also changes it; double click resets the view.",
                    "Зум области просмотра (только клиент). Колесо мыши над "
                    "превью тоже меняет его; двойной клик сбрасывает вид.")}),
                "pan_x": ("INT", {"default": 0, "min": -5000, "max": 5000, "step": 1,
                                  "display": "number", "tooltip": _T(
                    "Horizontal pan in pixels (client-side only). Drag the "
                    "preview to pan when zoomed in.",
                    "Смещение по горизонтали в пикселях (только клиент). Тяните "
                    "превью для панорамирования при зуме > 1.")}),
                "pan_y": ("INT", {"default": 0, "min": -5000, "max": 5000, "step": 1,
                                  "display": "number", "tooltip": _T(
                    "Vertical pan in pixels (client-side only). Drag the "
                    "preview to pan when zoomed in.",
                    "Смещение по вертикали в пикселях (только клиент). Тяните "
                    "превью для панорамирования при зуме > 1.")}),
                "preview_width": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 2,
                                          "display": "number", "tooltip": _T(
                    "Max width of cached preview PNGs (0 = original; the S image "
                    "is twice as wide as a single side).",
                    "Макс. ширина кэшируемых PNG-превью (0 = оригинал; изображение "
                    "S вдвое шире одной стороны).")}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs: Any) -> bool:
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return float("nan")

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("wipe_preview", "side_by_side_preview", "diff_frames")
    OUTPUT_TOOLTIPS = (
        _T("Frames split at wipe_position: left part from B, right part from A.",
           "Кадры, разрезанные по wipe_position: слева часть B, справа часть A."),
        _T("Full frames glued side by side: B on the left, A on the right.",
           "Полные кадры, склеенные рядом: B слева, A справа."),
        _T("Amplified per-frame difference |A-B|*diff_scale+0.5.",
           "Усиленная покадровая разница |A-B|*diff_scale+0.5."),
    )

    def main(self, image_a: torch.Tensor, image_b: torch.Tensor, view_mode: str,
             wipe_position: int, diff_scale: float, resize_mode: str, resize_method: str,
             length_mode: str, swap_sides: bool, zoom: float, pan_x: int, pan_y: int,
             preview_width: int, unique_id: Optional[str] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = _to_images(image_a)
        b = _to_images(image_b)
        if a.shape[0] == 0 or b.shape[0] == 0:
            raise ValueError(_T("Empty image on input", "Пустое изображение на входе"))
        if swap_sides:
            a, b = b, a
        # 1. Выравнивание размера / Size alignment
        if (a.shape[1], a.shape[2]) != (b.shape[1], b.shape[2]):
            if resize_mode == "none":
                raise ValueError(_T(
                    f"Size mismatch A={a.shape[2]}x{a.shape[1]} B={b.shape[2]}x{b.shape[1]}; "
                    "set resize_mode",
                    f"Несовпадение размеров A={a.shape[2]}x{a.shape[1]} "
                    f"B={b.shape[2]}x{b.shape[1]}; задайте resize_mode"))
            if resize_mode == "to_a":
                b = _resize_frames(b, a.shape[2], a.shape[1], resize_method)
            elif resize_mode == "to_b":
                a = _resize_frames(a, b.shape[2], b.shape[1], resize_method)
            else:
                w, h = min(a.shape[2], b.shape[2]), min(a.shape[1], b.shape[1])
                a = _resize_frames(a, w, h, resize_method)
                b = _resize_frames(b, w, h, resize_method)
        # 2. Выравнивание длины батча / Batch length alignment
        if a.shape[0] != b.shape[0]:
            if length_mode == "require_equal":
                raise ValueError(_T(
                    f"Length mismatch A={a.shape[0]} B={b.shape[0]}; set length_mode",
                    f"Несовпадение длины A={a.shape[0]} B={b.shape[0]}; задайте length_mode"))
            if length_mode == "trim_shortest":
                n = min(a.shape[0], b.shape[0])
                a, b = a[:n], b[:n]
            else:
                n = max(a.shape[0], b.shape[0])
                if a.shape[0] < n:
                    a = torch.cat([a, torch.zeros(n - a.shape[0], *a.shape[1:])], 0)
                if b.shape[0] < n:
                    b = torch.cat([b, torch.zeros(n - b.shape[0], *b.shape[1:])], 0)
        # 3. Визуальные выходы / Visual outputs
        diff_frames = ((a - b) * diff_scale + 0.5).clamp(0, 1)
        split = round(a.shape[2] * max(0, min(100, int(wipe_position))) / 100.0)
        wipe_preview = torch.cat([b[:, :, :split], a[:, :, split:]], dim=2)
        side_by_side_preview = torch.cat([b, a], dim=2)
        # 4. Кэш превью и метаданные / Preview cache and metadata
        opts = (f"{resize_mode}|{resize_method}|{length_mode}|{int(bool(swap_sides))}|"
                f"{preview_width}|{round(diff_scale, 2)}")
        key = hashlib.sha256((_fp(a) + _fp(b) + opts).encode()).hexdigest()[:16]
        pf = _ensure_pngs(key, a, b, preview_width, diff_scale)
        META_BY_KEY[key] = {"w": int(a.shape[2]), "h": int(a.shape[1]),
                            "frames": int(a.shape[0]), "preview_frames": pf,
                            "stats": _frame_stats(a, b, pf)}
        if unique_id is not None:
            KEY_BY_NODE[str(unique_id)] = key
            logger.info(f"[AGSoft Image Compare] node {unique_id} -> key {key}")
        return (wipe_preview, side_by_side_preview, diff_frames)


#==============================================================================
# Регистрация / Registration
#==============================================================================
NODE_CLASS_MAPPINGS = {
    "AGSoftImageCompare": AGSoftImageCompare,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftImageCompare": "🖼️↔️AGSoft Image Compare",
}