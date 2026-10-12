"""
==============================================================================
AGSoft_Video_Compare.py
==============================================================================
Node: 🎬AGSoft Video Compare
Version: v10.29
Author / Автор: AGSoft
Date / Дата: 12.10.2026
------------------------------------------------------------------------------
Description / Описание:

A server-side node for frame-accurate visual quality control of two video
sources. It accepts two inputs of any compatible type (VIDEO objects of the
ComfyUI video API, IMAGE frame batches, dictionaries with 'images'/'pixels'
keys, or raw torch tensors), converts each side into a float32 frame tensor
of shape [N, H, W, C] in the 0..1 range (5D batches are flattened, uint8
ranges are divided by 255), and then compares the two sides frame by frame.

Time window. start_second and compare_duration (0 = until the end of the
clip) limit the comparison to a sub-interval of BOTH sides. For VIDEO objects
the window is applied BEFORE decoding through the official trim API
(as_trimmed), which returns a new trimmed object, so skipped footage never
enters RAM; for sources without that API (IMAGE batches, tensors, dicts) the
window is applied after materialization as a copy-free slice view, and the
window seconds are converted to frames using the image_fps widget, which
supplies the assumed frame rate for inputs that carry no fps metadata of
their own (it is ignored for VIDEO inputs, which carry their own fps).

Alignment. resize_mode / resize_method bring both sides to a common
resolution (B resized to A, A resized to B, both resized to the smaller side,
or a strict bilingual error); length_mode reconciles different frame counts
(trim to the shortest, pad the shorter side with black frames, or a strict
bilingual error).

Outputs (fixed order, IMAGE only). wipe_preview — every frame split
vertically at wipe_position percent: the left part comes from B, the right
part from A. side_by_side_preview — full frames glued horizontally, B on the
left, A on the right. diff_frames — the amplified per-frame absolute
difference |A-B| * diff_scale + 0.5, clamped to 0..1. frame_step makes the
outputs use every N-th frame, while the preview streams always keep all
frames of the window.

Preview pipeline. The node encodes four H.264 preview streams with PyAV:
A (reference), B (candidate), D (amplified difference) and S (horizontal
concat of full frames B|A for the side-by-side view). Streams live in the
agsoft_video_compare subfolder of the ComfyUI temp folder; the folder is
purged on every ComfyUI start, lazily recreated before each write, and
trimmed to the six most recent cache keys. A cache key is a hash of both
sides' content fingerprints plus all processing options, so streams are
re-encoded only when the result would actually change. Streams are served to
the browser through a local HTTP endpoint with Range support and manual
chunked reads; client disconnects during streaming (scrubbing, prebuffering)
are caught and logged at debug level only. A second endpoint returns per-node
cache metadata (key, fps, frames, width, height, duration) with no-store
caching. A third, loopback-only endpoint accepts a raw PNG body from the
player, validates the PNG signature and size, and writes the file into the
ComfyUI output folder as AGSoft_VideoCompare_<mode>_<timestamp>.png.

Execution behaviour. The node is a graph terminal (OUTPUT_NODE): a workflow
containing only loaders and this node is valid and executes without any
downstream connection. IS_CHANGED always forces recomputation; VALIDATE_INPUTS
accepts everything, and unsupported input types raise a bilingual runtime
error. Empty inputs raise a bilingual error as well.
---
Серверная нода покадрового визуального контроля качества двух видеоисточников.
Принимает два входа любого совместимого типа (объекты VIDEO видео-API ComfyUI,
батчи кадров IMAGE, словари с ключами 'images'/'pixels' или сырые тензоры
torch), приводит каждую сторону к float32-тензору кадров формы [N, H, W, C] в
диапазоне 0..1 (5D-батчи схлопываются, uint8-диапазон делится на 255), после
чего сравнивает стороны покадрово.

Временное окно. start_second и compare_duration (0 = до конца ролика)
ограничивают сравнение подинтервалом ОБЕИХ сторон. Для объектов VIDEO окно
применяется ДО декодирования через официальный trim-API (as_trimmed), который
возвращает новый обрезанный объект, поэтому пропущенный материал никогда не
попадает в RAM; для источников без этого API (батчи IMAGE, тензоры, словари)
окно применяется после материализации как срез-view без копирования, а
секунды окна переводятся в кадры виджетом image_fps, задающим предполагаемую
частоту кадров для входов без собственных метаданных fps (для входов VIDEO он
игнорируется — у них свой fps).

Выравнивание. resize_mode / resize_method приводят обе стороны к общему
разрешению (B к размеру A, A к размеру B, обе к меньшей стороне, либо строгая
двуязычная ошибка); length_mode согласует разное число кадров (обрезка до
короткого, добивка короткой стороны чёрными кадрами, либо строгая двуязычная
ошибка).

Выходы (фиксированный порядок, только IMAGE). wipe_preview — каждый кадр
разрезан вертикально по wipe_position процентов: левая часть из B, правая из
A. side_by_side_preview — полные кадры, склеенные горизонтально: B слева, A
справа. diff_frames — усиленная покадровая абсолютная разница
|A-B| * diff_scale + 0.5, зажатая в 0..1. frame_step заставляет выходы брать
каждый N-й кадр, тогда как превью-потоки всегда содержат все кадры окна.

Конвейер превью. Нода кодирует четыре превью-потока H.264 через PyAV: A
(эталон), B (кандидат), D (усиленная разница) и S (горизонтальная склейка
полных кадров B|A для режима side-by_side). Потоки лежат в подпапке
agsoft_video_compare временной папки ComfyUI; папка полностью очищается при
каждом старте ComfyUI, лениво пересоздаётся перед каждой записью и усекается
до шести последних ключей кэша. Ключ кэша — хеш контентных отпечатков обеих
сторон плюс все параметры обработки, поэтому потоки перекодируются только
когда результат действительно меняется. Потоки отдаются браузеру через
локальный HTTP-эндпоинт с поддержкой Range и ручным чанковым чтением; обрывы
соединения клиентом (перемотка, пребуфер) перехватываются и логируются только
на уровне debug. Второй эндпоинт отдаёт метаданные кэша по ноде (key, fps,
frames, width, height, duration) с no-store кэшированием. Третий, loopback-
only эндпоинт принимает сырое PNG-тело из плеера, валидирует PNG-сигнатуру и
размер и пишет файл в папку output ComfyUI под именем
AGSoft_VideoCompare_<mode>_<timestamp>.png.

Поведение при выполнении. Нода является терминалом графа (OUTPUT_NODE):
workflow, содержащий только загрузчики и эту ноду, валиден и выполняется без
нижестоящих подключений. IS_CHANGED всегда форсирует пересчёт; VALIDATE_INPUTS
принимает всё, а неподдерживаемые типы входов вызывают двуязычную ошибку
времени выполнения. Пустые входы также вызывают двуязычную ошибку.

Features / Возможности:
⚡ Wildcard inputs: VIDEO objects, IMAGE batches, dicts and raw tensors are
   all normalized to one float32 [N, H, W, C] 0..1 frame tensor per side.
   Wildcard-входы: объекты VIDEO, батчи IMAGE, словари и сырые тензоры
   нормализуются к одному float32-тензору кадров [N, H, W, C] 0..1 на сторону.
⚡ RAM-safe time window: decoded-only window via as_trimmed for VIDEO, copy-
   free slice view plus image_fps seconds-to-frames conversion for fps-less
   inputs; long clips never load in full.
   RAM-безопасное окно: для VIDEO декодируется только окно через as_trimmed,
   для входов без fps — срез-view без копирования плюс перевод секунд в кадры
   по image_fps; длинные ролики никогда не загружаются целиком.
⚡ Three fixed IMAGE outputs with exact formulas: wipe split at
   wipe_position (B left / A right), horizontal B|A glue, amplified |A-B|.
   Три фиксированных выхода IMAGE с точными формулами: wipe-разрез по
   wipe_position (B слева / A справа), горизонтальная склейка B|A, усиленная
   |A-B|.
⚡ Four PyAV-encoded preview streams (A, B, D, S) in a temp cache purged on
   every ComfyUI start, lazily recreated, LRU-limited to six keys, re-encoded
   only on a real content/option change.
   Четыре превью-потока (A, B, D, S), кодированных PyAV, во временном кэше,
   очищаемом при каждом старте ComfyUI, лениво пересоздаваемом, с LRU-лимитом
   в шесть ключей и перекодированием только при реальном изменении.
⚡ Local HTTP endpoints: Range-aware chunked stream serving with silent
   client-disconnect handling, no-store metadata, and a loopback-only PNG
   save endpoint with signature and size validation into the output folder.
   Локальные HTTP-эндпоинты: чанковая отдача потоков с Range и тихим
   перехватом отключений клиента, no-store метаданные, loopback-only
   эндпоинт сохранения PNG с валидацией сигнатуры и размера в папку output.
⚡ Memory guards: with frame_step = 1 all outputs are computed on tensor
   views without a single full-size copy; larger steps use advanced indexing.
   Защита памяти: при frame_step = 1 все выходы считаются на view тензоров
   без единой полноразмерной копии; большие шаги используют advanced indexing.
⚡ Terminal node: executes standalone with nothing connected; always
   recomputes; bilingual runtime errors for empty or unsupported inputs.
   Терминальная нода: выполняется автономно без подключений; всегда
   пересчитывается; двуязычные ошибки времени выполнения для пустых или
   неподдерживаемых входов.
==============================================================================
"""

import os
import re
import json
import math
import asyncio
import time
import hashlib
import shutil
import logging
import folder_paths
import torch
import comfy.utils
from server import PromptServer
from aiohttp import web
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("[AGSoft Video Compare] v10.29 loaded (comparison window applies to IMAGE-batch inputs via image_fps)")
try:
    import av
except Exception:
    av = None
    logger.warning("[AGSoft Video Compare] PyAV missing: preview streams disabled")

#==============================================================================
# Кэш во временной папке ComfyUI / Cache in the ComfyUI temp folder
#==============================================================================
CACHE_DIR = os.path.join(folder_paths.get_temp_directory(), "agsoft_video_compare")
KEY_BY_NODE = {}
META_BY_KEY = {}
LRU_MAX_KEYS = 6

# Все ошибки «клиент закрыл сокет» на Windows / All "client closed socket" errors on Windows
_CLIENT_GONE = (ConnectionError, ConnectionResetError, OSError, asyncio.CancelledError)
# PNG-сигнатура для валидации сохраняемого кадра / PNG signature for save validation
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

def _purge_cache():
    """Полная очистка кэша при старте ComfyUI / Full cache purge on ComfyUI start."""
    try:
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"[AGSoft Video Compare] cache purge skipped: {e}")
_purge_cache()

def _ensure_cache_dir():
    """Лениво пересоздаёт папку кэша, если её удалили извне /
    Lazily recreates the cache dir if it was removed externally."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return True
    except Exception as e:
        logger.warning(f"[AGSoft Video Compare] cache dir create failed: {e}")
        return False

def _lru_trim():
    """Держит не более LRU_MAX_KEYS ключей / Keeps at most LRU_MAX_KEYS keys."""
    if not os.path.isdir(CACHE_DIR):
        return
    try:
        keys = {}
        for fn in os.listdir(CACHE_DIR):
            m = re.match(r"cmp_([0-9a-f]{16})_", fn)
            if m:
                p = os.path.join(CACHE_DIR, fn)
                keys[m.group(1)] = max(keys.get(m.group(1), 0), os.path.getmtime(p))
        order = sorted(keys.items(), key=lambda kv: kv[1])
        for key, _ in order[:-LRU_MAX_KEYS] if len(order) > LRU_MAX_KEYS else []:
            for fn in os.listdir(CACHE_DIR):
                if key in fn:
                    os.remove(os.path.join(CACHE_DIR, fn))
            META_BY_KEY.pop(key, None)
    except Exception as e:
        logger.warning(f"[AGSoft Video Compare] lru trim skipped: {e}")

_T = lambda en, ru: en + "\n---\n" + ru

#==============================================================================
# Утилиты / Utilities
#==============================================================================
def _to_frames(video, start_second=0.0, compare_duration=0.0, fallback_fps=24.0):
    """VIDEO или IMAGE → (frames, fps) с окном сравнения.
    Окно применяется официальным video.as_trimmed() (новый объект, декодится
    только окно); если API нет — fallback: срез-view после полного декода.
    У входов без собственного fps (батч IMAGE, dict без fps) секунды окна
    переводятся в кадры по fallback_fps.
    VIDEO or IMAGE to (frames, fps) with the comparison window.
    The window is applied via the official video.as_trimmed() (a new object,
    only the window is decoded); without that API — fallback: a copy-free
    slice view after the full decode. For inputs without their own fps
    (IMAGE batch, dict without fps) the window seconds are converted to
    frames via fallback_fps."""
    want_trim = (start_second > 0.0) or (compare_duration > 0.0)
    trimmed_obj = False
    if want_trim and not isinstance(video, (torch.Tensor, dict)):
        at = getattr(video, "as_trimmed", None)
        if callable(at):
            try:
                tv = at(float(start_second), float(compare_duration), False)
                if tv is not None:
                    video = tv
                    trimmed_obj = True
                else:
                    logger.warning("[AGSoft Video Compare] as_trimmed returned None: window beyond clip?")
            except Exception as e:
                logger.warning(f"[AGSoft Video Compare] as_trimmed failed: {e}")
        else:
            logger.warning("[AGSoft Video Compare] as_trimmed unsupported: full decode + slice fallback")
    if isinstance(video, torch.Tensor):
        t = video
        fps = 0.0
    else:
        t, fps = None, 0.0
        gc = getattr(video, "get_components", None)
        if callable(gc):
            try:
                comp = gc()
                if comp is not None:
                    t = getattr(comp, "images", None)
                    fr = getattr(comp, "frame_rate", None)
                    if fr is not None:
                        fps = float(fr)
            except Exception as e:
                logger.warning(f"[AGSoft Video Compare] get_components failed: {e}")
        if t is None and hasattr(video, "get_pixels"):
            try:
                t = video.get_pixels()
            except Exception as e:
                logger.warning(f"[AGSoft Video Compare] get_pixels failed: {e}")
        if t is None and isinstance(video, dict):
            t = video.get("images") or video.get("pixels")
            fr = video.get("frame_rate") or video.get("fps")
            if fr is not None:
                fps = float(fr)
    if t is None:
        raise TypeError(_T("Unsupported input type", "Неподдерживаемый тип входа") + f": {type(video).__name__}")
    if t.dim() == 5:
        t = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
    t = t.detach().cpu()
    if t.dtype != torch.float32:
        t = t.float()
    if t.numel() > 0 and t.max() > 1.5:
        t = t / 255.0
    # Нет собственного fps (IMAGE-вход) — берём предполагаемый из виджета.
    # No own fps (IMAGE input) — take the assumed one from the widget.
    if fps <= 0:
        fps = float(fallback_fps) if fallback_fps and fallback_fps > 0 else 24.0
    # Fallback-окно: срез-view без копирования / Fallback window: copy-free slice view
    if want_trim and not trimmed_obj and fps > 0 and t.shape[0] > 0:
        s0 = int(round(start_second * fps))
        s0 = max(0, min(s0, t.shape[0] - 1))
        s1 = t.shape[0] if compare_duration <= 0 else min(t.shape[0], s0 + int(round(compare_duration * fps)))
        if s1 <= s0:
            s1 = min(t.shape[0], s0 + 1)
        t = t[s0:s1]
    return t, fps

def _fp(t):
    """Дешёвый контентный отпечаток / Cheap content fingerprint."""
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    flat = t.flatten()
    step = max(1, flat.numel() // 262144)
    samp = flat[::step][:262144]
    h.update(samp.contiguous().numpy().tobytes())
    return h.hexdigest()[:8]

def _resize_frames(frames, w, h, method):
    s = frames[..., :3].movedim(-1, 1)
    s = comfy.utils.common_upscale(s, w, h, method, "disabled")
    return s.movedim(1, -1)

def _write_mp4(path, frames, fps, max_w, diff_pair=None, diff_scale=1.0):
    """Кодирование превью-потока H.264 напрямую через PyAV (core-зависимость
    ComfyUI): без рулетки плагинов imageio, из-за которой mp4 не писались.
    Encode an H.264 preview stream directly via PyAV (a ComfyUI core
    dependency): no imageio plugin roulette, which previously broke mp4."""
    if av is None:
        return False
    try:
        n = frames.shape[0] if diff_pair is None else diff_pair[0].shape[0]
        if n == 0:
            return False
        def _frame(i):
            if diff_pair is not None:
                fr = (diff_pair[0][i] - diff_pair[1][i]) * diff_scale + 0.5
            else:
                fr = frames[i]
            fr = fr.clamp(0, 1).unsqueeze(0)
            if max_w and fr.shape[2] > max_w:
                nh = max(2, int(round(fr.shape[1] * max_w / fr.shape[2])))
                fr = _resize_frames(fr, max_w, nh, "lanczos")
            if fr.shape[1] % 2:
                fr = torch.cat([fr, fr[:, -1:]], 1)
            if fr.shape[2] % 2:
                fr = torch.cat([fr, fr[:, -1:]], 2)
            return (fr[0] * 255.0).round().to(torch.uint8).numpy()
        first = _frame(0)
        h, w = first.shape[0], first.shape[1]
        container = av.open(path, "w")
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for i in range(n):
            arr = first if i == 0 else _frame(i)
            vframe = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return True
    except Exception as e:
        logger.warning(f"[AGSoft Video Compare] mp4 encode failed: {e}")
        return False

def _ensure_streams(key, a, b, fps, preview_width, diff_scale):
    """Кэш-потоки A/B/D/S; кодируются только при промахе.
    S = горизонтальная склейка ПОЛНЫХ кадров (B слева, A справа) для режима
    side_by_side: оба ролика всегда видны целиком.
    Cache streams A/B/D/S; encoded only on miss.
    S = horizontal concat of FULL frames (B left, A right) for the
    side_by_side mode: both clips always fully visible."""
    if not _ensure_cache_dir():
        return None, None, None, None
    pa = os.path.join(CACHE_DIR, f"cmp_{key}_A.mp4")
    pb = os.path.join(CACHE_DIR, f"cmp_{key}_B.mp4")
    pd = os.path.join(CACHE_DIR, f"cmp_{key}_D.mp4")
    ps = os.path.join(CACHE_DIR, f"cmp_{key}_S.mp4")
    if not os.path.isfile(pa):
        _write_mp4(pa, a, fps, preview_width)
    if not os.path.isfile(pb):
        _write_mp4(pb, b, fps, preview_width)
    if not os.path.isfile(pd):
        _write_mp4(pd, None, fps, preview_width, diff_pair=(a, b), diff_scale=diff_scale)
    if not os.path.isfile(ps):
        side = torch.cat([b, a], dim=2)
        _write_mp4(ps, side, fps, preview_width)
    _lru_trim()
    return pa, pb, pd, ps

#==============================================================================
# HTTP-эндпоинты / HTTP endpoints
#==============================================================================
def _safe_cache_path(name):
    p = os.path.realpath(os.path.join(CACHE_DIR, name))
    if not p.startswith(os.path.realpath(CACHE_DIR) + os.sep):
        return None
    return p if os.path.isfile(p) else None

@PromptServer.instance.routes.get("/agsoft/video_compare/stream")
async def agsoft_video_compare_stream(request):
    key = request.rel_url.query.get("key", "")
    side = request.rel_url.query.get("side", "A")
    if not re.fullmatch(r"[0-9a-f]{16}", key) or side not in ("A", "B", "D", "S"):
        return web.Response(status=400)
    path = _safe_cache_path(f"cmp_{key}_{side}.mp4")
    if path is None:
        logger.warning(f"[AGSoft Video Compare] stream missing: cmp_{key}_{side}.mp4")
        return web.Response(status=404)
    size = os.path.getsize(path)
    start, end = 0, size - 1
    ranged = False
    rng = request.headers.get("Range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m:
            ranged = True
            if m.group(1):
                start = int(m.group(1))
            if m.group(2):
                end = min(end, int(m.group(2)))
    headers = {"Accept-Ranges": "bytes", "Content-Type": "video/mp4",
               "Content-Length": str(end - start + 1)}
    if ranged:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    resp = web.StreamResponse(status=206 if ranged else 200, headers=headers)
    try:
        await resp.prepare(request)
    except _CLIENT_GONE:
        # Клиент отвалился до начала отдачи / Client gone before first byte
        return resp
    try:
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1024 * 1024, left))
                if not chunk:
                    break
                await resp.write(chunk)
                left -= len(chunk)
        try:
            await resp.write_eof()
        except _CLIENT_GONE:
            pass
    except _CLIENT_GONE:
        # Клиент закрыл сокет (перемотка/пребуфер) — это НЕ ошибка сервера.
        # The client closed the socket (scrub/prebuffer) — NOT a server error.
        logger.debug(f"[AGSoft Video Compare] stream {side} dropped by client (key {key})")
    return resp

@PromptServer.instance.routes.post("/agsoft/video_compare/save_frame")
async def agsoft_video_compare_save_frame(request):
    """Сохранение кадра сравнения из плеера: loopback-only, валидация PNG,
    имя из режима и метки времени, запись в папку output.
    Save a comparison frame from the player: loopback-only, PNG validation,
    name from mode + timestamp, written into the output folder."""
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
    name = f"AGSoft_VideoCompare_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.realpath(os.path.join(folder_paths.get_output_directory(), name))
    if not path.startswith(os.path.realpath(folder_paths.get_output_directory()) + os.sep):
        return web.Response(status=400)
    try:
        with open(path, "wb") as f:
            f.write(data)
    except Exception as e:
        logger.warning(f"[AGSoft Video Compare] save frame failed: {e}")
        return web.Response(status=500)
    logger.info(f"[AGSoft Video Compare] frame saved: {name}")
    return web.json_response({"ok": True, "name": name})

@PromptServer.instance.routes.get("/agsoft/video_compare/meta")
async def agsoft_video_compare_meta(request):
    node = request.rel_url.query.get("node", "")
    key = KEY_BY_NODE.get(node)
    if not key:
        return web.json_response({"key": None}, headers={"Cache-Control": "no-store"})
    meta = META_BY_KEY.get(key, {})
    meta = dict(meta)
    meta["key"] = key
    return web.json_response(meta, headers={"Cache-Control": "no-store"})

#==============================================================================
# Нода: 🎬AGSoft Video Compare
#==============================================================================
class AGSoft_VideoCompare:
    CATEGORY = "AGSoft/Video"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    # Нода-вьювер сама терминал графа: workflow без подключённых выходов
    # валиден и выполняется (фикс prompt_no_outputs).
    # The viewer node is a graph terminal itself: a workflow with no
    # connected outputs stays valid and executes (prompt_no_outputs fix).
    OUTPUT_NODE = True
    DESCRIPTION = _T(
        "Visually compares two videos (VIDEO objects or IMAGE frame batches) "
        "within an optional time window (start_second + compare_duration, "
        "0 = whole clip): the window is decoded via the official "
        "video.as_trimmed(), so long clips never load into RAM in full; for "
        "IMAGE-batch inputs (no fps metadata) the window seconds are "
        "converted to frames via image_fps. UI: synced player with a "
        "draggable wipe slider, side_by_side (server stream S with both full "
        "frames) and difference modes, plus a save button that stores the "
        "current comparison frame as PNG into the output folder. Outputs: "
        "wipe_preview, side_by_side_preview, diff_frames. Terminal node: runs "
        "even with nothing connected.",
        "Визуально сравнивает два видео (объекты VIDEO или батчи IMAGE) в "
        "опциональном временном окне (start_second + compare_duration, 0 = весь "
        "ролик): окно декодируется официальным video.as_trimmed(), поэтому "
        "длинные ролики не загружаются в RAM целиком; для входов батчем IMAGE "
        "(без метаданных fps) секунды окна переводятся в кадры через image_fps. "
        "UI: синхронный плеер с перетаскиваемым wipe-слайдером, режимы "
        "side_by_side (серверный поток S с оба полными кадрами) и difference, "
        "плюс кнопка сохранения текущего кадра сравнения в PNG в папку output. "
        "Выходы: wipe_preview, side_by_side_preview, diff_frames. Терминальная "
        "нода: выполняется даже без подключённых выходов.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("*", {"tooltip": _T(
                    "Reference side: VIDEO object or IMAGE frame batch.",
                    "Эталон: объект VIDEO или батч кадров IMAGE.")}),
                "video_b": ("*", {"tooltip": _T(
                    "Candidate side: VIDEO object or IMAGE frame batch.",
                    "Кандидат: объект VIDEO или батч кадров IMAGE.")}),
                "view_mode": (["wipe", "side_by_side", "difference"],
                              {"default": "wipe", "tooltip": _T(
                                  "UI view mode: wipe slider, server stream S with both "
                                  "full frames side by side, or the amplified difference "
                                  "stream.",
                                  "Режим просмотра UI: wipe-слайдер, серверный поток S с "
                                  "оба полными кадрами рядом или усиленный поток разницы.")}),
                "wipe_position": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1,
                                          "display": "number", "tooltip": _T(
                    "Divider position in percent: left of it shows B, right shows A. "
                    "Also used for the wipe_preview output and the saved frame.",
                    "Позиция разделителя в процентах: слева B, справа A. "
                    "Используется для выхода wipe_preview и сохраняемого кадра.")}),
                "diff_scale": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                         "display": "number", "tooltip": _T(
                    "Amplification of the visual difference: |A-B|*scale+0.5.",
                    "Усиление визуальной разницы: |A-B|*scale+0.5.")}),
                "start_second": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.1,
                                           "display": "number", "tooltip": _T(
                    "Window start in seconds for BOTH sides (e.g. 10.0 to skip the "
                    "intro). Decoded via video.as_trimmed(), so skipped part never "
                    "loads into RAM.",
                    "Начало окна в секундах для ОБЕИХ сторон (например 10.0, чтобы "
                    "пропустить вступление). Декодируется через video.as_trimmed(), "
                    "пропущенная часть в RAM не загружается.")}),
                "compare_duration": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 86400.0, "step": 0.5,
                                               "display": "number", "tooltip": _T(
                    "Window length in seconds (default 20). 0 = compare until the end "
                    "of the clip — WARNING: full decode of long clips needs huge RAM.",
                    "Длина окна в секундах (по умолчанию 20). 0 = сравнивать до конца "
                    "ролика — ВНИМАНИЕ: полный декод длинных роликов требует огромной "
                    "RAM.")}),
                "image_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.1,
                                        "display": "number", "tooltip": _T(
                    "Assumed fps for IMAGE-batch inputs (they carry no fps metadata): "
                    "converts the window seconds to frames and times the preview "
                    "streams. Ignored for VIDEO inputs, which carry their own fps.",
                    "Предполагаемый fps для входов батчем IMAGE (у них нет метаданных "
                    "fps): переводит секунды окна в кадры и задаёт тайминг превью-"
                    "потоков. Игнорируется для входов VIDEO — у них свой fps.")}),
                "resize_mode": (["to_a", "to_b", "min", "none"], {"default": "to_a", "tooltip": _T(
                    "Size mismatch handling: resize B to A, A to B, both to the "
                    "smaller, or none (error on mismatch).",
                    "Обработка несовпадения размеров: B к A, A к B, оба к "
                    "меньшему, или none (ошибка при несовпадении).")}),
                "resize_method": (["lanczos", "bicubic", "nearest"], {"default": "lanczos", "tooltip": _T(
                    "Resampling algorithm used when sizes differ.",
                    "Алгоритм ресемплинга при различии размеров.")}),
                "length_mode": (["trim_shortest", "pad_with_black", "require_equal"],
                                {"default": "trim_shortest", "tooltip": _T(
                    "Length mismatch handling: trim to the shortest, pad the "
                    "shorter with black frames, or require equal lengths.",
                    "Обработка разной длины: обрезка до короткого, добивка "
                    "короткого чёрными кадрами или требование равенства.")}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 240, "step": 1,
                                       "display": "number", "tooltip": _T(
                    "Use every N-th frame for the image outputs. Preview streams "
                    "always use all frames of the window.",
                    "Брать каждый N-й кадр для выходов IMAGE. Превью-потоки всегда "
                    "используют все кадры окна.")}),
                "preview_width": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 2,
                                          "display": "number", "tooltip": _T(
                    "Max width of cached preview streams (0 = original; stream S is "
                    "twice as wide as a single clip).",
                    "Макс. ширина кэшируемых превью-потоков (0 = оригинал; поток S "
                    "вдвое шире одного клипа).")}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("wipe_preview", "side_by_side_preview", "diff_frames")
    OUTPUT_TOOLTIPS = (
        _T("Frames split at wipe_position: left part from B, right part from A.",
           "Кадры, разрезанные по wipe_position: слева часть B, справа часть A."),
        _T("Full frames glued side by side: B on the left, A on the right "
           "(same layout as the side_by_side player mode).",
           "Полные кадры, склеенные рядом: B слева, A справа (та же раскладка, "
           "что и в режиме плеера side_by_side)."),
        _T("Amplified per-frame difference |A-B|*diff_scale+0.5 (compared frames).",
           "Усиленная покадровая разница |A-B|*diff_scale+0.5 (сравниваемые кадры)."),
    )

    def main(self, video_a, video_b, view_mode, wipe_position, diff_scale,
             start_second, compare_duration, image_fps, resize_mode, resize_method,
             length_mode, frame_step, preview_width, unique_id=None):
        a, fps_a = _to_frames(video_a, start_second, compare_duration, image_fps)
        b, fps_b = _to_frames(video_b, start_second, compare_duration, image_fps)
        if a.shape[0] == 0 or b.shape[0] == 0:
            raise ValueError(_T("Empty video on input", "Пустое видео на входе"))
        fps = fps_a if fps_a > 0 else (fps_b if fps_b > 0 else 24.0)
        # 1. Выравнивание размера / Size alignment
        if (a.shape[1], a.shape[2]) != (b.shape[1], b.shape[2]):
            if resize_mode == "none":
                raise ValueError(_T(
                    f"Size mismatch A={a.shape[2]}x{a.shape[1]} B={b.shape[2]}x{b.shape[1]}; set resize_mode",
                    f"Несовпадение размеров A={a.shape[2]}x{a.shape[1]} B={b.shape[2]}x{b.shape[1]}; задайте resize_mode"))
            if resize_mode == "to_a":
                w, h = a.shape[2], a.shape[1]
                b = _resize_frames(b, w, h, resize_method)
            elif resize_mode == "to_b":
                w, h = b.shape[2], b.shape[1]
                a = _resize_frames(a, w, h, resize_method)
            else:
                w, h = min(a.shape[2], b.shape[2]), min(a.shape[1], b.shape[1])
                a = _resize_frames(a, w, h, resize_method)
                b = _resize_frames(b, w, h, resize_method)
        # 2. Выравнивание длины / Length alignment
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
        n = a.shape[0]
        step = max(1, int(frame_step))
        # Без копии при step=1: выходы на view (фикс OOM на длинных роликах).
        # No copy at step=1: outputs run on views (OOM fix on long clips).
        if step > 1:
            idx = list(range(0, n, step))
            as_, bs_ = a[idx], b[idx]
        else:
            as_, bs_ = a, b
        # 3. Визуальные выходы / Visual outputs
        diff_frames = ((as_ - bs_) * diff_scale + 0.5).clamp(0, 1)
        split = round(as_.shape[2] * max(0, min(100, int(wipe_position))) / 100.0)
        wipe_preview = torch.cat([bs_[:, :, :split], as_[:, :, split:]], dim=2)
        side_by_side_preview = torch.cat([bs_, as_], dim=2)
        # 4. Кэш потоков / Stream cache
        opts = (f"{resize_mode}|{resize_method}|{length_mode}|{frame_step}|"
                f"{preview_width}|{round(diff_scale,2)}|{round(fps,3)}|"
                f"{round(start_second,3)}|{round(compare_duration,3)}")
        key = hashlib.sha256((_fp(a) + _fp(b) + opts).encode()).hexdigest()[:16]
        _ensure_streams(key, a, b, fps, preview_width, diff_scale)
        META_BY_KEY[key] = {"fps": fps, "frames": n, "w": a.shape[2], "h": a.shape[1],
                            "duration": n / fps if fps else 0.0}
        if unique_id is not None:
            KEY_BY_NODE[str(unique_id)] = key
            logger.info(f"[AGSoft Video Compare] node {unique_id} -> key {key}")
        return (wipe_preview, side_by_side_preview, diff_frames)

#==============================================================================
# Регистрация / Registration
#==============================================================================
NODE_CLASS_MAPPINGS = {
    "AGSoft_VideoCompare": AGSoft_VideoCompare,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_VideoCompare": "🎬AGSoft Video Compare",
}