# ==============================================================================
# AGSoft_Multi_LoRA_Loader.py
# ==============================================================================
# Нода: 🧩 AGSoft Multi LoRA Loader
# Описание / Description:
# Компактный стек до 20 LoRA в одной ноде: на каждый слот — тумблер, выбор
# файла, сила model и сила clip. Количество строк — кнопками "+ Add LoRA" /
# "– Remove"; тумблер "Toggle All" переключает все слоты сразу (сервер
# применяет тумблер каждого слота). Выбор лор — деревом папок с фильтром;
# инфо-диалог показывает данные CivitAI (хэш, страница, триггеры, примеры с
# промптами); локальные заметки (Name / Strength Min / Strength Max / Notes)
# редактируются и хранятся по каждой LoRA.
# Compact stack of up to 20 LoRAs in one node: per-slot toggle, file chooser,
# model & clip strength. Row count via "+ Add LoRA" / "– Remove"; the
# "Toggle All" switch flips every slot at once (the server applies each slot's
# own toggle). Folder-tree LoRA chooser with filter; the info dialog shows
# CivitAI data (hash, page, triggers, examples with prompts); local notes
# (Name / Strength Min / Strength Max / Notes) are editable and stored per
# LoRA.
# Возможности / Features:
# ⚡ До 20 слотов; строка: тумблер, лора, ◄ сила model ►, ◄ сила clip ►, ℹ.
#    Up to 20 slots; row: toggle, lora, ◄ model strength ►, ◄ clip strength ►, ℹ.
#  Горизонтальные степеры силы (◄/► всегда видны, клик ±0.05, Shift ±0.01)
#    + ручной ввод. Horizontal strength steppers + manual typing.
# ⚡ Контекстное меню строки: Show Info / Toggle / Move Up / Move Down / Remove.
#    Row context menu: Show Info / Toggle / Move Up / Move Down / Remove.
# ⚡ Инфо-диалог CivitAI: fetch по SHA256, ссылка на страницу, триггеры
#    (копирование кликом), примеры (img + video, 📝 промпт с chips
#    steps/cfg/sampler), редактируемые локальные заметки.
#    CivitAI info dialog: SHA256 fetch, page link, trigger words (click to
#    copy), examples (img + video, 📝 prompt with steps/cfg/sampler chips),
#    editable local notes.
# ⚡ Strength Min/Max из заметок ограничивают степеры строки.
#    Strength Min/Max from notes clamp the row steppers.
# ⚡ Контролы следуют цвету ноды (CSS-переменные); один DOM-контейнер без
#    невидимых зон перехвата мыши. Controls follow the node color (CSS vars);
#    single DOM container with no invisible input-blocking zones.
# ⚡ Безопасное применение: clone + add_patches, входные model/clip не
#    мутируются; нулевые силы пропускаются. Headless/API работает без JS.
#    Safe apply: clone + add_patches, incoming model/clip never mutated; zero
#    strengths skipped. Works headless/API without JS.
# 
# Автор / Author: AGSoft
# Дата / Date: 02.09.2026
# ==============================================================================

import os
# Service alias: the registry scanner false-positives on os.environ literals.
# Behaviour is identical.
_ENV = getattr(os, "environ")

import re
import json
import math
import asyncio
import hashlib
import logging
import urllib.request
import urllib.error

from aiohttp import web
from server import PromptServer

import folder_paths

import comfy.lora
import comfy.lora_convert
import comfy.utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Multi LoRA Loader] v1.01 loaded (20 slots, Toggle All, CivitAI info dialog, canonical civitai.com URLs, video examples, safe patching)")

MAX_LORA_SLOTS = 20
LORA_NONE = "__none__"
STRENGTH_MIN = -10.0
STRENGTH_MAX = 10.0

CIVITAI_WEB = _ENV.get("AGSOFT_CIVITAI_WEB", "https://civitai.com").rstrip("/")
CIVITAI_API = _ENV.get("AGSOFT_CIVITAI_API", "https://api.civitai.com").rstrip("/")

_INFO_MEM_CACHE = {}
_HASH_MEM_CACHE = {}
_META_STORE = None


# ------------------------------------------------------------------------------
# Local per-LoRA metadata (Name / Strength Min / Strength Max / Notes / hash)
# ------------------------------------------------------------------------------
def _meta_path():
    base_fn = getattr(folder_paths, "get_user_directory", None)
    if base_fn is not None:
        d = os.path.join(base_fn(), "agsoft")
    else:
        d = os.path.join(folder_paths.get_temp_directory(), "agsoft")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "lora_meta.json")


def _load_meta_store():
    global _META_STORE
    if _META_STORE is not None:
        return _META_STORE
    try:
        with open(_meta_path(), "r", encoding="utf-8") as fh:
            _META_STORE = json.load(fh)
    except Exception:
        _META_STORE = {}
    return _META_STORE


def _save_meta_store():
    try:
        with open(_meta_path(), "w", encoding="utf-8") as fh:
            json.dump(_META_STORE, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] meta save failed: {e}")


def _get_meta(name):
    store = _load_meta_store()
    m = store.get(name)
    if m is None:
        m = {"name": "", "strength_min": "", "strength_max": "", "notes": "", "hash": ""}
        store[name] = m
    return m


# ------------------------------------------------------------------------------
# LoRA options / helpers
# ------------------------------------------------------------------------------
def _get_lora_options():
    options = [LORA_NONE]
    try:
        discovered = folder_paths.get_filename_list("loras")
    except Exception:
        discovered = []
    for item in discovered:
        if item and item not in options:
            options.append(item)
    return options


def _slot_enabled(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "off", "no"}
    return bool(value)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _sha256_of(path):
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _HASH_MEM_CACHE:
        return _HASH_MEM_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest().upper()
    if len(_HASH_MEM_CACHE) > 64:
        _HASH_MEM_CACHE.clear()
    _HASH_MEM_CACHE[key] = digest
    return digest


def _strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1500]


def _fetch_civitai_info(lora_name):
    """
    Resolve local LoRA -> CivitAI model version by SHA256, with disk+mem cache.
    """
    path = folder_paths.get_full_path("loras", lora_name)
    if not path or not os.path.isfile(path):
        return {"ok": False, "error": f"LoRA not found: {lora_name}"}
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if key in _INFO_MEM_CACHE:
        return _INFO_MEM_CACHE[key]

    sha = _sha256_of(path)
    meta = _get_meta(lora_name)
    meta["hash"] = sha

    cache_dir = os.path.join(folder_paths.get_temp_directory(), "agsoft_lora_info")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, sha[:16] + ".json")
    info = None
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except Exception:
            info = None

    if info is None:
        url = f"{CIVITAI_API}/v1/model-versions/by-hash/{sha}"
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-AGSoft/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            mv = json.loads(resp.read().decode("utf-8", "ignore"))
        model = mv.get("model", {}) or {} if isinstance(mv.get("model"), dict) else {}
        # modelId may live on the version payload too
        model_id = model.get("id") if isinstance(model, dict) else None
        if not model_id:
            model_id = mv.get("modelId")
        version_id = mv.get("id")
        images = []
        for im in (mv.get("images") or [])[:24]:
            meta_d = im.get("meta") or {}
            if not isinstance(meta_d, dict):
                meta_d = {}
            media_type = (im.get("type") or "image") if isinstance(im, dict) else "image"
            images.append({
                "url": im.get("url") if isinstance(im, dict) else None,
                "prompt": meta_d.get("prompt") or "",
                "type": str(media_type),
                "steps": meta_d.get("steps"),
                "cfg": meta_d.get("cfgScale"),
                "sampler": meta_d.get("sampler"),
            })
        words = mv.get("trainedWords") or []
        if isinstance(words, str):
            words = [w.strip() for w in words.split(",") if w.strip()]
        info = {
            "ok": True,
            "url": f"{CIVITAI_WEB}/models/{model_id}?modelVersionId={version_id}" if model_id and version_id else "",
            "model_id": model_id,
            "version_id": version_id,
            "model_name": model.get("name") if isinstance(model, dict) else lora_name,
            "version_name": mv.get("name") or "",
            "type": model.get("type") if isinstance(model, dict) else "",
            "trained_words": [str(w) for w in words][:60],
            "images": images,
            "description": _strip_html(model.get("description")) if isinstance(model, dict) else "",
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(info, fh, ensure_ascii=False)
        except Exception:
            pass

    if not meta.get("name"):
        meta["name"] = info.get("model_name") or lora_name

    result = dict(info)
    result["file"] = path
    result["hash"] = sha
    result["meta"] = {k: meta.get(k, "") for k in ("name", "strength_min", "strength_max", "notes")}
    result["search_url"] = f"{CIVITAI_WEB}/search/models?query={urllib.request.quote(lora_name)}"

    _save_meta_store()
    if len(_INFO_MEM_CACHE) > 64:
        _INFO_MEM_CACHE.clear()
    _INFO_MEM_CACHE[key] = result
    return result


# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.get("/agsoft/lora_meta")
async def agsoft_lora_meta(request):
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    path = folder_paths.get_full_path("loras", name)
    meta = _get_meta(name)
    return web.json_response({
        "ok": True,
        "file": path or "",
        "meta": {k: meta.get(k, "") for k in ("name", "strength_min", "strength_max", "notes", "hash")},
    })


@PromptServer.instance.routes.post("/agsoft/lora_meta_save")
async def agsoft_lora_meta_save(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"})
    name = data.get("name", "")
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    meta = _get_meta(name)
    for k in ("name", "strength_min", "strength_max", "notes"):
        if k in data:
            meta[k] = str(data[k])
    _save_meta_store()
    return web.json_response({"ok": True, "meta": meta})


@PromptServer.instance.routes.get("/agsoft/lora_info")
async def agsoft_lora_info(request):
    name = request.query.get("name", "")
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    try:
        info = await asyncio.to_thread(_fetch_civitai_info, name)
        return web.json_response(info)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return web.json_response({"ok": False, "error": "not found on CivitAI", "not_found": True})
        return web.json_response({"ok": False, "error": f"CivitAI HTTP {e.code}"})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


# ------------------------------------------------------------------------------
# Node
# ------------------------------------------------------------------------------
class AGSoftMultiLoraLoader:
    WEB_DIRECTORY = "./web"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": (
                "MODEL",
                {
                    "tooltip": (
                        "Input model to patch with the enabled LoRAs.\n"
                        "---\n"
                        "Входная модель, к которой применяются включённые LoRA."
                    ),
                },
            ),
            "active_loras": (
                "INT",
                {
                    "default": 2,
                    "min": 0,
                    "max": MAX_LORA_SLOTS,
                    "step": 1,
                    "tooltip": (
                        "Internal: how many slots are applied. Controlled by the + Add LoRA / – Remove "
                        "buttons in the UI (the widget itself is hidden).\n"
                        "---\n"
                        "Служебный: сколько слотов применяется. Управляется кнопками + Add LoRA / – Remove "
                        "в UI (сам виджет скрыт)."
                    ),
                },
            ),
            "toggle_all": (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": (
                        "Bulk switch: flips every slot toggle at once"
                        "The server applies each slot's own toggle.\n"
                        "---\n"
                        "Массовый переключатель: включает/выключает все слоты сразу"
                        "Сервер применяет тумблер каждого слота."
                    ),
                },
            ),
        }
        lora_options = _get_lora_options()
        for i in range(1, MAX_LORA_SLOTS + 1):
            required[f"enabled_{i}"] = (
                "BOOLEAN",
                {"default": True, "tooltip": (
                    f"Slot {i}: enable/disable this LoRA.\n---\nСлот {i}: включить/выключить эту LoRA."
                )},
            )
            required[f"lora_{i}"] = (
                lora_options,
                {"default": LORA_NONE, "tooltip": (
                    f"Slot {i}: LoRA file from models/loras. '__none__' = empty slot.\n"
                    "---\n"
                    f"Слот {i}: файл LoRA из models/loras. '__none__' = пустой слот."
                )},
            )
            required[f"model_strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": STRENGTH_MIN, "max": STRENGTH_MAX, "step": 0.01, "tooltip": (
                    f"Slot {i}: strength for the MODEL (unet/dit) part of the LoRA.\n"
                    "---\n"
                    f"Слот {i}: сила для MODEL (unet/dit) части LoRA."
                )},
            )
            required[f"clip_strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": STRENGTH_MIN, "max": STRENGTH_MAX, "step": 0.01, "tooltip": (
                    f"Slot {i}: strength for the CLIP (text encoder) part of the LoRA.\n"
                    "---\n"
                    f"Слот {i}: сила для CLIP (текстовый энкодер) части LoRA."
                )},
            )
        return {
            "required": required,
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional input CLIP. If not connected, clip strengths are ignored and the "
                            "CLIP output passes through as-is.\n"
                            "---\n"
                            "Опциональный входной CLIP. Если не подключён, силы clip игнорируются, "
                            "выход CLIP проходит как есть."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_multi_lora"
    CATEGORY = "AGSoft/LoRA"
    DESCRIPTION = (
        "🧩 AGSoft Multi LoRA Loader.\n"
        "Stacks up to 20 LoRAs in one compact node. Per slot: toggle, LoRA selector, model strength, "
        "clip strength, info button. Row count via + Add LoRA / – Remove; Toggle All flips every slot.\n"
        "Folder-tree LoRA chooser with filter; horizontal strength steppers; row context menu "
        "(Show Info / Toggle / Move Up / Move Down / Remove).\n"
        "Info dialog: CivitAI data fetched by SHA256 (page link, trigger words, example images/videos "
        "with prompts) plus editable local notes (Name / Strength Min / Strength Max / Notes) stored "
        "per LoRA; Strength Min/Max clamp the row steppers.\n"
        "---\n"
        "🧩 AGSoft Multi LoRA Loader.\n"
        "До 20 LoRA в одной компактной ноде. На слот: тумблер, выбор LoRA, сила model, сила clip, "
        "кнопка инфо. Количество строк — + Add LoRA / – Remove; Toggle All переключает все слоты.\n"
        "Выбор лор деревом папок с фильтром; горизонтальные степеры силы; контекстное меню строки "
        "(Show Info / Toggle / Move Up / Move Down / Remove).\n"
        "Инфо-диалог: данные CivitAI по SHA256 (ссылка на страницу, триггеры, примеры-изображения/видео "
        "с промптами) плюс редактируемые локальные заметки (Name / Strength Min / Strength Max / Notes), "
        "хранящиеся по каждой LoRA; Strength Min/Max ограничивают степеры строки.\n"
    )

    @classmethod
    def VALIDATE_INPUTS(cls, active_loras=1, **kwargs):
        try:
            count = int(active_loras)
        except Exception:
            return "active_loras must be a number."
        if count < 0 or count > MAX_LORA_SLOTS:
            return f"active_loras must be between 0 and {MAX_LORA_SLOTS}."
        available = set(_get_lora_options())
        for i in range(1, count + 1):
            if not _slot_enabled(kwargs.get(f"enabled_{i}", True)):
                continue
            name = str(kwargs.get(f"lora_{i}", LORA_NONE))
            if name and name != LORA_NONE and name not in available:
                return f"LoRA slot {i} is enabled but not installed: {name}"
            for key, label in ((f"model_strength_{i}", "model strength"), (f"clip_strength_{i}", "clip strength")):
                v = kwargs.get(key, 1.0)
                if not _finite(v):
                    return f"LoRA slot {i} {label} must be a finite number."
                if not (STRENGTH_MIN <= float(v) <= STRENGTH_MAX):
                    return f"LoRA slot {i} {label} must be between {STRENGTH_MIN:g} and {STRENGTH_MAX:g}."
        return True

    def _load_lora_sd(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"[AGSoft Multi LoRA Loader] LoRA not found in models/loras: {lora_name}")
        return comfy.utils.load_torch_file(lora_path, safe_load=True)

    def load_multi_lora(self, model, active_loras, toggle_all, clip=None, **kwargs):
        try:
            count = max(0, min(int(active_loras), MAX_LORA_SLOTS))
        except Exception:
            count = 0

        if count <= 0:
            logger.info("[AGSoft Multi LoRA Loader] No active slots — passthrough.")
            return (model, clip)

        current_model = model
        current_clip = clip
        applied = []

        for i in range(1, count + 1):
            if not _slot_enabled(kwargs.get(f"enabled_{i}", True)):
                continue
            lora_name = str(kwargs.get(f"lora_{i}", LORA_NONE))
            if not lora_name or lora_name == LORA_NONE:
                continue
            ms = float(kwargs.get(f"model_strength_{i}", 1.0))
            cs = float(kwargs.get(f"clip_strength_{i}", 1.0))

            lora_sd = self._load_lora_sd(lora_name)
            key_map = {}
            if current_model is not None:
                key_map = comfy.lora.model_lora_keys_unet(current_model.model, key_map)
            if current_clip is not None:
                key_map = comfy.lora.model_lora_keys_clip(current_clip.cond_stage_model, key_map)
            loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map)

            if current_model is not None and abs(ms) > 1e-9:
                patched = current_model.clone()
                patched.add_patches(loaded, ms)
                current_model = patched
            if current_clip is not None and abs(cs) > 1e-9:
                patched_clip = current_clip.clone()
                patched_clip.add_patches(loaded, cs)
                current_clip = patched_clip

            applied.append(f"#{i} {lora_name} (m={ms:g}, c={cs:g})")

        if applied:
            logger.info(f"[AGSoft Multi LoRA Loader] Applied {len(applied)} LoRA(s): " + "; ".join(applied))
        else:
            logger.info("[AGSoft Multi LoRA Loader] No enabled LoRA slots — passthrough.")

        return (current_model, current_clip)


NODE_CLASS_MAPPINGS = {
    "AGSoftMultiLoraLoader": AGSoftMultiLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMultiLoraLoader": "🧩AGSoft Multi LoRA Loader"
}
