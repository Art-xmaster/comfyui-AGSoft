"""
AGSoft_Multi_LoRA_Loader.py
Нода: 🧩 AGSoft Multi LoRA Loader

Описание / Description:
Компактный стек до 20 LoRA в одной ноде: на каждый слот — тумблер, выбор
файла, сила model и сила clip. Количество строк — кнопками "+ Add LoRA" /
"– Remove"; тумблер "Toggle All" переключает все слоты сразу (сервер
применяет тумблер каждого слота). Выбор лор — деревом папок с фильтром;
инфо-диалог показывает данные CivitAI (хэш, страница, триггеры, примеры с
промптами); локальные заметки (Name / Strength Min / Strength Max / Notes)
редактируются и хранятся по каждой LoRA. Пресеты стека сохраняются,
загружаются и удаляются по имени (user/agsoft/lora_presets).

Compact stack of up to 20 LoRAs in one node: per-slot toggle, file chooser,
model & clip strength. Row count via "+ Add LoRA" / "– Remove"; the
"Toggle All" switch flips every slot at once (the server applies each slot's
own toggle). Folder-tree LoRA chooser with filter; the info dialog shows
CivitAI data (hash, page, triggers, examples with prompts); local notes
(Name / Strength Min / Strength Max / Notes) are editable and stored per
LoRA. Stack presets are saved, loaded and deleted by name
(user/agsoft/lora_presets).

Возможности / Features:
⚡ До 20 слотов; строка: тумблер, лора, ◄ сила model ►, ◄ сила clip ►, ℹ.
  Up to 20 slots; row: toggle, lora, ◄ model strength ►, ◄ clip strength ►, ℹ.
⚡ Горизонтальные степеры силы (◄/► всегда видны, клик ±0.05, Shift ±0.01)
  и ручной ввод. Horizontal strength steppers + manual typing.
⚡ Контекстное меню строки: Show Info / Toggle / Move Up / Move Down / Remove.
  Row context menu: Show Info / Toggle / Move Up / Move Down / Remove.
⚡ Инфо-диалог CivitAI: fetch по SHA256, ссылка на страницу, триггеры
  (копирование кликом), примеры (img + video, 📝 промпт с chips
  steps/cfg/sampler), редактируемые локальные заметки.
  CivitAI info dialog: SHA256 fetch, page link, trigger words (click to
  copy), examples (img + video, 📝 prompt with steps/cfg/sampler chips),
  editable local notes.
⚡ Strength Min/Max из заметок ограничивают степеры строки.
  Strength Min/Max from notes clamp the row steppers.
⚡ Список лор обновляется при каждом открытии chooser'а (эндпоинт
  /agsoft/lora_list) — без перезагрузки браузера.
  The LoRA list refreshes on every chooser open (endpoint /agsoft/lora_list)
  — no browser restart needed.
⚡ Пресеты стека: именованные снимки слотов, подтверждение перезаписи и
  удаления, живой список при открытии; имена лор возвращаются к родному
  написанию списка, поэтому пресет переживает перезагрузку.
  Stack presets: named slot snapshots, overwrite/delete confirmation, live
  list on open; lora names are mapped back to the list's native spelling so
  presets survive a restart.
⚡ Контролы следуют цвету ноды (CSS-переменные); один DOM-контейнер без
  невидимых зон перехвата мыши. Controls follow the node color (CSS vars);
  single DOM container with no invisible input-blocking zones.
⚡ Безопасное применение: clone + add_patches, входные model/clip не
  мутируются; нулевые силы пропускаются. Headless/API работает без JS.
  Safe apply: clone + add_patches, incoming model/clip never mutated; zero
  strengths skipped. Works headless/API without JS.

Автор / Author: AGSoft
Дата / Date: 27.11.2026
"""

import hashlib
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
from aiohttp import web
from server import PromptServer

logger = logging.getLogger(__name__)
# logger.info( "[AGSoft Multi LoRA Loader] v1.17 loaded (20 slots, Toggle All, CivitAI info dialog, live lora list, stack presets)")

MAX_LORA_SLOTS = 20
LORA_NONE = "none"
STRENGTH_MIN = -10.0
STRENGTH_MAX = 10.0
CIVITAI_WEB = "https://civitai.com"
META_FIELDS = ("name", "strength_min", "strength_max", "notes", "hash")
EDITABLE_META_FIELDS = ("name", "strength_min", "strength_max", "notes")
HASH_CACHE_LIMIT = 64
HASH_CHUNK_SIZE = 1024 * 1024
ZERO_STRENGTH_EPS = 1e-9
PRESET_NAME_MAX = 64
PRESET_VERSION = 1
PRESET_SUFFIX = ".json"
_URL_SAFE = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"
)

_HASH_MEM_CACHE: Dict[Tuple[str, float, int], str] = {}
_META_STORE: Optional[Dict[str, Dict[str, str]]] = None


def _sanitize_lora_name(name: Any) -> str:
    """Normalize a relative LoRA name and reject traversal (EN) /
    Нормализует относительное имя LoRA и отсекает выход из корня (RU)."""
    clean = str(name or "").replace("\\", "/").strip()
    if not clean or os.path.isabs(clean) or ".." in clean.split("/"):
        return ""
    return clean


def _lora_name_key(name: Any) -> str:
    """Separator-insensitive comparison key for LoRA names (EN) /
    Ключ сравнения имён LoRA без учёта разделителей пути (RU)."""
    return str(name or "").replace("\\", "/").strip()


def _lora_native_name(name: Any) -> str:
    """Map a stored name back to the exact folder_paths spelling (EN) /
    Возвращает имя в точном написании списка folder_paths (RU)."""
    key = _lora_name_key(name)
    if not key:
        return ""
    if key == LORA_NONE:
        return LORA_NONE
    for native in _get_lora_options():
        if _lora_name_key(native) == key:
            return native
    return _sanitize_lora_name(name)


def _resolve_lora_path(name: str) -> Optional[str]:
    """Resolve a LoRA name strictly inside the registered loras roots (EN) /
    Резолвит имя LoRA строго внутри зарегистрированных корней loras (RU)."""
    clean = _lora_native_name(name)
    if not clean:
        return None
    try:
        path = folder_paths.get_full_path("loras", clean)
        if not path:
            return None
        real = os.path.realpath(path)
        roots = folder_paths.get_folder_paths("loras")
    except Exception as e:
        logger.warning(
            f"[AGSoft Multi LoRA Loader] path resolve failed for {clean}: {e}"
        )
        return None
    for root in roots:
        root_real = os.path.realpath(root)
        if real == root_real or real.startswith(root_real + os.sep):
            return real
    logger.warning(
        f"[AGSoft Multi LoRA Loader] path outside loras roots rejected: {clean}"
    )
    return None


def _is_loopback(request: web.Request) -> bool:
    """Mutating endpoints accept local callers only (EN) /
    Мутирующие эндпоинты принимают только локальные вызовы (RU)."""
    return (request.remote or "") in {"127.0.0.1", "::1", "localhost"}


def _url_quote(value: str) -> str:
    """Percent-encode a query value (EN) /
    Percent-кодирует значение query-параметра (RU)."""
    parts: List[str] = []
    for byte in value.encode("utf-8"):
        char = chr(byte)
        parts.append(char if char in _URL_SAFE else f"%{byte:02X}")
    return "".join(parts)


def _meta_path() -> str:
    """Per-user metadata file location (EN) /
    Расположение пользовательского файла заметок (RU)."""
    try:
        base = folder_paths.get_user_directory()
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] user dir unavailable: {e}")
        base = folder_paths.get_temp_directory()
    directory = os.path.join(base, "agsoft")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"[AGSoft Multi LoRA Loader] meta dir create failed: {e}")
    return os.path.join(directory, "lora_meta.json")


def _empty_meta() -> Dict[str, str]:
    return {key: "" for key in META_FIELDS}


def _load_meta_store() -> Dict[str, Dict[str, str]]:
    """Load the notes store once and keep it as a module cache (EN) /
    Загружает хранилище заметок один раз и держит его как кэш модуля (RU)."""
    global _META_STORE
    if _META_STORE is not None:
        return _META_STORE
    path = _meta_path()
    if not os.path.isfile(path):
        _META_STORE = {}
        return _META_STORE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _META_STORE = data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] meta load failed: {e}")
        _META_STORE = {}
    return _META_STORE


def _save_meta_store() -> None:
    store = _load_meta_store()
    try:
        with open(_meta_path(), "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] meta save failed: {e}")


def _get_meta(name: str) -> Dict[str, str]:
    store = _load_meta_store()
    meta = store.get(name)
    if not isinstance(meta, dict):
        meta = _empty_meta()
        store[name] = meta
    else:
        for key in META_FIELDS:
            meta.setdefault(key, "")
    return meta


def _meta_view(meta: Dict[str, str]) -> Dict[str, str]:
    return {key: str(meta.get(key, "")) for key in META_FIELDS}


def _get_lora_options() -> List[str]:
    """LoRA dropdown values, empty slot first (EN) /
    Значения выпадающего списка LoRA, пустой слот первым (RU)."""
    options = [LORA_NONE]
    seen = {LORA_NONE}
    try:
        discovered = folder_paths.get_filename_list("loras")
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] lora list failed: {e}")
        discovered = []
    for item in discovered:
        if item and item not in seen:
            seen.add(item)
            options.append(item)
    return options


def _slot_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "off", "no"}
    return bool(value)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha256_of(path: str) -> str:
    """SHA256 with an mtime/size cache key so unchanged files skip rehash (EN) /
    SHA256 с ключом кэша по mtime/size — неизменённые файлы не хэшируются снова (RU)."""
    stat = os.stat(path)
    key = (path, stat.st_mtime, stat.st_size)
    cached = _HASH_MEM_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    result = digest.hexdigest().upper()
    if len(_HASH_MEM_CACHE) >= HASH_CACHE_LIMIT:
        _HASH_MEM_CACHE.clear()
    _HASH_MEM_CACHE[key] = result
    return result


def _sanitize_preset_name(name: Any) -> str:
    """Keep unicode letters/digits, space, underscore, hyphen; cap length (EN) /
    Оставляет unicode-буквы/цифры, пробел, подчёркивание, дефис; режет длину (RU)."""
    clean = "".join(
        ch for ch in str(name or "").strip()
        if ch.isalnum() or ch in " _-"
    ).strip()
    return clean[:PRESET_NAME_MAX]


def _presets_dir() -> str:
    """Per-user preset root, survives pack updates (EN) /
    Пользовательский корень пресетов, переживает обновление пака (RU)."""
    try:
        base = folder_paths.get_user_directory()
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] user dir unavailable: {e}")
        base = folder_paths.get_temp_directory()
    directory = os.path.join(base, "agsoft", "lora_presets")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"[AGSoft Multi LoRA Loader] presets dir create failed: {e}")
    return directory


def _preset_path(name: str) -> Optional[str]:
    """Resolve a preset file strictly inside the presets root (EN) /
    Резолвит файл пресета строго внутри корня пресетов (RU)."""
    clean = _sanitize_preset_name(name)
    if not clean:
        return None
    root = os.path.realpath(_presets_dir())
    candidate = os.path.realpath(os.path.join(root, clean + PRESET_SUFFIX))
    try:
        if os.path.commonpath([root, candidate]) != root:
            return None
    except ValueError as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] preset path rejected: {e}")
        return None
    return candidate


def _default_preset_slot() -> Dict[str, Any]:
    return {
        "enabled": True,
        "lora": LORA_NONE,
        "model_strength": 1.0,
        "clip_strength": 1.0,
    }


def _clamp_strength(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number):
        return 1.0
    return max(STRENGTH_MIN, min(STRENGTH_MAX, number))


def _normalize_preset(data: Any) -> Dict[str, Any]:
    """Validate and clip a preset payload to the node's slot contract (EN) /
    Валидирует и обрезает payload пресета под контракт слотов ноды (RU)."""
    if not isinstance(data, dict):
        data = {}
    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list):
        raw_slots = []
    slots: List[Dict[str, Any]] = []
    for item in raw_slots[:MAX_LORA_SLOTS]:
        if not isinstance(item, dict):
            continue
        lora = _lora_native_name(item.get("lora", LORA_NONE))
        slots.append(
            {
                "enabled": _slot_enabled(item.get("enabled", True)),
                "lora": lora or LORA_NONE,
                "model_strength": _clamp_strength(item.get("model_strength", 1.0)),
                "clip_strength": _clamp_strength(item.get("clip_strength", 1.0)),
            }
        )
    try:
        count = int(data.get("active_loras", len(slots)))
    except (TypeError, ValueError):
        count = len(slots)
    count = max(0, min(count, MAX_LORA_SLOTS))
    slots = slots[:count]
    while len(slots) < count:
        slots.append(_default_preset_slot())
    return {"version": PRESET_VERSION, "active_loras": count, "slots": slots}


@PromptServer.instance.routes.get("/agsoft/lora_meta")
async def agsoft_lora_meta(request: web.Request) -> web.Response:
    name = _sanitize_lora_name(request.query.get("name", ""))
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    path = _resolve_lora_path(name)
    meta = _get_meta(name)
    return web.json_response(
        {"ok": True, "file": path or "", "meta": _meta_view(meta)}
    )


@PromptServer.instance.routes.post("/agsoft/lora_meta_save")
async def agsoft_lora_meta_save(request: web.Request) -> web.Response:
    if not _is_loopback(request):
        return web.json_response({"ok": False, "error": "local only"}, status=403)
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] bad meta json: {e}")
        return web.json_response({"ok": False, "error": "bad json"})
    if not isinstance(data, dict):
        return web.json_response({"ok": False, "error": "bad json"})
    name = _sanitize_lora_name(data.get("name", ""))
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    meta = _get_meta(name)
    for key in EDITABLE_META_FIELDS:
        if key in data:
            meta[key] = str(data[key])
    _save_meta_store()
    return web.json_response({"ok": True, "meta": _meta_view(meta)})


@PromptServer.instance.routes.get("/agsoft/lora_info")
async def agsoft_lora_info(request: web.Request) -> web.Response:
    """Return SHA256 + local metadata; the frontend fetches CivitAI by hash (EN) /
    Отдаёт SHA256 и локальные заметки; CivitAI запрашивает фронтенд по хэшу (RU)."""
    name = _sanitize_lora_name(request.query.get("name", ""))
    if not name:
        return web.json_response({"ok": False, "error": "no lora name"})
    path = _resolve_lora_path(name)
    if not path or not os.path.isfile(path):
        return web.json_response(
            {"ok": False, "error": f"LoRA not found: {name}"}
        )
    try:
        sha = _sha256_of(path)
    except Exception as e:
        logger.error(f"[AGSoft Multi LoRA Loader] hash failed for {name}: {e}")
        return web.json_response({"ok": False, "error": f"hash failed: {e}"})
    meta = _get_meta(name)
    meta["hash"] = sha
    _save_meta_store()
    return web.json_response(
        {
            "ok": True,
            "hash": sha,
            "file": path,
            "meta": _meta_view(meta),
            "search_url": f"{CIVITAI_WEB}/search/models?query={_url_quote(name)}",
        }
    )


@PromptServer.instance.routes.get("/agsoft/lora_list")
async def agsoft_lora_list(request: web.Request) -> web.Response:
    """Live LoRA filename list; folder mtime invalidates the folder cache (EN) /
    Живой список файлов LoRA; mtime папки инвалидирует кэш списка (RU)."""
    try:
        items = folder_paths.get_filename_list("loras")
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] lora list failed: {e}")
        items = []
    return web.json_response({"ok": True, "list": [x for x in items if x]})


@PromptServer.instance.routes.get("/agsoft/lora_presets")
async def agsoft_lora_presets(request: web.Request) -> web.Response:
    """Preset names in case-insensitive order (EN) /
    Имена пресетов в порядке без учёта регистра (RU)."""
    try:
        entries = os.listdir(_presets_dir())
    except OSError as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] presets list failed: {e}")
        entries = []
    names = [
        item[: -len(PRESET_SUFFIX)]
        for item in entries
        if item.endswith(PRESET_SUFFIX)
    ]
    names.sort(key=str.lower)
    return web.json_response({"ok": True, "list": names})


@PromptServer.instance.routes.get("/agsoft/lora_preset")
async def agsoft_lora_preset(request: web.Request) -> web.Response:
    name = _sanitize_preset_name(request.query.get("name", ""))
    path = _preset_path(name) if name else None
    if not path or not os.path.isfile(path):
        return web.json_response({"ok": False, "error": "preset not found"})
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] preset load failed: {e}")
        return web.json_response({"ok": False, "error": "preset read failed"})
    return web.json_response(
        {"ok": True, "name": name, "preset": _normalize_preset(data)}
    )


@PromptServer.instance.routes.post("/agsoft/lora_preset_save")
async def agsoft_lora_preset_save(request: web.Request) -> web.Response:
    if not _is_loopback(request):
        return web.json_response({"ok": False, "error": "local only"}, status=403)
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] bad preset json: {e}")
        return web.json_response({"ok": False, "error": "bad json"})
    if not isinstance(data, dict):
        return web.json_response({"ok": False, "error": "bad json"})
    name = _sanitize_preset_name(data.get("name", ""))
    path = _preset_path(name) if name else None
    if not path:
        return web.json_response({"ok": False, "error": "bad preset name"})
    preset = _normalize_preset(data)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(preset, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[AGSoft Multi LoRA Loader] preset save failed: {e}")
        return web.json_response({"ok": False, "error": "preset write failed"})
    return web.json_response({"ok": True, "name": name})


@PromptServer.instance.routes.post("/agsoft/lora_preset_delete")
async def agsoft_lora_preset_delete(request: web.Request) -> web.Response:
    if not _is_loopback(request):
        return web.json_response({"ok": False, "error": "local only"}, status=403)
    try:
        data = await request.json()
    except Exception as e:
        logger.warning(f"[AGSoft Multi LoRA Loader] bad preset json: {e}")
        return web.json_response({"ok": False, "error": "bad json"})
    if not isinstance(data, dict):
        return web.json_response({"ok": False, "error": "bad json"})
    name = _sanitize_preset_name(data.get("name", ""))
    path = _preset_path(name) if name else None
    if not path or not os.path.isfile(path):
        return web.json_response({"ok": False, "error": "preset not found"})
    try:
        os.remove(path)
    except OSError as e:
        logger.error(f"[AGSoft Multi LoRA Loader] preset delete failed: {e}")
        return web.json_response({"ok": False, "error": "preset delete failed"})
    return web.json_response({"ok": True, "name": name})


class AGSoftMultiLoraLoader:
    """Compact stack of up to 20 LoRAs with per-slot controls (EN) /
    Компактный стек до 20 LoRA с поконтрольным управлением по слотам (RU)."""

    WEB_DIRECTORY = "./web"
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
        "The LoRA list is fetched live on every chooser open — newly downloaded files appear without "
        "a browser restart.\n"
        "Stack presets: save / load / delete named snapshots of the active slots, with overwrite and "
        "delete confirmation; presets live in the user directory and survive pack updates.\n"
        "---\n"
        "🧩 AGSoft Multi LoRA Loader.\n"
        "До 20 LoRA в одной компактной ноде. На слот: тумблер, выбор LoRA, сила model, сила clip, "
        "кнопка инфо. Количество строк — + Add LoRA / – Remove; Toggle All переключает все слоты.\n"
        "Выбор лор деревом папок с фильтром; горизонтальные степеры силы; контекстное меню строки "
        "(Show Info / Toggle / Move Up / Move Down / Remove).\n"
        "Инфо-диалог: данные CivitAI по SHA256 (ссылка на страницу, триггеры, примеры-изображения/видео "
        "с промптами) плюс редактируемые локальные заметки (Name / Strength Min / Strength Max / Notes), "
        "хранящиеся по каждой LoRA; Strength Min/Max ограничивают степеры строки.\n"
        "Список лор запрашивается вживую при каждом открытии chooser'а — свеже скачанные файлы видны "
        "без перезагрузки браузера.\n"
        "Пресеты стека: сохранение / загрузка / удаление именованных снимков активных слотов с "
        "подтверждением перезаписи и удаления; пресеты лежат в пользовательской папке и переживают "
        "обновление пака.\n"
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        required: Dict[str, Any] = {
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
                        "Bulk switch: flips every slot toggle at once. "
                        "The server applies each slot's own toggle.\n"
                        "---\n"
                        "Массовый переключатель: включает/выключает все слоты сразу. "
                        "Сервер применяет тумблер каждого слота."
                    ),
                },
            ),
        }
        lora_options = _get_lora_options()
        for i in range(1, MAX_LORA_SLOTS + 1):
            required[f"enabled_{i}"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": (
                        f"Slot {i}: enable/disable this LoRA.\n"
                        "---\n"
                        f"Слот {i}: включить/выключить эту LoRA."
                    ),
                },
            )
            required[f"lora_{i}"] = (
                lora_options,
                {
                    "default": LORA_NONE,
                    "tooltip": (
                        f"Slot {i}: LoRA file from models/loras. 'none' = empty slot.\n"
                        "---\n"
                        f"Слот {i}: файл LoRA из models/loras. 'none' = пустой слот."
                    ),
                },
            )
            required[f"model_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": STRENGTH_MIN,
                    "max": STRENGTH_MAX,
                    "step": 0.01,
                    "tooltip": (
                        f"Slot {i}: strength for the MODEL (unet/dit) part of the LoRA.\n"
                        "---\n"
                        f"Слот {i}: сила для MODEL (unet/dit) части LoRA."
                    ),
                },
            )
            required[f"clip_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": STRENGTH_MIN,
                    "max": STRENGTH_MAX,
                    "step": 0.01,
                    "tooltip": (
                        f"Slot {i}: strength for the CLIP (text encoder) part of the LoRA.\n"
                        "---\n"
                        f"Слот {i}: сила для CLIP (текстовый энкодер) части LoRA."
                    ),
                },
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

    @classmethod
    def VALIDATE_INPUTS(
        cls, active_loras: int = 2, **kwargs: Any
    ) -> Union[bool, str]:
        try:
            count = int(active_loras)
        except (TypeError, ValueError):
            return "active_loras must be a number."
        if count < 0 or count > MAX_LORA_SLOTS:
            return f"active_loras must be between 0 and {MAX_LORA_SLOTS}."
        available = {_lora_name_key(x) for x in _get_lora_options()}
        for i in range(1, count + 1):
            if not _slot_enabled(kwargs.get(f"enabled_{i}", True)):
                continue
            name = str(kwargs.get(f"lora_{i}", LORA_NONE))
            if name and name != LORA_NONE and _lora_name_key(name) not in available:
                return f"LoRA slot {i} is enabled but not installed: {name}"
            for key, label in (
                (f"model_strength_{i}", "model strength"),
                (f"clip_strength_{i}", "clip strength"),
            ):
                v = kwargs.get(key, 1.0)
                if not _finite(v):
                    return f"LoRA slot {i} {label} must be a finite number."
                if not (STRENGTH_MIN <= float(v) <= STRENGTH_MAX):
                    return (
                        f"LoRA slot {i} {label} must be between "
                        f"{STRENGTH_MIN:g} and {STRENGTH_MAX:g}."
                    )
        return True

    def _load_lora_sd(self, lora_name: str) -> Dict[str, Any]:
        """Load a LoRA state dict from a contained path (EN) /
        Загружает state dict LoRA из пути внутри разрешённого корня (RU)."""
        lora_path = _resolve_lora_path(lora_name)
        if not lora_path:
            raise FileNotFoundError(
                f"[AGSoft Multi LoRA Loader] LoRA not found in models/loras: {lora_name}"
            )
        return comfy.utils.load_torch_file(lora_path, safe_load=True)

    def load_multi_lora(
        self,
        model: Any,
        active_loras: int,
        toggle_all: bool,
        clip: Any = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        # toggle_all is a UI bulk switch; each slot's own toggle is authoritative.
        try:
            count = max(0, min(int(active_loras), MAX_LORA_SLOTS))
        except (TypeError, ValueError):
            count = 0
        if count <= 0:
            logger.info("[AGSoft Multi LoRA Loader] No active slots — passthrough.")
            return (model, clip)
        current_model = model
        current_clip = clip
        applied: List[str] = []
        for i in range(1, count + 1):
            if not _slot_enabled(kwargs.get(f"enabled_{i}", True)):
                continue
            lora_name = str(kwargs.get(f"lora_{i}", LORA_NONE))
            if not lora_name or lora_name == LORA_NONE:
                continue
            ms = float(kwargs.get(f"model_strength_{i}", 1.0))
            cs = float(kwargs.get(f"clip_strength_{i}", 1.0))
            lora_sd = self._load_lora_sd(lora_name)
            key_map: Dict[str, Any] = {}
            if current_model is not None:
                key_map = comfy.lora.model_lora_keys_unet(
                    current_model.model, key_map
                )
            if current_clip is not None:
                key_map = comfy.lora.model_lora_keys_clip(
                    current_clip.cond_stage_model, key_map
                )
            loaded = comfy.lora.load_lora(
                comfy.lora_convert.convert_lora(lora_sd), key_map
            )
            if current_model is not None and abs(ms) > ZERO_STRENGTH_EPS:
                patched = current_model.clone()
                patched.add_patches(loaded, ms)
                current_model = patched
            if current_clip is not None and abs(cs) > ZERO_STRENGTH_EPS:
                patched_clip = current_clip.clone()
                patched_clip.add_patches(loaded, cs)
                current_clip = patched_clip
            applied.append(f"#{i} {lora_name} (m={ms:g}, c={cs:g})")
        if applied:
            logger.info(
                f"[AGSoft Multi LoRA Loader] Applied {len(applied)} LoRA(s): "
                + "; ".join(applied)
            )
        else:
            logger.info(
                "[AGSoft Multi LoRA Loader] No enabled LoRA slots — passthrough."
            )
        return (current_model, current_clip)


NODE_CLASS_MAPPINGS = {"AGSoftMultiLoraLoader": AGSoftMultiLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMultiLoraLoader": "🧩 AGSoft Multi LoRA Loader"
}