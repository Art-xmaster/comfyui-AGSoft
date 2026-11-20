"""
==============================================================================
AGSoft_Pipe.py
==============================================================================
Нода: 🧩 AGSoft Pipe Master / 🔀 AGSoft Pipe Link
Описание / Description:
    Универсальная шина параметров. Master собирает любые подключённые значения
    в один поток AGS_PIPE, Link разбирает этот поток обратно на отдельные
    выходы. Один провод вместо десятка: модель, CLIP, VAE, кондиционирование,
    латент, изображения, аудио и скалярные параметры едут по одной шине.
    ---
    Universal parameter bus. Master packs any connected values into a single
    AGS_PIPE stream, Link unpacks that stream back into separate outputs.
    One wire instead of ten: model, CLIP, VAE, conditioning, latent, images,
    audio and scalar parameters travel along a single bus.

Возможности / Features:
    ⚡ До 32 входов Master: новый слот появляется сам при подключении предыдущего
      Up to 32 Master inputs: a new slot appears once the previous one is used
    ⚡ До 32 выходов Link: количество задаётся счётчиком, запись — на каждый выход
      Up to 32 Link outputs: count is set by a counter, an entry per output
    ⚡ Переименование входов и шаблоны имён с автонумерацией повторов
      Input renaming and name presets with automatic numbering of duplicates
    ⚡ Стабильные ID записей: переименование не рвёт выбор, сделанный в Link
      Stable entry IDs: renaming never breaks the selection made in Link
    ⚡ Слияние шин: вход bus_in подмешивает вышестоящую шину в текущую
      Bus merging: bus_in mixes an upstream bus into the current one
    ⚡ Транзит шины: выход bus у Link позволяет ставить цепочку Link подряд
      Bus passthrough: the bus output of Link allows chaining Link nodes
    ⚡ Автовыбор записи по типу принимающего входа при первом подключении
      Entry auto-selection by the receiving input type on first connection
    ⚡ Zero-copy: значения передаются ссылками, тензоры не копируются
      Zero-copy: values travel by reference, tensors are never copied

Архитектура / Architecture:
    Payload: {"v": 1, "entries": [{"id", "key", "name", "type", "value"}, ...]}
    - Ключ входа in_N неизменяем: переименование меняет только подпись и мету,
      поэтому индексы слотов, ключи prompt и валидация остаются целыми.
      The in_N key is immutable: renaming changes only the label and metadata,
      so slot indices, prompt keys and validation stay intact.
    - ID записи: L:N — локальный вход Master, U:... — запись вышестоящей шины.
      Entry ID: L:N is a local Master input, U:... comes from an upstream bus.
    - Сокет bus закреплён на индексе 0 у обеих нод: рост и усадка хвоста никогда
      не сдвигают индексы уже подключённых линков.
      The bus socket is pinned to index 0 on both nodes: growing or trimming
      the tail never shifts indices of already connected links.
    - В workflow сериализуется только мета (имена, типы, выборы) в скрытом
      виджете ags_meta; значения живут исключительно в памяти выполнения.
      Only metadata (names, types, selections) is serialized into the workflow
      via the hidden ags_meta widget; values live in execution memory only.
    - Отсутствующая запись даёт None на выходе, а не падение графа.
      A missing entry yields None on the output instead of breaking the graph.

Совместимость / Compatibility:
    Работа идёт через модель графа и DOM-виджеты, без canvas-хуков, поэтому
    поведение одинаково в классическом рендере и в Nodes 2.0.
    Everything goes through the graph model and DOM widgets, without canvas
    hooks, so behaviour is identical in the classic renderer and in Nodes 2.0.

Автор / Author: AGSoft
Дата / Date: 20.11.2026
==============================================================================
"""

import json
import logging

import torch

from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, VAE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Pipe] v11.20 loaded (universal parameter bus: Master + Link)")

MAX_SLOTS = 32
MAX_OUT = 32
BUS_TYPE = "AGS_PIPE"

TIP_BUS_IN = (
    "Optional upstream bus to merge into this Master. "
    "--- "
    "Опциональная вышестоящая шина для слияния в этот Master."
)
TIP_SLOT_IN = (
    "Any value: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING, etc. "
    "--- "
    "Любое значение: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING и т.д."
)
TIP_META = (
    "Internal metadata (names, types, selections). Managed by the UI. "
    "--- "
    "Служебные метаданные (имена, типы, выборы). Управляются интерфейсом."
)
TIP_BUS_LINK = (
    "Bus stream from a Master or another Link. "
    "--- "
    "Поток шины от Master или другого Link."
)
TIP_COUNT = (
    "Number of data outputs (1-32). Shrink removes only unconnected tail slots. "
    "--- "
    "Число выходов данных (1-32). Уменьшение снимает только неподключённые хвостовые слоты."
)
TIP_BUS_OUT = (
    "Passthrough bus for chaining another Link. "
    "--- "
    "Транзитная шина для подключения следующего Link."
)
TIP_DATA_OUT = (
    "Selected bus entry value; None if the entry is missing. "
    "--- "
    "Значение выбранной записи шины; None, если запись отсутствует."
)


def _load_meta(raw: str) -> dict:
    # Meta travels through a hidden widget; tolerate hand-edited workflows.
    # Мета едет через скрытый виджет; допускаем ручные правки workflow.
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _detect_type(value) -> str:
    # Fallback type detection when the UI has not recorded the source type.
    # Определение типа, если UI не записал тип источника.
    if isinstance(value, ModelPatcher):
        return "MODEL"
    if isinstance(value, CLIP):
        return "CLIP"
    if isinstance(value, VAE):
        return "VAE"
    if isinstance(value, dict) and "samples" in value:
        return "LATENT"
    if isinstance(value, torch.Tensor):
        return "IMAGE"
    if isinstance(value, list):
        return "CONDITIONING"
    return "*"


def _unique_name(name: str, seen: set) -> str:
    base = name or "entry"
    candidate = base
    n = 2
    while candidate in seen:
        candidate = f"{base}_{n}"
        n += 1
    seen.add(candidate)
    return candidate


class AGSoftPipeMaster:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "bus_in": (BUS_TYPE, {"tooltip": TIP_BUS_IN}),
        }
        for i in range(1, MAX_SLOTS + 1):
            optional[f"in_{i}"] = ("*", {"tooltip": TIP_SLOT_IN})
        optional["ags_meta"] = ("STRING", {
            "default": "{}",
            "multiline": True,
            "tooltip": TIP_META,
        })
        return {"required": {}, "optional": optional}

    RETURN_TYPES = (BUS_TYPE,)
    RETURN_NAMES = ("bus",)
    OUTPUT_TOOLTIPS = (TIP_BUS_OUT,)
    FUNCTION = "collect"
    CATEGORY = "AGSoft/Utils"
    DESCRIPTION = (
        "Collects any connected values into a single bus stream. "
        "--- "
        "Собирает любые подключённые значения в единый поток шины."
    )
    WEB_DIRECTORY = "./web"

    def collect(self, bus_in=None, ags_meta: str = "{}", **kwargs) -> tuple:
        meta = _load_meta(ags_meta)
        names = meta.get("names") or {}
        types = meta.get("types") or {}
        seen = set()
        entries = []
        if isinstance(bus_in, dict):
            for e in bus_in.get("entries") or []:
                if not isinstance(e, dict):
                    continue
                eid = str(e.get("id", ""))
                if not eid.startswith("U:"):
                    eid = "U:" + eid
                entries.append({
                    "id": eid,
                    "key": str(e.get("key", "")),
                    "name": _unique_name(str(e.get("name", "entry")), seen),
                    "type": str(e.get("type", "*")),
                    "value": e.get("value"),
                })
        for i in range(1, MAX_SLOTS + 1):
            key = f"in_{i}"
            if key not in kwargs or kwargs[key] is None:
                continue
            value = kwargs[key]
            entries.append({
                "id": f"L:{i}",
                "key": key,
                "name": _unique_name(str(names.get(key, key)), seen),
                "type": str(types.get(key)) if types.get(key) else _detect_type(value),
                "value": value,
            })
        return ({"v": 1, "entries": entries},)


class AGSoftPipeLink:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "bus": (BUS_TYPE, {"tooltip": TIP_BUS_LINK}),
            "count": ("INT", {
                "default": 1,
                "min": 1,
                "max": MAX_OUT,
                "step": 1,
                "tooltip": TIP_COUNT,
            }),
        }
        optional = {
            "ags_meta": ("STRING", {
                "default": "{}",
                "multiline": True,
                "tooltip": TIP_META,
            }),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = (BUS_TYPE,) + ("*",) * MAX_OUT
    RETURN_NAMES = ("bus",) + tuple(f"out_{i}" for i in range(1, MAX_OUT + 1))
    OUTPUT_TOOLTIPS = (TIP_BUS_OUT,) + (TIP_DATA_OUT,) * MAX_OUT
    FUNCTION = "dispatch"
    CATEGORY = "AGSoft/Utils"
    DESCRIPTION = (
        "Extracts selected bus entries into separate outputs and passes the bus on. "
        "--- "
        "Извлекает выбранные записи шины в отдельные выходы и передаёт шину дальше."
    )
    WEB_DIRECTORY = "./web"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types, **kwargs) -> bool:
        # Selections arrive through metadata, not through widget lists.
        # Выборы приходят через метаданные, а не через списки виджетов.
        return True

    def dispatch(self, bus, count: int, ags_meta: str = "{}", **kwargs) -> tuple:
        meta = _load_meta(ags_meta)
        entries = bus.get("entries") or [] if isinstance(bus, dict) else []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e, dict)}
        by_name = {str(e.get("name")): e for e in entries if isinstance(e, dict)}
        count = max(1, min(MAX_OUT, int(count)))
        values = []
        for i in range(1, MAX_OUT + 1):
            if i > count:
                values.append(None)
                continue
            rec = meta.get(f"out_{i}") or {}
            entry = by_id.get(str(rec.get("id"))) or by_name.get(str(rec.get("name")))
            if entry is None:
                logger.warning("[AGSoft Pipe] Link out_%s: entry not found on bus", i)
                values.append(None)
            else:
                values.append(entry.get("value"))
        return (bus,) + tuple(values)


NODE_CLASS_MAPPINGS = {
    "AGSoftPipeMaster": AGSoftPipeMaster,
    "AGSoftPipeLink": AGSoftPipeLink,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftPipeMaster": "🧩AGSoft Pipe Master",
    "AGSoftPipeLink": "🔀AGSoft Pipe Link",
}