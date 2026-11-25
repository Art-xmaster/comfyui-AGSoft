"""
==============================================================================
AGSoft_SetGet.py
==============================================================================
Нода: 🧩 AGSoft Set / 🧩 AGSoft Get
Описание / Description:
    Беспроводная пара: Set принимает значение любого типа под именем,
    Get отдаёт это значение в любой точке графа без провода.
    ---
    Wireless pair: Set stores an any-type value under a name,
    Get delivers that value anywhere in the graph without a wire.

Возможности / Features:
    ⚡ Любой тип: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING, AUDIO, VIDEO…
      Any type: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING, AUDIO, VIDEO…
    ⚡ Один Set питает сколько угодно Get, значения идут ссылками, без копий
      One Set feeds any number of Gets, values travel by reference, zero-copy
    ⚡ Связь разрешается интерфейсом: кэш и частичный прогон как с проводом
      The link is resolved by the UI: caching and partial runs act like a wire
    ⚡ Жёсткая проверка: дубли имён и разрывы пары блокируют запуск
      Strict checks: duplicate names and broken pairs block the run
    ⚡ Set можно ставить в разрыв провода: выход value транзитный
      Set can sit inline: the value output is a passthrough

Архитектура / Architecture:
    Python-классы — регистрационные заглушки: пара виртуальная и в промт
    не попадает, ядро резолвит выход Get прямо в источник входа Set.
    ---
    The Python classes are registration stubs: the pair is virtual and never
    enters the prompt, the core resolves a Get output straight to the Set
    input source.

Автор / Author: AGSoft
Дата / Date: 26.11.2026
==============================================================================
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft SetGet] v11.26 loaded (wireless any-type Set/Get pair)")

TIP_VALUE_IN = (
    "Any value: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING, AUDIO, VIDEO, etc. "
    "--- "
    "Любое значение: MODEL, CLIP, VAE, IMAGE, LATENT, CONDITIONING, AUDIO, VIDEO и т.д."
)
TIP_NAME_SET = (
    "Unique channel name. Duplicates in one graph block the run. "
    "--- "
    "Уникальное имя канала. Дубликаты в одном графе блокируют запуск."
)
TIP_NAME_GET = (
    "Pick the Set channel name to read from. "
    "--- "
    "Выберите имя канала Set, из которого читать значение."
)
TIP_VALUE_OUT = (
    "The wired value, passed by reference. "
    "--- "
    "Подключённое значение, передаётся ссылкой."
)


class AGSoftSet:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "name": ("STRING", {"default": "", "tooltip": TIP_NAME_SET}),
        }
        optional = {
            "value": ("*", {"tooltip": TIP_VALUE_IN}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = (TIP_VALUE_OUT,)
    FUNCTION = "passthrough"
    CATEGORY = "AGSoft/Utils"
    DESCRIPTION = (
        "Stores any connected value under a unique name for wireless Gets. "
        "--- "
        "Сохраняет любое подключённое значение под уникальным именем для Get."
    )
    WEB_DIRECTORY = "./web"

    def passthrough(self, name: str = "", value=None) -> tuple:
        # The pair is resolved by the UI; this body only serves registration.
        # Пара разрешается интерфейсом; это тело нужно только для регистрации.
        return (value,)


class AGSoftGet:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "name": ([" "], {"tooltip": TIP_NAME_GET}),
        }
        return {"required": required}

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = (TIP_VALUE_OUT,)
    FUNCTION = "passthrough"
    CATEGORY = "AGSoft/Utils"
    DESCRIPTION = (
        "Reads a named Set channel anywhere in the graph without a wire. "
        "--- "
        "Читает именованный канал Set в любой точке графа без провода."
    )
    WEB_DIRECTORY = "./web"

    def passthrough(self, name: str = " ") -> tuple:
        # Virtual node: the core substitutes the Set source before execution.
        # Виртуальная нода: ядро подставляет источник Set до исполнения.
        return (None,)


NODE_CLASS_MAPPINGS = {
    "AGSoftSet": AGSoftSet,
    "AGSoftGet": AGSoftGet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftSet": "🧩AGSoft Set",
    "AGSoftGet": "🧩AGSoft Get",
}