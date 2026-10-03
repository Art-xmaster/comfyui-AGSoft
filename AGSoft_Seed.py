"""
==============================================================================
AGSoft_Seed.py
==============================================================================
Node: 🎲AGSoft Seed
Version: v10.03
Universal seed generator whose seed output is a list: thanks to
OUTPUT_IS_LIST a downstream node (e.g. a sampler) runs once per list
element, i.e. one generation per seed — a cycle of `count` runs.
INPUTS:
seed       - base seed of the cycle; widget bounds are float64-safe
             (±9007199254740991), negatives allowed. The frontend adds
             its own control before/after generate widget automatically.
             After mapping into [min_seed, max_seed] the actual seed is
             written back into this widget by the JS frontend.
mode       - how the cycle varies:
             fixed     - every iteration repeats the base;
             increment - base + iteration index;
             decrement - base - iteration index;
             randomize - iteration 0 = base, iterations > 0 are hashed
             (deterministic per base).
offset     - added to the base before range mapping.
count      - number of generations in the cycle (seeds in the list).
min_seed   - lower allowed bound for cycle seeds (may be negative).
max_seed   - upper allowed bound; if min > max the bounds are swapped.
             The base and every cycle seed are wrapped into the range
             with a negative-safe modulo.
OUTPUTS:
seed     - list of cycle seeds (INT, OUTPUT_IS_LIST); the first element
           equals the mapped base seed.
seed_str - the same cycle as a comma-separated string for filenames.
FRONTEND (see AGSoft_Seed.js):
- live "Used Seed: N" line shows the actual first seed of the cycle;
- the mapped seed is written back to the seed widget, so the widget,
  the line and output[0] always match;
- 🎲 Randomize sets a new random base; 📋 Copy Used Seed copies the seed;
- node height is hard-fixed (title + widget rows + DOM block).
------------------------------------------------------------------------------
Нода: 🎲AGSoft Seed
Версия: v10.03
Универсальный генератор сидов, у которого выход seed — список: благодаря
OUTPUT_IS_LIST даунстрим-нода (например семплер) выполняется по одному
элементу списка, то есть одна генерация на сид — цикл из `count` запусков.
ВХОДЫ:
seed       - базовый сид цикла; границы виджета float64-безопасные
             (±9007199254740991), отрицательные разрешены. Фронтенд сам
             добавляет свой control before/after generate.
             После отображения в [min_seed, max_seed] фактический сид
             пишется обратно в этот виджет JS-фронтендом.
mode       - как меняется цикл:
             fixed     - каждая итерация повторяет базу;
             increment - база + индекс итерации;
             decrement - база - индекс итерации;
             randomize - итерация 0 = база, итерации > 0 хэшируются
             (детерминированно от базы).
offset     - добавляется к базе до отображения в диапазон.
count      - число генераций в цикле (сидов в списке).
min_seed   - нижняя допустимая граница сидов цикла (может быть отрицательной).
max_seed   - верхняя граница; если min > max, границы меняются местами.
             База и каждый сид цикла заворачиваются в диапазон
             отрицательно-безопасным modulo.
ВЫХОДЫ:
seed     - список сидов цикла (INT, OUTPUT_IS_LIST); первый элемент равен
           отображённой базе.
seed_str - тот же цикл строкой через запятую для имён файлов.
ФРОНТЕНД (см. AGSoft_Seed.js):
- живая строка "Used Seed: N" показывает фактический первый сид цикла;
- отображённый сид пишется обратно в виджет seed, поэтому виджет,
  строка и output[0] всегда совпадают;
- 🎲 Randomize ставит новую случайную базу; 📋 Copy Used Seed копирует сид;
- высота ноды жёстко фиксирована (заголовок + строки виджетов + DOM-блок).
Author: AGSoft
Date: 03.10.2026
==============================================================================
"""

import os
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Seed] v10.03 loaded (Seed node: bounded seed widget, uniform cycle)")

SAFE = 9007199254740991

class AGSoftSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": -SAFE,
                    "max": SAFE,
                    "tooltip": (
                        "Base seed of the cycle (float64-safe bounds).\n"
                        "The actual seed is written back here after mapping into [min_seed, max_seed].\n"
                        "---\n"
                        "Базовый сид цикла (float64-безопасные границы).\n"
                        "После отображения в [min_seed, max_seed] сюда пишется фактический сид."
                    ),
                }),
                "mode": (["fixed", "increment", "decrement", "randomize"], {
                    "default": "randomize",
                    "tooltip": (
                        "How cycle seeds vary: fixed repeats the base,\n"
                        "increment/decrement shift by the iteration index, randomize hashes iterations > 0.\n"
                        "---\n"
                        "Как меняются сиды цикла: fixed повторяет базу,\n"
                        "increment/decrement сдвигают на индекс итерации, randomize хэширует итерации > 0."
                    ),
                }),
            },
            "optional": {
                "offset": ("INT", {
                    "default": 0,
                    "tooltip": (
                        "Value added to the base seed before range mapping.\n"
                        "---\n"
                        "Значение, добавляемое к базовому сиду до отображения в диапазон."
                    ),
                }),
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100000,
                    "tooltip": (
                        "Number of generations in the cycle (one run per seed).\n"
                        "---\n"
                        "Число генераций в цикле (один запуск на сид)."
                    ),
                }),
                "min_seed": ("INT", {
                    "default": -SAFE,
                    "min": -SAFE,
                    "max": 0,
                    "tooltip": (
                        "Lower allowed bound for cycle seeds; may be negative.\n"
                        "Bounds are auto-swapped if min > max.\n"
                        "---\n"
                        "Нижняя допустимая граница сидов цикла; может быть отрицательной.\n"
                        "Границы меняются местами, если min > max."
                    ),
                }),
                "max_seed": ("INT", {
                    "default": SAFE,
                    "min": 0,
                    "max": SAFE,
                    "tooltip": (
                        "Upper allowed bound for cycle seeds.\n"
                        "Bounds are auto-swapped if min > max.\n"
                        "---\n"
                        "Верхняя допустимая граница сидов цикла.\n"
                        "Границы меняются местами, если min > max."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "seed_str")
    OUTPUT_TOOLTIPS = (
        "List of cycle seeds: downstream runs once per seed.\n"
        "First element equals the mapped base seed.\n"
        "---\n"
        "Список сидов цикла: даунстрим выполняется по одному сиду.\n"
        "Первый элемент равен отображённой базе.",
        "The same cycle as a comma-separated string for filenames.\n"
        "---\n"
        "Тот же цикл строкой через запятую для имён файлов.",
    )
    # Seed output is a per-run list, seed_str stays scalar / Выход seed — список по запускам, seed_str остаётся скаляром
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "process"
    CATEGORY = "AGSoft/Utils"
    DESCRIPTION = (
        "Universal seed generator with per-seed loop execution.\n"
        "The seed output is a list of count seeds; downstream runs once per seed.\n"
        "mode controls the cycle: fixed / increment / decrement / randomize (hash).\n"
        "min_seed..max_seed wraps every cycle seed into the allowed range.\n"
        "seed_str gives the same cycle as a comma-separated string for filenames.\n"
        "The frontend shows and keeps the actual used seed in the seed widget.\n"
        "---\n"
        "Универсальный генератор сидов с запуском генераций по сидам.\n"
        "Выход seed — список из count сидов; даунстрим выполняется по одному сиду.\n"
        "mode управляет циклом: fixed / increment / decrement / randomize (хэш).\n"
        "min_seed..max_seed заворачивает каждый сид цикла в допустимый диапазон.\n"
        "seed_str отдаёт тот же цикл строкой через запятую для имён файлов.\n"
        "Фронтенд показывает и держит в виджете seed фактический использованный сид."
    )
    WEB_DIRECTORY = "./web"

    @staticmethod
    def _derive(base, idx, mode, lo, hi):
        # First iteration always uses the base seed: what you see is what you get
        # Первая итерация всегда использует базовый сид: что видишь, то и получишь
        if idx <= 0 or mode == "fixed":
            return base
        if mode == "randomize":
            rng = random.Random(base + idx * 0x9E3779B97F4A7C15)
            return rng.randint(lo, hi)
        step = idx if mode == "increment" else -idx
        return base + step

    @staticmethod
    def _clamp(v, lo, hi):
        # Wrap value into allowed [lo, hi] range, negatives safe / Зацикливание значения в диапазон [lo, hi], отрицательные безопасны
        if hi < lo:
            lo, hi = hi, lo
        return lo + ((v - lo) % ((hi - lo) + 1))

    def process(self, seed, mode, offset=0, count=1, min_seed=-SAFE, max_seed=SAFE):
        # Base seed with offset, clamped to allowed range / Базовый сид со смещением, зажатый в допустимый диапазон
        base = self._clamp(seed + offset, min_seed, max_seed)
        # Cycle of seeds: one seed per generation run / Цикл сидов: по одному сиду на запуск генерации
        seeds = [self._clamp(self._derive(base, i, mode, min_seed, max_seed), min_seed, max_seed) for i in range(count)]
        seed_str = ",".join(str(s) for s in seeds)
        return (seeds, seed_str)

NODE_CLASS_MAPPINGS = {
    "AGSoftSeed": AGSoftSeed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftSeed": "🎲AGSoft Seed"
}