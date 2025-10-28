# AGSoft_Loop_Base.py
# Автор: AGSoft
# Дата: 28 октября 2025 г.

import random
from decimal import Decimal, ROUND_HALF_UP

MAX_SEED = 18446744073709551615


# ───────────────────────────────
#   AGSoft_Loop_Integer
# ───────────────────────────────

class AGSoft_Loop_Integer:
    DESCRIPTION = "Generates an inclusive sequence of integers (end value is included).\nГенерирует включающую последовательность целых чисел (конечное значение включается)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_at_step": ("INT", {
                    "default": 0,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Starting number of the sequence.\nExample: start=0, end=3 → [0, 1, 2, 3]\nНачальное число последовательности.\nПример: start=0, end=3 → [0, 1, 2, 3]"
                }),
                "end_at_step": ("INT", {
                    "default": 10,
                    "min": -10000,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Ending number — this value IS included.\nExample: start=5, end=5 → [5]\nКонечное число — оно ВКЛЮЧАЕТСЯ.\nПример: start=5, end=5 → [5]"
                }),
                "jump": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Step size between numbers. Must be ≥1.\nExample: jump=2 → [0, 2, 4, 6]\nШаг между числами. Должен быть ≥1.\nПример: jump=2 → [0, 2, 4, 6]"
                })
            }
        }

    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "AGSoft/Utility"

    def generate(self, start_at_step: int, end_at_step: int, jump: int):
        if jump <= 0:
            raise ValueError("Jump must be greater than 0.")
        values = []
        current = start_at_step
        while current <= end_at_step:
            values.append(current)
            current += jump
        return (values,)

    @classmethod
    def IS_CHANGED(cls, start_at_step, end_at_step, jump):
        return f"{start_at_step}_{end_at_step}_{jump}"

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Integer"


# ───────────────────────────────
#   AGSoft_Loop_Float (с выбором precision)
# ───────────────────────────────

class AGSoft_Loop_Float:
    DESCRIPTION = "Generates an inclusive sequence of float values with selectable decimal precision (1, 2, or 3 digits).\nГенерирует включающую последовательность дробных чисел с выбором точности (1, 2 или 3 знака после запятой)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_at_step": ("FLOAT", {
                    "default": 0.000,
                    "min": -10000.000,
                    "max": 10000.000,
                    "step": 0.001,
                    "tooltip": "Starting number of the sequence.\nExample: start=0.0, end=1.0, jump=0.3 → [0.0, 0.3, 0.6, 0.9]\nНачальное число последовательности.\nПример: start=0.0, end=1.0, jump=0.3 → [0.0, 0.3, 0.6, 0.9]"
                }),
                "end_at_step": ("FLOAT", {
                    "default": 3.000,
                    "min": -10000.000,
                    "max": 10000.000,
                    "step": 0.001,
                    "tooltip": "Ending number — this value IS included.\nExample: start=1.555, end=2.000, jump=0.222 → [1.555, 1.777, 2.000] (with 3 decimals)\nКонечное число — оно ВКЛЮЧАЕТСЯ.\nПример: start=1.555, end=2.000, jump=0.222 → [1.555, 1.777, 2.000] (с 3 знаками)"
                }),
                "jump": ("FLOAT", {
                    "default": 1.000,
                    "min": 0.001,
                    "max": 1000.000,
                    "step": 0.001,
                    "tooltip": "Step size between numbers. Must be >0.\nExample: jump=0.125 → [0.000, 0.125, 0.250, ...]\nШаг между числами. Должен быть >0.\nПример: jump=0.125 → [0.000, 0.125, 0.250, ...]"
                }),
                "precision": (["1", "2", "3"], {
                    "default": "2",
                    "tooltip": "Number of decimal places: 1 → 0.1, 2 → 0.01, 3 → 0.001.\nExample: precision=1, start=0.05 → becomes 0.1\nКоличество знаков после запятой: 1 → 0.1, 2 → 0.01, 3 → 0.001.\nПример: precision=1, start=0.05 → станет 0.1"
                })
            }
        }

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "AGSoft/Utility"

    def generate(self, start_at_step: float, end_at_step: float, jump: float, precision: str):
        if jump <= 0:
            raise ValueError("Jump must be greater than 0.")
        
        prec = int(precision)
        if prec == 1:
            quantize_str = '0.1'
        elif prec == 2:
            quantize_str = '0.01'
        else:  # prec == 3
            quantize_str = '0.001'
        
        quantize_decimal = Decimal(quantize_str)
        current = Decimal(str(start_at_step))
        end = Decimal(str(end_at_step))
        step = Decimal(str(jump))
        values = []

        while current <= end:
            rounded_val = float(current.quantize(quantize_decimal, rounding=ROUND_HALF_UP))
            values.append(rounded_val)
            current += step

        return (values,)

    @classmethod
    def IS_CHANGED(cls, start_at_step, end_at_step, jump, precision):
        return f"{start_at_step}_{end_at_step}_{jump}_p{precision}"

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Float"


# ───────────────────────────────
#   AGSoft_Loop_Random_Seed
# ───────────────────────────────

class AGSoft_Loop_Random_Seed:
    DESCRIPTION = "Generates N unique random seeds for A/B testing or batch generation.\nГенерирует N уникальных случайных seed'ов для A/B-тестов или пакетной генерации."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of random seeds to generate.\nExample: count=3 → [42, 1024, 999]\nСколько случайных seed'ов сгенерировать.\nПример: count=3 → [42, 1024, 999]"
                }),
                "min_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_SEED,
                    "step": 1,
                    "tooltip": "Minimum possible seed value.\nМинимальное значение seed."
                }),
                "max_seed": ("INT", {
                    "default": 10000,
                    "min": 0,
                    "max": MAX_SEED,
                    "step": 1,
                    "tooltip": "Maximum possible seed value.\nМаксимальное значение seed."
                }),
                "seed_for_random": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": MAX_SEED,
                    "step": 1,
                    "tooltip": "Seed for the random generator itself (for reproducibility).\nSeed для генератора случайных чисел (для воспроизводимости)."
                })
            }
        }

    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "AGSoft/Utility"

    def generate(self, count: int, min_seed: int, max_seed: int, seed_for_random: int):
        if min_seed > max_seed:
            raise ValueError("min_seed must be ≤ max_seed.")
        if count <= 0:
            return ([],)
        if count > (max_seed - min_seed + 1):
            raise ValueError("Too many seeds requested for the given range.")
        rng = random.Random(seed_for_random)
        seeds = rng.sample(range(min_seed, max_seed + 1), k=count)
        return (seeds,)

    @classmethod
    def IS_CHANGED(cls, count, min_seed, max_seed, seed_for_random):
        return f"{count}_{min_seed}_{max_seed}_{seed_for_random}"

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Random Seed"


# ───────────────────────────────
#   РЕГИСТРАЦИЯ ВСЕХ НОД
# ───────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AGSoft_Loop_Integer": AGSoft_Loop_Integer,
    "AGSoft_Loop_Float": AGSoft_Loop_Float,
    "AGSoft_Loop_Random_Seed": AGSoft_Loop_Random_Seed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Loop_Integer": "AGSoft Loop Integer",
    "AGSoft_Loop_Float": "AGSoft Loop Float",
    "AGSoft_Loop_Random_Seed": "AGSoft Loop Random Seed",
}