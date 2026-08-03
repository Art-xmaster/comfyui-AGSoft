# ==============================================================================
# AGSoft_MiniMax_Base.py
# ==============================================================================
# Нода: 🎬 AGSoft MiniMax Base
# Автор: AGSoft
# Дата: 04.08.2026 г.
#
# Описание / Description:
# Нода AGSoft MiniMax Base — расчет параметров видео для MiniMax H3.
# FPS фиксирован 24, все размеры кратны 32,
# количество кадров — последовательность 5, 22, 39, 56, 73... (17*N + 5).
# ==============================================================================

import math


# ========================================================================
# Утилиты
# ========================================================================
def fit_to_multiple(value: int, multiple: int = 32) -> int:
    """
    Подгоняет значение к ближайшему большему числу, кратному multiple.

    Примеры:
     100 → 128 (при multiple=32)
     128 → 128 (уже кратно)
     33 → 64
    """
    return ((value + multiple - 1) // multiple) * multiple


def fit_length_to_17n5(value: int) -> int:
    """
    Подгоняет число под последовательность: 5, 22, 39, 56, 73...
    Формула: max(5, v) + (5 - max(5, v) % 17) % 17.

    Примеры:
     5 → 5
     100 → 107
     240 → 243
    """
    v = max(5, int(value))
    return v + (5 - v % 17) % 17


# ========================================================================
# Пресеты — ОДИН плоский список
# ========================================================================
PRESET_LIST = [
    # 1:1 Квадрат
    "512×512 (1:1)",
    "576×576 (1:1)",
    "640×640 (1:1)",
    "704×704 (1:1)",
    "768×768 (1:1)",
    "832×832 (1:1)",
    "896×896 (1:1)",
    "960×960 (1:1)",
    "1024×1024 (1:1)",
    "1280×1280 (1:1)",
    # 3:2 Фото
    "480×320 (3:2)",
    "576×384 (3:2)",
    "672×448 (3:2)",
    "768×512 (3:2)",
    "864×576 (3:2)",
    "960×640 (3:2)",
    "1056×704 (3:2)",
    "1152×768 (3:2)",
    "1248×832 (3:2)",
    "1536×1024 (3:2)",
    # 4:3 Стандарт
    "512×384 (4:3)",
    "640×480 (4:3)",
    "768×576 (4:3)",
    "896×672 (4:3)",
    "1024×768 (4:3)",
    "1152×864 (4:3)",
    "1280×960 (4:3)",
    "1408×1056 (4:3)",
    "1536×1152 (4:3)",
    "2048×1536 (4:3)",
    # 16:9 Кино/ТВ
    "512×288 (16:9)",
    "640×352 (16:9)",
    "768×448 (16:9)",
    "896×512 (16:9)",
    "1024×576 (16:9)",
    "1152×640 (16:9)",
    "1280×704 (16:9)",
    "1536×864 (16:9)",
    "1920×1088 (16:9)",
    "2048×1152 (16:9)",
]

PRESET_MAP = {
    "512×512 (1:1)": (512, 512),
    "576×576 (1:1)": (576, 576),
    "640×640 (1:1)": (640, 640),
    "704×704 (1:1)": (704, 704),
    "768×768 (1:1)": (768, 768),
    "832×832 (1:1)": (832, 832),
    "896×896 (1:1)": (896, 896),
    "960×960 (1:1)": (960, 960),
    "1024×1024 (1:1)": (1024, 1024),
    "1280×1280 (1:1)": (1280, 1280),
    "480×320 (3:2)": (480, 320),
    "576×384 (3:2)": (576, 384),
    "672×448 (3:2)": (672, 448),
    "768×512 (3:2)": (768, 512),
    "864×576 (3:2)": (864, 576),
    "960×640 (3:2)": (960, 640),
    "1056×704 (3:2)": (1056, 704),
    "1152×768 (3:2)": (1152, 768),
    "1248×832 (3:2)": (1248, 832),
    "1536×1024 (3:2)": (1536, 1024),
    "512×384 (4:3)": (512, 384),
    "640×480 (4:3)": (640, 480),
    "768×576 (4:3)": (768, 576),
    "896×672 (4:3)": (896, 672),
    "1024×768 (4:3)": (1024, 768),
    "1152×864 (4:3)": (1152, 864),
    "1280×960 (4:3)": (1280, 960),
    "1408×1056 (4:3)": (1408, 1056),
    "1536×1152 (4:3)": (1536, 1152),
    "2048×1536 (4:3)": (2048, 1536),
    "512×288 (16:9)": (512, 288),
    "640×352 (16:9)": (640, 352),
    "768×448 (16:9)": (768, 448),
    "896×512 (16:9)": (896, 512),
    "1024×576 (16:9)": (1024, 576),
    "1152×640 (16:9)": (1152, 640),
    "1280×704 (16:9)": (1280, 704),
    "1536×864 (16:9)": (1536, 864),
    "1920×1088 (16:9)": (1920, 1088),
    "2048×1152 (16:9)": (2048, 1152),
}

# ========================================================================
# Соотношения сторон для режима Megapixels
# ========================================================================
ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]


# ========================================================================
# Нода
# ========================================================================
class AGSoft_MiniMax_Base:

    CATEGORY = "AGSoft/Video"
    FUNCTION = "main"

    DESCRIPTION = (
        "AGSoft MiniMax Base — universal video parameter calculator for MiniMax H3.\n"
        "Automatically normalizes all values to the required formats:\n"
        "• Frame width and height — always multiples of 32.\n"
        "• Frame count — follows the sequence: 5, 22, 39, 56, 73... (17×N+5), min 5.\n"
        "• Frame rate (FPS) — fixed at 24 (integer and float outputs).\n\n"
        "Supports three size modes: Preset, Custom, Megapixels.\n"
        "Invert orientation available for Preset and Custom modes.\n"
        "Frame count can be calculated from seconds (2–15 sec) or entered manually.\n\n"
        "Use this node before connecting to MiniMax H3 sampler nodes.\n\n"
        "═══════════════════════════════════════\n\n"
        "AGSoft MiniMax Base — универсальный калькулятор параметров для MiniMax H3.\n"
        "Автоматически нормализует все значения под требования формата:\n"
        "• Ширина и высота кадра — всегда кратны 32.\n"
        "• Количество кадров — последовательность: 5, 22, 39, 56, 73... (17×N+5), мин. 5.\n"
        "• Частота кадров (FPS) — фиксирована на 24 (целый и дробный выходы).\n\n"
        "Три режима выбора размера: Preset, Custom, Megapixels.\n"
        "Инверсия сторон работает в Preset и Custom.\n"
        "Количество кадров — авторасчёт из секунд (2–15 сек) или ручной ввод.\n\n"
        "Используйте эту ноду перед подключением к сэмплеру MiniMax H3."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ========================================================
                # mode
                # ========================================================
                "mode": (
                    ["Preset", "Custom", "Megapixels"],
                    {
                        "default": "Preset",
                        "tooltip": (
                            "Frame size selection mode.\n\n"
                            "• Preset — choose from 40 predefined sizes grouped by aspect ratio.\n"
                            "  Aspect ratios: 1:1 (Square), 3:2 (Photo), 4:3 (Standard), 16:9 (Cinema).\n"
                            "  All sizes are already multiples of 32. Invert orientation supported.\n\n"
                            "• Custom — manually enter width and height in pixels.\n"
                            "  Values are automatically rounded UP to the nearest multiple of 32.\n"
                            "  Example: 100×100 → 128×128. Invert orientation supported.\n\n"
                            "• Megapixels — specify target resolution in megapixels + aspect ratio.\n"
                            "  Width and height calculated automatically, rounded to multiples of 32.\n"
                            "  Invert orientation NOT applied (aspect ratio defines orientation).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Режим выбора размера кадра.\n\n"
                            "• Preset — выбор из 40 готовых размеров по соотношениям сторон.\n"
                            "  Форматы: 1:1 (Квадрат), 3:2 (Фото), 4:3 (Стандарт), 16:9 (Кино).\n"
                            "  Все размеры уже кратны 32. Доступна инверсия сторон.\n\n"
                            "• Custom — ручной ввод ширины и высоты в пикселях.\n"
                            "  Значения автоматически округляются ВВЕРХ до кратности 32.\n"
                            "  Пример: 100×100 → 128×128. Доступна инверсия сторон.\n\n"
                            "• Megapixels — задаёте мегапиксели + соотношение сторон.\n"
                            "  Размеры рассчитываются автоматически, кратность 32 соблюдается.\n"
                            "  Инверсия НЕ применяется (соотношение уже задаёт ориентацию)."
                        )
                    }
                ),
                # ========================================================
                # preset
                # ========================================================
                "preset": (
                    PRESET_LIST,
                    {
                        "default": "1280×704 (16:9)",
                        "tooltip": (
                            "Predefined frame size with aspect ratio label.\n\n"
                            "Format: WIDTH×HEIGHT (ASPECT_RATIO)\n"
                            "Examples:\n"
                            "  • 1024×1024 (1:1) — square, 1 megapixel\n"
                            "  • 1280×704 (16:9) — HD landscape, cinema format\n"
                            "  • 704×1280 (9:16) — vertical video (after invert)\n\n"
                            "All sizes are multiples of 32 — safe for MiniMax H3.\n"
                            "Grouped by aspect ratio for easy navigation:\n"
                            "  1:1 → 3:2 → 4:3 → 16:9\n\n"
                            "Use 'invert_orientation' to swap width↔height if needed.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Готовый размер кадра с меткой соотношения сторон.\n\n"
                            "Формат: ШИРИНА×ВЫСОТА (СООТНОШЕНИЕ)\n"
                            "Примеры:\n"
                            "  • 1024×1024 (1:1) — квадрат, 1 мегапиксель\n"
                            "  • 1280×704 (16:9) — HD горизонтальное, киноформат\n"
                            "  • 704×1280 (9:16) — вертикальное видео (после инверсии)\n\n"
                            "Все размеры кратны 32 — безопасно для MiniMax H3.\n"
                            "Сгруппированы по соотношениям для удобной навигации:\n"
                            "  1:1 → 3:2 → 4:3 → 16:9\n\n"
                            "Используйте 'invert_orientation' для смены ширины и высоты."
                        )
                    }
                ),
                # ========================================================
                # invert_orientation
                # ========================================================
                "invert_orientation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Swap width and height values.\n\n"
                            "Use cases:\n"
                            "  • Quickly create vertical video from landscape preset.\n"
                            "    Example: 1280×704 → 704×1280 (9:16 vertical)\n"
                            "  • Swap custom dimensions without re-entering values.\n\n"
                            "Works in: Preset and Custom modes.\n"
                            "Does NOT work in: Megapixels mode (aspect ratio defines orientation).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Поменять ширину и высоту местами.\n\n"
                            "Сценарии использования:\n"
                            "  • Быстрое создание вертикального видео из горизонтального пресета.\n"
                            "    Пример: 1280×704 → 704×1280 (вертикальное 9:16)\n"
                            "  • Смена размеров без повторного ввода значений.\n\n"
                            "Работает в: Preset и Custom.\n"
                            "НЕ работает в: Megapixels (соотношение уже задаёт ориентацию)."
                        )
                    }
                ),
                # ========================================================
                # custom_width
                # ========================================================
                "custom_width": (
                    "INT",
                    {
                        "default": 864,
                        "min": 64,
                        "max": 8192,
                        "step": 32,
                        "display": "number",
                        "tooltip": (
                            "Custom frame width in pixels (used in Custom mode).\n\n"
                            "• Automatically rounded UP to nearest multiple of 32.\n"
                            "  Example: 100 → 128, 1900 → 1920, 1920 → 1920.\n"
                            "• Minimum: 64 pixels.\n"
                            "• Maximum: 8192 pixels.\n"
                            "• Step: 32 (for convenience).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Своя ширина кадра в пикселях (режим Custom).\n\n"
                            "• Автоматически округляется ВВЕРХ до кратности 32.\n"
                            "  Пример: 100 → 128, 1900 → 1920, 1920 → 1920.\n"
                            "• Минимум: 64 пикселя.\n"
                            "• Максимум: 8192 пикселя.\n"
                            "• Шаг: 32 (для удобства)."
                        )
                    }
                ),
                # ========================================================
                # custom_height
                # ========================================================
                "custom_height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 64,
                        "max": 8192,
                        "step": 32,
                        "display": "number",
                        "tooltip": (
                            "Custom frame height in pixels (used in Custom mode).\n\n"
                            "• Automatically rounded UP to nearest multiple of 32.\n"
                            "  Example: 100 → 128, 1080 → 1088, 1088 → 1088.\n"
                            "• Minimum: 64 pixels.\n"
                            "• Maximum: 8192 pixels.\n"
                            "• Step: 32 (for convenience).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Своя высота кадра в пикселях (режим Custom).\n\n"
                            "• Автоматически округляется ВВЕРХ до кратности 32.\n"
                            "  Пример: 100 → 128, 1080 → 1088, 1088 → 1088.\n"
                            "• Минимум: 64 пикселя.\n"
                            "• Максимум: 8192 пикселя.\n"
                            "• Шаг: 32 (для удобства)."
                        )
                    }
                ),
                # ========================================================
                # megapixels_value
                # ========================================================
                "megapixels_value": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": (
                            "Target resolution in megapixels (MP) for Megapixels mode.\n\n"
                            "• 1.0 MP = 1,000,000 pixels (e.g. ~1024×1024).\n"
                            "• Range: 0.1 to 8.0 MP.\n"
                            "• Step: 0.1 MP (10,000 pixels).\n"
                            "• Width and height calculated automatically:\n"
                            "  – Based on selected aspect_ratio.\n"
                            "  – Rounded to nearest multiple of 32.\n\n"
                            "Examples:\n"
                            "  • 0.15 MP + 16:9 → ~512×288\n"
                            "  • 1.0 MP + 1:1 → ~1024×1024\n"
                            "  • 2.0 MP + 16:9 → ~1920×1088 (2K)\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Целевое разрешение в мегапикселях (MP) для режима Megapixels.\n\n"
                            "• 1.0 MP = 1 000 000 пикселей (например, ~1024×1024).\n"
                            "• Диапазон: 0.1 – 8.0 MP.\n"
                            "• Шаг: 0.1 MP (10 000 пикселей).\n"
                            "• Ширина и высота рассчитываются автоматически:\n"
                            "  – На основе выбранного aspect_ratio.\n"
                            "  – Округляются до кратности 32.\n\n"
                            "Примеры:\n"
                            "  • 0.15 MP + 16:9 → ~512×288\n"
                            "  • 1.0 MP + 1:1 → ~1024×1024\n"
                            "  • 2.0 MP + 16:9 → ~1920×1088 (2K)"
                        )
                    }
                ),
                # ========================================================
                # aspect_ratio
                # ========================================================
                "aspect_ratio": (
                    ASPECT_RATIOS,
                    {
                        "default": "16:9",
                        "tooltip": (
                            "Target aspect ratio for Megapixels mode.\n\n"
                            "Available ratios and typical use:\n"
                            "  • 1:1 — square, social media (Instagram)\n"
                            "  • 3:2 — classic photo, 35mm film\n"
                            "  • 2:3 — vertical photo, portrait\n"
                            "  • 4:3 — standard monitor, iPad\n"
                            "  • 3:4 — vertical standard\n"
                            "  • 16:9 — widescreen, YouTube, TV, cinema\n"
                            "  • 9:16 — vertical video, Stories, Reels, Shorts\n"
                            "  • 21:9 — ultrawide, cinematic\n"
                            "  • 9:21 — vertical ultrawide\n\n"
                            "Final size = closest match to target MP while keeping this ratio.\n"
                            "All sizes are multiples of 32.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Целевое соотношение сторон для режима Megapixels.\n\n"
                            "Доступные форматы и их применение:\n"
                            "  • 1:1 — квадрат, соцсети (Instagram)\n"
                            "  • 3:2 — классическое фото, 35мм плёнка\n"
                            "  • 2:3 — вертикальное фото, портрет\n"
                            "  • 4:3 — стандартный монитор, iPad\n"
                            "  • 3:4 — вертикальный стандарт\n"
                            "  • 16:9 — широкий экран, YouTube, ТВ, кино\n"
                            "  • 9:16 — вертикальное видео, Stories, Reels, Shorts\n"
                            "  • 21:9 — сверхширокий, кинематографичный\n"
                            "  • 9:21 — вертикальный сверхширокий\n\n"
                            "Итоговый размер — ближайший к целевому MP при данном соотношении.\n"
                            "Все размеры кратны 32."
                        )
                    }
                ),
                # ========================================================
                # frame_count_source
                # ========================================================
                "frame_count_source": (
                    ["From seconds", "Manual"],
                    {
                        "default": "From seconds",
                        "tooltip": (
                            "How to determine the total number of frames.\n\n"
                            "• From seconds — automatic calculation using the formula:\n"
                            "  max(5, round(sec×24)) + (5 - (max(5, round(sec×24)) % 17)) % 17\n"
                            "  Example: 10 sec × 24 FPS = 240 → aligned to 243 frames.\n"
                            "  The result always follows the sequence: 5, 22, 39, 56, 73...\n\n"
                            "• Manual — you specify the exact frame count.\n"
                            "  The value is automatically aligned UP to the nearest valid number\n"
                            "  in the sequence 5, 22, 39, 56...\n"
                            "  Example: you enter 100 → output will be 107.\n\n"
                            "HINT: Use 'From seconds' for quick setup. Use 'Manual' for precise\n"
                            "control over exact frame count (e.g. for looped animations).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Способ определения общего количества кадров.\n\n"
                            "• From seconds — автоматический расчёт по формуле:\n"
                            "  max(5, round(сек×24)) + (5 - (max(5, round(сек×24)) % 17)) % 17\n"
                            "  Пример: 10 сек × 24 FPS = 240 → выравнивается до 243 кадров.\n"
                            "  Результат всегда в последовательности: 5, 22, 39, 56, 73...\n\n"
                            "• Manual — вы указываете точное количество кадров.\n"
                            "  Значение автоматически выравнивается ВВЕРХ до ближайшего\n"
                            "  допустимого числа в последовательности 5, 22, 39, 56...\n"
                            "  Пример: вводите 100 → на выходе будет 107.\n\n"
                            "СОВЕТ: используйте 'From seconds' для быстрой настройки.\n"
                            "Используйте 'Manual' для точного контроля (например, для зацикленных анимаций)."
                        )
                    }
                ),
                # ========================================================
                # length_seconds
                # ========================================================
                "length_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 60.0,
                        "step": 1.0,
                        "display": "number",
                        "tooltip": (
                            "Desired video duration in seconds.\n\n"
                            "Used ONLY when frame_count_source = 'From seconds'.\n"
                            "Can be driven by an external Float node (duration socket).\n\n"
                            "The frame count is calculated as:\n"
                            "  max(5, round(sec×24)) + (5 - (max(5, round(sec×24)) % 17)) % 17\n\n"
                            "Examples:\n"
                            "  • 5 sec × 24 FPS → 124 frames\n"
                            "  • 10 sec × 24 FPS → 243 frames\n"
                            "  • 15 sec × 24 FPS → 362 frames\n\n"
                            "Range: 1 to 60 seconds (MiniMax H3 clip limit).\n\n"
                            "Note: the output frame count will be slightly different from\n"
                            "seconds × 24 due to alignment to the 17N+5 sequence.\n"
                            "This is REQUIRED by MiniMax H3.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Желаемая длительность видео в секундах.\n\n"
                            "Используется ТОЛЬКО при frame_count_source = 'From seconds'.\n"
                            "Можно подключить внешний Float-узел (сокет duration).\n\n"
                            "Количество кадров рассчитывается по формуле:\n"
                            "  max(5, round(сек×24)) + (5 - (max(5, round(сек×24)) % 17)) % 17\n\n"
                            "Примеры:\n"
                            "  • 5 сек × 24 FPS → 124 кадра\n"
                            "  • 10 сек × 24 FPS → 243 кадра\n"
                            "  • 15 сек × 24 FPS → 362 кадра\n\n"
                            "Диапазон: 1 – 60 секунд (лимит клипа MiniMax H3).\n\n"
                            "Заметьте: итоговое число кадров будет немного отличаться от\n"
                            "секунды × 24 из-за выравнивания под 17N+5.\n"
                            "Это ТРЕБОВАНИЕ MiniMax H3."
                        )
                    }
                ),
                # ========================================================
                # frame_count
                # ========================================================
                "frame_count": (
                    "INT",
                    {
                        "default": 124,
                        "min": 5,
                        "max": 99999,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Exact frame count for Manual mode.\n\n"
                            "Used ONLY when frame_count_source = 'Manual'.\n\n"
                            "The value is automatically aligned UP to the nearest valid number\n"
                            "in the MiniMax H3 sequence: 5, 22, 39, 56, 73, 90, 107, 124...\n\n"
                            "Formula: 17 × N + 5 (where N = 0, 1, 2, 3...), minimum 5.\n\n"
                            "Examples:\n"
                            "  • You enter: 100 → Output: 107 (nearest 17N+5 >= 100)\n"
                            "  • You enter: 124 → Output: 124 (already valid)\n"
                            "  • You enter: 5 → Output: 5 (minimum)\n"
                            "  • You enter: 200 → Output: 209 (nearest 17N+5 >= 200)\n\n"
                            "Range: 5 to 99999 frames.\n\n"
                            "HINT: typical H3 videos: 124 (5 sec @24fps), 243 (10 sec @24fps).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Точное количество кадров для Manual режима.\n\n"
                            "Используется ТОЛЬКО при frame_count_source = 'Manual'.\n\n"
                            "Значение автоматически выравнивается ВВЕРХ до ближайшего\n"
                            "допустимого числа из последовательности MiniMax H3:\n"
                            "5, 22, 39, 56, 73, 90, 107, 124...\n\n"
                            "Формула: 17 × N + 5 (где N = 0, 1, 2, 3...), минимум 5.\n\n"
                            "Примеры:\n"
                            "  • Вводите: 100 → На выходе: 107 (ближайшее 17N+5 >= 100)\n"
                            "  • Вводите: 124 → На выходе: 124 (уже допустимое)\n"
                            "  • Вводите: 5 → На выходе: 5 (минимум)\n"
                            "  • Вводите: 200 → На выходе: 209 (ближайшее 17N+5 >= 200)\n\n"
                            "Диапазон: 5 – 99999 кадров.\n\n"
                            "СОВЕТ: типичные значения H3: 124 (5 сек @24fps), 243 (10 сек @24fps)."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "fps_int", "fps_float", "total_frames")

    # ====================================================================
    # Megapixels → размер кадра
    # ====================================================================
    @staticmethod
    def _megapixels_to_size(mp, ratio):
        """
        Вычисляет (width, height) по мегапикселям и соотношению сторон.
        Все значения подгоняются под кратность 32.
        """
        w_r, h_r = map(int, ratio.split(":"))
        target = mp * 1_000_000
        x = math.sqrt(target / (w_r * h_r))
        w = fit_to_multiple(round(w_r * x), 32)
        h = fit_to_multiple(round(h_r * x), 32)
        return max(64, w), max(64, h)

    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count):

        # ================================================================
        # 1. Ширина и высота кадра
        # ================================================================
        if mode == "Preset":
            w, h = PRESET_MAP[preset]
            if invert_orientation:
                w, h = h, w
            width = fit_to_multiple(w, 32)
            height = fit_to_multiple(h, 32)

        elif mode == "Custom":
            w, h = custom_width, custom_height
            if invert_orientation:
                w, h = h, w
            width = fit_to_multiple(w, 32)
            height = fit_to_multiple(h, 32)

        elif mode == "Megapixels":
            width, height = self._megapixels_to_size(megapixels_value, aspect_ratio)

        else:
            width, height = 1280, 704

        # ================================================================
        # 2. Частота кадров (FPS) — у H3 всегда 24
        # ================================================================
        fps_int = 24
        fps_float = 24.0

        # ================================================================
        # 3. Количество кадров
        # ================================================================
        if frame_count_source == "Manual":
            total = fit_length_to_17n5(frame_count)
        else:
            # max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17
            total = fit_length_to_17n5(round(length_seconds * 24))
        total = max(5, total)

        return (width, height, fps_int, fps_float, total)


# ========================================================================
# Регистрация
# ========================================================================
NODE_CLASS_MAPPINGS = {"AGSoft_MiniMax_Base": AGSoft_MiniMax_Base}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_MiniMax_Base": "🎬 AGSoft MiniMax Base"}