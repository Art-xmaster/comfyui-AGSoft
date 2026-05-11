"""
# AGSoft LTX Base
# Автор: AGSoft
# Дата: 11.05.2026 г.
Нода AGSoft LTX Base — расчет параметров видео для моделей LTX.
Все размеры кратны 32, количество кадров — последовательность 1, 9, 17, 25...
"""


def fit_to_multiple(value: int, multiple: int = 32) -> int:
    """
    Подгоняет значение к ближайшему большему числу, кратному multiple.
    
    Примеры:
        100 → 128 (при multiple=32)
        128 → 128 (уже кратно)
        33  → 64
    """
    return ((value + multiple - 1) // multiple) * multiple


def fit_length_to_step(value: int, step: int = 8) -> int:
    """
    Подгоняет число под последовательность: 1, 9, 17, 25...
    Формула: step × N + 1, где N >= 0.
    
    Примеры:
        1   → 1
        5   → 9
        17  → 17
        120 → 121
        129 → 129
    """
    if value <= 1:
        return 1
    return ((value - 1 + step - 1) // step) * step + 1


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
# Megapixels
# ========================================================================
ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]


# ========================================================================
# Нода
# ========================================================================
class AGSoft_LTX_Base:

    CATEGORY = "AGSoft/Video"
    FUNCTION = "main"

    DESCRIPTION = (
        "AGSoft LTX Base — universal video parameter calculator for LTX models.\n"
        "Automatically normalizes all values to the required formats:\n"
        "• Frame width and height — always multiples of 32.\n"
        "• Frame count — follows the sequence: 1, 9, 17, 25, 33... (N×8+1).\n"
        "• Frame rate (FPS) — integer and float (2 decimal places).\n\n"
        "Supports three size modes: Preset, Custom, Megapixels.\n"
        "Invert orientation available for Preset and Custom modes.\n"
        "Frame count can be calculated from seconds or entered manually.\n\n"
        "Use this node before connecting to LTX Video Sampler or other LTX nodes.\n\n"
        "═══════════════════════════════════════\n\n"
        "AGSoft LTX Base — универсальный калькулятор параметров для моделей LTX.\n"
        "Автоматически нормализует все значения под требования формата:\n"
        "• Ширина и высота кадра — всегда кратны 32.\n"
        "• Количество кадров — последовательность: 1, 9, 17, 25, 33... (N×8+1).\n"
        "• Частота кадров (FPS) — целое и дробное (2 знака после запятой).\n\n"
        "Три режима выбора размера: Preset, Custom, Megapixels.\n"
        "Инверсия сторон работает в Preset и Custom.\n"
        "Количество кадров — авторасчёт из секунд или ручной ввод.\n\n"
        "Используйте эту ноду перед подключением к LTX Video Sampler или другим LTX-нодам."
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
                            "  • 768×1344 (9:16) — vertical video (after invert)\n\n"
                            "All sizes are multiples of 32 — safe for LTX models.\n"
                            "Grouped by aspect ratio for easy navigation:\n"
                            "  1:1 → 3:2 → 4:3 → 16:9\n\n"
                            "Use 'invert_orientation' to swap width↔height if needed.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Готовый размер кадра с меткой соотношения сторон.\n\n"
                            "Формат: ШИРИНА×ВЫСОТА (СООТНОШЕНИЕ)\n"
                            "Примеры:\n"
                            "  • 1024×1024 (1:1) — квадрат, 1 мегапиксель\n"
                            "  • 1280×704 (16:9) — HD горизонтальное, киноформат\n"
                            "  • 768×1344 (9:16) — вертикальное видео (после инверсии)\n\n"
                            "Все размеры кратны 32 — безопасно для моделей LTX.\n"
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
                        "default": 1280,
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
                        "default": 704,
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
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Target resolution in megapixels (MP) for Megapixels mode.\n\n"
                            "• 1.0 MP = 1,000,000 pixels (e.g. ~1024×1024).\n"
                            "• Range: 0.1 to 8.0 MP.\n"
                            "• Step: 0.01 MP (10,000 pixels).\n"
                            "• Width and height calculated automatically:\n"
                            "  – Based on selected aspect_ratio.\n"
                            "  – Rounded to nearest multiple of 32.\n\n"
                            "Examples:\n"
                            "  • 0.15 MP + 16:9 → ~512×288\n"
                            "  • 1.0 MP + 1:1  → ~1024×1024\n"
                            "  • 2.0 MP + 16:9 → ~1920×1088\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Целевое разрешение в мегапикселях (MP) для режима Megapixels.\n\n"
                            "• 1.0 MP = 1 000 000 пикселей (например, ~1024×1024).\n"
                            "• Диапазон: 0.1 – 8.0 MP.\n"
                            "• Шаг: 0.01 MP (10 000 пикселей).\n"
                            "• Ширина и высота рассчитываются автоматически:\n"
                            "  – На основе выбранного aspect_ratio.\n"
                            "  – Округляются до кратности 32.\n\n"
                            "Примеры:\n"
                            "  • 0.15 MP + 16:9 → ~512×288\n"
                            "  • 1.0 MP + 1:1  → ~1024×1024\n"
                            "  • 2.0 MP + 16:9 → ~1920×1088"
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
                # frame_rate_float
                # ========================================================
                "frame_rate_float": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 0.01,
                        "max": 120.0,
                        "step": 0.01,
                        "display": "number",
                        "tooltip": (
                            "Frame rate in frames per second (FPS).\n\n"
                            "Standard values:\n"
                            "  • 23.976 — cinema (NTSC film)\n"
                            "  • 24.0   — cinema (film standard)\n"
                            "  • 25.0   — PAL TV (Europe, Russia)\n"
                            "  • 29.97  — NTSC TV (USA, Japan)\n"
                            "  • 30.0   — internet video, gaming\n"
                            "  • 50.0   — smooth motion, slow-mo source\n"
                            "  • 60.0   — high frame rate, gaming content\n\n"
                            "Two outputs provided:\n"
                            "  • frame_rate_int — integer part (e.g. 25)\n"
                            "  • frame_rate_float — full value with 2 decimals (e.g. 25.0, 29.97)\n\n"
                            "Used in frame count calculation:\n"
                            "  frames = ((sec × FPS_int − 1 + 7) // 8) × 8 + 1\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Частота кадров (FPS).\n\n"
                            "Стандартные значения:\n"
                            "  • 23.976 — кино (NTSC)\n"
                            "  • 24.0   — кино (стандарт)\n"
                            "  • 25.0   — PAL ТВ (Европа, Россия)\n"
                            "  • 29.97  — NTSC ТВ (США, Япония)\n"
                            "  • 30.0   — интернет-видео, игры\n"
                            "  • 50.0   — плавное движение, источник для slow-mo\n"
                            "  • 60.0   — высокий FPS, игровой контент\n\n"
                            "Два выхода:\n"
                            "  • frame_rate_int — целая часть (например, 25)\n"
                            "  • frame_rate_float — полное значение с 2 знаками (например, 25.0, 29.97)\n\n"
                            "Участвует в расчёте кадров:\n"
                            "  кадры = ((сек × FPS_int − 1 + 7) // 8) × 8 + 1"
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
                            "  ((length_seconds × FPS_int − 1 + 7) // 8) × 8 + 1\n"
                            "  Example: 5 sec × 25 FPS = 125 → aligned to 129 frames.\n"
                            "  The result always follows the sequence: 1, 9, 17, 25, 33...\n\n"
                            "• Manual — you specify the exact frame count.\n"
                            "  The value is automatically aligned UP to the nearest valid number\n"
                            "  in the sequence 1, 9, 17, 25...\n"
                            "  Example: you enter 100 → output will be 105 (nearest 8N+1).\n\n"
                            "HINT: Use 'From seconds' for quick setup. Use 'Manual' for precise\n"
                            "control over exact frame count (e.g. for looped animations).\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Способ определения общего количества кадров.\n\n"
                            "• From seconds — автоматический расчёт по формуле:\n"
                            "  ((length_seconds × FPS_int − 1 + 7) // 8) × 8 + 1\n"
                            "  Пример: 5 сек × 25 FPS = 125 → выравнивается до 129 кадров.\n"
                            "  Результат всегда в последовательности: 1, 9, 17, 25, 33...\n\n"
                            "• Manual — вы указываете точное количество кадров.\n"
                            "  Значение автоматически выравнивается ВВЕРХ до ближайшего\n"
                            "  допустимого числа в последовательности 1, 9, 17, 25...\n"
                            "  Пример: вводите 100 → на выходе будет 105 (ближайшее 8N+1).\n\n"
                            "СОВЕТ: используйте 'From seconds' для быстрой настройки.\n"
                            "Используйте 'Manual' для точного контроля (например, для зацикленных анимаций)."
                        )
                    }
                ),
                # ========================================================
                # length_seconds
                # ========================================================
                "length_seconds": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 3600,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Desired video duration in seconds.\n\n"
                            "Used ONLY when frame_count_source = 'From seconds'.\n\n"
                            "The actual frame count is calculated as:\n"
                            "  ((length_seconds × FPS_int − 1 + 7) // 8) × 8 + 1\n\n"
                            "Examples:\n"
                            "  • 3 sec × 24 FPS → 71 frames (not 72!)\n"
                            "  • 5 sec × 25 FPS → 129 frames\n"
                            "  • 10 sec × 30 FPS → 297 frames\n\n"
                            "Range: 1 to 3600 seconds (1 hour).\n\n"
                            "Note: the output frame count will be slightly different from\n"
                            "seconds × FPS due to alignment to the 8N+1 sequence.\n"
                            "This is REQUIRED by LTX models.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Желаемая длительность видео в секундах.\n\n"
                            "Используется ТОЛЬКО при frame_count_source = 'From seconds'.\n\n"
                            "Количество кадров рассчитывается по формуле:\n"
                            "  ((length_seconds × FPS_int − 1 + 7) // 8) × 8 + 1\n\n"
                            "Примеры:\n"
                            "  • 3 сек × 24 FPS → 71 кадр (не 72!)\n"
                            "  • 5 сек × 25 FPS → 129 кадров\n"
                            "  • 10 сек × 30 FPS → 297 кадров\n\n"
                            "Диапазон: 1 – 3600 секунд (1 час).\n\n"
                            "Заметьте: итоговое число кадров будет немного отличаться от\n"
                            "секунды × FPS из-за выравнивания под 8N+1.\n"
                            "Это ТРЕБОВАНИЕ моделей LTX."
                        )
                    }
                ),
                # ========================================================
                # frame_count
                # ========================================================
                "frame_count": (
                    "INT",
                    {
                        "default": 129,
                        "min": 1,
                        "max": 99999,
                        "step": 1,
                        "display": "number",
                        "tooltip": (
                            "Exact frame count for Manual mode.\n\n"
                            "Used ONLY when frame_count_source = 'Manual'.\n\n"
                            "The value is automatically aligned UP to the nearest valid number\n"
                            "in the LTX sequence: 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97...\n\n"
                            "Formula: N × 8 + 1 (where N = 0, 1, 2, 3...)\n\n"
                            "Examples:\n"
                            "  • You enter: 100  →  Output: 105  (nearest 8N+1 >= 100)\n"
                            "  • You enter: 129  →  Output: 129  (already valid)\n"
                            "  • You enter: 1    →  Output: 1    (minimum)\n"
                            "  • You enter: 200  →  Output: 201  (nearest 8N+1 >= 200)\n\n"
                            "Range: 1 to 99999 frames.\n\n"
                            "HINT: Typical LTX videos range from 33 to 257 frames.\n"
                            "Common values: 129 (5 sec @25fps), 161, 193.\n\n"
                            "═══════════════════════════════════════\n\n"
                            "Точное количество кадров для Manual режима.\n\n"
                            "Используется ТОЛЬКО при frame_count_source = 'Manual'.\n\n"
                            "Значение автоматически выравнивается ВВЕРХ до ближайшего\n"
                            "допустимого числа из последовательности LTX:\n"
                            "1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97...\n\n"
                            "Формула: N × 8 + 1 (где N = 0, 1, 2, 3...)\n\n"
                            "Примеры:\n"
                            "  • Вводите: 100  →  На выходе: 105  (ближайшее 8N+1 >= 100)\n"
                            "  • Вводите: 129  →  На выходе: 129  (уже допустимое)\n"
                            "  • Вводите: 1    →  На выходе: 1    (минимум)\n"
                            "  • Вводите: 200  →  На выходе: 201  (ближайшее 8N+1 >= 200)\n\n"
                            "Диапазон: 1 – 99999 кадров.\n\n"
                            "СОВЕТ: типичные LTX видео — от 33 до 257 кадров.\n"
                            "Частые значения: 129 (5 сек @25fps), 161, 193."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("width", "height", "frame_rate_int", "frame_rate_float", "frame_count")

    @staticmethod
    def _megapixels_to_size(mp, ratio):
        """
        Вычисляет (width, height) по мегапикселям и соотношению сторон.
        Все значения подгоняются под кратность 32.
        """
        import math
        w_r, h_r = map(int, ratio.split(":"))
        target = mp * 1_000_000
        x = math.sqrt(target / (w_r * h_r))
        w = fit_to_multiple(round(w_r * x), 32)
        h = fit_to_multiple(round(h_r * x), 32)
        return max(64, w), max(64, h)

    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_rate_float, frame_count_source,
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
        # 2. Частота кадров (FPS)
        # ================================================================
        fps_float = round(frame_rate_float, 2)
        fps_int = int(fps_float)

        # ================================================================
        # 3. Количество кадров
        # ================================================================
        if frame_count_source == "Manual":
            total = fit_length_to_step(frame_count, 8)
        else:
            # ((a * b - 1 + 7) // 8) * 8 + 1
            total = ((length_seconds * fps_int - 1 + 7) // 8) * 8 + 1
            total = max(1, total)

        return (width, height, fps_int, fps_float, total)


# ========================================================================
# Регистрация
# ========================================================================
NODE_CLASS_MAPPINGS = {"AGSoft_LTX_Base": AGSoft_LTX_Base}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_LTX_Base": "🎬 AGSoft LTX Base"}