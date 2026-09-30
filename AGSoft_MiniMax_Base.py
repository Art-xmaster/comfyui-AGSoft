"""
AGSoft_MiniMax_Base.py
Нода / Node: 🎬AGSoft MiniMax Base
Версия / Version: v09.60
Калькулятор параметров видео для MiniMax H3.
Ширина/высота кратны 32 (Preset / Custom / Megapixels), FPS фиксирован 24,
число кадров — последовательность 17N+5 (5, 22, 39, ...) из секунд (Seconds)
или вручную (Frames). Выход duration_seconds — точная длительность клипа
(total_frames / 24). Цепочка copy_options передаёт все параметры калькулятора
ведомым нодам.
Video parameter calculator for MiniMax H3. Width/height are multiples of 32
(Preset / Custom / Megapixels), FPS fixed at 24, frame count follows the
17N+5 sequence from seconds (Seconds) or manually (Frames). duration_seconds
output = exact clip duration (total_frames / 24). The copy_options chain
passes all calculator parameters to follower nodes.
Автор / Author: AGSoft
Дата / Date: 28.09.2026
"""

import math

# Имя сокета цепочки / chain socket name
COPY_SOCKET = "copy_options"
COPY_TYPE = "AGSOFT_MINIMAX_COPY"


#========================================================================
# Утилиты / Utilities
#========================================================================

def fit_to_multiple(value: int, multiple: int = 32) -> int:
    """Округляет ВВЕРХ до кратного multiple / Rounds UP to a multiple. 100 → 128."""
    return ((value + multiple - 1) // multiple) * multiple


def fit_length_to_17n5(value: int) -> int:
    """Выравнивает ВВЕРХ до 17N+5 (5, 22, 39, ...) / Aligns UP to 17N+5. 100 → 107."""
    v = max(5, int(value))
    return v + (5 - v % 17) % 17


#========================================================================
# Пресеты / Presets
#========================================================================

PRESET_LIST = [
    # 1:1 Квадрат / Square
    "512×512 (1:1)", "576×576 (1:1)", "640×640 (1:1)", "704×704 (1:1)",
    "768×768 (1:1)", "832×832 (1:1)", "896×896 (1:1)", "960×960 (1:1)",
    "1024×1024 (1:1)", "1280×1280 (1:1)",
    # 3:2 Фото / Photo
    "480×320 (3:2)", "576×384 (3:2)", "672×448 (3:2)", "768×512 (3:2)",
    "864×576 (3:2)", "960×640 (3:2)", "1056×704 (3:2)", "1152×768 (3:2)",
    "1248×832 (3:2)", "1536×1024 (3:2)",
    # 4:3 Стандарт / Standard
    "512×384 (4:3)", "640×480 (4:3)", "768×576 (4:3)", "896×672 (4:3)",
    "1024×768 (4:3)", "1152×864 (4:3)", "1280×960 (4:3)", "1408×1056 (4:3)",
    "1536×1152 (4:3)", "2048×1536 (4:3)",
    # 16:9 Кино/ТВ / Cinema/TV
    "512×288 (16:9)", "640×352 (16:9)", "768×448 (16:9)", "896×512 (16:9)",
    "1024×576 (16:9)", "1152×640 (16:9)", "1280×704 (16:9)", "1536×864 (16:9)",
    "1920×1056 (16:9)", "1920×1088 (16:9)", "2048×1152 (16:9)",
]

PRESET_MAP = {
    "512×512 (1:1)": (512, 512), "576×576 (1:1)": (576, 576),
    "640×640 (1:1)": (640, 640), "704×704 (1:1)": (704, 704),
    "768×768 (1:1)": (768, 768), "832×832 (1:1)": (832, 832),
    "896×896 (1:1)": (896, 896), "960×960 (1:1)": (960, 960),
    "1024×1024 (1:1)": (1024, 1024), "1280×1280 (1:1)": (1280, 1280),
    "480×320 (3:2)": (480, 320), "576×384 (3:2)": (576, 384),
    "672×448 (3:2)": (672, 448), "768×512 (3:2)": (768, 512),
    "864×576 (3:2)": (864, 576), "960×640 (3:2)": (960, 640),
    "1056×704 (3:2)": (1056, 704), "1152×768 (3:2)": (1152, 768),
    "1248×832 (3:2)": (1248, 832), "1536×1024 (3:2)": (1536, 1024),
    "512×384 (4:3)": (512, 384), "640×480 (4:3)": (640, 480),
    "768×576 (4:3)": (768, 576), "896×672 (4:3)": (896, 672),
    "1024×768 (4:3)": (1024, 768), "1152×864 (4:3)": (1152, 864),
    "1280×960 (4:3)": (1280, 960), "1408×1056 (4:3)": (1408, 1056),
    "1536×1152 (4:3)": (1536, 1152), "2048×1536 (4:3)": (2048, 1536),
    "512×288 (16:9)": (512, 288), "640×352 (16:9)": (640, 352),
    "768×448 (16:9)": (768, 448), "896×512 (16:9)": (896, 512),
    "1024×576 (16:9)": (1024, 576), "1152×640 (16:9)": (1152, 640),
    "1280×704 (16:9)": (1280, 704), "1536×864 (16:9)": (1536, 864),
    "1920×1056 (16:9)": (1920, 1056), "1920×1088 (16:9)": (1920, 1088),
    "2048×1152 (16:9)": (2048, 1152),
}

ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]


#========================================================================
# Краткие двуязычные тултипы / Concise bilingual tooltips
#========================================================================

_T = lambda en, ru: en + "\n---\n" + ru

CALC_TOOLTIPS = {
    "mode": _T(
        "Frame size mode: Preset (predefined sizes), Custom (own W/H), Megapixels (MP + ratio). Unused widgets hide automatically.",
        "Режим размера кадра: Preset (готовые размеры), Custom (свои W/H), Megapixels (MP + соотношение). Лишние виджеты скрываются автоматически."
    ),
    "preset": _T(
        "Predefined size WIDTH×HEIGHT (aspect), all multiples of 32. invert_orientation swaps W/H.",
        "Готовый размер ШИРИНА×ВЫСОТА (соотношение), все кратны 32. invert_orientation меняет W/H местами."
    ),
    "invert_orientation": _T(
        "Swaps width and height. Works in all modes (in Megapixels it swaps the computed size).",
        "Меняет ширину и высоту местами. Работает во всех режимах (в Megapixels — расчётный размер)."
    ),
    "custom_width": _T(
        "Frame width in px (Custom). Rounded UP to a multiple of 32. Range 64–8192, step 32.",
        "Ширина кадра в px (Custom). Округляется ВВЕРХ до кратности 32. Диапазон 64–8192, шаг 32."
    ),
    "custom_height": _T(
        "Frame height in px (Custom). Rounded UP to a multiple of 32. Range 64–8192, step 32.",
        "Высота кадра в px (Custom). Округляется ВВЕРХ до кратности 32. Диапазон 64–8192, шаг 32."
    ),
    "megapixels_value": _T(
        "Target resolution in MP (0.01–8.0, step 0.01). W/H computed from aspect_ratio, multiples of 32.",
        "Целевое разрешение в MP (0.01–8.0, шаг 0.01). W/H считаются по aspect_ratio, кратны 32."
    ),
    "aspect_ratio": _T(
        "Aspect ratio for Megapixels: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21.",
        "Соотношение сторон для Megapixels: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21."
    ),
    "frame_count_source": _T(
        "Seconds — frames = round(sec×24) aligned UP to 17N+5. Frames — exact count aligned UP to 17N+5.",
        "Seconds — кадры = round(сек×24) ВВЕРХ до 17N+5. Frames — точное число с выравниванием ВВЕРХ до 17N+5."
    ),
    "length_seconds": _T(
        "Duration in seconds (1–60, Seconds only). Frames = round(sec×24) aligned UP to 17N+5, so it differs slightly from sec×24.",
        "Длительность в секундах (1–60, только Seconds). Кадры = round(сек×24) ВВЕРХ до 17N+5, поэтому число немного отличается от сек×24."
    ),
    "frame_count": _T(
        "Exact frame count (Frames only), aligned UP to 17N+5: 100 → 107, 124 → 124.",
        "Точное число кадров (только Frames), выравнивается ВВЕРХ до 17N+5: 100 → 107, 124 → 124."
    ),
    "copy": _T(
        "Chain: the output passes all calculator parameters; a follower with the input repeats the master and collapses its calculator.",
        "Цепочка: выход передаёт все параметры калькулятора; ведомая нода со входом повторяет мастера и схлопывает свой калькулятор."
    ),
}


#========================================================================
# Виджеты калькулятора / Calculator widgets
#========================================================================

def _calc_widgets():
    """Строит виджеты калькулятора / Builds the calculator widgets."""
    return {
        "mode": (["Preset", "Custom", "Megapixels"], {"default": "Preset", "tooltip": CALC_TOOLTIPS["mode"]}),
        "preset": (PRESET_LIST, {"default": "1280×704 (16:9)", "tooltip": CALC_TOOLTIPS["preset"]}),
        "invert_orientation": ("BOOLEAN", {"default": False, "tooltip": CALC_TOOLTIPS["invert_orientation"]}),
        "custom_width": ("INT", {"default": 864, "min": 64, "max": 8192, "step": 32, "display": "number", "tooltip": CALC_TOOLTIPS["custom_width"]}),
        "custom_height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 32, "display": "number", "tooltip": CALC_TOOLTIPS["custom_height"]}),
        "megapixels_value": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "display": "number", "tooltip": CALC_TOOLTIPS["megapixels_value"]}),
        "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": CALC_TOOLTIPS["aspect_ratio"]}),
        "frame_count_source": (["Seconds", "Frames"], {"default": "Seconds", "tooltip": CALC_TOOLTIPS["frame_count_source"]}),
        "length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 1.0, "display": "number", "tooltip": CALC_TOOLTIPS["length_seconds"]}),
        "frame_count": ("INT", {"default": 124, "min": 5, "max": 99999, "step": 1, "display": "number", "tooltip": CALC_TOOLTIPS["frame_count"]}),
    }


#========================================================================
# Расчёты / Calculator math
#========================================================================

def _calc_size(mode, preset, invert_orientation, custom_width, custom_height,
               megapixels_value, aspect_ratio):
    """Ширина/высота по режиму, кратность 32, инверсия во всех режимах.
    Width/height per mode, multiple of 32, invert in all modes."""
    if mode == "Preset":
        w, h = PRESET_MAP[preset]
        if invert_orientation:
            w, h = h, w
        return fit_to_multiple(w, 32), fit_to_multiple(h, 32)

    if mode == "Custom":
        w, h = custom_width, custom_height
        if invert_orientation:
            w, h = h, w
        return fit_to_multiple(w, 32), fit_to_multiple(h, 32)

    if mode == "Megapixels":
        w_r, h_r = map(int, aspect_ratio.split(":"))
        target = megapixels_value * 1_000_000
        x = math.sqrt(target / (w_r * h_r))
        w = fit_to_multiple(round(w_r * x), 32)
        h = fit_to_multiple(round(h_r * x), 32)
        if invert_orientation:
            w, h = h, w
        return max(64, w), max(64, h)

    return 1280, 704


def _calc_frames(frame_count_source, length_seconds, frame_count):
    """Число кадров 17N+5 по источнику / Frame count 17N+5 per source."""
    if frame_count_source == "Frames":
        total = fit_length_to_17n5(frame_count)
    else:
        total = fit_length_to_17n5(round(length_seconds * 24))
    return max(5, total)


#========================================================================
# Нода / Node
#========================================================================

class AGSoft_MiniMax_Base:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "Video parameter calculator for MiniMax H3: width/height multiples of 32 "
        "(Preset — 41 sizes, Custom — own W/H, Megapixels — MP + ratio); FPS 24; frame "
        "count from Seconds or Frames aligned UP to 17N+5; orientation invert in all "
        "modes; duration_seconds output = exact duration (total_frames / 24); "
        "copy_options chain passes all parameters to followers; live info line: "
        "⚙ W×H • MP • ~aspect • frames • exact sec."
        "---\n"
        "Калькулятор параметров видео для MiniMax H3: ширина/высота кратны 32 "
        "(Preset — 41 готовый размер, Custom — свои W/H, Megapixels — MP + соотношение); "
        "FPS 24; число кадров по Seconds или Frames с выравниванием ВВЕРХ до 17N+5; "
        "инверсия сторон во всех режимах; выход duration_seconds — точная длительность "
        "(total_frames / 24); цепочка copy_options передаёт все параметры ведомым нодам; "
        "живая инфострока: ⚙ W×H • MP • ~формат • кадры • точные сек.\n"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": _calc_widgets(),
            "optional": {
                COPY_SOCKET: (COPY_TYPE, {"tooltip": CALC_TOOLTIPS["copy"]}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT", "FLOAT", COPY_TYPE)
    RETURN_NAMES = ("width", "height", "fps_int", "fps_float", "total_frames",
                    "duration_seconds", COPY_SOCKET)

    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count, **kwargs):
        # 0. Цепочка: значения мастера заменяют свои виджеты.
        # Chain: the master's values replace own widgets.
        copy = kwargs.get(COPY_SOCKET, None)
        if isinstance(copy, dict):
            mode = copy.get("mode", mode)
            preset = copy.get("preset", preset)
            invert_orientation = copy.get("invert_orientation", invert_orientation)
            custom_width = copy.get("custom_width", custom_width)
            custom_height = copy.get("custom_height", custom_height)
            megapixels_value = copy.get("megapixels_value", megapixels_value)
            aspect_ratio = copy.get("aspect_ratio", aspect_ratio)
            frame_count_source = copy.get("frame_count_source", frame_count_source)
            length_seconds = copy.get("length_seconds", length_seconds)
            frame_count = copy.get("frame_count", frame_count)

        # 1. Размер / Size
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)

        # 2. FPS — всегда 24 / always 24
        fps_int, fps_float = 24, 24.0

        # 3. Кадры 17N+5 / Frames 17N+5
        total = _calc_frames(frame_count_source, length_seconds, frame_count)

        # 4. Точная длительность / Exact duration
        duration_seconds = total / 24.0

        # 5. Выход цепочки / Chain output
        out_copy = {
            "mode": mode,
            "preset": preset,
            "invert_orientation": invert_orientation,
            "custom_width": custom_width,
            "custom_height": custom_height,
            "megapixels_value": megapixels_value,
            "aspect_ratio": aspect_ratio,
            "frame_count_source": frame_count_source,
            "length_seconds": length_seconds,
            "frame_count": frame_count,
        }

        return (width, height, fps_int, fps_float, total,
                duration_seconds, out_copy)


#========================================================================
# Регистрация / Registration
#========================================================================

NODE_CLASS_MAPPINGS = {
    "AGSoft_MiniMax_Base": AGSoft_MiniMax_Base,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_MiniMax_Base": "🎬AGSoft MiniMax Base",
}