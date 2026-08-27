# ==============================================================================
# AGSoft_Aspect_Ratio.py
# ==============================================================================
# Нода: 📐AGSoft Aspect Ratio
# Описание / Description:
# Калькулятор размеров из пропорции. Выбираешь пресет или свою пропорцию,
# фиксируешь одну сторону (width/height/longest/shortest/megapixels) и
# получаешь точные width/height, выровненные под кратность (8/32/64) —
# готово к подключению в любую ноду ресайза/генерации.
# Dimension calculator from an aspect ratio. Pick a preset or a custom ratio,
# anchor one side (width/height/longest/shortest/megapixels) and get exact
# width/height aligned to a multiple (8/32/64) — ready to plug into any
# resize/generation node.
#
# Возможности / Features:
# ⚡ 15 пресетов + custom (поддержка "16:9", "1.85:1", "2.39:1").
#   15 presets + custom ("16:9", "1.85:1", "2.39:1" supported).
# ⚡ 5 режимов фиксации: width / height / longest / shortest / megapixels.
#   5 anchor modes: width / height / longest / shortest / megapixels.
# ⚡ Округление под кратность: floor / round / ceil.
#   Multiple rounding: floor / round / ceil.
# ⚡ Выходы: width, height, ratio (float), строка "1920×1080 (16:9)".
#   Outputs: width, height, ratio (float), display string "1920×1080 (16:9)".
#
# Автор / Author: AGSoft
# Дата / Date: 27.08.2026
# ==============================================================================
import math

RATIO_PRESETS = [
    "1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9",
    "4:5", "3:4", "2:3", "9:16", "1:2", "9:21", "custom",
]
BASE_MODES = ["width", "height", "longest", "shortest", "megapixels"]
ROUND_MODES = ["floor", "round", "ceil"]


def _parse_ratio(text):
    s = str(text).strip()
    for sep in ("x", "X", "/", ",", ";"):
        s = s.replace(sep, ":")
    parts = [p for p in s.split(":") if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"Неверный формат пропорции: '{text}' (ожидалось W:H)")
    w = float(parts[0]); h = float(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Значения пропорции должны быть > 0: '{text}'")
    return w, h


def _to_multiple(value, multiple, mode):
    v = max(1.0, float(value))
    if multiple <= 1:
        return int(round(v))
    if mode == "floor":
        return max(multiple, math.floor(v / multiple) * multiple)
    if mode == "ceil":
        return max(multiple, math.ceil(v / multiple) * multiple)
    return max(multiple, round(v / multiple) * multiple)


class AGSoft_Aspect_Ratio:
    DESCRIPTION = (
        "📐 AGSoft Aspect Ratio.\n"
        "Calculates exact width/height from an aspect ratio preset (or custom), anchored by "
        "width/height/longest/shortest/megapixels, aligned to a multiple.\n"
        "---\n"
        "📐 AGSoft Aspect Ratio.\n"
        "Вычисляет точные width/height из пропорции (пресет или custom), с фиксацией "
        "width/height/longest/shortest/megapixels и выравниванием под кратность."
    )
    CATEGORY = "AGSoft/nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (RATIO_PRESETS, {
                    "default": "1:1",
                    "tooltip": "Preset aspect ratio. 'custom' = use your own from custom_ratio.\n---\nПресет пропорции. 'custom' = своя из custom_ratio."}),
                "custom_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": "Custom ratio (W:H), used only when preset='custom'. Supports '16:9', '1.85:1', '2.39:1'.\n---\nСвоя пропорция (W:H), работает только при preset='custom'. Поддержка '16:9', '1.85:1', '2.39:1'."}),
                "base": (BASE_MODES, {
                    "default": "width",
                    "tooltip": "Which value to fix: width / height / longest / shortest side / total megapixels.\n---\nЧто фиксировать: width / height / длинную / короткую сторону / суммарные мегапиксели."}),
                "base_value": ("FLOAT", {
                    "default": 1024.0, "min": 1.0, "max": 16384.0, "step": 1.0,
                    "tooltip": "Fixed side value (px) for width/height/longest/shortest.\n-\nЗначение фиксируемой стороны (px) для width/height/longest/shortest."}),
            "megapixels_value": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Total megapixels, step 0.1 (used only when base='megapixels').\n-\nСуммарные мегапиксели, шаг 0.1 (работает только при base='megapixels')."}),
                "multiple": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Align dimensions to a multiple (8/32/64). 1 = no rounding.\n---\nВыравнивание размеров под кратность (8/32/64). 1 = без округления."}),
                "rounding": (ROUND_MODES, {
                    "default": "round",
                    "tooltip": "Rounding method: floor (down) / round (nearest) / ceil (up).\n---\nМетод округления: floor (вниз) / round (ближайшее) / ceil (вверх)."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "ratio", "display")
    FUNCTION = "calculate"

    def calculate(self, preset, custom_ratio, base, base_value, megapixels_value, multiple, rounding):
        ratio_text = custom_ratio if preset == "custom" else preset
        rw, rh = _parse_ratio(ratio_text)
        ratio = rw / rh
        val = float(base_value)

        if base == "width":
            w, h = val, val * rh / rw
        elif base == "height":
            h, w = val, val * rw / rh
        elif base == "longest":
            if rw >= rh: w, h = val, val * rh / rw
            else:        h, w = val, val * rw / rh
        elif base == "shortest":
            if rw <= rh: w, h = val, val * rh / rw
            else:        h, w = val, val * rw / rh
        else: # megapixels
            target = float(megapixels_value) * 1_000_000
            w = math.sqrt(target * ratio)
            h = math.sqrt(target / ratio)
        W = _to_multiple(w, multiple, rounding)
        H = _to_multiple(h, multiple, rounding)
        display = f"{W}×{H} ({ratio_text})"
        return {"result": (W, H, ratio, display),
                "ui": {"display": [display], "width": [W], "height": [H]}}


NODE_CLASS_MAPPINGS = {"AGSoft_Aspect_Ratio": AGSoft_Aspect_Ratio}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Aspect_Ratio": "📐AGSoft Aspect Ratio"}