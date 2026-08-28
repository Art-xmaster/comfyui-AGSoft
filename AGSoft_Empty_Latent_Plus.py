# ==============================================================================
# AGSoft_Empty_Latent_Plus.py
# ==============================================================================
# Нода: 🧊AGSoft Empty Latent Plus
# Описание / Description:
# Универсальный пустой латент: одна нода заменяет AGSoft Empty Latent /
# QwenImage / Flux.2 / Krea2. Тип модели сам выбирает число каналов и фактор
# сжатия латента. Движок размера заимствован из 📐AGSoft Aspect Ratio:
# пропорции (пресеты + custom), якоря width/height/longest/shortest/megapixels,
# выравнивание под кратность (floor/round/ceil); плюс готовые Preset-размеры
# и ручной режим Custom W×H.
# Universal empty latent: one node replaces AGSoft Empty Latent / QwenImage /
# Flux.2 / Krea2. The model type picks latent channels & compression factor.
# The size engine is borrowed from 📐AGSoft Aspect Ratio: ratio presets +
# custom, anchors width/height/longest/shortest/megapixels, multiple alignment
# (floor/round/ceil); plus ready Preset sizes and a manual Custom W×H mode.
#
# Возможности / Features:
# ⚡ Одна нода вместо нескольких: тип модели задаёт каналы и фактор сжатия.
#   One node instead of several: model type sets channels & latent factor.
# ⚡ Три режима размера: Ratio (пропорция+якорь) / Preset (готовые WxH) /
#   Custom (вручную). / Three size modes: Ratio / Preset / Custom.
# ⚡ Движок размера из 📐AGSoft Aspect Ratio: пресеты пропорций + custom,
#   якоря, мегапиксели, кратность. / Size engine from 📐AGSoft Aspect Ratio.
# ⚡ Округление floor/round/ceil + финальная гарантия кратности фактору латента.
#   floor/round/ceil rounding + final latent-factor divisibility guarantee.
# ⚡ Выходы: latent, width_px, height_px, display (+ ui для живой строки в JS).
#   Outputs: latent, width_px, height_px, display (+ ui for the live JS line).
#
# Автор / Author: AGSoft
# Дата / Date: 28.08.2026
# ==============================================================================

import torch
import math

RATIO_PRESETS = [
    "1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9",
    "4:5", "3:4", "2:3", "9:16", "1:2", "9:21", "custom",
]
BASE_MODES = ["width", "height", "longest", "shortest", "megapixels"]
ROUND_MODES = ["floor", "round", "ceil"]

# Пресеты готовых размеров (как в старых AGSoft Empty Latent): "Ориентация - WxH (пропорция)".
# W/H/пропорция не хранятся картой — парсятся из имени (_parse_size_preset).
SIZE_PRESETS = [
    # --- Square ---
    "Square - 448x448 (1:1)",
    "Square - 512x512 (1:1)",
    "Square - 576x576 (1:1)",
    "Square - 640x640 (1:1)",
    "Square - 768x768 (1:1)",
    "Square - 896x896 (1:1)",
    "Square - 1024x1024 (1:1)",
    "Square - 1152x1152 (1:1)",
    "Square - 1280x1280 (1:1)",
    "Square - 1440x1440 (1:1)",
    "Square - 1536x1536 (1:1)",
    "Square - 1920x1920 (1:1)",
    "Square - 2048x2048 (1:1)",
    # --- Portrait ---
    "Portrait - 384x512 (3:4)",
    "Portrait - 480x640 (3:4)",
    "Portrait - 512x768 (2:3)",
    "Portrait - 512x1024 (1:2)",
    "Portrait - 720x1280 (9:16)",
    "Portrait - 768x1024 (3:4)",
    "Portrait - 768x1152 (2:3)",
    "Portrait - 768x1280 (3:5)",
    "Portrait - 768x1344 (9:16)",
    "Portrait - 816x1920 (21:9)",
    "Portrait - 832x1152 (3:4)",
    "Portrait - 832x1216 (13:19)",
    "Portrait - 864x1152 (3:4)",
    "Portrait - 896x1088 (14:17)",
    "Portrait - 896x1152 (7:9)",
    "Portrait - 896x1344 (2:3)",
    "Portrait - 896x1536 (7:12)",
    "Portrait - 960x1024 (15:16)",
    "Portrait - 960x1088 (15:17)",
    "Portrait - 960x1280 (3:4)",
    "Portrait - 1024x1280 (4:5)",
    "Portrait - 1024x1536 (2:3)",
    "Portrait - 1080x1920 (9:16)",
    "Portrait - 1088x1856 (~6:10)",
    "Portrait - 1088x1920 (17:30)",
    "Portrait - 1280x1536 (5:6)",
    "Portrait - 1280x1920 (2:3)",
    "Portrait - 1344x1728 (7:9)",
    "Portrait - 1440x1920 (3:4)",
    "Portrait - 1440x2560 (9:16)",
    "Portrait - 1536x2048 (3:4)",
    # --- Landscape ---
    "Landscape - 512x384 (4:3)",
    "Landscape - 640x480 (4:3)",
    "Landscape - 768x512 (3:2)",
    "Landscape - 832x480 (16:9)",
    "Landscape - 1024x512 (2:1)",
    "Landscape - 1024x768 (4:3)",
    "Landscape - 1024x960 (16:15)",
    "Landscape - 1088x896 (17:14)",
    "Landscape - 1088x960 (17:15)",
    "Landscape - 1152x768 (3:2)",
    "Landscape - 1152x832 (9:7)",
    "Landscape - 1152x896 (9:7)",
    "Landscape - 1152x704 (16:9)",
    "Landscape - 1216x832 (19:13)",
    "Landscape - 1280x720 (16:9)",
    "Landscape - 1280x768 (5:3)",
    "Landscape - 1280x864 (4:3)",
    "Landscape - 1280x960 (4:3)",
    "Landscape - 1280x1024 (5:4)",
    "Landscape - 1344x768 (7:4)",
    "Landscape - 1344x896 (3:2)",
    "Landscape - 1360x768 (~16:9)",
    "Landscape - 1536x1024 (3:2)",
    "Landscape - 1536x1280 (6:5)",
    "Landscape - 1600x900 (16:9)",
    "Landscape - 1728x1344 (9:7)",
    "Landscape - 1792x1024 (7:4)",
    "Landscape - 1856x1088 (~16:9)",
    "Landscape - 1920x1024 (15:8)",
    "Landscape - 1920x1080 (16:9)",
    "Landscape - 1920x1280 (3:2)",
    "Landscape - 1920x1440 (4:3)",
    "Landscape - 1920x816 (20:9)",
    "Landscape - 2048x768 (8:3)",
    "Landscape - 2048x1152 (16:9)",
    "Landscape - 2560x1080 (21:9)",
    "Landscape - 3840x2160 (16:9)",
]

def _parse_size_preset(name):
    """'Portrait - 896x1152 (3:4)' -> (896, 1152, '3:4'). Понимает и '×'."""
    s = str(name).strip()
    if " - " not in s or " (" not in s:
        raise ValueError(f"Неверный формат пресета размера: '{name}'")
    dims = s.split(" - ", 1)[1].split(" (", 1)[0].replace("×", "x").lower()
    ratio = s.rsplit("(", 1)[1].rstrip(")")
    w, h = dims.split("x", 1)
    return int(w), int(h), ratio


# Тип модели -> каналы латента и фактор сжатия (latent = pixels / factor).
# Одинаковые по форме латенты объединены в одну группу.
MODEL_PRESETS = {
    "SD1.5 / SDXL":                     {"channels": 4,  "factor": 8},
    "SD3 / FLUX.1 / Krea2 / QwenImage": {"channels": 16, "factor": 8},
    "FLUX.2 / Flux2-klein":             {"channels": 128, "factor": 16},
}

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

class AGSoft_Empty_Latent_Plus:
    DESCRIPTION = (
        "🧊 AGSoft Empty Latent Plus.\n"
        "Universal empty latent for SD1.5/SDXL, SD3/FLUX.1, FLUX.2, Krea2, QwenImage.\n"
        "Size engine from 📐AGSoft Aspect Ratio: ratio presets/custom, anchors, megapixels, multiple alignment; plus ready Presets and Custom W×H.\n"
        "Outputs latent + real pixel W/H + display string.\n"
        "---\n"
        "🧊 AGSoft Empty Latent Plus.\n"
        "Универсальный пустой латент для SD1.5/SDXL, SD3/FLUX.1, FLUX.2, Krea2, QwenImage.\n"
        "Движок размера из 📐AGSoft Aspect Ratio: пропорции, якоря, мегапиксели, кратность; плюс готовые Preset и Custom W×H.\n"
        "Выдаёт латент + реальные W/H в пикселях + строку display."
    )

    CATEGORY = "AGSoft/nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (list(MODEL_PRESETS.keys()), {
                    "default": "SD3 / FLUX.1 / Krea2 / QwenImage",
                    "tooltip": "Latent format: channels & compression factor.\nSD1.5/SDXL=4ch·8, SD3/FLUX.1/Krea2/Qwen=16ch·8, FLUX.2=128ch·16.\n---\nФормат латента: каналы и фактор сжатия.\nSD1.5/SDXL=4ch·8, SD3/FLUX.1/Krea2/Qwen=16ch·8, FLUX.2=128ch·16."}),
                "size_mode": (["Ratio", "Preset", "Custom"], {
                    "default": "Ratio",
                    "tooltip": "How the size is set: Ratio = proportion+anchor, Preset = ready sizes, Custom = manual W×H.\n---\nКак задаётся размер: Ratio = пропорция+якорь, Preset = готовые размеры, Custom = вручную W×H."}),
                "size_preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Ready-made resolution (orientation, WxH, ratio). Used only when size_mode = Preset.\n---\nГотовое разрешение (ориентация, WxH, пропорция). Работает только при size_mode = Preset."}),
                "ratio_preset": (RATIO_PRESETS, {
                    "default": "1:1",
                    "tooltip": "Aspect ratio preset (W:H). Choose 'custom' to enter your own in custom_ratio.\n---\nПресет пропорции (W:H). Выберите 'custom', чтобы задать свою в custom_ratio."}),
                "custom_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": "Your own aspect ratio (W:H), e.g. 16:9, 1.85:1, 2.39:1. Used only when ratio_preset = custom.\n---\nСвоя пропорция (W:H), напр. 16:9, 1.85:1, 2.39:1. Работает только при ratio_preset = custom."}),
                "base": (BASE_MODES, {
                    "default": "width",
                    "tooltip": "Which side/value is fixed: width / height / longest / shortest side or total megapixels.\n---\nЧто фиксировано: width / height / длинная / короткая сторона или суммарные мегапиксели."}),
                "base_value": ("FLOAT", {
                    "default": 1024.0, "min": 1.0, "max": 16384.0, "step": 1.0,
                    "tooltip": "Value (px) of the fixed side for width/height/longest/shortest anchors.\n---\nЗначение (px) фиксируемой стороны для якорей width/height/longest/shortest."}),
                "megapixels_value": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1,
                    "tooltip": "Total image size in megapixels (step 0.1). Used only when base = megapixels.\n---\nОбщий размер в мегапикселях (шаг 0.1). Только при base = megapixels."}),
                "width": ("INT", {
                    "default": 1024, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Latent width in pixels. Used only when size_mode = Custom.\n---\nШирина в пикселях. Работает только при size_mode = Custom."}),
                "height": ("INT", {
                    "default": 1024, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Latent height in pixels. Used only when size_mode = Custom.\n---\nВысота в пикселях. Работает только при size_mode = Custom."}),
                "multiple": ("INT", {
                    "default": 64, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Align W/H to a multiple (8/32/64/112). 1 = no rounding.\n---\nВыравнивание W/H под кратность (8/32/64/112). 1 = без округления."}),
                "rounding": (ROUND_MODES, {
                    "default": "round",
                    "tooltip": "Rounding method when aligning to the multiple: floor / round / ceil.\n---\nМетод округления при выравнивании: floor (вниз) / round (ближайшее) / ceil (вверх)."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of empty latent images in the batch.\n---\nКоличество пустых латентных изображений в батче."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width_px", "height_px", "display")
    FUNCTION = "generate"

    def generate(self, model_type, size_mode, size_preset,
                 ratio_preset, custom_ratio, base, base_value, megapixels_value,
                 width, height, multiple, rounding, batch_size):
        # Если в старом воркфлоу остался неизвестный key — берём первый пресет.
        mp = MODEL_PRESETS.get(model_type) or next(iter(MODEL_PRESETS.values()))
        ch, factor = mp["channels"], mp["factor"]

        if size_mode == "Custom":
            w, h = float(width), float(height)
            label = "custom"
        elif size_mode == "Preset":
            pw, ph, label = _parse_size_preset(size_preset)
            w, h = float(pw), float(ph)
        else:
            ratio_text = custom_ratio if ratio_preset == "custom" else ratio_preset
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
            else:  # megapixels
                target = float(megapixels_value) * 1_000_000
                w = math.sqrt(target * ratio)
                h = math.sqrt(target / ratio)
            label = ratio_text

        W = _to_multiple(w, multiple, rounding)
        H = _to_multiple(h, multiple, rounding)
        # Гарантия кратности фактору латента (как финальная проверка в старых нодах).
        W = max(factor, (W // factor) * factor)
        H = max(factor, (H // factor) * factor)
        lw, lh = W // factor, H // factor

        latent = torch.zeros([int(batch_size), ch, lh, lw], device="cpu")
        display = f"{W}×{H} ({label}) → latent {lw}×{lh} · ch {ch}"
        return {"result": ({"samples": latent}, W, H, display),
                "ui": {"display": [display], "width": [W], "height": [H]}}

NODE_CLASS_MAPPINGS = {"AGSoft_Empty_Latent_Plus": AGSoft_Empty_Latent_Plus}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Empty_Latent_Plus": "🧊AGSoft Empty Latent Plus"}