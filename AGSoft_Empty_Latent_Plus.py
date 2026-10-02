# ==============================================================================
# AGSoft_Empty_Latent_Plus.py
# ==============================================================================
# Node: 🧊AGSoft Empty Latent Plus
# Version: v1.03
#
# Universal empty latent generator for ComfyUI.
# The node receives a VAE input and automatically infers the latent format
# from it: number of channels and the spatial compression factor.
# No manual model-type selector is needed; any VAE-based pipeline is
# supported (SD1.5/SDXL, SD3/FLUX.1, FLUX.2, Krea2, QwenImage, etc.).
#
# SIZE MODES (size_mode):
#   Ratio  - aspect ratio (preset or custom W:H) plus a fixed anchor:
#            width / height / longest / shortest side or total megapixels.
#   Preset - ready-made resolutions in the form "Orientation - WxH (ratio)".
#   Custom - manual width and height in pixels.
#
# OPTIONS:
#   invert_orientation - swaps width and height; active in all three modes.
#   multiple           - align W/H to a multiple (8/32/64/112); 1 = off.
#   rounding           - alignment method: floor / round (banker's) / ceil.
#   A final guarantee is applied afterwards: W/H are always divisible by the
#   latent compression factor.
#
# OUTPUTS:
#   latent    - empty latent tensor (batch_size x channels x lh x lw).
#   width_px  - final width in pixels.
#   height_px - final height in pixels.
#   display   - summary string "W×H (ratio) → latent lw×lh · ch N".
#   ui        - the same data for the live info line in the JS frontend.
#
# FRONTEND BEHAVIOUR (see AGSoft_Empty_Latent_Plus.js):
#   - the live line shows "W×H (ratio) · VAE auto" before execution and the
#     exact display string after execution;
#   - the last result is stored in the workflow JSON and survives ComfyUI tab
#     switching and workflow reloads; it resets only on manual change;
#   - unused widgets are hidden automatically per size_mode.
#
# ------------------------------------------------------------------------------
#
# Нода: 🧊AGSoft Empty Latent Plus
# Версия: v1.03
#
# Универсальный генератор пустого латента для ComfyUI.
# Нода принимает вход VAE и автоматически определяет по нему формат латента:
# число каналов и фактор сжатия. Ручной выбор типа модели не нужен;
# поддерживаются любые VAE-пайплайны (SD1.5/SDXL, SD3/FLUX.1, FLUX.2, Krea2,
# QwenImage и т.д.).
#
# РЕЖИМЫ РАЗМЕРА (size_mode):
#   Ratio  - пропорция (пресет или своя W:H) плюс фиксированный якорь:
#            width / height / длинная / короткая сторона или мегапиксели.
#   Preset - готовые разрешения вида "Ориентация - WxH (пропорция)".
#   Custom - ручные ширина и высота в пикселях.
#
# ОПЦИИ:
#   invert_orientation - меняет ширину и высоту местами; активна во всех режимах.
#   multiple           - выравнивание W/H под кратность (8/32/64/112); 1 = выкл.
#   rounding           - метод выравнивания: floor / round (банковский) / ceil.
#   Затем применяется финальная гарантия: W/H всегда кратны фактору сжатия.
#
# ВЫХОДЫ:
#   latent    - пустой тензор латента (batch_size x channels x lh x lw).
#   width_px  - итоговая ширина в пикселях.
#   height_px - итоговая высота в пикселях.
#   display   - строка "W×H (пропорция) → latent lw×lh · ch N".
#   ui        - те же данные для живой инфостроки в JS-фронтенде.
#
# ПОВЕДЕНИЕ ФРОНТЕНДА (см. AGSoft_Empty_Latent_Plus.js):
#   - живая строка показывает "W×H (пропорция) · VAE auto" до выполнения
#     и точную строку display после выполнения;
#   - последний результат сохраняется в JSON воркфлоу и переживает
#     переключение вкладок и перезагрузку воркфлоу; сброс — только при ручном
#     изменении параметров;
#   - неиспользуемые виджеты скрываются автоматически по size_mode.
#
# Author: AGSoft
# Date: 02.10.2026
# ==============================================================================

import torch
import math


RATIO_PRESETS = [
    "1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9",
    "4:5", "3:4", "2:3", "9:16", "1:2", "9:21", "custom",
]

BASE_MODES = ["width", "height", "longest", "shortest", "megapixels"]

ROUND_MODES = ["floor", "round", "ceil"]

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
    s = str(name).strip()

    if "-" not in s or "(" not in s:
        raise ValueError(f"Неверный формат пресета размера: '{name}'")

    dims = s.split("-", 1)[1].split("(", 1)[0].replace("×", "x").lower()
    ratio = s.rsplit("(", 1)[1].rstrip(")").strip()

    w, h = dims.split("x", 1)

    return int(w.strip()), int(h.strip()), ratio


def _pick(obj, *names):
    if obj is None:
        return None

    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)

        if value is not None:
            return value

    return None


def _int_or(value, default):
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) > 0 else None

        value = int(float(value))

        return value if value > 0 else default
    except Exception:
        return default


def _get_vae_spec(vae):
    """
    Каналы латента и фактор сжатия из подключённого VAE.
    Fallback: 4 channels, factor 8.
    """
    if vae is None:
        return 4, 8

    roots = [
        vae,
        getattr(vae, "vae", None),
        getattr(vae, "model", None),
        getattr(vae, "first_stage_model", None),
    ]

    objects = []

    for root in roots:
        if root is None:
            continue

        objects.append(root)

        for attr_name in ("config", "model_config", "encoder", "decoder", "first_stage_model"):
            child = getattr(root, attr_name, None)

            if child is None:
                continue

            objects.append(child)

            for sub_attr_name in ("config", "encoder", "decoder"):
                sub_child = getattr(child, sub_attr_name, None)

                if sub_child is not None:
                    objects.append(sub_child)

    for obj in list(objects):
        if isinstance(obj, dict):
            for key in ("encoder", "decoder", "vae_config", "unet_config"):
                nested = obj.get(key)

                if nested is not None:
                    objects.append(nested)

    channels = None
    factor = None

    for obj in objects:
        if channels is None:
            channels = _pick(obj, "latent_channels", "z_channels", "latent_dim", "output_channels")

        if factor is None:
            factor = _pick(
                obj,
                "downscale_ratio",
                "spatial_compression_ratio",
                "latent_downscale_ratio",
                "compression_ratio",
                "downscale_factor",
            )

    return _int_or(channels, 4), _int_or(factor, 8)


def _parse_ratio(text):
    s = str(text).strip()

    for sep in ("x", "X", "/", ",", ";"):
        s = s.replace(sep, ":")

    parts = [p for p in s.split(":") if p.strip() != ""]

    if len(parts) != 2:
        raise ValueError(f"Неверный формат пропорции: '{text}' (ожидалось W:H)")

    w = float(parts[0])
    h = float(parts[1])

    if w <= 0 or h <= 0:
        raise ValueError(f"Значения пропорции должны быть > 0: '{text}'")

    return w, h


def _to_multiple(value, multiple, mode):
    v = max(1.0, float(value))
    multiple = int(multiple)

    if multiple <= 1:
        return int(round(v))

    if mode == "floor":
        return max(multiple, int(math.floor(v / multiple)) * multiple)

    if mode == "ceil":
        return max(multiple, int(math.ceil(v / multiple)) * multiple)

    return max(multiple, int(round(v / multiple)) * multiple)


class AGSoft_Empty_Latent_Plus:
    DESCRIPTION = (
        "Universal empty latent: channels & compression factor are inferred from the connected VAE.\n"
        "Size modes: Ratio (ratio + anchor), Preset (ready WxH), Custom (manual W×H).\n"
        "Anchors: width / height / longest / shortest / megapixels; invert_orientation swaps W/H in all modes.\n"
        "multiple/rounding align the size (floor/round/ceil) with a final latent-factor divisibility guarantee.\n"
        "Outputs: latent, width_px, height_px, display.\n"
        "Live line: 'W×H (ratio) · VAE auto' before run, exact result after run.\n"
        "---\n"
        "Универсальный пустой латент: каналы и фактор сжатия определяются из подключённого VAE.\n"
        "Режимы: Ratio (пропорция + якорь), Preset (готовые WxH), Custom (свои W×H).\n"
        "Якоря: width / height / длинная / короткая / мегапиксели; invert_orientation меняет W/H во всех режимах.\n"
        "multiple/rounding выравнивают размер (floor/round/ceil) с финальной гарантией кратности фактору латента.\n"
        "Выходы: latent, width_px, height_px, display.\n"
        "Живая строка: 'W×H (пропорция) · VAE auto' до запуска, точный результат после."
    )

    CATEGORY = "AGSoft/nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {
                    "tooltip": "VAE used to infer the latent format (channels & compression factor).\n---\nVAE, по которому определяются каналы и фактор сжатия латента."}),

                "size_mode": (["Ratio", "Preset", "Custom"], {
                    "default": "Ratio",
                    "tooltip": "How the size is set:\nRatio = ratio + anchor, Preset = ready sizes, Custom = manual W×H.\n---\nКак задаётся размер:\nRatio = пропорция + якорь, Preset = готовые размеры, Custom = свои W×H."}),

                "size_preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Ready-made resolution (orientation, WxH, ratio).\nUsed only when size_mode = Preset.\n---\nГотовое разрешение (ориентация, WxH, пропорция).\nРаботает только при size_mode = Preset."}),

                "ratio_preset": (RATIO_PRESETS, {
                    "default": "1:1",
                    "tooltip": "Aspect ratio preset (W:H).\n'custom' enables your own ratio in custom_ratio.\n---\nПресет пропорции (W:H).\n'custom' включает свою пропорцию в custom_ratio."}),

                "custom_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": "Your own aspect ratio (W:H), e.g. 16:9, 1.85:1, 2.39:1.\nUsed only when ratio_preset = custom.\n---\nСвоя пропорция (W:H), напр. 16:9, 1.85:1, 2.39:1.\nРаботает только при ratio_preset = custom."}),

                "base": (BASE_MODES, {
                    "default": "megapixels",
                    "tooltip": "What is fixed: width / height / longest / shortest side\nor total megapixels.\n---\nЧто фиксировано: width / height / длинная / короткая сторона\nили суммарные мегапиксели."}),

                "base_value": ("FLOAT", {
                    "default": 1024.0, "min": 1.0, "max": 16384.0, "step": 1.0,
                    "tooltip": "Value (px) of the fixed side.\nUsed for width/height/longest/shortest anchors.\n---\nЗначение фиксируемой стороны (px).\nДля якорей width/height/longest/shortest."}),

                "megapixels_value": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01,
                    "tooltip": "Total image size in megapixels.\nUsed only when base = megapixels.\n---\nОбщий размер изображения в мегапикселях.\nТолько при base = megapixels."}),

                "width": ("INT", {
                    "default": 1024, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Width in pixels.\nUsed only when size_mode = Custom.\n---\nШирина в пикселях.\nТолько при size_mode = Custom."}),

                "height": ("INT", {
                    "default": 1024, "min": 8, "max": 8192, "step": 8,
                    "tooltip": "Height in pixels.\nUsed only when size_mode = Custom.\n---\nВысота в пикселях.\nТолько при size_mode = Custom."}),

                "multiple": ("INT", {
                    "default": 64, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Align W/H to a multiple (8/16/32/64).\n1 = no alignment.\n---\nВыравнивание W/H под кратность (8/16/32/64).\n1 = без выравнивания."}),

                "rounding": (ROUND_MODES, {
                    "default": "round",
                    "tooltip": "Rounding method for the multiple:\nfloor (down) / round (nearest) / ceil (up).\n---\nМетод округления под кратность:\nfloor (вниз) / round (ближайшее) / ceil (вверх)."}),

                "invert_orientation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Swap width and height.\nWorks in all three size modes.\n---\nПоменять ширину и высоту местами.\nРаботает во всех трёх режимах."}),

                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of empty latents in the batch.\n---\nКоличество пустых латентов в батче."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width_px", "height_px", "display")
    FUNCTION = "generate"

    def generate(self, vae, size_mode, size_preset,
                 ratio_preset, custom_ratio, base, base_value, megapixels_value,
                 width, height, multiple, rounding, invert_orientation, batch_size):

        ch, factor = _get_vae_spec(vae)

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
                if rw >= rh:
                    w, h = val, val * rh / rw
                else:
                    h, w = val, val * rw / rh
            elif base == "shortest":
                if rw <= rh:
                    w, h = val, val * rh / rw
                else:
                    h, w = val, val * rw / rh
            else:
                target = float(megapixels_value) * 1_000_000
                w = math.sqrt(target * ratio)
                h = math.sqrt(target / ratio)

            label = ratio_text

        # Инверсия ориентации — во всех трёх режимах.
        if invert_orientation:
            w, h = h, w
            label = f"{label} ↕"

        W = _to_multiple(w, multiple, rounding)
        H = _to_multiple(h, multiple, rounding)

        W = max(factor, (W // factor) * factor)
        H = max(factor, (H // factor) * factor)

        lw, lh = W // factor, H // factor

        latent = torch.zeros([int(batch_size), ch, lh, lw], device="cpu")

        display = f"{W}×{H} ({label}) → latent {lw}×{lh} · ch {ch}"

        return {
            "result": ({"samples": latent}, W, H, display),
            "ui": {
                "display": [display],
                "width": [W],
                "height": [H],
                "channels": [ch],
                "factor": [factor],
            },
        }


NODE_CLASS_MAPPINGS = {"AGSoft_Empty_Latent_Plus": AGSoft_Empty_Latent_Plus}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Empty_Latent_Plus": "🧊AGSoft Empty Latent Plus"}