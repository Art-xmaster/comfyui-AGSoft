"""
==============================================================================
AGSoft_Empty_Latent_Plus.py
==============================================================================
Нода: 🧊AGSoft Empty Latent Plus
Описание / Description:
Универсальный генератор пустого латента. Формат латента (число каналов и
фактор пространственного сжатия) определяется автоматически из подключённого
VAE — ручной выбор типа модели не нужен. Три режима задания размера
(Ratio / Preset / Custom), якорь пропорции, инверсия ориентации, выравнивание
под кратность и финальная гарантия кратности фактору латента.
Universal empty latent generator. The latent format (channel count and
spatial compression factor) is inferred automatically from the connected
VAE - no manual model-type selector is needed. Three size modes
(Ratio / Preset / Custom), a ratio anchor, orientation inversion, multiple
alignment and a final divisibility guarantee by the latent factor.
Возможности / Features:
⚡ VAE auto: каналы из latent_channels / z_channels / z_dim / embed_dim,
фактор из downscale_ratio и методов сжатия; fallback 4 канала / фактор 8.
VAE auto: channels from latent_channels / z_channels / z_dim / embed_dim,
factor from downscale_ratio and compression methods; fallback 4 ch / factor 8.
⚡ Режимы: Ratio (пропорция + якорь), Preset (готовые WxH), Custom (свои W×H).
Modes: Ratio (ratio + anchor), Preset (ready WxH), Custom (manual W×H).
⚡ Якоря: width / height / longest / shortest / megapixels.
Anchors: width / height / longest / shortest / megapixels.
⚡ invert_orientation меняет W/H местами во всех трёх режимах.
invert_orientation swaps W/H in all three modes.
⚡ Выравнивание под кратность: floor / round (банковский) / ceil.
Multiple alignment: floor / round (banker's) / ceil.
⚡ В Custom шаг стрелок width/height равен multiple (JS).
In Custom the width/height arrow step equals multiple (JS).
⚡ Статусная строка: "1344×768 • 1.03MP • ~16:9 • latent 168×96 • ch 4".
Status line: "1344×768 • 1.03MP • ~16:9 • latent 168×96 • ch 4".
⚡ Выходы: latent, width_px, height_px, display.
Outputs: latent, width_px, height_px, display.
⚡ JS: живая инфострока, автоскрытие виджетов, свободная ширина ноды,
автовысота, поддержка Classic и Nodes 2.0.
JS: live info line, auto-hiding widgets, free node width, auto height,
Classic and Nodes 2.0 support.
Автор / Author: AGSoft
Дата / Date: 24.11.2026
==============================================================================
"""
import math
import logging
from typing import Any, Dict, Tuple

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Empty Latent Plus] v11.24 loaded (VAE auto spec, 3 size modes, MP status line, auto-hide UI)")

RATIO_PRESETS = [
    "1:1", "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9",
    "4:5", "3:4", "2:3", "9:16", "1:2", "9:21", "custom",
]
BASE_MODES = ["width", "height", "longest", "shortest", "megapixels"]
ROUND_MODES = ["floor", "round", "ceil"]
SIZE_MODES = ["Ratio", "Preset", "Custom"]

# Типовые пропорции для статусной строки (порядок важен при равенстве).
# Common ratios for the status line (order matters on ties).
RATIO_LABELS = (
    (1, 1), (5, 4), (4, 3), (3, 2), (16, 10), (16, 9), (2, 1), (21, 9),
    (4, 5), (3, 4), (2, 3), (5, 8), (9, 16), (1, 2), (9, 21),
)

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


def _pick(obj: Any, *names: str) -> Any:
    """
    First non-empty field from a fallback chain of names.
    Первое непустое поле из цепочки имён-кандидатов.
    """
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _int_or(value: Any, default: int) -> int:
    """
    Positive int from scalars or single-element lists.
    Положительное целое из скаляра или списка из одного элемента.
    """
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) > 0 else None
        number = int(float(value))
        return number if number > 0 else default
    except (TypeError, ValueError, IndexError):
        return default


def _get_vae_spec(vae: Any) -> Tuple[int, int]:
    """
    Latent channels and spatial factor; latent_dim is dimensionality, not channels.
    Каналы латента и пространственный фактор; latent_dim — размерность, не каналы.
    """
    if vae is None:
        return 4, 8
    roots = [vae]
    if hasattr(vae, "vae"):
        roots.append(vae.vae)
    if hasattr(vae, "model"):
        roots.append(vae.model)
    if hasattr(vae, "first_stage_model"):
        roots.append(vae.first_stage_model)
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
            for sub_name in ("config", "encoder", "decoder"):
                sub_child = getattr(child, sub_name, None)
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
            channels = _pick(obj, "latent_channels", "z_channels", "z_dim", "embed_dim")
        if factor is None:
            factor = _pick(
                obj,
                "downscale_ratio",
                "spatial_compression_ratio",
                "latent_downscale_ratio",
                "compression_ratio",
                "downscale_factor",
            )
    # Modern VAE API exposes the factor as a method.
    # Современный API VAE отдаёт фактор методом.
    if factor is None and hasattr(vae, "spacial_compression_encode"):
        try:
            factor = vae.spacial_compression_encode()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("[AGSoft Empty Latent Plus] compression probe skipped: %s", exc)
    ch = _int_or(channels, 4)
    fac = _int_or(factor, 8)
    # Sanity bounds keep a wrong probe from breaking the latent.
    # Границы разумного не дают ошибочному зонду сломать латент.
    if ch > 128:
        ch = 4
    if fac > 64:
        fac = 8
    return ch, fac


def _parse_ratio(text: Any) -> Tuple[float, float]:
    """
    Parse '16:9', '16x9', '1920x1080', '1.85:1', '2.39:1'.
    Разбор '16:9', '16x9', '1920x1080', '1.85:1', '2.39:1'.
    """
    s = str(text).strip()
    for sep in ("x", "X", "/", ",", ";"):
        s = s.replace(sep, ":")
    parts = [p for p in s.split(":") if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid ratio format / Неверный формат пропорции: '{text}' (expected W:H / ожидалось W:H)"
        )
    try:
        w, h = float(parts[0]), float(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid ratio numbers / Неверные числа пропорции: '{text}'"
        ) from exc
    if not math.isfinite(w) or not math.isfinite(h) or w <= 0 or h <= 0:
        raise ValueError(
            f"Ratio values must be > 0 / Значения пропорции должны быть > 0: '{text}'"
        )
    return w, h


def _parse_size_preset(name: Any) -> Tuple[int, int, str]:
    """
    Read 'Orientation - WxH (ratio)' presets.
    Чтение пресетов вида 'Ориентация - WxH (пропорция)'.
    """
    s = str(name).strip()
    if "-" not in s or "(" not in s:
        raise ValueError(f"Invalid size preset / Неверный пресет размера: '{name}'")
    dims = s.split("-", 1)[1].split("(", 1)[0].replace("×", "x").lower()
    ratio = s.rsplit("(", 1)[1].rstrip(")").strip()
    parts = dims.split("x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid size preset / Неверный пресет размера: '{name}'")
    try:
        w, h = int(parts[0].strip()), int(parts[1].strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid size preset / Неверный пресет размера: '{name}'") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size preset / Неверный пресет размера: '{name}'")
    return w, h, ratio


def _to_multiple(value: float, multiple: int, mode: str) -> int:
    """
    Align one side, banker's rounding for 'round'.
    Выравнивание стороны, банковское округление для 'round'.
    """
    v = max(1.0, float(value))
    m = max(1, int(multiple))
    if m <= 1:
        return int(round(v))
    if mode == "floor":
        return max(m, int(math.floor(v / m)) * m)
    if mode == "ceil":
        return max(m, int(math.ceil(v / m)) * m)
    return max(m, int(round(v / m)) * m)


def _ratio_text(width: int, height: int) -> str:
    """
    Nearest common ratio for the status line, '~' marks an approximation.
    Ближайшая типовая пропорция для статусной строки, '~' — приближение.
    """
    target = width / height
    best = RATIO_LABELS[0]
    best_err = None
    for rw, rh in RATIO_LABELS:
        err = abs(rw / rh - target)
        if best_err is None or err < best_err - 1e-12:
            best_err = err
            best = (rw, rh)
    prefix = "" if best_err < 1e-9 else "~"
    return f"{prefix}{best[0]}:{best[1]}"


class AGSoft_Empty_Latent_Plus:
    DESCRIPTION = (
        "🧊 AGSoft Empty Latent Plus - universal empty latent.\n"
        "Channels and the spatial compression factor are inferred from the connected VAE.\n"
        "Size modes: Ratio (ratio + anchor), Preset (ready WxH), Custom (manual W×H).\n"
        "Anchors: width / height / longest / shortest / megapixels.\n"
        "invert_orientation swaps W/H in all modes; multiple/rounding align the size,\n"
        "then W/H are guaranteed divisible by the latent factor.\n"
        "In Custom mode the width/height arrow step follows 'multiple'.\n"
        "Status line: 'W×H • N.NNMP • ~ratio • latent lw×lh • ch N'.\n"
        "Outputs: latent, width_px, height_px, display.\n"
        "---\n"
        "🧊 AGSoft Empty Latent Plus - универсальный пустой латент.\n"
        "Каналы и фактор пространственного сжатия определяются из подключённого VAE.\n"
        "Режимы: Ratio (пропорция + якорь), Preset (готовые WxH), Custom (свои W×H).\n"
        "Якоря: width / height / длинная / короткая / мегапиксели.\n"
        "invert_orientation меняет W/H во всех режимах; multiple/rounding выравнивают размер,\n"
        "затем W/H гарантированно кратны фактору латента.\n"
        "В режиме Custom шаг стрелок width/height равен 'multiple'.\n"
        "Статусная строка: 'W×H • N.NNMP • ~пропорция • latent lw×lh • ch N'.\n"
        "Выходы: latent, width_px, height_px, display."
    )
    CATEGORY = "AGSoft/nodes"
    FUNCTION = "generate"
    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width_px", "height_px", "display")
    OUTPUT_TOOLTIPS = (
        "Empty latent tensor: batch_size x channels x lh x lw.\n"
        "---\nПустой тензор латента: batch_size x каналы x lh x lw.",
        "Final width in pixels.\n---\nИтоговая ширина в пикселях.",
        "Final height in pixels.\n---\nИтоговая высота в пикселях.",
        "Summary 'W×H • N.NNMP • ~ratio • latent lw×lh • ch N'.\n"
        "---\nИтог 'W×H • N.NNMP • ~пропорция • latent lw×lh • ch N'.",
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "vae": ("VAE", {
                    "tooltip": (
                        "VAE used to infer the latent format (channels and compression factor).\n"
                        "---\n"
                        "VAE, по которому определяются каналы и фактор сжатия латента."
                    ),
                }),
                "size_mode": (SIZE_MODES, {
                    "default": "Ratio",
                    "tooltip": (
                        "How the size is set: Ratio = ratio + anchor, Preset = ready sizes, "
                        "Custom = manual W×H.\n"
                        "---\n"
                        "Как задаётся размер: Ratio = пропорция + якорь, Preset = готовые размеры, "
                        "Custom = свои W×H."
                    ),
                }),
                "size_preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": (
                        "Ready-made resolution (orientation, WxH, ratio). "
                        "Used only when size_mode = Preset.\n"
                        "---\n"
                        "Готовое разрешение (ориентация, WxH, пропорция). "
                        "Только при size_mode = Preset."
                    ),
                }),
                "ratio_preset": (RATIO_PRESETS, {
                    "default": "1:1",
                    "tooltip": (
                        "Aspect ratio preset (W:H). 'custom' enables your own ratio.\n"
                        "---\n"
                        "Пресет пропорции (W:H). 'custom' включает свою пропорцию."
                    ),
                }),
                "custom_ratio": ("STRING", {
                    "default": "16:9",
                    "tooltip": (
                        "Your own ratio: 16:9, 1.85:1, 2.39:1, 16x9, 1920x1080. "
                        "Used only when ratio_preset = custom.\n"
                        "---\n"
                        "Своя пропорция: 16:9, 1.85:1, 2.39:1, 16x9, 1920x1080. "
                        "Только при ratio_preset = custom."
                    ),
                }),
                "base": (BASE_MODES, {
                    "default": "megapixels",
                    "tooltip": (
                        "What is fixed: width / height / longest / shortest side or total megapixels.\n"
                        "---\n"
                        "Что фиксировано: width / height / длинная / короткая сторона "
                        "или суммарные мегапиксели."
                    ),
                }),
                "base_value": ("FLOAT", {
                    "default": 1024.0, "min": 1.0, "max": 16384.0, "step": 1.0,
                    "tooltip": (
                        "Value (px) of the fixed side for width/height/longest/shortest.\n"
                        "---\n"
                        "Значение фиксируемой стороны (px) для width/height/longest/shortest."
                    ),
                }),
                "megapixels_value": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01,
                    "tooltip": (
                        "Total image area in megapixels. Used only when base = megapixels.\n"
                        "---\n"
                        "Общая площадь изображения в мегапикселях. Только при base = megapixels."
                    ),
                }),
                "width": ("INT", {
                    "default": 1024, "min": 1, "max": 8192, "step": 64,
                    "tooltip": (
                        "Manual width (size_mode = Custom). The arrow step equals 'multiple'.\n"
                        "---\n"
                        "Ручная ширина (size_mode = Custom). Шаг стрелок равен 'multiple'."
                    ),
                }),
                "height": ("INT", {
                    "default": 1024, "min": 1, "max": 8192, "step": 64,
                    "tooltip": (
                        "Manual height (size_mode = Custom). The arrow step equals 'multiple'.\n"
                        "---\n"
                        "Ручная высота (size_mode = Custom). Шаг стрелок равен 'multiple'."
                    ),
                }),
                "multiple": ("INT", {
                    "default": 64, "min": 1, "max": 128, "step": 1,
                    "tooltip": (
                        "Align W/H to a multiple (8/16/32/64). 1 = no alignment. "
                        "Also sets the width/height arrow step in Custom mode.\n"
                        "---\n"
                        "Выравнивание W/H под кратность (8/16/32/64). 1 = без выравнивания. "
                        "Также задаёт шаг стрелок width/height в режиме Custom."
                    ),
                }),
                "rounding": (ROUND_MODES, {
                    "default": "round",
                    "tooltip": (
                        "Rounding method: floor (down) / round (nearest) / ceil (up).\n"
                        "---\n"
                        "Метод округления: floor (вниз) / round (ближайшее) / ceil (вверх)."
                    ),
                }),
                "invert_orientation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Swap width and height. Works in all three size modes.\n"
                        "---\n"
                        "Поменять ширину и высоту местами. Работает во всех трёх режимах."
                    ),
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": (
                        "Number of empty latents in the batch.\n"
                        "---\n"
                        "Количество пустых латентов в батче."
                    ),
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs) -> Any:
        """
        Early validation before the graph runs.
        Ранняя проверка до запуска графа.
        """
        size_mode = kwargs.get("size_mode", "Ratio")
        if size_mode == "Preset":
            name = kwargs.get("size_preset")
            if isinstance(name, str):
                try:
                    _parse_size_preset(name)
                except (ValueError, TypeError, IndexError) as exc:
                    return str(exc)
        if size_mode == "Ratio":
            ratio_preset = kwargs.get("ratio_preset", "1:1")
            text = kwargs.get("custom_ratio", "16:9") if ratio_preset == "custom" else ratio_preset
            if isinstance(text, str):
                try:
                    _parse_ratio(text)
                except (ValueError, TypeError) as exc:
                    return str(exc)
        return True

    def generate(self, vae: Any, size_mode: str, size_preset: str, ratio_preset: str,
                 custom_ratio: str, base: str, base_value: float, megapixels_value: float,
                 width: int, height: int, multiple: int, rounding: str,
                 invert_orientation: bool, batch_size: int) -> Dict[str, Any]:
        ch, factor = _get_vae_spec(vae)
        try:
            if size_mode == "Custom":
                w, h = float(width), float(height)
            elif size_mode == "Preset":
                pw, ph, _ = _parse_size_preset(size_preset)
                w, h = float(pw), float(ph)
            else:
                ratio_text = custom_ratio if ratio_preset == "custom" else ratio_preset
                rw, rh = _parse_ratio(ratio_text)
                ratio_src = rw / rh
                val = float(base_value)
                if base == "width":
                    w, h = val, val / ratio_src
                elif base == "height":
                    w, h = val * ratio_src, val
                elif base == "longest":
                    if rw >= rh:
                        w, h = val, val / ratio_src
                    else:
                        w, h = val * ratio_src, val
                elif base == "shortest":
                    if rw <= rh:
                        w, h = val, val / ratio_src
                    else:
                        w, h = val * ratio_src, val
                else:
                    target = float(megapixels_value) * 1_000_000
                    w = math.sqrt(target * ratio_src)
                    h = math.sqrt(target / ratio_src)
        except (ValueError, TypeError, IndexError) as exc:
            logger.warning("[AGSoft Empty Latent Plus] %s", exc)
            raise ValueError(str(exc)) from exc
        if invert_orientation:
            w, h = h, w
        W = _to_multiple(w, multiple, rounding)
        H = _to_multiple(h, multiple, rounding)
        # Final guarantee: the latent grid must fit the VAE factor.
        # Финальная гарантия: сетка латента должна делиться на фактор VAE.
        W = max(factor, (W // factor) * factor)
        H = max(factor, (H // factor) * factor)
        lw, lh = W // factor, H // factor
        latent = torch.zeros([int(batch_size), ch, lh, lw], device="cpu")
        megapixels = (W * H) / 1_000_000.0
        display = (
            f"{W}×{H} • {megapixels:.2f}MP • {_ratio_text(W, H)} • "
            f"latent {lw}×{lh} • ch {ch}"
        )
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