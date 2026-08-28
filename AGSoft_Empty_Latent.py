# ==============================================================================
# AGSoft_Empty_Latent.py
# ==============================================================================
# Ноды: AGSoft Empty Latent / QwenImage / Flux.2 / Krea2 (все в одном файле).
# Исправления (по comfy/latent_formats.py из репо ComfyUI):
#   QwenImage: 16 каналов (формат Wan21), фактор 8,
#    добавлен виджет divisibility (default 64) в КОНЕЦ списка виджетов.
#   Flux.2: 128 каналов, фактор 16 (latent_channels=128, spacial_downscale_ratio=16).
#   SD1.5/SDXL: 4/8; Krea2: 16/8 - без изменений.
#   Пресеты как в AGSoft Empty Latent Plus: список SIZE_PRESETS + парсер
#    _parse_size_preset (старые size_map удалены; сохранённые старые пресеты
#    продолжают работать, W и H читаются из имени пресета).
#   Расположение виджетов и выходы НЕ менялись (compat со старыми воркфлоу).
# Автор: AGSoft
# Дата: 28.08.2026
# ==============================================================================

import torch
import math

# ==============================================================================
# ПРЕСЕТЫ РАЗМЕРОВ: "Ориентация - WxH (пропорция)". W/H парсятся из имени.
# ==============================================================================
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

# ==============================================================================
# НОДА: AGSoft Empty Latent (SD1.5 / SDXL) - 4 канала, фактор 8
# ==============================================================================
class AGSoft_Empty_Latent:
    """Создает пустой латентный тензор заданного размера и количества батчей."""
    CHANNELS = 4
    LATENT_FACTOR = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom", "Megapixels"], {
                    "tooltip": "Choose how to define the image size:\n- Preset: Use predefined sizes (Square/Portrait/Landscape)\n- Custom: Manually enter width and height\n- Megapixels: Specify target resolution in megapixels\n\nВыберите способ задания размера изображения:\n- Preset: Использовать предустановленные размеры\n- Custom: Вручную указать ширину и высоту\n- Megapixels: Задать разрешение в мегапикселях"}),
                "preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Select a predefined resolution and aspect ratio.\nВыберите предустановленное разрешение и соотношение сторон."}),
                "width": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Width in pixels (must be divisible by 8).\nШирина в пикселях (должна быть кратна 8)."}),
                "height": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Height in pixels (must be divisible by 8).\nВысота в пикселях (должна быть кратна 8)."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider",
                    "tooltip": "Target resolution in megapixels (e.g., 1.0 = 1,000,000 pixels).\nЦелевое разрешение в мегапикселях (например, 1.0 = 1 000 000 пикселей)."}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {
                    "tooltip": "Target aspect ratio for the megapixel-based resolution.\nЦелевое соотношение сторон для разрешения, заданного в мегапикселях."}),
                "divisibility": (["8", "16", "32", "64", "112", "128"], {
                    "default": "64",
                    "tooltip": "Ensure the final width and height are divisible by this value.\nГарантирует, что итоговые ширина и высота кратны этому числу."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of latent images to generate.\nКоличество латентных изображений для генерации."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width_px", "height_px", "width_latent", "height_latent")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/nodes"
    DESCRIPTION = (
        "Generates an empty latent tensor with specified dimensions.\n"
        "Supports presets, custom size, and megapixel-based resolution.\n\n"
        "Создает пустой латентный тензор с указанными размерами.\n"
        "Поддерживает пресеты, произвольный ввод и задание разрешения через мегапиксели."
    )

    def generate(self, size_mode, preset, width, height, megapixels, aspect_ratio, divisibility, batch_size):
        divisibility = int(divisibility)
        if size_mode == "Preset":
            width, height, _ = _parse_size_preset(preset)
        elif size_mode == "Custom":
            pass  # width/height already set
        elif size_mode == "Megapixels":
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
            target_pixels = megapixels * 1_000_000
            x = math.sqrt(target_pixels / (w_ratio * h_ratio))
            width = round(w_ratio * x / divisibility) * divisibility
            height = round(h_ratio * x / divisibility) * divisibility
            width = max(divisibility, width)
            height = max(divisibility, height)
        width = max(self.LATENT_FACTOR, (width // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        height = max(self.LATENT_FACTOR, (height // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        latent_width = width // self.LATENT_FACTOR
        latent_height = height // self.LATENT_FACTOR
        latent = torch.zeros([batch_size, self.CHANNELS, latent_height, latent_width], device="cpu")
        return ({"samples": latent}, width, height, latent_width, latent_height)

# ==============================================================================
# НОДА: AGSoft Empty Latent QwenImage - 16 каналов (формат Wan21), фактор 8
# ==============================================================================
class AGSoft_Empty_Latent_QwenImage:
    """Создает пустой латентный тензор с 16 каналами для моделей QwenImage."""
    CHANNELS = 16
    LATENT_FACTOR = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom", "Megapixels"], {
                    "tooltip": "Choose how to define the image size:\n- Preset: Use predefined sizes (Square/Portrait/Landscape)\n- Custom: Manually enter width and height\n- Megapixels: Specify target resolution in megapixels\n\nВыберите способ задания размера изображения:\n- Preset: Использовать предустановленные размеры\n- Custom: Вручную указать ширину и высоту\n- Megapixels: Задать разрешение в мегапикселях"}),
                "preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Select a predefined resolution and aspect ratio.\nВыберите предустановленное разрешение и соотношение сторон."}),
                "width": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Width in pixels (must be divisible by 8).\nШирина в пикселях (должна быть кратна 8)."}),
                "height": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Height in pixels (must be divisible by 8).\nВысота в пикселях (должна быть кратна 8)."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider",
                    "tooltip": "Target resolution in megapixels (e.g., 1.0 = 1,000,000 pixels).\nЦелевое разрешение в мегапикселях (например, 1.0 = 1 000 000 пикселей)."}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {
                    "tooltip": "Target aspect ratio for the megapixel-based resolution.\nЦелевое соотношение сторон для разрешения, заданного в мегапикселях."}),
                "divisibility": (["8", "16", "32", "64", "112", "128"], {
                    "default": "64",
                    "tooltip": "Ensure the final width and height are divisible by this value.\nГарантирует, что итоговые ширина и высота кратны этому числу."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of latent images to generate.\nКоличество латентных изображений для генерации."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width_px", "height_px", "width_latent", "height_latent")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/nodes"
    DESCRIPTION = (
        "Generates an empty latent tensor with 16 channels (Wan21 format) for QwenImage models.\n"
        "Supports presets, custom size, and megapixel-based resolution.\n\n"
        "Создает пустой латентный тензор с 16 каналами (формат Wan21) для моделей QwenImage.\n"
        "Поддерживает пресеты, произвольный ввод и задание разрешения через мегапиксели."
    )

    def generate(self, size_mode, preset, width, height, megapixels, aspect_ratio, divisibility, batch_size):
        divisibility = int(divisibility)
        if size_mode == "Preset":
            width, height, _ = _parse_size_preset(preset)
        elif size_mode == "Custom":
            pass  # width/height already set
        elif size_mode == "Megapixels":
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
            target_pixels = megapixels * 1_000_000
            x = math.sqrt(target_pixels / (w_ratio * h_ratio))
            width = round(w_ratio * x / divisibility) * divisibility
            height = round(h_ratio * x / divisibility) * divisibility
            width = max(divisibility, width)
            height = max(divisibility, height)
        width = max(self.LATENT_FACTOR, (width // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        height = max(self.LATENT_FACTOR, (height // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        latent_width = width // self.LATENT_FACTOR
        latent_height = height // self.LATENT_FACTOR
        latent = torch.zeros([batch_size, self.CHANNELS, latent_height, latent_width], device="cpu")
        return ({"samples": latent}, width, height, latent_width, latent_height)

# ==============================================================================
# НОДА: AGSoft Empty Latent Flux2 - 128 каналов, фактор 16
# ==============================================================================
class AGSoft_Empty_Latent_Flux2:
    """Создает пустой латентный тензор для моделей FLUX.2 (128 каналов, фактор 16)."""
    CHANNELS = 128
    LATENT_FACTOR = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom", "Megapixels"], {
                    "tooltip": "Choose how to define the image size:\n- Preset: Use predefined sizes (Square/Portrait/Landscape)\n- Custom: Manually enter width and height\n- Megapixels: Specify target resolution in megapixels\n\nВыберите способ задания размера изображения:\n- Preset: Использовать предустановленные размеры\n- Custom: Вручную указать ширину и высоту\n- Megapixels: Задать разрешение в мегапикселях"}),
                "preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Select a predefined resolution and aspect ratio.\nВыберите предустановленное разрешение и соотношение сторон."}),
                "width": ("INT", {
                    "default": 1024, "min": 16, "max": 4096, "step": 8,
                    "tooltip": "Width in pixels (must be divisible by 16 for FLUX.2).\nШирина в пикселях (должна быть кратна 16 для FLUX.2)."}),
                "height": ("INT", {
                    "default": 1024, "min": 16, "max": 4096, "step": 8,
                    "tooltip": "Height in pixels (must be divisible by 16 for FLUX.2).\nВысота в пикселях (должна быть кратна 16 для FLUX.2)."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Target resolution in megapixels (e.g., 1.0 = 1,000,000 pixels).\nЦелевое разрешение в мегапикселях (например, 1.0 = 1 000 000 пикселей)."}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {
                    "tooltip": "Target aspect ratio for the megapixel-based resolution.\nUsed only when Megapixels mode is selected.\n\nЦелевое соотношение сторон для разрешения, заданного в мегапикселях.\nИспользуется только при выборе режима Megapixels."}),
                "divisibility": (["8", "16", "32", "64", "112", "128"], {
                    "default": "64",
                    "tooltip": "Ensure the final width and height are divisible by this value.\nГарантирует, что итоговые ширина и высота кратны этому числу."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of latent images to generate.\nКоличество латентных изображений для генерации."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width_px", "height_px", "width_latent", "height_latent")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/nodes"
    DESCRIPTION = (
        "Generates an empty latent tensor with specified dimensions for FLUX.2 models (128 channels, factor 16).\n"
        "Supports presets, custom size, and megapixel-based resolution.\n\n"
        "Создает пустой латентный тензор для моделей FLUX.2 (128 каналов, фактор 16).\n"
        "Поддерживает пресеты, произвольный ввод и задание разрешения через мегапиксели."
    )

    def generate(self, size_mode, preset, width, height, megapixels, aspect_ratio, divisibility, batch_size):
        divisibility = int(divisibility)
        if size_mode == "Preset":
            width, height, _ = _parse_size_preset(preset)
        elif size_mode == "Custom":
            pass  # width/height already set
        elif size_mode == "Megapixels":
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
            target_pixels = megapixels * 1_000_000
            x = math.sqrt(target_pixels / (w_ratio * h_ratio))
            width = round(w_ratio * x / divisibility) * divisibility
            height = round(h_ratio * x / divisibility) * divisibility
            width = max(divisibility, width)
            height = max(divisibility, height)
        width = max(self.LATENT_FACTOR, (width // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        height = max(self.LATENT_FACTOR, (height // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        latent_width = width // self.LATENT_FACTOR
        latent_height = height // self.LATENT_FACTOR
        latent = torch.zeros([batch_size, self.CHANNELS, latent_height, latent_width], device="cpu")
        return ({"samples": latent}, width, height, latent_width, latent_height)

# ==============================================================================
# НОДА: AGSoft Empty Latent Krea2 - 16 каналов, фактор 8
# (перенесена из AGSoft_Empty_Latent_Krea2.py - старый файл удалить)
# ==============================================================================
class AGSoft_Empty_Latent_Krea2:
    """Создаёт пустой латентный тензор с 16 каналами для моделей Krea2."""
    CHANNELS = 16
    LATENT_FACTOR = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom", "Megapixels"], {
                    "tooltip": "Choose how to define the image size:\n- Preset: Use predefined sizes (Square/Portrait/Landscape)\n- Custom: Manually enter width and height\n- Megapixels: Specify target resolution in megapixels\n\n---\n\nВыберите способ задания размера изображения:\n- Preset: Использовать предустановленные размеры\n- Custom: Вручную указать ширину и высоту\n- Megapixels: Задать разрешение в мегапикселях"}),
                "preset": (SIZE_PRESETS, {
                    "default": "Square - 1024x1024 (1:1)",
                    "tooltip": "Select a predefined resolution and aspect ratio.\nВыберите предустановленное разрешение и соотношение сторон."}),
                "width": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Width in pixels (must be divisible by 8).\nШирина в пикселях (должна быть кратна 8)."}),
                "height": ("INT", {
                    "default": 1024, "min": 8, "max": 4096, "step": 8,
                    "tooltip": "Height in pixels (must be divisible by 8).\nВысота в пикселях (должна быть кратна 8)."}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Target resolution in megapixels (e.g., 1.0 = 1,000,000 pixels).\nЦелевое разрешение в мегапикселях (например, 1.0 = 1 000 000 пикселей)."}),
                "aspect_ratio": (["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {
                    "tooltip": "Target aspect ratio for the megapixel-based resolution.\nЦелевое соотношение сторон для разрешения, заданного в мегапикселях."}),
                "divisibility": (["8", "16", "32", "64", "128"], {
                    "default": "64",
                    "tooltip": "Ensure the final width and height are divisible by this value.\nГарантирует, что итоговые ширина и высота кратны этому числу."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Number of latent images to generate.\nКоличество латентных изображений для генерации."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width_px", "height_px", "width_latent", "height_latent")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/nodes"
    DESCRIPTION = (
        "Generates an empty latent tensor with 16 channels for Krea2 models.\n"
        "Supports presets, custom size, and megapixel-based resolution.\n\n"
        "Создаёт пустой латентный тензор с 16 каналами для моделей Krea2.\n"
        "Поддерживает пресеты, произвольный ввод и задание разрешения через мегапиксели."
    )

    def generate(self, size_mode, preset, width, height, megapixels, aspect_ratio, divisibility, batch_size):
        divisibility = int(divisibility)
        if size_mode == "Preset":
            width, height, _ = _parse_size_preset(preset)
        elif size_mode == "Custom":
            pass  # width/height already set
        elif size_mode == "Megapixels":
            w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
            target_pixels = megapixels * 1_000_000
            x = math.sqrt(target_pixels / (w_ratio * h_ratio))
            width = round(w_ratio * x / divisibility) * divisibility
            height = round(h_ratio * x / divisibility) * divisibility
            width = max(divisibility, width)
            height = max(divisibility, height)
        width = max(self.LATENT_FACTOR, (width // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        height = max(self.LATENT_FACTOR, (height // self.LATENT_FACTOR) * self.LATENT_FACTOR)
        latent_width = width // self.LATENT_FACTOR
        latent_height = height // self.LATENT_FACTOR
        latent = torch.zeros([batch_size, self.CHANNELS, latent_height, latent_width], device="cpu")
        return ({"samples": latent}, width, height, latent_width, latent_height)

# ==============================================================================
# РЕГИСТРАЦИЯ НОД
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "AGSoft_Empty_Latent": AGSoft_Empty_Latent,
    "AGSoft_Empty_Latent_QwenImage": AGSoft_Empty_Latent_QwenImage,
    "AGSoft_Empty_Latent_Flux2": AGSoft_Empty_Latent_Flux2,
    "AGSoft_Empty_Latent_Krea2": AGSoft_Empty_Latent_Krea2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Empty_Latent": "AGSoft Empty Latent",
    "AGSoft_Empty_Latent_QwenImage": "AGSoft Empty Latent QwenImage",
    "AGSoft_Empty_Latent_Flux2": "AGSoft Empty Latent Flux.2",
    "AGSoft_Empty_Latent_Krea2": "AGSoft Empty Latent Krea2",
}