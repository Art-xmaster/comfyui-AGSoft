import torch

class AGSoft_Empty_Latent_QwenImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom"],),
                "preset": ([ "1008x1008 (1:1)", 
                             "1120x1120 (1:1)", 
                             "1344x1344 (1:1)", 
                             "1568x1568 (1:1)", 
                             "1792x1792 (1:1)", 
                             "2016x2016 (1:1)", 
                             "896x1120 (4:5)", 
                             "896x1344 (2:3)", 
                             "1008x1344 (3:4)", 
                             "1008x1792 (9:16)", 
                             "1120x1456 (10:13)", 
                             "1120x1680 (2:3)", 
                             "1120x1792 (5:8)", 
                             "1120x1904 (10:17)",
                             "1120x2016 (5:6)",
                             "1120x2240 (1:2)", 
                             "1232x2016 (11:18)", 
                             "1232x2464 (1:2)", 
                             "1344x1792 (3:4)", 
                             "1344x2016 (2:3)", 
                             "1344x2240 (3:5)", 
                             "1344x2688 (1:2)", 
                             "1456x1792 (13:16)", 
                             "1456x2912 (1:2)",
                             "1568x2352 (2:3)", 
                             "1680x2016 (5:6)",
                             "1680x2240 (3:4)", 
                             "1792x2240 (4:5)"],),
                "width": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 8}),
                "invert": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width_px", "height_px", "width_latent", "height_latent")
    FUNCTION = "generate"
    CATEGORY = "AGSoft/nodes"

    def generate(self, size_mode, preset, width, height, invert, batch_size):
        # Определяем ширину и высоту
        if size_mode == "Preset":
            size_map = {
                "1008x1008 (1:1)": (1008, 1008),
                "1120x1120 (1:1)": (1120, 1120),
                "1344x1344 (1:1)": (1344, 1344),
                "1568x1568 (1:1)": (1568, 1568),
                "1792x1792 (1:1)": (1792, 1792),
                "2016x2016 (1:1)": (2016, 2016),
                "896x1120 (4:5)": (896, 1120),
                "896x1344 (2:3)": (896, 1344),
                "1008x1344 (3:4)": (1008, 1344),
                "1008x1792 (9:16)": (1008, 1792),
                "1120x1456 (10:13)": (1120, 1456),
                "1120x1680 (2:3)": (1120, 1680),
                "1120x1792 (5:8)": (1120, 1792),
                "1120x1904 (10:17)": (1120, 1904),
                "1120x2016 (5:6)": (1120, 2016),
                "1120x2240 (1:2)": (1120, 2240),
                "1232x2016 (11:18)": (1232, 2016),
                "1232x2464 (1:2)": (1232, 2464),
                "1344x1792 (3:4)": (1344, 1792),
                "1344x2016 (2:3)": (1344, 2016),
                "1344x2240 (3:5)": (1344, 2240),
                "1344x2688 (1:2)": (1344, 2688),
                "1456x1792 (13:16)": (1456, 1792),
                "1456x2912 (1:2)": (1456, 2912),
                "1568x2352 (2:3)": (1568, 2352),
                "1680x2016 (5:6)": (1680, 2016),
                "1680x2240 (3:4)": (1680, 2240),
                "1792x2240 (4:5)": (1792, 2240),
                 }
            width, height = size_map[preset]
        
        # Инвертируем размеры, если включено
        if invert:
            width, height = height, width
        
        # Проверяем кратность 8 (обязательно для Stable Diffusion)
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)

        # Рассчитываем размеры в латентном пространстве
        latent_width = width // 8
        latent_height = height // 8

        # Создаём пустой латентный тензор
        latent = torch.zeros([batch_size, 4, latent_height, latent_width])

        return ({
            "samples": latent
        }, width, height, latent_width, latent_height)

# Маппинг для ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Empty_Latent_QwenImage": AGSoft_Empty_Latent_QwenImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Empty_Latent_QwenImage": "AGSoft Empty Latent QwenImage"

}


