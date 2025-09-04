import torch

class AGSoft_Empty_Latent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["Preset", "Custom"],),
                "preset": (["512x512", "640x640", "768x768", "1024x1024", "512x768", "512x1024", "768x1024", "768x1280", "768x1344", "832x1152", "832x1216", "896x1088", "896x1152", "960x1024", "960x1088"],),
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
                "512x512": (512, 512),
                "640x640": (640, 640),
                "768x768": (768, 768),
                "1024x1024": (1024, 1024),
                "512x768": (512, 768),
                "512x1024": (512, 1024),
                "768x1024": (768, 1024),
                "768x1280": (768, 1280),
                "768x1344": (768, 1344),
                "832x1152": (832, 1152),
                "832x1216": (832, 1216),
                "896x1088": (896, 1088),
                "896x1152": (896, 1152),
                "960x1024": (960, 1024),
                "960x1088": (960, 1088),
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
    "AGSoft_Empty_Latent": AGSoft_Empty_Latent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Empty_Latent": "AGSoft Empty Latent"
}