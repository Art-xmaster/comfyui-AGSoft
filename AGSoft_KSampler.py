# AGSoft_KSampler.py
# Автор: AGSoft
# Дата: 28 октября 2025 г.

import comfy.samplers
from nodes import common_ksampler
import json
from typing import Dict, Any, Tuple, Optional

class AGSoft_KSampler:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent.\nМодель, используемая для денойзинга входного латентного изображения."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise.\nСлучайный сид, используемый для генерации шума."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process.\nКоличество шагов, используемых в процессе денойзинга."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.\nМасштаб Classifier-Free Guidance (CFG) балансирует между креативностью и следованием промпту. Более высокие значения дают изображения, точнее соответствующие промпту, но чрезмерно высокие значения ухудшают качество."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.\nАлгоритм, используемый при сэмплировании; может влиять на качество, скорость и стиль генерируемого изображения."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image.\nПланировщик определяет, как шум постепенно удаляется для формирования изображения."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image.\nУсловие, описывающее атрибуты, которые вы хотите включить в изображение."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image.\nУсловие, описывающее атрибуты, которые вы хотите исключить из изображения."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise.\nЛатентное изображение для денойзинга."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.\nСтепень применяемого денойзинга; более низкие значения сохраняют структуру исходного изображения, что позволяет использовать режим image-to-image."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent into an image (optional).\nVAE-модель для декодирования латента в изображение (опционально)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "latent",
        "image",
        "seed_str",
        "steps_str",
        "cfg_str",
        "sampler_str",
        "scheduler_str",
        "denoise_str",
        "params_json"
    )
    OUTPUT_TOOLTIPS = (
        "The denoised latent.\nДеноизированный латент.",
        "The decoded image (if VAE is connected).\nДекодированное изображение (если подключен VAE).",
        "Seed as string.\nСид в виде строки.",
        "Steps as string.\nКоличество шагов в виде строки.",
        "CFG as string.\nЗначение CFG в виде строки.",
        "Sampler name.\nИмя сэмплера.",
        "Scheduler name.\nИмя планировщика.",
        "Denoise level as string.\nУровень денойзинга в виде строки.",
        "Full parameters as JSON.\nПолные параметры в формате JSON."
    )
    FUNCTION = "sample_with_params"
    CATEGORY = "AGSoft/Utility"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image. Includes built-in VAE decoding and parameter logging.\nИспользует указанную модель и позитивное/негативное условия для денойзинга латентного изображения. Включает встроенный VAE-декодер и логирование параметров."

    def sample_with_params(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        vae=None
    ):
        # 1. Сэмплирование
        result = common_ksampler(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent=latent_image,
            denoise=denoise
        )
        latent_dict = result[0]

        # 2. VAE Decode — только если vae передан
        if vae is not None:
            samples = latent_dict["samples"]
            images = vae.decode(samples)
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            # Создаём пустой тензор IMAGE (1x1x1x3), чтобы ComfyUI не ругался
            import torch
            images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        # 3. Подготавливаем строковые значения
        seed_str = str(seed)
        steps_str = str(steps)
        cfg_str = f"{cfg:.2f}"
        denoise_str = f"{denoise:.2f}"

        # 4. JSON с параметрами
        params_json = json.dumps({
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise
        }, ensure_ascii=False, indent=2)

        return (
            latent_dict,
            images,
            seed_str,
            steps_str,
            cfg_str,
            sampler_name,
            scheduler,
            denoise_str,
            params_json
        )

# Регистрация
NODE_CLASS_MAPPINGS = {"AGSoft_KSampler": AGSoft_KSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_KSampler": "AGSoft KSampler"}