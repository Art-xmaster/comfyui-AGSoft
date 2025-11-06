# AGSoft_Loop_Samplers.py
# Автор: AGSoft
# Дата: 28 октября 2025 г.

import comfy.samplers

class AGSoft_Loop_Samplers:
    DESCRIPTION = "Select up to 20 samplers for batch processing. Connect directly to KSampler.sampler_name.\nВыберите до 20 сэмплеров для пакетной обработки. Подключайте напрямую к KSampler.sampler_name."

    @classmethod
    def INPUT_TYPES(cls):
        # 🔥 Получаем список ТОЧНО как в KSampler
        try:
            samplers = list(comfy.samplers.KSampler.SAMPLERS)
        except AttributeError:
            # Fallback на SAMPLER_NAMES
            try:
                samplers = list(comfy.samplers.SAMPLER_NAMES)
            except AttributeError:
                samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpmpp_2m", "ddim"]
        
        options = ["none"] + samplers
        return {
            "required": {
                "sampler_1": (options, {"default": "none"}),
                "sampler_2": (options, {"default": "none"}),
                "sampler_3": (options, {"default": "none"}),
                "sampler_4": (options, {"default": "none"}),
                "sampler_5": (options, {"default": "none"}),
                "sampler_6": (options, {"default": "none"}),
                "sampler_7": (options, {"default": "none"}),
                "sampler_8": (options, {"default": "none"}),
                "sampler_9": (options, {"default": "none"}),
                "sampler_10": (options, {"default": "none"}),
                "sampler_11": (options, {"default": "none"}),
                "sampler_12": (options, {"default": "none"}),
                "sampler_13": (options, {"default": "none"}),
                "sampler_14": (options, {"default": "none"}),
                "sampler_15": (options, {"default": "none"}),
                "sampler_16": (options, {"default": "none"}),
                "sampler_17": (options, {"default": "none"}),
                "sampler_18": (options, {"default": "none"}),
                "sampler_19": (options, {"default": "none"}),
                "sampler_20": (options, {"default": "none"}),
            }
        }

    # 🔥 Тип выхода — тот же, что у KSampler
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("samplers_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "get_samplers"
    CATEGORY = "AGSoft/Utility"

    def get_samplers(self, **kwargs):
        return ([v for v in kwargs.values() if v != "none"],)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return str(sorted(kwargs.items()))

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Samplers"


NODE_CLASS_MAPPINGS = {"AGSoft_Loop_Samplers": AGSoft_Loop_Samplers}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Loop_Samplers": "AGSoft Loop Samplers"}