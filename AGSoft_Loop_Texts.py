# AGSoft_Loop_Texts.py
# Автор: AGSoft
# Дата: 28 октября 2025 г.

class AGSoft_Loop_Texts:
    DESCRIPTION = "Dynamically collects multiple text inputs into a single list for batch processing (e.g., Prompt Switch, ForEach, etc.).\nДинамически собирает несколько текстовых входов в один список для пакетной обработки (например, Prompt Switch, ForEach и т.д.)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_of_inputs": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "How many text inputs to create. Each input will be named text_1, text_2, etc.\nExample: number_of_inputs=3 → creates text_1, text_2, text_3\nСколько текстовых входов создать. Каждый вход будет называться text_1, text_2 и т.д.\nПример: number_of_inputs=3 → создаются text_1, text_2, text_3"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "loop_texts"
    CATEGORY = "AGSoft/Utility"

    def loop_texts(self, number_of_inputs, **kwargs):
        text_list = []
        for i in range(1, number_of_inputs + 1):
            key = f"text_{i}"
            if key in kwargs and kwargs[key] is not None:
                text_list.append(kwargs[key])
        return (text_list,)

    @classmethod
    def IS_CHANGED(cls, number_of_inputs, **kwargs):
        return float("nan")  # Always re-execute

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Texts"


NODE_CLASS_MAPPINGS = {"AGSoft_Loop_Texts": AGSoft_Loop_Texts}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Loop_Texts": "AGSoft Loop Texts"}