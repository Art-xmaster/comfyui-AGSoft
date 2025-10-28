# AGSoft_Loop_Images.py
# Автор: AGSoft
# Дата: 28 октября 2025 г.

class AGSoft_Loop_Images:
    DESCRIPTION = "Dynamically collects multiple image inputs into a single list for batch processing (e.g., Save Image Batch, ForEach, etc.).\nДинамически собирает несколько входов изображений в один список для пакетной обработки (например, Save Image Batch, ForEach и т.д.)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_of_images": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "How many image inputs to create. Each input will be named image_1, image_2, etc.\nExample: number_of_images=3 → creates image_1, image_2, image_3\nСколько входов для изображений создать. Каждый вход будет называться image_1, image_2 и т.д.\nПример: number_of_images=3 → создаются image_1, image_2, image_3"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "loop_images"
    CATEGORY = "AGSoft/Utility"

    def loop_images(self, number_of_images, **kwargs):
        image_list = []
        for i in range(1, number_of_images + 1):
            image_key = f"image_{i}"
            if image_key in kwargs and kwargs[image_key] is not None:
                image_list.append(kwargs[image_key])
        return (image_list,)

    @classmethod
    def IS_CHANGED(cls, number_of_images, **kwargs):
        return float("nan")  # Always re-execute to reflect image changes

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Images"


NODE_CLASS_MAPPINGS = {"AGSoft_Loop_Images": AGSoft_Loop_Images}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Loop_Images": "AGSoft Loop Images"}