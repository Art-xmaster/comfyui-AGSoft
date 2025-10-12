import torch
import numpy as np
from PIL import Image
import comfy.utils

# Словарь методов интерполяции
INTERPOLATION_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "area": Image.BOX,  # PIL не поддерживает area напрямую, но box подходит
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS
}

class AGSoft_Img_Res_MP:
    """
    Нода для изменения размера изображения с учётом:
    - Мегапикселей (MP)
    - Метода интерполяции
    - Кратности
    - Всегда сохраняет пропорции изображения
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 50.0, "step": 0.1}),
                "interpolation": (list(INTERPOLATION_METHODS.keys()), {"default": "lanczos"}),
                "multiple_of": ([0, 1, 2, 4, 8, 16, 32, 64, 112, 128], {"default": 0})
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "resize_image"
    CATEGORY = "AGSoft/Image"

    def resize_image(self, image, target_megapixels, interpolation, multiple_of):

        # Конвертируем тензор в PIL
        batch_size, height, width, channels = image.shape
        img_np = image.cpu().numpy()

        # Список для выходных изображений
        resized_images = []

        for i in range(batch_size):
            pil_img = Image.fromarray((img_np[i] * 255).astype(np.uint8))

            # Вычисляем целевой размер
            target_pixels = int(target_megapixels * 1_000_000)

            # Пропорциональный ресайз по мегапикселям
            original_w, original_h = pil_img.size
            if original_w <= 0 or original_h <= 0:
                original_w, original_h = 1, 1  # Защита от деления на ноль
            
            ratio = (target_pixels / (original_w * original_h)) ** 0.5
            new_w = max(1, int(original_w * ratio))
            new_h = max(1, int(original_h * ratio))

            # Округление до кратности
            if multiple_of > 0:
                if multiple_of > new_w or multiple_of > new_h:
                    # Если кратность больше размера, используем минимальную кратность
                    new_w = max(multiple_of, new_w)
                    new_h = max(multiple_of, new_h)
                else:
                    new_w = max(multiple_of, (new_w // multiple_of) * multiple_of)
                    new_h = max(multiple_of, (new_h // multiple_of) * multiple_of)

            # Финальная проверка: размеры должны быть >= 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            # Применяем ресайз
            resample = INTERPOLATION_METHODS[interpolation]
            resized_img = pil_img.resize((new_w, new_h), resample=resample)

            # Конвертируем обратно в тензор
            resized_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_tensor = torch.from_numpy(resized_np)[None,]
            resized_images.append(resized_tensor)

        final_image = torch.cat(resized_images, dim=0)

        return (final_image, new_w, new_h)


# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Res_MP": AGSoft_Img_Res_MP
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Res_MP": "AGSoft Image Resize MP"

}
