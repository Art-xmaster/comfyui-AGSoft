import torch
import numpy as np
from PIL import Image, ImageOps
import comfy.utils
import nodes

# Узел для добавления паддинга к изображению с возможностью фитинга и генерации маски
class AGSoft_Img_Pad_for_Outpainting:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Входное изображение
            },
            "optional": {
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_mode": (["constant", "edge", "reflect", "symmetric"], {"default": "constant"}),
                "background_color": (["gray", "black", "white"], {"default": "gray"}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 512, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "pad_image"
    CATEGORY = "AGSoft/Image"

    def pad_image(self, image,
                  pad_left=0, pad_top=0, pad_right=0, pad_bottom=0,
                  pad_mode="constant", background_color="gray",
                  feathering=40, invert_mask=False):

        # Преобразуем тензор изображения в numpy массив
        batch, height, width, channels = image.shape
        image_np = image.cpu().numpy()

        # Выбираем цвет фона
        color_map = {
            "gray": 128,
            "black": 0,
            "white": 255
        }
        fill_value = color_map[background_color] / 255.0  # Нормализуем до [0,1]

        # Применяем паддинг к каждому изображению в батче
        padded_images = []
        masks = []

        for i in range(batch):
            img = image_np[i]  # Получаем одно изображение из батча

            # Паддинг изображения
            if pad_mode == "constant":
                padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    mode='constant', constant_values=fill_value)
            else:
                padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                    mode=pad_mode)

            # Создаем маску: 1 — оригинальное изображение, 0 — паддинг
            mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.float32)
            mask[pad_top:pad_top + height, pad_left:pad_left + width] = 1.0

            # Применяем feathering (размытие краев маски)
            if feathering > 0:
                mask = self.feather_mask(mask, feathering)

            # Инвертируем маску, если нужно
            if invert_mask:
                mask = 1.0 - mask

            # Добавляем результаты
            padded_images.append(padded_img)
            masks.append(mask)

        # Конвертируем обратно в тензоры
        padded_images = torch.from_numpy(np.stack(padded_images)).float()
        masks = torch.from_numpy(np.stack(masks)).float()

        # Возвращаем изображение, маску и размеры
        new_width = width + pad_left + pad_right
        new_height = height + pad_top + pad_bottom

        return (padded_images, masks, new_width, new_height)

    def feather_mask(self, mask, feathering):
        """Применяет размытие к краям маски для плавного перехода"""
        from scipy import ndimage
        mask = ndimage.gaussian_filter(mask, sigma=feathering / 6)  # 6 — эмпирический коэффициент
        return mask


# Регистрация узла в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Pad_for_Outpainting": AGSoft_Img_Pad_for_Outpainting
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Pad_for_Outpainting": "AGSoft Img Pad for Outpainting"
}