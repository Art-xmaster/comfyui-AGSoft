import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

class AGSoft_Img_Res:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["fixed_size", "percentage"],),
                "target_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "target_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "percentage": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "resize_if_larger": (["none", "width", "height", "both"],),
                "resize_if_smaller": (["none", "width", "height", "both"],),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
                "interpolation": (["lanczos", "bilinear", "bicubic", "nearest"],),
                "crop_if_needed": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "AGSoft/Image"

    def get_interpolation_method(self, interpolation):
        """Возвращает метод интерполяции PIL"""
        methods = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST
        }
        return methods.get(interpolation, Image.LANCZOS)

    def resize_image(self, image, resize_mode, target_width, target_height, percentage, 
                     resize_if_larger, resize_if_smaller, keep_aspect_ratio, 
                     interpolation, crop_if_needed):
        # Преобразуем из тензора в PIL Image
        batch = image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
        resized_batch = []

        interp_method = self.get_interpolation_method(interpolation)

        for img_tensor in batch:
            # Преобразуем тензор в PIL
            pil_image = TF.to_pil_image(img_tensor)

            orig_width, orig_height = pil_image.size

            # Определяем целевые размеры
            if resize_mode == "percentage":
                target_width = int(orig_width * (percentage / 100.0))
                target_height = int(orig_height * (percentage / 100.0))
                # Убеждаемся, что размеры не нулевые
                target_width = max(1, target_width)
                target_height = max(1, target_height)

            new_width = orig_width
            new_height = orig_height

            # Условия изменения размера для больших изображений
            should_resize_larger = False
            if resize_if_larger != "none":
                if resize_if_larger == "width" and orig_width > target_width:
                    should_resize_larger = True
                elif resize_if_larger == "height" and orig_height > target_height:
                    should_resize_larger = True
                elif resize_if_larger == "both" and (orig_width > target_width or orig_height > target_height):
                    should_resize_larger = True

            if should_resize_larger:
                if keep_aspect_ratio:
                    if resize_if_larger == "width":
                        ratio = target_width / orig_width
                        new_width = target_width
                        new_height = int(orig_height * ratio)
                    elif resize_if_larger == "height":
                        ratio = target_height / orig_height
                        new_height = target_height
                        new_width = int(orig_width * ratio)
                    elif resize_if_larger == "both":
                        # Выбираем минимальное соотношение для сохранения пропорций
                        ratio_w = target_width / orig_width
                        ratio_h = target_height / orig_height
                        ratio = min(ratio_w, ratio_h)
                        new_width = int(orig_width * ratio)
                        new_height = int(orig_height * ratio)
                else:
                    new_width = target_width
                    new_height = target_height

            # Условия изменения размера для меньших изображений
            should_resize_smaller = False
            if resize_if_smaller != "none":
                if resize_if_smaller == "width" and orig_width < target_width:
                    should_resize_smaller = True
                elif resize_if_smaller == "height" and orig_height < target_height:
                    should_resize_smaller = True
                elif resize_if_smaller == "both" and (orig_width < target_width or orig_height < target_height):
                    should_resize_smaller = True

            if should_resize_smaller:
                if keep_aspect_ratio:
                    if resize_if_smaller == "width":
                        ratio = target_width / orig_width
                        new_width = target_width
                        new_height = int(orig_height * ratio)
                    elif resize_if_smaller == "height":
                        ratio = target_height / orig_height
                        new_height = target_height
                        new_width = int(orig_width * ratio)
                    elif resize_if_smaller == "both":
                        # Выбираем максимальное соотношение для сохранения пропорций
                        ratio_w = target_width / orig_width
                        ratio_h = target_height / orig_height
                        ratio = max(ratio_w, ratio_h)
                        new_width = int(orig_width * ratio)
                        new_height = int(orig_height * ratio)
                else:
                    new_width = target_width
                    new_height = target_height

            # Если размеры изменились — ресайзим
            if new_width != orig_width or new_height != orig_height:
                pil_image = pil_image.resize((new_width, new_height), interp_method)

            # Обрезка до целевых размеров, если требуется
            if crop_if_needed and (new_width != target_width or new_height != target_height):
                # Центрируем обрезку
                left = max(0, (new_width - target_width) // 2)
                top = max(0, (new_height - target_height) // 2)
                right = min(new_width, left + target_width)
                bottom = min(new_height, top + target_height)
                
                # Корректируем размеры, если нужно
                if right - left < target_width:
                    left = max(0, right - target_width)
                if bottom - top < target_height:
                    top = max(0, bottom - target_height)
                
                pil_image = pil_image.crop((left, top, left + target_width, top + target_height))

            # Если нужна обрезка и режим фиксированного размера, обрезаем до целевого размера
            elif crop_if_needed and resize_mode == "fixed_size":
                if pil_image.width > target_width or pil_image.height > target_height:
                    left = max(0, (pil_image.width - target_width) // 2)
                    top = max(0, (pil_image.height - target_height) // 2)
                    pil_image = pil_image.crop((left, top, left + target_width, top + target_height))

            # Преобразуем обратно в тензор
            resized_tensor = TF.to_tensor(pil_image)
            resized_batch.append(resized_tensor)

        # Собираем батч обратно
        resized_batch = torch.stack(resized_batch, dim=0)
        resized_batch = resized_batch.movedim(1, -1)  # [B, C, H, W] -> [B, H, W, C]

        return (resized_batch,)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Res": AGSoft_Img_Res
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Res": "AGSoft Image Resize"
}