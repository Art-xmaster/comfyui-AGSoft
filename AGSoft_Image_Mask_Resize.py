# AGSoft_Image_Mask_Resize.py
# Нода для изменения размера изображения и маски с сохранением пропорций.
# Одно поле ввода + выбор: масштабировать по ширине или по высоте.

import torch
import numpy as np
from PIL import Image

class AGSoft_Image_Mask_Resize:
    """
    Изменяет размер изображения и маски с сохранением пропорций.
    Пользователь указывает:
      - resize_target: "width" или "height"
      - size: целевое значение (если 0 → использовать scale_by)
    Пропорции всегда сохраняются.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resize_target": (["width", "height"], {"default": "width"}),
                "size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"],),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "device": (["cpu", "cuda"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "resize_image_and_mask"
    CATEGORY = "AGSoft/Mask"

    def resize_image_and_mask(
        self,
        image,
        mask,
        resize_target,
        size,
        scale_by,
        upscale_method,
        invert_mask,
        divisible_by,
        device,
    ):
        try:
            if image is None or mask is None:
                raise ValueError("Изображение или маска не предоставлены.")

            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            image = image.to(target_device)
            mask = mask.to(target_device)

            batch_size, orig_h, orig_w, _ = image.shape

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.repeat(batch_size, 1, 1)

            if invert_mask:
                mask = 1.0 - mask

            # === РАСЧЁТ ЦЕЛЕВЫХ РАЗМЕРОВ ===
            if size > 0:
                if resize_target == "width":
                    target_width = size
                    target_height = int(round(orig_h * (target_width / orig_w)))
                else:  # "height"
                    target_height = size
                    target_width = int(round(orig_w * (target_height / orig_h)))
            else:
                # size = 0 → используем scale_by
                target_width = int(round(orig_w * scale_by))
                target_height = int(round(orig_h * scale_by))

            # Приведение к кратности
            target_width = ((target_width + divisible_by - 1) // divisible_by) * divisible_by
            target_height = ((target_height + divisible_by - 1) // divisible_by) * divisible_by

            # Защита от нулевых размеров
            target_width = max(divisible_by, target_width)
            target_height = max(divisible_by, target_height)

            resized_images = []
            resized_masks = []

            resample_img = self._get_pil_resample(upscale_method)
            resample_mask = Image.NEAREST

            for i in range(batch_size):
                img_tensor = image[i]
                msk_tensor = mask[i]

                img_pil = self._tensor_to_pil(img_tensor, is_mask=False)
                msk_pil = self._tensor_to_pil(msk_tensor, is_mask=True)

                img_resized = img_pil.resize((target_width, target_height), resample=resample_img)
                msk_resized = msk_pil.resize((target_width, target_height), resample=resample_mask)

                resized_images.append(self._pil_to_tensor(img_resized, is_mask=False))
                resized_masks.append(self._pil_to_tensor(msk_resized, is_mask=True))

            final_image = torch.stack(resized_images, dim=0).to(target_device)
            final_mask = torch.stack(resized_masks, dim=0).to(target_device)

            return final_image, final_mask, target_width, target_height

        except Exception as e:
            raise RuntimeError(f"Ошибка при изменении размера: {str(e)}")

    def _tensor_to_pil(self, tensor, is_mask=False):
        if is_mask:
            array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(array, mode='L')
        else:
            array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if array.ndim == 2 or array.shape[-1] == 1:
                array = np.tile(array, (1, 1, 3))
            return Image.fromarray(array, mode='RGB')

    def _pil_to_tensor(self, pil_image, is_mask=False):
        if is_mask:
            if pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            array = np.array(pil_image).astype(np.float32) / 255.0
            return torch.from_numpy(array)
        else:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            array = np.array(pil_image).astype(np.float32) / 255.0
            return torch.from_numpy(array)

    def _get_pil_resample(self, method):
        mapping = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "area": Image.BOX,
            "lanczos": Image.LANCZOS
        }
        return mapping.get(method, Image.BILINEAR)


# --- РЕГИСТРАЦИЯ ---
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Mask_Resize": AGSoft_Image_Mask_Resize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Mask_Resize": "AGSoft Image & Mask Resize"
}