# AGSoft_Image_Mask_Resize_Adv.py
# Расширенная нода для одновременного изменения размера изображения и маски

import torch
import numpy as np
from PIL import Image

class AGSoft_Image_Mask_Resize_Adv:
    """
    Расширенная нода для изменения размера изображения и маски.
    Поддерживает:
      - 3 режима: stretch, crop, pad
      - 9 позиций выравнивания
      - выбор устройства (CPU/GPU)
      - цвет фона изображения и маски при pad (12 цветов)
      - инверсию маски
      - кратность размеров до 128
    """

    @classmethod
    def INPUT_TYPES(cls):
        image_colors = [
            "black", "white", "gray", "red", "green", "blue",
            "orange", "purple", "cyan", "yellow", "brown", "pink"
        ]
        mask_colors = [
            "black", "white", "gray", "red", "green", "blue",
            "orange", "purple", "cyan", "yellow", "brown", "pink"
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "upscale_method": (["nearest", "bilinear", "bicubic", "area", "lanczos"],),
                "resize_mode": (["stretch", "crop", "pad"],),
                "pad_image_color": (image_colors, {"default": "white"}),
                "pad_mask_color": (mask_colors, {"default": "black"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_position": (["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"],),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),  # ← увеличено до 128
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
        width,
        height,
        scale_by,
        upscale_method,
        resize_mode,
        pad_image_color,
        pad_mask_color,
        invert_mask,
        crop_position,
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

            target_width = width if width > 0 else int(orig_w * scale_by)
            target_height = height if height > 0 else int(orig_h * scale_by)

            # Кратность до 128
            target_width = ((target_width + divisible_by - 1) // divisible_by) * divisible_by
            target_height = ((target_height + divisible_by - 1) // divisible_by) * divisible_by

            resized_images = []
            resized_masks = []

            for i in range(batch_size):
                img_tensor = image[i]
                msk_tensor = mask[i]

                img_pil = self._tensor_to_pil(img_tensor, is_mask=False)
                msk_pil = self._tensor_to_pil(msk_tensor, is_mask=True)

                new_img, new_msk = self._apply_resize(
                    img_pil, msk_pil,
                    target_width, target_height,
                    resize_mode, pad_image_color, pad_mask_color,
                    crop_position, upscale_method
                )

                resized_images.append(self._pil_to_tensor(new_img, is_mask=False))
                resized_masks.append(self._pil_to_tensor(new_msk, is_mask=True))

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

    def _apply_resize(
        self, image, mask, width, height, mode,
        pad_image_color, pad_mask_color,
        crop_pos, method
    ):
        if mode == "stretch":
            resample = self._get_pil_resample(method)
            image_resized = image.resize((width, height), resample=resample)
            mask_resized = mask.resize((width, height), resample=Image.NEAREST)

        elif mode == "crop":
            ratio = min(width / image.width, height / image.height)
            new_w = int(image.width * ratio)
            new_h = int(image.height * ratio)
            resample = self._get_pil_resample(method)
            image_scaled = image.resize((new_w, new_h), resample=resample)
            mask_scaled = mask.resize((new_w, new_h), resample=Image.NEAREST)

            left, top = self._get_crop_coords(new_w, new_h, width, height, crop_pos)
            right = min(left + width, new_w)
            bottom = min(top + height, new_h)
            left = max(0, right - width)
            top = max(0, bottom - height)

            image_resized = image_scaled.crop((left, top, right, bottom))
            mask_resized = mask_scaled.crop((left, top, right, bottom))

        elif mode == "pad":
            ratio = min(width / image.width, height / image.height)
            new_w = int(image.width * ratio)
            new_h = int(image.height * ratio)
            resample = self._get_pil_resample(method)
            image_scaled = image.resize((new_w, new_h), resample=resample)
            mask_scaled = mask.resize((new_w, new_h), resample=Image.NEAREST)

            img_color = self._get_rgb_color(pad_image_color)
            msk_color = self._get_grayscale_color(pad_mask_color)

            bg_image = Image.new('RGB', (width, height), img_color)
            bg_mask = Image.new('L', (width, height), msk_color)

            left, top = self._get_pad_coords(new_w, new_h, width, height, crop_pos)
            left = max(0, left)
            top = max(0, top)

            bg_image.paste(image_scaled, (left, top))
            bg_mask.paste(mask_scaled, (left, top))

            image_resized = bg_image
            mask_resized = bg_mask

        return image_resized, mask_resized

    def _get_rgb_color(self, color_name):
        colors = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "cyan": (0, 255, 255),
            "yellow": (255, 255, 0),
            "brown": (165, 42, 42),
            "pink": (255, 192, 203),
        }
        return colors.get(color_name, (255, 255, 255))

    def _get_grayscale_color(self, color_name):
        colors = {
            "black": 0,
            "white": 255,
            "gray": 128,
            "red": 255,
            "green": 255,
            "blue": 255,
            "orange": 255,
            "purple": 255,
            "cyan": 255,
            "yellow": 255,
            "brown": 128,
            "pink": 255,
        }
        return colors.get(color_name, 0)

    def _get_pil_resample(self, method):
        mapping = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "area": Image.BOX,
            "lanczos": Image.LANCZOS
        }
        return mapping.get(method, Image.BILINEAR)

    def _get_crop_coords(self, src_w, src_h, dst_w, dst_h, pos):
        if src_w <= dst_w:
            left = 0
        elif pos in ["left", "top_left", "bottom_left"]:
            left = 0
        elif pos in ["right", "top_right", "bottom_right"]:
            left = src_w - dst_w
        else:
            left = (src_w - dst_w) // 2

        if src_h <= dst_h:
            top = 0
        elif pos in ["top", "top_left", "top_right"]:
            top = 0
        elif pos in ["bottom", "bottom_left", "bottom_right"]:
            top = src_h - dst_h
        else:
            top = (src_h - dst_h) // 2

        return left, top

    def _get_pad_coords(self, src_w, src_h, dst_w, dst_h, pos):
        if pos in ["left", "top_left", "bottom_left"]:
            left = 0
        elif pos in ["right", "top_right", "bottom_right"]:
            left = dst_w - src_w
        else:
            left = (dst_w - src_w) // 2

        if pos in ["top", "top_left", "top_right"]:
            top = 0
        elif pos in ["bottom", "bottom_left", "bottom_right"]:
            top = dst_h - src_h
        else:
            top = (dst_h - src_h) // 2

        return left, top


# --- РЕГИСТРАЦИЯ ---
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Mask_Resize_Adv": AGSoft_Image_Mask_Resize_Adv
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Mask_Resize_Adv": "AGSoft Image & Mask Resize Advanced"
}