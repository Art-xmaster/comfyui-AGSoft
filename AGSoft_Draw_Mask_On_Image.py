import torch
import numpy as np
from PIL import Image, ImageColor

class AGSoft_Draw_Mask_On_Image:
    """
    Рисует (закрашивает) область маски на изображении выбранным цветом по названию (например: red, blue, white).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color_name": ([
                    "white", "black", "red", "green", "blue",
                    "yellow", "cyan", "magenta", "gray", "lightgray",
                    "darkgray", "orange", "purple", "pink", "brown",
                    "lime", "navy", "teal", "olive", "maroon"
                ], {"default": "white"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw_mask"
    CATEGORY = "AGSoft/Mask"

    def draw_mask(self, image, mask, color_name, invert_mask, device):
        try:
            # Определяем устройство
            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

            image = image.to(target_device)
            mask = mask.to(target_device)

            batch_size, h, w, c = image.shape

            # Подготавливаем маску
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.repeat(batch_size, 1, 1)

            if invert_mask:
                mask = 1.0 - mask

            # Получаем RGB из названия цвета
            try:
                r, g, b = ImageColor.getrgb(color_name)
            except:
                r, g, b = 255, 255, 255  # fallback: white

            painted_images = []

            for i in range(batch_size):
                img_np = image[i].cpu().numpy()
                msk_np = mask[i].cpu().numpy()

                img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
                msk_uint8 = (msk_np * 255).clip(0, 255).astype(np.uint8)

                color_layer = np.full_like(img_uint8, [r, g, b], dtype=np.uint8)
                painted = np.where(msk_uint8[..., None] > 127, color_layer, img_uint8)

                painted_tensor = torch.from_numpy(painted.astype(np.float32) / 255.0).to(target_device)
                painted_images.append(painted_tensor)

            painted_batch = torch.stack(painted_images, dim=0)

            return (painted_batch,)

        except Exception as e:
            raise RuntimeError(f"Ошибка в AGSoft Draw Mask On Image: {str(e)}")


# --- РЕГИСТРАЦИЯ ---
NODE_CLASS_MAPPINGS = {
    "AGSoft_Draw_Mask_On_Image": AGSoft_Draw_Mask_On_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Draw_Mask_On_Image": "AGSoft Draw Mask On Image"
}