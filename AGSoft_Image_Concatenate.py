import torch
import numpy as np
from PIL import Image

class AGSoft_Image_Concatenate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "direction": (["horizontal", "vertical"], {"default": "horizontal"}),
                "resize_method": (["LANCZOS", "BILINEAR", "BICUBIC", "NEAREST"], {"default": "LANCZOS"}),
                "gap": ("INT", {"default": 0, "min": 0, "max": 100}),
                "gap_color": (["white", "black", "gray", "red", "green", "blue", "yellow"], {"default": "white"}),
                "scale_to_size": ("INT", {"default": 0, "min": 0, "max": 4096}),  # 0 = не изменять
                "multiple_of": ([0, 2, 8, 16, 32, 64], {"default": 0}),
                "keep_aspect_ratio": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "concatenate_images"
    CATEGORY = "AGSoft/Image"

    def get_gap_color_rgb(self, color_name):
        """Возвращает RGB значение для выбранного цвета gap"""
        colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0)
        }
        return colors.get(color_name, (255, 255, 255))

    def resize_image(self, img, target_size, axis, method, keep_aspect_ratio):
        """
        Изменяет размер изображения по заданной оси с возможностью сохранения пропорций.
        """
        w, h = img.size
        if axis == "horizontal":
            if keep_aspect_ratio:
                ratio = target_size / h if h > 0 else 1
                new_w = int(w * ratio)
                new_h = target_size
            else:
                new_h = target_size
                new_w = w
        else:  # vertical
            if keep_aspect_ratio:
                ratio = target_size / w if w > 0 else 1
                new_w = target_size
                new_h = int(h * ratio)
            else:
                new_w = target_size
                new_h = h

        # Применяем кратность
        if self.multiple_of > 0:
            new_w = round(new_w / self.multiple_of) * self.multiple_of
            new_h = round(new_h / self.multiple_of) * self.multiple_of

        resample = getattr(Image, method)
        return img.resize((new_w, new_h), resample)

    def tensor_to_pil(self, image_tensor):
        """
        Конвертирует тензор ComfyUI в PIL Image.
        """
        image_np = image_tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np[0])

    def pil_to_tensor(self, pil_image):
        """
        Конвертирует PIL Image в тензор ComfyUI.
        """
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]

    def concatenate_images(self, image1, image2, direction, resize_method, gap, gap_color, scale_to_size, multiple_of, keep_aspect_ratio):
        self.multiple_of = multiple_of
        gap_color_rgb = self.get_gap_color_rgb(gap_color)

        # Конвертируем тензоры в PIL
        img1 = self.tensor_to_pil(image1)
        img2 = self.tensor_to_pil(image2)

        # Приведение к одному размеру
        if direction == "horizontal":
            # Приводим по высоте
            target_size = scale_to_size if scale_to_size > 0 else max(img1.height, img2.height)
            img1 = self.resize_image(img1, target_size, "horizontal", resize_method, keep_aspect_ratio)
            img2 = self.resize_image(img2, target_size, "horizontal", resize_method, keep_aspect_ratio)
            canvas_width = img1.width + img2.width + gap
            canvas_height = max(img1.height, img2.height)
        else:
            # Приводим по ширине
            target_size = scale_to_size if scale_to_size > 0 else max(img1.width, img2.width)
            img1 = self.resize_image(img1, target_size, "vertical", resize_method, keep_aspect_ratio)
            img2 = self.resize_image(img2, target_size, "vertical", resize_method, keep_aspect_ratio)
            canvas_width = max(img1.width, img2.width)
            canvas_height = img1.height + img2.height + gap

        # Создаем пустое изображение с цветом gap
        canvas = Image.new("RGB", (canvas_width, canvas_height), gap_color_rgb)

        # Размещаем изображения по центру (по умолчанию)
        if direction == "horizontal":
            # Центрируем по вертикали
            y1 = (canvas_height - img1.height) // 2
            y2 = (canvas_height - img2.height) // 2
            canvas.paste(img1, (0, y1))
            canvas.paste(img2, (img1.width + gap, y2))
        else:
            # Центрируем по горизонтали
            x1 = (canvas_width - img1.width) // 2
            x2 = (canvas_width - img2.width) // 2
            canvas.paste(img1, (x1, 0))
            canvas.paste(img2, (x2, img1.height + gap))

        # Конвертируем обратно в тензор
        result_tensor = self.pil_to_tensor(canvas)
        width = canvas.width
        height = canvas.height

        return (result_tensor, width, height)


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Concatenate": AGSoft_Image_Concatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Concatenate": "AGSoft Image Concatenate"
}