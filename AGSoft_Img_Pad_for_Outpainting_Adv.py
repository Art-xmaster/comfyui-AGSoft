import torch
import numpy as np
from PIL import Image, ImageOps
import comfy.utils
import nodes

# Узел для добавления паддинга к изображению с возможностью фитинга и генерации маски
class AGSoft_Img_Pad_for_Outpainting_Adv:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Входное изображение
            },
            "optional": {
                # Ручной паддинг
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "pad_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                
                # Масштабирование до фиксированного размера (опционально)
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = не использовать"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = не использовать"}),
                "keep_proportions": ("BOOLEAN", {"default": True, "label_on": "Keep", "label_off": "Ignore"}),
                "resize_position": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {"default": "center"}),
                
                # Паддинг настройки
                "pad_mode": (["constant", "edge", "reflect", "symmetric"], {"default": "constant"}),
                "background_color": (["gray", "black", "white", "red", "blue", "green", "yellow", "transparent"], {"default": "gray"}),
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
                  target_width=0, target_height=0, keep_proportions=True, resize_position="center",
                  pad_mode="constant", background_color="gray",
                  feathering=40, invert_mask=False):

        # Преобразуем тензор изображения в numpy массив
        batch, orig_height, orig_width, channels = image.shape
        image_np = image.cpu().numpy()

        # Выбираем цвет фона
        color_map = {
            "gray": [128, 128, 128],
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "green": [0, 255, 0],
            "yellow": [255, 255, 0],
            "transparent": [0, 0, 0]  # Для прозрачности используем черный, маска будет управлять прозрачностью
        }
        
        # Нормализуем цвет к диапазону [0,1]
        if background_color != "transparent":
            fill_color = np.array(color_map[background_color]) / 255.0
        else:
            fill_color = np.array([0, 0, 0]) / 255.0  # Черный для прозрачного фона

        # Применяем паддинг к каждому изображению в батче
        padded_images = []
        masks = []

        for i in range(batch):
            img = image_np[i]  # Получаем одно изображение из батча
            current_height, current_width = img.shape[:2]
            
            # Если заданы целевые размеры, сначала масштабируем изображение
            if target_width > 0 and target_height > 0:
                # Рассчитываем размеры с учетом сохранения пропорций
                if keep_proportions:
                    # Рассчитываем коэффициенты масштабирования
                    scale_w = target_width / current_width
                    scale_h = target_height / current_height
                    
                    # Выбираем минимальный коэффициент для сохранения пропорций
                    scale = min(scale_w, scale_h)
                    new_width = int(current_width * scale)
                    new_height = int(current_height * scale)
                else:
                    # Масштабируем без сохранения пропорций
                    new_width = target_width
                    new_height = target_height
                
                # Масштабируем изображение
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img = np.array(img_resized).astype(np.float32) / 255.0
                
                # Рассчитываем паддинг для центрирования
                if resize_position == "center":
                    pad_left_new = (target_width - new_width) // 2
                    pad_right_new = target_width - new_width - pad_left_new
                    pad_top_new = (target_height - new_height) // 2
                    pad_bottom_new = target_height - new_height - pad_top_new
                elif resize_position == "top-left":
                    pad_left_new = 0
                    pad_right_new = target_width - new_width
                    pad_top_new = 0
                    pad_bottom_new = target_height - new_height
                elif resize_position == "top-right":
                    pad_left_new = target_width - new_width
                    pad_right_new = 0
                    pad_top_new = 0
                    pad_bottom_new = target_height - new_height
                elif resize_position == "bottom-left":
                    pad_left_new = 0
                    pad_right_new = target_width - new_width
                    pad_top_new = target_height - new_height
                    pad_bottom_new = 0
                elif resize_position == "bottom-right":
                    pad_left_new = target_width - new_width
                    pad_right_new = 0
                    pad_top_new = target_height - new_height
                    pad_bottom_new = 0
                
                # Добавляем паддинг для масштабированного изображения
                pad_left += pad_left_new
                pad_right += pad_right_new
                pad_top += pad_top_new
                pad_bottom += pad_bottom_new
                
                # Обновляем размеры
                current_width, current_height = new_width, new_height

            # Паддинг изображения - ИСПРАВЛЕННЫЙ КОД
            if pad_mode == "constant":
                # Для constant mode создаем паддинг с выбранным цветом
                if background_color == "transparent" or channels == 4:
                    # Для прозрачного фона или RGBA создаем черный фон
                    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                      mode='constant', constant_values=0)
                else:
                    # Создаем новое изображение с правильным цветом фона
                    new_height_total = current_height + pad_top + pad_bottom
                    new_width_total = current_width + pad_left + pad_right
                    
                    # Создаем изображение с цветом фона
                    padded_img = np.full((new_height_total, new_width_total, channels), 
                                       fill_color, dtype=np.float32)
                    
                    # Копируем оригинальное изображение в центр
                    padded_img[pad_top:pad_top + current_height, pad_left:pad_left + current_width] = img
            else:
                # Для других режимов паддинга используем стандартный подход
                padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                  mode=pad_mode)

            # Создаем маску: 1 — оригинальное изображение, 0 — паддинг
            mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), dtype=np.float32)
            mask[pad_top:pad_top + current_height, pad_left:pad_left + current_width] = 1.0

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

        # Рассчитываем финальные размеры
        final_width = orig_width + pad_left + pad_right
        final_height = orig_height + pad_top + pad_bottom
        
        # Если использовалось масштабирование, используем целевые размеры
        if target_width > 0 and target_height > 0:
            final_width = target_width
            final_height = target_height

        return (padded_images, masks, final_width, final_height)

    def feather_mask(self, mask, feathering):
        """Применяет размытие к краям маски для плавного перехода"""
        from scipy import ndimage
        # Используем гауссово размытие для создания плавного перехода
        if feathering > 0:
            mask = ndimage.gaussian_filter(mask, sigma=feathering / 6)  # 6 — эмпирический коэффициент
        return mask


# Регистрация узла в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Pad_for_Outpainting_Adv": AGSoft_Img_Pad_for_Outpainting_Adv
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Pad_for_Outpainting_Adv": "AGSoft Img Pad for Outpainting Adv"
}