"""
AGSoft Image Crop Plus - Расширенная нода для интерактивной обрезки изображений в ComfyUI
Автор: AGSoft
Дата: 03.06.2026
"""
import torch
import numpy as np
from PIL import Image
import json
import os
import folder_paths
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AGSoft_Image_Crop_Plus")

def pil_to_tensor(pil_image):
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).unsqueeze(0)

class AGSoft_Image_Crop_Plus:
    DESCRIPTION = """Interactive image cropping node. Choose between 4-point manual selection, preset aspect ratios, or manual dimensions. The node automatically aligns the output dimensions to a specified multiple (e.g., 8 or 64) for perfect AI model compatibility.
Интерактивная нода для обрезки изображений. Выберите режим: расстановка 4 точек вручную, готовые пропорции (пресеты) или ручной ввод размеров. Нода автоматически выравнивает итоговые размеры под заданную кратность (например, 8 или 64) для идеальной совместимости с нейросетями."""
    CATEGORY = "AGSoft/Image"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            input_dir = folder_paths.get_input_directory()
            image_files = []
            if os.path.exists(input_dir):
                valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
                image_files = sorted([f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions])
        except Exception as e:
            logger.error(f"Error getting input files: {e}")
            image_files = []
        
        return {
            "required": {
                "image_name": ([""] + image_files, {
                    "tooltip": "Select an image from the ComfyUI 'input' folder, or upload a new one via the button in the node interface. Leave empty to bypass cropping and pass the original image. Example: my_photo.png\nВыберите изображение из папки 'input' ComfyUI или загрузите новое через кнопку в интерфейсе ноды. Оставьте пустым, чтобы пропустить обрезку и передать исходное изображение. Пример: my_photo.png"
                }),
                "crop_mode": (["Points (4 clicks)", "Preset Ratio", "Manual Size"], {
                    "default": "Points (4 clicks)",
                    "tooltip": "Cropping method. 'Points (4 clicks)' lets you click 4 corners manually. 'Preset Ratio' creates a resizable box with fixed proportions. 'Manual Size' creates a box with exact pixel dimensions.\nРежим обрезки. 'Points (4 clicks)' позволяет вручную кликнуть 4 угла. 'Preset Ratio' создает изменяемую рамку с фиксированными пропорциями. 'Manual Size' создает рамку с точными размерами в пикселях."
                }),
                "aspect_ratio": (["1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"], {
                    "default": "1:1",
                    "tooltip": "Target aspect ratio for the crop box (used only in 'Preset Ratio' mode). The box will maintain this proportion when resized. Example: '16:9' for widescreen video, '1:1' for social media posts.\nЦелевое соотношение сторон для рамки обрезки (используется только в режиме 'Preset Ratio'). Рамка будет сохранять эту пропорцию при изменении размера. Пример: '16:9' для широкоформатного видео, '1:1' для постов в соцсетях."
                }),
                "manual_width": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Desired width of the crop area in pixels (used only in 'Manual Size' mode). The final output will be adjusted to be divisible by the 'multiple' parameter. Example: 512\nЖелаемая ширина области обрезки в пикселях (используется только в режиме 'Manual Size'). Итоговый размер будет скорректирован так, чтобы делиться на значение параметра 'multiple'. Пример: 512"
                }),
                "manual_height": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Desired height of the crop area in pixels (used only in 'Manual Size' mode). The final output will be adjusted to be divisible by the 'multiple' parameter. Example: 512\nЖелаемая высота области обрезки в пикселях (используется только в режиме 'Manual Size'). Итоговый размер будет скорректирован так, чтобы делиться на значение параметра 'multiple'. Пример: 512"
                }),
                "crop_coords": ("STRING", {
                    "default": "[]", "multiline": False,
                    "tooltip": "AUTOMATICALLY FILLED BY THE UI. Do not edit manually. Stores the coordinates of the 4 points or the crop rectangle (x, y, width, height) as a JSON string.\nЗАПОЛНЯЕТСЯ ИНТЕРФЕЙСОМ АВТОМАТИЧЕСКИ. Не редактируйте вручную. Хранит координаты 4 точек или прямоугольника обрезки (x, y, ширина, высота) в формате JSON."
                }),
                "multiple": ("INT", {
                    "default": 8, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Forces the final output width and height to be divisible by this number. This prevents dimension errors in AI models. Recommended: 8 (general), 32(LTX Video), 64 (SD, FLUX). Example: If crop is 515x300 and multiple=8, result is 512x296.\nЗаставляет итоговую ширину и высоту делиться на это число без остатка. Это предотвращает ошибки размеров в нейросетях. Рекомендуется: 8 (универсально), 32(LTX Video), 64 (SD, FLUX). Пример: если область 515x300, а кратность 8, результат будет 512x296."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "width", "height")
    FUNCTION = "crop_image"

    def crop_image(self, image_name, crop_mode, aspect_ratio, manual_width, manual_height, crop_coords, multiple):
        """
        Crops the image based on the selected mode and aligns dimensions.
        
        Outputs:
        - cropped_image (IMAGE): The resulting cropped and dimension-aligned image tensor, ready for AI generation. / Итоговое обрезанное изображение с выровненными размерами в формате тензора, готовое для генерации.
        - width (INT): The actual width of the output image in pixels (after 'multiple' alignment). / Фактическая ширина выходного изображения в пикселях (после выравнивания по 'multiple').
        - height (INT): The actual height of the output image in pixels (after 'multiple' alignment). / Фактическая высота выходного изображения в пикселях (после выравнивания по 'multiple').
        """
        if not image_name:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), 0, 0)
        
        try:
            image_path = folder_paths.get_annotated_filepath(image_name)
            if not os.path.exists(image_path):
                image_path = os.path.join(folder_paths.get_input_directory(), image_name)
            
            original_image = Image.open(image_path).convert('RGB')
            img_width, img_height = original_image.size
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), 0, 0)

        if not crop_coords or crop_coords == "[]" or crop_coords == "{}":
            return (pil_to_tensor(original_image), img_width, img_height)

        try:
            data = json.loads(crop_coords)
            
            # Поддержка обоих форматов: массив точек или объект с x,y,w,h
            if isinstance(data, list) and len(data) == 4:
                # Старый формат: 4 точки
                x_coords = [p['x'] for p in data]
                y_coords = [p['y'] for p in data]
                min_x = max(0, min(x_coords))
                min_y = max(0, min(y_coords))
                max_x = min(img_width, max(x_coords))
                max_y = min(img_height, max(y_coords))
                w = max_x - min_x
                h = max_y - min_y
            elif isinstance(data, dict) and 'x' in data and 'w' in data:
                # Новый формат: прямоугольник
                min_x = int(data['x'])
                min_y = int(data['y'])
                w = int(data['w'])
                h = int(data['h'])
            else:
                return (pil_to_tensor(original_image), img_width, img_height)

            # Применяем кратность
            w = max(multiple, (w // multiple) * multiple)
            h = max(multiple, (h // multiple) * multiple)
            
            # Ограничиваем границами изображения
            min_x = max(0, min(min_x, img_width - w))
            min_y = max(0, min(min_y, img_height - h))

            cropped_image = original_image.crop((min_x, min_y, min_x + w, min_y + h))
            logger.info(f"Cropped to: {w}x{h} at ({min_x}, {min_y})")
            
            return (pil_to_tensor(cropped_image), w, h)
            
        except Exception as e:
            logger.error(f"Crop error: {e}")
            return (pil_to_tensor(original_image), img_width, img_height)

NODE_CLASS_MAPPINGS = {"AGSoft Image Crop Plus": AGSoft_Image_Crop_Plus}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft Image Crop Plus": "✂️ AGSoft Image Crop Plus"}