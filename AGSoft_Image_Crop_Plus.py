"""
AGSoft Image Crop Plus - нода для обрезки изображений в ComfyUI
Автор: AGSoft
Версия: 2.0.0
Дата: 2026-06-01
"""
import torch
import numpy as np
from PIL import Image
import json
import os
import folder_paths
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AGSoft_Image_Crop_Plus")

# --- Вспомогательные функции ---
def pil_to_tensor(pil_image):
    """
    Преобразует PIL Image в тензор PyTorch (1, H, W, C) с диапазоном [0, 1]
    """
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(image_np).unsqueeze(0)
    return tensor_image

def tensor_to_pil(tensor_image):
    """
    Преобразует тензор PyTorch (1, H, W, C) в PIL Image
    """
    if tensor_image.dim() == 4:
        tensor_image = tensor_image[0]
    image_np = (tensor_image.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np, mode='RGB')

# --- Основной класс ноды ---
class AGSoft_Image_Crop_Plus:
    DESCRIPTION = """Interactive image cropping node. Select an image, click 4 points on the canvas to define a crop rectangle, and queue. The node automatically adjusts dimensions to a specified multiple for AI compatibility.
Интерактивная нода для обрезки изображений. Выберите изображение, кликните 4 точки на холсте для выделения прямоугольника обрезки и запустите генерацию. Нода автоматически подгоняет размеры под заданную кратность для совместимости с нейросетями."""
    
    CATEGORY = "AGSoft/Image"

    @classmethod
    def INPUT_TYPES(cls):
        # Получаем список файлов из папки input
        try:
            input_dir = folder_paths.get_input_directory()
            if os.path.exists(input_dir):
                image_files = []
                valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
                for file in os.listdir(input_dir):
                    if os.path.splitext(file)[1].lower() in valid_extensions:
                        image_files.append(file)
                image_files.sort()
            else:
                logger.warning(f"Папка input не найдена: {input_dir}")
                image_files = []
        except Exception as e:
            logger.error(f"Ошибка при получении списка файлов: {e}")
            image_files = []
        
        return {
            "required": {
                "image_name": ([""] + image_files, {
                    "tooltip": "Select an image from the ComfyUI input folder, or upload one via the button below. Leave empty to bypass cropping. / Выберите изображение из папки input или загрузите новое кнопкой. Оставьте пустым, чтобы пропустить обрезку.\nПример: my_photo.png"
                }),
                "crop_coords": ("STRING", {
                    "default": "[]",
                    "tooltip": "Automatically filled by the UI when you click 4 points on the preview. Do not edit manually. Format: JSON array of coordinates. / Автоматически заполняется интерфейсом при клике 4 точек. Не редактируйте вручную. Формат: JSON массив координат.\nПример: [{\"x\":100, \"y\":100}, {\"x\":200, \"y\":100}, {\"x\":200, \"y\":200}, {\"x\":100, \"y\":200}]"
                }),
                "multiple": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Forces output width/height to be divisible by this number for AI model compatibility. Recommended: 8 (general), 64 (SD/FLUX). Example: If crop is 513x300, multiple=8 gives 512x296. / Заставляет ширину и высоту обрезки делиться на это число для совместимости с нейросетями. Рекомендуется: 8 (общее), 64 (SD/FLUX). Пример: при области 513x300 и значении 8 результат будет 512x296."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "width", "height")

    FUNCTION = "crop_image"

    def crop_image(self, image_name, crop_coords, multiple):
        """
        Основная функция обрезки изображения
        """
        logger.info(f"Начало обработки: image_name={image_name}, multiple={multiple}")
        
        # Проверяем, выбрано ли изображение
        if not image_name:
            logger.warning("Изображение не выбрано, возвращаем пустое изображение")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0, 0)
        
        # Получаем полный путь к изображению
        try:
            image_path = folder_paths.get_annotated_filepath(image_name)
            if not os.path.exists(image_path):
                # Пробуем найти в папке input
                input_dir = folder_paths.get_input_directory()
                image_path = os.path.join(input_dir, image_name)
                if not os.path.exists(image_path):
                    logger.error(f"Файл не найден: {image_path}")
                    empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                    return (empty_image, 0, 0)
        except Exception as e:
            logger.error(f"Ошибка при получении пути к файлу: {e}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0, 0)
        
        # Загружаем изображение
        try:
            original_image = Image.open(image_path).convert('RGB')
            img_width, img_height = original_image.size
            logger.info(f"Изображение загружено: {img_width}x{img_height}")
        except Exception as e:
            logger.error(f"Не удалось загрузить изображение: {e}")
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (empty_image, 0, 0)
        
        # Проверяем наличие координат
        if not crop_coords or crop_coords == "[]":
            logger.info("Координаты не заданы, возвращаем исходное изображение")
            return (pil_to_tensor(original_image), img_width, img_height)
        
        # Парсим координаты
        try:
            points = json.loads(crop_coords)
            if not isinstance(points, list) or len(points) != 4:
                logger.warning(f"Некорректный формат точек: {points}")
                return (pil_to_tensor(original_image), img_width, img_height)
            
            # Проверяем каждую точку
            for p in points:
                if not isinstance(p, dict) or 'x' not in p or 'y' not in p:
                    logger.warning(f"Некорректная точка: {p}")
                    return (pil_to_tensor(original_image), img_width, img_height)
            
            # Получаем границы прямоугольника
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            
            min_x = max(0, min(x_coords))
            min_y = max(0, min(y_coords))
            max_x = min(img_width, max(x_coords))
            max_y = min(img_height, max(y_coords))
            
            # Проверяем, что прямоугольник имеет положительный размер
            if min_x >= max_x or min_y >= max_y:
                logger.warning(f"Прямоугольник имеет нулевой размер: ({min_x},{min_y})-({max_x},{max_y})")
                return (pil_to_tensor(original_image), img_width, img_height)
            
            # Вычисляем размеры области
            crop_width = max_x - min_x
            crop_height = max_y - min_y
            
            # Применяем кратность
            new_width = (crop_width // multiple) * multiple
            new_height = (crop_height // multiple) * multiple
            
            if new_width <= 0 or new_height <= 0:
                logger.warning(f"После применения кратности {multiple} размер стал нулевым")
                return (pil_to_tensor(original_image), img_width, img_height)
            
            # Корректируем координаты под новые размеры
            max_x = min_x + new_width
            max_y = min_y + new_height
            
            # Обрезаем изображение
            cropped_image = original_image.crop((min_x, min_y, max_x, max_y))
            logger.info(f"Обрезка выполнена: {new_width}x{new_height}")
            
            return (pil_to_tensor(cropped_image), new_width, new_height)
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}, crop_coords={crop_coords}")
            return (pil_to_tensor(original_image), img_width, img_height)
        except Exception as e:
            logger.error(f"Неожиданная ошибка при обрезке: {e}")
            return (pil_to_tensor(original_image), img_width, img_height)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "AGSoft Image Crop Plus": AGSoft_Image_Crop_Plus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft Image Crop Plus": "✂️AGSoft Image Crop Plus",
}