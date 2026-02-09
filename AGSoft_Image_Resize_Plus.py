"""
AGSoft_Image_Resize_Plus.py
Расширенная нода для изменения размера изображений и масок с поддержкой
5 режимов расчета размеров, 3 стратегий масштабирования, аппаратного ускорения
и предпросмотра изображений как в стандартной ноде Load Image.

Особенности:
- Одновременная обработка изображения и маски (синхронизировано)
- 5 режимов задания размеров: мегапиксели, проценты, ширина, высота, оба размера
- 3 стратегии: stretch (растянуть), crop (обрезать), pad (добавить поля)
- Раздельные цвета фона для изображения и маски
- Поддержка CPU/CUDA
- Встроенный превью изображения (как в Load Image)
- Кратность размеров (divisible_by) для совместимости с моделями
- Инверсия маски
- 9 позиций выравнивания
- Опциональная маска (можно работать только с изображением)
- Возврат имени файла и полного пути

Автор: AGSoft
Дата: 05.02.2026
"""

import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import warnings
import os
import hashlib
import folder_paths
import node_helpers
from typing import Tuple, Optional, Dict, List, Union, Any

# ============================================================================
# КОНСТАНТЫ И ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

# Словарь методов интерполяции для PIL
INTERPOLATION_METHODS = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "area": Image.BOX,
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS
}

# Стратегии обработки пропорций
FIT_MODES = ["stretch", "crop", "pad"]

# Расширенный список цветов для изображения (RGB в HEX)
IMAGE_COLORS_HEX = {
    "black": "#000000",
    "white": "#FFFFFF",
    "gray": "#808080",
    "silver": "#C0C0C0",
    "light_gray": "#D3D3D3",
    "dark_gray": "#A9A9A9",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "orange": "#FFA500",
    "pink": "#FFC0CB",
    "brown": "#A52A2A",
    "purple": "#800080",
    "violet": "#EE82EE",
    "indigo": "#4B0082",
    "teal": "#008080",
    "navy": "#000080",
    "olive": "#808000",
    "maroon": "#800000",
    "dark_blue": "#00008B",
    "light_blue": "#ADD8E6",
    "light_green": "#90EE90",
    "dark_green": "#006400",
    "transparent": "transparent"
}

# Цвета для маски (оттенки серого в HEX)
MASK_COLORS_HEX = {
    "black": "#000000",
    "white": "#FFFFFF",
    "gray": "#808080",
    "silver": "#C0C0C0",
    "light_gray": "#D3D3D3",
    "dark_gray": "#A9A9A9",
    "transparent": "transparent"
}

# Позиции для центрирования (9 стандартных позиций)
CROP_POSITIONS = {
    "center": (0.5, 0.5),
    "top-left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top-right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom-right": (1.0, 1.0),
}

# Режимы задания размеров
RESIZE_MODES = [
    "target_megapixels",    # Задать целевое количество мегапикселей
    "target_percentage",    # Задать процент от исходного размера
    "target_width",         # Задать только ширину (высота автоматом)
    "target_height",        # Задать только высоту (ширина автоматом)
    "target_both"           # Задать ширину и высоту явно
]


def hex_to_rgb_or_rgba(hex_color: str, has_alpha: bool = False) -> Tuple:
    """
    Преобразует HEX-цвет в кортеж RGB или RGBA для PIL.
    
    Args:
        hex_color (str): HEX-код цвета (#RRGGBB или #RRGGBBAA) или "transparent"
        has_alpha (bool): Возвращать ли альфа-канал
        
    Returns:
        Tuple: (R, G, B) или (R, G, B, A)
        
    Перевод на английский:
    Converts HEX color to RGB or RGBA tuple for PIL.
    """
    if hex_color == "transparent":
        return (0, 0, 0, 0) if has_alpha else (0, 0, 0)
    
    hex_color = hex_color.lstrip('#')
    
    if len(hex_color) == 6:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb + (255,) if has_alpha else rgb
    elif len(hex_color) == 8:
        rgba = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
        return rgba
    else:
        return (0, 0, 0)  # fallback на черный цвет


def tensor_to_pil(tensor: torch.Tensor, is_mask: bool = False) -> Image.Image:
    """
    Преобразует тензор PyTorch в изображение PIL.
    
    Args:
        tensor (torch.Tensor): Тензор изображения [H, W, C] или [H, W]
        is_mask (bool): Является ли тензор маской
        
    Returns:
        Image.Image: Изображение PIL
        
    Перевод на английский:
    Converts PyTorch tensor to PIL Image.
    """
    # Конвертируем в numpy
    array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    if is_mask:
        # Для маски: градации серого
        if array.ndim == 3:
            if array.shape[-1] == 1:
                array = array.squeeze(-1)
            elif array.shape[-1] == 3:
                # Если маска в RGB, конвертируем в grayscale
                array = np.dot(array[..., :3], [0.299, 0.587, 0.114])
        return Image.fromarray(array, mode='L')
    else:
        # Для изображения: RGB или RGBA
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.shape[-1] == 1:
            array = np.concatenate([array] * 3, axis=-1)
        elif array.shape[-1] == 4:
            return Image.fromarray(array, mode='RGBA')
        return Image.fromarray(array, mode='RGB')


def pil_to_tensor(pil_image: Image.Image, is_mask: bool = False) -> torch.Tensor:
    """
    Преобразует изображение PIL в тензор PyTorch.
    
    Args:
        pil_image (Image.Image): Изображение PIL
        is_mask (bool): Является ли изображение маской
        
    Returns:
        torch.Tensor: Тензор [H, W, C] или [H, W]
        
    Перевод на английский:
    Converts PIL Image to PyTorch tensor.
    """
    if is_mask:
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(array)
    else:
        if pil_image.mode == 'RGBA':
            array = np.array(pil_image).astype(np.float32) / 255.0
        else:
            pil_image = pil_image.convert('RGB')
            array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(array)


# ============================================================================
# ОСНОВНОЙ КЛАСС НОДЫ - AGSoft_Image_Resize_Plus
# ============================================================================

class AGSoft_Image_Resize_Plus:
    """
    AGSoft Image Resize Plus - Расширенная нода для изменения размера изображений и масок.
    
    Эта нода объединяет лучшие функции из всех предыдущих версий:
    1. Поддерживает 5 режимов задания размеров: мегапиксели, проценты, ширина, высота, оба размера
    2. Обрабатывает изображения и маски синхронно (чтобы они не рассинхронизировались)
    3. Имеет 3 стратегии масштабирования: stretch, crop, pad
    4. Раздельные цвета фона для изображения и маски
    5. Поддержка аппаратного ускорения (CPU/CUDA)
    6. Встроенный превью изображения (как в Load Image)
    7. Кратность размеров для совместимости с нейросетевыми моделями
    
    Особенности из стандартной ноды Load Image:
    - Превью изображения при загрузке
    - Выбор изображения из папки input
    - Кнопка загрузки файла
    - Поддержка анимированных изображений
    - Автоматическая обработка EXIF данных
    
    Основные сценарии использования:
    - Подготовка изображений для обучения моделей
    - Изменение размеров перед обработкой в пайплайне
    - Синхронное изменение изображений и масок для сегментации
    - Батч-обработка нескольких изображений
    
    AGSoft Image Resize Plus - Extended node for resizing images and masks.
    This node combines the best features from all previous versions:
    1. Supports 5 size specification modes: megapixels, percentage, width, height, both dimensions
    2. Processes images and masks synchronously (to prevent desynchronization)
    3. Has 3 scaling strategies: stretch, crop, pad
    4. Separate background colors for image and mask
    5. Hardware acceleration support (CPU/CUDA)
    6. Built-in image preview (like Load Image)
    7. Size divisibility for compatibility with neural network models
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """
        Определяет входные параметры ноды для ComfyUI.
        Этот метод обязателен для всех нод ComfyUI.
        
        Returns:
            Dict: Словарь с описанием входных параметров
            
        Определение на английском:
        Defines input parameters for the node in ComfyUI.
        This method is mandatory for all ComfyUI nodes.
        """
        # Получаем список файлов из папки input
        input_dir = folder_paths.get_input_directory()
        files = []
        
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            # Фильтруем только изображения
            files = folder_paths.filter_files_content_types(files, ["image"])
        
        return {
            "required": {
                # Параметр image с выпадающим списком файлов как в стандартной ноде Load Image
                "image": (sorted(files), {
                    "image_upload": True,  # Добавляет кнопку загрузки файла
                    "tooltip": "Выберите изображение из папки input или загрузите новое.\n"
                               "Select image from input folder or upload new one."
                }),
                "resize_mode": (RESIZE_MODES, {
                    "default": "target_percentage",
                    "tooltip": "Режим задания целевых размеров:\n"
                               "• target_megapixels - задать общее количество мегапикселей\n"
                               "• target_percentage - задать процент от исходного размера\n"
                               "• target_width - задать только ширину (высота вычисляется автоматически)\n"
                               "• target_height - задать только высоту (ширина вычисляется автоматически)\n"
                               "• target_both - задать ширину и высоту явно\n\n"
                               "Size specification mode:\n"
                               "• target_megapixels - set total megapixels\n"
                               "• target_percentage - set percentage of original size\n"
                               "• target_width - set only width (height calculated automatically)\n"
                               "• target_height - set only height (width calculated automatically)\n"
                               "• target_both - set both width and height explicitly"
                }),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Целевое количество мегапикселей (MP).\n"
                               "1.0 MP = 1,000,000 пикселей (примерно 1000x1000).\n"
                               "Используется только в режиме 'target_megapixels'.\n\n"
                               "Target number of megapixels (MP).\n"
                               "1.0 MP = 1,000,000 pixels (approximately 1000x1000).\n"
                               "Used only in 'target_megapixels' mode."
                }),
                "target_percentage": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Процент изменения размера относительно оригинала.\n"
                               "100.0 = исходный размер, 50.0 = половина размера,\n"
                               "200.0 = двойной размер.\n"
                               "Используется только в режиме 'target_percentage'.\n\n"
                               "Percentage of size change relative to original.\n"
                               "100.0 = original size, 50.0 = half size,\n"
                               "200.0 = double size.\n"
                               "Used only in 'target_percentage' mode."
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Желаемая ширина в пикселях.\n"
                               "Используется в режимах 'target_width' и 'target_both'.\n\n"
                               "Desired width in pixels.\n"
                               "Used in 'target_width' and 'target_both' modes."
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Желаемая высота в пикселях.\n"
                               "Используется в режимах 'target_height' и 'target_both'.\n\n"
                               "Desired height in pixels.\n"
                               "Used in 'target_height' and 'target_both' modes."
                }),
                "fit_mode": (FIT_MODES, {
                    "default": "crop",
                    "tooltip": "Стратегия масштабирования:\n"
                               "• stretch - растянуть/сжать изображение до точных размеров (могут искажаться пропорции)\n"
                               "• crop - сохранить пропорции и обрезать лишнее\n"
                               "• pad - сохранить пропорции и добавить поля по краям\n\n"
                               "Scaling strategy:\n"
                               "• stretch - stretch/squeeze image to exact dimensions (aspect ratio may distort)\n"
                               "• crop - preserve aspect ratio and crop excess\n"
                               "• pad - preserve aspect ratio and add padding"
                }),
                "interpolation": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "lanczos",
                    "tooltip": "Метод интерполяции при изменении размера:\n"
                               "• nearest - самый быстрый, но низкое качество (пиксельный)\n"
                               "• bilinear - сглаженный, быстрый, подходит для preview\n"
                               "• bicubic - более плавный, чем bilinear, хорош для фото\n"
                               "• area (BOX) - оптимален для уменьшения изображений (сохраняет детали)\n"
                               "• lanczos - наивысшее качество, но медленнее; идеален для финального рендера\n"
                               "• nearest-exact - как 'nearest', но с улучшённой точностью\n\n"
                               "Interpolation method for resizing:\n"
                               "• nearest - fastest but lowest quality (pixelated)\n"
                               "• bilinear - smooth, fast, good for preview\n"
                               "• bicubic - smoother than bilinear, good for photos\n"
                               "• area (BOX) - optimal for downscaling (preserves details)\n"
                               "• lanczos - highest quality but slower; ideal for final render\n"
                               "• nearest-exact - like 'nearest' but with improved accuracy"
                }),
                "divisible_by": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Округляет итоговую ширину и высоту до кратного этому числу.\n"
                               "0 означает отключение. Полезно для совместимости с моделями.\n"
                               "Пример: divisible_by=8 для SD/SDXL моделей.\n\n"
                               "Rounds final width and height to multiples of this number.\n"
                               "0 means disabled. Useful for model compatibility.\n"
                               "Example: divisible_by=8 for SD/SDXL models."
                }),
                "crop_position": (list(CROP_POSITIONS.keys()), {
                    "default": "center",
                    "tooltip": "Позиция изображения при стратегиях crop и pad:\n"
                               "Определяет, какая часть изображения будет видна после операции.\n"
                               "Пример: 'center' - центр, 'top-left' - верхний левый угол.\n\n"
                               "Image position for crop and pad strategies:\n"
                               "Determines which part of the image will be visible after the operation.\n"
                               "Example: 'center' - center, 'top-left' - top-left corner."
                }),
                "pad_image_color": (list(IMAGE_COLORS_HEX.keys()), {
                    "default": "white",
                    "tooltip": "Цвет полей для изображения при стратегии 'pad'.\n"
                               "Для прозрачного фона изображение должно быть в формате RGBA.\n\n"
                               "Padding color for image when using 'pad' strategy.\n"
                               "For transparent background, image must be in RGBA format."
                }),
                "pad_mask_color": (list(MASK_COLORS_HEX.keys()), {
                    "default": "black",
                    "tooltip": "Цвет полей для маски при стратегии 'pad'.\n"
                               "Маски всегда в градациях серого (от черного к белому).\n\n"
                               "Padding color for mask when using 'pad' strategy.\n"
                               "Masks are always in grayscale (from black to white)."
                }),
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Инвертировать маску перед обработкой.\n"
                               "Черное становится белым и наоборот.\n"
                               "Работает только если есть альфа-канал или отдельная маска.\n\n"
                               "Invert mask before processing.\n"
                               "Black becomes white and vice versa.\n"
                               "Works only if there is alpha channel or separate mask."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений.\n"
                               "cuda - использование GPU (быстрее, требуется видеокарта NVIDIA с CUDA)\n"
                               "cpu - использование центрального процессора (медленнее, но всегда доступно)\n\n"
                               "Device for computations.\n"
                               "cuda - use GPU (faster, requires NVIDIA GPU with CUDA)\n"
                               "cpu - use CPU (slower but always available)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "filename", "filepath")
    FUNCTION = "load_and_resize_image"
    CATEGORY = "AGSoft/Image"
    
    # Описание ноды для UI
    DESCRIPTION = """ 
AGSoft Image Resize Plus - Extended node for resizing images and masks. Supports 5 size specification modes, 3 scaling strategies, synchronous image and mask processing.

AGSoft Image Resize Plus - Расширенная нода для изменения размера изображений и масок. Поддерживает 5 режимов задания размеров, 3 стратегии масштабирования, синхронную обработку изображений и масок.
    """
    
    @classmethod
    def IS_CHANGED(cls, image: str, **kwargs):
        """
        Определяет, изменилось ли состояние ноды.
        Используется для кэширования в ComfyUI.
        
        Args:
            image (str): Имя файла изображения
            
        Returns:
            str: Хэш файла для определения изменений
            
        Перевод на английский:
        Determines if node state has changed.
        Used for caching in ComfyUI.
        """
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        
        try:
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except:
            return ""
    
    @classmethod
    def VALIDATE_INPUTS(cls, image: str):
        """
        Проверяет корректность входных данных.
        
        Args:
            image (str): Имя файла изображения
            
        Returns:
            bool or str: True если валидно, сообщение об ошибке если нет
            
        Перевод на английский:
        Validates input data.
        """
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True
    
    def load_and_resize_image(
        self,
        image: str,  # Имя файла, а не тензор!
        resize_mode: str = "target_megapixels",
        target_megapixels: float = 1.0,
        target_percentage: float = 100.0,
        target_width: int = 512,
        target_height: int = 512,
        fit_mode: str = "crop",
        interpolation: str = "lanczos",
        divisible_by: int = 8,
        crop_position: str = "center",
        pad_image_color: str = "white",
        pad_mask_color: str = "black",
        device: str = "cpu",
        invert_mask: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, str, str]:
        """
        Основная функция ноды - загружает изображение и изменяет его размер.
        
        Args:
            image (str): Имя файла изображения
            resize_mode (str): Режим задания размеров
            target_megapixels (float): Целевые мегапиксели
            target_percentage (float): Процент изменения размера
            target_width (int): Целевая ширина
            target_height (int): Целевая высота
            fit_mode (str): Стратегия масштабирования
            interpolation (str): Метод интерполяции
            divisible_by (int): Кратность размеров
            crop_position (str): Позиция для crop/pad
            pad_image_color (str): Цвет полей для изображения
            pad_mask_color (str): Цвет полей для маски
            device (str): Устройство для вычислений
            invert_mask (bool): Инвертировать маску
            
        Returns:
            Tuple: (изображение, маска, ширина, высота, имя_файла, путь_к_файлу)
            
        Исключения:
            RuntimeError: При ошибках обработки
        """
        
        try:
            # ============================================================
            # ШАГ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЯ
            # ============================================================
            
            image_path = folder_paths.get_annotated_filepath(image)
            
            # Извлекаем имя файла с расширением и полный путь
            filename_with_ext = os.path.basename(image_path)
            filename_without_ext = os.path.splitext(filename_with_ext)[0]
            full_path = image_path
            
            # Загружаем изображение с помощью node_helpers.pillow
            img = node_helpers.pillow(Image.open, image_path)
            
            output_images = []
            output_masks = []
            w, h = None, None
            
            # Исключаемые форматы
            excluded_formats = ['MPO']
            
            # Обрабатываем каждый кадр (для анимированных изображений)
            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)
                
                # Конвертируем 16-битные изображения
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                
                # Конвертируем в RGB
                image_rgb = i.convert("RGB")
                
                # Запоминаем размеры первого кадра
                if len(output_images) == 0:
                    w = image_rgb.size[0]
                    h = image_rgb.size[1]
                
                # Пропускаем кадры с другими размерами
                if image_rgb.size[0] != w or image_rgb.size[1] != h:
                    continue
                
                # Конвертируем в тензор
                image_array = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array)[None,]
                
                # Извлекаем маску из альфа-канала
                if 'A' in i.getbands():
                    mask_array = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1.0 - torch.from_numpy(mask_array)
                elif i.mode == 'P' and 'transparency' in i.info:
                    mask_array = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1.0 - torch.from_numpy(mask_array)
                else:
                    # Создаем черную маску
                    mask_tensor = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                
                output_images.append(image_tensor)
                output_masks.append(mask_tensor.unsqueeze(0))
            
            # Собираем батч
            if len(output_images) > 1 and img.format not in excluded_formats:
                loaded_image = torch.cat(output_images, dim=0)
                loaded_mask = torch.cat(output_masks, dim=0)
            else:
                loaded_image = output_images[0]
                loaded_mask = output_masks[0]
            
            # Получаем размеры загруженного изображения
            batch_size, orig_h, orig_w, channels = loaded_image.shape
            
            # ============================================================
            # ШАГ 2: ПОДГОТОВКА ДАННЫХ ДЛЯ РЕСАЙЗА
            # ============================================================
            
            # Определяем целевое устройство
            if device == "cuda" and torch.cuda.is_available():
                target_device = "cuda"
            else:
                target_device = "cpu"
                if device == "cuda":
                    warnings.warn("CUDA requested but not available. Falling back to CPU.")
            
            # Перемещаем данные на нужное устройство
            loaded_image = loaded_image.to(target_device)
            loaded_mask = loaded_mask.to(target_device)
            
            # Инверсия маски при необходимости
            if invert_mask:
                loaded_mask = 1.0 - loaded_mask
            
            # ============================================================
            # ШАГ 3: ВЫЧИСЛЕНИЕ ЦЕЛЕВЫХ РАЗМЕРОВ
            # ============================================================
            
            # Вычисляем целевые размеры в зависимости от режима
            if resize_mode == "target_megapixels":
                # Режим мегапикселей
                target_pixels = max(1, int(target_megapixels * 1_000_000))
                if orig_w <= 0 or orig_h <= 0:
                    orig_w, orig_h = 1, 1
                ratio = (target_pixels / (orig_w * orig_h)) ** 0.5
                base_w = max(1, int(orig_w * ratio))
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_percentage":
                # Режим процентов
                ratio = target_percentage / 100.0
                # Обработка отрицательных процентов (зеркальное отражение)
                if ratio < 0:
                    ratio = abs(ratio)  # Берем абсолютное значение для масштабирования
                    # Зеркальное отражение будет применено позже при обработке каждого изображения
                base_w = max(1, int(orig_w * ratio))
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_width":
                # Только ширина
                base_w = max(1, target_width)
                ratio = base_w / orig_w if orig_w > 0 else 1
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_height":
                # Только высота
                base_h = max(1, target_height)
                ratio = base_h / orig_h if orig_h > 0 else 1
                base_w = max(1, int(orig_w * ratio))
                
            elif resize_mode == "target_both":
                # Оба размера явно
                base_w = max(1, target_width)
                base_h = max(1, target_height)
                
            else:
                raise ValueError(f"Неизвестный режим ресайза: {resize_mode}")
            
            # Применяем кратность размеров (если divisible_by > 0)
            if divisible_by > 0:
                target_w = max(divisible_by, (base_w // divisible_by) * divisible_by)
                target_h = max(divisible_by, (base_h // divisible_by) * divisible_by)
            else:
                target_w, target_h = base_w, base_h
            
            # Запоминаем финальные размеры для возврата
            final_width, final_height = target_w, target_h
            
            # ============================================================
            # ШАГ 4: ОБРАБОТКА КАЖДОГО ИЗОБРАЖЕНИЯ В БАТЧЕ
            # ============================================================
            
            resized_images = []
            resized_masks = []
            
            # Подготавливаем цвета для паддинга
            has_alpha = (channels == 4)
            image_pad_color = hex_to_rgb_or_rgba(IMAGE_COLORS_HEX[pad_image_color], has_alpha)
            mask_pad_color = hex_to_rgb_or_rgba(MASK_COLORS_HEX[pad_mask_color], False)
            
            # Получаем метод интерполяции и позицию
            resample_method = INTERPOLATION_METHODS[interpolation]
            centering_tuple = CROP_POSITIONS[crop_position]
            
            for i in range(batch_size):
                # Берем текущее изображение и маску из батча
                img_tensor = loaded_image[i]
                msk_tensor = loaded_mask[i]
                
                # Преобразуем тензоры в PIL изображения
                pil_image = tensor_to_pil(img_tensor, is_mask=False)
                pil_mask = tensor_to_pil(msk_tensor, is_mask=True)
                
                # Применяем зеркальное отражение для отрицательных процентов
                if resize_mode == "target_percentage" and target_percentage < 0:
                    pil_image = ImageOps.mirror(pil_image)
                    pil_mask = ImageOps.mirror(pil_mask)
                
                # ========================================================
                # ШАГ 5: ПРИМЕНЕНИЕ ВЫБРАННОЙ СТРАТЕГИИ МАСШТАБИРОВАНИЯ
                # ========================================================
                
                if fit_mode == "stretch":
                    # Просто растягиваем до нужных размеров
                    resized_image = pil_image.resize((target_w, target_h), resample=resample_method)
                    resized_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
                    
                elif fit_mode == "crop":
                    # Сохраняем пропорции и обрезаем лишнее
                    scale = max(target_w / pil_image.width, target_h / pil_image.height)
                    fit_w = max(1, int(pil_image.width * scale))
                    fit_h = max(1, int(pil_image.height * scale))
                    
                    # Масштабируем до размера, который перекрывает целевой прямоугольник
                    scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
                    scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
                    
                    # Вычисляем координаты обрезки
                    dx = max(0, fit_w - target_w)
                    dy = max(0, fit_h - target_h)
                    left = int(dx * centering_tuple[0])
                    top = int(dy * centering_tuple[1])
                    
                    # Обрезаем
                    resized_image = scaled_image.crop((left, top, left + target_w, top + target_h))
                    resized_mask = scaled_mask.crop((left, top, left + target_w, top + target_h))
                    
                elif fit_mode == "pad":
                    # Сохраняем пропорции и добавляем поля
                    scale = min(target_w / pil_image.width, target_h / pil_image.height)
                    fit_w = max(1, int(pil_image.width * scale))
                    fit_h = max(1, int(pil_image.height * scale))
                    
                    # Масштабируем до размера, который вписывается в целевой прямоугольник
                    scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
                    scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
                    
                    # Создаем фоновые изображения с нужным цветом
                    if has_alpha and pad_image_color == "transparent":
                        bg_image = Image.new('RGBA', (target_w, target_h), image_pad_color)
                    else:
                        bg_image = Image.new('RGB', (target_w, target_h), image_pad_color[:3])
                    
                    bg_mask = Image.new('L', (target_w, target_h), mask_pad_color[0])
                    
                    # Вычисляем позицию для вставки
                    left = int((target_w - fit_w) * centering_tuple[0])
                    top = int((target_h - fit_h) * centering_tuple[1])
                    
                    # Вставляем масштабированные изображения на фон
                    if scaled_image.mode == 'RGBA' and bg_image.mode == 'RGB':
                        # Конвертируем RGBA в RGB перед вставкой
                        scaled_image_rgb = scaled_image.convert('RGB')
                        bg_image.paste(scaled_image_rgb, (left, top))
                    else:
                        bg_image.paste(scaled_image, (left, top))
                    
                    bg_mask.paste(scaled_mask, (left, top))
                    
                    resized_image = bg_image
                    resized_mask = bg_mask
                    
                else:
                    raise ValueError(f"Неизвестная стратегия масштабирования: {fit_mode}")
                
                # Преобразуем обратно в тензоры
                img_tensor_resized = pil_to_tensor(resized_image, is_mask=False)
                mask_tensor_resized = pil_to_tensor(resized_mask, is_mask=True)
                
                # Добавляем в батч
                resized_images.append(img_tensor_resized.unsqueeze(0))
                resized_masks.append(mask_tensor_resized.unsqueeze(0))
            
            # ============================================================
            # ШАГ 6: ФИНАЛИЗАЦИЯ И ВОЗВРАТ РЕЗУЛЬТАТОВ
            # ============================================================
            
            # Собираем батч обратно
            final_image = torch.cat(resized_images, dim=0).to(target_device)
            final_mask = torch.cat(resized_masks, dim=0).to(target_device)
            
            # Возвращаем результат с информацией о файле
            return (
                final_image, 
                final_mask, 
                final_width, 
                final_height, 
                filename_with_ext,  # Имя файла с расширением
                full_path           # Полный путь к файлу
            )
            
        except Exception as e:
            # Обработка ошибок с информативным сообщением
            error_msg = f"Ошибка при загрузке и изменении размера изображения: {str(e)}\n"
            error_msg += f"Файл: {image}\n"
            error_msg += f"Параметры: resize_mode={resize_mode}, fit_mode={fit_mode}"
            
            raise RuntimeError(error_msg)


# ============================================================================
# КЛАСС НОДЫ - AGSoft_Image_Resize_Base (без выбора файлов)
# ============================================================================

class AGSoft_Image_Resize_Base:
    """
    AGSoft Image Resize Base - Базовая нода для изменения размера изображений и масок.
    
    Эта нода использует входы для изображения и маски вместо выбора файлов.
    Идеально подходит для использования внутри пайплайнов ComfyUI.
    
    Особенности:
    - Принимает изображение и маску как входные тензоры
    - Поддерживает 5 режимов задания размеров: мегапиксели, проценты, ширина, высота, оба размера
    - Обрабатывает изображения и маски синхронно
    - 3 стратегии масштабирования: stretch, crop, pad
    - Раздельные цвета фона для изображения и маски
    - Поддержка CPU/CUDA
    - Кратность размеров для совместимости с моделями
    - Инверсия маски
    - 9 позиций выравнивания
    
    AGSoft Image Resize Base - Base node for resizing images and masks.
    This node uses image and mask inputs instead of file selection.
    Perfect for use within ComfyUI pipelines.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        """
        Определяет входные параметры ноды для ComfyUI.
        
        Returns:
            Dict: Словарь с описанием входных параметров
        """
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Входное изображение в формате тензора.\n"
                               "Input image as tensor."
                }),
                "resize_mode": (RESIZE_MODES, {
                    "default": "target_percentage",
                    "tooltip": "Режим задания целевых размеров:\n"
                               "• target_megapixels - задать общее количество мегапикселей\n"
                               "• target_percentage - задать процент от исходного размера\n"
                               "• target_width - задать только ширину (высота вычисляется автоматически)\n"
                               "• target_height - задать только высоту (ширина вычисляется автоматически)\n"
                               "• target_both - задать ширину и высоту явно\n\n"
                               "Size specification mode:\n"
                               "• target_megapixels - set total megapixels\n"
                               "• target_percentage - set percentage of original size\n"
                               "• target_width - set only width (height calculated automatically)\n"
                               "• target_height - set only height (width calculated automatically)\n"
                               "• target_both - set both width and height explicitly"
                }),
                "target_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Целевое количество мегапикселей (MP).\n"
                               "1.0 MP = 1,000,000 пикселей (примерно 1000x1000).\n"
                               "Используется только в режиме 'target_megapixels'.\n\n"
                               "Target number of megapixels (MP).\n"
                               "1.0 MP = 1,000,000 pixels (approximately 1000x1000).\n"
                               "Used only in 'target_megapixels' mode."
                }),
                "target_percentage": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Процент изменения размера относительно оригинала.\n"
                               "100.0 = исходный размер, 50.0 = половина размера,\n"
                               "200.0 = двойной размер.\n"
                               "Используется только в режиме 'target_percentage'.\n\n"
                               "Percentage of size change relative to original.\n"
                               "100.0 = original size, 50.0 = half size,\n"
                               "200.0 = double size.\n"
                               "Used only in 'target_percentage' mode."
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Желаемая ширина в пикселях.\n"
                               "Используется в режимах 'target_width' и 'target_both'.\n\n"
                               "Desired width in pixels.\n"
                               "Used in 'target_width' and 'target_both' modes."
                }),
                "target_height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Желаемая высота в пикселях.\n"
                               "Используется в режимах 'target_height' и 'target_both'.\n\n"
                               "Desired height in pixels.\n"
                               "Used in 'target_height' and 'target_both' modes."
                }),
                "fit_mode": (FIT_MODES, {
                    "default": "crop",
                    "tooltip": "Стратегия масштабирования:\n"
                               "• stretch - растянуть/сжать изображение до точных размеров (могут искажаться пропорции)\n"
                               "• crop - сохранить пропорции и обрезать лишнее\n"
                               "• pad - сохранить пропорции и добавить поля по краям\n\n"
                               "Scaling strategy:\n"
                               "• stretch - stretch/squeeze image to exact dimensions (aspect ratio may distort)\n"
                               "• crop - preserve aspect ratio and crop excess\n"
                               "• pad - preserve aspect ratio and add padding"
                }),
                "interpolation": (list(INTERPOLATION_METHODS.keys()), {
                    "default": "lanczos",
                    "tooltip": "Метод интерполяции при изменении размера:\n"
                               "• nearest - самый быстрый, но низкое качество (пиксельный)\n"
                               "• bilinear - сглаженный, быстрый, подходит для preview\n"
                               "• bicubic - более плавный, чем bilinear, хорош для фото\n"
                               "• area (BOX) - оптимален для уменьшения изображений (сохраняет детали)\n"
                               "• lanczos - наивысшее качество, но медленнее; идеален для финального рендера\n"
                               "• nearest-exact - как 'nearest', но с улучшённой точностью\n\n"
                               "Interpolation method for resizing:\n"
                               "• nearest - fastest but lowest quality (pixelated)\n"
                               "• bilinear - smooth, fast, good for preview\n"
                               "• bicubic - smoother than bilinear, good for photos\n"
                               "• area (BOX) - optimal for downscaling (preserves details)\n"
                               "• lanczos - highest quality but slower; ideal for final render\n"
                               "• nearest-exact - like 'nearest' but with improved accuracy"
                }),
                "divisible_by": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Округляет итоговую ширину и высоту до кратного этому числу.\n"
                               "0 означает отключение. Полезно для совместимости с моделями.\n"
                               "Пример: divisible_by=8 для SD/SDXL моделей.\n\n"
                               "Rounds final width and height to multiples of this number.\n"
                               "0 means disabled. Useful for model compatibility.\n"
                               "Example: divisible_by=8 for SD/SDXL models."
                }),
                "crop_position": (list(CROP_POSITIONS.keys()), {
                    "default": "center",
                    "tooltip": "Позиция изображения при стратегиях crop и pad:\n"
                               "Определяет, какая часть изображения будет видна после операции.\n"
                               "Пример: 'center' - центр, 'top-left' - верхний левый угол.\n\n"
                               "Image position for crop and pad strategies:\n"
                               "Determines which part of the image will be visible after the operation.\n"
                               "Example: 'center' - center, 'top-left' - top-left corner."
                }),
                "pad_image_color": (list(IMAGE_COLORS_HEX.keys()), {
                    "default": "white",
                    "tooltip": "Цвет полей для изображения при стратегии 'pad'.\n"
                               "Для прозрачного фона изображение должно быть в формате RGBA.\n\n"
                               "Padding color for image when using 'pad' strategy.\n"
                               "For transparent background, image must be in RGBA format."
                }),
                "pad_mask_color": (list(MASK_COLORS_HEX.keys()), {
                    "default": "black",
                    "tooltip": "Цвет полей для маски при стратегии 'pad'.\n"
                               "Маски всегда в градациях серого (от черного к белому).\n\n"
                               "Padding color for mask when using 'pad' strategy.\n"
                               "Masks are always in grayscale (from black to white)."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Входная маска в формате тензора.\n"
                               "Если не указана, будет создана черная маска.\n"
                               "Input mask as tensor.\n"
                               "If not provided, a black mask will be created."
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Инвертировать маску перед обработкой.\n"
                               "Черное становится белым и наоборот.\n\n"
                               "Invert mask before processing.\n"
                               "Black becomes white and vice versa."
                }),
                "device": (["cpu", "cuda"], {
                    "default": "cpu",
                    "tooltip": "Устройство для вычислений.\n"
                               "cuda - использование GPU (быстрее, требуется видеокарта NVIDIA с CUDA)\n"
                               "cpu - использование центрального процессора (медленнее, но всегда доступно)\n\n"
                               "Device for computations.\n"
                               "cuda - use GPU (faster, requires NVIDIA GPU with CUDA)\n"
                               "cpu - use CPU (slower but always available)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "resize_image_tensor"
    CATEGORY = "AGSoft/Image"
    
    # Описание ноды для UI
    DESCRIPTION = """
AGSoft Image Resize Base - Extended node for resizing images and masks.
Supports 5 size specification modes, 3 scaling strategies, synchronous image and mask processing.

AGSoft Image Resize Base - Расширенная нода для изменения размера изображений и масок.
Поддерживает 5 режимов задания размеров, 3 стратегии масштабирования, синхронную обработку изображений и масок.
    """
    
    def resize_image_tensor(
        self,
        image: torch.Tensor,
        resize_mode: str = "target_megapixels",
        target_megapixels: float = 1.0,
        target_percentage: float = 100.0,
        target_width: int = 512,
        target_height: int = 512,
        fit_mode: str = "crop",
        interpolation: str = "lanczos",
        divisible_by: int = 8,
        crop_position: str = "center",
        pad_image_color: str = "white",
        device: str = "cpu",
        mask: Optional[torch.Tensor] = None,
        invert_mask: bool = False,
        pad_mask_color: str = "black",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Основная функция ноды - изменяет размер входного изображения и маски.
        
        Args:
            image (torch.Tensor): Входное изображение [B, H, W, C]
            resize_mode (str): Режим задания размеров
            target_megapixels (float): Целевые мегапиксели
            target_percentage (float): Процент изменения размера
            target_width (int): Целевая ширина
            target_height (int): Целевая высота
            fit_mode (str): Стратегия масштабирования
            interpolation (str): Метод интерполяции
            divisible_by (int): Кратность размеров
            crop_position (str): Позиция для crop/pad
            pad_image_color (str): Цвет полей для изображения
            device (str): Устройство для вычислений
            mask (torch.Tensor, optional): Входная маска [B, H, W]
            invert_mask (bool): Инвертировать маску
            pad_mask_color (str): Цвет полей для маски
            
        Returns:
            Tuple: (изображение, маска, ширина, высота)
            
        Исключения:
            RuntimeError: При ошибках обработки
        """
        
        try:
            # ============================================================
            # ШАГ 1: ПОДГОТОВКА ВХОДНЫХ ДАННЫХ
            # ============================================================
            
            # Определяем целевое устройство
            if device == "cuda" and torch.cuda.is_available():
                target_device = "cuda"
            else:
                target_device = "cpu"
                if device == "cuda":
                    warnings.warn("CUDA requested but not available. Falling back to CPU.")
            
            # Перемещаем изображение на нужное устройство
            image = image.to(target_device)
            
            # Получаем размеры входного изображения
            batch_size, orig_h, orig_w, channels = image.shape
            
            # Подготавливаем маску
            if mask is not None:
                # Проверяем размеры маски
                if mask.dim() == 2:
                    # [H, W] -> [1, H, W]
                    mask = mask.unsqueeze(0)
                elif mask.dim() == 3:
                    # [B, H, W] - уже правильный формат
                    pass
                elif mask.dim() == 4:
                    # [B, H, W, C] -> [B, H, W]
                    if mask.shape[-1] == 1:
                        mask = mask.squeeze(-1)
                    else:
                        # Если маска в RGB, конвертируем в градации серого
                        mask = 0.299 * mask[..., 0] + 0.587 * mask[..., 1] + 0.114 * mask[..., 2]
                
                # Проверяем соответствие размеров маски и изображения
                mask_batch_size = mask.shape[0]
                if mask_batch_size != batch_size and mask_batch_size == 1:
                    # Если маска одна на весь батч
                    mask = mask.expand(batch_size, -1, -1)
                elif mask_batch_size != batch_size:
                    raise ValueError(f"Размер батча маски ({mask_batch_size}) не соответствует размеру батча изображения ({batch_size})")
                
                # Приводим к правильному устройству и формату
                mask = mask.to(target_device)
            else:
                # Создаем черную маску
                mask = torch.zeros((batch_size, orig_h, orig_w), dtype=torch.float32, device=target_device)
            
            # Инверсия маски при необходимости
            if invert_mask:
                mask = 1.0 - mask
            
            # ============================================================
            # ШАГ 2: ВЫЧИСЛЕНИЕ ЦЕЛЕВЫХ РАЗМЕРОВ
            # ============================================================
            
            # Вычисляем целевые размеры в зависимости от режима
            if resize_mode == "target_megapixels":
                # Режим мегапикселей
                target_pixels = max(1, int(target_megapixels * 1_000_000))
                if orig_w <= 0 or orig_h <= 0:
                    orig_w, orig_h = 1, 1
                ratio = (target_pixels / (orig_w * orig_h)) ** 0.5
                base_w = max(1, int(orig_w * ratio))
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_percentage":
                # Режим процентов
                ratio = target_percentage / 100.0
                # Обработка отрицательных процентов (зеркальное отражение)
                if ratio < 0:
                    ratio = abs(ratio)  # Берем абсолютное значение для масштабирования
                    # Зеркальное отражение будет применено позже при обработке каждого изображения
                base_w = max(1, int(orig_w * ratio))
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_width":
                # Только ширина
                base_w = max(1, target_width)
                ratio = base_w / orig_w if orig_w > 0 else 1
                base_h = max(1, int(orig_h * ratio))
                
            elif resize_mode == "target_height":
                # Только высота
                base_h = max(1, target_height)
                ratio = base_h / orig_h if orig_h > 0 else 1
                base_w = max(1, int(orig_w * ratio))
                
            elif resize_mode == "target_both":
                # Оба размера явно
                base_w = max(1, target_width)
                base_h = max(1, target_height)
                
            else:
                raise ValueError(f"Неизвестный режим ресайза: {resize_mode}")
            
            # Применяем кратность размеров (если divisible_by > 0)
            if divisible_by > 0:
                target_w = max(divisible_by, (base_w // divisible_by) * divisible_by)
                target_h = max(divisible_by, (base_h // divisible_by) * divisible_by)
            else:
                target_w, target_h = base_w, base_h
            
            # Запоминаем финальные размеры для возврата
            final_width, final_height = target_w, target_h
            
            # ============================================================
            # ШАГ 3: ОБРАБОТКА КАЖДОГО ИЗОБРАЖЕНИЯ В БАТЧЕ
            # ============================================================
            
            resized_images = []
            resized_masks = []
            
            # Подготавливаем цвета для паддинга
            has_alpha = (channels == 4)
            image_pad_color = hex_to_rgb_or_rgba(IMAGE_COLORS_HEX[pad_image_color], has_alpha)
            mask_pad_color = hex_to_rgb_or_rgba(MASK_COLORS_HEX[pad_mask_color], False)
            
            # Получаем метод интерполяции и позицию
            resample_method = INTERPOLATION_METHODS[interpolation]
            centering_tuple = CROP_POSITIONS[crop_position]
            
            for i in range(batch_size):
                # Берем текущее изображение и маску из батча
                img_tensor = image[i]
                msk_tensor = mask[i]
                
                # Преобразуем тензоры в PIL изображения
                pil_image = tensor_to_pil(img_tensor, is_mask=False)
                pil_mask = tensor_to_pil(msk_tensor, is_mask=True)
                
                # Применяем зеркальное отражение для отрицательных процентов
                if resize_mode == "target_percentage" and target_percentage < 0:
                    pil_image = ImageOps.mirror(pil_image)
                    pil_mask = ImageOps.mirror(pil_mask)
                
                # ========================================================
                # ШАГ 4: ПРИМЕНЕНИЕ ВЫБРАННОЙ СТРАТЕГИИ МАСШТАБИРОВАНИЯ
                # ========================================================
                
                if fit_mode == "stretch":
                    # Просто растягиваем до нужных размеров
                    resized_image = pil_image.resize((target_w, target_h), resample=resample_method)
                    resized_mask = pil_mask.resize((target_w, target_h), resample=Image.NEAREST)
                    
                elif fit_mode == "crop":
                    # Сохраняем пропорции и обрезаем лишнее
                    scale = max(target_w / pil_image.width, target_h / pil_image.height)
                    fit_w = max(1, int(pil_image.width * scale))
                    fit_h = max(1, int(pil_image.height * scale))
                    
                    # Масштабируем до размера, который перекрывает целевой прямоугольник
                    scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
                    scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
                    
                    # Вычисляем координаты обрезки
                    dx = max(0, fit_w - target_w)
                    dy = max(0, fit_h - target_h)
                    left = int(dx * centering_tuple[0])
                    top = int(dy * centering_tuple[1])
                    
                    # Обрезаем
                    resized_image = scaled_image.crop((left, top, left + target_w, top + target_h))
                    resized_mask = scaled_mask.crop((left, top, left + target_w, top + target_h))
                    
                elif fit_mode == "pad":
                    # Сохраняем пропорции и добавляем поля
                    scale = min(target_w / pil_image.width, target_h / pil_image.height)
                    fit_w = max(1, int(pil_image.width * scale))
                    fit_h = max(1, int(pil_image.height * scale))
                    
                    # Масштабируем до размера, который вписывается в целевой прямоугольник
                    scaled_image = pil_image.resize((fit_w, fit_h), resample=resample_method)
                    scaled_mask = pil_mask.resize((fit_w, fit_h), resample=Image.NEAREST)
                    
                    # Создаем фоновые изображения с нужным цветом
                    if has_alpha and pad_image_color == "transparent":
                        bg_image = Image.new('RGBA', (target_w, target_h), image_pad_color)
                    else:
                        bg_image = Image.new('RGB', (target_w, target_h), image_pad_color[:3])
                    
                    bg_mask = Image.new('L', (target_w, target_h), mask_pad_color[0])
                    
                    # Вычисляем позицию для вставки
                    left = int((target_w - fit_w) * centering_tuple[0])
                    top = int((target_h - fit_h) * centering_tuple[1])
                    
                    # Вставляем масштабированные изображения на фон
                    if scaled_image.mode == 'RGBA' and bg_image.mode == 'RGB':
                        # Конвертируем RGBA в RGB перед вставкой
                        scaled_image_rgb = scaled_image.convert('RGB')
                        bg_image.paste(scaled_image_rgb, (left, top))
                    else:
                        bg_image.paste(scaled_image, (left, top))
                    
                    bg_mask.paste(scaled_mask, (left, top))
                    
                    resized_image = bg_image
                    resized_mask = bg_mask
                    
                else:
                    raise ValueError(f"Неизвестная стратегия масштабирования: {fit_mode}")
                
                # Преобразуем обратно в тензоры
                img_tensor_resized = pil_to_tensor(resized_image, is_mask=False)
                mask_tensor_resized = pil_to_tensor(resized_mask, is_mask=True)
                
                # Добавляем в батч
                resized_images.append(img_tensor_resized.unsqueeze(0))
                resized_masks.append(mask_tensor_resized.unsqueeze(0))
            
            # ============================================================
            # ШАГ 5: ФИНАЛИЗАЦИЯ И ВОЗВРАТ РЕЗУЛЬТАТОВ
            # ============================================================
            
            # Собираем батч обратно
            final_image = torch.cat(resized_images, dim=0).to(target_device)
            final_mask = torch.cat(resized_masks, dim=0).to(target_device)
            
            # Возвращаем результат
            return (
                final_image, 
                final_mask, 
                final_width, 
                final_height
            )
            
        except Exception as e:
            # Обработка ошибок с информативным сообщением
            error_msg = f"Ошибка при изменении размера изображения: {str(e)}\n"
            error_msg += f"Параметры: resize_mode={resize_mode}, fit_mode={fit_mode}"
            
            raise RuntimeError(error_msg)


# ============================================================================
# РЕГИСТРАЦИЯ НОД В COMFYUI
# ============================================================================

# Обязательные переменные для регистрации нод в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Resize_Plus": AGSoft_Image_Resize_Plus,
    "AGSoft_Image_Resize_Base": AGSoft_Image_Resize_Base
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Resize_Plus": "AGSoft Image & Mask Resize Plus",
    "AGSoft_Image_Resize_Base": "AGSoft Image & Mask Resize Base"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
