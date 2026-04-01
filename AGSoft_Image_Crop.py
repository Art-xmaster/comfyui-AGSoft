"""
AGSoft Image Crop - нода для обрезки изображений в ComfyUI
==========================================================
Модуль предоставляет узел для гибкой обрезки изображений с поддержкой:
- 9 предустановленных позиций обрезки
- Процентной обрезки (относительно размера изображения)
- Сохранения пропорций исходного изображения
- Инверсной обрезки (обрезка краёв)
- Пользовательских координат обрезки
- Поддержки маски для точной обрезки
- Кэширования для оптимизации производительности
- Выбора устройства выполнения (CPU/GPU/Auto)

Автор: AGSoft
Версия: 2.0.0
Дата: 2026-03-31
"""

import torch
import hashlib
from typing import Dict, Tuple, Optional, Any, Union
from functools import lru_cache
from enum import Enum


class DeviceType(Enum):
    """Перечисление типов устройств для выполнения операций."""
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class AGSoft_Image_Crop:
    """
    Узел для обрезки изображения с расширенными возможностями.
    
    Ключевые особенности:
    - Поддержка батчей изображений (batch processing)
    - 9 предустановленных позиций + пользовательская
    - Процентная обрезка для адаптивных workflow
    - Сохранение пропорций исходного изображения
    - Инверсная обрезка (обрезка краёв с заданным процентом)
    - Поддержка маски для точной обрезки
    - Кэширование результатов для одинаковых входных данных
    - Выбор устройства выполнения (CPU/GPU/Auto)
    
    Входные параметры:
    - image: тензор изображения [B, H, W, C]
    - mask: опциональная маска [B, H, W] (должна совпадать по размеру с image)
    - width/height: размеры обрезки (пиксели или проценты)
    - position: позиция обрезки
    - use_percentage: интерпретировать width/height как проценты
    - maintain_aspect_ratio: сохранять пропорции изображения
    - inverse_crop_percent: обрезка краёв (0-49%)
    - start_x/start_y: начальные координаты (только для position="custom")
    - device: устройство выполнения (auto/cpu/gpu)
    - use_cache: использовать кэширование результатов
    
    Выходные данные:
    - cropped_image: обрезанное изображение [B, H_cropped, W_cropped, C]
    - cropped_mask: обрезанная маска [B, H_cropped, W_cropped] (если маска была на входе)
    - cropped_width: ширина обрезанного изображения
    - cropped_height: высота обрезанного изображения
    """

    def __init__(self) -> None:
        """Инициализация узла с кэшем."""
        super().__init__()
        # Кэш для хранения результатов обрезки
        self._cache: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], int, int]] = {}
        self._max_cache_size: int = 100  # Максимальный размер кэша
        
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Определение входных параметров узла.
        
        Returns:
            Dict[str, Dict]: Словарь с обязательными и опциональными параметрами
        """
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Входное изображение в формате [BATCH, HEIGHT, WIDTH, CHANNELS]\n"
                              "Input image tensor in format [BATCH, HEIGHT, WIDTH, CHANNELS]"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Ширина обрезки в пикселях или процентах (зависит от use_percentage)\n"
                              "Crop width in pixels or percentage (depends on use_percentage)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Высота обрезки в пикселях или процентах (зависит от use_percentage)\n"
                              "Crop height in pixels or percentage (depends on use_percentage)"
                }),
                "position": ([
                    "top-left", "top-center", "top-right",
                    "center-left", "center", "center-right",
                    "bottom-left", "bottom-center", "bottom-right",
                    "custom"
                ], {
                    "default": "center",
                    "tooltip": "Позиция области обрезки:\n"
                              "- top-left: верхний левый угол\n"
                              "- top-center: верхний центр\n"
                              "- top-right: верхний правый угол\n"
                              "- center-left: центр по вертикали, левый край\n"
                              "- center: центр изображения\n"
                              "- center-right: центр по вертикали, правый край\n"
                              "- bottom-left: нижний левый угол\n"
                              "- bottom-center: нижний центр\n"
                              "- bottom-right: нижний правый угол\n"
                              "- custom: пользовательские координаты (start_x, start_y)\n\n"
                              "Crop area position:\n"
                              "- top-left: top-left corner\n"
                              "- top-center: top center\n"
                              "- top-right: top-right corner\n"
                              "- center-left: center vertically, left edge\n"
                              "- center: image center\n"
                              "- center-right: center vertically, right edge\n"
                              "- bottom-left: bottom-left corner\n"
                              "- bottom-center: bottom center\n"
                              "- bottom-right: bottom-right corner\n"
                              "- custom: custom coordinates (start_x, start_y)"
                })
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Опциональная маска для обрезки (должна совпадать по размеру с изображением)\n"
                              "Optional mask for cropping (must match image dimensions)"
                }),
                "use_percentage": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Да / Yes",
                    "label_off": "Нет / No",
                    "tooltip": "Если включено, width и height интерпретируются как проценты от размера изображения\n"
                              "If enabled, width and height are interpreted as percentage of image size"
                }),
                "maintain_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Да / Yes",
                    "label_off": "Нет / No",
                    "tooltip": "Сохранять соотношение сторон исходного изображения\n"
                              "Maintain original image aspect ratio"
                }),
                "inverse_crop_percent": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 49,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Процент обрезки от каждого края изображения (0-49%):\n"
                              "Например, 10% обрежет по 10% с каждой стороны\n"
                              "Полезно для удаления артефактов по краям\n\n"
                              "Percentage to crop from each edge (0-49%):\n"
                              "Example: 10% crops 10% from each side\n"
                              "Useful for removing edge artifacts"
                }),
                "start_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Начальная координата X для обрезки (только при position='custom')\n"
                              "Должна быть меньше (ширина изображения - ширина обрезки)\n\n"
                              "Start X coordinate for cropping (only when position='custom')\n"
                              "Must be less than (image width - crop width)"
                }),
                "start_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Начальная координата Y для обрезки (только при position='custom')\n"
                              "Должна быть меньше (высота изображения - высота обрезки)\n\n"
                              "Start Y coordinate for cropping (only when position='custom')\n"
                              "Must be less than (image height - crop height)"
                }),
                "device": (["auto", "cpu", "gpu"], {
                    "default": "auto",
                    "tooltip": "Устройство для выполнения операций:\n"
                              "- auto: автоматический выбор (GPU если доступен, иначе CPU)\n"
                              "- cpu: принудительное использование CPU\n"
                              "- gpu: принудительное использование GPU (если доступен)\n\n"
                              "Device for operations:\n"
                              "- auto: automatic selection (GPU if available, else CPU)\n"
                              "- cpu: force CPU usage\n"
                              "- gpu: force GPU usage (if available)"
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Да / Yes",
                    "label_off": "Нет / No",
                    "tooltip": "Использовать кэширование для ускорения повторных операций с теми же параметрами\n"
                              "Use caching to speed up repeated operations with same parameters"
                }),
                "fill_color": ("STRING", {
                    "default": "black",
                    "tooltip": "Цвет для заполнения при выходе за границы (black, white, или RGB значения через запятую, например: 255,0,0)\n"
                              "Fill color when cropping out of bounds (black, white, or comma-separated RGB, e.g.: 255,0,0)"
                }),
                "interpolation": (["bilinear", "nearest", "bicubic", "area"], {
                    "default": "bilinear",
                    "tooltip": "Метод интерполяции для изменения размера (если требуется)\n"
                              "Interpolation method for resizing (if needed)"
                })
            }
        }

    RETURN_TYPES: Tuple[str, ...] = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES: Tuple[str, ...] = ("cropped_image", "cropped_mask", "cropped_width", "cropped_height")
    FUNCTION: str = "crop_image"
    CATEGORY: str = "AGSoft/Image"
    DESCRIPTION: str = """
    Обрезка изображения с расширенными возможностями.
    
    Возможности:
    • 9 предустановленных позиций + пользовательские координаты
    • Процентная обрезка (относительно размера изображения)
    • Сохранение пропорций исходного изображения
    • Инверсная обрезка (удаление краёв)
    • Поддержка батчей изображений и масок
    • Кэширование для оптимизации производительности
    • Выбор устройства выполнения (CPU/GPU/Auto)
    
    Примеры использования:
    1. Центрированная обрезка 512x512: position="center"
    2. Обрезка 50% от центра: use_percentage=True, width=50, height=50
    3. Удаление краёв на 5%: inverse_crop_percent=5
    4. Сохранение пропорций при обрезке: maintain_aspect_ratio=True
    5. Обрезка с маской: подключите маску к входу mask
    
    Crop image with advanced features.
    
    Features:
    • 9 preset positions + custom coordinates
    • Percentage cropping (relative to image size)
    • Maintain original image aspect ratio
    • Inverse cropping (remove edges)
    • Batch processing for images and masks
    • Caching for performance optimization
    • Device selection (CPU/GPU/Auto)
    
    Usage examples:
    1. Center crop 512x512: position="center"
    2. Crop 50% from center: use_percentage=True, width=50, height=50
    3. Remove 5% of edges: inverse_crop_percent=5
    4. Maintain aspect ratio: maintain_aspect_ratio=True
    5. Crop with mask: connect mask to mask input
    """

    def _get_device(self, device_choice: str) -> torch.device:
        """
        Определяет устройство для выполнения операций.
        
        Args:
            device_choice: Выбор устройства ("auto", "cpu", "gpu")
            
        Returns:
            torch.device: Устройство для выполнения операций
        """
        if device_choice == "cpu":
            return torch.device("cpu")
        elif device_choice == "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("[AGSoft Crop] Предупреждение: GPU не доступен, используется CPU / Warning: GPU not available, using CPU")
                return torch.device("cpu")
        else:  # auto
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_cache_key(self, image_hash: str, mask_hash: Optional[str], 
                           width: int, height: int, position: str,
                           use_percentage: bool, maintain_aspect_ratio: bool,
                           inverse_crop_percent: int, start_x: int, start_y: int) -> str:
        """
        Генерирует уникальный ключ для кэша на основе всех параметров.
        
        Args:
            image_hash: Хэш изображения
            mask_hash: Хэш маски (если есть)
            width: Ширина обрезки
            height: Высота обрезки
            position: Позиция обрезки
            use_percentage: Использовать проценты
            maintain_aspect_ratio: Сохранять пропорции
            inverse_crop_percent: Инверсная обрезка
            start_x: Начальная X координата
            start_y: Начальная Y координата
            
        Returns:
            str: Уникальный ключ для кэша
        """
        key_string = f"{image_hash}|{mask_hash}|{width}|{height}|{position}|{use_percentage}|{maintain_aspect_ratio}|{inverse_crop_percent}|{start_x}|{start_y}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_image_hash(self, image: torch.Tensor) -> str:
        """
        Вычисляет хэш изображения для кэширования.
        
        Args:
            image: Тензор изображения
            
        Returns:
            str: Хэш изображения
        """
        # Используем первые 1000 элементов для быстрого хэширования
        sample = image.flatten()[:1000]
        return hashlib.md5(sample.cpu().numpy().tobytes()).hexdigest()

    def _apply_inverse_crop(self, image: torch.Tensor, mask: Optional[torch.Tensor], 
                           percent: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Применяет инверсную обрезку (обрезает указанный процент с каждого края).
        
        Args:
            image: Входное изображение [B, H, W, C]
            mask: Входная маска [B, H, W] (опционально)
            percent: Процент обрезки от каждого края (0-49)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (обрезанное_изображение, обрезанная_маска)
        """
        if percent <= 0:
            return image, mask
            
        _, img_height, img_width, _ = image.shape
        
        # Вычисляем количество пикселей для обрезки с каждой стороны
        crop_pixels_h = int(img_height * (percent / 100.0))
        crop_pixels_w = int(img_width * (percent / 100.0))
        
        # Ограничиваем обрезку половиной изображения (макс 49% от каждого края)
        crop_pixels_h = min(crop_pixels_h, img_height // 2)
        crop_pixels_w = min(crop_pixels_w, img_width // 2)
        
        # Проверяем, что после обрезки остаётся хотя бы 1 пиксель
        if img_height - 2 * crop_pixels_h < 1 or img_width - 2 * crop_pixels_w < 1:
            return image, mask
            
        # Применяем обрезку к изображению
        cropped_image = image[:, crop_pixels_h:img_height - crop_pixels_h, 
                             crop_pixels_w:img_width - crop_pixels_w, :]
        
        # Применяем обрезку к маске, если она есть
        cropped_mask = None
        if mask is not None:
            cropped_mask = mask[:, crop_pixels_h:img_height - crop_pixels_h,
                               crop_pixels_w:img_width - crop_pixels_w]
        
        return cropped_image, cropped_mask

    def _calculate_crop_size(self, img_width: int, img_height: int, 
                            width: int, height: int, 
                            use_percentage: bool, 
                            maintain_aspect_ratio: bool) -> Tuple[int, int]:
        """
        Рассчитывает итоговый размер области обрезки.
        
        Args:
            img_width: Исходная ширина изображения
            img_height: Исходная высота изображения
            width: Запрошенная ширина (пиксели или проценты)
            height: Запрошенная высота (пиксели или проценты)
            use_percentage: Использовать проценты
            maintain_aspect_ratio: Сохранять пропорции
            
        Returns:
            Tuple[int, int]: (ширина_обрезки, высота_обрезки)
        """
        # Преобразуем проценты в пиксели при необходимости
        if use_percentage:
            crop_w = max(1, int(img_width * (width / 100.0)))
            crop_h = max(1, int(img_height * (height / 100.0)))
        else:
            crop_w = width
            crop_h = height
            
        # Сохраняем пропорции если нужно
        if maintain_aspect_ratio:
            original_aspect = img_width / img_height
            crop_aspect = crop_w / crop_h
            
            if crop_aspect > original_aspect:
                # Обрезка по ширине ограничена, подгоняем высоту
                crop_h = int(crop_w / original_aspect)
            else:
                # Обрезка по высоте ограничена, подгоняем ширину
                crop_w = int(crop_h * original_aspect)
                
        # Ограничиваем размерами изображения
        crop_w = max(1, min(crop_w, img_width))
        crop_h = max(1, min(crop_h, img_height))
        
        return crop_w, crop_h

    def _calculate_crop_position(self, img_width: int, img_height: int, 
                                crop_w: int, crop_h: int, 
                                position: str, start_x: int, start_y: int) -> Tuple[int, int]:
        """
        Рассчитывает координаты начала обрезки.
        
        Args:
            img_width: Исходная ширина изображения
            img_height: Исходная высота изображения
            crop_w: Ширина области обрезки
            crop_h: Высота области обрезки
            position: Позиция обрезки
            start_x: Пользовательская X координата
            start_y: Пользовательская Y координата
            
        Returns:
            Tuple[int, int]: (x, y) координаты начала обрезки
        """
        if position == "custom":
            x, y = start_x, start_y
        elif position == "top-left":
            x, y = 0, 0
        elif position == "top-center":
            x = (img_width - crop_w) // 2
            y = 0
        elif position == "top-right":
            x = img_width - crop_w
            y = 0
        elif position == "center-left":
            x = 0
            y = (img_height - crop_h) // 2
        elif position == "center":
            x = (img_width - crop_w) // 2
            y = (img_height - crop_h) // 2
        elif position == "center-right":
            x = img_width - crop_w
            y = (img_height - crop_h) // 2
        elif position == "bottom-left":
            x = 0
            y = img_height - crop_h
        elif position == "bottom-center":
            x = (img_width - crop_w) // 2
            y = img_height - crop_h
        elif position == "bottom-right":
            x = img_width - crop_w
            y = img_height - crop_h
        else:
            # Fallback на центр при неизвестной позиции
            x = (img_width - crop_w) // 2
            y = (img_height - crop_h) // 2
            
        # Ограничиваем координаты допустимыми значениями
        x = max(0, min(x, img_width - crop_w))
        y = max(0, min(y, img_height - crop_h))
        
        return x, y

    def _parse_fill_color(self, color_string: str) -> Tuple[float, float, float]:
        """
        Парсит строку с цветом заполнения.
        
        Args:
            color_string: Строка с цветом ("black", "white", или "R,G,B")
            
        Returns:
            Tuple[float, float, float]: RGB значения в диапазоне 0-1
        """
        color_string = color_string.lower().strip()
        
        if color_string == "black":
            return (0.0, 0.0, 0.0)
        elif color_string == "white":
            return (1.0, 1.0, 1.0)
        else:
            # Пытаемся распарсить RGB значения
            try:
                parts = color_string.split(',')
                if len(parts) == 3:
                    r = float(parts[0].strip()) / 255.0
                    g = float(parts[1].strip()) / 255.0
                    b = float(parts[2].strip()) / 255.0
                    return (r, g, b)
            except:
                pass
            # По умолчанию чёрный
            return (0.0, 0.0, 0.0)

    def _apply_out_of_bounds_fill(self, image: torch.Tensor, x: int, y: int, 
                                  crop_w: int, crop_h: int, fill_color: Tuple[float, float, float]) -> torch.Tensor:
        """
        Заполняет области, выходящие за границы изображения, указанным цветом.
        
        Args:
            image: Исходное изображение [B, H, W, C]
            x: Координата X начала обрезки
            y: Координата Y начала обрезки
            crop_w: Ширина обрезки
            crop_h: Высота обрезки
            fill_color: Цвет заполнения (R, G, B) в диапазоне 0-1
            
        Returns:
            torch.Tensor: Изображение с заполненными областями
        """
        batch_size, img_height, img_width, channels = image.shape
        
        # Создаём тензор с цветом заполнения
        fill_tensor = torch.tensor(fill_color, device=image.device, dtype=image.dtype)
        fill_tensor = fill_tensor.view(1, 1, 1, 3).expand(batch_size, crop_h, crop_w, 3)
        
        # Вычисляем области пересечения
        src_x_start = max(0, x)
        src_x_end = min(img_width, x + crop_w)
        src_y_start = max(0, y)
        src_y_end = min(img_height, y + crop_h)
        
        dst_x_start = max(0, -x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, -y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Копируем исходные пиксели в результирующий тензор
        result = fill_tensor.clone()
        if src_x_start < src_x_end and src_y_start < src_y_end:
            result[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                image[:, src_y_start:src_y_end, src_x_start:src_x_end, :]
        
        return result

    def crop_image(self, image: torch.Tensor, width: int, height: int, 
                   position: str, mask: Optional[torch.Tensor] = None,
                   use_percentage: bool = False, 
                   maintain_aspect_ratio: bool = False, 
                   inverse_crop_percent: int = 0, 
                   start_x: int = 0, start_y: int = 0,
                   device: str = "auto",
                   use_cache: bool = True,
                   fill_color: str = "black",
                   interpolation: str = "bilinear") -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """
        Основной метод обрезки изображения.
        
        Args:
            image: Входное изображение [B, H, W, C]
            width: Ширина обрезки
            height: Высота обрезки
            position: Позиция обрезки
            mask: Опциональная маска [B, H, W]
            use_percentage: Использовать проценты
            maintain_aspect_ratio: Сохранять пропорции
            inverse_crop_percent: Инверсная обрезка (0-49)
            start_x: Начальная X координата
            start_y: Начальная Y координата
            device: Устройство выполнения
            use_cache: Использовать кэширование
            fill_color: Цвет заполнения
            interpolation: Метод интерполяции
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], int, int]: 
            (обрезанное_изображение, обрезанная_маска, ширина, высота)
        """
        # Валидация входных данных
        if image is None:
            raise ValueError("Изображение не может быть пустым / Image cannot be None")
            
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получен {type(image)} / Expected torch.Tensor, got {type(image)}")
            
        if len(image.shape) != 4:
            raise ValueError(f"Ожидается 4D тензор [B,H,W,C], получена форма {image.shape} / Expected 4D tensor [B,H,W,C], got shape {image.shape}")
        
        # Валидация маски, если она предоставлена
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Маска должна быть torch.Tensor, получен {type(mask)} / Mask must be torch.Tensor, got {type(mask)}")
            if len(mask.shape) != 3:
                raise ValueError(f"Ожидается 3D тензор маски [B,H,W], получена форма {mask.shape} / Expected 3D mask tensor [B,H,W], got shape {mask.shape}")
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                raise ValueError(f"Размеры маски {mask.shape} не совпадают с размерами изображения {image.shape} / Mask shape {mask.shape} doesn't match image shape {image.shape}")
        
        # Выбираем устройство
        target_device = self._get_device(device)
        
        # Перемещаем тензоры на выбранное устройство
        original_device = image.device
        image = image.to(target_device)
        if mask is not None:
            mask = mask.to(target_device)
        
        # Проверяем кэш
        cache_key = None
        if use_cache:
            image_hash = self._get_image_hash(image)
            mask_hash = self._get_image_hash(mask) if mask is not None else None
            cache_key = self._generate_cache_key(image_hash, mask_hash, width, height, position,
                                                 use_percentage, maintain_aspect_ratio,
                                                 inverse_crop_percent, start_x, start_y)
            
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                # Возвращаем результат на исходное устройство
                return (cached_result[0].to(original_device), 
                       cached_result[1].to(original_device) if cached_result[1] is not None else None,
                       cached_result[2], cached_result[3])
        
        # Получаем размеры батча и изображения
        batch_size, img_height, img_width, channels = image.shape
        
        # Проверяем минимальные размеры
        if img_width < 1 or img_height < 1:
            raise ValueError(f"Некорректные размеры изображения: {img_width}x{img_height} / Invalid image size: {img_width}x{img_height}")
        
        # Шаг 1: Применяем инверсную обрезку (обрезаем края)
        if inverse_crop_percent > 0:
            image, mask = self._apply_inverse_crop(image, mask, inverse_crop_percent)
            # Обновляем размеры после инверсной обрезки
            _, img_height, img_width, _ = image.shape
        
        # Шаг 2: Рассчитываем размер области обрезки
        crop_w, crop_h = self._calculate_crop_size(
            img_width, img_height, width, height, 
            use_percentage, maintain_aspect_ratio
        )
        
        # Шаг 3: Рассчитываем координаты начала обрезки
        x, y = self._calculate_crop_position(
            img_width, img_height, crop_w, crop_h, 
            position, start_x, start_y
        )
        
        # Шаг 4: Проверяем, не выходит ли обрезка за границы
        out_of_bounds = (x < 0 or y < 0 or x + crop_w > img_width or y + crop_h > img_height)
        
        # Шаг 5: Выполняем обрезку или заполнение
        if out_of_bounds:
            # Парсим цвет заполнения
            fill_rgb = self._parse_fill_color(fill_color)
            # Обрезаем с заполнением
            cropped = self._apply_out_of_bounds_fill(image, x, y, crop_w, crop_h, fill_rgb)
            # Обрезаем маску (без заполнения, просто берём доступную область)
            if mask is not None:
                # Для маски используем чёрный цвет (0) для заполнения
                cropped_mask = torch.zeros((batch_size, crop_h, crop_w), device=target_device, dtype=mask.dtype)
                src_x_start = max(0, x)
                src_x_end = min(img_width, x + crop_w)
                src_y_start = max(0, y)
                src_y_end = min(img_height, y + crop_h)
                dst_x_start = max(0, -x)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                dst_y_start = max(0, -y)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                if src_x_start < src_x_end and src_y_start < src_y_end:
                    cropped_mask[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        mask[:, src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                cropped_mask = None
        else:
            # Обычная обрезка
            cropped = image[:, y:y + crop_h, x:x + crop_w, :]
            if mask is not None:
                cropped_mask = mask[:, y:y + crop_h, x:x + crop_w]
            else:
                cropped_mask = None
        
        # Сохраняем в кэш
        if use_cache and cache_key is not None:
            # Ограничиваем размер кэша
            if len(self._cache) >= self._max_cache_size:
                # Удаляем первый (самый старый) элемент
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[cache_key] = (cropped.clone(), cropped_mask.clone() if cropped_mask is not None else None, crop_w, crop_h)
        
        # Возвращаем результат на исходное устройство
        return (cropped.to(original_device), 
               cropped_mask.to(original_device) if cropped_mask is not None else None,
               crop_w, crop_h)


# Регистрация ноды для ComfyUI
NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "AGSoft Image Crop": AGSoft_Image_Crop
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "AGSoft Image Crop": "✂️AGSoft Image Crop"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
