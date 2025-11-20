import torch
import numpy as np
from PIL import Image
import comfy.utils
import logging
from scipy import ndimage
from typing import Tuple, Dict, Any, Optional

# Общий цветовой маппинг для обеих нод
COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "olive": (128, 128, 0),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 255),
    "aqua": (0, 255, 255),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),
    "transparent": (0, 0, 0)  # Для прозрачности используем черный, маска будет управлять прозрачностью
}

class AGSoft_Img_Pad:
    """
    Базовая нода для добавления паддинга к изображению с генерацией маски для аутпейнтинга.
    Позволяет добавлять пустое пространство вокруг изображения с различными режимами заполнения
    и создает маску для последующего инпейнтинга (inpainting) или аутпейнтинга (outpainting).
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Определяет входные параметры ноды с подробными подсказками на двух языках
        """
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to add padding to\n"
                               "Входное изображение, к которому добавляется паддинг"
                }),
            },
            "optional": {
                "pad_left": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Pixels to add on the left side\n"
                               "Пикселей для добавления слева"
                }),
                "pad_top": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Pixels to add on the top side\n"
                               "Пикселей для добавления сверху"
                }),
                "pad_right": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Pixels to add on the right side\n"
                               "Пикселей для добавления справа"
                }),
                "pad_bottom": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Pixels to add on the bottom side\n"
                               "Пикселей для добавления снизу"
                }),
                "pad_mode": (["constant", "edge", "reflect", "symmetric"], {
                    "default": "constant",
                    "tooltip": "Padding mode:\n"
                               "constant - fill with background color\n"
                               "edge - replicate edge pixels\n"
                               "reflect - reflect around edge\n"
                               "symmetric - symmetric reflection\n\n"
                               "Режим паддинга:\n"
                               "constant - заполнение фоновым цветом\n"
                               "edge - копирование пикселей с краев\n"
                               "reflect - отражение относительно края\n"
                               "symmetric - симметричное отражение"
                }),
                "background_color": (list(COLOR_MAP.keys()), {
                    "default": "gray",
                    "tooltip": "Background color for constant padding mode\n"
                               "Фоновый цвет для режима constant"
                }),
                "feathering": ("INT", {
                    "default": 40, 
                    "min": 0, 
                    "max": 512, 
                    "step": 1,
                    "tooltip": "Feathering radius for mask edges (soft transition)\n"
                               "Радиус растушевки краев маски (плавный переход)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask: 0=padding area, 1=original image area\n"
                               "Инвертировать маску: 0=область паддинга, 1=оригинальное изображение"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "pad_image"
    CATEGORY = "AGSoft/Image"
    DESCRIPTION = (
        "Adds padding around an image and generates a mask for inpainting/outpainting workflows.\n"
        "By default, mask covers the padding area (1=padding, 0=original image).\n\n"
        "Добавляет паддинг вокруг изображения и генерирует маску для workflows инпейнтинга/аутпейнтинга.\n"
        "По умолчанию, маска покрывает область паддинга (1=паддинг, 0=оригинальное изображение)."
    )
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def pad_image(
        self,
        image: torch.Tensor,
        pad_left: int = 0,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 0,
        pad_mode: str = "constant",
        background_color: str = "gray",
        feathering: int = 40,
        invert_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Основной метод обработки изображения с добавлением паддинга
        
        Args:
            image: Входной тензор изображения
            pad_left, pad_top, pad_right, pad_bottom: Параметры паддинга
            pad_mode: Режим заполнения паддинга
            background_color: Цвет фона
            feathering: Радиус растушевки маски
            invert_mask: Флаг инвертирования маски (меняет назначение маски)
        
        Returns:
            padded_images: Изображение с паддингом
            masks: Маска для inpainting/outpainting
            new_width: Новая ширина изображения
            new_height: Новая высота изображения
        """
        try:
            # Проверка входных данных
            if image is None or image.nelement() == 0:
                raise ValueError("Input image is empty or None")
            
            # Преобразуем тензор изображения в numpy массив
            batch, height, width, channels = image.shape
            image_np = image.cpu().numpy()
            
            # Выбираем цвет фона и нормализуем до [0,1]
            if background_color not in COLOR_MAP:
                self.logger.warning(f"Unknown background color: {background_color}. Using gray.")
                background_color = "gray"
            
            fill_value = np.array(COLOR_MAP[background_color]) / 255.0
            
            # Применяем паддинг к каждому изображению в батче
            padded_images = []
            masks = []
            
            for i in range(batch):
                img = image_np[i].copy()  # Получаем одно изображение из батча
                
                # Паддинг изображения
                if pad_mode == "constant":
                    # Создаем новое изображение с цветом фона
                    new_height = height + pad_top + pad_bottom
                    new_width_total = width + pad_left + pad_right
                    padded_img = np.full((new_height, new_width_total, channels), 
                                       fill_value, dtype=np.float32)
                    # Копируем оригинальное изображение в центр
                    padded_img[pad_top:pad_top + height, pad_left:pad_left + width] = img
                else:
                    # Для других режимов паддинга используем стандартный подход
                    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                      mode=pad_mode)
                
                # Создаем маску: 0 — оригинальное изображение, 1 — паддинг
                mask = np.ones((padded_img.shape[0], padded_img.shape[1]), dtype=np.float32)
                mask[pad_top:pad_top + height, pad_left:pad_left + width] = 0.0
                
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
            new_width = width + pad_left + pad_right
            new_height = height + pad_top + pad_bottom
            
            return (padded_images, masks, new_width, new_height)
        
        except Exception as e:
            self.logger.error(f"Error in pad_image: {str(e)}")
            raise RuntimeError(f"Image padding failed: {str(e)}")
    
    def feather_mask(self, mask: np.ndarray, feathering: int) -> np.ndarray:
        """
        Применяет размытие к краям маски для плавного перехода
        
        Args:
            mask: Входная маска
            feathering: Радиус растушевки
        
        Returns:
            Растушеванная маска
        """
        if feathering <= 0:
            return mask
        
        try:
            # Используем гауссово размытие для создания плавного перехода
            sigma = feathering / 6.0  # 6 — эмпирический коэффициент
            return ndimage.gaussian_filter(mask, sigma=sigma)
        except Exception as e:
            self.logger.warning(f"Feathering failed, returning original mask: {str(e)}")
            return mask


class AGSoft_Img_Pad_Adv(AGSoft_Img_Pad):
    """
    Расширенная нода для добавления паддинга к изображению с дополнительными функциями
    масштабирования и позиционирования. Включает все возможности базовой версии
    плюс контроль над целевыми размерами и сохранением пропорций.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Расширенные входные параметры с дополнительными опциями масштабирования
        """
        base_inputs = super().INPUT_TYPES()
        
        # Добавляем расширенные параметры для масштабирования
        optional_params = base_inputs["optional"]
        optional_params.update({
            "target_width": ("INT", {
                "default": 0, 
                "min": 0, 
                "max": 8192, 
                "step": 1,
                "tooltip": "Target width after padding (0 = disable scaling)\n"
                           "Целевая ширина после паддинга (0 = отключить масштабирование)"
            }),
            "target_height": ("INT", {
                "default": 0, 
                "min": 0, 
                "max": 8192, 
                "step": 1,
                "tooltip": "Target height after padding (0 = disable scaling)\n"
                           "Целевая высота после паддинга (0 = отключить масштабирование)"
            }),
            "keep_proportions": ("BOOLEAN", {
                "default": True,
                "label_on": "Keep",
                "label_off": "Ignore",
                "tooltip": "Maintain original aspect ratio when scaling\n"
                           "Сохранять оригинальные пропорции при масштабировании"
            }),
            "resize_position": (["center", "top-left", "top-right", "bottom-left", "bottom-right"], {
                "default": "center",
                "tooltip": "Position of resized image within target dimensions\n"
                           "Позиция масштабированного изображения внутри целевых размеров"
            }),
        })
        
        return {
            "required": base_inputs["required"],
            "optional": optional_params
        }
    
    DESCRIPTION = (
        "Advanced image padding with scaling options and precise positioning control.\n"
        "Includes all features of basic padding plus target dimensions and aspect ratio preservation.\n"
        "By default, mask covers the padding area (1=padding, 0=original image).\n\n"
        "Расширенный паддинг изображения с опциями масштабирования и точным контролем позиционирования.\n"
        "Включает все функции базового паддинга плюс целевые размеры и сохранение пропорций.\n"
        "По умолчанию, маска покрывает область паддинга (1=паддинг, 0=оригинальное изображение)."
    )
    
    def pad_image(
        self,
        image: torch.Tensor,
        pad_left: int = 0,
        pad_top: int = 0,
        pad_right: int = 0,
        pad_bottom: int = 0,
        target_width: int = 0,
        target_height: int = 0,
        keep_proportions: bool = True,
        resize_position: str = "center",
        pad_mode: str = "constant",
        background_color: str = "gray",
        feathering: int = 40,
        invert_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Расширенная версия метода обработки изображения с поддержкой масштабирования
        """
        try:
            # Проверка входных данных
            if image is None or image.nelement() == 0:
                raise ValueError("Input image is empty or None")
            
            # Преобразуем тензор изображения в numpy массив
            batch, orig_height, orig_width, channels = image.shape
            image_np = image.cpu().numpy()
            
            # Выбираем цвет фона и нормализуем до [0,1]
            if background_color not in COLOR_MAP:
                self.logger.warning(f"Unknown background color: {background_color}. Using gray.")
                background_color = "gray"
            
            fill_value = np.array(COLOR_MAP[background_color]) / 255.0
            
            # Применяем паддинг к каждому изображению в батче
            padded_images = []
            masks = []
            
            for i in range(batch):
                img = image_np[i].copy()  # Получаем одно изображение из батча
                current_height, current_width = orig_height, orig_width
                
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
                    
                    # Рассчитываем паддинг для позиционирования
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
                    
                    # Обновляем текущие размеры
                    current_width, current_height = new_width, new_height
                
                # Паддинг изображения
                if pad_mode == "constant":
                    # Создаем новое изображение с цветом фона
                    new_height_total = current_height + pad_top + pad_bottom
                    new_width_total = current_width + pad_left + pad_right
                    padded_img = np.full((new_height_total, new_width_total, channels), 
                                       fill_value, dtype=np.float32)
                    # Копируем оригинальное изображение в нужную позицию
                    padded_img[pad_top:pad_top + current_height, pad_left:pad_left + current_width] = img
                else:
                    # Для других режимов паддинга используем стандартный подход
                    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                                      mode=pad_mode)
                
                # Создаем маску: 0 — оригинальное изображение, 1 — паддинг
                mask = np.ones((padded_img.shape[0], padded_img.shape[1]), dtype=np.float32)
                mask[pad_top:pad_top + current_height, pad_left:pad_left + current_width] = 0.0
                
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
        
        except Exception as e:
            self.logger.error(f"Error in advanced pad_image: {str(e)}")
            raise RuntimeError(f"Advanced image padding failed: {str(e)}")


# Регистрация нод в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Img_Pad": AGSoft_Img_Pad,
    "AGSoft_Img_Pad_Adv": AGSoft_Img_Pad_Adv
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Img_Pad": "AGSoft Img Pad",
    "AGSoft_Img_Pad_Adv": "AGSoft Img Pad Adv"
}