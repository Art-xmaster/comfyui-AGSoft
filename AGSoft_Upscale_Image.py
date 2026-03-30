"""
AGSoft Upscale Image for ComfyUI
# AGSoft_Upscale_Image.py
# Автор: AGSoft
# Дата: 30 марта 2026 г.

Нода для увеличения изображений с поддержкой моделей апскейла.
Позволяет масштабировать изображения с сохранением пропорций,
изменять размер до конкретных значений, применять суперсемплинг
и округлять размеры для совместимости с другими нодами.
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import os
from typing import Dict, Tuple, List

from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
import comfy.model_management

# ============================================================================
# КОНВЕРТАЦИЯ
# ============================================================================

def tensor2pil(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    image = image.cpu().numpy()
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(image)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# ============================================================================
# СПИСОК МОДЕЛЕЙ
# ============================================================================

def get_upscale_models():
    models = ["None"]
    try:
        model_dir = folder_paths.get_folder_paths("upscale_models")[0]
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(('.pth', '.pt', '.ckpt', '.safetensors')):
                    models.append(f)
    except:
        pass
    return models

# ============================================================================
# НОДА
# ============================================================================

class AGSoft_Upscale_Image:
    """
    Увеличивает разрешение изображений с использованием моделей апскейла.
    Поддерживает три режима работы: масштабирование с коэффициентом,
    изменение по ширине и изменение по высоте с автоматическим сохранением пропорций.
    
    Upscales images using upscale models.
    Supports three operation modes: scaling by factor, resizing by width,
    and resizing by height with automatic aspect ratio preservation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": (
                        "Входное изображение для увеличения.\n"
                        "Может быть одиночным изображением или батчем.\n\n"
                        "Input image to upscale.\n"
                        "Can be a single image or a batch."
                    )
                }),
                "upscale_model": (get_upscale_models(), {
                    "default": "None",
                    "tooltip": (
                        "Модель для увеличения качества изображения.\n"
                        "Модели должны находиться в папке: ComfyUI/models/upscale_models/\n"
                        "Поддерживаются форматы: .pth, .pt, .ckpt, .safetensors\n"
                        "Выберите 'None' если не нужно использовать модель.\n\n"
                        "Model for image quality enhancement.\n"
                        "Models must be placed in: ComfyUI/models/upscale_models/\n"
                        "Supported formats: .pth, .pt, .ckpt, .safetensors\n"
                        "Select 'None' if no model is needed."
                    )
                }),
                "mode": (["rescale", "resize_width", "resize_height"], {
                    "default": "rescale",
                    "tooltip": (
                        "Режим работы:\n"
                        "• rescale - масштабирование с коэффициентом (умножает оба размера)\n"
                        "• resize_width - указать ширину, высота автоматически\n"
                        "• resize_height - указать высоту, ширина автоматически\n\n"
                        "Operation mode:\n"
                        "• rescale - scale by factor (multiplies both dimensions)\n"
                        "• resize_width - specify width, height auto\n"
                        "• resize_height - specify height, width auto"
                    )
                }),
                "rescale_factor": ("FLOAT", {
                    "default": 2.00,
                    "min": 0.50,
                    "max": 10.00,
                    "step": 0.01,
                    "tooltip": (
                        "Коэффициент масштабирования (только для режима rescale).\n"
                        "Примеры:\n"
                        "• 2.0 - увеличит изображение в 2 раза\n"
                        "• 0.5 - уменьшит изображение в 2 раза\n"
                        "• 1.5 - увеличит в 1.5 раза\n\n"
                        "Scale factor (only for rescale mode).\n"
                        "Examples:\n"
                        "• 2.0 - doubles the image size\n"
                        "• 0.5 - halves the image size\n"
                        "• 1.5 - scales by 1.5x"
                    )
                }),
                "resize_value": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": (
                        "Целевой размер (только для режимов resize_width/resize_height).\n"
                        "Для resize_width - это ширина в пикселях.\n"
                        "Для resize_height - это высота в пикселях.\n"
                        "Вторая сторона будет рассчитана автоматически с сохранением пропорций.\n\n"
                        "Target size (only for resize_width/resize_height modes).\n"
                        "For resize_width - width in pixels.\n"
                        "For resize_height - height in pixels.\n"
                        "The other dimension will be calculated automatically maintaining aspect ratio."
                    )
                }),
                "resampling_method": (["lanczos", "bicubic", "bilinear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": (
                        "Метод интерполяции при изменении размера:\n"
                        "• lanczos - наилучшее качество, рекомендуется для увеличения\n"
                        "• bicubic - хорошее качество, быстрее чем lanczos\n"
                        "• bilinear - среднее качество, быстро\n"
                        "• nearest - низкое качество, артефакты, но максимально быстро\n\n"
                        "Interpolation method for resizing:\n"
                        "• lanczos - best quality, recommended for upscaling\n"
                        "• bicubic - good quality, faster than lanczos\n"
                        "• bilinear - medium quality, fast\n"
                        "• nearest - low quality, artifacts, but fastest"
                    )
                }),
                "supersample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Суперсемплинг - улучшает качество при увеличении.\n"
                        "Сначала увеличивает изображение в 2 раза больше, затем уменьшает до нужного размера.\n"
                        "Рекомендуется включать для значительного увеличения.\n\n"
                        "Supersampling - improves quality when upscaling.\n"
                        "First upsamples 2x larger, then downsamples to target size.\n"
                        "Recommended for significant upscaling."
                    )
                }),
                "rounding_modulus": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": (
                        "Округление размеров до кратного значения.\n"
                        "Необходимо для совместимости с VAE и другими нодами.\n"
                        "Стандартное значение 8 подходит для большинства случаев.\n"
                        "Значение 1 отключает округление.\n\n"
                        "Round dimensions to multiples of this value.\n"
                        "Required for compatibility with VAE and other nodes.\n"
                        "Default value 8 works for most cases.\n"
                        "Value 1 disables rounding."
                    )
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": (
                        "Устройство для вычислений:\n"
                        "• auto - автоматический выбор (обычно GPU если доступен)\n"
                        "• cuda - принудительно использовать GPU\n"
                        "• cpu - принудительно использовать CPU\n\n"
                        "Compute device:\n"
                        "• auto - automatic selection (usually GPU if available)\n"
                        "• cuda - force GPU usage\n"
                        "• cpu - force CPU usage"
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "upscale"
    CATEGORY = "AGSoft/Image"
    
    DESCRIPTION = (
        "Увеличивает разрешение изображений с поддержкой моделей апскейла.\n"
        "Поддерживает три режима: масштабирование с коэффициентом, изменение по ширине, изменение по высоте.\n"
        "Автоматически сохраняет пропорции при изменении размера.\n"
        "Возвращает увеличенное изображение, а также его ширину и высоту.\n\n"
        "Upscales images with support for upscale models.\n"
        "Supports three modes: scale by factor, resize by width, resize by height.\n"
        "Automatically preserves aspect ratio when resizing.\n"
        "Returns the upscaled image along with its width and height."
    )
    
    def __init__(self):
        self.model_loader = None
        self.upscaler = None
        self.loaded_model = None
        self.current_model_name = None
    
    def upscale(self, image, upscale_model, mode, rescale_factor, resize_value, 
                resampling_method, supersample, rounding_modulus, device):
        
        # ============================================================
        # НАСТРОЙКА УСТРОЙСТВА
        # ============================================================
        if device == "auto":
            compute_device = comfy.model_management.get_torch_device()
        elif device == "cuda":
            compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            compute_device = torch.device("cpu")
        
        print(f"[AGSoft] Используется устройство: {compute_device}")
        
        # ============================================================
        # ЗАГРУЗКА МОДЕЛИ
        # ============================================================
        model = None
        if upscale_model != "None":
            if self.current_model_name != upscale_model:
                print(f"[AGSoft] Загрузка модели: {upscale_model}")
                self.model_loader = UpscaleModelLoader()
                self.loaded_model = self.model_loader.load_model(upscale_model)[0]
                # Перемещаем модель на выбранное устройство
                if hasattr(self.loaded_model, 'to'):
                    self.loaded_model = self.loaded_model.to(compute_device)
                self.upscaler = ImageUpscaleWithModel()
                self.current_model_name = upscale_model
            model = self.loaded_model
        
        result_images = []
        final_width = 0
        final_height = 0
        
        for i in range(image.shape[0]):
            img = tensor2pil(image[i])
            original_w, original_h = img.size
            print(f"[AGSoft] Исходный размер: {original_w}x{original_h}")
            
            # ============================================================
            # ШАГ 1: ПРИМЕНЯЕМ МОДЕЛЬ (если есть)
            # ============================================================
            if model is not None:
                img_tensor = pil2tensor(img)
                # Перемещаем тензор на выбранное устройство
                img_tensor = img_tensor.to(compute_device)
                img_tensor = self.upscaler.upscale(model, img_tensor)[0]
                img_tensor = img_tensor.cpu()
                img = tensor2pil(img_tensor)
                print(f"[AGSoft] После модели: {img.size}")
            else:
                print(f"[AGSoft] Модель не используется")
            
            # ============================================================
            # ШАГ 2: РАССЧИТЫВАЕМ ЦЕЛЕВОЙ РАЗМЕР
            # ============================================================
            current_w, current_h = img.size
            
            if mode == "rescale":
                # rescale применяется к ИСХОДНОМУ размеру
                target_w = int(original_w * rescale_factor)
                target_h = int(original_h * rescale_factor)
                print(f"[AGSoft] Режим rescale: исходный {original_w}x{original_h} * {rescale_factor} = {target_w}x{target_h}")
                print(f"[AGSoft] Текущий размер после модели: {current_w}x{current_h} → изменим до {target_w}x{target_h}")
                
            elif mode == "resize_width":
                target_w = resize_value
                target_h = int(original_h * (resize_value / original_w))
                print(f"[AGSoft] Режим resize_width: {original_w}x{original_h} → ширина {target_w}, высота {target_h}")
                
            else:  # resize_height
                target_h = resize_value
                target_w = int(original_w * (resize_value / original_h))
                print(f"[AGSoft] Режим resize_height: {original_w}x{original_h} → высота {target_h}, ширина {target_w}")
            
            # ============================================================
            # ШАГ 3: ИЗМЕНЯЕМ РАЗМЕР (если нужно)
            # ============================================================
            if target_w != current_w or target_h != current_h:
                resample_map = {
                    "lanczos": Image.Resampling.LANCZOS,
                    "bicubic": Image.Resampling.BICUBIC,
                    "bilinear": Image.Resampling.BILINEAR,
                    "nearest": Image.Resampling.NEAREST
                }
                resample = resample_map[resampling_method]
                
                if supersample and (target_w > current_w or target_h > current_h):
                    temp_img = img.resize((target_w * 2, target_h * 2), resample)
                    img = temp_img.resize((target_w, target_h), resample)
                    print(f"[AGSoft] Суперсемплинг применен")
                else:
                    img = img.resize((target_w, target_h), resample)
                
                print(f"[AGSoft] После изменения размера: {img.size}")
            else:
                print(f"[AGSoft] Изменение размера не требуется")
            
            # ============================================================
            # ШАГ 4: ОКРУГЛЯЕМ
            # ============================================================
            if rounding_modulus > 1:
                round_w = ((img.width + rounding_modulus - 1) // rounding_modulus) * rounding_modulus
                round_h = ((img.height + rounding_modulus - 1) // rounding_modulus) * rounding_modulus
                if round_w != img.width or round_h != img.height:
                    resample_map = {
                        "lanczos": Image.Resampling.LANCZOS,
                        "bicubic": Image.Resampling.BICUBIC,
                        "bilinear": Image.Resampling.BILINEAR,
                        "nearest": Image.Resampling.NEAREST
                    }
                    resample = resample_map[resampling_method]
                    img = img.resize((round_w, round_h), resample)
                    print(f"[AGSoft] Округлено до {round_w}x{round_h}")
            
            # Сохраняем размеры для выхода (берём размер последнего изображения в батче)
            final_width = img.width
            final_height = img.height
            
            result_images.append(pil2tensor(img))
        
        output = torch.cat(result_images, dim=0)
        print(f"[AGSoft] Итоговый результат: {output.shape[2]}x{output.shape[1]}")
        
        # Возвращаем изображение, ширину и высоту
        return (output, final_width, final_height)


# ============================================================================
# РЕГИСТРАЦИЯ
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "AGSoft Upscale Image": AGSoft_Upscale_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft Upscale Image": "🔍 AGSoft Upscale Image",
}