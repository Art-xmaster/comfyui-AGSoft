import node_helpers
import comfy.utils
import math
import torch
import comfy.model_management

# ==============================================================================
# Нода: AGSoftReferenceToLatent
# ==============================================================================
# Назначение:
#   Преобразует одно или несколько эталонных изображений в латентное пространство VAE
#   и добавляет их в conditioning в виде 'reference_latents'. Это позволяет моделям
#   (например, Flux) использовать несколько изображений в качестве визуальных референсов.
# ==============================================================================

class AGSoftReferenceToLatent:
    """
    Класс, реализующий кастомную ноду для ComfyUI.
    Преобразует reference-изображения в латенты и добавляет их в conditioning.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет входные параметры ноды.
        'required' - обязательные параметры, всегда отображаются.
        'optional' - опциональные параметры, могут быть подключены или нет.
        """
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Positive conditioning from CLIP text encoder.\n\n---\n\nПозитивный conditioning от текстового энкодера CLIP."
                }),
                "negative": ("CONDITIONING", {
                    "tooltip": "Negative conditioning from CLIP text encoder.\n\n---\n\nНегативный conditioning от текстового энкодера CLIP."
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE model used to encode images into latent space.\n\n---\n\nVAE модель, используемая для кодирования изображений в латентное пространство."
                }),
                "mode": (["1_image", "2_image", "3_image", "4_image", "5_image", 
                         "6_image", "7_image", "8_image", "9_image", "10_image"], {
                    "tooltip": "Number of reference images to use (1-10). Dynamically changes the number of 'imageX' inputs.\n\n---\n\nКоличество используемых эталонных изображений (1-10). Динамически изменяет количество входов 'imageX'."
                }),
            },
            "optional": {
                "image1_mask": ("MASK", {
                    "tooltip": "Mask for the first image. Determines which areas will be affected during generation. Scaled automatically to latent size.\n\n---\n\nМаска для первого изображения. Определяет, какие области будут затронуты во время генерации. Автоматически масштабируется до размера латентов."
                }),
                "image1": ("IMAGE", {
                    "tooltip": "First reference image (used as base latent).\n\n---\n\nПервое эталонное изображение (используется как базовый латент)."
                }),
                "image2": ("IMAGE", {
                    "tooltip": "Second reference image.\n\n---\n\nВторое эталонное изображение."
                }),
                "image3": ("IMAGE", {
                    "tooltip": "Third reference image.\n\n---\n\nТретье эталонное изображение."
                }),
                "image4": ("IMAGE", {
                    "tooltip": "Fourth reference image.\n\n---\n\nЧетвертое эталонное изображение."
                }),
                "image5": ("IMAGE", {
                    "tooltip": "Fifth reference image.\n\n---\n\nПятое эталонное изображение."
                }),
                "image6": ("IMAGE", {
                    "tooltip": "Sixth reference image.\n\n---\n\nШестое эталонное изображение."
                }),
                "image7": ("IMAGE", {
                    "tooltip": "Seventh reference image.\n\n---\n\nСедьмое эталонное изображение."
                }),
                "image8": ("IMAGE", {
                    "tooltip": "Eighth reference image.\n\n---\n\nВосьмое эталонное изображение."
                }),
                "image9": ("IMAGE", {
                    "tooltip": "Ninth reference image.\n\n---\n\nДевятое эталонное изображение."
                }),
                "image10": ("IMAGE", {
                    "tooltip": "Tenth reference image.\n\n---\n\nДесятое эталонное изображение."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "MASK", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "mask", "image")
    
    # Всплывающие подсказки для выходных слотов (RETURN_TOOLTIPS)
    RETURN_TOOLTIPS = (
        "Positive conditioning with reference_latents added.\n\n---\n\nПозитивный conditioning с добавленными reference_latents.",
        "Negative conditioning with reference_latents added.\n\n---\n\nНегативный conditioning с добавленными reference_latents.",
        "Latent representation of the first reference image (size: width/8 × height/8). Can be used as initial noise.\n\n---\n\nЛатентное представление первого эталонного изображения (размер: ширина/8 × высота/8). Можно использовать как начальный шум.",
        "Mask scaled to latent space dimensions (width/8 × height/8). Useful for inpainting.\n\n---\n\nМаска, масштабированная до размеров латентного пространства (ширина/8 × высота/8). Полезна для инпайтинга.",
        "First reference image (unchanged). Useful for inpainting preview.\n\n---\n\nПервое эталонное изображение (без изменений). Полезно для предпросмотра инпайтинга."
    )
    
    FUNCTION = "process"
    CATEGORY = "AGSoft/nodes"
    
    # Описание ноды, отображаемое в интерфейсе ComfyUI
    DESCRIPTION = """Converts reference images to latent space and adds them as reference_latents to conditioning.

**English:**
- Converts 1-10 reference images to latents using a VAE.
- Adds `reference_latents` to positive/negative conditioning for models like Flux.
- Outputs the latent of the first image, a scaled mask, and the original image.

**Русский:**
- Преобразует 1-10 эталонных изображений в латенты с помощью VAE.
- Добавляет `reference_latents` в позитивный/негативный conditioning для моделей типа Flux.
- Возвращает латент первого изображения, масштабированную маску и оригинал.
"""

    def process(self, positive, negative, vae, mode, image1_mask=None, 
               image1=None, image2=None, image3=None, image4=None, 
               image5=None, image6=None, image7=None, image8=None, 
               image9=None, image10=None):
        """
        Основной метод обработки ноды.
        Выполняет кодирование изображений, создание референс-латенти и обновление conditioning.
        """
        # --- Проверка наличия VAE ---
        if vae is None:
            raise RuntimeError("VAE is required. Please connect a VAE loader.")
        
        # --- Сбор активных изображений на основе режима (mode) ---
        all_images = [image1, image2, image3, image4, image5, 
                      image6, image7, image8, image9, image10]
        # Извлекаем число из строки вида "X_image"
        count = int(mode.split("_")[0])
        # Фильтруем None (неподключенные входы)
        images = [img for i, img in enumerate(all_images[:count]) if img is not None]
        
        if not images:
            raise RuntimeError("At least one reference image is required")
        
        # --- Сохраняем первое изображение для выхода ---
        first_image = images[0]
        
        # --- Получение размеров и подготовка к изменению размера ---
        # VAE требует, чтобы размеры изображения были кратны downscale_factor (обычно 8)
        original_height, original_width = first_image.shape[1], first_image.shape[2]
        downscale_factor = getattr(vae, 'downscale_factor', 8)
        
        # В современных VAE (SDXL, Flux) требуется кратность 16, но оставим логику как в оригинале
        valid_multiple = downscale_factor * 8
        
        def round_to_valid(x):
            """Округляет размер до ближайшего кратного valid_multiple вверх"""
            return ((x + valid_multiple - 1) // valid_multiple) * valid_multiple
        
        target_width = round_to_valid(original_width)
        target_height = round_to_valid(original_height)
        
        # Логирование для отладки (будет видно в консоли ComfyUI)
        print(f"[AGSoft Reference to Latent] Original image size: {original_width}×{original_height}")
        print(f"[AGSoft Reference to Latent] Rounded to VAE-valid size: {target_width}×{target_height}")
        print(f"[AGSoft Reference to Latent] Latent will be: {target_width // downscale_factor}×{target_height // downscale_factor}")
        
        # --- Основной цикл обработки всех эталонных изображений ---
        ref_latents = []      # Список латентов для conditioning
        vl_images = []        # Список уменьшенных копий для CLIP (не используются, но оставлены из оригинала)
        
        for idx, image in enumerate(images):
            # Меняем порядок осей из (B, H, W, C) в (B, C, H, W) для удобства обработки
            samples = image.movedim(-1, 1)
            
            # --- Создание миниатюры для CLIP vision (384x384) ---
            # Это нужно для совместимости с некоторыми моделями, но в текущей логике не используется
            current_total = samples.shape[3] * samples.shape[2]
            vl_total = int(384 * 384)
            vl_scale_by = math.sqrt(vl_total / current_total)
            vl_width = round(samples.shape[3] * vl_scale_by)
            vl_height = round(samples.shape[2] * vl_scale_by)
            
            s_vl = comfy.utils.common_upscale(samples, vl_width, vl_height, "area", "center")
            vl_image = s_vl.movedim(1, -1)  # обратно в (B, H, W, C)
            vl_images.append(vl_image)
            
            # --- Создание референс-латента для conditioning ---
            # Изменяем размер изображения до корректного для VAE
            vae_input = comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "center")
            vae_input = vae_input.movedim(1, -1)  # обратно в (B, H, W, C)
            # Кодируем в латентное пространство
            ref_latent = vae.encode(vae_input[:, :target_height, :target_width, :])
            ref_latents.append(ref_latent)
            
            print(f"[AGSoft Reference to Latent] Image {idx+1}: VAE input shape = {vae_input.shape}, latent shape = {ref_latent.shape}")
        
        # --- Добавление reference_latents к conditioning ---
        # Используем стандартную утилиту ComfyUI для безопасного добавления данных в conditioning
        positive_out = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
        negative_out = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        
        # --- Создание основного латента из первого изображения ---
        first_samples = first_image.movedim(-1, 1)
        vae_input = comfy.utils.common_upscale(first_samples, target_width, target_height, "lanczos", "center")
        vae_input = vae_input.movedim(1, -1)
        base_latent = vae.encode(vae_input[:, :target_height, :target_width, :])
        
        # Упаковываем в формат, ожидаемый другими нодами (например, KSampler)
        latent = {"samples": base_latent}
        
        # --- Обработка маски (если подана) ---
        noise_mask = None
        if image1_mask is not None:
            mask = image1_mask
            # Приводим маску к формату (B, 1, H, W) для масштабирования
            if mask.dim() == 2:
                mask_samples = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask_samples = mask.unsqueeze(1)
            else:
                mask_samples = mask.unsqueeze(0) if mask.dim() == 4 else mask
            
            if mask_samples is not None:
                # Вычисляем целевой размер латента
                latent_width = target_width // downscale_factor
                latent_height = target_height // downscale_factor
                # Масштабируем маску до размера латента
                noise_mask = comfy.utils.common_upscale(mask_samples, latent_width, latent_height, "area", "center")
                noise_mask = noise_mask.squeeze(1)  # Убираем лишнюю размерность канала
                latent["noise_mask"] = noise_mask
                
                print(f"[AGSoft Reference to Latent] Mask scaled to latent size: {latent_width}×{latent_height}")
        
        print(f"[AGSoft Reference to Latent] Final latent size: {base_latent.shape}")
        
        # Возвращаем результат: conditioning, латент, маску и первое изображение
        return (positive_out, negative_out, latent, noise_mask, first_image)


# --- Регистрация ноды для ComfyUI ---
# ComfyUI автоматически ищет эти словари в глобальной области видимости файлов в custom_nodes/
NODE_CLASS_MAPPINGS = {
    "AGSoftReferenceToLatent": AGSoftReferenceToLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftReferenceToLatent": "AGSoft Reference to Latent",
}