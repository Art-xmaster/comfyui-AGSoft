import torch

class AGSoft_Image_Crop:
    """
    Узел для обрезки изображения с возможностью выбора позиции обрезки,
    процентной обрезки, сохранения пропорций, инверсной обрезки и 
    задания начальных точек по осям X и Y.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяем входные параметры узла.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Входное изображение
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "position": ([
                    "top-left", "top-center", "top-right",
                    "center-left", "center", "center-right",
                    "bottom-left", "bottom-center", "bottom-right",
                    "custom"
                ], {
                    "default": "center"
                })
            },
            "optional": {
                "use_percentage": ("BOOLEAN", {
                    "default": False,
                    "label_on": "True",
                    "label_off": "False"
                }),
                "maintain_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "label_on": "True",
                    "label_off": "False"
                }),
                "inverse_crop_percent": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 49,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Процент обрезки от краев (0-49%)"
                }),
                "start_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Начальная точка по оси X (игнорируется, если position != custom)"
                }),
                "start_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Начальная точка по оси Y (игнорируется, если position != custom)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_image"
    CATEGORY = "AGSoft/Image"

    def crop_image(self, image, width, height, position, use_percentage=False, 
                   maintain_aspect_ratio=False, inverse_crop_percent=0, 
                   start_x=0, start_y=0):
        """
        Обрезает изображение согласно заданным параметрам.

        Параметры:
        - image: torch.Tensor, форма [B, H, W, C]
        - width: ширина обрезки (в пикселях или процентах)
        - height: высота обрезки (в пикселях или процентах)
        - position: позиция обрезки (например, "center", "custom")
        - use_percentage: если True, width и height интерпретируются как проценты
        - maintain_aspect_ratio: если True, сохраняется соотношение сторон
        - inverse_crop_percent: процент обрезки от краев (инверсная обрезка)
        - start_x: начальная точка по оси X (используется только при position="custom")
        - start_y: начальная точка по оси Y (используется только при position="custom")

        Возвращает:
        - torch.Tensor, обрезанное изображение
        """

        # Получаем размеры оригинального изображения
        batch_size, img_height, img_width, channels = image.shape

        # Применяем инверсную обрезку (обрезаем края)
        if inverse_crop_percent > 0:
            # Вычисляем количество пикселей для обрезки с каждой стороны
            crop_left = int(img_width * (inverse_crop_percent / 100.0))
            crop_right = int(img_width * (inverse_crop_percent / 100.0))
            crop_top = int(img_height * (inverse_crop_percent / 100.0))
            crop_bottom = int(img_height * (inverse_crop_percent / 100.0))
            
            # Убеждаемся, что обрезка не превышает половину изображения
            crop_left = min(crop_left, img_width // 2)
            crop_right = min(crop_right, img_width // 2)
            crop_top = min(crop_top, img_height // 2)
            crop_bottom = min(crop_bottom, img_height // 2)
            
            # Применяем инверсную обрезку
            if crop_left + crop_right < img_width and crop_top + crop_bottom < img_height:
                image = image[:, crop_top:img_height-crop_bottom, crop_left:img_width-crop_right, :]
                # Обновляем размеры после инверсной обрезки
                img_height = image.shape[1]
                img_width = image.shape[2]

        # Если используется процентная обрезка, преобразуем значения
        if use_percentage:
            crop_w = max(1, int(img_width * (width / 100.0)))
            crop_h = max(1, int(img_height * (height / 100.0)))
        else:
            crop_w = min(width, img_width)
            crop_h = min(height, img_height)

        # Если нужно сохранить пропорции, корректируем размеры
        if maintain_aspect_ratio:
            original_aspect = img_width / img_height
            crop_aspect = crop_w / crop_h

            if crop_aspect > original_aspect:
                # Обрезка по ширине ограничена, корректируем высоту
                crop_h = int(crop_w / original_aspect)
            else:
                # Обрезка по высоте ограничена, корректируем ширину
                crop_w = int(crop_h * original_aspect)

            # Убеждаемся, что размеры не превышают оригинальные
            crop_w = min(crop_w, img_width)
            crop_h = min(crop_h, img_height)

        # Ограничиваем размеры обрезки размерами оригинала
        crop_w = max(1, min(crop_w, img_width))
        crop_h = max(1, min(crop_h, img_height))

        # Вычисляем смещения (x, y) в зависимости от позиции
        if position == "custom":
            # Используем пользовательские начальные точки
            x = start_x
            y = start_y
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
            x, y = 0, 0  # по умолчанию

        # Убеждаемся, что координаты не выходят за границы
        x = max(0, min(x, img_width - crop_w))
        y = max(0, min(y, img_height - crop_h))

        # Выполняем обрезку
        cropped = image[:, y:y + crop_h, x:x + crop_w, :]

        return (cropped,)


# Регистрация узла в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Crop": AGSoft_Image_Crop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Crop": "AGSoft Image Crop"
}