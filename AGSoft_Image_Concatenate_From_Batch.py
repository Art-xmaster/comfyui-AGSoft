import torch
import numpy as np
from PIL import Image

class AGSoft_Image_Concatenate_From_Batch:
    """
    Нода для объединения изображений из батча в сетку с заданным количеством колонок
    """
    
    def __init__(self):
        """Инициализация ноды"""
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Определение входных параметров ноды
        """
        return {
            "required": {
                # Входной батч изображений
                "images": ("IMAGE",),
                
                # Количество колонок (1-100)
                "columns": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                
                # Направление заполнения сетки
                "fill_direction": ([
                    "row_by_row",         # По строкам (слева направо, сверху вниз)
                    "column_by_column"    # По колонкам (сверху вниз, слева направо)
                ], {"default": "row_by_row"}),
                
                # Цвет фона холста
                "background_color": ([
                    "black",              # Черный
                    "white",              # Белый
                    "gray"                # Серый
                ], {"default": "black"}),
                
                # Цвет отступов между изображениями
                "gap_color": ([
                    "white",              # Белый (по умолчанию)
                    "black",              # Черный
                    "gray",               # Серый
                    "red",                # Красный
                    "green",              # Зеленый
                    "blue",               # Синий
                    "yellow"              # Желтый
                ], {"default": "white"}),
                
                # Размер отступа между изображениями в пикселях
                "gap": ("INT", {
                    "default": 0,         # По умолчанию без отступа
                    "min": 0,             # Минимальное значение
                    "max": 100,           # Максимальное значение
                    "step": 1             # Шаг изменения
                }),
            }
        }

    # Типы возвращаемых значений
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    # Имена возвращаемых значений
    RETURN_NAMES = ("image", "width", "height")
    # Имя функции для вызова
    FUNCTION = "concatenate_from_batch"
    # Категория ноды в меню ComfyUI
    CATEGORY = "AGSoft/Image"

    def get_background_color_rgb(self, color_name):
        """
        Получение RGB значения для цвета фона
        """
        colors = {
            "black": (0, 0, 0),       # Черный
            "white": (255, 255, 255), # Белый
            "gray": (128, 128, 128)   # Серый
        }
        return colors.get(color_name, (0, 0, 0))

    def get_gap_color_rgb(self, color_name):
        """
        Получение RGB значения для цвета отступов
        """
        colors = {
            "white": (255, 255, 255),  # Белый
            "black": (0, 0, 0),        # Черный
            "gray": (128, 128, 128),   # Серый
            "red": (255, 0, 0),        # Красный
            "green": (0, 255, 0),      # Зеленый
            "blue": (0, 0, 255),       # Синий
            "yellow": (255, 255, 0)    # Желтый
        }
        return colors.get(color_name, (255, 255, 255))

    def tensor_to_pil(self, image_tensor):
        """
        Конвертация тензора ComfyUI в PIL Image
        """
        # Если тензор имеет 4 измерения и первое измерение = 1, убираем его
        if len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        
        # Конвертируем тензор в numpy массив и приводим к диапазону 0-255
        image_np = image_tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def pil_to_tensor(self, pil_image):
        """
        Конвертация PIL Image в тензор ComfyUI
        """
        # Конвертируем PIL изображение в numpy массив и нормализуем к диапазону 0-1
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        # Добавляем измерение батча если его нет
        if len(image_np.shape) == 3:
            image_np = image_np[None,]  # Добавляем размерность батча [1, H, W, C]
        return torch.from_numpy(image_np)

    def concatenate_from_batch(self, images, columns, fill_direction, background_color, gap_color, gap):
        """
        Основная функция объединения изображений из батча в сетку
        """
        
        # Проверяем корректность параметров
        if columns < 1:
            columns = 1
        if columns > 100:
            columns = 100
            
        # Конвертируем батч изображений в список PIL изображений
        pil_images = []
        
        # Обрабатываем каждое изображение в батче
        for i in range(images.shape[0]):
            # Извлекаем одно изображение из батча
            img_tensor = images[i]  # Тензор формы [H, W, C]
            # Добавляем измерение батча для корректной конвертации
            img_tensor_batched = img_tensor.unsqueeze(0)  # [1, H, W, C]
            pil_img = self.tensor_to_pil(img_tensor_batched)
            pil_images.append(pil_img)

        # Если нет изображений в батче, возвращаем пустое изображение
        if len(pil_images) == 0:
            empty_img = Image.new("RGB", (1, 1), self.get_background_color_rgb(background_color))
            result_tensor = self.pil_to_tensor(empty_img)
            return (result_tensor, 1, 1)

        # Определяем максимальные размеры изображений в батче
        max_width = max(img.width for img in pil_images)
        max_height = max(img.height for img in pil_images)

        # Рассчитываем количество строк
        total_images = len(pil_images)
        rows = (total_images + columns - 1) // columns  # Округление вверх

        # Рассчитываем размеры холста
        canvas_width = columns * max_width + (columns - 1) * gap
        canvas_height = rows * max_height + (rows - 1) * gap

        # Создаем холст с выбранным цветом фона
        background_rgb = self.get_background_color_rgb(background_color)
        canvas = Image.new("RGB", (canvas_width, canvas_height), background_rgb)

        # Размещаем изображения в сетке
        for i, img in enumerate(pil_images):
            if fill_direction == "row_by_row":
                # Заполнение по строкам: слева направо, сверху вниз
                row = i // columns
                col = i % columns
            else:
                # Заполнение по колонкам: сверху вниз, слева направо
                col = i // rows
                row = i % rows
                # Проверяем, чтобы не вышли за пределы колонок
                if col >= columns:
                    break

            # Вычисляем позицию для размещения изображения
            x_pos = col * (max_width + gap)
            y_pos = row * (max_height + gap)
            
            # Центрируем изображение в своей ячейке
            x_offset = x_pos + (max_width - img.width) // 2
            y_offset = y_pos + (max_height - img.height) // 2
            
            # Размещаем изображение
            canvas.paste(img, (x_offset, y_offset))

        # Если есть отступы и он больше 0, добавляем цветные линии между ячейками
        if gap > 0:
            gap_rgb = self.get_gap_color_rgb(gap_color)
            
            # Вертикальные линии отступов между колонками
            for col in range(columns - 1):
                x_pos = (col + 1) * max_width + col * gap
                for x in range(gap):
                    for y in range(canvas_height):
                        canvas.putpixel((x_pos + x, y), gap_rgb)
            
            # Горизонтальные линии отступов между строками
            for row in range(rows - 1):
                y_pos = (row + 1) * max_height + row * gap
                for y in range(gap):
                    for x in range(canvas_width):
                        canvas.putpixel((x, y_pos + y), gap_rgb)

        # Конвертируем результат обратно в тензор ComfyUI
        result_tensor = self.pil_to_tensor(canvas)
        width = canvas.width
        height = canvas.height

        return (result_tensor, width, height)


# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Concatenate_From_Batch": AGSoft_Image_Concatenate_From_Batch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Concatenate_From_Batch": "AGSoft Image Concatenate From Batch"
}