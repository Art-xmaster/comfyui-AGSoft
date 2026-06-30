# AGSoft_Load_Images_From_Dir.py
# Автор: AGSoft
# Дата: 30 июня 2026 г.
# Описание: Загружает изображения из папки по одному без изменения размера.
"""
AGSoft Load Images From Dir
Load images from folder one by one without resizing.
Загружает изображения из папки по одному без изменения размера.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any


class AGSoftLoadImagesFromDir:
    
    DESCRIPTION = (
        "Load images from directory one by one. Preserves original size and alpha channel.\n"
        "Загружает изображения из папки по одному. Сохраняет размер и альфа-канал."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to folder with images.\n"
                            "Путь к папке с изображениями."
                        )
                    }
                ),
                "filter_type": (
                    ["*.*", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "custom"],
                    {
                        "default": "*.png",
                        "tooltip": (
                            "File extension filter. *.* = all files.\n"
                            "Фильтр по расширению. *.* = все файлы."
                        )
                    }
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": (
                            "First file index (0-based, inclusive).\n"
                            "Индекс первого файла (с 0, включительно)."
                        )
                    }
                ),
                "end_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": (
                            "Last file index (inclusive).\n"
                            "Индекс последнего файла (включительно)."
                        )
                    }
                ),
            },
            "optional": {
                "custom_filter": (
                    "STRING",
                    {
                        "default": "*.png",
                        "multiline": False,
                        "tooltip": (
                            "Custom pattern (used when filter_type='custom').\n"
                            "Свой паттерн (если filter_type='custom')."
                        )
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "filename", "folder_path", "number_of_files")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "load_images"
    CATEGORY = "AGSoft/Image"

    def load_images(
        self,
        folder_path: str,
        filter_type: str,
        start_index: int,
        end_index: int,
        custom_filter: str = "*.png"
    ):
        """
        Load images and return as lists.
        """
        
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Определение паттерна
        pattern = custom_filter if filter_type == "custom" else filter_type
        search_pattern = os.path.join(folder_path, pattern)
        
        # Поиск и сортировка файлов
        file_list = sorted(
            [f for f in glob.glob(search_pattern) if os.path.isfile(f)],
            key=lambda x: os.path.basename(x).lower()
        )

        total_files = len(file_list)
        images_list: List[torch.Tensor] = []
        filenames_list: List[str] = []

        if total_files == 0:
            return (images_list, filenames_list, folder_path, total_files)

        # Корректировка индексов
        start_index = max(0, min(start_index, total_files - 1))
        end_index = max(start_index, min(end_index, total_files - 1))

        # Загрузка изображений
        for idx in range(start_index, end_index + 1):
            filepath = file_list[idx]
            filename = os.path.basename(filepath)
            
            try:
                # Открываем без изменений
                img = Image.open(filepath)
                img_np = np.array(img).astype(np.float32) / 255.0
                
                # Grayscale → RGB
                if img_np.ndim == 2:
                    img_np = np.stack([img_np] * 3, axis=-1)
                
                # [H, W, C] → [1, H, W, C]
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                
                images_list.append(img_tensor)
                filenames_list.append(filename)
                
            except Exception as e:
                print(f"[AGSoft] Error loading {filename}: {e}")

        return (images_list, filenames_list, folder_path, total_files)


NODE_CLASS_MAPPINGS = {
    "AGSoftLoadImagesFromDir": AGSoftLoadImagesFromDir
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadImagesFromDir": "🖼️ AGSoft Load Images From Dir"
}

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']