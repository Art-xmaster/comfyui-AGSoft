# ==============================================================================
# AGSoft_Load_Images_From_Dir.py
# ==============================================================================
# Нода: 🖼️AGSoft Load Images From Dir
#
# Описание / Description:
# Загружает изображения из папки по одному без изменения размера, сохраняя
# альфа-канал. 
# Loads images from a folder one by one without resizing, preserving the alpha
# channel.
#
# Возможности / Features:
# ⚡ Загрузка без изменений: сохраняются оригинальный размер и альфа-канал.
#    Load without changes: original size and alpha channel are preserved.
# ⚡ Фильтр расширений: один или несколько паттернов сразу в custom
#    (например '*.png, *.jpg' или 'png jpg'), с дедупликацией.
#    Extension filter: one or several patterns at once in custom
#    (e.g. '*.png, *.jpg' or 'png jpg'), with deduplication.
# ⚡ Рекурсия по подпапкам (**) — опционально.
#    Optional recursive search across all subfolders (**).
# ⚡ Порядок сортировки: natural / обратный / по дате файла (новые или старые сначала).
#    Sort order: natural / reversed / by file date (newest or oldest first).
# ⚡ Диапазон start_index / end_index (0 в конце = «до конца списка»)
#    и лимит количества max_images (0 = все).
#    start_index / end_index range (0 at end = "to the end of the list")
#    and a max_images count limit (0 = all).
# ⚡ Защита от пустой выборки: вместо падения ComfyUI нода выдаёт понятную
#    ошибку с путём, паттернами и режимом рекурсии.
#    Empty-selection guard: instead of crashing ComfyUI, the node raises a clear
#    error showing the path, patterns and recursion mode.
#
# Автор / Author: AGSoft
# Дата / Date: 03.08.2026
# ==============================================================================

import os
import re
import glob
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any


def _natural_key(s):
    """Ключ natural sort: img_2 встанет перед img_10. / Natural sort key."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _parse_filter_patterns(custom_filter):
    """Разбирает строку с несколькими паттернами (через запятую/точку с запятой/пробел).
    Токен без '*' и '.' трактуется как расширение: png -> *.png.
    Parses a string with multiple patterns (comma/semicolon/space separated).
    A token without '*' and '.' is treated as an extension: png -> *.png."""
    tokens = re.split(r"[,;\s]+", str(custom_filter or "").strip())
    patterns = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "*" not in tok and "." not in tok:
            tok = f"*.{tok}"
        patterns.append(tok)
    return patterns if patterns else ["*.*"]


class AGSoftLoadImagesFromDir:
    DESCRIPTION = (
        "Load images from a folder one by one without resizing, preserving the alpha channel.\n"
        "Supports multiple extension patterns (custom), optional recursive subfolder search, "
        "sort order choice (natural / reversed / by file date), a start/end range "
        "(0 at end = to the end of the list) and a max_images count limit.\n"
        "If nothing matches the filter, the node raises a clear error instead of crashing ComfyUI.\n"
        "---\n"
        "Загружает изображения из папки по одному без изменения размера, сохраняя альфа-канал.\n"
        "Поддерживает несколько паттернов расширений (custom), опциональную рекурсию по "
        "подпапкам, выбор порядка сортировки (natural / обратный / по дате файла), диапазон "
        "start/end (0 в конце = до конца списка) и лимит количества max_images.\n"
        "Если по фильтру ничего не найдено, нода выдаёт понятную ошибку вместо падения ComfyUI."
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
                            "Path to the folder with images.\n"
                            "---\n"
                            "Путь к папке с изображениями."
                        )
                    }
                ),
                "filter_type": (
                    ["*.*", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.gif", "custom"],
                    {
                        "default": "custom",
                        "tooltip": (
                            "File extension filter. *.* = all files. custom = use custom_filter.\n"
                            "---\n"
                            "Фильтр по расширению. *.* = все файлы. custom = использовать custom_filter."
                        )
                    }
                ),
                "custom_filter": (
                    "STRING",
                    {
                        "default": "*.png, *.jpg",
                        "tooltip": (
                            "Used when filter_type='custom'. One or more patterns separated by comma/semicolon/space.\n"
                            "Examples: '*.png, *.jpg' or 'png jpg' or '*.webp'. A bare extension (png) becomes *.png.\n"
                            "---\n"
                            "Используется при filter_type='custom'. Один или несколько паттернов через запятую/точку с запятой/пробел.\n"
                            "Примеры: '*.png, *.jpg' или 'png jpg' или '*.webp'. Расширение без точки (png) превращается в *.png."
                        )
                    }
                ),
                "recursive": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": (
                            "on = search images recursively in all nested subfolders (**).\n"
                            "off = only the selected folder.\n"
                            "---\n"
                            "on = искать изображения рекурсивно во всех вложенных подпапках (**).\n"
                            "off = только в выбранной папке."
                        )
                    }
                ),
                "sort_order": (
                    ["natural", "reversed", "date_newest", "date_oldest"],
                    {
                        "default": "natural",
                        "tooltip": (
                            "File ordering before the range is applied.\n"
                            "natural = natural sort by name (img_2 before img_10).\n"
                            "reversed = reverse of natural.\n"
                            "date_newest = by file modification date, newest first.\n"
                            "date_oldest = by file modification date, oldest first.\n"
                            "---\n"
                            "Порядок файлов перед применением диапазона.\n"
                            "natural = естественная сортировка по имени (img_2 перед img_10).\n"
                            "reversed = обратный порядок.\n"
                            "date_newest = по дате изменения файла, сначала новые.\n"
                            "date_oldest = по дате изменения файла, сначала старые."
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
                            "First file index (0-based, inclusive) in the sorted list.\n"
                            "---\n"
                            "Индекс первого файла (с 0, включительно) в отсортированном списке."
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
                            "Last file index (inclusive). 0 = to the end of the list.\n"
                            "---\n"
                            "Индекс последнего файла (включительно). 0 = до конца списка."
                        )
                    }
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": (
                            "Maximum number of images to load (from the selected range). 0 = all.\n"
                            "---\n"
                            "Максимум изображений для загрузки (из выбранного диапазона). 0 = все."
                        )
                    }
                ),
            },
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
        custom_filter: str,
        recursive: str,
        sort_order: str,
        start_index: int,
        end_index: int,
        max_images: int,
    ):
        """
        Load images and return as lists.
        """
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        # Паттерны: custom поддерживает несколько, остальные — один.
        # Patterns: custom supports several, the others - one.
        filter_type = str(filter_type or "*.png").strip()
        if filter_type.lower() == "custom":
            patterns = _parse_filter_patterns(custom_filter)
        else:
            patterns = [filter_type if filter_type else "*.*"]

        is_recursive = str(recursive or "off").strip().lower() == "on"
        sort_order = str(sort_order or "natural").strip().lower()

        # Сбор файлов по всем паттернам с дедупликацией (и рекурсией, если включена).
        # Collect files for all patterns, deduplicated (and recursive if enabled).
        seen = {}
        for pat in patterns:
            if is_recursive:
                found = glob.glob(os.path.join(folder_path, "**", pat), recursive=True)
            else:
                found = glob.glob(os.path.join(folder_path, pat))
            for f in found:
                if os.path.isfile(f):
                    key = os.path.normcase(os.path.abspath(f))
                    if key not in seen:
                        seen[key] = f

        # Сортировка согласно выбранному порядку. / Sort according to the chosen order.
        if sort_order == "reversed":
            file_list = sorted(
                seen.values(), key=lambda x: _natural_key(os.path.basename(x)), reverse=True
            )
        elif sort_order == "date_newest":
            file_list = sorted(seen.values(), key=lambda x: os.path.getmtime(x), reverse=True)
        elif sort_order == "date_oldest":
            file_list = sorted(seen.values(), key=lambda x: os.path.getmtime(x))
        else:  # natural
            file_list = sorted(seen.values(), key=lambda x: _natural_key(os.path.basename(x)))

        total_files = len(file_list)
        images_list: List[torch.Tensor] = []
        filenames_list: List[str] = []

        # ВАЖНО: пустые списки возвращать нельзя — ComfyUI (OUTPUT_IS_LIST) падает
        # с IndexError при нарезке пустого списка для следующих нод.
        # Вместо этого выдаём понятную ошибку. / IMPORTANT: never return empty lists -
        # ComfyUI (OUTPUT_IS_LIST) crashes with IndexError when slicing an empty list
        # for downstream nodes. Raise a clear error instead.
        if total_files == 0:
            raise ValueError(
                f"[AGSoft Load Images From Dir] В папке не найдено файлов по паттерну(ам): "
                f"{', '.join(patterns)} (recursive={'on' if is_recursive else 'off'}). "
                f"Папка: {folder_path}. Проверьте путь, фильтр и содержимое папки."
            )

        # Корректировка диапазона. end_index <= 0 = "до конца списка".
        # Range adjustment. end_index <= 0 = "to the end of the list".
        start = max(0, min(int(start_index), total_files - 1))
        if int(end_index) <= 0:
            end = total_files - 1
        else:
            end = max(start, min(int(end_index), total_files - 1))

        selected = file_list[start:end + 1]

        # Лимит количества. 0 = все. / Count limit. 0 = all.
        if int(max_images) > 0:
            selected = selected[:int(max_images)]

        # Загрузка изображений без изменений (размер/альфа как в оригинале).
        # Load images without changes (size/alpha as in the original).
        for filepath in selected:
            filename = os.path.basename(filepath)
            try:
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

        # Защита: файлы нашлись, но ни один не удалось прочитать.
        # Guard: files were found but none could be read.
        if not images_list:
            raise ValueError(
                f"[AGSoft Load Images From Dir] Не удалось загрузить ни одного изображения "
                f"(найдено файлов: {total_files}, но все повреждены или нечитаемы). Папка: {folder_path}."
            )

        print(
            f"[AGSoft Load Images From Dir] Паттерны: {', '.join(patterns)} | "
            f"recursive: {'on' if is_recursive else 'off'} | sort: {sort_order} | "
            f"найдено файлов: {total_files} | загружено: {len(images_list)} "
            f"(индексы {start}..{end}, лимит {int(max_images)})."
        )

        return (images_list, filenames_list, folder_path, total_files)


NODE_CLASS_MAPPINGS = {
    "AGSoftLoadImagesFromDir": AGSoftLoadImagesFromDir
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftLoadImagesFromDir": "🖼️ AGSoft Load Images From Dir"
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']