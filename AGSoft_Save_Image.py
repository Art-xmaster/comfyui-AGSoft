"""
AGSoft Save Image Node для ComfyUI
Автор: AGSoft
Дата: 03.02.2026 г.
"""

import os
import json
import numpy as np
import re
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL.JpegImagePlugin import JpegImageFile
import folder_paths
from collections import defaultdict
import torch


class AGSoftSaveImage:
    """
    Главный класс ноды для сохранения изображений в ComfyUI.
    
    Особенности:
    - Поддержка PNG, JPG, WebP, BMP форматов
    - Раздельные настройки сжатия/качества для каждого формата
    - Гибкая система путей (дата, пользовательские папки)
    - Опция перезаписи существующих файлов
    - Корректное сохранение workflow в JSON (совместим с ComfyUI)
    - Правильная нумерация файлов: всегда 001, 002, 003...
    - Полная поддержка batch processing
    - Детальные подсказки на русском/английском
    """
    
    def __init__(self):
        """Инициализация ноды с настройками по умолчанию."""
        # Получаем дефолтную папку для вывода из ComfyUI
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        
        # Запрещенные символы для Windows
        self.WINDOWS_FORBIDDEN = set('<>:"/\\|?*')
        self.MAX_PATH_LEN = 260
        
        # Поддерживаемые форматы и их параметры по умолчанию
        self.SUPPORTED_FORMATS = {
            "png": {
                "ext": "png", 
                "default_compression": 1,
                "compression_range": (0, 9),
                "compression_label": "Compression Level (0-9)",
                "compression_tooltip": "PNG compression: 0=no compression (fast, large files), 6=good balance, 9=maximum compression (slow, small files)\nСжатие PNG: 0=без сжатия (быстро, большие файлы), 6=хороший баланс, 9=максимальное (медленно, маленькие файлы)"
            },
            "jpg": {
                "ext": "jpg", 
                "default_compression": 90,
                "compression_range": (1, 100),
                "compression_label": "Quality (1-100)",
                "compression_tooltip": "JPEG quality: 1=worst quality (smallest file), 70=good balance, 90=high quality, 100=best quality (largest file)\nКачество JPEG: 1=худшее качество (маленький файл), 70=хороший баланс, 90=высокое качество, 100=лучшее качество (большой файл)"
            },
            "webp": {
                "ext": "webp", 
                "default_compression": 90,
                "compression_range": (1, 100),
                "compression_label": "Quality (1-100)",
                "compression_tooltip": "WebP quality: 1=worst quality, 75=good balance, 90=high quality, 100=best quality (lossless above 90)\nКачество WebP: 1=худшее качество, 75=хороший баланс, 90=высокое качество, 100=лучшее качество (без потерь выше 90)"
            },
            "bmp": {
                "ext": "bmp", 
                "default_compression": 0,
                "compression_range": (0, 0),  # BMP не поддерживает сжатие
                "compression_label": "Not Applicable",
                "compression_tooltip": "BMP format does not support compression\nФормат BMP не поддерживает сжатие"
            }
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Определяет входные параметры ноды для ComfyUI.
        Каждый параметр имеет тип, значение по умолчанию и подсказки.
        """
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image(s) to save\nИзображение(я) для сохранения"
                }),
                "filename_prefix": ("STRING", {
                    "default": "image",
                    "tooltip": "Base filename without extension\nБазовое имя файла без расширения"
                }),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional subdirectory in output folder (e.g., 'my_project')\nОпциональная подпапка в output (например, 'мой_проект')"
                }),
                "create_dated_subfolder": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "Create YYYY-MM-DD subfolder\nСоздать подпапку с датой ГГГГ-ММ-ДД"
                }),
                "image_format": (["png", "jpg", "webp", "bmp"], {
                    "default": "png",
                    "tooltip": "Output image format\nФормат выходного изображения"
                }),
                # Раздельные настройки сжатия для каждого формата
                "png_compression": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 9,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "PNG compression level: 0=no compression, 9=max compression\nУровень сжатия PNG: 0=без сжатия, 9=максимальное"
                }),
                "jpg_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "JPEG quality: 1=worst, 100=best\nКачество JPEG: 1=худшее, 100=лучшее"
                }),
                "webp_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "WebP quality: 1=worst, 100=best\nКачество WebP: 1=худшее, 100=лучшее"
                }),
                "overwrite_existing": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "Overwrite existing files with same name\nПерезаписывать существующие файлы с таким же именем"
                }),
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No",
                    "tooltip": "PNG: save metadata in image (ComfyUI readable)\nJPG/WebP/BMP: save workflow as separate JSON file\nPNG: сохранить метаданные в изображении (читается ComfyUI)\nJPG/WebP/BMP: сохранить workflow как отдельный JSON файл"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "saved_paths", "saved_dir")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "AGSoft/Image"
    
    DESCRIPTION = """
Advanced image saver node with format-specific compression settings, flexible path management, and robust file numbering system.

Продвинутый узел сохранения изображений с настройками сжатия для разных форматов, гибким управлением путями и надежной системой нумерации файлов.
    """
    
    # ==============================
    # ОБРАБОТКА ПУТЕЙ И ИМЕН ФАЙЛОВ
    # ==============================
    
    def process_output_path(self, user_path: str) -> str:
        """
        Обрабатывает пользовательский путь.
        """
        try:
            if not user_path:
                return self.output_dir
            
            try:
                dated_path = datetime.now().strftime(user_path)
            except Exception:
                dated_path = user_path
            
            dated_path = dated_path.replace("/", os.sep).replace("\\", os.sep)
            
            if os.path.isabs(dated_path):
                norm = os.path.normpath(dated_path)
                return norm
            else:
                combined = os.path.normpath(os.path.join(self.output_dir, dated_path))
                return combined
                
        except Exception as e:
            print(f"[AGSoft] Path processing error / Ошибка обработки пути: {str(e)}")
            return None
    
    def get_next_filename(self, directory: str, base_name: str, extension: str, 
                         overwrite: bool = False, is_batch: bool = False, 
                         batch_index: int = None) -> str:
        """
        Генерирует имя файла ВСЕГДА с номером: 001, 002, 003...
        
        Args:
            directory: Папка для сохранения
            base_name: Базовое имя файла
            extension: Расширение файла
            overwrite: Перезаписывать ли существующие файлы
            is_batch: Является ли частью batch
            batch_index: Индекс в batch (если есть)
            
        Returns:
            str: Полный путь к файлу
        """
        try:
            # Если перезапись разрешена
            if overwrite:
                if is_batch and batch_index is not None:
                    # Для batch с перезаписью: имя_001.ext, имя_002.ext
                    filename = f"{base_name}_{batch_index:03d}.{extension}"
                else:
                    # Для одиночного с перезаписью: имя_001.ext
                    filename = f"{base_name}_001.{extension}"
                return os.path.join(directory, filename)
            
            # Если перезапись НЕ разрешена
            if is_batch and batch_index is not None:
                # Для batch: имя_001.ext, имя_002.ext и т.д.
                base_with_index = f"{base_name}_{batch_index:03d}"
                
                # Ищем максимальный номер для этого batch индекса
                pattern = re.compile(rf'^{re.escape(base_with_index)}_(\d+)\.{re.escape(extension)}$')
                max_num = 0
                
                if os.path.exists(directory):
                    for f in os.listdir(directory):
                        if f.endswith(f".{extension}"):
                            match = pattern.match(f)
                            if match:
                                num = int(match.group(1))
                                if num > max_num:
                                    max_num = num
                
                # Если файл с таким batch индексом уже есть, добавляем суффикс
                if max_num > 0:
                    filename = f"{base_with_index}_{max_num+1:03d}.{extension}"
                else:
                    # Первый файл с таким batch индексом
                    filename = f"{base_with_index}_001.{extension}"
                    
                return os.path.join(directory, filename)
            
            # Для одиночных файлов (не batch)
            # Ищем все файлы с таким префиксом и находим максимальный общий номер
            max_num = 0
            pattern = re.compile(rf'^{re.escape(base_name)}_(\d+)\.{re.escape(extension)}$')
            
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    if f.endswith(f".{extension}"):
                        match = pattern.match(f)
                        if match:
                            num = int(match.group(1))
                            if num > max_num:
                                max_num = num
            
            # ВСЕГДА используем номер, начиная с 001
            next_num = max_num + 1
            filename = f"{base_name}_{next_num:03d}.{extension}"
            
            return os.path.join(directory, filename)
            
        except Exception as e:
            print(f"[AGSoft] Filename generation error / Ошибка генерации имени: {str(e)}")
            # Резервное имя
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(directory, f"{base_name}_{timestamp}.{extension}")
    
    # ==============================
    # ОБРАБОТКА WORKFLOW МЕТАДАННЫХ
    # ==============================
    
    def create_comfyui_workflow_json(self, prompt: dict, extra_pnginfo: dict) -> dict:
        """
        Создает JSON workflow в формате ComfyUI.
        Копирует структуру из ComfyUI_temp_*.json файлов.
        """
        try:
            # Берем полную структуру из extra_pnginfo если она есть
            if extra_pnginfo and 'workflow' in extra_pnginfo:
                workflow_data = extra_pnginfo['workflow']
                
                if 'prompt' not in workflow_data:
                    workflow_data['prompt'] = prompt
                
                return workflow_data
            
            # Если нет полного workflow в extra_pnginfo, создаем упрощенный
            workflow_data = {
                "id": str(hash(str(prompt)))[:36],
                "revision": 0,
                "last_node_id": 100,
                "last_link_id": 100,
                "nodes": [],
                "links": [],
                "groups": [],
                "config": {},
                "extra": {
                    "workflowRendererVersion": "LG",
                    "ue_links": [],
                    "ds": {"scale": 1.0, "offset": [0, 0]},
                    "links_added_by_ue": [],
                    "frontendVersion": "1.37.11",
                    "VHS_latentpreview": False,
                    "VHS_latentpreviewrate": 0,
                    "VHS_MetadataImage": True,
                    "VHS_KeepIntermediate": True
                },
                "version": 0.4,
                "prompt": prompt,
                "seed_widgets": {}
            }
            
            return workflow_data
            
        except Exception as e:
            print(f"[AGSoft] Failed to create workflow JSON / Ошибка создания JSON workflow: {str(e)}")
            return {
                "prompt": prompt or {},
                "workflow_info": "Generated by AGSoft Save Image"
            }
    
    def save_workflow_json_file(self, image_path: str, workflow_data: dict) -> str:
        """
        Сохраняет workflow в JSON файл в формате ComfyUI.
        """
        try:
            # Создаем имя для JSON файла (то же имя что у изображения)
            base_name = os.path.splitext(image_path)[0]
            json_path = base_name + ".json"
            
            # Добавляем информацию о сохранении
            if 'extra' not in workflow_data:
                workflow_data['extra'] = {}
            
            workflow_data['extra']['save_timestamp'] = datetime.now().isoformat()
            workflow_data['extra']['image_file'] = os.path.basename(image_path)
            workflow_data['extra']['saved_by'] = "AGSoft Save Image v1.5"
            
            # Сохраняем в файл
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, ensure_ascii=False, indent=2)
            
            print(f"[AGSoft] Workflow JSON saved / JSON workflow сохранен: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"[AGSoft] Failed to save workflow JSON / Не удалось сохранить JSON workflow: {str(e)}")
            return ""
    
    # ==============================
    # СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ
    # ==============================
    
    def save_single_image(self, image_tensor: torch.Tensor, save_dir: str, 
                         base_filename: str, image_format: str, 
                         png_compression: int, jpg_quality: int, webp_quality: int,
                         overwrite: bool, embed_workflow: bool,
                         prompt: dict = None, extra_pnginfo: dict = None,
                         is_batch: bool = False, batch_index: int = None) -> tuple:
        """
        Сохраняет одно изображение.
        Returns: (filepath, json_path)
        """
        json_path = ""
        
        try:
            # Создаем папку если не существует
            os.makedirs(save_dir, exist_ok=True)
            
            # Получаем правильное имя файла (ВСЕГДА с номером)
            extension = self.SUPPORTED_FORMATS[image_format]["ext"]
            filepath = self.get_next_filename(
                save_dir, base_filename, extension, 
                overwrite, is_batch, batch_index
            )
            
            # Конвертируем тензор в PIL Image
            img_array = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            
            if len(img_array.shape) == 4 and img_array.shape[0] == 1:
                img_array = img_array[0]
            
            img = Image.fromarray(img_array)
            
            # Подготовка параметров сохранения
            save_kwargs = {}
            
            if image_format == "png":
                compression_level = min(9, max(0, png_compression))
                save_kwargs["compress_level"] = compression_level
                
                if embed_workflow and prompt:
                    metadata = PngInfo()
                    
                    if prompt:
                        metadata.add_text("prompt", json.dumps(prompt))
                    
                    if extra_pnginfo:
                        for key, value in extra_pnginfo.items():
                            try:
                                metadata.add_text(key, json.dumps(value))
                            except:
                                pass
                    
                    save_kwargs["pnginfo"] = metadata
                    print(f"[AGSoft] PNG metadata embedded / Метаданные PNG встроены")
            
            elif image_format == "jpg":
                quality = min(100, max(1, jpg_quality))
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
                save_kwargs["progressive"] = True
                
                if embed_workflow and (prompt or extra_pnginfo):
                    workflow_data = self.create_comfyui_workflow_json(prompt, extra_pnginfo)
                    json_path = self.save_workflow_json_file(filepath, workflow_data)
            
            elif image_format == "webp":
                quality = min(100, max(1, webp_quality))
                save_kwargs["quality"] = quality
                save_kwargs["method"] = 4
                
                if embed_workflow and (prompt or extra_pnginfo):
                    workflow_data = self.create_comfyui_workflow_json(prompt, extra_pnginfo)
                    json_path = self.save_workflow_json_file(filepath, workflow_data)
            
            elif image_format == "bmp":
                if embed_workflow and (prompt or extra_pnginfo):
                    workflow_data = self.create_comfyui_workflow_json(prompt, extra_pnginfo)
                    json_path = self.save_workflow_json_file(filepath, workflow_data)
            
            # Сохраняем изображение
            img.save(filepath, **save_kwargs)
            
            # Логируем
            basename = os.path.basename(filepath)
            if image_format == "png":
                print(f"[AGSoft] PNG saved: {basename} (compression: {save_kwargs.get('compress_level', 'N/A')}/9)")
            elif image_format == "jpg":
                print(f"[AGSoft] JPG saved: {basename} (quality: {save_kwargs.get('quality', 'N/A')}/100)")
            elif image_format == "webp":
                print(f"[AGSoft] WebP saved: {basename} (quality: {save_kwargs.get('quality', 'N/A')}/100)")
            else:
                print(f"[AGSoft] {image_format.upper()} saved: {basename}")
            
            return filepath, json_path
            
        except Exception as e:
            print(f"[AGSoft] Image save failed / Ошибка сохранения изображения: {str(e)}")
            return "", ""
    
    # ==============================
    # ОСНОВНОЙ МЕТОД COMFYUI
    # ==============================
    
    def save_images(self, images, filename_prefix, output_path, 
                   create_dated_subfolder, image_format, 
                   png_compression, jpg_quality, webp_quality,
                   overwrite_existing, embed_workflow, 
                   prompt=None, extra_pnginfo=None):
        """
        Основной метод сохранения изображений.
        """
        saved_paths = []
        json_paths = []
        
        # Обработка пути сохранения
        try:
            base_save_dir = self.process_output_path(output_path)
            if base_save_dir is None:
                print("[AGSoft] Using default output directory / Использую дефолтную папку")
                base_save_dir = self.output_dir
            
            if create_dated_subfolder:
                date_str = datetime.now().strftime("%Y-%m-%d")
                final_save_dir = os.path.join(base_save_dir, date_str)
            else:
                final_save_dir = base_save_dir
            
            print(f"[AGSoft] Saving to / Сохраняю в: {final_save_dir}")
            
        except Exception as e:
            print(f"[AGSoft] Directory setup failed / Ошибка настройки папки: {str(e)}")
            return (images, "", "")
        
        # Сохраняем каждое изображение в пакете
        batch_size = images.shape[0]
        is_batch = batch_size > 1
        
        for i in range(batch_size):
            batch_index = i + 1 if is_batch else None
            
            saved_path, json_path = self.save_single_image(
                image_tensor=images[i] if is_batch else images,
                save_dir=final_save_dir,
                base_filename=filename_prefix,
                image_format=image_format,
                png_compression=png_compression,
                jpg_quality=jpg_quality,
                webp_quality=webp_quality,
                overwrite=overwrite_existing,
                embed_workflow=embed_workflow,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                is_batch=is_batch,
                batch_index=batch_index
            )
            
            if saved_path:
                saved_paths.append(saved_path)
                if json_path:
                    json_paths.append(json_path)
        
        # Формируем строку с путями - ТОЛЬКО пути к изображениям через запятую
        if saved_paths:
            # Для batch - объединяем все пути через запятую
            paths_string = ",".join(saved_paths)
        else:
            paths_string = ""
        
        return (images, paths_string, final_save_dir)

# ==============================
# РЕГИСТРАЦИЯ НОДЫ В COMFYUI
# ==============================

NODE_CLASS_MAPPINGS = {
    "AGSoftSaveImage": AGSoftSaveImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftSaveImage": "AGSoft Save Image"
}
