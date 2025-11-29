# AGSoft Styles CSV Loader
# Автор: AGSoft
# Дата: 29 ноября 2025 г.
#

import os
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
import folder_paths

logger = logging.getLogger(__name__)

class AGSoftStylesCSVLoader:
    """
    Нода для загрузки художественных стилей из CSV-файлов.
    Позволяет выбирать файл со стилями и конкретный стиль для генерации изображений.
    
    English: Node for loading artistic styles from CSV files.
    Allows selecting a style file and a specific style for image generation.
    """
    
    # Имя папки со стилями внутри директории ноды
    STYLES_FOLDER = "styles"
    
    @classmethod
    def get_node_dir(cls) -> str:
        """Возвращает директорию, где находится этот файл ноды"""
        return os.path.dirname(os.path.abspath(__file__))
    
    @classmethod
    def get_styles_folder_path(cls) -> str:
        """Возвращает полный путь к папке со стилями внутри директории ноды"""
        return os.path.join(cls.get_node_dir(), cls.STYLES_FOLDER)
    
    @classmethod
    def get_available_style_files(cls) -> List[str]:
        """
        Возвращает список доступных CSV-файлов со стилями.
        
        English: Returns a list of available CSV files with styles.
        """
        styles_folder = cls.get_styles_folder_path()
        
        # Проверяем, существует ли папка styles
        if not os.path.exists(styles_folder):
            os.makedirs(styles_folder, exist_ok=True)
            return []
        
        # Собираем список CSV-файлов
        files = [
            f for f in os.listdir(styles_folder)
            if os.path.isfile(os.path.join(styles_folder, f))
            and f.lower().endswith('.csv')
        ]
        
        # Сортируем файлы для удобства
        return sorted(files)
    
    @classmethod
    def load_styles_csv(cls, styles_path: str) -> Dict[str, List[str]]:
        """
        Загружает CSV файл со стилями.
        Формат CSV: style_name,positive_prompt,negative_prompt
        Игнорирует первую строку (заголовок).
        
        Args:
            styles_path: Путь к CSV файлу со стилями
            
        Returns:
            Словарь стилей. Ключ - название стиля, значение - [positive_prompt, negative_prompt]
        
        English: Loads CSV file with styles.
        CSV format: style_name,positive_prompt,negative_prompt
        Ignores the first line (header).
        """
        styles = {}
        if not os.path.exists(styles_path):
            logger.error(f"Style file not found: {styles_path}")
            return {"Error: File not found": ["", ""]}
        
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    logger.warning(f"Empty file: {styles_path}")
                    return {"Error: Empty file": ["", ""]}
                
                # Пропускаем заголовок и обрабатываем остальные строки
                for i, line in enumerate(lines[1:], 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Разделяем строку с учетом кавычек
                    parts = [
                        part.strip().strip('"') for part in re.split(
                            ',(?=(?:[^"]*"[^"]*")*[^"]*$)', line
                        )
                    ]
                    
                    if len(parts) >= 3:
                        style_name = parts[0]
                        positive_prompt = parts[1]
                        negative_prompt = parts[2]
                        styles[style_name] = [positive_prompt, negative_prompt]
                    else:
                        logger.warning(f"Invalid format in line {i}: {line}")
            
            return styles
        except Exception as e:
            logger.error(f"Error loading styles from {styles_path}: {str(e)}")
            return {"Error loading styles": ["", str(e)]}
    
    @classmethod
    def get_styles_for_file(cls, style_file: str) -> List[str]:
        """
        Возвращает список стилей для указанного файла.
        
        English: Returns a list of styles for the specified file.
        """
        if style_file.startswith("No style files found"):
            return ["No styles available"]
        
        styles_folder = cls.get_styles_folder_path()
        file_path = os.path.join(styles_folder, style_file)
        styles = cls.load_styles_csv(file_path)
        return list(styles.keys()) if styles else ["No styles found"]
    
    @classmethod
    def VALIDATE_INPUTS(cls, style_file, style_name, **kwargs) -> bool:
        """
        Валидация входных данных при загрузке workflow.
        Если выбранный стиль не найден в текущем файле, возвращаем False,
        что заставит ComfyUI перезагрузить ноду с корректными значениями.
        
        English: Validates input data when loading workflow.
        If the selected style is not found in the current file, returns False,
        which will force ComfyUI to reload the node with correct values.
        """
        try:
            # Если это сообщения об ошибках, пропускаем валидацию
            if (style_file.startswith("No style files found") or 
                style_name.startswith("No styles")):
                return True
            
            # Проверяем, существует ли выбранный файл
            styles_folder = cls.get_styles_folder_path()
            file_path = os.path.join(styles_folder, style_file)
            if not os.path.exists(file_path):
                return False
            
            # Загружаем стили из файла
            styles = cls.load_styles_csv(file_path)
            
            # Проверяем, есть ли выбранный стиль в файле
            if style_name not in styles:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error during input validation: {e}")
            return True  # В случае ошибки пропускаем валидацию
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Определяет входные параметры ноды для интерфейса ComfyUI.
        
        English: Defines the input parameters of the node for the ComfyUI interface.
        """
        style_files = cls.get_available_style_files()
        
        # Если нет файлов, показываем сообщение
        if not style_files:
            return {
                "required": {
                    "style_file": (["No style files found. Create styles folder with CSV files"], {
                        "default": "No style files found. Create styles folder with CSV files",
                        "tooltip": "No style files found in the styles folder\n"
                                   "Нет файлов со стилями в папке styles"
                    }),
                    "style_name": (["No styles available"], {
                        "default": "No styles available",
                        "tooltip": "No styles available. Add CSV files to the styles folder\n"
                                   "Нет доступных стилей. Добавьте CSV файлы в папку styles"
                    }),
                },
                "optional": {
                    "reload_files": ("BOOLEAN", {
                        "default": False,
                        "label_on": "Reload Files ✓",
                        "label_off": "Reload Files",
                        "tooltip": "Reload the list of style files from disk\n"
                                   "Перезагрузить список файлов стилей с диска"
                    })
                }
            }
        
        # Загружаем стили из первого файла для начального отображения
        first_file = style_files[0]
        style_names = cls.get_styles_for_file(first_file)
        
        return {
            "required": {
                "style_file": (style_files, {
                    "default": first_file,
                    "tooltip": "Select a CSV file containing styles\n"
                               "Выберите CSV файл, содержащий стили"
                }),
                "style_name": (style_names, {
                    "default": style_names[0] if style_names else "No styles found",
                    "tooltip": "Select a style to apply to your generation\n"
                               "Выберите стиль для применения к генерации"
                }),
            },
            "optional": {
                "reload_files": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Reload Files ✓",
                    "label_off": "Reload Files",
                    "tooltip": "Reload the list of style files from disk\n"
                               "Перезагрузить список файлов стилей с диска"
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive prompt", "negative prompt")
    FUNCTION = "load_style"
    CATEGORY = "AGSoft/nodes"
    
    DESCRIPTION = (
        "Loads artistic styles from CSV files for prompt generation. "
        "Select a style file and a specific style to get positive and negative prompts.\n"
        "Загружает художественные стили из CSV файлов для генерации промптов. "
        "Выберите файл стилей и конкретный стиль для получения позитивного и негативного промптов."
    )
    
    def load_style(
        self, 
        style_file: str, 
        style_name: str,
        reload_files: bool = False,
        unique_id: Optional[str] = None,
        extra_pnginfo: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Загружает выбранный стиль и возвращает позитивный и негативный промпты.
        
        Args:
            style_file: Имя CSV-файла со стилями
            style_name: Название стиля для загрузки
            reload_files: Флаг для перезагрузки списка файлов
            unique_id: Уникальный ID ноды (используется ComfyUI)
            extra_pnginfo: Дополнительная информация (используется ComfyUI)
            
        Returns:
            Кортеж из двух строк: (positive_prompt, negative_prompt)
        
        English: Loads the selected style and returns positive and negative prompts.
        """
        try:
            # Если выбрано сообщение об ошибке, возвращаем пустые промпты
            if style_file.startswith("No style files found") or style_name.startswith("No styles"):
                return ("", "")
            
            # Полный путь к выбранному файлу стилей
            styles_folder = self.get_styles_folder_path()
            file_path = os.path.join(styles_folder, style_file)
            
            # Загрузка стилей из файла
            styles = self.load_styles_csv(file_path)
            
            # Проверка наличия запрошенного стиля
            if style_name in styles:
                positive_prompt, negative_prompt = styles[style_name]
                return (positive_prompt, negative_prompt)
            else:
                # Если запрошенный стиль не найден, возвращаем первый доступный
                if styles:
                    first_style = next(iter(styles))
                    positive_prompt, negative_prompt = styles[first_style]
                    return (positive_prompt, negative_prompt)
                else:
                    return ("", "")
        except Exception as e:
            error_msg = f"Error loading style: {str(e)}"
            logger.error(error_msg)
            return (error_msg, "Error loading negative prompt")

# Серверная часть для обработки API запросов
class AGSoftServer:
    @classmethod
    def setup_routes(cls):
        """Настраивает HTTP endpoints для работы с файлами стилей"""
        try:
            from aiohttp import web
            import server
            
            @server.PromptServer.instance.routes.get("/agsoft_get_style_files")
            async def get_style_files(request):
                """Возвращает список доступных CSV файлов со стилями"""
                try:
                    files = AGSoftStylesCSVLoader.get_available_style_files()
                    return web.json_response(files)
                except Exception as e:
                    logger.error(f"Error getting style files: {e}")
                    return web.json_response({"error": str(e)}, status=500)
            
            @server.PromptServer.instance.routes.post("/agsoft_load_styles")
            async def load_styles(request):
                """Загружает стили из указанного CSV файла"""
                try:
                    data = await request.json()
                    filename = data.get('filename')
                    if not filename:
                        return web.json_response({'error': 'No filename provided'}, status=400)
                    
                    styles_folder = AGSoftStylesCSVLoader.get_styles_folder_path()
                    file_path = os.path.join(styles_folder, filename)
                    styles = AGSoftStylesCSVLoader.load_styles_csv(file_path)
                    
                    return web.json_response({'styles': list(styles.keys())})
                except Exception as e:
                    logger.error(f"Error loading styles: {e}")
                    return web.json_response({'error': str(e)}, status=500)
        except Exception as e:
            logger.error(f"Failed to register server routes: {e}")

# Создаем папку styles при загрузке ноды
try:
    styles_folder = AGSoftStylesCSVLoader.get_styles_folder_path()
    os.makedirs(styles_folder, exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create styles folder: {e}")

# Регистрируем серверные маршруты
try:
    AGSoftServer.setup_routes()
except Exception as e:
    logger.error(f"Failed to setup server routes: {e}")

# Обязательная регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "AGSoft_Styles_CSV_Loader": AGSoftStylesCSVLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Styles_CSV_Loader": "AGSoft Styles CSV Loader"
}

# Указание директории для веб-файлов
WEB_DIRECTORY = "./web"