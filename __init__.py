"""
Динамический загрузчик модулей для ComfyUI.
Автоматически импортирует все Python-файлы из текущей директории
и объединяет их маппинги узлов для системы ComfyUI.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Any, Set
import sys

# Настройка логирования для отладки и мониторинга процесса загрузки
logger = logging.getLogger(__name__)

# Глобальные словари для хранения маппингов классов и отображаемых имен
# Эти переменные будут экспортированы и использованы ComfyUI
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Множество файлов, которые нужно игнорировать при сканировании
# __init__.py всегда игнорируется, так как это текущий файл
IGNORED_MODULES: Set[str] = {'__init__.py'}

# Множество модулей, которые обязательно должны быть загружены
# Если эти модули не загрузятся, будет выдано предупреждение
REQUIRED_MODULES: Set[str] = set()

def load_modules():
    """
    Основная функция загрузки модулей.
    Сканирует текущую директорию, импортирует все Python-файлы
    (кроме игнорируемых) и объединяет их маппинги.
    """
    try:
        # Получаем путь к директории, где находится текущий файл (__init__.py)
        # Path(__file__) - путь к текущему файлу
        # .parent - родительская директория (папка с плагинами)
        current_dir = Path(__file__).parent
        logger.debug(f"Сканируем директорию: {current_dir}")
        
        # Счетчики для статистики загрузки
        loaded_modules = []      # Успешно загруженные модули
        failed_modules = []      # Модули с ошибками
        
        # Используем glob для поиска всех .py файлов в директории
        # Это более надежный способ, чем os.listdir()
        for py_file in current_dir.glob('*.py'):
            # Получаем имя файла (например, "my_node.py")
            filename = py_file.name
            
            # Пропускаем игнорируемые файлы
            if filename in IGNORED_MODULES:
                logger.debug(f"Пропускаем игнорируемый файл: {filename}")
                continue
            
            # Получаем имя модуля без расширения (.py)
            # Например, из "my_node.py" получаем "my_node"
            module_name = py_file.stem
            logger.debug(f"Попытка загрузки модуля: {module_name}")
            
            try:
                # Динамически импортируем модуль
                # f'.{module_name}' - относительный импорт (например, .my_node)
                # package=__name__ - указываем текущий пакет как контекст
                module = importlib.import_module(f'.{module_name}', package=__name__)
                
                # Объединяем маппинги из загруженного модуля
                _merge_module_mappings(module, module_name)
                
                # Добавляем в список успешно загруженных
                loaded_modules.append(module_name)
                logger.info(f"Успешно загружен модуль: {module_name}")
                
            except ImportError as e:
                # Обработка ошибок импорта (например, отсутствующие зависимости)
                failed_modules.append(module_name)
                logger.error(f"Ошибка импорта модуля {module_name}: {e}")
                continue
            except Exception as e:
                # Обработка других ошибок (синтаксические ошибки и т.д.)
                failed_modules.append(module_name)
                logger.error(f"Неожиданная ошибка в модуле {module_name}: {e}")
                continue
        
        # Проверка обязательных модулей
        missing_required = REQUIRED_MODULES - set(loaded_modules)
        if missing_required:
            logger.error(f"Обязательные модули не загружены: {missing_required}")
        
        # Вывод статистики загрузки
        logger.info(f"Загрузка завершена: {len(loaded_modules)} модулей успешно, "
                   f"{len(failed_modules)} с ошибками")
        
        if failed_modules:
            logger.warning(f"Модули с ошибками: {failed_modules}")
            
    except Exception as e:
        # Обработка критических ошибок в самой функции загрузки
        logger.critical(f"Критическая ошибка при загрузке модулей: {e}")
        raise

def _merge_module_mappings(module, module_name: str):
    """
    Вспомогательная функция для объединения маппингов из модуля.
    
    Args:
        module: Загруженный модуль Python
        module_name: Имя модуля для логирования
    """
    try:
        # Проверяем и объединяем NODE_CLASS_MAPPINGS
        # hasattr проверяет наличие атрибута в модуле
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            new_classes = module.NODE_CLASS_MAPPINGS
            
            # Проверка на дубликаты ключей для предотвращения конфликтов
            duplicates = set(NODE_CLASS_MAPPINGS.keys()) & set(new_classes.keys())
            if duplicates:
                logger.warning(f"Найдены дубликаты классов в модуле {module_name}: {duplicates}")
            
            # Обновляем глобальный словарь новыми маппингами
            NODE_CLASS_MAPPINGS.update(new_classes)
            logger.debug(f"Добавлены классы из {module_name}: {list(new_classes.keys())}")
        
        # Проверяем и объединяем NODE_DISPLAY_NAME_MAPPINGS
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            new_names = module.NODE_DISPLAY_NAME_MAPPINGS
            
            # Проверка на дубликаты отображаемых имен
            duplicates = set(NODE_DISPLAY_NAME_MAPPINGS.keys()) & set(new_names.keys())
            if duplicates:
                logger.warning(f"Найдены дубликаты имен в модуле {module_name}: {duplicates}")
            
            # Обновляем глобальный словарь отображаемых имен
            NODE_DISPLAY_NAME_MAPPINGS.update(new_names)
            logger.debug(f"Добавлены имена из {module_name}: {list(new_names.keys())}")
            
    except Exception as e:
        logger.error(f"Ошибка при объединении маппингов из модуля {module_name}: {e}")
        raise

# Немедленная загрузка модулей при импорте пакета
# Это необходимо для ComfyUI, чтобы маппинги были доступны сразу
try:
    load_modules()
except Exception as e:
    logger.critical(f"Фатальная ошибка при инициализации пакета: {e}")
    # Не прерываем выполнение, чтобы не сломать весь ComfyUI
    pass

# Экспортируем только необходимые переменные
# ComfyUI будет использовать именно эти имена
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Дополнительная проверка, что маппинги не пустые
if not NODE_CLASS_MAPPINGS:
    logger.warning("NODE_CLASS_MAPPINGS пуст - возможно, не найдено модулей с узлами")
else:
    logger.info(f"Загружено {len(NODE_CLASS_MAPPINGS)} классов узлов")

if not NODE_DISPLAY_NAME_MAPPINGS:
    logger.info("NODE_DISPLAY_NAME_MAPPINGS пуст - будут использованы имена классов по умолчанию")
else:
    logger.info(f"Загружено {len(NODE_DISPLAY_NAME_MAPPINGS)} отображаемых имен")
