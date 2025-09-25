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
IGNORED_MODULES: Set[str] = {'__init__.py'}

# Множество модулей, которые обязательно должны быть загружены
REQUIRED_MODULES: Set[str] = set()

# Имя пакета для отображения в логах
PACKAGE_DISPLAY_NAME = "comfyui-AGSoft"

def load_modules():
    """
    Основная функция загрузки модулей.
    Сканирует текущую директорию, импортирует все Python-файлы
    (кроме игнорируемых) и объединяет их маппинги.
    """
    try:
        current_dir = Path(__file__).parent
        logger.debug(f"Сканируем директорию: {current_dir}")
        
        loaded_modules = []
        failed_modules = []  # Будем хранить кортежи (имя, ошибка)

        for py_file in current_dir.glob('*.py'):
            filename = py_file.name
            if filename in IGNORED_MODULES:
                logger.debug(f"Пропускаем игнорируемый файл: {filename}")
                continue
            
            module_name = py_file.stem
            logger.debug(f"Попытка загрузки модуля: {module_name}")
            
            try:
                module = importlib.import_module(f'.{module_name}', package=__name__)
                _merge_module_mappings(module, module_name)
                loaded_modules.append(module_name)
            except ImportError as e:
                failed_modules.append((module_name, str(e)))
                logger.error(f"Ошибка импорта модуля {module_name}: {e}")
            except Exception as e:
                failed_modules.append((module_name, str(e)))
                logger.error(f"Неожиданная ошибка в модуле {module_name}: {e}")

        # Проверка обязательных модулей
        missing_required = REQUIRED_MODULES - set(loaded_modules)
        if missing_required:
            logger.error(f"Обязательные модули не загружены: {missing_required}")

        # Финальный вывод
        if not failed_modules:
            logger.info(f"Все модули {PACKAGE_DISPLAY_NAME} успешно загружены.")
        else:
            for mod_name, error in failed_modules:
                logger.error(f"Модуль '{mod_name}' не загружен: {error}")
            logger.warning(f"Загружено {len(loaded_modules)} модулей, ошибки в {len(failed_modules)} модулях.")

    except Exception as e:
        logger.critical(f"Критическая ошибка при загрузке модулей: {e}")
        raise

def _merge_module_mappings(module, module_name: str):
    """
    Вспомогательная функция для объединения маппингов из модуля.
    """
    try:
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            new_classes = module.NODE_CLASS_MAPPINGS
            duplicates = set(NODE_CLASS_MAPPINGS.keys()) & set(new_classes.keys())
            if duplicates:
                logger.warning(f"Найдены дубликаты классов в модуле {module_name}: {duplicates}")
            NODE_CLASS_MAPPINGS.update(new_classes)
            logger.debug(f"Добавлены классы из {module_name}: {list(new_classes.keys())}")
        
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            new_names = module.NODE_DISPLAY_NAME_MAPPINGS
            duplicates = set(NODE_DISPLAY_NAME_MAPPINGS.keys()) & set(new_names.keys())
            if duplicates:
                logger.warning(f"Найдены дубликаты имен в модуле {module_name}: {duplicates}")
            NODE_DISPLAY_NAME_MAPPINGS.update(new_names)
            logger.debug(f"Добавлены имена из {module_name}: {list(new_names.keys())}")
            
    except Exception as e:
        logger.error(f"Ошибка при объединении маппингов из модуля {module_name}: {e}")
        raise

# Немедленная загрузка модулей при импорте пакета
try:
    load_modules()
except Exception as e:
    logger.critical(f"Фатальная ошибка при инициализации пакета: {e}")
    pass

# Экспортируем только необходимые переменные
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Дополнительная проверка, что маппинги не пустые
if not NODE_CLASS_MAPPINGS:
    logger.warning("NODE_CLASS_MAPPINGS пуст - возможно, не найдено модулей с узлами")
else:
    logger.debug(f"Загружено {len(NODE_CLASS_MAPPINGS)} классов узлов")

if not NODE_DISPLAY_NAME_MAPPINGS:
    logger.debug("NODE_DISPLAY_NAME_MAPPINGS пуст - будут использованы имена классов по умолчанию")
else:
    logger.debug(f"Загружено {len(NODE_DISPLAY_NAME_MAPPINGS)} отображаемых имен")
