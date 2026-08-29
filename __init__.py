"""
Dynamic module loader for ComfyUI.
Automatically imports all Python files from the current directory
and merges their node mappings for the ComfyUI system.

Динамический загрузчик модулей для ComfyUI.
Автоматически импортирует все Python-файлы из текущей директории
и объединяет их маппинги узлов для системы ComfyUI.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Set

# Configure logging for debugging and monitoring the loading process.
# Настройка логирования для отладки и мониторинга процесса загрузки.
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# Global dictionaries for storing class mappings and display names.
# These variables will be exported and used by ComfyUI.
#
# Глобальные словари для хранения маппингов классов и отображаемых имён.
# Эти переменные будут экспортированы и использованы ComfyUI.
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Set of files to ignore during scanning.
# Множество файлов, которые нужно игнорировать при сканировании.
IGNORED_MODULES: Set[str] = {"__init__.py"}

# Set of modules that must be loaded.
# Множество модулей, которые обязательно должны быть загружены.
REQUIRED_MODULES: Set[str] = set()

# Package name to display in logs.
# Имя пакета для отображения в логах.
PACKAGE_DISPLAY_NAME = "comfyui-AGSoft"

# Web directory for ComfyUI frontend extensions.
# Веб-директория для фронтенд-расширений ComfyUI.
WEB_DIRECTORY = "./web"


def _bilingual(english: str, russian: str) -> str:
    """
    Return a bilingual message: English first, then Russian.

    Возвращает двуязычное сообщение: сначала английский, затем русский.
    """
    return f"{english} / {russian}"


def load_modules() -> None:
    """
    Main module loading function.
    Scans the current directory, imports all Python files except ignored ones,
    and merges their mappings.

    Основная функция загрузки модулей.
    Сканирует текущую директорию, импортирует все Python-файлы, кроме игнорируемых,
    и объединяет их маппинги.
    """
    try:
        current_dir = Path(__file__).resolve().parent

        logger.debug(
            _bilingual(
                f"Scanning directory: {current_dir}",
                f"Сканируем директорию: {current_dir}",
            )
        )

        loaded_modules = []
        failed_modules = []
        # Stores tuples: (module name, error).
        # Хранит кортежи: (имя модуля, ошибка).

        for py_file in sorted(current_dir.glob("*.py")):
            filename = py_file.name

            if filename in IGNORED_MODULES:
                logger.debug(
                    _bilingual(
                        f"Skipping ignored file: {filename}",
                        f"Пропускаем игнорируемый файл: {filename}",
                    )
                )
                continue

            module_name = py_file.stem

            # Skip files whose names cannot be used as Python module names.
            # Пропускаем файлы, имена которых нельзя использовать как имена Python-модулей.
            if not module_name.isidentifier():
                failed_modules.append(
                    (
                        module_name,
                        "Invalid Python module name / Неверное имя Python-модуля",
                    )
                )

                logger.error(
                    _bilingual(
                        f"Module name '{module_name}' is not a valid Python identifier",
                        f"Имя модуля '{module_name}' не является допустимым Python-идентификатором",
                    )
                )
                continue

            logger.debug(
                _bilingual(
                    f"Attempting to load module: {module_name}",
                    f"Попытка загрузки модуля: {module_name}",
                )
            )

            try:
                module = importlib.import_module(
                    f".{module_name}",
                    package=__package__ or __name__,
                )

                _merge_module_mappings(module, module_name)
                loaded_modules.append(module_name)

            except ImportError as e:
                failed_modules.append((module_name, str(e)))

                logger.error(
                    _bilingual(
                        f"Import error in module '{module_name}': {e}",
                        f"Ошибка импорта модуля {module_name}: {e}",
                    )
                )

            except Exception as e:
                failed_modules.append((module_name, str(e)))

                logger.error(
                    _bilingual(
                        f"Unexpected error in module '{module_name}': {e}",
                        f"Неожиданная ошибка в модуле {module_name}: {e}",
                    )
                )

        # Check required modules.
        # Проверка обязательных модулей.
        missing_required = REQUIRED_MODULES - set(loaded_modules)

        if missing_required:
            logger.error(
                _bilingual(
                    f"Required modules were not loaded: {sorted(missing_required)}",
                    f"Обязательные модули не загружены: {sorted(missing_required)}",
                )
            )

        # Final report.
        # Финальный отчёт.
        if not failed_modules:
            logger.info(
                _bilingual(
                    f"All {PACKAGE_DISPLAY_NAME} modules loaded successfully.",
                    f"Все модули {PACKAGE_DISPLAY_NAME} успешно загружены.",
                )
            )
        else:
            for mod_name, error in failed_modules:
                logger.error(
                    _bilingual(
                        f"Module '{mod_name}' was not loaded: {error}",
                        f"Модуль '{mod_name}' не загружен: {error}",
                    )
                )

            logger.warning(
                _bilingual(
                    f"Loaded {len(loaded_modules)} modules, errors in {len(failed_modules)} modules.",
                    f"Загружено {len(loaded_modules)} модулей, ошибки в {len(failed_modules)} модулях.",
                )
            )

    except Exception as e:
        logger.critical(
            _bilingual(
                f"Critical error while loading modules: {e}",
                f"Критическая ошибка при загрузке модулей: {e}",
            )
        )
        raise


def _merge_module_mappings(module: Any, module_name: str) -> None:
    """
    Helper function for merging mappings from a module.

    Вспомогательная функция для объединения маппингов из модуля.
    """
    try:
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            new_classes = module.NODE_CLASS_MAPPINGS

            duplicates = set(NODE_CLASS_MAPPINGS.keys()) & set(new_classes.keys())

            if duplicates:
                logger.warning(
                    _bilingual(
                        f"Duplicate node classes found in module '{module_name}': {sorted(duplicates)}",
                        f"Найдены дубликаты классов узлов в модуле {module_name}: {sorted(duplicates)}",
                    )
                )

            NODE_CLASS_MAPPINGS.update(new_classes)

            logger.debug(
                _bilingual(
                    f"Classes added from module '{module_name}': {list(new_classes.keys())}",
                    f"Добавлены классы из {module_name}: {list(new_classes.keys())}",
                )
            )

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            new_names = module.NODE_DISPLAY_NAME_MAPPINGS

            duplicates = set(NODE_DISPLAY_NAME_MAPPINGS.keys()) & set(new_names.keys())

            if duplicates:
                logger.warning(
                    _bilingual(
                        f"Duplicate display names found in module '{module_name}': {sorted(duplicates)}",
                        f"Найдены дубликаты отображаемых имён в модуле {module_name}: {sorted(duplicates)}",
                    )
                )

            NODE_DISPLAY_NAME_MAPPINGS.update(new_names)

            logger.debug(
                _bilingual(
                    f"Display names added from module '{module_name}': {list(new_names.keys())}",
                    f"Добавлены отображаемые имена из {module_name}: {list(new_names.keys())}",
                )
            )

    except Exception as e:
        logger.error(
            _bilingual(
                f"Error while merging mappings from module '{module_name}': {e}",
                f"Ошибка при объединении маппингов из модуля {module_name}: {e}",
            )
        )
        raise


# Load modules immediately when the package is imported.
# Немедленная загрузка модулей при импорте пакета.
try:
    load_modules()
except Exception as e:
    logger.critical(
        _bilingual(
            f"Fatal error during package initialization: {e}",
            f"Фатальная ошибка при инициализации пакета: {e}",
        )
    )

    # Do not stop ComfyUI startup if this package fails.
    # Не останавливать запуск ComfyUI, если этот пакет не смог инициализироваться.

# Export only the required variables.
# Экспортируем только необходимые переменные.
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

# Additional check that mappings are not empty.
# Дополнительная проверка, что маппинги не пустые.
if not NODE_CLASS_MAPPINGS:
    logger.warning(
        _bilingual(
            "NODE_CLASS_MAPPINGS is empty - no modules with nodes were found.",
            "NODE_CLASS_MAPPINGS пуст - возможно, не найдено модулей с узлами.",
        )
    )
else:
    logger.debug(
        _bilingual(
            f"Loaded {len(NODE_CLASS_MAPPINGS)} node classes.",
            f"Загружено {len(NODE_CLASS_MAPPINGS)} классов узлов.",
        )
    )

if not NODE_DISPLAY_NAME_MAPPINGS:
    logger.debug(
        _bilingual(
            "NODE_DISPLAY_NAME_MAPPINGS is empty - class names will be used by default.",
            "NODE_DISPLAY_NAME_MAPPINGS пуст - будут использованы имена классов по умолчанию.",
        )
    )
else:
    logger.debug(
        _bilingual(
            f"Loaded {len(NODE_DISPLAY_NAME_MAPPINGS)} display names.",
            f"Загружено {len(NODE_DISPLAY_NAME_MAPPINGS)} отображаемых имён.",
        )
    )