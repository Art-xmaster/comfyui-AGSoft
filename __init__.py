import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# Получаем путь к текущей директории
current_dir = os.path.dirname(__file__)

# Проходим по всем файлам в директории
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Убираем .py
        module = importlib.import_module(f'.{module_name}', package=__name__)

        # Предполагаем, что в каждом модуле есть NODE_CLASS_MAPPINGS и NODE_DISPLAY_NAME_MAPPINGS
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
