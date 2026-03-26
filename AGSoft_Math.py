"""
AGSoft Math Nodes для ComfyUI
Файл: AGSoft_Math.py
Путь: ComfyUI/custom_nodes/comfyui-AGSoft/AGSoft_Math.py

Набор математических нод:
- AGSoft Int: целочисленная константа
- AGSoft Float: константа с плавающей точкой (с точностью 1-2 знака)
- AGSoft Math Expression: вычисление математических выражений с 4 переменными (a, b, c, d)
- AGSoft Show Any: универсальная нода для отображения любых значений (INT, FLOAT, STRING и их списков)
  с дополнительными выходами int и float (списками)

Категория: AGSoft/Math
"""

import re
import hashlib
from typing import Any, Dict, Tuple, Optional, Union, List

# ----------------------------------------------------------------------------
# Базовые классы для переиспользования логики
# ----------------------------------------------------------------------------

class AGSoftBaseNode:
    """
    Базовый класс для всех AGSoft нод.
    Предоставляет общие методы и атрибуты.
    """
    
    # Категория для всех нод в этом файле
    CATEGORY = "AGSoft/Math"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Базовый метод INPUT_TYPES - должен быть переопределён в дочерних классах.
        Возвращает словарь с типами входных данных.
        """
        return {"required": {}}
    
    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        """
        Указывает ComfyUI, когда нужно пересчитывать ноду.
        Возвращаем текущее время, чтобы нода всегда пересчитывалась,
        если она зависит от внешних данных.
        """
        return float("NaN")  # Всегда пересчитывать


# ----------------------------------------------------------------------------
# Основные ноды
# ----------------------------------------------------------------------------

class AGSoftInt(AGSoftBaseNode):
    """
    Нода для создания целочисленной константы.
    Позволяет задать целое число вручную или через спиннер.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Определяет входной параметр для целого числа.
        
        Returns:
            Словарь с конфигурацией входов
        """
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,          # Значение по умолчанию
                    "min": -1000000,       # Минимальное значение
                    "max": 1000000,        # Максимальное значение
                    "step": 1,             # Шаг изменения
                    "display": "number",   # Отображение как число
                    "tooltip": (
                        "RU: Целое число. Диапазон от -1,000,000 до 1,000,000.\n"
                        "EN: Integer value. Range from -1,000,000 to 1,000,000.\n"
                        "Пример: 42, -7, 100500"
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "get_int"
    
    def get_int(self, value: int) -> Tuple[int]:
        """
        Возвращает целое число.
        
        Args:
            value: Целое число из входа
            
        Returns:
            Кортеж с целым числом
        """
        return (value,)


class AGSoftFloat(AGSoftBaseNode):
    """
    Нода для создания константы с плавающей точкой.
    Позволяет задать дробное число вручную с точностью 1 или 2 знака после запятой.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Определяет входные параметры для числа с плавающей точкой.
        - value: само число
        - precision: количество знаков после запятой (1 или 2)
        
        Returns:
            Словарь с конфигурацией входов
        """
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,        # Значение по умолчанию
                    "min": -1000000.0,     # Минимальное значение
                    "max": 1000000.0,      # Максимальное значение
                    "step": 0.01,          # Шаг изменения
                    "display": "number",   # Отображение как число
                    "tooltip": (
                        "RU: Число с плавающей точкой. Диапазон от -1,000,000 до 1,000,000.\n"
                        "EN: Floating point number. Range from -1,000,000 to 1,000,000.\n"
                        "Пример: 3.14, -0.5, 2.71828"
                    )
                }),
                "precision": (["1", "2"], {
                    "default": "2",         # Количество знаков после запятой по умолчанию
                    "tooltip": (
                        "RU: Количество знаков после запятой (1 или 2).\n"
                        "EN: Number of decimal places (1 or 2).\n"
                        "Пример: 2 → 3.14, 1 → 3.1"
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_float"
    
    def get_float(self, value: float, precision: str) -> Tuple[float]:
        """
        Возвращает число с плавающей точкой с заданной точностью.
        
        Args:
            value: Число из входа
            precision: Количество знаков после запятой (1 или 2 в виде строки)
            
        Returns:
            Кортеж с числом с плавающей точкой (округлённым до указанной точности)
        """
        # Преобразуем строку в целое число
        precision_int = int(precision)
        # Округляем число до указанной точности
        rounded_value = round(value, precision_int)
        return (rounded_value,)


class AGSoftMathExpression(AGSoftBaseNode):
    """
    Нода для вычисления математических выражений.
    Поддерживает переменные a, b, c, d и базовые математические операции.
    Все переменные опциональны - можно использовать только те, которые нужны.
    Принимает любые типы данных на входах (INT, FLOAT и другие).
    
    Безопасное выполнение выражений через ограниченный словарь функций.
    Возвращает результат в трёх форматах: int, float, text.
    """
    
    # Кэш для оптимизации: хранит последние результаты вычислений
    _cache: Dict[str, Tuple[float, str]] = {}
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Определяет входные параметры для математического выражения.
        - expression: строка с математическим выражением (обязательный)
        - a, b, c, d: числа для переменных (опциональные, принимают любые типы)
        
        Returns:
            Словарь с конфигурацией входов
        """
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "a + b",    # Выражение по умолчанию
                    "multiline": False,    # Однострочное поле ввода
                    "tooltip": (
                        "RU: Математическое выражение.\n"
                        "EN: Mathematical expression.\n\n"
                        "Доступные переменные: a, b, c, d\n"
                        "Доступные операции: +, -, *, /, **, %, //\n"
                        "Доступные функции: abs, round, min, max, pow, sqrt, sin, cos, tan\n"
                        "Примеры:\n"
                        "  - a + b * 2\n"
                        "  - (a - b) / c\n"
                        "  - sqrt(a**2 + b**2)\n"
                        "  - sin(pi * a / 180)\n"
                        "  - a + b + c + d"
                    )
                }),
            },
            "optional": {
                "a": ("*", {
                    "default": 0,
                    "tooltip": (
                        "RU: Значение переменной a (опционально).\n"
                        "EN: Value of variable a (optional).\n"
                        "Принимает любые типы: INT, FLOAT и другие.\n"
                        "Используется в выражении как 'a'"
                    )
                }),
                "b": ("*", {
                    "default": 0,
                    "tooltip": (
                        "RU: Значение переменной b (опционально).\n"
                        "EN: Value of variable b (optional).\n"
                        "Принимает любые типы: INT, FLOAT и другие.\n"
                        "Используется в выражении как 'b'"
                    )
                }),
                "c": ("*", {
                    "default": 0,
                    "tooltip": (
                        "RU: Значение переменной c (опционально).\n"
                        "EN: Value of variable c (optional).\n"
                        "Принимает любые типы: INT, FLOAT и другие.\n"
                        "Используется в выражении как 'c'"
                    )
                }),
                "d": ("*", {
                    "default": 0,
                    "tooltip": (
                        "RU: Значение переменной d (опционально).\n"
                        "EN: Value of variable d (optional).\n"
                        "Принимает любые типы: INT, FLOAT и другие.\n"
                        "Используется в выражении как 'd'"
                    )
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("int", "float", "text")
    FUNCTION = "evaluate_expression"
    
    @classmethod
    def _get_cache_key(cls, expression: str, a: float, b: float, c: float, d: float) -> str:
        """
        Генерирует уникальный ключ для кэша на основе выражения и всех значений.
        
        Args:
            expression: Математическое выражение
            a, b, c, d: Значения переменных
            
        Returns:
            Хеш-строка для использования в качестве ключа кэша
        """
        # Создаём строку с выражением и значениями всех переменных
        data = f"{expression}|{a}|{b}|{c}|{d}"
        # Возвращаем MD5 хеш (быстрый и достаточно уникальный)
        return hashlib.md5(data.encode()).hexdigest()
    
    @classmethod
    def _get_safe_globals(cls) -> Dict[str, Any]:
        """
        Возвращает безопасный словарь глобальных функций для eval().
        
        Returns:
            Словарь с разрешёнными математическими функциями и константами
        """
        # Импортируем math только если нужно (ленивая загрузка)
        import math
        
        # Разрешённые функции и константы
        safe_dict = {
            # Математические функции
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            # Математические константы
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,  # 2 * pi
            'inf': float('inf'),
        }
        return safe_dict
    
    def evaluate_expression(self, 
                           expression: str, 
                           a: Any = 0, 
                           b: Any = 0, 
                           c: Any = 0, 
                           d: Any = 0) -> Tuple[int, float, str]:
        """
        Вычисляет математическое выражение и возвращает результат в трёх форматах.
        
        Args:
            expression: Строка с выражением для вычисления
            a, b, c, d: Значения переменных (все опциональны, принимают любые типы)
            
        Returns:
            Кортеж (целочисленный результат, float результат, текстовый результат)
        """
        try:
            # Проверяем наличие выражения
            if not expression or not expression.strip():
                raise ValueError("Expression cannot be empty")
            
            # Очищаем выражение от лишних пробелов
            clean_expression = expression.strip()
            
            # Преобразуем все переменные в float для вычислений
            # Пробуем преобразовать в число, если не получается - используем 0
            def to_float(value: Any) -> float:
                if value is None:
                    return 0.0
                try:
                    # Если это число (int, float) или строка с числом
                    return float(value)
                except (ValueError, TypeError):
                    # Если не удалось преобразовать, возвращаем 0
                    print(f"[AGSoft] Warning: Could not convert '{value}' to float, using 0")
                    return 0.0
            
            a_float = to_float(a)
            b_float = to_float(b)
            c_float = to_float(c)
            d_float = to_float(d)
            
            # Проверяем кэш для оптимизации
            cache_key = self._get_cache_key(clean_expression, a_float, b_float, c_float, d_float)
            if cache_key in self._cache:
                result_float, result_text = self._cache[cache_key]
                # Преобразуем в int и возвращаем
                result_int = int(result_float) if result_float.is_integer() else int(result_float)
                return (result_int, result_float, result_text)
            
            # Базовые проверки безопасности
            # Запрещаем некоторые опасные конструкции
            dangerous_patterns = [
                r'__',           # Магические методы
                r'import',       # Импорты
                r'exec',         # Exec
                r'eval',         # Eval (рекурсивный вызов)
                r'open',         # Открытие файлов
                r'globals',      # Глобальные переменные
                r'locals',       # Локальные переменные
                r'__builtins__', # Встроенные функции
                r'breakpoint',   # Отладка
                r'compile',      # Компиляция кода
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, clean_expression, re.IGNORECASE):
                    raise ValueError(f"Expression contains forbidden pattern: {pattern}")
            
            # Создаём словарь для вычисления
            # Безопасный словарь с разрешёнными функциями
            safe_globals = self._get_safe_globals()
            
            # Словарь с переменными (все опциональны)
            locals_dict = {
                'a': a_float,
                'b': b_float,
                'c': c_float,
                'd': d_float,
            }
            
            # Вычисляем выражение в безопасной среде
            # Используем ограниченный globals и словарь locals
            result = eval(clean_expression, safe_globals, locals_dict)
            
            # Преобразуем результат в float
            result_float = float(result)
            
            # Преобразуем в int (если число целое, берём int, иначе усекаем)
            if result_float.is_integer():
                result_int = int(result_float)
            else:
                # Для нецелых чисел возвращаем int с усечением (как в обычных калькуляторах)
                result_int = int(result_float)
            
            # Формируем строку для отображения (только результат, без выражения)
            if result_float.is_integer():
                # Если результат целый, показываем как целое число
                result_text = f"{result_int}"
            else:
                # Если результат не целый, показываем до 10 знаков после запятой
                # Убираем лишние нули в конце
                result_text = f"{result_float:.10f}".rstrip('0').rstrip('.')
                # Если после удаления нулей получилось целое число, убираем десятичную точку
                if result_text.endswith('.'):
                    result_text = result_text[:-1]
            
            # Сохраняем в кэш
            self._cache[cache_key] = (result_float, result_text)
            
            # Ограничиваем размер кэша (не более 1000 элементов)
            if len(self._cache) > 1000:
                # Удаляем первый элемент (простой способ очистки)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            
            return (result_int, result_float, result_text)
            
        except ZeroDivisionError:
            error_msg = "Division by zero"
            print(f"[AGSoft] {error_msg} in expression: {expression}")
            return (0, 0.0, "Error: Division by zero")
            
        except SyntaxError as e:
            error_msg = f"Syntax error: {str(e)}"
            print(f"[AGSoft] {error_msg} in expression: {expression}")
            return (0, 0.0, f"Error: {error_msg}")
            
        except NameError as e:
            error_msg = f"Unknown variable or function: {str(e)}"
            print(f"[AGSoft] {error_msg} in expression: {expression}")
            return (0, 0.0, f"Error: {error_msg}")
            
        except Exception as e:
            error_msg = f"Evaluation error: {str(e)}"
            print(f"[AGSoft] {error_msg} in expression: {expression}")
            return (0, 0.0, f"Error: {error_msg}")
    
    @classmethod
    def IS_CHANGED(cls, expression: str, a: Any = 0, 
                   b: Any = 0, c: Any = 0, 
                   d: Any = 0) -> float:
        """
        Переопределяем IS_CHANGED для оптимизации.
        Возвращаем значение на основе хеша всех параметров.
        """
        # Преобразуем все в float для единообразия
        def to_float(value: Any) -> float:
            if value is None:
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        a_float = to_float(a)
        b_float = to_float(b)
        c_float = to_float(c)
        d_float = to_float(d)
        
        # Генерируем ключ на основе всех параметров
        key = cls._get_cache_key(expression, a_float, b_float, c_float, d_float)
        # Используем хеш как флаг изменения
        # Возвращаем число, основанное на хеше (но в пределах float)
        hash_int = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        return float(hash_int) / 2**32  # Нормализуем к диапазону [0, 1]


# ----------------------------------------------------------------------------
# Универсальная нода для отображения любых значений (аналог AGSoft Show Text)
# С дополнительными выходами int и float (списками)
# ----------------------------------------------------------------------------

class AGSoftShowAny:
    """
    Универсальная нода для отображения значений любых типов (INT, FLOAT, STRING).
    Полностью повторяет механизм AGSoft Show Text, но с поддержкой любых типов входных данных.
    Поддерживает списки значений.
    
    Дополнительные выходы:
    - int: все значения, преобразованные в целые числа (список)
    - float: все значения, преобразованные в числа с плавающей точкой (список)
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict]:
        """
        Определяет входные параметры для отображения любого значения.
        - value: значение любого типа для отображения (INT, FLOAT, STRING, списки)
        
        Returns:
            Словарь с конфигурацией входов
        """
        return {
            "required": {
                "value": ("*", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "INT", "FLOAT")
    RETURN_NAMES = ("text", "int", "float")
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True, True)  # Все выходы могут быть списками
    
    CATEGORY = "AGSoft/Math"
    DESCRIPTION = """A universal node for displaying values of any type (INT, FLOAT, STRING) directly in the ComfyUI interface.
Supports lists of values.
Useful for debugging, previewing results, or showing outputs from other nodes.

Additional outputs:
- int: all values converted to integers (list)
- float: all values converted to floats (list)

\n
Универсальная нода для отображения значений любого типа (INT, FLOAT, STRING) непосредственно в интерфейсе ComfyUI.
Поддерживает списки значений.
Полезна для отладки, просмотра результатов или показа выходных данных других нод.

Дополнительные выходы:
- int: все значения, преобразованные в целые числа (список)
- float: все значения, преобразованные в числа с плавающей точкой (список)"""
    
    def convert_value_to_string(self, value: Any) -> str:
        """
        Преобразует значение любого типа в строку для отображения.
        
        Args:
            value: Значение для преобразования
            
        Returns:
            Строковое представление значения
        """
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, float):
            # Форматируем float для красивого отображения
            if value.is_integer():
                return f"{int(value)}"
            else:
                # Показываем до 10 знаков после запятой, убираем лишние нули
                formatted = f"{value:.10f}".rstrip('0').rstrip('.')
                return formatted
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, tuple)):
            # Для списков рекурсивно преобразуем каждый элемент
            converted = [self.convert_value_to_string(item) for item in value]
            return "[" + ", ".join(converted) + "]"
        elif isinstance(value, dict):
            # Для словарей преобразуем ключи и значения
            converted = {str(k): self.convert_value_to_string(v) for k, v in value.items()}
            return str(converted)
        else:
            # Для всего остального используем str()
            return str(value)
    
    def convert_to_int(self, value: Any) -> int:
        """
        Преобразует значение в целое число.
        
        Args:
            value: Значение для преобразования
            
        Returns:
            Целое число (0 если преобразование невозможно)
        """
        if value is None:
            return 0
        
        try:
            # Пробуем преобразовать в float, затем в int
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def convert_to_float(self, value: Any) -> float:
        """
        Преобразует значение в число с плавающей точкой.
        
        Args:
            value: Значение для преобразования
            
        Returns:
            Число с плавающей точкой (0.0 если преобразование невозможно)
        """
        if value is None:
            return 0.0
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def notify(self, value, unique_id=None, extra_pnginfo=None):
        """
        Отображает значение в интерфейсе ComfyUI и возвращает числовые выходы.
        
        Args:
            value: Значение для отображения (может быть одиночным или списком)
            unique_id: Уникальный ID ноды (для сохранения состояния)
            extra_pnginfo: Дополнительная информация (для сохранения состояния)
            
        Returns:
            Словарь с данными для UI и результат (текст, int, float)
        """
        # Преобразуем все значения в строки
        text_strings = []
        int_values = []
        float_values = []
        
        # value уже является списком из-за INPUT_IS_LIST = True
        # Преобразуем каждый элемент в списке
        for v in value:
            # Преобразуем в строку для отображения
            text_strings.append(self.convert_value_to_string(v))
            
            # Преобразуем в int и float для выходов
            int_values.append(self.convert_to_int(v))
            float_values.append(self.convert_to_float(v))
        
        # Сохраняем состояние ноды в workflow (как в AGSoftShowText)
        if unique_id is not None and extra_pnginfo is not None:
            try:
                if not isinstance(extra_pnginfo, list):
                    print("[AGSoft] Error: extra_pnginfo is not a list")
                elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                    print("[AGSoft] Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
                else:
                    workflow = extra_pnginfo[0]["workflow"]
                    node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])), None)
                    if node:
                        node["widgets_values"] = [text_strings]
            except Exception as e:
                print(f"[AGSoft] Error saving node state: {e}")
        
        # Возвращаем данные для UI и результат (текст, int, float)
        # Все три выхода возвращаются как списки
        return {"ui": {"text": text_strings}, "result": (text_strings, int_values, float_values)}


# ----------------------------------------------------------------------------
# Регистрация нод
# ----------------------------------------------------------------------------

# Словарь для маппинга классов нод
NODE_CLASS_MAPPINGS = {
    "AGSoftInt": AGSoftInt,
    "AGSoftFloat": AGSoftFloat,
    "AGSoftMathExpression": AGSoftMathExpression,
    "AGSoftShowAny": AGSoftShowAny,
}

# Словарь для отображаемых имён нод
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftInt": "🔢AGSoft Integer",
    "AGSoftFloat": "🔢AGSoft Float",
    "AGSoftMathExpression": "🧮AGSoft Math Expression",
    "AGSoftShowAny": "👁️AGSoft Show Any",
}
