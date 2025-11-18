# AGSoft_Text.py
# Автор: AGSoft
# Дата: 18 ноября 2025 г.
#

from __future__ import annotations
import re
import os
import sys
import json

# Импортируем необходимые модули из ComfyUI.
# Важно: используем только те, что гарантированно доступны в базовой установке.
from typing import Any, Dict, List, Tuple, Union

class AGSoftTextMultiline:
    """
    Нода для ввода многострочного текста.
    Позволяет вводить и редактировать длинные текстовые строки с поддержкой переносов строк.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,  # Разрешает многострочный ввод
                        "dynamicPrompts": True,  # Поддержка динамических промптов (например, через {variable})
                        "tooltip": "The text to be processed. Supports multiline input and dynamic prompts.\nТекст для обработки. Поддерживает многострочный ввод и динамические промпты."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A node for entering and processing multi-line text.
Useful for writing long prompts or complex instructions that require line breaks.
\n
Нода для ввода и обработки многострочного текста.
Полезна для написания длинных промптов или сложных инструкций, требующих переносов строк."""

    def process(self, text: str) -> Tuple[str]:
        """
        Просто возвращает введенный текст без изменений.
        Это базовая нода-контейнер для текста.
        """
        return (text,)


class AGSoftTextReplace:
    """
    Нода для поиска и замены текста.
    Позволяет выполнить до трех операций замены в одной строке.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": False,
                        "dynamicPrompts": True,
                        "tooltip": "The original text in which to perform replacements.\nИсходный текст, в котором будут выполняться замены."
                    },
                ),
                "find1": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The substring to find for the first replacement.\nПодстрока, которую нужно найти для первой замены."
                    },
                ),
                "replace1": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The string to replace the first found substring with.\nСтрока, на которую будет заменена первая найденная подстрока."
                    },
                ),
                "case_sensitive1": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether the first search is case-sensitive.\nУчитывать ли регистр при первом поиске."
                    },
                ),
                "whole_word1": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether the first search matches whole words only.\nСоответствует ли первому поиску только целые слова."
                    },
                ),
                "find2": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The substring to find for the second replacement.\nПодстрока, которую нужно найти для второй замены."
                    },
                ),
                "replace2": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The string to replace the second found substring with.\nСтрока, на которую будет заменена вторая найденная подстрока."
                    },
                ),
                "case_sensitive2": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether the second search is case-sensitive.\nУчитывать ли регистр при втором поиске."
                    },
                ),
                "whole_word2": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether the second search matches whole words only.\nСоответствует ли второму поиску только целые слова."
                    },
                ),
                "find3": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The substring to find for the third replacement.\nПодстрока, которую нужно найти для третьей замены."
                    },
                ),
                "replace3": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The string to replace the third found substring with.\nСтрока, на которую будет заменена третья найденная подстрока."
                    },
                ),
                "case_sensitive3": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether the third search is case-sensitive.\nУчитывать ли регистр при третьем поиске."
                    },
                ),
                "whole_word3": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether the third search matches whole words only.\nСоответствует ли третьему поиску только целые слова."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A node for performing multiple find-and-replace operations on a single text string.
You can define up to three different search and replace pairs, with options for case sensitivity and whole-word matching.
\n
Нода для выполнения нескольких операций поиска и замены в одной текстовой строке.
Вы можете определить до трех разных пар поиска и замены с опциями чувствительности к регистру и соответствия целым словам."""

    def process(
        self,
        text: str,
        find1: str,
        replace1: str,
        case_sensitive1: bool,
        whole_word1: bool,
        find2: str,
        replace2: str,
        case_sensitive2: bool,
        whole_word2: bool,
        find3: str,
        replace3: str,
        case_sensitive3: bool,
        whole_word3: bool,
    ) -> Tuple[str]:
        """
        Выполняет последовательную замену подстрок в тексте.
        Замены применяются в порядке от 1 к 3.
        """
        result = text

        # --- Замена 1 ---
        if find1:
            flags = 0 if case_sensitive1 else re.IGNORECASE
            pattern = re.escape(find1)
            if whole_word1:
                pattern = r'\b' + pattern + r'\b'
            result = re.sub(pattern, replace1, result, flags=flags)

        # --- Замена 2 ---
        if find2:
            flags = 0 if case_sensitive2 else re.IGNORECASE
            pattern = re.escape(find2)
            if whole_word2:
                pattern = r'\b' + pattern + r'\b'
            result = re.sub(pattern, replace2, result, flags=flags)

        # --- Замена 3 ---
        if find3:
            flags = 0 if case_sensitive3 else re.IGNORECASE
            pattern = re.escape(find3)
            if whole_word3:
                pattern = r'\b' + pattern + r'\b'
            result = re.sub(pattern, replace3, result, flags=flags)

        return (result,)


class AGSoftTextOperation:
    """
    Нода для выполнения различных текстовых операций.
    Позволяет преобразовать текст одним из предопределенных способов.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text to apply the operation to. This is the source string that will be modified.\nТекст, к которому будет применена операция. Это исходная строка, которая будет изменена."
                    },
                ),
                "operation": (
                    [
                        "uppercase",
                        "lowercase",
                        "capitalize",
                        "invert_case",
                        "reverse",
                        "trim",
                        "remove_spaces",
                        "count_words",
                        "split_lines",
                        "join_lines",
                    ],
                    {
                        "default": "uppercase",
                        "tooltip": "Select the operation to perform on the input text.\nВыберите операцию, которую нужно выполнить над входным текстом.\n\n- uppercase: Converts all characters to uppercase.\n  Преобразует все символы в верхний регистр.\n- lowercase: Converts all characters to lowercase.\n  Преобразует все символы в нижний регистр.\n- capitalize: Capitalizes the first letter of the entire string.\n  Делает первую букву всей строки заглавной.\n- invert_case: Inverts the case of each character (upper becomes lower and vice versa).\n  Инвертирует регистр каждого символа (верхний становится нижним и наоборот).\n- reverse: Reverses the order of characters in the string.\n  Переворачивает порядок символов в строке.\n- trim: Removes leading and trailing whitespace (spaces, tabs, newlines) from the string.\n  Удаляет начальные и конечные пробельные символы (пробелы, табуляции, переводы строк) из строки.\n- remove_spaces: Removes ALL whitespace characters (spaces, tabs, newlines) from the string.\n  Удаляет ВСЕ пробельные символы (пробелы, табуляции, переводы строк) из строки.\n- count_words: Counts the number of words in the text. The result is output to the 'count' socket.\n  Подсчитывает количество слов в тексте. Результат выводится в сокет 'count'.\n- split_lines: Splits the text into lines and outputs them as a single string joined by newlines. Useful for processing multi-line text.\n  Разбивает текст на строки и выводит их как одну строку, соединенную символами новой строки. Полезно для обработки многострочного текста.\n- join_lines: Joins all lines of the text into a single line, separated by a space. Useful for flattening multi-line text.\n  Объединяет все строки текста в одну строку, разделенные пробелом. Полезно для сглаживания многострочного текста."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING") # text, count, lines
    RETURN_NAMES = ("text", "count", "lines")
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A node for applying common text transformations.
Choose from a list of predefined operations like uppercase, lowercase, etc.
The 'count_words' operation outputs the word count to the 'count' socket.
The 'split_lines' operation outputs the joined lines to the 'lines' socket as a single string separated by newlines.
\n
Нода для применения распространенных текстовых преобразований.
Выберите одну из предопределенных операций, таких как верхний регистр, нижний регистр и т.д.
Операция 'count_words' выводит количество слов в сокет 'count'.
Операция 'split_lines' выводит объединенные строки в сокет 'lines' как одну строку, разделенную символами новой строки."""

    def process(self, text: str, operation: str) -> Tuple[str, int, str]:
        """
        Применяет выбранную текстовую операцию к входной строке.
        """
        result_text = text
        result_count = 0
        result_lines = ""

        if operation == "uppercase":
            result_text = text.upper()
        elif operation == "lowercase":
            result_text = text.lower()
        elif operation == "capitalize":
            result_text = text.capitalize()
        elif operation == "invert_case":
            result_text = "".join(
                char.lower() if char.isupper() else char.upper() for char in text
            )
        elif operation == "reverse":
            result_text = text[::-1]
        elif operation == "trim":
            # Используем strip(), который удаляет ВСЕ пробельные символы с начала и конца строки.
            result_text = text.strip()
        elif operation == "remove_spaces":
            # Используем join(split()) для удаления ВСЕХ пробельных символов (включая табуляции и переводы строк).
            # split() без аргумента разбивает по ЛЮБЫМ пробельным символам.
            # join() соединяет без разделителя.
            result_text = ''.join(text.split())
        elif operation == "count_words":
            # Простой подсчет слов по пробелам и другим разделителям
            import re
            words = re.findall(r'\S+', text.strip())
            result_count = len(words) if words else 0
            result_text = text # Возвращаем исходный текст в текстовый сокет
        elif operation == "split_lines":
            # Разбиваем по символам новой строки, убираем пустые строки
            # splitlines() корректно обрабатывает \n, \r\n, \r
            lines = [line for line in text.splitlines() if line.strip()]
            result_lines = '\n'.join(lines)
            result_text = text # Возвращаем исходный текст в текстовый сокет
        elif operation == "join_lines":
            # Объединяем строки, убирая пустые и лишние пробелы
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            result_text = ' '.join(lines) # Объединяем пробелом

        return (result_text, result_count, result_lines)


class AGSoftTextInputSwitchX2:
    """
    Нода-переключатель для выбора одного из двух текстовых входов.
    Полезна для создания условных потоков данных.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text1": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The first text option.\nПервый текстовый вариант."
                    },
                ),
                "text2": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The second text option.\nВторой текстовый вариант."
                    },
                ),
                "input_selector": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2,
                        "step": 1,
                        "tooltip": "Select which input text to output (1 or 2).\nВыберите, какой входной текст выводить (1 или 2)."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A switch node that outputs one of two text inputs based on a selector value.
Useful for conditional logic or A/B testing within your workflow.
\n
Нода-переключатель, которая выводит один из двух текстовых входов в зависимости от значения селектора.
Полезна для условной логики или A/B тестирования в вашем рабочем процессе."""

    def process(self, text1: str, text2: str, input_selector: int) -> Tuple[str]:
        """
        Возвращает текст в зависимости от значения селектора.
        Если селектор равен 1, возвращает text1; если 2, возвращает text2.
        """
        if input_selector == 1:
            return (text1,)
        elif input_selector == 2:
            return (text2,)
        else:
            # На случай, если значение выйдет за пределы, возвращаем первый текст.
            return (text1,)


class AGSoftTextInputSwitchX4:
    """
    Нода-переключатель для выбора одного из четырех текстовых входов.
    Расширенная версия X2 для более сложных условий.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text1": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The first text option.\nПервый текстовый вариант."
                    },
                ),
                "text2": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The second text option.\nВторой текстовый вариант."
                    },
                ),
                "text3": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The third text option.\nТретий текстовый вариант."
                    },
                ),
                "text4": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The fourth text option.\nЧетвертый текстовый вариант."
                    },
                ),
                "input_selector": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "Select which input text to output (1-4).\nВыберите, какой входной текст выводить (1-4)."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A switch node that outputs one of four text inputs based on a selector value.
Extends the X2 switch for more complex branching logic.
\n
Нода-переключатель, которая выводит один из четырех текстовых входов в зависимости от значения селектора.
Расширяет X2 переключатель для более сложной логики ветвления."""

    def process(
        self, text1: str, text2: str, text3: str, text4: str, input_selector: int
    ) -> Tuple[str]:
        """
        Возвращает текст в зависимости от значения селектора.
        """
        options = [text1, text2, text3, text4]
        # Безопасно: если input_selector выйдет за пределы [1,4], возвращаем первый
        index = max(0, min(input_selector - 1, len(options) - 1))
        return (options[index],)


class AGSoftTextInputSwitchX8:
    """
    Нода-переключатель для выбора одного из восьми текстовых входов.
    Максимально гибкая версия для сложных сценариев.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text1": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The first text option.\nПервый текстовый вариант."
                    },
                ),
                "text2": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The second text option.\nВторой текстовый вариант."
                    },
                ),
                "text3": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The third text option.\nТретий текстовый вариант."
                    },
                ),
                "text4": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The fourth text option.\nЧетвертый текстовый вариант."
                    },
                ),
                "text5": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The fifth text option.\nПятый текстовый вариант."
                    },
                ),
                "text6": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The sixth text option.\nШестой текстовый вариант."
                    },
                ),
                "text7": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The seventh text option.\nСедьмой текстовый вариант."
                    },
                ),
                "text8": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The eighth text option.\nВосьмой текстовый вариант."
                    },
                ),
                "input_selector": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Select which input text to output (1-8).\nВыберите, какой входной текст выводить (1-8)."
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A switch node that outputs one of eight text inputs based on a selector value.
Ideal for workflows requiring many discrete text choices.
\n
Нода-переключатель, которая выводит один из восьми текстовых входов в зависимости от значения селектора.
Идеальна для рабочих процессов, требующих множества дискретных текстовых вариантов."""

    def process(
        self,
        text1: str,
        text2: str,
        text3: str,
        text4: str,
        text5: str,
        text6: str,
        text7: str,
        text8: str,
        input_selector: int,
    ) -> Tuple[str]:
        """
        Возвращает текст в зависимости от значения селектора.
        """
        options = [text1, text2, text3, text4, text5, text6, text7, text8]
        # Безопасно: если input_selector выйдет за пределы [1,8], возвращаем первый
        index = max(0, min(input_selector - 1, len(options) - 1))
        return (options[index],)


class AGSoftShowText:
    """
    Нода для отображения текста в интерфейсе ComfyUI.
    Полезна для отладки, просмотра промптов или вывода результатов работы других нод.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """A node for displaying text directly in the ComfyUI interface.
Useful for debugging, previewing prompts, or showing results from other nodes.
\n
Нода для отображения текста непосредственно в интерфейсе ComfyUI.
Полезна для отладки, просмотра промптов или показа результатов работы других нод."""

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        # Обновляем widgets_values в workflow для сохранения состояния
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next((x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])), None)
                if node:
                    node["widgets_values"] = [text]
        
        # Возвращаем данные для UI и результат
        return {"ui": {"text": text}, "result": (text,)}



# --- РЕГИСТРАЦИЯ НОД ---

# Обязательный словарь для регистрации классов нод.
NODE_CLASS_MAPPINGS = {
    "AGSoft Text Multiline": AGSoftTextMultiline,
    "AGSoft Text Replace": AGSoftTextReplace,
    "AGSoft Text Operation": AGSoftTextOperation,
    "AGSoft Text Input Switch X2": AGSoftTextInputSwitchX2,
    "AGSoft Text Input Switch X4": AGSoftTextInputSwitchX4,
    "AGSoft Text Input Switch X8": AGSoftTextInputSwitchX8,
    "AGSoft Show Text": AGSoftShowText,
}

# Словарь для отображения красивых имен нод в интерфейсе ComfyUI.
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft Text Multiline": "AGSoft Text Multiline",
    "AGSoft Text Replace": "AGSoft Text Replace",
    "AGSoft Text Operation": "AGSoft Text Operation",
    "AGSoft Text Input Switch X2": "AGSoft Text Input Switch X2",
    "AGSoft Text Input Switch X4": "AGSoft Text Input Switch X4",
    "AGSoft Text Input Switch X8": "AGSoft Text Input Switch X8",
    "AGSoft Show Text": "AGSoft Show Text",
}