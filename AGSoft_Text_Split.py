# AGSoft Text Split
# Автор: AGSoft
# Дата: 13 ноября 2025 г.
#
# Описание: Разбивает текст на части и создаёт отдельные выходы для каждой части.
# Количество выходов определяется параметром number_of_outputs.
# Можно указать, с какого по счету разделителя начинать разбивку.

from __future__ import annotations
from comfy.comfy_types import IO, InputTypeDict
from typing import Tuple, List

class AGSoftTextSplit:
    """
    Нода для разбиения текста на отдельные строки с фиксированным количеством выходов.
    Количество выходов задаётся параметром number_of_outputs.
    Можно указать, с какого по счету разделителя начинать разбивку.
    """
    
    MAX_OUTPUTS = 50  # Максимальное количество выходов

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """Определяет типы входных параметров ноды."""
        return {
            "required": {
                # Основной текст для разбиения
                "text": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "tooltip": "The text to be split into parts. / Текст, который будет разбит на части."
                    }
                ),
                # Выбор разделителя
                "delimiter": (
                    [
                        "custom",
                        ",",
                        ".",
                        "New Line",
                        "Space",
                        ";",
                        "|",
                        " - ",
                        " | ",
                        " :: ",
                        " >>> ",
                        " • ",
                        " — ",
                        " ... ",
                    ],
                    {
                        "default": "Space",
                        "tooltip": "The separator used to split the text. Choose from presets or 'custom'.\n"
                                   "Presets:\n"
                                   "- 'custom' : Enter your own delimiter in the next field.\n"
                                   "- ',' : Comma separator.\n"
                                   "- '.' : Period separator.\n"
                                   "- 'New Line' : Splits by line breaks (\\n).\n"
                                   "- 'Space' : Splits by single space character.\n"
                                   "- ';' : Semicolon separator.\n"
                                   "- '|' : Vertical bar separator.\n"
                                   "- ' - ' : Dash with spaces.\n"
                                   "- ' | ' : Pipe with spaces.\n"
                                   "- ' :: ' : Double colon with spaces.\n"
                                   "- ' >>> ' : Triple greater-than with spaces.\n"
                                   "- ' • ' : Bullet point with spaces.\n"
                                   "- ' — ' : Em dash with spaces.\n"
                                   "- ' ... ' : Ellipsis with spaces.\n"
                                   "/ Разделитель, используемый для разбиения текста. Выберите из предустановленных вариантов или 'custom'.\n"
                                   "Предустановленные значения:\n"
                                   "- 'custom' : Введите свой собственный разделитель в следующем поле.\n"
                                   "- ',' : Запятая.\n"
                                   "- '.' : Точка.\n"
                                   "- 'New Line' : Разбивает по переносам строк (\\n).\n"
                                   "- 'Space' : Разбивает по одиночному пробелу.\n"
                                   "- ';' : Точка с запятой.\n"
                                   "- '|' : Вертикальная черта.\n"
                                   "- ' - ' : Дефис с пробелами.\n"
                                   "- ' | ' : Пайп с пробелами.\n"
                                   "- ' :: ' : Двойное двоеточие с пробелами.\n"
                                   "- ' >>> ' : Три знака больше с пробелами.\n"
                                   "- ' • ' : Эмодзи-точка с пробелами.\n"
                                   "- ' — ' : Длинное тире с пробелами.\n"
                                   "- ' ... ' : Многоточие с пробелами."
                    },
                ),
                # Поле для пользовательского разделителя (активно только если выбрано 'custom')
                "custom_delimiter": (
                    IO.STRING,
                    {
                        "default": " ",
                        "tooltip": "Enter a custom delimiter if 'custom' is selected above. Ignored otherwise.\n"
                                   "Example: ' | ', '---', or even an emoji like '⭐'.\n"
                                   "/ Введите пользовательский разделитель, если выше выбрано 'custom'. Игнорируется в противном случае.\n"
                                   "Пример: ' | ', '---', или даже эмодзи, например '⭐'."
                    },
                ),
                # Опция для очистки пробелов
                "clean_whitespace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, removes leading and trailing whitespace from each resulting part before output.\n"
                                   "/ Если установлено, удаляет начальные и конечные пробелы из каждой полученной части перед выводом."
                    },
                ),
                # Опция для игнорирования пустых строк
                "ignore_empty_strings": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, empty strings (or strings that become empty after cleaning whitespace) will be ignored and not included in the final outputs.\n"
                                   "/ Если установлено, пустые строки (или строки, которые становятся пустыми после очистки пробелов) будут проигнорированы и не будут включены в конечные выходы."
                    },
                ),
                # Количество выходов
                "number_of_outputs": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": cls.MAX_OUTPUTS,
                        "step": 1,
                        "tooltip": "How many text outputs to create. Each output will be named string_1, string_2, etc.\n"
                                   "Example: number_of_outputs=3 → creates string_1, string_2, string_3.\n"
                                   "/ Сколько текстовых выходов создать. Каждый выход будет называться string_1, string_2 и т.д.\n"
                                   "Пример: number_of_outputs=3 → создаются string_1, string_2, string_3."
                    }
                ),
                # Начинать разбивку с разделителя под номером
                "start_from_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Start splitting from this delimiter number (1-based index).\n"
                                   "Example: text='111 222 333 777 999 888', start_from_index=4 → parts=['777', '999', '888']\n"
                                   "If the index is larger than the number of parts, all outputs will be empty.\n"
                                   "/ Начинать разбивку с этого номера разделителя (индексация с 1).\n"
                                   "Пример: text='111 222 333 777 999 888', start_from_index=4 → parts=['777', '999', '888']\n"
                                   "Если индекс больше количества частей, все выходы будут пустыми."
                    }
                )
            }
        }

    # Фиксированное количество выходов (максимальное)
    RETURN_TYPES = ("STRING",) * MAX_OUTPUTS
    RETURN_NAMES = tuple(f"string_{i+1}" for i in range(MAX_OUTPUTS))
    FUNCTION = "split_text"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Splits a single input text string into multiple parts based on a configurable delimiter.\n"
        "Creates fixed number of outputs (string_1, string_2, etc.) up to number_of_outputs.\n"
        "Option to start splitting from a specific delimiter number (1-based index).\n"
        "Useful for processing individual segments of text in loops or with ForEach nodes.\n"
        "--- \n"
        "Разбивает одну входную текстовую строку на несколько частей по настраиваемому разделителю.\n"
        "Создает фиксированное количество выходов (string_1, string_2 и т.д.) до number_of_outputs.\n"
        "Возможность начать разбивку с определенного номера разделителя (индексация с 1).\n"
        "Полезна для обработки отдельных сегментов текста в циклах или с помощью нод ForEach."
    )

    def split_text(self, text: str, delimiter: str, custom_delimiter: str, clean_whitespace: bool, ignore_empty_strings: bool, number_of_outputs: int, start_from_index: int) -> Tuple[str, ...]:
        """
        Основная логика разбиения текста.
        Возвращает кортеж строк фиксированной длины (MAX_OUTPUTS).
        Начинает разбивку с указанного индекса (1-based).
        """
        # Преобразуем именованные разделители в символы
        actual_delim = delimiter
        if delimiter == "New Line":
            actual_delim = "\n"
        elif delimiter == "Space":
            actual_delim = " "
        elif delimiter == "custom":
            actual_delim = custom_delimiter

        # Разбиваем текст на части
        parts = text.split(actual_delim)

        # Обработка опций
        if clean_whitespace:
            parts = [part.strip() for part in parts]
        if ignore_empty_strings:
            parts = [part for part in parts if part != ""]

        # Учитываем start_from_index (1-based index)
        # Если start_from_index больше длины parts, то parts будет пустым списком
        start_idx = max(0, start_from_index - 1)  # Конвертируем в 0-based index
        if start_idx < len(parts):
            parts = parts[start_idx:]
        else:
            parts = []

        # Создаём кортеж фиксированной длины
        result_parts = parts[:number_of_outputs] + [""] * (self.MAX_OUTPUTS - len(parts[:number_of_outputs]))
        
        return tuple(result_parts)

    @classmethod
    def IS_CHANGED(cls, text, delimiter, custom_delimiter, clean_whitespace, ignore_empty_strings, number_of_outputs, start_from_index):
        """
        Указывает ComfyUI, что нода всегда должна пересчитываться при изменении входов.
        """
        return float("nan")  # Всегда пересчитывать

    @classmethod
    def OUTPUT_NODE(cls):
        """
        Указывает, что эта нода является выходной (Output Node).
        """
        return True

    @classmethod
    def DISPLAY_NAME(cls):
        """
        Возвращает имя ноды, которое будет отображаться в UI ComfyUI.
        """
        return "AGSoft Text Split"

# ================ РЕГИСТРАЦИЯ НОДЫ ================
NODE_CLASS_MAPPINGS = {
    "AGSoftTextSplit": AGSoftTextSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftTextSplit": "AGSoft Text Split",
}