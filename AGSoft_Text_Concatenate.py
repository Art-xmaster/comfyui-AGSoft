# AGSoft Text Concatenate X2, X4, X8
# Автор: AGSoft
# Дата: 02 ноября 2025 г.

from __future__ import annotations
from comfy.comfy_types import IO, InputTypeDict
from typing import Tuple, List


def _build_input_types(text_count: int) -> InputTypeDict:
    """Вспомогательная функция для генерации INPUT_TYPES в зависимости от количества текстовых входов."""
    inputs = {}
    ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
    for i in range(1, text_count + 1):
        en_ord = ordinals.get(i, f"{i}th")
        ru_ord = f"{i}-я"
        inputs[f"text_{i}"] = (
            IO.STRING,
            {
                "multiline": True,
                "tooltip": f"The {en_ord} text string to concatenate. / {ru_ord} текстовая строка для объединения."
            }
        )

    inputs["delimiter"] = (
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
            "default": ",",
            "tooltip": "The separator used between concatenated strings. Choose from presets or 'custom'.\n\n"
                       "Presets:\n"
                       "- 'custom' : Enter your own delimiter in the next field.\n"
                       "- ',' : Comma separator.\n"
                       "- '.' : Period separator.\n"
                       "- 'New Line' : Inserts a line break (\\n) in the output.\n"
                       "- 'Space' : Inserts a single space character.\n"
                       "- ';' : Semicolon separator.\n"
                       "- '|' : Vertical bar separator.\n"
                       "- ' - ' : Dash with spaces.\n"
                       "- ' | ' : Pipe with spaces.\n"
                       "- ' :: ' : Double colon with spaces.\n"
                       "- ' >>> ' : Triple greater-than with spaces.\n"
                       "- ' • ' : Bullet point with spaces.\n"
                       "- ' — ' : Em dash with spaces.\n"
                       "- ' ... ' : Ellipsis with spaces.\n"
                       "/ Разделитель, используемый между объединяемыми строками. Выберите из предустановленных вариантов или 'custom'.\n\n"
                       "Предустановленные значения:\n"
                       "- 'custom' : Введите свой собственный разделитель в следующем поле.\n"
                       "- ',' : Запятая.\n"
                       "- '.' : Точка.\n"
                       "- 'New Line' : Вставляет перенос строки (\\n) в выводе.\n"
                       "- 'Space' : Вставляет один пробел.\n"
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
    )

    inputs["custom_delimiter"] = (
        IO.STRING,
        {
            "default": ",",
            "tooltip": "Enter a custom delimiter if 'custom' is selected above. Ignored otherwise.\n"
                       "Example: ' | ', '---', or even an emoji like '⭐'.\n"
                       "/ Введите пользовательский разделитель, если выше выбрано 'custom'. Игнорируется в противном случае.\n"
                       "Пример: ' | ', '---', или даже эмодзи, например '⭐'."
        },
    )

    inputs["clean_whitespace"] = (
        "BOOLEAN",
        {
            "default": True,
            "tooltip": "If true, removes leading and trailing whitespace from each input string before concatenation.\n"
                       "/ Если установлено, удаляет начальные и конечные пробелы из каждой входной строки перед объединением."
        },
    )

    inputs["ignore_empty_strings"] = (
        "BOOLEAN",
        {
            "default": True,
            "tooltip": "If true, empty strings (or strings that become empty after cleaning whitespace) will be ignored and not included in the final result.\n"
                       "/ Если установлено, пустые строки (или строки, которые становятся пустыми после очистки пробелов) будут проигнорированы и не будут включены в конечный результат."
        },
    )

    return {"required": inputs}


def _concatenate_core(
    texts: List[str],
    delimiter: str,
    custom_delimiter: str,
    clean_whitespace: bool,
    ignore_empty_strings: bool,
) -> str:
    """Основная логика объединения строк."""
    if clean_whitespace:
        texts = [t.strip() for t in texts]

    if ignore_empty_strings:
        texts = [t for t in texts if t != ""]

    # Преобразуем именованные разделители в символы
    actual_delim = delimiter
    if delimiter == "New Line":
        actual_delim = "\n"
    elif delimiter == "Space":
        actual_delim = " "
    elif delimiter == "custom":
        actual_delim = custom_delimiter

    return actual_delim.join(texts)


# ================ Нода X2 ================
class AGSoftTextConcatenateX2:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_input_types(2)

    RETURN_TYPES = (IO.STRING,)
    OUTPUT_TOOLTIPS = ("Concatenated result of 2 input strings. / Результат объединения 2 входных строк.",)
    FUNCTION = "execute"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Concatenates 2 text strings with a configurable delimiter. Empty strings are ignored by default.\n"
        "---\n"
        "Объединяет 2 текстовые строки с настраиваемым разделителем. Пустые строки игнорируются по умолчанию."
    )

    def execute(
        self,
        text_1: str,
        text_2: str,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        result = _concatenate_core(
            texts=[text_1, text_2],
            delimiter=delimiter,
            custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace,
            ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


# ================ Нода X4 ================
class AGSoftTextConcatenateX4:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_input_types(4)

    RETURN_TYPES = (IO.STRING,)
    OUTPUT_TOOLTIPS = ("Concatenated result of 4 input strings. / Результат объединения 4 входных строк.",)
    FUNCTION = "execute"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Concatenates 4 text strings with a configurable delimiter. Empty strings are ignored by default.\n"
        "---\n"
        "Объединяет 4 текстовые строки с настраиваемым разделителем. Пустые строки игнорируются по умолчанию."
    )

    def execute(
        self,
        text_1: str,
        text_2: str,
        text_3: str,
        text_4: str,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        result = _concatenate_core(
            texts=[text_1, text_2, text_3, text_4],
            delimiter=delimiter,
            custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace,
            ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


# ================ Нода X8 ================
class AGSoftTextConcatenateX8:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_input_types(8)

    RETURN_TYPES = (IO.STRING,)
    OUTPUT_TOOLTIPS = ("Concatenated result of 8 input strings. / Результат объединения 8 входных строк.",)
    FUNCTION = "execute"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Concatenates 8 text strings with a configurable delimiter. Empty strings are ignored by default.\n"
        "---\n"
        "Объединяет 8 текстовых строк с настраиваемым разделителем. Пустые строки игнорируются по умолчанию."
    )

    def execute(
        self,
        text_1: str,
        text_2: str,
        text_3: str,
        text_4: str,
        text_5: str,
        text_6: str,
        text_7: str,
        text_8: str,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        result = _concatenate_core(
            texts=[text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8],
            delimiter=delimiter,
            custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace,
            ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


# ================ РЕГИСТРАЦИЯ ================
NODE_CLASS_MAPPINGS = {
    "AGSoftTextConcatenateX2": AGSoftTextConcatenateX2,
    "AGSoftTextConcatenateX4": AGSoftTextConcatenateX4,
    "AGSoftTextConcatenateX8": AGSoftTextConcatenateX8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftTextConcatenateX2": "AGSoft Text Concatenate X2",
    "AGSoftTextConcatenateX4": "AGSoft Text Concatenate X4",
    "AGSoftTextConcatenateX8": "AGSoft Text Concatenate X8",
}