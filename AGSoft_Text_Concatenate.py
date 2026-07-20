# ==============================================================================
# AGSoft_Text_Concatenate.py
# ==============================================================================
# Нода: 📝🪡AGSoft Text Concatenate (Dynamic & Static X2/X4/X8)
#
# ОСОБЕННОСТИ:
# Нода для склеивания текстовых строк с настраиваемыми разделителями,
# поддержкой динамического количества входов и умной фильтрацией пустых значений.
# 
# Автор: AGSoft
# Дата: 20.07.2026
# ==============================================================================

from __future__ import annotations

from comfy.comfy_types import IO, InputTypeDict
from typing import Tuple, List, Any


def _process_escape_sequences(text: str) -> str:
    """Обрабатывает escape-последовательности в строке."""
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    text = text.replace("\\r", "\r")
    return text


def _concatenate_core(
    texts: List[Any],
    delimiter: str,
    custom_delimiter: str,
    clean_whitespace: bool,
    ignore_empty_strings: bool,
) -> str:
    """Основная логика объединения строк."""
    # 1. Гарантируем, что все элементы - строки. Обрабатываем None.
    texts = [str(t) if t is not None else "" for t in texts]

    # 2. Очистка пробелов по краям (если включена)
    if clean_whitespace:
        texts = [t.strip() for t in texts]

    # 3. Фильтрация пустых строк
    if ignore_empty_strings:
        texts = [t for t in texts if t.strip() != ""]

    # 4. Преобразуем именованные разделители в символы
    actual_delim = delimiter
    if delimiter == "New Line":
        actual_delim = "\n"
    elif delimiter == "New Line X2":
        actual_delim = "\n\n"
    elif delimiter == "Space":
        actual_delim = " "
    elif delimiter == "custom":
        actual_delim = _process_escape_sequences(custom_delimiter)

    return actual_delim.join(texts)


def _build_static_input_types(text_count: int) -> InputTypeDict:
    """Вспомогательная функция для генерации INPUT_TYPES для статических нод (X2, X4, X8)."""
    inputs = {}
    ordinals = {1: "1st", 2: "2nd", 3: "3rd"}
    
    for i in range(1, text_count + 1):
        en_ord = ordinals.get(i, f"{i}th")
        ru_ord = f"{i}-я"
        inputs[f"text_{i}"] = (
            IO.STRING,
            {
                "multiline": True,
                "default": "",
                "tooltip": f"The {en_ord} text string to concatenate. / {ru_ord} текстовая строка для объединения.",
            },
        )

    inputs["reverse_order"] = (
        "BOOLEAN",
        {
            "default": False,
            "tooltip": (
                "If true, concatenates strings in reverse order (from last active input to first).\n"
                "Example: text_1='A', text_2='B', text_3='C' → result will be 'C, B, A'\n"
                "---\n"
                "Если включено, объединяет строки в обратном порядке (от последнего активного входа к первому).\n"
                "Пример: text_1='A', text_2='B', text_3='C' → результат будет 'C, B, A'"
            ),
        },
    )

    inputs["delimiter"] = (
        [
            "custom", ",", ".", "New Line", "New Line X2", "Space", ";", "|",
            " - ", " | ", " :: ", " >>> ", " • ", " — ", " ... ",
        ],
        {
            "default": ",",
            "tooltip": (
                "The separator used between concatenated strings. Choose from presets or 'custom'.\n\n"
                "Presets:\n"
                "- 'custom' : Enter your own delimiter in the next field. Supports escape sequences: \\n (newline), \\t (tab), \\r (carriage return).\n"
                "- ',' : Comma separator.\n"
                "- '.' : Period separator.\n"
                "- 'New Line' : Inserts a line break (\\n) in the output.\n"
                "- 'New Line X2' : Inserts a blank line (\\n\\n) for visual spacing between paragraphs.\n"
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
                "---\n"
                "Разделитель, используемый между объединяемыми строками. Выберите из предустановленных вариантов или 'custom'.\n\n"
                "Предустановленные значения:\n"
                "- 'custom' : Введите свой собственный разделитель в следующем поле. Поддерживает escape-последовательности: \\n (перенос строки), \\t (табуляция), \\r (возврат каретки).\n"
                "- ',' : Запятая.\n"
                "- '.' : Точка.\n"
                "- 'New Line' : Вставляет перенос строки (\\n) в выводе.\n"
                "- 'New Line X2' : Вставляет пустую строку (\\n\\n) для визуального разделения абзацев.\n"
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
            ),
        },
    )

    inputs["custom_delimiter"] = (
        IO.STRING,
        {
            "default": "",
            "tooltip": (
                "Enter a custom delimiter if 'custom' is selected above. Ignored otherwise.\n"
                "Supports escape sequences: \\n (newline), \\t (tab), \\r (carriage return).\n"
                "Example: '\\n\\n' for blank line, ' | ', '---', or even an emoji like '⭐'.\n"
                "---\n"
                "Введите пользовательский разделитель, если выше выбрано 'custom'. Игнорируется в противном случае.\n"
                "Поддерживает escape-последовательности: \\n (перенос строки), \\t (табуляция), \\r (возврат каретки).\n"
                "Пример: '\\n\\n' для пустой строки, ' | ', '---', или даже эмодзи, например '⭐'."
            ),
        },
    )

    inputs["clean_whitespace"] = (
        "BOOLEAN",
        {
            "default": True,
            "tooltip": (
                "If true, removes leading and trailing whitespace from each input string before concatenation.\n"
                "Example: '  hello  ' → 'hello'\n"
                "---\n"
                "Если установлено, удаляет начальные и конечные пробелы из каждой входной строки перед объединением.\n"
                "Пример: '  hello  ' → 'hello'"
            ),
        },
    )

    inputs["ignore_empty_strings"] = (
        "BOOLEAN",
        {
            "default": True,
            "tooltip": (
                "If true, empty strings (or strings that become empty after cleaning whitespace) will be ignored and not included in the final result.\n"
                "This includes strings containing only spaces, tabs, or newlines.\n"
                "---\n"
                "Если установлено, пустые строки (или строки, которые становятся пустыми после очистки пробелов) будут проигнорированы и не будут включены в конечный результат.\n"
                "Это включает строки, состоящие только из пробелов, табуляций или переносов строк."
            ),
        },
    )

    return {"required": inputs}


# ================= ДИНАМИЧЕСКАЯ НОДА (JS) =================
class AGSoftTextConcatenate:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "number_of_inputs": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Number of text inputs to create (2-32). Each input will be named text_1, text_2, etc.\n"
                            "Example: number_of_inputs=3 → creates text_1, text_2, text_3\n"
                            "Inputs are managed dynamically via frontend JS.\n"
                            "---\n"
                            "Количество текстовых входов (2-32). Каждый вход будет называться text_1, text_2 и т.д.\n"
                            "Пример: number_of_inputs=3 → создаются text_1, text_2, text_3\n"
                            "Входы управляются динамически через JS на фронтенде."
                        ),
                    },
                ),
                "reverse_order": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "If true, concatenates strings in reverse order (from last active input to first).\n"
                            "Example: text_1='A', text_2='B', text_3='C' → result will be 'C, B, A'\n"
                            "---\n"
                            "Если включено, объединяет строки в обратном порядке (от последнего активного входа к первому).\n"
                            "Пример: text_1='A', text_2='B', text_3='C' → результат будет 'C, B, A'"
                        ),
                    },
                ),
                "delimiter": (
                    [
                        "custom", ",", ".", "New Line", "New Line X2", "Space", ";", "|",
                        " - ", " | ", " :: ", " >>> ", " • ", " — ", " ... ",
                    ],
                    {
                        "default": ",",
                        "tooltip": (
                            "The separator used between concatenated strings. Choose from presets or 'custom'.\n\n"
                            "Presets:\n"
                            "- 'custom' : Enter your own delimiter in the next field. Supports escape sequences: \\n (newline), \\t (tab), \\r (carriage return).\n"
                            "- 'New Line X2' : Inserts a blank line (\\n\\n) for visual spacing between paragraphs.\n"
                            "---\n"
                            "Разделитель. 'custom' поддерживает \\n, \\t, \\r. 'New Line X2' вставляет \\n\\n."
                        ),
                    },
                ),
                "custom_delimiter": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": (
                            "Enter a custom delimiter if 'custom' is selected above. Supports \\n, \\t, \\r.\n"
                            "---\n"
                            "Пользовательский разделитель. Поддерживает \\n, \\t, \\r."
                        ),
                    },
                ),
                "clean_whitespace": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Strip leading/trailing whitespace.\n/ Удалить пробелы по краям."},
                ),
                "ignore_empty_strings": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Skip empty strings (including whitespace-only).\n/ Пропускать пустые строки (включая состоящие только из пробелов)."},
                ),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    OUTPUT_TOOLTIPS = ("Concatenated text result.\n/ Результат объединения.",)
    FUNCTION = "execute"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Dynamically concatenates 2 to 32 text strings. Inputs are managed via frontend JS.\n"
        "---\n"
        "Динамически объединяет от 2 до 32 строк. Входы управляются через JS на фронтенде."
    )

    def execute(
        self,
        number_of_inputs: int,
        reverse_order: bool,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
        **kwargs,
    ) -> Tuple[str]:
        texts = []
        for i in range(1, number_of_inputs + 1):
            texts.append(kwargs.get(f"text_{i}"))

        if reverse_order:
            texts.reverse()

        result = _concatenate_core(
            texts=texts,
            delimiter=delimiter,
            custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace,
            ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


# ================= СТАТИЧЕСКИЕ НОДЫ (X2, X4, X8) =================
class AGSoftTextConcatenateX2:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_static_input_types(2)

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
        reverse_order: bool,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        texts = [text_1, text_2]
        if reverse_order:
            texts.reverse()
            
        result = _concatenate_core(
            texts=texts, delimiter=delimiter, custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace, ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


class AGSoftTextConcatenateX4:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_static_input_types(4)

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
        reverse_order: bool,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        texts = [text_1, text_2, text_3, text_4]
        if reverse_order:
            texts.reverse()
            
        result = _concatenate_core(
            texts=texts, delimiter=delimiter, custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace, ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


class AGSoftTextConcatenateX8:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return _build_static_input_types(8)

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
        reverse_order: bool,
        delimiter: str,
        custom_delimiter: str,
        clean_whitespace: bool,
        ignore_empty_strings: bool,
    ) -> Tuple[str]:
        texts = [text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8]
        if reverse_order:
            texts.reverse()
            
        result = _concatenate_core(
            texts=texts, delimiter=delimiter, custom_delimiter=custom_delimiter,
            clean_whitespace=clean_whitespace, ignore_empty_strings=ignore_empty_strings,
        )
        return (result,)


# ================= РЕГИСТРАЦИЯ =================
NODE_CLASS_MAPPINGS = {
    "AGSoftTextConcatenate": AGSoftTextConcatenate,
    "AGSoftTextConcatenateX2": AGSoftTextConcatenateX2,
    "AGSoftTextConcatenateX4": AGSoftTextConcatenateX4,
    "AGSoftTextConcatenateX8": AGSoftTextConcatenateX8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftTextConcatenate": "📝🪡AGSoft Text Concatenate",
    "AGSoftTextConcatenateX2": "📝🪡AGSoft Text Concatenate X2",
    "AGSoftTextConcatenateX4": "📝🪡AGSoft Text Concatenate X4",
    "AGSoftTextConcatenateX8": "📝🪡AGSoft Text Concatenate X8",
}