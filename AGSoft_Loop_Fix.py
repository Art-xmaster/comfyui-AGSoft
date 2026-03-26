# AGSoft_Loop_Fix.py
# Автор: AGSoft
# Дата: 26 марта 2026 г.

# ───────────────────────────────
#   AGSoft_Loop_Fix_Float
# ───────────────────────────────

class AGSoft_Loop_Fix_Float:
    DESCRIPTION = "Provides 10 float input fields with selectable precision (1-2 decimals). All values including zero are passed.\nПредоставляет 10 полей для ввода float с выбором точности (1-2 знака). Все значения, включая ноль, передаются."

    @classmethod
    def INPUT_TYPES(cls):
        # Создаём 10 полей для ввода
        required = {
            "number_of_inputs": ("INT", {
                "default": 10,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "How many inputs to use (1-10).\nСколько входов использовать (1-10)."
            }),
            "precision": (["1", "2"], {
                "default": "2",
                "tooltip": "Number of decimal places (1 or 2).\nКоличество знаков после запятой (1 или 2)."
            })
        }
        
        # Добавляем 10 полей float_1 до float_10
        for i in range(1, 11):
            required[f"float_{i}"] = ("FLOAT", {
                "default": 0.0,
                "min": -10000.0,
                "max": 10000.0,
                "step": 0.01,
                "tooltip": f"Float value {i}.\nFloat значение {i}."
            })
        
        return {"required": required}

    RETURN_TYPES = ("FLOAT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "loop_floats"
    CATEGORY = "AGSoft/Utility"

    def loop_floats(self, number_of_inputs, precision, **kwargs):
        prec = int(precision)
        float_list = []
        
        for i in range(1, number_of_inputs + 1):
            key = f"float_{i}"
            value = kwargs.get(key, 0.0)
            
            # Округляем до указанной точности
            rounded_value = round(value, prec)
            float_list.append(rounded_value)
        
        return (float_list,)

    @classmethod
    def IS_CHANGED(cls, number_of_inputs, precision, **kwargs):
        return float("nan")

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Fix Float"


# ───────────────────────────────
#   AGSoft_Loop_Fix_Integer
# ───────────────────────────────

class AGSoft_Loop_Fix_Integer:
    DESCRIPTION = "Provides 10 integer input fields. All values including zero are passed.\nПредоставляет 10 полей для ввода integer. Все значения, включая ноль, передаются."

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "number_of_inputs": ("INT", {
                "default": 10,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "How many inputs to use (1-10).\nСколько входов использовать (1-10)."
            })
        }
        
        # Добавляем 10 полей int_1 до int_10
        for i in range(1, 11):
            required[f"int_{i}"] = ("INT", {
                "default": 0,
                "min": -10000,
                "max": 10000,
                "step": 1,
                "tooltip": f"Integer value {i}.\nInteger значение {i}."
            })
        
        return {"required": required}

    RETURN_TYPES = ("INT",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "loop_ints"
    CATEGORY = "AGSoft/Utility"

    def loop_ints(self, number_of_inputs, **kwargs):
        int_list = []
        
        for i in range(1, number_of_inputs + 1):
            key = f"int_{i}"
            value = kwargs.get(key, 0)
            int_list.append(value)
        
        return (int_list,)

    @classmethod
    def IS_CHANGED(cls, number_of_inputs, **kwargs):
        return float("nan")

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Fix Integer"


# ───────────────────────────────
#   AGSoft_Loop_Fix_Text
# ───────────────────────────────

class AGSoft_Loop_Fix_Text:
    DESCRIPTION = "Provides 10 text input fields. All values including empty strings are passed.\nПредоставляет 10 текстовых полей. Все значения, включая пустые строки, передаются."

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "number_of_inputs": ("INT", {
                "default": 10,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "How many inputs to use (1-10).\nСколько входов использовать (1-10)."
            })
        }
        
        # Добавляем 10 текстовых полей
        for i in range(1, 11):
            required[f"text_{i}"] = ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": f"Text value {i}.\nТекстовое значение {i}."
            })
        
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "loop_texts"
    CATEGORY = "AGSoft/Utility"

    def loop_texts(self, number_of_inputs, **kwargs):
        text_list = []
        
        for i in range(1, number_of_inputs + 1):
            key = f"text_{i}"
            value = kwargs.get(key, "")
            text_list.append(value if value is not None else "")
        
        return (text_list,)

    @classmethod
    def IS_CHANGED(cls, number_of_inputs, **kwargs):
        return float("nan")

    @classmethod
    def DISPLAY_NAME(cls):
        return "AGSoft Loop Fix Text"


# ───────────────────────────────
#   РЕГИСТРАЦИЯ НОД
# ───────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AGSoft_Loop_Fix_Float": AGSoft_Loop_Fix_Float,
    "AGSoft_Loop_Fix_Integer": AGSoft_Loop_Fix_Integer,
    "AGSoft_Loop_Fix_Text": AGSoft_Loop_Fix_Text,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Loop_Fix_Float": "🔢 AGSoft Loop Fix Float",
    "AGSoft_Loop_Fix_Integer": "🔢 AGSoft Loop Fix Integer",
    "AGSoft_Loop_Fix_Text": "📝 AGSoft Loop Fix Text",
}