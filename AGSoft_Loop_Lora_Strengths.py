# AGSoft_Loop_Lora_Strengths.py
# Автор: AGSoft
# Дата: 22 ноября 2025 г.

class AGSoftLoopLoraStrengths:
    """
    Generates synchronized lists of LoRA strength values and formatted text labels.
    Slots with strength exactly 0.0 are skipped.
    All other values are formatted without loss of precision (e.g., 0.001 stays 0.001).
    
    Генерирует синхронизированные списки: числовые значения силы LoRA и текстовые метки.
    Слоты со значением точно 0.0 пропускаются.
    Все остальные значения форматируются без потери точности (напр., 0.001 остаётся 0.001).
    """
    DESCRIPTION = (
        "Outputs two synchronized lists:\n"
        "- FLOAT: LoRA strength values (excluding exactly 0.0) for batch processing\n"
        "- STRING: formatted as 'lora_strength = X' with full precision (e.g., 0.001, 1.5, -2)\n"
        "Only values exactly 0.0 are skipped.\n\n"
        "Возвращает два синхронизированных списка:\n"
        "- FLOAT: значения силы LoRA (исключая точно 0.0) для пакетной обработки\n"
        "- STRING: в формате 'lora_strength = X' с полной точностью (например, 0.001, 1.5, -2)\n"
        "Пропускаются только значения, точно равные 0.0."
    )

    @classmethod
    def INPUT_TYPES(cls):
        float_config = {
            "default": 0.0,
            "min": -100.0,
            "max": 100.0,
            "step": 0.01,
            "tooltip": (
                "LoRA strength for diffusion model. Set to 0.0 to skip this slot.\n"
                "Сила LoRA для модели диффузии. Установите 0.0, чтобы пропустить этот слот."
            )
        }
        inputs = {"required": {}}
        for i in range(1, 21):
            default_val = 1.0 if i == 1 else 0.0
            inputs["required"][f"strength_model_{i}"] = ("FLOAT", {
                **float_config,
                "default": default_val,
                "tooltip": (
                    f"Strength for slot {i}. 0.0 = skip.\n"
                    f"Сила для слота {i}. 0.0 = пропустить."
                )
            })
        return inputs

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("strengths", "strengths_text")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "get_strengths"
    CATEGORY = "AGSoft/Utility"

    def _format_val(self, val: float) -> str:
        s = str(val)
        if s.endswith('.0'):
            s = s[:-2]
        return s

    def get_strengths(self, **kwargs):
        strengths = []
        texts = []
        for i in range(1, 21):
            val = kwargs.get(f"strength_model_{i}", 0.0)
            if abs(val) < 1e-8:  # skip only exact 0.0
                continue
            strengths.append(val)
            formatted = self._format_val(val)
            texts.append(f"lora_strength = {formatted}")
        return (strengths, texts)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        active = tuple(
            kwargs[f"strength_model_{i}"]
            for i in range(1, 21)
            if abs(kwargs[f"strength_model_{i}"]) > 1e-8
        )
        return hash(active) % (2**64)


NODE_CLASS_MAPPINGS = {
    "AGSoft Loop Lora Strengths": AGSoftLoopLoraStrengths
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft Loop Lora Strengths": "AGSoft Loop Lora Strengths"
}