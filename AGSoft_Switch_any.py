# AGSoft_Switch_any.py
# Автор: AGSoft
# Дата: 14 июня 2026 г.
# Описание: Нода-переключатель с динамическими входами для выбора одного из множества входов любого типа

from __future__ import annotations
from typing import Any, Dict, Tuple, List


class AGSoftSwitchAny:
    """
    Нода-переключатель с динамическими входами.
    
    Позволяет выбрать один из N входов (любого типа) и передать его на выход.
    Количество входов настраивается динамически через параметр number_of_inputs.
    Выбор входа осуществляется через комбо бокс с выпадающим списком.
    """
    
    DESCRIPTION = """A universal switch node with dynamic inputs that accepts ANY data type.

Use this node to route different types of data (images, text, latents, models, audio, video, etc.) through a single output based on a selector.

Features:
- Dynamic number of inputs (2 to 30)
- Combo box for easy input selection
- Supports ALL data types (images, text, latents, models, audio, video, etc.)
- Two inputs are always required (minimum)

The selected input is passed to the output unchanged.

---

Универсальная нода-переключатель с динамическими входами, принимающая ЛЮБОЙ тип данных.

Используйте эту ноду для маршрутизации различных типов данных (изображения, текст, латенты, модели, аудио, видео и т.д.) через один выход на основе селектора.

Возможности:
- Динамическое количество входов (от 2 до 30)
- Комбо бокс для удобного выбора входа
- Поддерживает ВСЕ типы данных (изображения, текст, латенты, модели, аудио, видео и т.д.)
- Два входа всегда обязательны (минимум)

Выбранный вход передается на выход без изменений."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Определяет типы входных параметров ноды.
        
        Returns:
            Dict[str, Any]: Словарь с обязательными (required) параметрами.
        """
        # Генерируем все возможные опции (от input_1 до input_30)
        # Это нужно для прохождения валидации ComfyUI
        # JavaScript будет показывать только нужное количество опций
        all_options = [f"input_{i}" for i in range(1, 31)]
        
        return {
            "required": {
                # Комбо бокс для выбора, какой вход вывести
                "selected_input": (all_options, {
                    "default": "input_1",
                    "tooltip": """Select which input to pass to the output from the dropdown list.
The list shows all available inputs based on number_of_inputs setting.
Example: select "input_2" → output receives data from input_2
---
Выберите из выпадающего списка, какой вход передать на выход.
Список показывает все доступные входы на основе настройки number_of_inputs.
Пример: выберите "input_2" → выход получает данные из input_2"""
                }),
                
                # Параметр для указания количества входов
                "number_of_inputs": ("INT", {
                    "default": 2,  # По умолчанию 2 входа (минимум)
                    "min": 2,  # Минимум 2 входа (обязательное требование)
                    "max": 30,  # Максимум 30 входов
                    "step": 1,
                    "tooltip": """Number of dynamic inputs to create (minimum 2, maximum 30).
Each input will be named input_1, input_2, input_3, etc.
The combo box above will automatically update to show only available inputs.
Example: number_of_inputs=5 → creates input_1, input_2, input_3, input_4, input_5
---
Количество динамических входов для создания (минимум 2, максимум 30).
Каждый вход будет называться input_1, input_2, input_3 и т.д.
Комбо бокс выше автоматически обновится, показав только доступные входы.
Пример: number_of_inputs=5 → создаются input_1, input_2, input_3, input_4, input_5"""
                }),
            },
        }

    # Типы возвращаемых данных - один выход любого типа
    RETURN_TYPES = ("*",)
    
    # Имена возвращаемых значений (для отображения в интерфейсе)
    RETURN_NAMES = ("output",)
    
    # Имя метода, который будет выполняться при обработке ноды
    FUNCTION = "switch_input"
    
    # Категория ноды (для организации в меню ComfyUI)
    CATEGORY = "AGSoft/Utility"
    
    def switch_input(self, selected_input: str, number_of_inputs: int, **kwargs) -> Tuple[Any]:
        """
        Основной метод выполнения ноды.
        
        Args:
            selected_input (str): Имя выбранного входа (например, "input_1", "input_2")
            number_of_inputs (int): Общее количество входов
            **kwargs: Словарь со всеми динамическими входами (input_1, input_2, ...)
        
        Returns:
            Tuple[Any]: Кортеж с одним элементом - выбранным входом
        """
        # Проверяем, существует ли такой вход в kwargs
        if selected_input not in kwargs:
            # Если вход не найден, используем первый доступный
            available_inputs = [k for k in kwargs.keys() if k.startswith("input_")]
            if available_inputs:
                selected_input = available_inputs[0]
            else:
                # Если совсем нет входов, возвращаем None
                return (None,)
        
        # Получаем значение выбранного входа
        selected_value = kwargs[selected_input]
        
        # Возвращаем кортеж с одним элементом (требование ComfyUI)
        return (selected_value,)


# ─────────────────────────────────────────────────────────────────────────────
# РЕГИСТРАЦИЯ НОДЫ В ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AGSoft_Switch_any": AGSoftSwitchAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Switch_any": "🔀 AGSoft Switch Any",
}