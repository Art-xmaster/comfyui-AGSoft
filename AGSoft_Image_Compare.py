# AGSoft_Image_Compare.py
# Автор: AGSoft
# Дата: 14 июня 2026 г.
# Описание: Улучшенная нода для визуального сравнения двух изображений с множеством режимов, зумом и панорамированием.
import logging
from typing import Dict, Any, Tuple, Optional
import torch
from nodes import PreviewImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AGSoftImageCompare(PreviewImage):
    """
    Advanced node for visual comparison of two images with multiple modes, zoom, and pan.
    Улучшенная нода для визуального сравнения двух изображений с множеством режимов, зумом и панорамированием.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (
                    ["Slider", "Difference", "Side-by-Side"],
                    {
                        "default": "Slider",
                        "tooltip": """Comparison mode:
Slider: Interactive slider to reveal image_2 over image_1.
Difference: Shows pixel differences (black means identical).
Side-by-Side: Splits the view vertically.
Режим сравнения:
Slider: Интерактивный слайдер для проявления image_2 поверх image_1.
Difference: Показывает различия пикселей (черный цвет означает идентичность).
Side-by-Side: Делит вертикально пополам."""
                    }
                ),
                "zoom": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": """Zoom level for both images. 1.0 is 100% (fit to node).
Уровень масштабирования обоих изображений. 1.0 — это 100% (вписать в ноду)."""
                    }
                ),
                "pan_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -5000,
                        "max": 5000,
                        "step": 1,
                        "tooltip": """Horizontal pan offset in pixels. Useful with zoom > 1.0.
Смещение по горизонтали в пикселях. Полезно при зуме > 1.0."""
                    }
                ),
                "pan_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -5000,
                        "max": 5000,
                        "step": 1,
                        "tooltip": """Vertical pan offset in pixels. Useful with zoom > 1.0.
Смещение по вертикали в пикселях. Полезно при зуме > 1.0."""
                    }
                ),
            },
            "optional": {
                "image_1": (
                    "IMAGE",
                    {
                        "tooltip": """First image (Background layer).
Первое изображение (Фоновый слой)."""
                    }
                ),
                "image_2": (
                    "IMAGE",
                    {
                        "tooltip": """Second image (Foreground layer).
Второе изображение (Передний план)."""
                    }
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2")
    FUNCTION = "compare_images"
    CATEGORY = "AGSoft/Image"
    OUTPUT_NODE = True
    DESCRIPTION = """Advanced image comparison tool. Supports Slider, Difference, and Side-by-Side modes. Includes Zoom and Pan widgets to inspect details. Press 'Space' while hovering the node to temporarily hide the second image.
Продвинутый инструмент сравнения изображений. Поддерживает режимы Slider, Difference и Side-by-Side. Включает виджеты Zoom и Pan для изучения деталей. Нажмите 'Space' (Пробел), наведя курсор на ноду, чтобы временно скрыть второе изображение."""

    def compare_images(
        self,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        mode: str = "Slider",
        zoom: float = 1.0,
        pan_x: int = 0,
        pan_y: int = 0,
        filename_prefix: str = "AGSoft_Compare_",
        prompt: Optional[Any] = None,
        extra_pnginfo: Optional[Any] = None
    ) -> Dict[str, Any]:
        
        ui_data: Dict[str, Any] = {"image_1": [], "image_2": []}
        
        if image_1 is not None:
            try:
                res_1 = self.save_images(image_1, filename_prefix + "1_", prompt, extra_pnginfo)
                ui_data["image_1"] = res_1.get("ui", {}).get("images", [])
                logger.info("Successfully processed image_1 for comparison.")
            except Exception as e:
                logger.error(f"Error saving image_1: {e}")
                
        if image_2 is not None:
            try:
                res_2 = self.save_images(image_2, filename_prefix + "2_", prompt, extra_pnginfo)
                ui_data["image_2"] = res_2.get("ui", {}).get("images", [])
                logger.info("Successfully processed image_2 for comparison.")
            except Exception as e:
                logger.error(f"Error saving image_2: {e}")

        return {
            "result": (image_1, image_2),
            "ui": ui_data
        }


NODE_CLASS_MAPPINGS = {
    "AGSoftImageCompare": AGSoftImageCompare
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftImageCompare": "↔️ AGSoft Image Compare"
}