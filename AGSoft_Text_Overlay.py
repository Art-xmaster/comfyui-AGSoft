# AGSoft Text Overlay
# Автор: AGSoft
# Дата: 02 ноября 2025 г.

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_MAPPING = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "olive": (128, 128, 0),
    "lime": (0, 255, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 255),
    "aqua": (0, 255, 255),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),
}

ALIGN_OPTIONS = ["center", "top", "bottom"]
JUSTIFY_OPTIONS = ["center", "left", "right"]
TEXT_ALIGN_OPTIONS = ["left", "center", "right"]

def get_color_list():
    return ["custom"] + list(COLOR_MAPPING.keys())

def get_color_values(color_name: str, hex_color: str, color_map: dict) -> tuple:
    if color_name == "custom":
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except:
            return (255, 255, 255)
    return color_map.get(color_name, (255, 255, 255))

def tensor2pil(image_tensor):
    return Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class AGSoftTextOverlay:
    @classmethod
    def INPUT_TYPES(s):
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        os.makedirs(font_dir, exist_ok=True)
        file_list = [f for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
        if not file_list:
            file_list = ["default (missing fonts)"]
        color_list = get_color_list()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to overlay text on.\nВходное изображение, на которое будет наложен текст."}),
                "text": ("STRING", {"multiline": True, "default": "text", "tooltip": "The text to be overlaid.\nТекст, который будет наложен на изображение."}),
                "font_name": (file_list, {"tooltip": "The font file to use for the text.\nФайл шрифта, который будет использован для текста."}),
                "font_size": ("INT", {"default": 50, "min": 1, "max": 1024, "tooltip": "The size of the font.\nРазмер шрифта."}),
                "font_color": (color_list, {"default": "white", "tooltip": "The color of the text. Choose 'custom' to use a specific hex color.\nЦвет текста. Выберите 'custom', чтобы использовать конкретный HEX-цвет."}),
                "text_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Opacity of the text (0 = transparent, 1 = opaque).\nПрозрачность текста (0 = прозрачный, 1 = непрозрачный)."}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20, "tooltip": "Width of the text stroke (outline).\nШирина обводки текста."}),
                "stroke_color": (color_list, {"default": "black", "tooltip": "Color of the text stroke.\nЦвет обводки текста."}),
                "shadow_x": ("INT", {"default": 0, "min": -100, "max": 100, "tooltip": "Horizontal offset of the shadow.\nГоризонтальное смещение тени."}),
                "shadow_y": ("INT", {"default": 0, "min": -100, "max": 100, "tooltip": "Vertical offset of the shadow.\nВертикальное смещение тени."}),
                # shadow_blur УДАЛЁН
                "shadow_color": (color_list, {"default": "black", "tooltip": "Color of the shadow.\nЦвет тени."}),
                "box_padding": ("INT", {"default": 20, "min": 0, "max": 200, "tooltip": "Padding around the text for the background box.\nОтступ вокруг текста для фона (рамки)."}),
                "box_color": (color_list, {"default": "black", "tooltip": "Background color behind the text.\nЦвет фона за текстом."}),
                "box_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Opacity of the background box (0 = transparent, 1 = opaque).\nПрозрачность фона за текстом."}),
                "box_radius": ("INT", {"default": 20, "min": 0, "max": 100, "tooltip": "Corner radius of the background box.\nРадиус скругления углов фона."}),
                "align": (ALIGN_OPTIONS, {"tooltip": "Vertical alignment of the text block (top, center, bottom).\nВертикальное выравнивание текстового блока (вверх, по центру, вниз)."}),
                "justify": (JUSTIFY_OPTIONS, {"tooltip": "Horizontal alignment of the text block (left, center, right).\nГоризонтальное выравнивание текстового блока (влево, по центру, вправо)."}),
                "text_align": (TEXT_ALIGN_OPTIONS, {"default": "left", "tooltip": "Text alignment inside the block (left, center, right).\nВыравнивание текста внутри блока (влево, по центру, вправо)."}),
                "margins": ("INT", {"default": 0, "min": -1024, "max": 1024, "tooltip": "Margins from the edges of the image.\nОтступы от краев изображения."}),
                "line_spacing": ("INT", {"default": 10, "min": -1024, "max": 1024, "tooltip": "Spacing between lines of text.\nИнтервал между строками текста."}),
                "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "X position of the text block.\nГоризонтальная позиция текстового блока."}),
                "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "Y position of the text block.\nВертикальная позиция текстового блока."}),
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1, "tooltip": "Rotation angle of the text block in degrees.\nУгол поворота текстового блока в градусах."}),
            },
            "optional": {
                "font_color_hex": ("STRING", {"multiline": False, "default": "#FFFFFF", "tooltip": "HEX color code for custom font color.\nHEX-код для кастомного цвета шрифта."}),
                "stroke_color_hex": ("STRING", {"multiline": False, "default": "#000000", "tooltip": "HEX color code for custom stroke color.\nHEX-код для кастомного цвета обводки."}),
                "shadow_color_hex": ("STRING", {"multiline": False, "default": "#000000", "tooltip": "HEX color code for custom shadow color.\nHEX-код для кастомного цвета тени."}),
                "box_color_hex": ("STRING", {"multiline": False, "default": "#000000", "tooltip": "HEX color code for custom box background color.\nHEX-код для кастомного цвета фона."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_text"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = "Overlays text onto an image with customizable font, color, alignment, stroke, shadow, background box, and opacity.\nНакладывает текст на изображение с настраиваемым шрифтом, цветом, выравниванием, обводкой, тенью, фоном и прозрачностью."

    def overlay_text(self, image, text, font_name, font_size,
                     font_color, text_opacity,
                     stroke_width, stroke_color,
                     shadow_x, shadow_y, shadow_color,
                     box_padding, box_color, box_opacity, box_radius,
                     margins, line_spacing,
                     position_x, position_y,
                     align, justify, text_align,
                     rotation_angle=0.0,
                     font_color_hex='#FFFFFF',
                     stroke_color_hex='#000000',
                     shadow_color_hex='#000000',
                     box_color_hex='#000000'):
        text_color = get_color_values(font_color, font_color_hex, COLOR_MAPPING)
        stroke_col = get_color_values(stroke_color, stroke_color_hex, COLOR_MAPPING)
        shadow_col = get_color_values(shadow_color, shadow_color_hex, COLOR_MAPPING)
        box_col = get_color_values(box_color, box_color_hex, COLOR_MAPPING)

        output_images = []
        for i in range(image.shape[0]):
            img = image[i]
            back_image = tensor2pil(img).convert('RGBA')
            W, H = back_image.size

            # --- ШАГ 1: Создаём временный слой и рисуем текст в (0, 0) ---
            temp_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_layer)

            font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = ImageFont.load_default()

            lines = text.split('\n')
            y_offset = 0
            line_widths = []
            for line in lines:
                bbox = temp_draw.textbbox((0, y_offset), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_widths.append(line_width)
                temp_draw.text((0, y_offset), line, fill=(255, 255, 255, 255), font=font,
                               stroke_width=stroke_width, stroke_fill=(255, 255, 255, 255))
                line_height = bbox[3] - bbox[1]
                y_offset += line_height + line_spacing

            max_line_width = max(line_widths) if line_widths else 0

            # --- ШАГ 2: Получаем маску и её координаты ---
            text_mask = temp_layer.convert('L')
            bbox = text_mask.getbbox()
            if bbox is None:
                output_images.append(pil2tensor(back_image.convert('RGB')))
                continue

            min_x, min_y, max_x, max_y = bbox
            mask_width = max_x - min_x
            mask_height = max_y - min_y

            # --- ШАГ 3: Определяем целевую позицию маски на изображении ---
            area_left = margins
            area_right = W - margins
            area_top = margins
            area_bottom = H - margins

            if justify == "left":
                target_x = area_left
            elif justify == "right":
                target_x = area_right - mask_width
            else:  # center
                target_x = area_left + (area_right - area_left - mask_width) // 2

            if align == "top":
                target_y = area_top
            elif align == "bottom":
                target_y = area_bottom - mask_height
            else:  # center
                target_y = area_top + (area_bottom - area_top - mask_height) // 2

            # --- ШАГ 4: Применяем position_x/y как смещение маски ---
            final_x = target_x + position_x
            final_y = target_y + position_y

            # --- ШАГ 5: Сдвигаем весь слой так, чтобы маска оказалась в (final_x, final_y) ---
            dx = final_x - min_x
            dy = final_y - min_y
            shifted_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            shifted_layer.paste(temp_layer, (dx, dy), temp_layer)

            # --- ШАГ 6: Применяем вращение, если нужно ---
            rotated_layer = shifted_layer
            if abs(rotation_angle) > 0.01:
                center_x = final_x + mask_width // 2
                center_y = final_y + mask_height // 2
                rotated_layer = shifted_layer.rotate(
                    -rotation_angle,
                    center=(center_x, center_y),
                    resample=Image.BICUBIC,
                    expand=False
                )

            # --- ШАГ 7: Создаём финальный слой с цветом, прозрачностью и тенью ---
            text_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))

            # Тень
            if shadow_x != 0 or shadow_y != 0:
                shadow_temp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_temp)
                y_off = 0
                for line in lines:
                    shadow_draw.text((0, y_off), line, fill=(*shadow_col, int(255 * text_opacity)), font=font)
                    bbox = shadow_draw.textbbox((0, y_off), line, font=font)
                    y_off += (bbox[3] - bbox[1]) + line_spacing
                shadow_shifted = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                shadow_shifted.paste(shadow_temp, (dx, dy), shadow_temp)
                if abs(rotation_angle) > 0.01:
                    shadow_shifted = shadow_shifted.rotate(
                        -rotation_angle,
                        center=(final_x + mask_width // 2, final_y + mask_height // 2),
                        resample=Image.BICUBIC,
                        expand=False
                    )
                shadow_final = Image.new('RGBA', (W, H), (0, 0, 0, 0))
                shadow_final.paste(shadow_shifted, (shadow_x, shadow_y), shadow_shifted)
                text_layer = Image.alpha_composite(text_layer, shadow_final)

            # Основной текст
            main_temp = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            main_draw = ImageDraw.Draw(main_temp)
            y_off = 0
            for line in lines:
                # Применяем text_align: смещение строки внутри блока
                if text_align == "left":
                    x_offset = 0
                elif text_align == "right":
                    x_offset = max_line_width - line_widths[len(lines) - len(lines) + lines.index(line)]  # просто line_widths[i]
                else:  # center
                    x_offset = (max_line_width - line_widths[lines.index(line)]) // 2
                main_draw.text((x_offset, y_off), line, fill=(*text_color, int(255 * text_opacity)), font=font,
                               stroke_width=stroke_width, stroke_fill=(*stroke_col, int(255 * text_opacity)))
                bbox = main_draw.textbbox((x_offset, y_off), line, font=font)
                y_off += (bbox[3] - bbox[1]) + line_spacing
            main_shifted = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            main_shifted.paste(main_temp, (dx, dy), main_temp)
            if abs(rotation_angle) > 0.01:
                main_shifted = main_shifted.rotate(
                    -rotation_angle,
                    center=(final_x + mask_width // 2, final_y + mask_height // 2),
                    resample=Image.BICUBIC,
                    expand=False
                )
            text_layer = Image.alpha_composite(text_layer, main_shifted)

            # --- ШАГ 8: Обновляем координаты маски после вращения ---
            rotated_mask = text_layer.convert('L')
            rotated_bbox = rotated_mask.getbbox()
            if rotated_bbox:
                min_x_final, min_y_final, max_x_final, max_y_final = rotated_bbox
            else:
                min_x_final = final_x
                min_y_final = final_y
                max_x_final = final_x + mask_width
                max_y_final = final_y + mask_height

            # --- ШАГ 9: Рисуем рамку с возможным скруглением ---
            box_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            if box_padding > 0 and box_opacity > 0:
                box_draw = ImageDraw.Draw(box_layer)
                box_alpha = int(255 * box_opacity)
                box_coords = [
                    min_x_final - box_padding,
                    min_y_final - box_padding,
                    max_x_final + box_padding,
                    max_y_final + box_padding
                ]
                if box_radius > 0:
                    box_draw.rounded_rectangle(
                        box_coords,
                        radius=box_radius,
                        fill=(*box_col, box_alpha)
                    )
                else:
                    box_draw.rectangle(
                        box_coords,
                        fill=(*box_col, box_alpha)
                    )

            # --- ШАГ 10: Собираем результат ---
            result = Image.alpha_composite(back_image, box_layer)
            result = Image.alpha_composite(result, text_layer).convert('RGB')
            output_images.append(pil2tensor(result))

        final_image = torch.cat(output_images, dim=0)
        return (final_image,)

NODE_CLASS_MAPPINGS = {"AGSoft Text Overlay": AGSoftTextOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft Text Overlay": "AGSoft Text Overlay"}