"""
==============================================================================
AGSoft_MiniMax_H3.py
==============================================================================
Ноды / Nodes: 🎬AGSoft MiniMax H3 Ref2V, 🎬AGSoft MiniMax H3 I2V
Версия / Version: v09.27
Описание / Description:
Пара нод кондиционирования для MiniMax H3 (ref2va / t2va+fl2va) с полным
локальным воспроизведением логики нативных нодов ComfyUI и калькулятором
параметров AGSoft: размер кратен 32, кадры — последовательность 17N+5
(5, 22, 39, ...), FPS фиксирован 24, длительность задаётся в секундах
(Seconds) или кадрах (Frames).
Ref2V кодирует промпт вместе с презентацией референсов (изображения,
видео, парные аудиодорожки, отдельные аудио) и передаёт DiT блоки
minimax_refs; I2V работает с текстом и опциональными первым/последним
кадрами (minimax_keyframes). Обе ноды отдают пустой совместный AV-латент
(видео 24 канала + аудио 32 канала) рассчитанного размера.
---
A pair of conditioning nodes for MiniMax H3 (ref2va / t2va+fl2va) with a
full local re-implementation of the native ComfyUI nodes' logic plus the
AGSoft parameter calculator: sizes are multiples of 32, frame count follows
the 17N+5 sequence (5, 22, 39, ...), FPS is fixed at 24, duration is set in
seconds (Seconds) or frames (Frames).
Ref2V encodes the prompt together with the reference presentation (images,
videos, paired soundtracks, standalone audio) and passes minimax_refs blocks
to the DiT; I2V works with text and optional first/last keyframes
(minimax_keyframes). Both nodes output an empty joint AV latent (24-channel
video + 32-channel audio) of the calculated size.
Возможности / Features:
⚡ Полная локальная логика H3: кодирование рефов через VAE/audio_vae,
   токенизация с minimax_ref_items, кондиционирование minimax_refs /
   minimax_keyframes, пустой AV-латент (NestedTensor видео+аудио).
   Full local H3 logic: ref encoding via VAE/audio_vae, tokenization with
   minimax_ref_items, minimax_refs / minimax_keyframes conditioning, empty
   AV latent (video+audio NestedTensor).
⚡ Калькулятор: Preset (40 размеров) / Custom / Megapixels + Seconds/Frames,
   кратность 32, кадры 17N+5, FPS 24, инверсия сторон в Preset/Custom.
   Calculator: Preset (40 sizes) / Custom / Megapixels + Seconds/Frames,
   multiples of 32, 17N+5 frames, FPS 24, orientation invert in Preset/Custom.
⚡ Ref2V: динамические входы рефов по группам (image до 10, video до 3,
   video_audio до 3, audio до 10): подключение к последнему слоту группы
   создаёт следующий, отключение последнего подключённого тримит лишний.
   Ref2V: dynamic ref inputs per group (image up to 10, video up to 3,
   video_audio up to 3, audio up to 10): connecting the last slot of a group
   creates the next one, disconnecting the last connected trims the extra.
⚡ I2V: first_frame (stretch до холста) и last_frame (cover-crop с центром),
   режим чистого text-to-video без кадров.
   I2V: first_frame (stretch to canvas) and last_frame (center cover-crop),
   plus a pure text-to-video mode without frames.
⚡ Автоскрытие неиспользуемых виджетов калькулятора по режимам (JS),
   высота ноды подгоняется под видимые виджеты, prompt растягивается до низа.
   Auto-hiding of unused calculator widgets per mode (JS), node height fits
   the visible widgets, the prompt stretches to the bottom.
⚡ Живая инфострока (DOM-виджет 18px между последним combo и prompt):
   ⚙ W×H • сек • кадры; пересчитывается при любом изменении виджетов.
   Live info line (18px DOM widget between the last combo and the prompt):
   ⚙ W×H • sec • frames; recomputed on any widget change.
⚡ ref_image_size строго match/max — обязательные значения combo H3.
   ref_image_size strictly match/max — the required H3 combo values.
Автор / Author: AGSoft
Дата / Date: 27.09.2026
==============================================================================
"""
import math
import torch
import logging
import comfy.utils
import comfy.audio
import comfy.model_management
import comfy.nested_tensor
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print("[AGSoft MiniMax H3] v09.27 loaded (Ref2V + I2V, unified detailed docs and tooltips)")
#========================================================================
# Константы H3 / H3 constants
#========================================================================
FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
#========================================================================
# Утилиты калькулятора / Calculator utilities
#========================================================================
def fit_to_multiple(value: int, multiple: int = 32) -> int:
    """
    Подгоняет значение к ближайшему большему числу, кратному multiple.
    Rounds value UP to the nearest multiple of multiple.
    100 → 128, 128 → 128, 33 → 64
    """
    return ((value + multiple - 1) // multiple) * multiple
def fit_length_to_17n5(value: int) -> int:
    """
    Подгоняет число под последовательность: 5, 22, 39, 56, 73... (17*N+5).
    Aligns value UP to the sequence: 5, 22, 39, 56, 73... (17*N+5).
    5 → 5, 100 → 107, 240 → 243
    """
    v = max(5, int(value))
    return v + (5 - v % 17) % 17
#========================================================================
# Утилиты H3 / H3 utilities
#========================================================================
def align_frame_count(n):
    """Выравнивает кадр вверх под 17N+5 / Aligns frame UP to 17N+5"""
    while n % 17 != 5:
        n += 1
    return n
def video_latent_t(frame_count):
    """Temporal-размер видео-латента H3 / H3 video latent temporal size"""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
def temporal_shape(length):
    """
    Возвращает (frame_count, latent_t, audio_t) для совместного AV-латента.
    Returns (frame_count, latent_t, audio_t) for the joint AV latent.
    """
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)
def adapt_canvas(width, height):
    """
    Холст референс-видео: короткая сторона 768, потолок площади 768*1344,
    округление по осям до 32 / Reference video canvas: 768 short edge,
    768*1344 area cap, per-axis round to 32.
    """
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    )
def _resize(image, width, height, crop):
    """Ресайз тензора изображения lanczos'ом / Resize image tensor with lanczos"""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)
def _encode_ref_audio(audio_vae, audio):
    """
    Ресемплирует аудио до частоты audio_vae и кодирует в латент (32 канала).
    Resamples audio to the audio_vae rate and encodes it into a 32ch latent.
    """
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = comfy.audio.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]
def _empty_av_latent(width, height, length, batch_size=1):
    """
    Пустой совместный AV-латент H3: видео [B,24,T,H//16,W//16] +
    аудио [B,32,2,A] в NestedTensor; возвращает (latent, frame_count).
    Empty joint H3 AV latent: video [B,24,T,H//16,W//16] + audio
    [B,32,2,A] as NestedTensor; returns (latent, frame_count).
    """
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count
#========================================================================
# Пресеты размеров / Size presets
#========================================================================
PRESET_LIST = [
    "512×512 (1:1) ", "576×576 (1:1) ", "640×640 (1:1) ", "704×704 (1:1) ",
    "768×768 (1:1) ", "832×832 (1:1) ", "896×896 (1:1) ", "960×960 (1:1) ",
    "1024×1024 (1:1) ", "1280×1280 (1:1) ",
    "480×320 (3:2) ", "576×384 (3:2) ", "672×448 (3:2) ", "768×512 (3:2) ",
    "864×576 (3:2) ", "960×640 (3:2) ", "1056×704 (3:2) ", "1152×768 (3:2) ",
    "1248×832 (3:2) ", "1536×1024 (3:2) ",
    "512×384 (4:3) ", "640×480 (4:3) ", "768×576 (4:3) ", "896×672 (4:3) ",
    "1024×768 (4:3) ", "1152×864 (4:3) ", "1280×960 (4:3) ", "1408×1056 (4:3) ",
    "1536×1152 (4:3) ", "2048×1536 (4:3) ",
    "512×288 (16:9) ", "640×352 (16:9) ", "768×448 (16:9) ", "896×512 (16:9) ",
    "1024×576 (16:9) ", "1152×640 (16:9) ", "1280×704 (16:9) ", "1536×864 (16:9) ",
    "1920×1088 (16:9) ", "2048×1152 (16:9) ",
]
PRESET_MAP = {
    "512×512 (1:1) ": (512, 512), "576×576 (1:1) ": (576, 576),
    "640×640 (1:1) ": (640, 640), "704×704 (1:1) ": (704, 704),
    "768×768 (1:1) ": (768, 768), "832×832 (1:1) ": (832, 832),
    "896×896 (1:1) ": (896, 896), "960×960 (1:1) ": (960, 960),
    "1024×1024 (1:1) ": (1024, 1024), "1280×1280 (1:1) ": (1280, 1280),
    "480×320 (3:2) ": (480, 320), "576×384 (3:2) ": (576, 384),
    "672×448 (3:2) ": (672, 448), "768×512 (3:2) ": (768, 512),
    "864×576 (3:2) ": (864, 576), "960×640 (3:2) ": (960, 640),
    "1056×704 (3:2) ": (1056, 704), "1152×768 (3:2) ": (1152, 768),
    "1248×832 (3:2) ": (1248, 832), "1536×1024 (3:2) ": (1536, 1024),
    "512×384 (4:3) ": (512, 384), "640×480 (4:3) ": (640, 480),
    "768×576 (4:3) ": (768, 576), "896×672 (4:3) ": (896, 672),
    "1024×768 (4:3) ": (1024, 768), "1152×864 (4:3) ": (1152, 864),
    "1280×960 (4:3) ": (1280, 960), "1408×1056 (4:3) ": (1408, 1056),
    "1536×1152 (4:3) ": (1536, 1152), "2048×1536 (4:3) ": (2048, 1536),
    "512×288 (16:9) ": (512, 288), "640×352 (16:9) ": (640, 352),
    "768×448 (16:9) ": (768, 448), "896×512 (16:9) ": (896, 512),
    "1024×576 (16:9) ": (1024, 576), "1152×640 (16:9) ": (1152, 640),
    "1280×704 (16:9) ": (1280, 704), "1536×864 (16:9) ": (1536, 864),
    "1920×1088 (16:9) ": (1920, 1088), "2048×1152 (16:9) ": (2048, 1152),
}
ASPECT_RATIOS = ["1:1 ", "3:2 ", "2:3 ", "4:3 ", "3:4 ", "16:9 ", "9:16 ", "21:9 ", "9:21 "]
#========================================================================
# Подробные двуязычные тултипы калькулятора / Detailed bilingual tooltips
#========================================================================
_T = lambda en, ru: en + "\n---\n" + ru
CALC_TOOLTIPS = {
    "mode": _T(
        "Frame size selection mode. Unused widgets of the inactive modes are hidden automatically (JS).\n\n"
        "• Preset — choose from 40 predefined sizes grouped by aspect ratio (1:1, 3:2, 4:3, 16:9). All sizes are already multiples of 32. Invert orientation supported.\n"
        "• Custom — manually enter width and height in pixels; values are rounded UP to the nearest multiple of 32 (100 → 128). Invert orientation supported.\n"
        "• Megapixels — specify target resolution in megapixels plus aspect ratio; width/height are computed automatically and rounded to multiples of 32. Invert orientation NOT applied (the ratio defines orientation).\n",
        "Режим выбора размера кадра. Неиспользуемые виджеты неактивных режимов скрываются автоматически (JS).\n\n"
        "• Preset — выбор из 40 готовых размеров по соотношениям сторон (1:1, 3:2, 4:3, 16:9). Все размеры уже кратны 32. Инверсия сторон работает.\n"
        "• Custom — ручной ввод ширины и высоты в пикселях; значения округляются ВВЕРХ до кратности 32 (100 → 128). Инверсия сторон работает.\n"
        "• Megapixels — целевое разрешение в мегапикселях плюс соотношение сторон; ширина/высота считаются автоматически и округляются до кратности 32. Инверсия НЕ применяется (соотношение задаёт ориентацию)."
    ),
    "preset": _T(
        "Predefined frame size with an aspect ratio label, format WIDTH×HEIGHT (ASPECT).\n\n"
        "Examples: 1024×1024 (1:1) square; 1280×704 (16:9) HD landscape; 1536×1024 (3:2) photo.\n"
        "All sizes are multiples of 32 — safe for MiniMax H3. Use invert_orientation to swap width↔height (e.g. 1280×704 → 704×1280 vertical).\n",
        "Готовый размер кадра с меткой соотношения сторон, формат ШИРИНА×ВЫСОТА (СООТНОШЕНИЕ).\n\n"
        "Примеры: 1024×1024 (1:1) квадрат; 1280×704 (16:9) HD горизонталь; 1536×1024 (3:2) фото.\n"
        "Все размеры кратны 32 — безопасно для MiniMax H3. Используйте invert_orientation для смены ширины↔высоты (например 1280×704 → 704×1280 вертикально)."
    ),
    "invert_orientation": _T(
        "Swap width and height values.\n\n"
        "Use cases: quick vertical video from a landscape preset (1280×704 → 704×1280); swapping custom dimensions without re-entering values.\n"
        "Works in Preset and Custom modes; NOT applied in Megapixels (the aspect ratio already defines orientation).\n",
        "Поменять ширину и высоту местами.\n\n"
        "Сценарии: быстрое вертикальное видео из горизонтального пресета (1280×704 → 704×1280); смена своих размеров без повторного ввода.\n"
        "Работает в режимах Preset и Custom; НЕ применяется в Megapixels (соотношение уже задаёт ориентацию)."
    ),
    "custom_width": _T(
        "Custom frame width in pixels (Custom mode only).\n\n"
        "• Rounded UP to the nearest multiple of 32: 100 → 128, 1900 → 1920.\n"
        "• Range 64–8192, step 32 for convenience.\n",
        "Своя ширина кадра в пикселях (только режим Custom).\n\n"
        "• Округляется ВВЕРХ до кратности 32: 100 → 128, 1900 → 1920.\n"
        "• Диапазон 64–8192, шаг 32 для удобства."
    ),
    "custom_height": _T(
        "Custom frame height in pixels (Custom mode only).\n\n"
        "• Rounded UP to the nearest multiple of 32: 100 → 128, 1080 → 1088.\n"
        "• Range 64–8192, step 32 for convenience.\n",
        "Своя высота кадра в пикселях (только режим Custom).\n\n"
        "• Округляется ВВЕРХ до кратности 32: 100 → 128, 1080 → 1088.\n"
        "• Диапазон 64–8192, шаг 32 для удобства."
    ),
    "megapixels_value": _T(
        "Target resolution in megapixels (Megapixels mode only), range 0.1–8.0, step 0.01.\n\n"
        "Width/height are computed from the value plus aspect_ratio and rounded to multiples of 32.\n"
        "Examples: 0.15 MP + 16:9 → ~512×288; 1.0 MP + 1:1 → ~1024×1024; 2.0 MP + 16:9 → ~1920×1088.\n",
        "Целевое разрешение в мегапикселях (только режим Megapixels), диапазон 0.1–8.0, шаг 0.1.\n\n"
        "Ширина/высота вычисляются из значения плюс aspect_ratio и округляются до кратности 32.\n"
        "Примеры: 0.15 MP + 16:9 → ~512×288; 1.0 MP + 1:1 → ~1024×1024; 2.0 MP + 16:9 → ~1920×1088."
    ),
    "aspect_ratio": _T(
        "Target aspect ratio for Megapixels mode: 1:1 square, 3:2 photo, 2:3 vertical photo, 4:3 standard, 3:4 vertical standard, 16:9 widescreen, 9:16 vertical video, 21:9 ultrawide, 9:21 vertical ultrawide.\n\n"
        "Final size is the closest match to the target MP while keeping this ratio; all sizes are multiples of 32.\n",
        "Целевое соотношение сторон для режима Megapixels: 1:1 квадрат, 3:2 фото, 2:3 вертикальное фото, 4:3 стандарт, 3:4 вертикальный стандарт, 16:9 широкий экран, 9:16 вертикальное видео, 21:9 сверхширокий, 9:21 вертикальный сверхширокий.\n\n"
        "Итоговый размер — ближайший к целевым MP при сохранении этого соотношения; все размеры кратны 32."
    ),
    "frame_count_source": _T(
        "How the total frame count is set; the inactive source widget hides automatically (JS).\n\n"
        "• Seconds — automatic: round(sec×24) aligned UP to the 17N+5 sequence (5, 22, 39, 56, ...). 5s → 124 frames, 10s → 243 frames.\n"
        "• Frames — manual exact count, aligned UP to the same sequence (100 → 107, 124 → 124).\n"
        "HINT: Seconds for quick setup; Frames for precise control (loops, continuation clips).\n",
        "Способ задания общего числа кадров; неактивный виджет источника скрывается автоматически (JS).\n\n"
        "• Seconds — авторасчёт: round(сек×24) с выравниванием ВВЕРХ до последовательности 17N+5 (5, 22, 39, 56, ...). 5 сек → 124 кадра, 10 сек → 243 кадра.\n"
        "• Frames — ручное точное число кадров с выравниванием ВВЕРХ до той же последовательности (100 → 107, 124 → 124).\n"
        "СОВЕТ: Seconds для быстрой настройки; Frames для точного контроля (циклы, клипы-продолжения)."
    ),
    "length_seconds": _T(
        "Desired video duration in seconds (used only when frame_count_source = Seconds). Range 1–60, step 1.\n\n"
        "Frame count = round(sec×24) aligned UP to 17N+5, so the result differs slightly from sec×24 — this is REQUIRED by MiniMax H3.\n"
        "Examples: 5s → 124 frames, 10s → 243, 15s → 362. Trained H3 range is ~124–362 frames.\n",
        "Желаемая длительность видео в секундах (используется только при frame_count_source = Seconds). Диапазон 1–60, шаг 1.\n\n"
        "Число кадров = round(сек×24) с выравниванием ВВЕРХ до 17N+5, поэтому результат немного отличается от сек×24 — это ТРЕБОВАНИЕ MiniMax H3.\n"
        "Примеры: 5 сек → 124 кадра, 10 сек → 243, 15 сек → 362. Обученный диапазон H3 — ~124–362 кадра."
    ),
    "frame_count": _T(
        "Exact frame count (used only when frame_count_source = Frames). Range 5–99999.\n\n"
        "Aligned UP to the MiniMax H3 sequence 17N+5: 5, 22, 39, 56, 73, 90, 107, 124...\n"
        "Examples: 100 → 107; 124 → 124 (already valid); 200 → 209.\n"
        "HINT: typical H3 clips are 124 (5s @24fps) and 243 (10s @24fps).\n",
        "Точное число кадров (используется только при frame_count_source = Frames). Диапазон 5–99999.\n\n"
        "Выравнивается ВВЕРХ до последовательности MiniMax H3 17N+5: 5, 22, 39, 56, 73, 90, 107, 124...\n"
        "Примеры: 100 → 107; 124 → 124 (уже допустимо); 200 → 209.\n"
        "СОВЕТ: типичные клипы H3 — 124 (5 сек @24fps) и 243 (10 сек @24fps)."
    ),
}
def _ref_tip(en, ru):
    return _T(en, ru)
#========================================================================
# Общие виджеты калькулятора для обеих нод / Shared calculator widgets
#========================================================================
def _calc_widgets():
    """
    Строит required-виджеты калькулятора с подробными тултипами для обеих нод.
    Builds the required calculator widgets with detailed tooltips for both nodes.
    """
    return {
        "mode": (["Preset", "Custom", "Megapixels"], {"default": "Preset", "tooltip": CALC_TOOLTIPS["mode"]}),
        "preset": (PRESET_LIST, {"default": "1280×704 (16:9)", "tooltip": CALC_TOOLTIPS["preset"]}),
        "invert_orientation": ("BOOLEAN", {"default": False, "tooltip": CALC_TOOLTIPS["invert_orientation"]}),
        "custom_width": ("INT", {"default": 864, "min": 64, "max": 8192, "step": 32, "display": "number", "tooltip": CALC_TOOLTIPS["custom_width"]}),
        "custom_height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 32, "display": "number", "tooltip": CALC_TOOLTIPS["custom_height"]}),
        "megapixels_value": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "display": "number", "tooltip": CALC_TOOLTIPS["megapixels_value"]}),
        "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9", "tooltip": CALC_TOOLTIPS["aspect_ratio"]}),
        "frame_count_source": (["Seconds", "Frames"], {"default": "Seconds", "tooltip": CALC_TOOLTIPS["frame_count_source"]}),
        "length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 1.0, "display": "number", "tooltip": CALC_TOOLTIPS["length_seconds"]}),
        "frame_count": ("INT", {"default": 124, "min": 5, "max": 99999, "step": 1, "display": "number", "tooltip": CALC_TOOLTIPS["frame_count"]}),
    }
def _calc_size(mode, preset, invert_orientation, custom_width, custom_height,
               megapixels_value, aspect_ratio):
    """
    Ширина/высота по режиму калькулятора, кратность 32 / Width/height per calculator mode, multiple of 32
    """
    if mode == "Preset":
        w, h = PRESET_MAP[preset]
        if invert_orientation:
            w, h = h, w
        return fit_to_multiple(w, 32), fit_to_multiple(h, 32)
    if mode == "Custom":
        w, h = custom_width, custom_height
        if invert_orientation:
            w, h = h, w
        return fit_to_multiple(w, 32), fit_to_multiple(h, 32)
    if mode == "Megapixels":
        w_r, h_r = map(int, aspect_ratio.split(":"))
        target = megapixels_value * 1_000_000
        x = math.sqrt(target / (w_r * h_r))
        w = fit_to_multiple(round(w_r * x), 32)
        h = fit_to_multiple(round(h_r * x), 32)
        return max(64, w), max(64, h)
    return 1280, 704
def _calc_frames(frame_count_source, length_seconds, frame_count):
    """
    Число кадров 17N+5 по источнику / Frame count 17N+5 per source
    """
    if frame_count_source == "Frames":
        total = fit_length_to_17n5(frame_count)
    else:
        total = fit_length_to_17n5(round(length_seconds * 24))
    return max(5, total)
#========================================================================
# Нода 1: AGSoft MiniMax H3 Ref2V
#========================================================================
class AGSoft_MiniMax_H3_Ref2V:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "AGSoft MiniMax H3 Ref2V — local Reference-to-Video conditioning for MiniMax H3 "
        "with the AGSoft parameter calculator.\n"
        "Pipeline: empty joint AV latent (video 24ch + audio 32ch) of the calculated size; "
        "reference images resized per ref_image_size (match = down-only to the generation pixel "
        "area, max = 2048px short edge) and encoded by the video VAE; reference videos adapted to "
        "the 768-short-edge canvas, trimmed to the generation length and aligned down to 17N+5 "
        "(min 5 frames), encoded by the video VAE; paired soundtracks (ref_video_audio_N for "
        "ref_video_N) and standalone audio resampled to the audio VAE rate and encoded; the prompt "
        "is tokenized together with the reference presentation (minimax_ref_items) and the "
        "resulting CONDITIONING carries minimax_refs blocks for the DiT.\n"
        "References enter the prompt by 1-based ordinals per type in connection order: "
        "ref_image_0 → <Picture 1>, ref_video_0 → <Video 1>, audio → <Audio j> (a video soundtrack "
        "label is emitted right before its video label).\n"
        "Ref inputs are dynamic per group (connected+1 slots, JS): image up to 10, video up to 3, "
        "video_audio up to 3, audio up to 10. Calculator widgets auto-hide per mode; the live info "
        "line (DOM widget) shows ⚙ W×H • sec • frames between the last combo and the prompt.\n"
        "Outputs: positive (CONDITIONING), LATENT, width, height, fps_int, fps_float, total_frames.\n"
        "---\n"
        "AGSoft MiniMax H3 Ref2V — локальное кондиционирование Reference-to-Video для MiniMax H3 "
        "с калькулятором параметров AGSoft.\n"
        "Конвейер: пустой совместный AV-латент (видео 24 канала + аудио 32 канала) рассчитанного "
        "размера; референс-изображения ресайзятся по ref_image_size (match = только вниз до площади "
        "генерации, max = короткая сторона 2048px) и кодируются видео-VAE; референс-видео "
        "приводятся к холсту с короткой стороной 768, обрезаются до длины генерации и выравниваются "
        "вниз до 17N+5 (минимум 5 кадров), кодируются видео-VAE; парные дорожки (ref_video_audio_N "
        "для ref_video_N) и отдельные аудио ресемплируются до частоты audio_vae и кодируются; "
        "промпт токенизируется вместе с презентацией референсов (minimax_ref_items), и итоговый "
        "CONDITIONING несёт блоки minimax_refs для DiT.\n"
        "Референсы упоминаются в промпте порядковыми тегами с 1 по типу в порядке подключения: "
        "ref_image_0 → <Picture 1>, ref_video_0 → <Video 1>, аудио → <Audio j> (метка дорожки видео "
        "ставится сразу перед меткой своего видео).\n"
        "Входы рефов динамические по группам (подключено+1 слотов, JS): image до 10, video до 3, "
        "video_audio до 3, audio до 10. Виджеты калькулятора автоскрываются по режимам; живая "
        "инфострока (DOM-виджет) показывает ⚙ W×H • сек • кадры между последним combo и prompt.\n"
        "Выходы: positive (CONDITIONING), LATENT, width, height, fps_int, fps_float, total_frames."
    )
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(10):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": _ref_tip(
                f"Optional reference image #{i+1} in connection order → <Picture {i+1}> in the prompt. "
                "Downscaled to a 2048px short edge if larger, never upscaled. "
                "Dynamic group: connecting the last free slot creates the next one (JS).",
                f"Опциональный референс-изображение #{i+1} в порядке подключения → <Picture {i+1}> в промпте. "
                "Уменьшается до короткой стороны 2048px если больше, никогда не увеличивается. "
                "Динамическая группа: подключение к последнему свободному слоту создаёт следующий (JS).")})
        for i in range(3):
            optional[f"ref_video_{i}"] = ("VIDEO", {"tooltip": _ref_tip(
                f"Optional reference video #{i+1} → <Video {i+1}> in the prompt. Frames are treated as "
                "24 fps, adapted to the 768-short-edge canvas, trimmed to the generation length and "
                "aligned down to 17N+5 (minimum 5 frames, else error).",
                f"Опциональный референс-видео #{i+1} → <Video {i+1}> в промпте. Кадры считаются 24 fps, "
                "приводятся к холсту с короткой стороной 768, обрезаются до длины генерации и "
                "выравниваются вниз до 17N+5 (минимум 5 кадров, иначе ошибка).")})
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": _ref_tip(
                f"Soundtrack paired by index with ref_video_{i} (ref_video_audio_{i} ↔ ref_video_{i}). "
                "Gets its own <Audio j> label emitted right before its <Video k> label; the text encoder "
                "sees the video at 2 fps with timestamps. Encoded by the audio VAE when connected.",
                f"Аудиодорожка, парная по индексу с ref_video_{i} (ref_video_audio_{i} ↔ ref_video_{i}). "
                "Получает собственную метку <Audio j> сразу перед меткой своего <Video k>; текст-энкодер "
                "видит видео на 2 fps с таймстампами. Кодируется audio_vae при подключении.")})
        for i in range(10):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": _ref_tip(
                f"Optional standalone reference audio #{i+1} (voice/timbre) → next <Audio j> ordinal in "
                "prompt order. Resampled to the audio VAE rate and encoded. Recommended to combine with "
                "at least one image or video reference.",
                f"Опциональный отдельный референс-аудио #{i+1} (голос/тембр) → следующий порядковый "
                "<Audio j> в порядке промпта. Ресемплируется до частоты audio_vae и кодируется. "
                "Рекомендуется сочетать хотя бы с одним изображением или видео-референсом.")})
        return {
            "required": {
                **_calc_widgets(),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip": _ref_tip(
                    "Reference image sizing for the H3 encoder. ONLY these two values exist in the H3 "
                    "combo — both are required.\n"
                    "• match — each reference is scaled (down only, aspect kept) to the generation's "
                    "pixel area; cheap and consistent with the output framing.\n"
                    "• max — references use the reference pipeline's 2048px short edge for best identity "
                    "fidelity; reference tokens ride through every sampling step, so 'max' can be several "
                    "times slower.",
                    "Размер референс-изображений для энкодера H3. В combo H3 существуют ТОЛЬКО эти два "
                    "значения — оба обязательны.\n"
                    "• match — каждый референс масштабируется (только вниз, пропорции сохраняются) до "
                    "площади пикселей генерации; дёшево и согласовано с кадрированием выхода.\n"
                    "• max — референсы используют короткую сторону 2048px конвейера референсов для лучшей "
                    "передачи внешности; токены референсов проходят каждый шаг семплирования, поэтому "
                    "'max' может быть в несколько раз медленнее.")}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _ref_tip(
                    "Text prompt. References are addressed by 1-based ordinals per type in connection "
                    "order: ref_image_0 → <Picture 1>, ref_video_0 → <Video 1>, audio refs and video "
                    "soundtracks → <Audio j> (a soundtrack label is emitted right before its video).\n"
                    "Typical pattern: 'Subject_definitions: <Subject 1>: character from <Picture 1> ...' "
                    "followed by the scene description using the same tags.",
                    "Текстовый промпт. Референсы адресуются порядковыми тегами с 1 по типу в порядке "
                    "подключения: ref_image_0 → <Picture 1>, ref_video_0 → <Video 1>, аудио-рефы и дорожки "
                    "видео → <Audio j> (метка дорожки ставится сразу перед меткой своего видео).\n"
                    "Типовой шаблон: 'Subject_definitions: <Subject 1>: персонаж из <Picture 1> ...' "
                    "далее описание сцены с теми же тегами.")}),
                "clip": ("CLIP", {"tooltip": _ref_tip(
                    "CLIP / text encoder (Qwen3-VL family for H3). Encodes the prompt together with the "
                    "reference presentation (minimax_ref_items: images, 2 fps video frames with "
                    "timestamps, audio placeholders).",
                    "CLIP / текст-энкодер (семейство Qwen3-VL для H3). Кодирует промпт вместе с презентацией "
                    "референсов (minimax_ref_items: изображения, кадры видео на 2 fps с таймстампами, "
                    "аудио-плейсхолдеры).")}),
                "vae": ("VAE", {"tooltip": _ref_tip(
                    "Video VAE (24-channel latent). Encodes resized reference images and trimmed reference "
                    "videos into minimax_refs latents. Without it references only condition the text "
                    "encoder (no DiT ref blocks).",
                    "Видео-VAE (24-канальный латент). Кодирует ресайзнутые референс-изображения и обрезанные "
                    "референс-видео в латенты minimax_refs. Без него референсы кондиционируют только "
                    "текст-энкодер (без блоков рефов для DiT).")}),
                "audio_vae": ("VAE", {"tooltip": _ref_tip(
                    "Audio VAE (32-channel, 40 Hz latent). Encodes paired soundtracks and standalone "
                    "reference audio after resampling to its rate. Without it audio refs only condition "
                    "the text encoder.",
                    "Аудио-VAE (32-канальный, 40 Гц латент). Кодирует парные дорожки и отдельные "
                    "референс-аудио после ресемплирования до своей частоты. Без него аудио-рефы "
                    "кондиционируют только текст-энкодер.")}),
            },
            "optional": optional,
        }
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("positive", "LATENT", "width", "height", "fps_int", "fps_float", "total_frames")
    OUTPUT_TOOLTIPS = (
        _ref_tip("CONDITIONING: encoded prompt with the reference presentation; carries 'minimax_refs' "
                 "blocks (image/video/video_audio/audio latents with grid sizes) for the H3 DiT.",
                 "CONDITIONING: закодированный промпт с презентацией референсов; несёт блоки "
                 "'minimax_refs' (латенты изображений/видео/видео_аудио/аудио с размерами сетки) для DiT H3."),
        _ref_tip("Empty joint audio-video latent: video [B,24,latent_t,H//16,W//16] + audio [B,32,2,audio_t] "
                 "as a NestedTensor, sizes derived from width/height/total_frames (17N+5).",
                 "Пустой совместный аудио-видео латент: видео [B,24,latent_t,H//16,W//16] + аудио "
                 "[B,32,2,audio_t] в NestedTensor; размеры получены из width/height/total_frames (17N+5)."),
        _ref_tip("Generation frame width, multiple of 32 / Ширина кадра генерации, кратна 32.",
                 "Ширина кадра генерации, кратна 32."),
        _ref_tip("Generation frame height, multiple of 32 / Высота кадра генерации, кратна 32.",
                 "Высота кадра генерации, кратна 32."),
        _ref_tip("FPS as integer, always 24 for MiniMax H3 / FPS целым, всегда 24 для MiniMax H3.",
                 "FPS целым, всегда 24 для MiniMax H3."),
        _ref_tip("FPS as float, always 24.0 for MiniMax H3 / FPS дробным, всегда 24.0 для MiniMax H3.",
                 "FPS дробным, всегда 24.0 для MiniMax H3."),
        _ref_tip("Total frame count aligned to 17N+5 (5, 22, 39, ...); equals the AV latent temporal grid.",
                 "Общее число кадров, выровненное до 17N+5 (5, 22, 39, ...); равно временной сетке AV-латента."),
    )
    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count, ref_image_size, prompt,
             clip, vae, audio_vae, **refs):
        # 1. Калькулятор размера / Size calculator
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)
        fps_int, fps_float = 24, 24.0
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        # 2. Пустой AV-латент / Empty AV latent
        latent, frame_count = _empty_av_latent(width, height, total)
        # 3. Референсы (логика H3 ref2va) / References (H3 ref2va logic)
        ref_items = []
        ref_blocks = []
        for i in range(10):
            img = refs.get(f"ref_image_{i}")
            if img is None: continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            if vae is not None:
                z = vae.encode(resized)
                ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
        for i in range(3):
            video_frames = refs.get(f"ref_video_{i}")
            if video_frames is None: continue
            soundtrack = refs.get(f"ref_video_audio_{i}")
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames / референс-видео MiniMax H3 требуют минимум 5 кадров")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            if soundtrack is not None:
                ref_items.append({"type": "audio"})
                sample_idx = list(range(0, frames.shape[0], FPS // 2))
                qwen_frames = frames[sample_idx]
                ref_items.append({"type": "video", "data": qwen_frames, "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            if vae is None:
                continue
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent
            })
        for i in range(10):
            audio = refs.get(f"ref_audio_{i}")
            if audio is None: continue
            ref_items.append({"type": "audio"})
            audio_latent, ref_audio_t = (None, 0)
            if audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
        # 4. Токенизация и кондиционирование / Tokenization and conditioning
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return (cond, latent, width, height, fps_int, fps_float, total)
#========================================================================
# Нода 2: AGSoft MiniMax H3 I2V
#========================================================================
class AGSoft_MiniMax_H3_I2V:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "AGSoft MiniMax H3 I2V — local Image-to-Video (t2va / fl2va) conditioning for MiniMax H3 "
        "with the AGSoft parameter calculator.\n"
        "Pipeline: empty joint AV latent (video 24ch + audio 32ch) of the calculated size; optional "
        "first_frame is stretched plain to the generation canvas (geometry anchor, resolved to frame "
        "index 0); optional last_frame is aspect-preserving cover-cropped to the canvas (follower, "
        "resolved to frame index length-1); both are encoded by the video VAE into minimax_keyframes "
        "blocks attached to the CONDITIONING; the prompt is tokenized together with the keyframe images.\n"
        "Without any keyframes the node works as pure text-to-video. With both frames the model "
        "interpolates the motion between them.\n"
        "Calculator widgets auto-hide per mode; the live info line (DOM widget) shows ⚙ W×H • sec • "
        "frames between the last combo and the prompt.\n"
        "Outputs: positive (CONDITIONING), LATENT, width, height, fps_int, fps_float, total_frames.\n"
        "---\n"
        "AGSoft MiniMax H3 I2V — локальное кондиционирование Image-to-Video (t2va / fl2va) для "
        "MiniMax H3 с калькулятором параметров AGSoft.\n"
        "Конвейер: пустой совместный AV-латент (видео 24 канала + аудио 32 канала) рассчитанного "
        "размера; опциональный first_frame растягивается по холсту генерации без сохранения пропорций "
        "(геометрический якорь, кадр 0); опциональный last_frame обрезается по центру с сохранением "
        "пропорций (cover-crop, последний кадр length-1); оба кодируются видео-VAE в блоки "
        "minimax_keyframes, прикрепляемые к CONDITIONING; промпт токенизируется вместе с изображениями "
        "ключевых кадров.\n"
        "Без ключевых кадров нода работает как чистый text-to-video. С обоими кадрами модель "
        "интерполирует движение между ними.\n"
        "Виджеты калькулятора автоскрываются по режимам; живая инфострока (DOM-виджет) показывает "
        "⚙ W×H • сек • кадры между последним combo и prompt.\n"
        "Выходы: positive (CONDITIONING), LATENT, width, height, fps_int, fps_float, total_frames."
    )
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_calc_widgets(),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _ref_tip(
                    "Text prompt describing the scene and the motion.\n"
                    "• first_frame only: the motion starts from it.\n"
                    "• last_frame only: the motion ends there.\n"
                    "• both: the model interpolates between the keyframes.\n"
                    "• none: pure text-to-video generation.",
                    "Текстовый промпт, описывающий сцену и движение.\n"
                    "• Только first_frame: движение начинается с него.\n"
                    "• Только last_frame: движение заканчивается на нём.\n"
                    "• Оба: модель интерполирует между ключевыми кадрами.\n"
                    "• Ни одного: чистая генерация text-to-video.")}),
                "clip": ("CLIP", {"tooltip": _ref_tip(
                    "CLIP / text encoder (Qwen3-VL family for H3). Encodes the prompt together with the "
                    "keyframe images (first/last frame) when they are connected.",
                    "CLIP / текст-энкодер (семейство Qwen3-VL для H3). Кодирует промпт вместе с "
                    "изображениями ключевых кадров (первый/последний), если они подключены.")}),
                "vae": ("VAE", {"tooltip": _ref_tip(
                    "Video VAE (24-channel latent). Encodes the connected keyframes into minimax_keyframes "
                    "blocks. Required when first_frame or last_frame is used.",
                    "Видео-VAE (24-канальный латент). Кодирует подключённые ключевые кадры в блоки "
                    "minimax_keyframes. Обязателен при использовании first_frame или last_frame.")}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": _ref_tip(
                    "Optional first keyframe. Geometry anchor: plain stretch to the generation canvas "
                    "(aspect may distort). Resolved to frame index 0 and encoded by the video VAE into "
                    "minimax_keyframes.",
                    "Опциональный первый ключевой кадр. Геометрический якорь: растягивание по холсту "
                    "генерации без сохранения пропорций. Приводится к индексу кадра 0 и кодируется "
                    "видео-VAE в minimax_keyframes.")}),
                "last_frame": ("IMAGE", {"tooltip": _ref_tip(
                    "Optional last keyframe. Follower: aspect-preserving cover-crop (center) to the "
                    "generation canvas. Resolved to frame index length-1 and encoded by the video VAE "
                    "into minimax_keyframes.",
                    "Опциональный последний ключевой кадр. Последователь: обрезка по центру с сохранением "
                    "пропорций (cover-crop) по холсту генерации. Приводится к индексу кадра length-1 и "
                    "кодируется видео-VAE в minimax_keyframes.")}),
            },
        }
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("positive", "LATENT", "width", "height", "fps_int", "fps_float", "total_frames")
    OUTPUT_TOOLTIPS = (
        _ref_tip("CONDITIONING: encoded prompt (+ keyframe images); carries 'minimax_keyframes' blocks "
                 "(frame index + VAE latent) for the H3 DiT when keyframes are connected.",
                 "CONDITIONING: закодированный промпт (+ изображения ключевых кадров); несёт блоки "
                 "'minimax_keyframes' (индекс кадра + латент VAE) для DiT H3 при подключённых кадрах."),
        _ref_tip("Empty joint audio-video latent: video [B,24,latent_t,H//16,W//16] + audio [B,32,2,audio_t] "
                 "as a NestedTensor, sizes derived from width/height/total_frames (17N+5).",
                 "Пустой совместный аудио-видео латент: видео [B,24,latent_t,H//16,W//16] + аудио "
                 "[B,32,2,audio_t] в NestedTensor; размеры получены из width/height/total_frames (17N+5)."),
        _ref_tip("Generation frame width, multiple of 32 / Ширина кадра генерации, кратна 32.",
                 "Ширина кадра генерации, кратна 32."),
        _ref_tip("Generation frame height, multiple of 32 / Высота кадра генерации, кратна 32.",
                 "Высота кадра генерации, кратна 32."),
        _ref_tip("FPS as integer, always 24 for MiniMax H3 / FPS целым, всегда 24 для MiniMax H3.",
                 "FPS целым, всегда 24 для MiniMax H3."),
        _ref_tip("FPS as float, always 24.0 for MiniMax H3 / FPS дробным, всегда 24.0 для MiniMax H3.",
                 "FPS дробным, всегда 24.0 для MiniMax H3."),
        _ref_tip("Total frame count aligned to 17N+5 (5, 22, 39, ...); equals the AV latent temporal grid.",
                 "Общее число кадров, выровненное до 17N+5 (5, 22, 39, ...); равно временной сетке AV-латента."),
    )
    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count, prompt, clip, vae,
             first_frame=None, last_frame=None):
        # 1. Калькулятор размера / Size calculator
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)
        fps_int, fps_float = 24, 24.0
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        # 2. Пустой AV-латент / Empty AV latent
        latent, frame_count = _empty_av_latent(width, height, total)
        # 3. Ключевые кадры (логика H3 I2V) / Keyframes (H3 I2V logic)
        images = []
        keyframes = []
        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
        # 4. Токенизация и кондиционирование / Tokenization and conditioning
        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        return (cond, latent, width, height, fps_int, fps_float, total)
#========================================================================
# Регистрация / Registration
#========================================================================
NODE_CLASS_MAPPINGS = {
    "AGSoft_MiniMax_H3_Ref2V": AGSoft_MiniMax_H3_Ref2V,
    "AGSoft_MiniMax_H3_I2V": AGSoft_MiniMax_H3_I2V,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_MiniMax_H3_Ref2V": "🎬AGSoft MiniMax H3 Ref2V",
    "AGSoft_MiniMax_H3_I2V": "🎬AGSoft MiniMax H3 I2V",
}