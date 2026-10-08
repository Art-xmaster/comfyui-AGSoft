"""
AGSoft_MiniMax_H3.py
Ноды / Nodes: 🎬AGSoft MiniMax H3 Ref2V, 🎬AGSoft MiniMax H3 I2V
Версия / Version: v09.48
Описание / Description:
Пара нод кондиционирования для MiniMax H3 (ref2va / t2va+fl2va) с полным
локальным воспроизведением логики нативных нодов ComfyUI и калькулятором
параметров AGSoft: размер кратен 32, кадры — последовательность 17N+5
(5, 22, 39, ...), FPS фиксирован 24, длительность задаётся в секундах
(Seconds) или кадрах (Frames).
Ref2V кодирует промпт вместе с презентацией референсов (изображения,
видео ПОТОКОМ КАДРОВ (IMAGE), парные аудиодорожки, отдельные аудио) и
передаёт DiT блоки minimax_refs; I2V работает с текстом и опциональными
первым/последним кадрами (minimax_keyframes). Обе ноды отдают пустой
совместный AV-латент (видео 24 канала + аудио 32 канала) рассчитанного
размера.
A pair of conditioning nodes for MiniMax H3 (ref2va / t2va+fl2va) with a
full local re-implementation of the native ComfyUI nodes' logic plus the
AGSoft parameter calculator: sizes are multiples of 32, frame count follows
the 17N+5 sequence (5, 22, 39, ...), FPS is fixed at 24, duration is set in
seconds (Seconds) or frames (Frames).
Ref2V encodes the prompt together with the reference presentation (images,
videos as FRAME STREAMS (IMAGE), paired soundtracks, standalone audio) and
passes minimax_refs blocks to the DiT; I2V works with text and optional
first/last keyframes (minimax_keyframes). Both nodes output an empty joint
AV latent (24-channel video + 32-channel audio) of the calculated size.
Возможности / Features:
⚡ Полная локальная логика H3: кодирование рефов через VAE/audio_vae,
   токенизация с minimax_ref_items, кондиционирование minimax_refs /
   minimax_keyframes, пустой AV-латент (NestedTensor видео+аудио).
   Full local H3 logic: ref encoding via VAE/audio_vae, tokenization with
   minimax_ref_items, minimax_refs / minimax_keyframes conditioning, empty
   AV latent (video+audio NestedTensor).
⚡ Калькулятор: Preset (40 размеров) / Custom / Megapixels + Seconds/Frames,
   кратность 32, кадры 17N+5, FPS 24, инверсия сторон во ВСЕХ режимах
   (в Megapixels меняет местами расчётный размер) — как в 🎬AGSoft MiniMax Base.
   Calculator: Preset (40 sizes) / Custom / Megapixels + Seconds/Frames,
   multiples of 32, 17N+5 frames, FPS 24, orientation invert in ALL modes
   (in Megapixels it swaps the computed size) — same as 🎬AGSoft MiniMax Base.
⚡ Выход duration_seconds — точная длительность клипа (total_frames / 24),
   как в 🎬AGSoft MiniMax Base.
   duration_seconds output — exact clip duration (total_frames / 24),
   same as 🎬AGSoft MiniMax Base.
⚡ vae / audio_vae опциональны (1:1 с нативом): без VAE рефы кондиционируют
   только текст-энкодер (без блоков minimax_refs / minimax_keyframes).
   vae / audio_vae are optional (1:1 with native): without VAE the refs only
   condition the text encoder (no minimax_refs / minimax_keyframes blocks).
⚡ Ref2V: динамические входы рефов по группам, лимиты 1:1 с нативом
   (image до 9, video до 3, video_audio до 3, audio до 3); ref_video_N —
   тип IMAGE (поток кадров) 1:1 с нативом: подключаются кадры, а не объект
   VIDEO; защитная нормализация тензора [F,H,W,C] и диапазона 0..1.
   Ref2V: dynamic ref inputs per group with native limits (image up to 9,
   video up to 3, video_audio up to 3, audio up to 3); ref_video_N is IMAGE
   type (frame stream) 1:1 with native: frames connect, not a VIDEO object;
   defensive normalization of the [F,H,W,C] tensor and 0..1 range.
⚡ I2V: first_frame (stretch до холста) и last_frame (cover-crop с центром),
   режим чистого text-to-video без кадров.
   I2V: first_frame (stretch to canvas) and last_frame (center cover-crop),
   plus a pure text-to-video mode without frames.
⚡ Автоскрытие неиспользуемых виджетов калькулятора по режимам (JS),
   invert_orientation виден во всех режимах, высота ноды подгоняется под
   видимые виджеты, prompt растягивается до низа.
   Auto-hiding of unused calculator widgets per mode (JS), invert_orientation
   visible in all modes, node height fits the visible widgets, the prompt
   stretches to the bottom.
⚡ Живая инфострока как в Base: ⚙ W×H • MP • ~формат • кадры • точные сек.
   Live info line same as Base: ⚙ W×H • MP • ~aspect • frames • exact sec.
⚡ ref_image_size строго match/max — обязательные значения combo H3.
   ref_image_size strictly match/max — the required H3 combo values.
Автор / Author: AGSoft
Дата / Date: 09.10.2026
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
#print("[AGSoft MiniMax H3] v09.48 loaded (ref_video = IMAGE frame stream, Base-style live line)")
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
    """Округляет ВВЕРХ до кратного multiple / Rounds UP to a multiple. 100 → 128."""
    return ((value + multiple - 1) // multiple) * multiple
def fit_length_to_17n5(value: int) -> int:
    """Выравнивает ВВЕРХ до 17N+5 (5, 22, 39, ...) / Aligns UP to 17N+5. 100 → 107."""
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
def _video_to_frames(video):
    """
    Приводит вход ref_video_N к тензору кадров [F, H, W, C]: штатно приходит
    IMAGE-тензор; защитно поддерживается и объект VideoFromFile нового API
    (get_components()/get_pixels()). Normalizes the ref_video_N input to a
    [F, H, W, C] frames tensor: normally an IMAGE tensor arrives; a new-API
    VideoFromFile object (get_components()/get_pixels()) is supported
    defensively.
    """
    t = None
    if isinstance(video, torch.Tensor):
        t = video
    else:
        gc = getattr(video, "get_components", None)
        if callable(gc):
            try:
                comp = gc()
                if comp is not None:
                    imgs = getattr(comp, "images", None)
                    if isinstance(imgs, torch.Tensor):
                        t = imgs
            except Exception as e:
                logger.warning(f"[AGSoft MiniMax H3] get_components failed: {e}")
        if t is None and hasattr(video, "get_pixels"):
            t = video.get_pixels()
        if t is None and isinstance(video, dict):
            t = video.get("images") or video.get("pixels")
    if t is None:
        raise TypeError(f"Unsupported ref_video input type: {type(video).__name__}")
    if t.dim() == 5:  # [B, F, H, W, C] → [F, H, W, C]
        t = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
    t = t.detach().cpu()
    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return t
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
# Краткие двуязычные тултипы калькулятора / Concise bilingual tooltips
#========================================================================
_T = lambda en, ru: en + "\n---\n" + ru
CALC_TOOLTIPS = {
    "mode": _T(
        "Frame size mode: Preset (predefined), Custom (own W/H), Megapixels (MP + ratio). Unused widgets hide automatically.",
        "Режим размера кадра: Preset (готовые), Custom (свои W/H), Megapixels (MP + соотношение). Лишние виджеты скрываются автоматически."),
    "preset": _T(
        "Predefined size WIDTH×HEIGHT (aspect), all multiples of 32. invert_orientation swaps W/H.",
        "Готовый размер ШИРИНА×ВЫСОТА (соотношение), все кратны 32. invert_orientation меняет W/H местами."),
    "invert_orientation": _T(
        "Swaps width and height. Works in ALL modes (in Megapixels it swaps the computed size).",
        "Меняет ширину и высоту местами. Работает во ВСЕХ режимах (в Megapixels — расчётный размер)."),
    "custom_width": _T(
        "Frame width in px (Custom). Rounded UP to a multiple of 32. Range 64–8192, step 32.",
        "Ширина кадра в px (Custom). Округляется ВВЕРХ до кратности 32. Диапазон 64–8192, шаг 32."),
    "custom_height": _T(
        "Frame height in px (Custom). Rounded UP to a multiple of 32. Range 64–8192, step 32.",
        "Высота кадра в px (Custom). Округляется ВВЕРХ до кратности 32. Диапазон 64–8192, шаг 32."),
    "megapixels_value": _T(
        "Target resolution in MP (0.01–8.0, step 0.01). W/H computed from aspect_ratio, multiples of 32.",
        "Целевое разрешение в MP (0.01–8.0, шаг 0.01). W/H считаются по aspect_ratio, кратны 32."),
    "aspect_ratio": _T(
        "Aspect ratio for Megapixels: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21.",
        "Соотношение сторон для Megapixels: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9, 9:21."),
    "frame_count_source": _T(
        "Seconds — frames = round(sec×24) aligned UP to 17N+5. Frames — exact count aligned UP to 17N+5.",
        "Seconds — кадры = round(сек×24) ВВЕРХ до 17N+5. Frames — точное число с выравниванием ВВЕРХ до 17N+5."),
    "length_seconds": _T(
        "Duration in seconds (1–60, Seconds only). Frames = round(sec×24) aligned UP to 17N+5, so it differs slightly from sec×24.",
        "Длительность в секундах (1–60, только Seconds). Кадры = round(сек×24) ВВЕРХ до 17N+5, поэтому число немного отличается от сек×24."),
    "frame_count": _T(
        "Exact frame count (Frames only), aligned UP to 17N+5: 100 → 107, 124 → 124.",
        "Точное число кадров (только Frames), выравнивается ВВЕРХ до 17N+5: 100 → 107, 124 → 124."),
}
def _ref_tip(en, ru):
    return _T(en, ru)
#========================================================================
# Общие виджеты калькулятора для обеих нод / Shared calculator widgets
#========================================================================
def _calc_widgets():
    """Строит виджеты калькулятора / Builds the calculator widgets."""
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
    """Ширина/высота по режиму, кратность 32, инверсия во всех режимах.
    Width/height per mode, multiple of 32, invert in all modes."""
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
        if invert_orientation:
            w, h = h, w
        return max(64, w), max(64, h)
    return 1280, 704
def _calc_frames(frame_count_source, length_seconds, frame_count):
    """Число кадров 17N+5 по источнику / Frame count 17N+5 per source."""
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
        "Local Reference-to-Video conditioning for MiniMax H3 with the AGSoft calculator.\n"
        "• Size: Preset / Custom / Megapixels, multiples of 32; invert works in ALL modes.\n"
        "• Frames: Seconds or Frames → 17N+5 sequence, FPS 24; duration_seconds = total/24.\n"
        "• Refs enter the prompt 1-based per type in connection order: <Picture i>, "
        "<Video k>, <Audio j>; a video soundtrack label goes right before its video.\n"
        "• ref_video_N is an IMAGE input (frame stream at 24 fps) — connect frames, "
        "not a VIDEO object, same as the native node.\n"
        "• vae / audio_vae are OPTIONAL: without them refs only condition the text encoder.\n"
        "• ref_image_size: match (down to generation area) / max (2048px short edge, slower).\n"
        "• Dynamic ref slots (connected+1, JS); live info line ⚙ W×H • MP • ~aspect • frames • exact sec.\n"
        "---\n"
        "Локальное кондиционирование Reference-to-Video для MiniMax H3 с калькулятором AGSoft.\n"
        "• Размер: Preset / Custom / Megapixels, кратность 32; инверсия во ВСЕХ режимах.\n"
        "• Кадры: Seconds или Frames → последовательность 17N+5, FPS 24; duration_seconds = total/24.\n"
        "• Референсы в промпте с 1 по типу в порядке подключения: <Picture i>, <Video k>, "
        "<Audio j>; метка дорожки видео ставится сразу перед меткой своего видео.\n"
        "• ref_video_N — вход IMAGE (поток кадров 24 fps): подключайте кадры, а не объект "
        "VIDEO, как в нативной ноде.\n"
        "• vae / audio_vae ОПЦИОНАЛЬНЫ: без них рефы кондиционируют только текст-энкодер.\n"
        "• ref_image_size: match (вниз до площади генерации) / max (короткая сторона 2048px, медленнее).\n"
        "• Динамические слоты рефов (подключено+1, JS); живая инфострока "
        "⚙ W×H • MP • ~формат • кадры • точные сек."
    )
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "vae": ("VAE", {"tooltip": _ref_tip(
                "Video VAE (24ch latent). OPTIONAL: without it reference images/videos only condition the text encoder.",
                "Видео-VAE (24ch латент). ОПЦИОНАЛЕН: без него референс-изображения/видео кондиционируют только текст-энкодер.")}),
            "audio_vae": ("VAE", {"tooltip": _ref_tip(
                "Audio VAE (32ch, 40 Hz latent). OPTIONAL: without it reference audio only conditions the text encoder.",
                "Аудио-VAE (32ch, 40 Гц латент). ОПЦИОНАЛЕН: без него референс-аудио кондиционируют только текст-энкодер.")}),
        }
        for i in range(9):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": _ref_tip(
                f"Optional reference image #{i+1} → <Picture {i+1}> in the prompt. Downscaled to 2048px short edge if larger. Dynamic group (connected+1, JS).",
                f"Опциональный референс-изображение #{i+1} → <Picture {i+1}> в промпте. Уменьшается до короткой стороны 2048px если больше. Динамическая группа (подключено+1, JS).")})
        for i in range(3):
            optional[f"ref_video_{i}"] = ("IMAGE", {"tooltip": _ref_tip(
                f"Optional reference video #{i+1} as a FRAME STREAM (IMAGE batch at 24 fps, 2–15 s) → <Video {i+1}>. "
                "Connect frames (IMAGE), not a VIDEO object — same as the native node. "
                "Adapted to the 768 canvas, trimmed to generation length, aligned down to 17N+5 (min 5).",
                f"Опциональный референс-видео #{i+1} ПОТОКОМ КАДРОВ (батч IMAGE, 24 fps, 2–15 с) → <Video {i+1}>. "
                "Подключайте кадры (IMAGE), а не объект VIDEO — как в нативной ноде. "
                "Приводится к холсту 768, обрезается до длины генерации, выравнивается вниз до 17N+5 (минимум 5).")})
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": _ref_tip(
                f"Soundtrack paired by index with ref_video_{i}. Gets its own <Audio j> label right before its <Video k>.",
                f"Аудиодорожка, парная по индексу с ref_video_{i}. Получает собственную метку <Audio j> сразу перед своим <Video k>.")})
        for i in range(3):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": _ref_tip(
                f"Optional standalone reference audio #{i+1} (voice/timbre) → next <Audio j>. Combine with at least one image/video ref.",
                f"Опциональный отдельный референс-аудио #{i+1} (голос/тембр) → следующий <Audio j>. Сочетайте хотя бы с одним изображением/видео.")})
        return {
            "required": {
                **_calc_widgets(),
                "ref_image_size": (["match", "max"], {"default": "match", "tooltip": _ref_tip(
                    "Reference image sizing. match = down-only to the generation pixel area; max = 2048px short edge for best identity, several times slower.",
                    "Размер референс-изображений. match = только вниз до площади генерации; max = короткая сторона 2048px для лучшей внешности, в несколько раз медленнее.")}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _ref_tip(
                    "Text prompt. Refs addressed by 1-based tags per type: <Picture 1>, <Video 1>, <Audio 1>... Typical: 'Subject_definitions: <Subject 1>: character from <Picture 1>...'.",
                    "Текстовый промпт. Референсы адресуются тегами с 1 по типу: <Picture 1>, <Video 1>, <Audio 1>... Типово: 'Subject_definitions: <Subject 1>: персонаж из <Picture 1>...'.")}),
                "clip": ("CLIP", {"tooltip": _ref_tip(
                    "CLIP / text encoder (Qwen3-VL for H3). Encodes the prompt with the reference presentation (minimax_ref_items).",
                    "CLIP / текст-энкодер (Qwen3-VL для H3). Кодирует промпт с презентацией референсов (minimax_ref_items).")}),
            },
            "optional": optional,
        }
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "LATENT", "width", "height", "fps_int", "fps_float", "total_frames", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        _ref_tip("CONDITIONING: encoded prompt with the reference presentation; carries 'minimax_refs' blocks for the H3 DiT.",
                 "CONDITIONING: закодированный промпт с презентацией референсов; несёт блоки 'minimax_refs' для DiT H3."),
        _ref_tip("Empty joint audio-video latent: video [B,24,latent_t,H//16,W//16] + audio [B,32,2,audio_t] NestedTensor.",
                 "Пустой совместный аудио-видео латент: видео [B,24,latent_t,H//16,W//16] + аудио [B,32,2,audio_t] NestedTensor."),
        _ref_tip("Generation frame width, multiple of 32.", "Ширина кадра генерации, кратна 32."),
        _ref_tip("Generation frame height, multiple of 32.", "Высота кадра генерации, кратна 32."),
        _ref_tip("FPS as integer, always 24.", "FPS целым, всегда 24."),
        _ref_tip("FPS as float, always 24.0.", "FPS дробным, всегда 24.0."),
        _ref_tip("Total frame count aligned to 17N+5.", "Общее число кадров, выровненное до 17N+5."),
        _ref_tip("Exact clip duration in seconds (total_frames / 24).", "Точная длительность клипа в секундах (total_frames / 24)."),
    )
    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count, ref_image_size, prompt,
             clip, vae=None, audio_vae=None, **refs):
        # 1. Калькулятор размера / Size calculator
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)
        fps_int, fps_float = 24, 24.0
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        duration_seconds = total / 24.0
        # 2. Пустой AV-латент / Empty AV latent
        latent, frame_count = _empty_av_latent(width, height, total)
        # 3. Референсы (логика H3 ref2va) / References (H3 ref2va logic)
        ref_items = []
        ref_blocks = []
        for i in range(9):
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
            video_frames = _video_to_frames(video_frames)
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
        for i in range(3):
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
        return (cond, latent, width, height, fps_int, fps_float, total, duration_seconds)
#========================================================================
# Нода 2: AGSoft MiniMax H3 I2V
#========================================================================
class AGSoft_MiniMax_H3_I2V:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    DESCRIPTION = (
        "Local Image-to-Video (t2va/fl2va) conditioning for MiniMax H3 with the AGSoft calculator.\n"
        "• Size and frames as in Ref2V; invert in ALL modes; duration_seconds = total/24.\n"
        "• first_frame = geometry anchor (plain stretch, frame 0); last_frame = follower "
        "(center cover-crop, last frame); both → minimax_keyframes; no frames = pure text-to-video.\n"
        "• vae is OPTIONAL: without it keyframes only condition the text encoder.\n"
        "---\n"
        "Локальное кондиционирование Image-to-Video (t2va/fl2va) для MiniMax H3 с калькулятором AGSoft.\n"
        "• Размер и кадры как в Ref2V; инверсия во ВСЕХ режимах; duration_seconds = total/24.\n"
        "• first_frame = геометрический якорь (растягивание, кадр 0); last_frame = последователь "
        "(cover-crop по центру, последний кадр); оба → minimax_keyframes; без кадров = чистый text-to-video.\n"
        "• vae ОПЦИОНАЛЕН: без него ключевые кадры кондиционируют только текст-энкодер."
    )
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_calc_widgets(),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _ref_tip(
                    "Text prompt describing the scene and motion. first_frame only = motion starts from it; last_frame only = ends there; both = interpolation; none = text-to-video.",
                    "Текстовый промпт со сценой и движением. Только first_frame = движение начинается с него; только last_frame = заканчивается на нём; оба = интерполяция; ни одного = text-to-video.")}),
                "clip": ("CLIP", {"tooltip": _ref_tip(
                    "CLIP / text encoder (Qwen3-VL for H3). Encodes the prompt with the keyframe images.",
                    "CLIP / текст-энкодер (Qwen3-VL для H3). Кодирует промпт с изображениями ключевых кадров.")}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": _ref_tip(
                    "Video VAE (24ch latent). OPTIONAL: without it keyframes only condition the text encoder (no minimax_keyframes blocks).",
                    "Видео-VAE (24ch латент). ОПЦИОНАЛЕН: без него ключевые кадры кондиционируют только текст-энкодер (без блоков minimax_keyframes).")}),
                "first_frame": ("IMAGE", {"tooltip": _ref_tip(
                    "Optional first keyframe. Geometry anchor: plain stretch to the canvas, frame 0, encoded into minimax_keyframes.",
                    "Опциональный первый ключевой кадр. Геометрический якорь: растягивание по холсту, кадр 0, кодируется в minimax_keyframes.")}),
                "last_frame": ("IMAGE", {"tooltip": _ref_tip(
                    "Optional last keyframe. Follower: center cover-crop to the canvas, last frame, encoded into minimax_keyframes.",
                    "Опциональный последний ключевой кадр. Последователь: cover-crop по центру, последний кадр, кодируется в minimax_keyframes.")}),
            },
        }
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "LATENT", "width", "height", "fps_int", "fps_float", "total_frames", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        _ref_tip("CONDITIONING: encoded prompt (+ keyframe images); carries 'minimax_keyframes' blocks when keyframes are connected.",
                 "CONDITIONING: закодированный промпт (+ ключевые кадры); несёт блоки 'minimax_keyframes' при подключённых кадрах."),
        _ref_tip("Empty joint audio-video latent: video [B,24,latent_t,H//16,W//16] + audio [B,32,2,audio_t] NestedTensor.",
                 "Пустой совместный аудио-видео латент: видео [B,24,latent_t,H//16,W//16] + аудио [B,32,2,audio_t] NestedTensor."),
        _ref_tip("Generation frame width, multiple of 32.", "Ширина кадра генерации, кратна 32."),
        _ref_tip("Generation frame height, multiple of 32.", "Высота кадра генерации, кратна 32."),
        _ref_tip("FPS as integer, always 24.", "FPS целым, всегда 24."),
        _ref_tip("FPS as float, always 24.0.", "FPS дробным, всегда 24.0."),
        _ref_tip("Total frame count aligned to 17N+5.", "Общее число кадров, выровненное до 17N+5."),
        _ref_tip("Exact clip duration in seconds (total_frames / 24).", "Точная длительность клипа в секундах (total_frames / 24)."),
    )
    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source,
             length_seconds, frame_count, prompt, clip, vae=None,
             first_frame=None, last_frame=None):
        # 1. Калькулятор размера / Size calculator
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)
        fps_int, fps_float = 24, 24.0
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        duration_seconds = total / 24.0
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
        if keyframes and vae is not None:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
        return (cond, latent, width, height, fps_int, fps_float, total, duration_seconds)
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