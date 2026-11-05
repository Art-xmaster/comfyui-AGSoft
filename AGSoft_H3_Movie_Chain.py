"""
==============================================================================
AGSoft_H3_Movie_Chain.py
==============================================================================
Ноды / Nodes:
🎬AGSoft H3 Ref2V Movie Master — голова цепи: глобальные ресурсы фильма и
   сцена 1; кодирует референсы один раз на весь фильм.
🎬AGSoft H3 Ref2V Movie Link — звено цепи: ровно одна следующая сцена; всё
   глобальное читает из шин, своих ресурсов не держит.
Описание / Description:
Цепочечная раскадровка фильма MiniMax H3 на Ref-модели (Ref2V): без рефов
нода работает как text-to-video. Master объявляет клип: clip, vae, audio_vae,
размер кадра (калькулятор Preset/Custom/Megapixels, кратность 32), fps фильма,
старт исходной дорожки и полный набор референсов (9 image, 3 video с парными
дорожками, 3 audio). Референсы кодируются один раз и едут по шине chain_base
вместе с ресурсами, нормализованной волной источника, курсором аудио и
накопленными сырыми кусками. Шина chain_advanced везёт параметры семплирования
и декода (steps, cfg, sampler, scheduler, denoise, decode_mode, tile_size,
overlap, temporal_size, temporal_overlap); у Link её вход опционален:
подключил — виджеты схлопываются и сцена живёт на значениях шины, отключил —
правишь сцену вручную.
Каждая сцена проходит один конвейер: калькулятор кадров (17N+5) → Ref2V-
кондиционирование поверх презентации референсов → ключевые кадры
minimax_keyframes (липсинк-гайд кадра 0 от отрезанного куска source_audio плюс
мульти-гайды в стиле 📍Add Guide с проверкой границ, клипами 17k+5 и кропом
аудио по остатку дорожки) → встроенное семплирование (логика 🧩KSampler Plus,
кэш по ноде: смена виджетов декода не пересемплирует, только передекодирует) →
декод VAE tiled/standard/off с подробным логом плиток и VRAM → срезы
продолжения last_frame (самый последний кадр) и context_images (последние N).
Аудио: source_audio режется по сценам встык от source_audio_start_seconds
(Master) и от курсора шины (Link); source_audio_cut отдаёт сырой точный кусок
сцены для сведения, stitched_audio собирает куски вверх по цепи (на последнем
Link — полный трек фильма), audio — саундтрек, сгенерированный моделью.
---
Chain-based MiniMax H3 movie storyboard on the Ref model (Ref2V): without refs
the node works as text-to-video. Master declares the clip: clip, vae,
audio_vae, frame size (Preset/Custom/Megapixels calculator, multiples of 32),
movie fps, source track start and the full reference set (9 image, 3 video
with paired soundtracks, 3 audio). References encode once and travel
chain_base together with resources, the normalized source wave, the audio
cursor and the accumulated raw pieces. The chain_advanced bus carries sampling
and decode params (steps, cfg, sampler, scheduler, denoise, decode_mode,
tile_size, overlap, temporal_size, temporal_overlap); on Link its input is
optional: linked — widgets collapse and the scene runs on bus values,
unlinked — the scene is tuned manually.
Every scene passes one pipeline: frame calculator (17N+5) → Ref2V
conditioning over the reference presentation → minimax_keyframes keyframes
(frame-0 lip sync guide from the cut source_audio piece plus multi guides in
📍Add Guide style with bounds check, 17k+5 clips and audio crop by track
remainder) → built-in sampling (🧩KSampler Plus logic, per-node cache: decode
widget changes re-decode only, no re-sampling) → VAE decode tiled/standard/off
with verbose tile and VRAM log → continuation slices last_frame (the very last
frame) and context_images (last N).
Audio: source_audio is cut per scene contiguously from
source_audio_start_seconds (Master) and from the bus cursor (Link);
source_audio_cut gives the raw exact scene piece for muxing, stitched_audio
assembles pieces up the chain (full movie track on the last Link), audio is
the model-generated soundtrack.
Возможности / Features:
⚡ Только Ref2V: без подключённых рефов сцена генерируется как text-to-video
   Ref2V only: with no refs linked the scene generates as text-to-video
⚡ Две шины: chain_base (ресурсы, fps, рефы, курсор и куски аудио) и
   chain_advanced (семплинг и декод); у Link chain_advanced опциональна
   Two buses: chain_base (resources, fps, refs, audio cursor and pieces) and
   chain_advanced (sampling and decode); on Link chain_advanced is optional
⚡ Референсы кодируются один раз на Master и переиспользуются всеми сценами
   References encode once on Master and are reused by every scene
⚡ Липсинк сцены: отрезанный кусок source_audio входит в minimax_keyframes
   как аудио-гайд кадра 0; мульти-гайды как в 📍Add Guide (границы, 17k+5)
   Scene lip sync: the cut source_audio piece enters minimax_keyframes as a
   frame-0 audio guide; multi guides like 📍Add Guide (bounds, 17k+5)
⚡ Встроенный семплер и декод с кэшем по ноде и логом плиток/VRAM как в
   🧩KSampler Plus; без model нода отдаёт conditioning и пустой латент
   Built-in sampler and decode with per-node cache and tile/VRAM log like
   🧩KSampler Plus; without model the node yields conditioning and empty latent
⚡ Срезы продолжения: last_frame (самый последний кадр), context_images
   (последние N), latent для апскейла и склейки
   Continuation slices: last_frame (the very last frame), context_images
   (last N), latent for upscale and stitch
⚡ UI: виджеты сверху (калькулятор, fps над seed, семплер, декод,
   ref_image_size, context_frames, группа AddGuide), prompt внизу; живая
   инфострока формата Base только на Master, между виджетами и prompt
   UI: widgets on top (calculator, fps above seed, sampler, decode,
   ref_image_size, context_frames, AddGuide group), prompt at the bottom;
   the Base-format live line on Master only, between widgets and prompt
⚡ Автоскрытие: виджеты калькулятора по режимам, пары секунд/кадры гайдов по
   счётчику и position_mode, виджеты семплера/декода на Link при подключённой
   chain_advanced; динамические сокеты рефов (подключено+1) и гайдов
   Auto-hiding: calculator widgets per mode, guide seconds/frames pairs per
   counter and position_mode, sampler/decode widgets on Link while
   chain_advanced is linked; dynamic ref (connected+1) and guide sockets
Автор / Author: AGSoft
Дата / Date: 05.11.2026
==============================================================================
"""

import math
import time
import logging
import torch
import comfy.utils
import comfy.audio
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.nested_tensor
import node_helpers
from nodes import common_ksampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft H3 Movie Chain] v11.05 loaded (source_audio_start_seconds, detailed headers)")

FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_GUIDES = 16
DEFAULT_CONTEXT_FRAMES = 5
MAX_CONTEXT_FRAMES = 32
PLACEHOLDER_SIZE = 64
DEFAULT_AUDIO_SAMPLE_RATE = 32000
BASE_TYPE = "AGS_MOVIE_BASE"
ADV_TYPE = "AGS_MOVIE_ADV"
LOG_PREFIX = "[AGSoft H3 Movie Chain]"
_SAMPLE_CACHE = {}
_CACHE_LIMIT = 8

_T = lambda en, ru: en + "\n---\n" + ru

PRESET_LIST = [
    "512×512 (1:1)", "576×576 (1:1)", "640×640 (1:1)", "704×704 (1:1)",
    "768×768 (1:1)", "832×832 (1:1)", "896×896 (1:1)", "960×960 (1:1)",
    "1024×1024 (1:1)", "1280×1280 (1:1)",
    "480×320 (3:2)", "576×384 (3:2)", "672×448 (3:2)", "768×512 (3:2)",
    "864×576 (3:2)", "960×640 (3:2)", "1056×704 (3:2)", "1152×768 (3:2)",
    "1248×832 (3:2)", "1536×1024 (3:2)",
    "512×384 (4:3)", "640×480 (4:3)", "768×576 (4:3)", "896×672 (4:3)",
    "1024×768 (4:3)", "1152×864 (4:3)", "1280×960 (4:3)", "1408×1056 (4:3)",
    "1536×1152 (4:3)", "2048×1536 (4:3)",
    "512×288 (16:9)", "640×352 (16:9)", "768×448 (16:9)", "896×512 (16:9)",
    "1024×576 (16:9)", "1152×640 (16:9)", "1280×704 (16:9)", "1536×864 (16:9)",
    "1920×1088 (16:9)", "2048×1152 (16:9)",
]

PRESET_MAP = {
    "512×512 (1:1)": (512, 512), "576×576 (1:1)": (576, 576),
    "640×640 (1:1)": (640, 640), "704×704 (1:1)": (704, 704),
    "768×768 (1:1)": (768, 768), "832×832 (1:1)": (832, 832),
    "896×896 (1:1)": (896, 896), "960×960 (1:1)": (960, 960),
    "1024×1024 (1:1)": (1024, 1024), "1280×1280 (1:1)": (1280, 1280),
    "480×320 (3:2)": (480, 320), "576×384 (3:2)": (576, 384),
    "672×448 (3:2)": (672, 448), "768×512 (3:2)": (768, 512),
    "864×576 (3:2)": (864, 576), "960×640 (3:2)": (960, 640),
    "1056×704 (3:2)": (1056, 704), "1152×768 (3:2)": (1152, 768),
    "1248×832 (3:2)": (1248, 832), "1536×1024 (3:2)": (1536, 1024),
    "512×384 (4:3)": (512, 384), "640×480 (4:3)": (640, 480),
    "768×576 (4:3)": (768, 576), "896×672 (4:3)": (896, 672),
    "1024×768 (4:3)": (1024, 768), "1152×864 (4:3)": (1152, 864),
    "1280×960 (4:3)": (1280, 960), "1408×1056 (4:3)": (1408, 1056),
    "1536×1152 (4:3)": (1536, 1152), "2048×1536 (4:3)": (2048, 1536),
    "512×288 (16:9)": (512, 288), "640×352 (16:9)": (640, 352),
    "768×448 (16:9)": (768, 448), "896×512 (16:9)": (896, 512),
    "1024×576 (16:9)": (1024, 576), "1152×640 (16:9)": (1152, 640),
    "1280×704 (16:9)": (1280, 704), "1536×864 (16:9)": (1536, 864),
    "1920×1088 (16:9)": (1920, 1088), "2048×1152 (16:9)": (2048, 1152),
}

ASPECT_RATIOS = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"]


def fit_to_multiple(value: int, multiple: int = 32) -> int:
    """Округляет ВВЕРХ до кратного multiple / Rounds UP to a multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def fit_length_to_17n5(value: int) -> int:
    """Выравнивает ВВЕРХ до 17N+5 / Aligns UP to 17N+5."""
    v = max(5, int(value))
    return v + (5 - v % 17) % 17


def video_latent_t(frame_count: int) -> int:
    """Temporal-размер видео-латента H3 / H3 video latent temporal size."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _calc_size(mode, preset, invert_orientation, custom_width, custom_height,
               megapixels_value, aspect_ratio):
    """Ширина/высота по режиму, кратны 32 / Width/height per mode, mult of 32."""
    if mode == "Preset":
        w, h = PRESET_MAP[preset]
    elif mode == "Custom":
        w, h = custom_width, custom_height
    else:
        w_r, h_r = map(int, aspect_ratio.split(":"))
        x = math.sqrt(megapixels_value * 1_000_000 / (w_r * h_r))
        w, h = round(w_r * x), round(h_r * x)
    if invert_orientation:
        w, h = h, w
    return max(64, fit_to_multiple(w, 32)), max(64, fit_to_multiple(h, 32))


def _calc_frames(frame_count_source, length_seconds, frame_count):
    """Кадры 17N+5 по источнику / Frame count 17N+5 per source."""
    if frame_count_source == "Frames":
        return fit_length_to_17n5(frame_count)
    return fit_length_to_17n5(round(length_seconds * 24))


def _resize(image, width, height, crop):
    """Ресайз тензора изображения lanczos / Resize image tensor with lanczos."""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    """Пустой совместный AV-латент H3 / Empty joint H3 AV latent."""
    frame_count = fit_length_to_17n5(length)
    latent_t = video_latent_t(frame_count)
    audio_t = round(frame_count / FPS * AUDIO_LATENT_FPS)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _vae_kind(vae):
    """Возвращает 'audio' или 'video' по классу и модулю модели.
    Returns 'audio' or 'video' by model class name and module."""
    if vae is None:
        return None
    fsm = vae.first_stage_model if hasattr(vae, "first_stage_model") else None
    if fsm is None:
        return None
    tag = (type(fsm).__name__ + " " + str(type(fsm).__module__)).lower()
    return "audio" if "audio" in tag else "video"


def _sort_vaes(vae, audio_vae):
    """Ставит VAE по своим сокетам, даже если их перепутали местами.
    Puts VAEs into their sockets even if connected swapped."""
    if _vae_kind(vae) == "audio" and _vae_kind(audio_vae) != "audio":
        logger.warning("%s VAE sockets were swapped, auto-corrected", LOG_PREFIX)
        return audio_vae, vae
    return vae, audio_vae


def _video_latent(vae, pixels):
    """Кодирует пиксели видео-VAE с проверкой типа модели.
    Encodes pixels with the video VAE after a model kind check."""
    if _vae_kind(vae) != "video":
        raise ValueError("Image encoding needs the VIDEO VAE in the vae socket / "
                         "для изображений нужен VIDEO VAE в сокете vae")
    return vae.encode(pixels)


def _norm_waveform(wav):
    """Нормализует волну к [B, C, T] и клампит / Normalizes wave to [B, C, T]."""
    wav = wav.detach().cpu().float()
    if wav.dim() == 1:
        wav = wav.view(1, 1, -1)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(1)
    elif wav.dim() > 3:
        wav = wav.reshape(-1, wav.shape[-2], wav.shape[-1])
    return wav.clamp(-1.0, 1.0)


def _encode_ref_audio(audio_vae, audio):
    """Ресемплирует до частоты audio_vae и кодирует в 32ch латент; вход
    подаётся раскладкой [B, T, C] как в рабочих нодах AGSoft MiniMax H3.
    Resamples to the audio_vae rate and encodes into a 32ch latent; the input
    uses the [B, T, C] layout as in the proven AGSoft MiniMax H3 nodes."""
    if _vae_kind(audio_vae) != "audio":
        raise ValueError("Reference audio needs the AUDIO VAE in the audio_vae socket / "
                         "для референс-аудио нужен AUDIO VAE в сокете audio_vae")
    waveform = _norm_waveform(audio["waveform"])
    sr = int(audio.get("sample_rate", 44100))
    vae_sr = audio_vae.audio_sample_rate if hasattr(audio_vae, "audio_sample_rate") else 32000
    if sr != vae_sr:
        waveform = comfy.audio.resample(waveform, sr, vae_sr)
    if waveform.shape[-1] < 1:
        waveform = torch.zeros((*waveform.shape[:-1], vae_sr), dtype=waveform.dtype)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _slice_piece(wav, sr, start, length):
    """Сэмпл-точный срез волны / Sample-accurate waveform slice."""
    want = max(1, int(round(length * sr)))
    a = int(round(start * sr))
    b = a + want
    n = wav.shape[-1]
    if a >= n:
        return torch.zeros((*wav.shape[:-1], want), dtype=wav.dtype), True
    piece = wav[..., a:min(b, n)]
    padded = False
    if piece.shape[-1] < want:
        pad = torch.zeros((*piece.shape[:-1], want - piece.shape[-1]), dtype=piece.dtype)
        piece = torch.cat([piece, pad], dim=-1)
        padded = True
    return piece, padded


def _source_track(seconds, sr):
    """Тишина-фолбэк дорожки сцены / Silence fallback for the scene track."""
    want = max(1, int(round(seconds * sr)))
    return {"waveform": torch.zeros(1, 1, want), "sample_rate": sr}


def _audio_dict(piece, sr):
    """Собирает AUDIO-словарь ComfyUI / Builds a ComfyUI AUDIO dict."""
    return {"waveform": piece, "sample_rate": sr}


def _placeholder_image():
    """Пустой плейсхолдер кадра (чётный) / Empty even-size frame placeholder."""
    return torch.zeros((1, PLACEHOLDER_SIZE, PLACEHOLDER_SIZE, 3), dtype=torch.float32)


def _gen_audio_silence(audio_vae=None):
    """Плейсхолдер сгенерированного аудио / Generated-audio placeholder."""
    sr = audio_vae.audio_sample_rate if hasattr(audio_vae, "audio_sample_rate") else None
    return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
            "sample_rate": int(sr) if sr else DEFAULT_AUDIO_SAMPLE_RATE}


def _video_to_frames(video):
    """Приводит вход к тензору кадров [F, H, W, C] / Normalizes to [F, H, W, C]."""
    t = None
    if isinstance(video, torch.Tensor):
        t = video
    elif hasattr(video, "get_pixels"):
        t = video.get_pixels()
    if t is None:
        raise TypeError(f"Unsupported ref_video input type: {type(video).__name__}")
    if t.dim() == 5:
        t = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])
    t = t.detach().cpu()
    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return t


def _to_frames(img):
    """Приводит IMAGE к 4D [F, H, W, C] / Normalizes IMAGE to 4D [F, H, W, C]."""
    if img.dim() == 5:
        b, f, h, w, c = img.shape
        return img.reshape(b * f, h, w, c)
    return img


def adapt_canvas(width, height):
    """Холст референс-видео 768 / Reference video canvas 768."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _guide_frames(image):
    """Кадры гайда: одиночный или клип 17k+5 / Guide frames: single or 17k+5 clip."""
    n = image.shape[0]
    if n < 5:
        return 1
    while n % 17 != 5:
        n -= 1
    return n


def _build_refs(width, height, vae, audio_vae, refs, ref_image_size):
    """Кодирует глобальные референсы один раз на весь фильм.
    Encodes the global references once for the whole movie."""
    ref_items, ref_blocks = [], []
    needs_video_vae = False
    for i in range(MAX_REF_IMAGES):
        img = refs.get(f"ref_image_{i}")
        if img is None:
            continue
        needs_video_vae = True
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
            z = _video_latent(vae, resized)
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})
    for i in range(MAX_REF_VIDEOS):
        video_frames = refs.get(f"ref_video_{i}")
        if video_frames is None:
            continue
        needs_video_vae = True
        soundtrack = refs.get(f"ref_video_audio_{i}")
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None and audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
        if audio_latent is not None:
            ref_items.append({"type": "audio"})
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
            raise ValueError("Ref videos need at least 5 frames / минимум 5 кадров")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_idx],
                          "timestamps": [k / 2.0 for k in range(len(sample_idx))]})
        if vae is None:
            continue
        z = _video_latent(vae, frames)
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16,
                           "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                           "latent": z, "audio_latent": audio_latent})
    for i in range(MAX_REF_AUDIOS):
        audio = refs.get(f"ref_audio_{i}")
        if audio is None:
            continue
        audio_latent, ref_audio_t = (None, 0)
        if audio_vae is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        if audio_latent is None:
            continue
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})
    if needs_video_vae and vae is None:
        raise ValueError("Reference mode needs the VIDEO VAE in the vae socket / "
                         "режим референсов требует VIDEO VAE в сокете vae")
    return ref_items, ref_blocks


def _gen_ref2v(width, height, total, prompt, clip, ref_items, ref_blocks):
    """Промпт сцены поверх презентации референсов / Scene prompt over ref presentation."""
    latent, frame_count = _empty_av_latent(width, height, total)
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items or [])
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    return cond, latent, frame_count


def _collect_keyframes(cond):
    """Читает текущий список minimax_keyframes / Reads current minimax_keyframes."""
    return list(cond[0][1].get("minimax_keyframes", []))


def _set_keyframes(cond, keyframes):
    """Пишет minimax_keyframes обратно, если список не пуст.
    Writes minimax_keyframes back when the list is not empty."""
    if not keyframes:
        return cond
    return node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})


def _source_lip_sync(keyframes, audio_vae, piece, sr, audio_t):
    """Кусок source_audio как аудио-гайд кадра 0 (липсинк сцены).
    The source_audio piece as a frame-0 audio guide (scene lip sync)."""
    if piece is None or audio_vae is None:
        return
    z, rt = _encode_ref_audio(audio_vae, {"waveform": piece, "sample_rate": sr})
    max_rt = max(1, math.floor(audio_t))
    if rt > max_rt:
        z = z[..., :max_rt].clone()
    keyframes.append({"resolved_frame_index": 0, "audio_latent": z})


def _apply_guides(keyframes, width, height, fc, audio_t, vae, audio_vae,
                  number_of_guides, position_mode, fps, slots):
    """Мульти-гайды как в 📍AGSoft Add Guide: границы, клипы 17k+5, кроп аудио.
    Multi guides like 📍AGSoft Add Guide: bounds, 17k+5 clips, audio crop."""
    try:
        from comfy.ldm.minimax.model import FRAME_RESCALE
    except Exception:
        FRAME_RESCALE = 0.25
    for i in range(1, int(number_of_guides) + 1):
        img = slots.get(f"image_{i}")
        aud = slots.get(f"audio_{i}")
        if img is None and aud is None:
            continue
        if position_mode == "Seconds":
            value = float(slots.get(f"seconds_{i}") or 0.0)
            fi = -1 if value < 0 else int(round(value * fps))
        else:
            value = int(slots.get(f"frames_{i}") or 0)
            fi = -1 if value < 0 else int(value)
        resolved = fi if fi >= 0 else fc + fi
        gf = _guide_frames(img) if img is not None else 1
        if resolved < 0 or resolved + gf > fc:
            raise ValueError(f"guide #{i}: frame {resolved} + {gf} outside {fc} / "
                             f"гайд #{i}: кадр {resolved} + {gf} вне {fc}")
        kf = {"resolved_frame_index": resolved}
        if img is not None:
            if vae is None:
                raise ValueError(f"guide #{i}: image needs the VIDEO VAE / "
                                 f"гайд #{i}: изображению нужен VIDEO VAE")
            kf["latent"] = _video_latent(vae, _resize(img[:gf], width, height, "center"))
        if aud is not None:
            if audio_vae is None:
                raise ValueError(f"guide #{i}: audio needs the AUDIO VAE / "
                                 f"гайд #{i}: аудио требует AUDIO VAE")
            z, rt = _encode_ref_audio(audio_vae, aud)
            max_rt = math.floor(audio_t - FRAME_RESCALE * resolved)
            if max_rt < 1:
                raise ValueError(f"guide #{i}: frame {resolved} past audio track end / "
                                 f"гайд #{i}: кадр {resolved} за концом аудиодорожки")
            if rt > max_rt:
                z = z[..., :max_rt].clone()
            kf["audio_latent"] = z
        keyframes.append(kf)


def _vram_stats():
    """Снимок VRAM (MB): использовано и свободно / VRAM snapshot (MB): used and free."""
    if not torch.cuda.is_available():
        return None
    try:
        used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        return (used_mb, free_mb)
    except Exception:
        return None


def _sig_tuple(sigmas):
    """Сигмы кортежем float для ключа кэша / Sigmas as a float tuple for the cache key."""
    if sigmas is None:
        return None
    if isinstance(sigmas, torch.Tensor):
        return tuple(float(x) for x in sigmas.detach().cpu().flatten().tolist())
    return tuple(float(x) for x in sigmas)


def _sampling_key(model, positive, negative, latent_image, sigmas,
                  seed, steps, cfg, sampler_name, scheduler, denoise):
    """Ключ фазы семплирования: идентичность входов + параметры.
    Sampling-phase key: identity of upstream inputs plus parameters."""
    refs = (model, positive, negative, latent_image, sigmas)
    params = (int(seed), int(steps), float(cfg), str(sampler_name), str(scheduler),
              float(denoise), _sig_tuple(sigmas))
    return (refs, params)


def _key_match(entry, key):
    """Сравнение ключа кэша по идентичности ссылок и параметрам.
    Cache key comparison by reference identity and parameters."""
    refs, params = key
    erefs, eparams = entry["refs"], entry["params"]
    if len(refs) != len(erefs):
        return False
    for a, b in zip(refs, erefs):
        if a is not b:
            return False
    return params == eparams


def _create_empty_cond(positive):
    """Пустой конд на основе структуры позитивного.
    Empty conditioning based on the positive structure."""
    empty = []
    for cond_pair in positive:
        if isinstance(cond_pair, (list, tuple)) and len(cond_pair) >= 1:
            cond_tensor = cond_pair[0]
            extra = cond_pair[1] if len(cond_pair) > 1 else {}
            empty_tensor = torch.zeros_like(cond_tensor) if isinstance(cond_tensor, torch.Tensor) else cond_tensor
            empty_extra = {}
            if isinstance(extra, dict):
                for key, value in extra.items():
                    empty_extra[key] = torch.zeros_like(value) if isinstance(value, torch.Tensor) else value
            empty.append([empty_tensor, empty_extra])
        else:
            empty.append(cond_pair)
    return empty


def _split_av_latent(latent_dict):
    """H3: NestedTensor((video, audio)) -> (video, audio) / unpack the joint latent."""
    samples = latent_dict["samples"] if isinstance(latent_dict, dict) else latent_dict
    tensors = samples.tensors if hasattr(samples, "tensors") else None
    if tensors is None and (samples.is_nested if hasattr(samples, "is_nested") else False):
        try:
            tensors = samples.unbind()
        except Exception:
            tensors = None
    if tensors is not None and len(tensors) >= 2:
        return tensors[0], tensors[1]
    if tensors is not None and len(tensors) == 1:
        return tensors[0], None
    if isinstance(latent_dict, dict) and "audio" in latent_dict:
        return samples, latent_dict["audio"]
    return samples, None


def _normalize_audio(decoded, audio_vae=None):
    """Нормализует аудио к стандарту ComfyUI [B, C, N] / Normalize audio to [B, C, N]."""
    if decoded is None:
        return None
    waveform, sample_rate = None, None
    if isinstance(decoded, dict):
        waveform = decoded.get("waveform", None)
        sample_rate = decoded.get("sample_rate", None)
    elif isinstance(decoded, torch.Tensor):
        waveform = decoded
    if waveform is None:
        return None
    if sample_rate is None:
        sample_rate = audio_vae.audio_sample_rate if hasattr(audio_vae, "audio_sample_rate") else None
    if sample_rate is None:
        sample_rate = audio_vae.sample_rate if hasattr(audio_vae, "sample_rate") else None
    if sample_rate is None:
        sample_rate = DEFAULT_AUDIO_SAMPLE_RATE
    if (waveform.is_nested if hasattr(waveform, "is_nested") else False) or hasattr(waveform, "tensors"):
        try:
            waveform = waveform.tensors[0] if hasattr(waveform, "tensors") else waveform.to_dense()
        except Exception:
            return None
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 3:
        b, d1, d2 = waveform.shape
        if d2 <= 8 < d1:
            waveform = waveform.movedim(-1, 1)
    if waveform.dim() != 3:
        logger.warning("%s unexpected audio waveform dims: %d", LOG_PREFIX, waveform.dim())
        return None
    return {"waveform": waveform.detach().cpu().float().contiguous(),
            "sample_rate": int(sample_rate)}


def _decode_audio(audio_vae, audio_latent):
    """Декод аудио-латента через audio_vae / Decode the audio latent via audio_vae."""
    decoded = None
    try:
        if hasattr(audio_vae, "decode_audio"):
            decoded = audio_vae.decode_audio(audio_latent)
        elif hasattr(audio_vae, "decode"):
            decoded = audio_vae.decode(audio_latent)
    except Exception as e:
        logger.warning("%s audio decode error: %s: %s", LOG_PREFIX, type(e).__name__, e)
        return None
    return _normalize_audio(decoded, audio_vae)


def _decode_tiled_logged(vae, latent, tile_size, overlap, temporal_size, temporal_overlap):
    """Плиточный декод с логом реальных параметров / Tiled decode logging real params."""
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    t_comp = vae.temporal_compression_decode() if hasattr(vae, "temporal_compression_decode") else None
    if t_comp is not None:
        ts = max(2, int(temporal_size) // int(t_comp))
        to = max(1, min(ts // 2, int(temporal_overlap) // int(t_comp)))
    else:
        ts, to = None, None
    s_comp = vae.spacial_compression_decode() if hasattr(vae, "spacial_compression_decode") else 8
    if not s_comp or int(s_comp) < 1:
        s_comp = 8
    tx = max(1, int(tile_size) // int(s_comp))
    ov = max(0, int(overlap) // int(s_comp))
    tile_info = {"tile_px": int(tile_size), "overlap_px": int(overlap),
                 "tile_lat": tx, "overlap_lat": ov, "spacial": int(s_comp),
                 "temporal": int(t_comp) if t_comp is not None else None,
                 "t_frames": ts, "t_overlap": to}
    try:
        images = vae.decode_tiled(latent, tile_x=tx, tile_y=tx, overlap=ov,
                                  tile_t=ts, overlap_t=to)
    except TypeError:
        images = vae.decode_tiled(latent, tile_x=tx, tile_y=tx, overlap=ov)
        tile_info["t_frames"] = None
        tile_info["t_overlap"] = None
    return images, tile_info


def _decode_and_log(vae, latent, mode, tile_size, overlap, temporal_size,
                    temporal_overlap, tag):
    """Обёртка декода с подробным логом / Decode wrapper with a verbose log."""
    samples = latent["samples"] if isinstance(latent, dict) else latent
    if hasattr(samples, "is_nested") and samples.is_nested:
        try:
            samples = samples.unbind()[0]
        except Exception:
            pass
    shape = tuple(int(x) for x in samples.shape) if hasattr(samples, "shape") else "?"
    logger.info("%s VAE decode start | mode=%s | latent shape=%s", tag, mode, shape)
    vram_before = _vram_stats()
    t0 = time.time()
    try:
        if mode == "standard":
            images = vae.decode(samples)
            actual_mode = "standard"
            tile_info = None
        else:
            try:
                images, tile_info = _decode_tiled_logged(vae, samples, tile_size, overlap,
                                                         temporal_size, temporal_overlap)
                actual_mode = "tiled"
            except Exception as e:
                logger.warning("%s tiled decode failed, fallback standard: %s: %s",
                               tag, type(e).__name__, e)
                t1 = time.time()
                images = vae.decode(samples)
                actual_mode = "standard (fallback after tiled error)"
                tile_info = None
                logger.info("%s fallback standard decode in %.2fs", tag, time.time() - t1)
    except Exception as e:
        logger.warning("%s VAE decode error: %s: %s", tag, type(e).__name__, e)
        return None
    dt = time.time() - t0
    if images is None:
        logger.warning("%s VAE decode returned None in %.2fs", tag, dt)
        return None
    if images.dim() == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    out_shape = tuple(int(x) for x in images.shape)
    vram_after = _vram_stats()
    mem_line = ""
    if vram_before is not None and vram_after is not None:
        delta = vram_after[0] - vram_before[0]
        mem_line = (f" | VRAM used {vram_before[0]:.0f} -> {vram_after[0]:.0f} MB "
                    f"(delta {delta:+.0f}), free {vram_after[1]:.0f} MB")
    tiles_line = ""
    if tile_info:
        tiles_line = (f" | tiles px={tile_info['tile_px']} overlap_px={tile_info['overlap_px']} "
                      f"latent_tiles={tile_info['tile_lat']} overlap_lat={tile_info['overlap_lat']} "
                      f"spacial_comp={tile_info['spacial']}")
        if tile_info.get("temporal") is not None:
            tiles_line += (f" | temporal_comp={tile_info['temporal']} "
                           f"t_frames={tile_info['t_frames']} t_overlap={tile_info['t_overlap']}")
    logger.info("%s VAE decode done | mode=%s | output shape=%s time=%.2fs%s%s",
                tag, actual_mode, out_shape, dt, tiles_line, mem_line)
    return images


def _run_sampling(model, positive, negative, latent_image, sigmas, seed, steps, cfg,
                  sampler_name, scheduler, denoise, tag):
    """Один прогон семплирования без декода / One sampling pass without decode."""
    if negative is None:
        negative = _create_empty_cond(positive)
    t0 = time.time()
    if sigmas is None:
        result = common_ksampler(model=model, seed=seed, steps=steps, cfg=cfg,
                                 sampler_name=sampler_name, scheduler=scheduler,
                                 positive=positive, negative=negative,
                                 latent=latent_image, denoise=denoise)
        latent_dict = result[0]
        steps_used = int(steps)
        scheduler_used = str(scheduler)
        denoise_used = float(denoise)
    else:
        latent_samples = latent_image["samples"]
        noise = comfy.sample.prepare_noise(latent_samples, seed, None)
        sampler = comfy.samplers.sampler_object(sampler_name)
        samples = comfy.samplers.sample(
            model, noise, positive, negative, cfg,
            comfy.model_management.get_torch_device(), sampler, sigmas,
            model_options=model.model_options if hasattr(model, "model_options") else {},
            latent_image=latent_samples, denoise_mask=None, callback=None,
            disable_pbar=False, seed=seed)
        latent_dict = {"samples": samples}
        if isinstance(latent_image, dict):
            for key, value in latent_image.items():
                if key not in latent_dict:
                    latent_dict[key] = value
        steps_used = max(1, len(sigmas) - 1)
        scheduler_used = "manual sigmas"
        denoise_used = float(sigmas[0])
    dt = time.time() - t0
    logger.info("%s sampling done in %.2fs | steps=%d cfg=%.2f sampler=%s scheduler=%s "
                "denoise=%.2f", tag, dt, steps_used, cfg, sampler_name,
                scheduler_used, denoise_used)
    video_lat, audio_lat = _split_av_latent(latent_dict)
    return latent_dict, video_lat, audio_lat


def _sample_cached(uid, model, positive, negative, latent_image, sigmas, seed, steps,
                   cfg, sampler_name, scheduler, denoise, tag):
    """Семплирование с кэшем по ноде: смена виджетов декода не пересемплирует.
    Sampling with per-node cache: decode widget changes do not re-sample."""
    key = _sampling_key(model, positive, negative, latent_image, sigmas,
                        seed, steps, cfg, sampler_name, scheduler, denoise)
    entry = _SAMPLE_CACHE.get(uid) if uid is not None else None
    if entry is not None and _key_match(entry, key):
        logger.info("%s CACHE HIT: sampling skipped, decode re-run only", tag)
        return entry["latent"], entry["video"], entry["audio"]
    latent_dict, video_lat, audio_lat = _run_sampling(
        model, positive, negative, latent_image, sigmas, seed, steps, cfg,
        sampler_name, scheduler, denoise, tag)
    if uid is not None:
        _SAMPLE_CACHE[uid] = {"refs": key[0], "params": key[1], "latent": latent_dict,
                              "video": video_lat, "audio": audio_lat}
        while len(_SAMPLE_CACHE) > _CACHE_LIMIT:
            _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
        logger.info("%s sampling executed, result cached for re-decode", tag)
    return latent_dict, video_lat, audio_lat


def _sample_scene(model, negative, sigmas, latent_empty, cond, seed, steps, cfg,
                  sampler_name, scheduler, denoise, decode_mode, tile_size, overlap,
                  temporal_size, temporal_overlap, vae, audio_vae, uid, tag):
    """Семплирование сцены и декод; без model отдаёт плейсхолдеры.
    Scene sampling and decode; without model returns placeholders."""
    if model is None:
        logger.info("%s sampling skipped: model not connected", tag)
        return latent_empty, _placeholder_image(), _gen_audio_silence(audio_vae), False
    latent_dict, video_lat, audio_lat = _sample_cached(
        uid, model, cond, negative, latent_empty, sigmas, seed, steps, cfg,
        sampler_name, scheduler, denoise, tag)
    images = None
    if vae is None or video_lat is None or str(decode_mode) == "off":
        if vae is None:
            reason = "vae not connected"
        elif video_lat is None:
            reason = "no video latent"
        else:
            reason = "mode=off"
        logger.info("%s VAE decode skipped (%s): images = empty %dx%d placeholder",
                    tag, reason, PLACEHOLDER_SIZE, PLACEHOLDER_SIZE)
    else:
        images = _decode_and_log(vae, video_lat, str(decode_mode), tile_size, overlap,
                                 temporal_size, temporal_overlap, tag)
    if images is None:
        images = _placeholder_image()
        return latent_dict, images, _gen_audio_silence(audio_vae), False
    gen_audio = None
    if audio_vae is not None and audio_lat is not None:
        gen_audio = _decode_audio(audio_vae, audio_lat)
    if gen_audio is None:
        gen_audio = _gen_audio_silence(audio_vae)
    return latent_dict, images, gen_audio, True


def _slice_last_context(images, context_frames, decoded):
    """Последний кадр и последние N кадров как в Slice-нодах.
    The very last frame and the last N frames like the Slice nodes."""
    if not decoded:
        return _placeholder_image(), _placeholder_image()
    pixels = _to_frames(images)
    total = pixels.shape[0]
    ctx = min(int(context_frames), total)
    if ctx < int(context_frames):
        logger.info("%s segment has only %d frames, context reduced to %d",
                    LOG_PREFIX, total, ctx)
    return pixels[-1:], pixels[-ctx:]


def _calc_widgets():
    """Виджеты калькулятора размера / Size calculator widgets."""
    return {
        "mode": (["Preset", "Custom", "Megapixels"], {"default": "Megapixels", "tooltip": _T(
            "Frame size mode. Unused widgets hide automatically (JS).",
            "Режим размера кадра. Лишние виджеты скрываются автоматически (JS).")}),
        "preset": (PRESET_LIST, {"default": "1280×704 (16:9)", "tooltip": _T(
            "Predefined size WIDTHxHEIGHT, multiples of 32.",
            "Готовый размер ШИРИНАxВЫСОТА, кратны 32.")}),
        "invert_orientation": ("BOOLEAN", {"default": False, "tooltip": _T(
            "Swaps width and height in all modes.",
            "Меняет ширину и высоту местами во всех режимах.")}),
        "custom_width": ("INT", {"default": 864, "min": 64, "max": 8192, "step": 32, "tooltip": _T(
            "Frame width in px (Custom), rounded up to 32.",
            "Ширина кадра в px (Custom), округление вверх до 32.")}),
        "custom_height": ("INT", {"default": 480, "min": 64, "max": 8192, "step": 32, "tooltip": _T(
            "Frame height in px (Custom), rounded up to 32.",
            "Высота кадра в px (Custom), округление вверх до 32.")}),
        "megapixels_value": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": _T(
            "Target area in MP; W/H computed from aspect_ratio.",
            "Целевая площадь в MP; W/H считаются по aspect_ratio.")}),
        "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1", "tooltip": _T(
            "Aspect ratio for Megapixels mode.",
            "Соотношение сторон для режима Megapixels.")}),
        "frame_count_source": (["Seconds", "Frames"], {"default": "Seconds", "tooltip": _T(
            "Seconds or exact Frames; aligned UP to 17N+5.",
            "Секунды или точные кадры; выравнивание ВВЕРХ до 17N+5.")}),
        "length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": _T(
            "Scene duration in seconds (Seconds only).",
            "Длительность сцены в секундах (только Seconds).")}),
        "frame_count": ("INT", {"default": 124, "min": 5, "max": 99999, "step": 1, "tooltip": _T(
            "Exact scene frame count (Frames only).",
            "Точное число кадров сцены (только Frames).")}),
    }


def _fps_widget():
    """Виджет fps: только Master, значение идёт по chain_base.
    The fps widget: Master only, its value travels chain_base."""
    return {
        "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                          "tooltip": _T(
            "Movie FPS: converts guide seconds to frames and travels in chain_base "
            "to every Link (H3 native is 24).",
            "FPS фильма: переводит секунды гайдов в кадры и идёт по chain_base во все "
            "Link (родной для H3 — 24).")}),
    }


def _sampler_widgets():
    """Виджеты встроенного семплера / Built-in sampler widgets."""
    return {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                         "control_after_generate": True, "tooltip": _T(
            "The random seed used for creating the noise.",
            "Сид для генерации шума.")}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": _T(
            "Steps count, unless sigmas or chain_advanced provide their own.",
            "Число шагов, если сигмы или chain_advanced не задали свои.")}),
        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                          "round": 0.01, "tooltip": _T(
            "CFG scale. Usually 1.0 for H3.",
            "Масштаб CFG. Обычно 1.0 для H3.")}),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": _T(
            "The algorithm used when sampling.",
            "Алгоритм сэмплирования.")}),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": _T(
            "The scheduler controls how noise is gradually removed.",
            "Планировщик определяет, как удаляется шум.")}),
    }


def _decode_widgets():
    """Виджеты денoйза и встроенного декода / Denoise and built-in decode widgets."""
    return {
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": _T(
            "Denoise amount (used only when sigmas are NOT connected).",
            "Степень денойзинга (только если сигмы НЕ подключены).")}),
        "decode_mode": (["tiled", "standard", "off"], {"default": "standard", "tooltip": _T(
            "How the built-in VAE decodes: tiled (low VRAM), standard (faster), off.",
            "Как встроенный VAE декодирует: tiled (мало VRAM), standard (быстрее), off.")}),
        "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "tooltip": _T(
            "Tiled decode: tile size in pixels (like native VAE Decode (Tiled)).",
            "Плиточный декод: размер плитки в пикселях (как в нативной VAE Decode (Tiled)).")}),
        "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": _T(
            "Tiled decode: overlap between tiles in pixels; auto-clamped to tile_size / 4.",
            "Плиточный декод: перекрытие плиток в пикселях; авто-кламп до tile_size / 4.")}),
        "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": _T(
            "Only used for video VAEs: frames to decode at a time.",
            "Только для видео-VAE: сколько кадров декодировать за проход.")}),
        "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": _T(
            "Only used for video VAEs: frames to overlap; auto-clamped.",
            "Только для видео-VAE: перекрытие в кадрах; авто-кламп.")}),
    }


def _context_widget():
    """Виджет размера контекста продолжения / Continuation context size widget."""
    return {
        "context_frames": ("INT", {"default": DEFAULT_CONTEXT_FRAMES, "min": 1,
                                   "max": MAX_CONTEXT_FRAMES, "step": 1, "tooltip": _T(
            "How many last frames go into context_images (1-32).",
            "Сколько последних кадров отдать в context_images (1-32).")}),
    }


def _guide_widgets():
    """Виджеты группы AddGuide без fps: счётчик, режим, позиции.
    AddGuide group widgets without fps: counter, mode, positions."""
    out = {
        "number_of_guides": ("INT", {"default": 0, "min": 0, "max": MAX_GUIDES, "step": 1,
                                     "tooltip": _T(
            "How many guide slots are active (0 = none); slots hold image/audio pairs.",
            "Сколько слотов гайдов активно (0 = нет); слоты держат пары image/audio.")}),
        "position_mode": (["Seconds", "Frames"], {"default": "Seconds", "tooltip": _T(
            "Guide position: Seconds via the bus fps or Frames directly; -1 = last frame.",
            "Позиция гайдов: Seconds через fps из шины или Frames напрямую; -1 = последний кадр.")}),
    }
    for i in range(1, MAX_GUIDES + 1):
        out[f"seconds_{i}"] = ("FLOAT", {"default": 0.0, "min": -1.0, "max": 3600.0,
                                        "step": 0.05, "tooltip": _T(
            f"Position of guide #{i} in seconds, -1 = last frame.",
            f"Позиция направляющей #{i} в секундах, -1 = последний кадр.")})
        out[f"frames_{i}"] = ("INT", {"default": 0, "min": -1, "max": 9999, "step": 1,
                                      "tooltip": _T(
            f"Position of guide #{i} in frames, -1 = last frame.",
            f"Позиция направляющей #{i} в кадрах, -1 = последний кадр.")})
    return out


def _guide_inputs(optional):
    """Входы гайдов (image/audio) в optional / Guide inputs (image/audio) in optional."""
    for i in range(1, MAX_GUIDES + 1):
        optional[f"image_{i}"] = ("IMAGE", {"tooltip": _T(
            f"Guide image/clip #{i}; batches >=5 frames become 17k+5 clips.",
            f"Направляющее изображение/клип #{i}; батчи >=5 кадров становятся клипами 17k+5.")})
        optional[f"audio_{i}"] = ("AUDIO", {"tooltip": _T(
            f"Guide audio #{i} for lip sync at the guide position.",
            f"Направляющее аудио #{i} для липсинка на позиции гайда.")})
    return optional


def _adv_dict(steps, cfg, sampler_name, scheduler, denoise, decode_mode,
              tile_size, overlap, temporal_size, temporal_overlap):
    """Словарь шины chain_advanced / The chain_advanced bus dict."""
    return {
        "steps": int(steps), "cfg": float(cfg), "sampler_name": str(sampler_name),
        "scheduler": str(scheduler), "denoise": float(denoise),
        "decode_mode": str(decode_mode), "tile_size": int(tile_size),
        "overlap": int(overlap), "temporal_size": int(temporal_size),
        "temporal_overlap": int(temporal_overlap),
    }


def _scene_body(clip, vae, audio_vae, width, height, prompt, total,
                number_of_guides, position_mode, fps, context_frames,
                seed, steps, cfg, sampler_name, scheduler, decode_mode, tile_size,
                overlap, temporal_size, temporal_overlap, model, negative, sigmas,
                denoise, source_piece, source_sr, ref_items, ref_blocks, slots,
                uid, tag):
    """Общее тело сцены: кондиционирование, семплирование, срезы.
    Shared scene body: conditioning, sampling, slices."""
    cond, latent_empty, fc = _gen_ref2v(width, height, total, prompt, clip,
                                        ref_items, ref_blocks)
    audio_t = round(fc / FPS * AUDIO_LATENT_FPS)
    keyframes = _collect_keyframes(cond)
    _source_lip_sync(keyframes, audio_vae, source_piece, source_sr, audio_t)
    _apply_guides(keyframes, width, height, fc, audio_t, vae, audio_vae,
                  number_of_guides, position_mode, fps, slots)
    cond = _set_keyframes(cond, keyframes)
    latent_out, images, gen_audio, decoded = _sample_scene(
        model, negative, sigmas, latent_empty, cond, seed, steps, cfg,
        sampler_name, scheduler, denoise, decode_mode, tile_size, overlap,
        temporal_size, temporal_overlap, vae, audio_vae, uid, tag)
    last_frame_out, context_images = _slice_last_context(images, context_frames, decoded)
    return cond, latent_out, images, gen_audio, last_frame_out, context_images, fc


class AGSoft_H3_Ref2V_Movie_Master:
    """🎬 Голова цепи: глобальные ресурсы, рефы, семплер и сцена 1.
    Chain head: global resources, refs, sampler and scene 1."""
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False, False,
                      False, False, False, False, False, False, False, False)

    DESCRIPTION = (
        "Head of the movie chain: global resources (clip, vae, audio_vae, size, fps, refs) "
        "and scene 1. Refs encode once and travel chain_base; sampling/decode params travel "
        "chain_advanced. Built-in sampler with per-node cache and tiled decode; the cut "
        "source_audio piece guides lip sync at frame 0 and leaves as source_audio_cut.\n"
        "---\n"
        "Голова цепи фильма: глобальные ресурсы (clip, vae, audio_vae, размер, fps, рефы) и "
        "сцена 1. Рефы кодируются один раз и идут по chain_base; параметры семплинга/декода — "
        "по chain_advanced. Встроенный семплер с кэшем по ноде и плиточным декодом; отрезанный "
        "кусок source_audio ведёт липсинк на кадре 0 и выходит как source_audio_cut."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        required.update(_calc_widgets())
        required.update({
            "source_audio_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0,
                                                      "max": 3600.0, "step": 0.001,
                                                      "tooltip": _T(
                "Start offset in seconds inside the source track for scene 1 (ms precision).",
                "Старт в секундах внутри исходной дорожки для сцены 1 (точность мс).")}),
        })
        required.update(_fps_widget())
        required.update(_sampler_widgets())
        required.update(_decode_widgets())
        required.update({
            "ref_image_size": (["match", "max"], {"default": "match", "tooltip": _T(
                "Reference image sizing: match area or 2048px short edge.",
                "Размер реф-изображений: по площади или короткая сторона 2048px.")}),
        })
        required.update(_context_widget())
        required.update(_guide_widgets())
        required.update({
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _T(
                "Scene 1 prompt; refs addressed as <Picture i>, <Video k>, <Audio j>.",
                "Промпт сцены 1; рефы адресуются как <Picture i>, <Video k>, <Audio j>.")}),
        })
        optional = {
            "model": ("MODEL", {"tooltip": _T(
                "First socket. Optional model: linking it enables built-in sampling and "
                "decode; without it the node only builds conditioning and empty latents.",
                "Первый сокет. Опциональная модель: подключение включает встроенные "
                "семплинг и декод; без неё нода только строит conditioning и пустые латенты.")}),
            "clip": ("CLIP", {"tooltip": _T(
                "Second socket. Text encoder for every scene; ALWAYS link it — the node "
                "cannot encode prompts without a CLIP (guarded in main).",
                "Второй сокет. Текст-энкодер для всех сцен; подключайте ВСЕГДА — без CLIP "
                "нода не закодирует промпты (guard в main).")}),
            "negative": ("CONDITIONING", {"tooltip": _T(
                "Optional negative conditioning. Empty conditioning is created if absent.",
                "Опциональный негатив. Без него создаётся пустой конд.")}),
            "sigmas": ("SIGMAS", {"tooltip": _T(
                "Optional custom sigmas. If connected, steps/scheduler are ignored.",
                "Опциональные сигмы. Если подключены, steps/scheduler игнорируются.")}),
            "vae": ("VAE", {"tooltip": _T("VIDEO VAE for the whole chain.",
                                          "ВИДЕО-VAE для всей цепи.")}),
            "audio_vae": ("VAE", {"tooltip": _T("AUDIO VAE for the whole chain.",
                                                "АУДИО-VAE для всей цепи.")}),
            "source_audio": ("AUDIO", {"tooltip": _T(
                "Source track cut per scene from source_audio_start_seconds; its piece "
                "guides scene lip sync.",
                "Исходная дорожка, режется по сценам от source_audio_start_seconds; её "
                "кусок ведёт липсинк сцены.")}),
        }
        for i in range(MAX_REF_IMAGES):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": _T(
                f"Reference image #{i + 1} for the whole movie.",
                f"Референс-изображение #{i + 1} для всего фильма.")})
        for i in range(MAX_REF_VIDEOS):
            optional[f"ref_video_{i}"] = ("IMAGE", {"tooltip": _T(
                f"Reference video #{i + 1} as frame stream (IMAGE batch).",
                f"Референс-видео #{i + 1} потоком кадров (батч IMAGE).")})
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": _T(
                f"Soundtrack paired with ref_video_{i}.",
                f"Дорожка, парная с ref_video_{i}.")})
        for i in range(MAX_REF_AUDIOS):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": _T(
                f"Standalone reference audio #{i + 1} (voice/timbre).",
                f"Отдельный референс-аудио #{i + 1} (голос/тембр).")})
        _guide_inputs(optional)
        return {"required": required, "optional": optional,
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = (BASE_TYPE, ADV_TYPE, "CONDITIONING", "LATENT", "IMAGE", "AUDIO",
                    "AUDIO", "AUDIO", "IMAGE", "IMAGE",
                    "INT", "INT", "INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("chain_base", "chain_advanced", "conditioning", "latent", "images",
                    "audio", "source_audio_cut", "stitched_audio",
                    "last_frame", "context_images",
                    "width", "height", "fps_int", "fps_float",
                    "total_frames", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        _T("Base bus: resources, fps, encoded refs, audio cursor, track pieces.",
           "Базовая шина: ресурсы, fps, закодированные рефы, курсор аудио, куски дорожки."),
        _T("Advanced bus: steps, cfg, sampler, scheduler, denoise, decode params.",
           "Продвинутая шина: steps, cfg, sampler, scheduler, denoise, параметры декода."),
        _T("Scene 1 conditioning (refs + keyframe guides).",
           "Conditioning сцены 1 (рефы + ключевые гайды)."),
        _T("Sampled scene latent (empty AV latent without model).",
           "Семплированный латент сцены (пустой AV-латент без модели)."),
        _T("Decoded scene frames (placeholder without sampling/decode).",
           "Декодированные кадры сцены (плейсхолдер без семплинга/декода)."),
        _T("Generated scene audio decoded from the AV latent.",
           "Сгенерированное аудио сцены, декодированное из AV-латента."),
        _T("RAW exactly cut source_audio piece of this scene (silence without source).",
           "СЫРОЙ точно порезанный кусок source_audio этой сцены (тишина без источника)."),
        _T("Source track assembled up to this node (full track on last Link).",
           "Исходная дорожка до этой ноды (полный трек на последнем Link)."),
        _T("The VERY LAST decoded frame of the scene for the next scene anchor.",
           "САМЫЙ ПОСЛЕДНИЙ декодированный кадр сцены для якоря следующей сцены."),
        _T("Last N decoded frames for references / motion control.",
           "Последние N декодированных кадров для референсов / контроля движения."),
        _T("Frame width in px.", "Ширина кадра в px."),
        _T("Frame height in px.", "Высота кадра в px."),
        _T("Movie FPS as integer (from the fps widget, bus value).",
           "FPS фильма целым (из виджета fps, значение шины)."),
        _T("Movie FPS as float (from the fps widget, bus value).",
           "FPS фильма дробным (из виджета fps, значение шины)."),
        _T("Exact scene frame count (17N+5).", "Точное число кадров сцены (17N+5)."),
        _T("Exact scene duration in seconds (frames / 24).",
           "Точная длительность сцены в секундах (кадры / 24)."),
    )

    def main(self, mode, preset, invert_orientation, custom_width, custom_height,
             megapixels_value, aspect_ratio, frame_count_source, length_seconds,
             frame_count, source_audio_start_seconds, fps, seed, steps, cfg,
             sampler_name, scheduler, denoise, decode_mode, tile_size, overlap,
             temporal_size, temporal_overlap, ref_image_size, context_frames,
             number_of_guides, position_mode, prompt, model=None, clip=None,
             negative=None, sigmas=None, vae=None, audio_vae=None, source_audio=None,
             unique_id=None, **slots):
        if clip is None:
            raise ValueError("clip input is not connected: the Movie Master needs a text "
                             "encoder / вход clip не подключён: Movie Master требует "
                             "текст-энкодер")
        vae, audio_vae = _sort_vaes(vae, audio_vae)
        fps_eff = float(fps)
        width, height = _calc_size(mode, preset, invert_orientation, custom_width,
                                   custom_height, megapixels_value, aspect_ratio)
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        duration = total / 24.0
        start = max(0.0, float(source_audio_start_seconds))
        piece, sr, wave, padded = (None, 44100, None, False)
        if source_audio is not None:
            sr = int(source_audio.get("sample_rate", 44100))
            wave = _norm_waveform(source_audio["waveform"])
            piece, padded = _slice_piece(wave, sr, start, duration)
        if padded:
            logger.warning("%s source shorter than scene 1 window", LOG_PREFIX)
        ref_items, ref_blocks = _build_refs(width, height, vae, audio_vae,
                                            slots, ref_image_size)
        cond, latent_out, images, gen_audio, last_out, ctx, fc = _scene_body(
            clip, vae, audio_vae, width, height, prompt, total, number_of_guides,
            position_mode, fps_eff, context_frames, seed, steps, cfg, sampler_name,
            scheduler, decode_mode, tile_size, overlap, temporal_size,
            temporal_overlap, model, negative, sigmas, denoise, piece, sr,
            ref_items, ref_blocks, slots,
            str(unique_id) if unique_id is not None else None, "[scene1]")
        source_out = _audio_dict(piece, sr) if piece is not None else _source_track(duration, sr)
        base_chain = {
            "clip": clip, "vae": vae, "audio_vae": audio_vae,
            "width": width, "height": height, "fps": fps_eff,
            "ref_items": ref_items, "ref_blocks": ref_blocks,
            "wave": wave, "sr": sr, "audio_cursor": start + duration,
            "scene_count": 1, "pieces": [source_out["waveform"]],
        }
        adv_chain = _adv_dict(steps, cfg, sampler_name, scheduler, denoise, decode_mode,
                              tile_size, overlap, temporal_size, temporal_overlap)
        return (base_chain, adv_chain, cond, latent_out, images, gen_audio,
                source_out, source_out, last_out, ctx,
                width, height, int(round(fps_eff)), fps_eff, fc, fc / 24.0)


class AGSoft_H3_Ref2V_Movie_Link:
    """🎬 Звено цепи: только своя сцена, всё остальное из шин.
    Chain link: only own scene, everything else from the buses."""
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "main"
    WEB_DIRECTORY = "./web"
    OUTPUT_IS_LIST = (False, False, False, False, False, False, False, False,
                      False, False, False, False, False, False, False, False)

    DESCRIPTION = (
        "Next scene: reads resources, fps and encoded refs from chain_base; sampling/decode "
        "params from chain_advanced when linked (widgets collapse), else from own widgets. "
        "Own inputs: prompt, duration, guides, seed. source_audio_cut = raw scene piece; "
        "stitched_audio on the last Link = full movie track.\n"
        "---\n"
        "Следующая сцена: читает ресурсы, fps и закодированные рефы из chain_base; параметры "
        "семплинга/декода из chain_advanced, когда подключена (виджеты схлопнуты), иначе из "
        "своих виджетов. Свои входы: промпт, длительность, гайды, сид. source_audio_cut = "
        "сырой кусок сцены; stitched_audio на последнем Link = полный трек фильма."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "chain_base": (BASE_TYPE, {"tooltip": _T(
                "Base chain bus from Master or previous Link: resources, fps, refs, cursor.",
                "Базовая шина от Master или предыдущего Link: ресурсы, fps, рефы, курсор.")}),
            "frame_count_source": (["Seconds", "Frames"], {"default": "Seconds", "tooltip": _T(
                "Seconds or exact Frames; aligned UP to 17N+5.",
                "Секунды или точные кадры; выравнивание ВВЕРХ до 17N+5.")}),
            "length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": _T(
                "Scene duration in seconds (Seconds only).",
                "Длительность сцены в секундах (только Seconds).")}),
            "frame_count": ("INT", {"default": 124, "min": 5, "max": 99999, "step": 1, "tooltip": _T(
                "Exact scene frame count (Frames only).",
                "Точное число кадров сцены (только Frames).")}),
        }
        required.update(_sampler_widgets())
        required.update(_decode_widgets())
        required.update(_context_widget())
        required.update(_guide_widgets())
        required.update({
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": _T(
                "This scene prompt.", "Промпт этой сцены.")}),
        })
        optional = {
            "chain_advanced": (ADV_TYPE, {"tooltip": _T(
                "Optional advanced bus: link it and sampling/decode widgets collapse to bus "
                "values; unlink to tune this scene manually.",
                "Опциональная продвинутая шина: подключи — виджеты семплинга/декода схлопнутся "
                "до значений шины; отключи — настройка сцены вручную.")}),
            "model": ("MODEL", {"tooltip": _T(
                "Optional model: linking it enables built-in sampling and decode.",
                "Опциональная модель: подключение включает встроенные семплинг и декод.")}),
            "negative": ("CONDITIONING", {"tooltip": _T(
                "Optional negative conditioning. Empty conditioning is created if absent.",
                "Опциональный негатив. Без него создаётся пустой конд.")}),
            "sigmas": ("SIGMAS", {"tooltip": _T(
                "Optional custom sigmas. If connected, steps/scheduler are ignored.",
                "Опциональные сигмы. Если подключены, steps/scheduler игнорируются.")}),
        }
        _guide_inputs(optional)
        return {"required": required, "optional": optional,
                "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = (BASE_TYPE, ADV_TYPE, "CONDITIONING", "LATENT", "IMAGE", "AUDIO",
                    "AUDIO", "AUDIO", "IMAGE", "IMAGE",
                    "INT", "INT", "INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("chain_base", "chain_advanced", "conditioning", "latent", "images",
                    "audio", "source_audio_cut", "stitched_audio",
                    "last_frame", "context_images",
                    "width", "height", "fps_int", "fps_float",
                    "total_frames", "duration_seconds")
    OUTPUT_TOOLTIPS = (
        _T("Base bus for the next Link (fps carried on).",
           "Базовая шина для следующего Link (fps несётся дальше)."),
        _T("Advanced bus for the next Link (effective values of this node).",
           "Продвинутая шина для следующего Link (фактические значения этой ноды)."),
        _T("This scene conditioning (refs + keyframe guides).",
           "Conditioning сцены (рефы + ключевые гайды)."),
        _T("Sampled scene latent (empty AV latent without model).",
           "Семплированный латент сцены (пустой AV-латент без модели)."),
        _T("Decoded scene frames (placeholder without sampling/decode).",
           "Декодированные кадры сцены (плейсхолдер без семплинга/декода)."),
        _T("Generated scene audio decoded from the AV latent.",
           "Сгенерированное аудио сцены, декодированное из AV-латента."),
        _T("RAW exactly cut source_audio piece of this scene (silence without source).",
           "СЫРОЙ точно порезанный кусок source_audio этой сцены (тишина без источника)."),
        _T("Source track assembled up to this node (full track on last Link).",
           "Исходная дорожка до этой ноды (полный трек на последнем Link)."),
        _T("The VERY LAST decoded frame of the scene for the next scene anchor.",
           "САМЫЙ ПОСЛЕДНИЙ декодированный кадр сцены для якоря следующей сцены."),
        _T("Last N decoded frames for references / motion control.",
           "Последние N декодированных кадров для референсов / контроля движения."),
        _T("Frame width in px.", "Ширина кадра в px."),
        _T("Frame height in px.", "Высота кадра в px."),
        _T("Movie FPS as integer (from chain_base).",
           "FPS фильма целым (из chain_base)."),
        _T("Movie FPS as float (from chain_base).",
           "FPS фильма дробным (из chain_base)."),
        _T("Exact scene frame count (17N+5).", "Точное число кадров сцены (17N+5)."),
        _T("Exact scene duration in seconds (frames / 24).",
           "Точная длительность сцены в секундах (кадры / 24)."),
    )

    def main(self, chain_base, frame_count_source, length_seconds, frame_count, seed,
             steps, cfg, sampler_name, scheduler, denoise, decode_mode, tile_size,
             overlap, temporal_size, temporal_overlap, context_frames,
             number_of_guides, position_mode, prompt, model=None, negative=None,
             sigmas=None, chain_advanced=None, unique_id=None, **slots):
        if not isinstance(chain_base, dict) or "clip" not in chain_base:
            raise ValueError("Input chain_base is not a movie bus / вход chain_base не шина")
        adv = chain_advanced if isinstance(chain_advanced, dict) else None
        if adv is not None:
            steps = int(adv.get("steps", steps))
            cfg = float(adv.get("cfg", cfg))
            sampler_name = str(adv.get("sampler_name", sampler_name))
            scheduler = str(adv.get("scheduler", scheduler))
            denoise = float(adv.get("denoise", denoise))
            decode_mode = str(adv.get("decode_mode", decode_mode))
            tile_size = int(adv.get("tile_size", tile_size))
            overlap = int(adv.get("overlap", overlap))
            temporal_size = int(adv.get("temporal_size", temporal_size))
            temporal_overlap = int(adv.get("temporal_overlap", temporal_overlap))
        clip = chain_base["clip"]
        vae, audio_vae = _sort_vaes(chain_base.get("vae"), chain_base.get("audio_vae"))
        width = chain_base["width"]
        height = chain_base["height"]
        fps_eff = float(chain_base.get("fps", FPS))
        scene_index = chain_base["scene_count"] + 1
        cursor = float(chain_base["audio_cursor"])
        total = _calc_frames(frame_count_source, length_seconds, frame_count)
        duration = total / 24.0
        wave, sr = chain_base.get("wave"), int(chain_base.get("sr", 44100))
        piece, padded = None, False
        if wave is not None:
            piece, padded = _slice_piece(wave, sr, cursor, duration)
        if padded:
            logger.warning("%s source shorter than scene %s window", LOG_PREFIX, scene_index)
        cond, latent_out, images, gen_audio, last_out, ctx, fc = _scene_body(
            clip, vae, audio_vae, width, height, prompt, total, number_of_guides,
            position_mode, fps_eff, context_frames, seed, steps, cfg, sampler_name,
            scheduler, decode_mode, tile_size, overlap, temporal_size,
            temporal_overlap, model, negative, sigmas, denoise, piece, sr,
            chain_base.get("ref_items"), chain_base.get("ref_blocks"), slots,
            str(unique_id) if unique_id is not None else None, f"[scene{scene_index}]")
        source_out = _audio_dict(piece, sr) if piece is not None else _source_track(duration, sr)
        pieces = chain_base["pieces"] + [source_out["waveform"]]
        new_base = dict(chain_base)
        new_base.update({
            "vae": vae, "audio_vae": audio_vae,
            "audio_cursor": cursor + duration,
            "scene_count": scene_index,
            "pieces": pieces,
        })
        stitched = {"waveform": torch.cat(pieces, dim=-1), "sample_rate": sr}
        adv_out = _adv_dict(steps, cfg, sampler_name, scheduler, denoise, decode_mode,
                            tile_size, overlap, temporal_size, temporal_overlap)
        return (new_base, adv_out, cond, latent_out, images, gen_audio,
                source_out, stitched, last_out, ctx,
                width, height, int(round(fps_eff)), fps_eff, fc, fc / 24.0)


NODE_CLASS_MAPPINGS = {
    "AGSoft_H3_Ref2V_Movie_Master": AGSoft_H3_Ref2V_Movie_Master,
    "AGSoft_H3_Ref2V_Movie_Link": AGSoft_H3_Ref2V_Movie_Link,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_H3_Ref2V_Movie_Master": "🎬AGSoft H3 Ref2V Movie Master",
    "AGSoft_H3_Ref2V_Movie_Link": "🎬AGSoft H3 Ref2V Movie Link",
}