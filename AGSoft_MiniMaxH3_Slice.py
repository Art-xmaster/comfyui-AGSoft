"""
==============================================================================
AGSoft_MiniMaxH3_Slice.py
==============================================================================
Ноды / Nodes:
🎞️🧊AGSoft MiniMaxH3 Latent Slice — вырезает контекст из ЛАТЕНТА (декод внутри).
🎞️🖼️AGSoft MiniMaxH3 Images Slice — вырезает контекст из готовых IMAGE (без декода).

Описание / Description:
Пара нод подготовки контекста для продолжения сюжета MiniMax H3.
Обе отдают last_frame (самый последний кадр, всегда 4D IMAGE) для входа
first_frame ноды MiniMax H3 Image to Video следующей сцены — так модель
продолжает сюжет с той же внешностью, позой и сценой — и context_images
(последние N кадров) для референсов и контроля движения.
Latent Slice принимает готовый латент сцены и сам декодирует его через video
VAE (NestedTensor поддерживается: автоматически берётся видео-часть).
Images Slice принимает УЖЕ декодированные IMAGE — повторного декода нет,
нода мгновенная и подходит слабым машинам, где декод всё равно уже сделан
для превью/сохранения.

A pair of nodes that prepare continuation context for MiniMax H3.
Both output last_frame (the very last frame, always 4D IMAGE) for the
first_frame input of the next scene's MiniMax H3 Image to Video node — this
is how the model continues the story with the same character, pose and scene
— and context_images (last N frames) for references and motion control.
Latent Slice takes a completed scene latent and decodes it itself via the
video VAE (NestedTensor supported: the video part is picked automatically).
Images Slice takes ALREADY decoded IMAGE — no repeated decode, so the node is
instant and suits weak machines where the decode was already done for
preview/saving.

Возможности / Features:
⚡ last_frame всегда 4D [B, H, W, C] — совместим с энкодером Qwen3VL
(иначе "too many values to unpack").
last_frame is always 4D [B, H, W, C] — compatible with the Qwen3VL encoder
(otherwise "too many values to unpack").
⚡ context_frames (1-124) регулирует размер контекста: больше кадров = лучше
консистентность движения, но больше токенов и VRAM.
context_frames (1-124) controls context size: more frames = better motion
consistency, but more tokens and VRAM.
⚡ Кэш по содержимому входа: тот же латент не декодируется дважды, те же кадры
возвращаются теми же тензорами (стабильная память, без лишних аллокаций).
Content-keyed cache: the same latent is never decoded twice, identical frames
come back as the same tensors (stable memory, no extra allocations).
⚡ Поддержка NestedTensor (видео+аудио) и IMAGE 4D/5D.
NestedTensor (video+audio) and 4D/5D IMAGE support.
⚡ JS не требуется — входы статические.
No JS required — static inputs.
Автор / Author: AGSoft
Дата / Date: 25.11.2026
==============================================================================
"""

import hashlib
import logging
import weakref
from collections import OrderedDict

import torch

# print("[AGSoft MiniMaxH3 Slice] v11.25 loaded (context slicing with content-hash cache)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Параметры по умолчанию / Default parameters
# ------------------------------------------------------------------------------
DEFAULT_CONTEXT_FRAMES = 5
MIN_CONTEXT_FRAMES = 1
MAX_CONTEXT_FRAMES = 124

# Кэш нарезки: ключ — содержимое входа / Slice cache keyed by input content
CACHE_MAX_ENTRIES = 8
CACHE_MAX_BYTES = 1024 * 1024 * 1024
CACHE_CHUNK_BYTES = 8 * 1024 * 1024
_SLICE_CACHE = OrderedDict()


def _to_frames(img: torch.Tensor) -> torch.Tensor:
    """Приводит IMAGE к 4D [F, H, W, C] / Normalize IMAGE to 4D [F, H, W, C]."""
    if img.dim() == 5:
        B, F, H, W, C = img.shape
        return img.reshape(B * F, H, W, C)
    return img


def _slice_context(pixels: torch.Tensor, context_frames: int, tag: str) -> tuple:
    """
    Отдает (last_frame, context_images) из 4D пикселей.
    Returns (last_frame, context_images) from 4D pixels.
    """
    total = pixels.shape[0]
    ctx = min(context_frames, total)
    if ctx < context_frames:
        print(
            f"[{tag}] В отрезке только {total} кадров, контекст уменьшен до {ctx}.\n"
            f"Segment has only {total} frames, context reduced to {ctx}."
        )
    start = total - ctx  # срез по индексу: -0 вернул бы весь ролик / index slice: -0 means all
    # clone отвязывает результат от буфера всей сцены / clone detaches from the scene buffer
    last_frame = pixels[-1:].clone()         # [1, H, W, C] — последний кадр / the very last frame
    context_images = pixels[start:].clone()  # [N, H, W, C] — последние N кадров / last N frames
    return last_frame, context_images


def _tensor_hash(t: torch.Tensor) -> str:
    """
    Отпечаток содержимого: форма + dtype + device + байты, чанками без пиков RAM.
    Content fingerprint: shape + dtype + device + bytes, chunked without RAM spikes.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{tuple(t.shape)}|{t.dtype}|{t.device}".encode("utf-8"))
    src = t.detach().contiguous()
    try:
        flat = src.view(torch.uint8).reshape(-1)
    except RuntimeError as e:
        # Экзотический dtype приводим к float32 / Cast exotic dtype to float32 for hashing
        logger.debug("[AGSoft MiniMaxH3 Slice] byte view fallback: %s", e)
        flat = src.to(torch.float32).contiguous().view(torch.uint8).reshape(-1)
    for i in range(0, flat.numel(), CACHE_CHUNK_BYTES):
        chunk = flat[i:i + CACHE_CHUNK_BYTES].to("cpu", copy=False)
        h.update(chunk.numpy().tobytes())
    return h.hexdigest()


def _cache_get(key: tuple):
    """LRU-поиск готового результата / LRU lookup of a ready result."""
    hit = _SLICE_CACHE.get(key)
    if hit is not None:
        _SLICE_CACHE.move_to_end(key)
    return hit


def _entry_bytes(entry: tuple) -> int:
    """Память, которую держит запись кэша / Memory held by a cache entry."""
    return sum(t.numel() * t.element_size() for t in entry[:2])


def _cache_put(key: tuple, entry: tuple) -> None:
    """Запись с вытеснением по числу и объёму / Store with eviction by count and bytes."""
    _SLICE_CACHE[key] = entry
    _SLICE_CACHE.move_to_end(key)
    total = sum(_entry_bytes(e) for e in _SLICE_CACHE.values())
    while len(_SLICE_CACHE) > 1 and (
        len(_SLICE_CACHE) > CACHE_MAX_ENTRIES or total > CACHE_MAX_BYTES
    ):
        _, old = _SLICE_CACHE.popitem(last=False)
        total -= _entry_bytes(old)


class AGSoftMiniMaxH3LatentSlice:
    """
    🎞️🧊 Вырезает контекст продолжения из готового ЛАТЕНТА (декод VAE внутри).
    Cuts continuation context out of a completed LATENT (VAE decode inside).
    """

    DESCRIPTION = (
        "🎞️🧊AGSoft MiniMaxH3 Latent Slice.\n"
        "Cuts continuation context out of a completed MiniMax H3 scene latent. The node decodes "
        "the latent itself via the connected video VAE (NestedTensor supported: the 5D video part "
        "is picked automatically, the audio part is ignored) and returns last_frame — the very "
        "last frame as a 4D IMAGE for the first_frame input of the next scene's MiniMax H3 Image "
        "to Video node — plus context_images, the last N frames, for references / motion control. "
        "This is how the next scene continues the story: same character, pose, scene and camera. "
        "Use this node when the workflow has NO decoded frames yet; if your workflow already "
        "decodes the scene (preview, Video Save, VSR), prefer 🎞️🖼️Images Slice to avoid paying for "
        "a second decode on weak machines.\n"
        "---\n"
        "🎞️🧊AGSoft MiniMaxH3 Latent Slice.\n"
        "Вырезает контекст продолжения из готового латента сцены MiniMax H3. Нода сама декодирует "
        "латент через подключенный video VAE (NestedTensor поддерживается: видео-часть 5D берётся "
        "автоматически, аудио-часть игнорируется) и возвращает last_frame — самый последний кадр "
        "в формате 4D IMAGE для входа first_frame ноды MiniMax H3 Image to Video следующей сцены — "
        "плюс context_images, последние N кадров, для референсов и контроля движения. Так следующая "
        "сцена продолжает сюжет: те же персонаж, поза, сцена и камера. Используйте эту ноду, когда в "
        "ворке ЕЩЁ НЕТ декодированных кадров; если ворк уже декодирует сцену (превью, Video Save, "
        "VSR), берите 🎞️🖼️Images Slice, чтобы не платить за повторный декод на слабых машинах."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_input": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Completed latent of the previous scene (sampler output, NestedTensor "
                            "video+audio or plain tensor). The video part is decoded to cut context.\n"
                            "---\n"
                            "Готовый латент предыдущей сцены (выход сэмплера, NestedTensor "
                            "видео+аудио или обычный тензор). Видео-часть декодируется для нарезки контекста."
                        )
                    }
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 video VAE (minimax_h3_video_vae_fp16) used to decode the "
                            "latent into frames.\n"
                            "---\n"
                            "Video VAE MiniMax H3 (minimax_h3_video_vae_fp16) для декодирования "
                            "латента в кадры."
                        )
                    }
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": DEFAULT_CONTEXT_FRAMES,
                        "min": MIN_CONTEXT_FRAMES,
                        "max": MAX_CONTEXT_FRAMES,
                        "step": 1,
                        "tooltip": (
                            "How many last frames go into context_images (1-124). More frames = "
                            "better motion consistency, but more Qwen3VL tokens and VRAM. 1-3 for "
                            "weak GPUs, 5-8 for hard character consistency.\n"
                            "---\n"
                            "Сколько последних кадров отдать в context_images (1-124). Больше кадров = "
                            "лучше консистентность движения, но больше токенов Qwen3VL и VRAM. "
                            "1-3 для слабых GPU, 5-8 для жёсткой консистентности персонажа."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("last_frame", "context_images")
    OUTPUT_TOOLTIPS = (
        "The very last frame as 4D IMAGE — into first_frame of the next scene.\n"
        "---\n"
        "Самый последний кадр в 4D IMAGE — на вход first_frame следующей сцены.",
        "Last N frames as 4D IMAGE — references and motion control.\n"
        "---\n"
        "Последние N кадров в 4D IMAGE — референсы и контроль движения.",
    )
    FUNCTION = "slice_latent"
    CATEGORY = "AGSoft/MiniMaxH3"

    def slice_latent(self, latent_input: dict, vae, context_frames: int) -> tuple:
        samples = latent_input["samples"]
        # NestedTensor: берем видео-часть (5D) / NestedTensor: take the video part (5D)
        if hasattr(samples, "tensors"):
            vid = next((t for t in samples.tensors if t.dim() == 5), samples.tensors[0])
        else:
            vid = samples
        # Тот же латент = тот же результат без декода / Same latent = same result, no decode
        key = ("latent", context_frames, _tensor_hash(vid))
        hit = _cache_get(key)
        if hit is not None and hit[2]() is vae:
            return (hit[0], hit[1])
        # Декодируем в пиксели и приводим к 4D / Decode to pixels and normalize to 4D
        pixels = _to_frames(vae.decode(vid))
        last_frame, context_images = _slice_context(
            pixels, context_frames, "AGSoft MiniMaxH3 Latent Slice"
        )
        _cache_put(key, (last_frame, context_images, weakref.ref(vae)))
        return (last_frame, context_images)


class AGSoftMiniMaxH3ImagesSlice:
    """
    🎞️🖼️ Вырезает контекст продолжения из УЖЕ декодированных IMAGE. Без VAE.
    Cuts continuation context out of ALREADY decoded IMAGE. No VAE.
    """

    DESCRIPTION = (
        "🎞️🖼️AGSoft MiniMaxH3 Images Slice.\n"
        "Cuts continuation context out of ALREADY decoded frames (IMAGE) of the previous scene — "
        "no VAE involved, pure tensor slicing, so it is instant and memory-cheap even on weak GPUs. "
        "Returns last_frame — the very last frame as a 4D IMAGE for the first_frame input of the "
        "next scene's MiniMax H3 Image to Video node — plus context_images, the last N frames, for "
        "references / motion control. Use this node whenever the workflow already decodes the scene "
        "(preview, AGSoft Video Save, RTX VSR etc.): since the decode was paid anyway, slicing "
        "pixels costs nothing. Accepts 4D [F, H, W, C] and 5D [B, F, H, W, C] IMAGE tensors.\n"
        "---\n"
        "🎞️🖼️AGSoft MiniMaxH3 Images Slice.\n"
        "Вырезает контекст продолжения из УЖЕ декодированных кадров (IMAGE) предыдущей сцены — без "
        "VAE, чистая нарезка тензора, поэтому мгновенно и дёшево по памяти даже на слабых GPU. "
        "Возвращает last_frame — самый последний кадр в формате 4D IMAGE для входа first_frame "
        "ноды MiniMax H3 Image to Video следующей сцены — плюс context_images, последние N кадров, "
        "для референсов и контроля движения. Используйте эту ноду, когда ворк уже декодирует сцену "
        "(превью, AGSoft Video Save, RTX VSR и т.п.): раз декод всё равно уже оплачен, нарезка "
        "пикселей ничего не стоит. Принимает IMAGE 4D [F, H, W, C] и 5D [B, F, H, W, C]."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Already decoded frames of the previous scene (VAE Decode / Video Save "
                            "path output). 4D or 5D IMAGE tensors accepted.\n"
                            "---\n"
                            "Уже декодированные кадры предыдущей сцены (выход цепочки VAE Decode / "
                            "Video Save). Принимаются IMAGE 4D и 5D."
                        )
                    }
                ),
                "context_frames": (
                    "INT",
                    {
                        "default": DEFAULT_CONTEXT_FRAMES,
                        "min": MIN_CONTEXT_FRAMES,
                        "max": MAX_CONTEXT_FRAMES,
                        "step": 1,
                        "tooltip": (
                            "How many last frames go into context_images (1-124). More frames = "
                            "better motion consistency, but more Qwen3VL tokens and VRAM downstream.\n"
                            "---\n"
                            "Сколько последних кадров отдать в context_images (1-124). Больше кадров = "
                            "лучше консистентность движения, но больше токенов Qwen3VL и VRAM дальше по цепочке."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("last_frame", "context_images")
    OUTPUT_TOOLTIPS = (
        "The very last frame as 4D IMAGE — into first_frame of the next scene.\n"
        "---\n"
        "Самый последний кадр в 4D IMAGE — на вход first_frame следующей сцены.",
        "Last N frames as 4D IMAGE — references and motion control.\n"
        "---\n"
        "Последние N кадров в 4D IMAGE — референсы и контроль движения.",
    )
    FUNCTION = "slice_images"
    CATEGORY = "AGSoft/MiniMaxH3"

    def slice_images(self, images: torch.Tensor, context_frames: int) -> tuple:
        pixels = _to_frames(images)
        # Хэшируем только хвост — он и уходит на выход / Hash only the tail we output
        total = pixels.shape[0]
        ctx = min(context_frames, total)
        key = ("images", context_frames, _tensor_hash(pixels[total - ctx:]))
        hit = _cache_get(key)
        if hit is not None:
            return (hit[0], hit[1])
        last_frame, context_images = _slice_context(
            pixels, context_frames, "AGSoft MiniMaxH3 Images Slice"
        )
        _cache_put(key, (last_frame, context_images, None))
        return (last_frame, context_images)


# Маппинг для ComfyUI / ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "AGSoftMiniMaxH3LatentSlice": AGSoftMiniMaxH3LatentSlice,
    "AGSoftMiniMaxH3ImagesSlice": AGSoftMiniMaxH3ImagesSlice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMiniMaxH3LatentSlice": "🎞️🧊AGSoft MiniMaxH3 Latent Slice",
    "AGSoftMiniMaxH3ImagesSlice": "🎞️🖼️AGSoft MiniMaxH3 Images Slice",
}