# ==============================================================================
# AGSoft_MiniMaxH3_Stitch.py
# ==============================================================================
# Ноды / Nodes:
# 🎬🖼️AGSoft MiniMaxH3 Stitch Images — мгновенная склейка готовых IMAGE.
# 🎬🧊AGSoft MiniMaxH3 Stitch Latent — склейка из ЛАТЕНТОВ (видео+аудио).
#
# Описание / Description:
# Пара нод финальной сборки многосценного ролика MiniMax H3.
# Stitch Images склеивает уже декодированные IMAGE-отрезки (без VAE, мгновенно)
# и опционально собирает аудио из латентов тех же отрезков с пропорциональной
# подрезкой под trim — синхрон сохраняется.
# Stitch Latent склеивает латенты (NestedTensor: видео+аудио) в общий LATENT;
# при подключенном video VAE дополнительно декодирует сегменты ПОСЕГМЕНТНО
# (экономия VRAM) в общий IMAGE; без VAE работает мгновенно и отдаёт заглушку.
# A pair of nodes for the final assembly of a multi-scene MiniMax H3 movie.
# Stitch Images joins already decoded IMAGE segments (no VAE, instant) and
# optionally stitches audio from the same segments' latents with proportional
# trim — sync is preserved.
# Stitch Latent concatenates latents (NestedTensor: video+audio) into a single
# LATENT; with a connected video VAE it additionally decodes segments ONE BY ONE
# (VRAM friendly) into a single IMAGE; without a VAE it works instantly and
# returns a dummy placeholder.
#
# Возможности / Features:
# ⚡ Динамическое число входов (inputs_count, 2-50) через JS-сокеты.
#   Stitch Latent: latent_N (LATENT). Stitch Images: images_N (IMAGE) + latent_N.
# Dynamic number of inputs (inputs_count, 2-50) via JS sockets.
#   Stitch Latent: latent_N (LATENT). Stitch Images: images_N (IMAGE) + latent_N.
# ⚡ Trim дубля кадра-якоря на стыке (цепочка first_frame); аудио режется
#   пропорционально — A/V синхрон не разъезжается.
# Anchor-frame duplicate trim at joins (first_frame chaining); audio is trimmed
# proportionally — A/V sync stays intact.
# ⚡ Поддержка NestedTensor MiniMax H3 (видео+аудио) и обычных тензоров.
# MiniMax H3 NestedTensor (video+audio) and plain tensor support.
# ⚡ Конкат в CPU-памяти — бережет VRAM на слабых GPU.
# Concat in CPU memory — saves VRAM on weak GPUs.
# ⚡ ВАЖНО: видео из склеенного латента НЕ декодировать (мерцание/дёргание из-за
#   темпорального контекста VAE и фазы чанков). stitched_latent — переносчик
#   АУДИО для VAE Decode Audio. Видео собираем только в пиксельном пространстве.
# IMPORTANT: do NOT decode video from a stitched latent (flicker/jitter caused
# by temporal VAE context and chunk phase). stitched_latent is an AUDIO carrier
# for VAE Decode Audio. Assemble video in pixel space only.
#
# Автор / Author: AGSoft
# Дата / Date: 07.09.2026
# ==============================================================================

# ------------------------------------------------------------------------------
# Параметры по умолчанию / Default parameters
# ------------------------------------------------------------------------------
DEFAULT_TRIM = 1
MIN_SEGMENTS = 2
MAX_SEGMENTS = 50
SEGMENT_COUNTS = [str(i) for i in range(MIN_SEGMENTS, MAX_SEGMENTS + 1)]

import copy
import re
import torch


def _to_frames(img):
    """Приводит IMAGE к 4D [F, H, W, C] / Normalize IMAGE to 4D [F, H, W, C]."""
    if img.dim() == 5:
        B, F, H, W, C = img.shape
        return img.reshape(B * F, H, W, C)
    return img


def _split_parts(lat):
    """
    Разбирает LATENT (обычный или NestedTensor) на тип оболочки и части.
    Splits LATENT (plain or NestedTensor) into shell type and parts.
    """
    s = lat["samples"]
    if hasattr(s, "tensors"):
        return type(s), list(s.tensors)
    return None, [s]


def _rebuild_nested(nested_type, fallback_samples, new_parts):
    """
    Собирает NestedTensor обратно; fallback — копирование оболочки.
    Rebuilds NestedTensor; fallback — shell copy.
    """
    if nested_type is None:
        return new_parts[0] if len(new_parts) == 1 else new_parts
    try:
        return nested_type(new_parts)
    except Exception:
        out = copy.copy(fallback_samples)
        out.tensors = new_parts
        return out


def _collect(kwargs, prefix):
    """
    Собирает динамические входы (latent_N / images_N) из kwargs по индексу.
    Collects dynamic inputs (latent_N / images_N) from kwargs by index.
    """
    items = {}
    for key, value in kwargs.items():
        if not key.startswith(prefix):
            continue
        match = re.search(r"\d+", key)
        if not match:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            items[int(match.group())] = value
    return [items[i] for i in sorted(items.keys())]


def _check_count(items, expected, label):
    """
    Проверка количества подключенных входов / Check connected inputs count.
    """
    if len(items) < MIN_SEGMENTS:
        raise ValueError(
            f"[AGSoft Stitch] Нужно подключить как минимум {MIN_SEGMENTS} отрезка ({label})!\n"
            f"At least {MIN_SEGMENTS} segments ({label}) must be connected!"
        )
    if len(items) < expected:
        raise ValueError(
            f"[AGSoft Stitch] Подключено {len(items)} из {expected} выбранных входов ({label}).\n"
            f"Connected {len(items)} of {expected} selected inputs ({label})."
        )


class AGSoftMiniMaxH3StitchLatent:
    """
    🎬🧊Склейка из латентов: общий LATENT (переносчик аудио) + IMAGE через
    посегментный декод VAE (опционально).
    Latent-based stitching: combined LATENT (audio carrier) + IMAGE via
    per-segment VAE decode (optional).
    """

    # JS для динамических входов latent_N / JS for dynamic latent_N inputs.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬🧊AGSoft MiniMaxH3 Stitch Latent.\n"
        "Stitches 2-50 completed MiniMax H3 latents (NestedTensor: video + audio parts) into one "
        "production. Video parts are concatenated over the frame axis, audio parts over the time "
        "axis. If a video VAE is connected, segments are decoded one by one (VRAM friendly) and "
        "pixels are concatenated into a single IMAGE with anchor-frame duplicate trimming at joins. "
        "Without a VAE the node works instantly (latent-only concat) and outputs a dummy IMAGE "
        "placeholder. IMPORTANT: use stitched_latent only as an AUDIO carrier (VAE Decode Audio); "
        "do NOT decode video from a concatenated latent — the temporal VAE context and chunk phase "
        "change, causing flicker/jitter. Number of inputs is set dynamically via inputs_count "
        "(JS adds latent_N sockets).\n"
        "---\n"
        "🎬🧊AGSoft MiniMaxH3 Stitch Latent.\n"
        "Склеивает 2-50 готовых латентов MiniMax H3 (NestedTensor: видео+аудио) в один ролик. "
        "Видео-части конкатенируются по оси кадров, аудио-части — по оси времени. Если подключен "
        "video VAE, сегменты декодируются посегментно (экономия VRAM), а пиксели склеиваются в один "
        "IMAGE с подрезкой дубля кадра-якоря на стыках. Без VAE нода работает мгновенно (только "
        "склейка латентов) и отдаёт заглушку вместо IMAGE. ВАЖНО: stitched_latent используйте только "
        "как переносчик АУДИО (VAE Decode Audio); НЕ декодируйте видео из склеенного латента — "
        "темпоральный контекст VAE и фаза чанков меняются, появляется мерцание/дёргание. Количество "
        "входов задаётся динамически через inputs_count (JS добавляет сокеты latent_N)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputs_count": (
                    SEGMENT_COUNTS,
                    {
                        "default": "2",
                        "tooltip": (
                            "Number of segments to stitch (2-50). JS adds latent_N sockets; "
                            "connect scene latents in order 1..N — order defines stitch order.\n"
                            "---\n"
                            "Количество отрезков для склейки (2-50). JS добавляет сокеты latent_N; "
                            "подключайте латенты сцен по порядку 1..N — порядок определяет порядок склейки."
                        )
                    }
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 video VAE (minimax_h3_video_vae_fp16). Connected = "
                            "per-segment decode into a single IMAGE. Not connected = latent-only "
                            "concat (instant), IMAGE is a dummy placeholder.\n"
                            "---\n"
                            "Video VAE MiniMax H3 (minimax_h3_video_vae_fp16). Подключен = "
                            "посегментный декод в один IMAGE. Не подключен = только склейка латентов "
                            "(мгновенно), IMAGE — заглушка."
                        )
                    }
                ),
                "trim_first_frames": (
                    "INT",
                    {
                        "default": DEFAULT_TRIM, "min": 0, "max": 8, "step": 1,
                        "tooltip": (
                            "How many first PIXEL frames to drop from segments 2+ (the anchor-frame "
                            "duplicate produced by first_frame chaining). Applies to the IMAGE output "
                            "only. 0 = keep all frames.\n"
                            "---\n"
                            "Сколько первых ПИКСЕЛЬНЫХ кадров отрезать у отрезков 2+ (дубль кадра-якоря "
                            "из цепочки first_frame). Действует только на выход IMAGE. 0 = не резать."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("stitched_latent", "images")
    FUNCTION = "stitch_latent"
    CATEGORY = "AGSoft/MiniMaxH3"

    def stitch_latent(self, inputs_count="2", vae=None,
                      trim_first_frames=DEFAULT_TRIM, **kwargs):
        latents = _collect(kwargs, "latent_")
        _check_count(latents, int(inputs_count or 2), "latent")

        video_parts, audio_parts = [], []
        nested_type = None

        for lat in latents:
            nt, parts = _split_parts(lat)
            if nt is not None:
                nested_type = nt
            v = next((t for t in parts if t.dim() == 5), None)
            a = next((t for t in parts if t.dim() != 5), None)
            if v is None:
                raise ValueError("В латенте нет видео-части (5D).\nNo video part (5D) in latent.")
            video_parts.append(v)
            if a is not None:
                audio_parts.append(a)

        # Проверка одинакового разрешения / Same resolution check
        base = video_parts[0].shape
        for v in video_parts[1:]:
            if v.shape[1:] != base[1:]:
                raise ValueError(
                    f"Разное разрешение отрезков: {tuple(base)} vs {tuple(v.shape)}.\n"
                    f"Segment resolution mismatch."
                )

        # Склеиваем латенты: видео по кадрам (dim 2), аудио по времени (dim -1)
        # Concat latents: video over frames (dim 2), audio over time (dim -1)
        new_parts = [torch.cat(video_parts, dim=2)]
        if audio_parts:
            new_parts.append(torch.cat(audio_parts, dim=-1))
        stitched = _rebuild_nested(nested_type, latents[0]["samples"], new_parts)

        # Декод ПОСЕГМЕНТНО только если vae подключен / Per-segment decode only if vae connected
        if vae is not None:
            img_chunks = []
            for i, v in enumerate(video_parts):
                px = _to_frames(vae.decode(v)).cpu()
                if i > 0 and trim_first_frames > 0:
                    px = px[trim_first_frames:]
                img_chunks.append(px)
            images = torch.cat(img_chunks, dim=0)
        else:
            # Заглушка, как в AGSoft Save Image Plus / Dummy, like in AGSoft Save Image Plus
            images = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

        return ({"samples": stitched}, images)


class AGSoftMiniMaxH3StitchImages:
    """
    🎬️🖼️Мгновенная склейка готовых IMAGE + склейка аудио из латентов.
    Без декода VAE.
    Instant stitching of ready IMAGE tensors + audio stitching from latents.
    No VAE decode.
    """

    # JS для динамических входов images_N + latent_N / JS for dynamic images_N + latent_N inputs.
    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬🖼️AGSoft MiniMaxH3 Stitch Images.\n"
        "Instantly stitches 2-50 already decoded IMAGE segments into one movie (pure tensor concat "
        "in CPU memory, no VAE involved — milliseconds even on weak GPUs). The anchor-frame "
        "duplicate at each join (produced by first_frame chaining) is trimmed via "
        "trim_first_frames. Optionally accepts the same segments' latents (latent_N, paired with "
        "images_N by index) and stitches their audio parts with proportional trimming, so A/V sync "
        "stays intact after the pixel trim. Outputs: images -> AGSoft Video Save.images; "
        "stitched_latent -> VAE Decode Audio -> AGSoft Video Save.audio. If no latents are "
        "connected, stitched_latent returns a silent dummy placeholder. Number of inputs is set "
        "dynamically via inputs_count (JS adds images_N and latent_N sockets).\n"
        "---\n"
        "🎬🖼️AGSoft MiniMaxH3 Stitch Images.\n"
        "Мгновенно склеивает 2-50 уже декодированных IMAGE-отрезков в один ролик (чистый конкат "
        "тензоров в CPU-памяти, без VAE — миллисекунды даже на слабых GPU). Дубль кадра-якоря на "
        "каждом стыке (из цепочки first_frame) подрезается через trim_first_frames. Опционально "
        "принимает латенты тех же отрезков (latent_N, пара к images_N по индексу) и склеивает их "
        "аудио-части с пропорциональной подрезкой — A/V синхрон сохраняется после пиксельного trim. "
        "Выходы: images -> AGSoft Video Save.images; stitched_latent -> VAE Decode Audio -> "
        "AGSoft Video Save.audio. Если латенты не подключены, stitched_latent отдаёт тихую заглушку. "
        "Количество входов задаётся динамически через inputs_count (JS добавляет сокеты images_N и latent_N)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputs_count": (
                    SEGMENT_COUNTS,
                    {
                        "default": "2",
                        "tooltip": (
                            "Number of segments to stitch (2-50). JS adds images_N (and latent_N) "
                            "sockets; connect decoded frames in order 1..N — order defines stitch order.\n"
                            "---\n"
                            "Количество отрезков для склейки (2-50). JS добавляет сокеты images_N "
                            "(и latent_N); подключайте декодированные кадры по порядку 1..N — порядок "
                            "определяет порядок склейки."
                        )
                    }
                ),
                "trim_first_frames": (
                    "INT",
                    {
                        "default": DEFAULT_TRIM, "min": 0, "max": 8, "step": 1,
                        "tooltip": (
                            "How many first frames to drop from segments 2+ (the anchor-frame "
                            "duplicate produced by first_frame chaining). Audio latents are trimmed "
                            "proportionally — A/V sync is preserved. 0 = keep all frames.\n"
                            "---\n"
                            "Сколько первых кадров отрезать у отрезков 2+ (дубль кадра-якоря из "
                            "цепочки first_frame). Аудио-латенты режутся пропорционально — A/V "
                            "синхрон сохраняется. 0 = не резать."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("images", "stitched_latent")
    FUNCTION = "stitch_images"
    CATEGORY = "AGSoft/MiniMaxH3"

    def stitch_images(self, inputs_count="2", trim_first_frames=DEFAULT_TRIM, **kwargs):
        images_in = _collect(kwargs, "images_")
        latents_in = _collect(kwargs, "latent_")
        _check_count(images_in, int(inputs_count or 2), "images")

        # Пары (images, latent) по позициям / (images, latent) pairs by position
        positions = [
            (_to_frames(img).cpu(),
             latents_in[i] if i < len(latents_in) else None)
            for i, img in enumerate(images_in)
        ]

        # Видео: конкат с trim / Video: concat with trim
        video_chunks = [p[0] for p in positions]
        out = [video_chunks[0]] + [c[trim_first_frames:] for c in video_chunks[1:]]
        images = torch.cat(out, dim=0)

        # Аудио из латентов, пропорциональный trim / Audio from latents, proportional trim
        video_parts, audio_parts = [], []
        nested_type = None
        first_samples = None

        for i, (chunk, lat) in enumerate(positions):
            if lat is None:
                continue
            nt, parts = _split_parts(lat)
            if nt is not None:
                nested_type = nt
            if first_samples is None:
                first_samples = lat["samples"]
            v = next((t for t in parts if t.dim() == 5), None)
            a = next((t for t in parts if t.dim() != 5), None)
            if v is not None:
                video_parts.append(v)
            if a is not None and i > 0 and trim_first_frames > 0:
                # Сколько аудио-шагов стоит один пиксельный кадр этого отрезка
                # How many audio steps one pixel frame of this segment costs
                f_pix = chunk.shape[0]
                t_aud = a.shape[-1]
                trim_a = min(t_aud, round(trim_first_frames * t_aud / f_pix))
                a = a[..., trim_a:]
            if a is not None:
                audio_parts.append(a)

        if video_parts or audio_parts:
            new_parts = []
            if video_parts:
                new_parts.append(torch.cat(video_parts, dim=2))
            if audio_parts:
                new_parts.append(torch.cat(audio_parts, dim=-1))
            stitched = _rebuild_nested(nested_type, first_samples, new_parts)
        else:
            # Латенты не подключены — заглушка / No latents connected — dummy
            stitched = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)

        return (images, {"samples": stitched})


# Маппинг для ComfyUI / ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "AGSoftMiniMaxH3StitchLatent": AGSoftMiniMaxH3StitchLatent,
    "AGSoftMiniMaxH3StitchImages": AGSoftMiniMaxH3StitchImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMiniMaxH3StitchLatent": "🎬🧊AGSoft MiniMaxH3 Stitch Latent",
    "AGSoftMiniMaxH3StitchImages": "🎬🖼️AGSoft MiniMaxH3 Stitch Images",
}