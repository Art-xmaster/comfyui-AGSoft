# ==============================================================================
# AGSoft_Stitch_Video_Audio.py
# ==============================================================================
# Ноды / Nodes:
# 🎬🔊AGSoft Stitch Video & Audio — склейка готовых IMAGE + AUDIO.
# 🎬🔊AGSoft Latent Stitch Video & Audio — склейка видео/аудио из ЛАТЕНТОВ.
#
# Описание / Description:
# Первая нода мгновенно склеивает уже декодированные IMAGE-отрезки и готовые
# AUDIO-отрезки. Вторая нода принимает разделённые латенты:
#   - video_latent_N — видео-латент сегмента;
#   - audio_latent_N — аудио-латент сегмента.
# Видео декодируется через video_vae, аудио декодируется через audio_vae.
# Если какой-то вход не подключён, пришёл из забайпасенной группы или равен
# None, он просто игнорируется. Ошибки количества нет: inputs_count только
# управляет числом видимых сокетов.
#
# First node instantly stitches already decoded IMAGE segments and ready AUDIO
# segments. Second node accepts separated latents:
#   - video_latent_N — segment video latent;
#   - audio_latent_N — segment audio latent.
# Video is decoded via video_vae, audio is decoded via audio_vae.
# If an input is not connected, comes from a bypassed group or is None, it is
# ignored. There is no count error: inputs_count only controls visible sockets.
#
# Возможности / Features:
# ⚡ Динамическое число входов (inputs_count, 2-50) через JS-сокеты.
#    Dynamic number of inputs (inputs_count, 2-50) via JS sockets.
# ⚡ Неподключённые/забайпассенные входы отсеиваются без ошибок.
#    Unconnected/bypassed inputs are filtered without errors.
# ⚡ Trim дубля кадра-якоря на стыке (цепочка first_frame).
#    Anchor-frame duplicate trim at joins (first_frame chaining).
# ⚡ Пропорциональная подрезка аудио для сохранения A/V синхрона.
#    Proportional audio trim to preserve A/V sync.
# ⚡ Конкат в CPU-памяти — бережет VRAM на слабых GPU.
#    Concat in CPU memory — saves VRAM on weak GPUs.
#
# Автор / Author: AGSoft
# Дата / Date: 16.09.2026
# ==============================================================================


# ------------------------------------------------------------------------------
# Параметры по умолчанию / Default parameters
# ------------------------------------------------------------------------------

DEFAULT_FPS = 24.0
DEFAULT_TRIM = 1
DEFAULT_SAMPLE_RATE = 44100

MIN_SEGMENTS = 2
MAX_SEGMENTS = 50

SEGMENT_COUNTS = [str(i) for i in range(MIN_SEGMENTS, MAX_SEGMENTS + 1)]


import re
import torch
import torch.nn.functional as F


# ==============================================================================
# Helpers
# ==============================================================================

def _as_bool(value):
    """
    Безопасно приводит значение к bool.
    Safely converts value to bool.
    """
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def _clean_value(value):
    """
    Убирает списочные обёртки и отсеивает None.
    Removes list wrappers and filters None values.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for v in value:
            if v is not None:
                return v
        return None

    return value


def _collect_connected_map(kwargs, prefix, validator):
    """
    Собирает только реально подключённые динамические входы:
    images_1 / audio_2 / video_latent_3 / audio_latent_4 и т.д.
    Collects only actually connected dynamic inputs:
    images_1 / audio_2 / video_latent_3 / audio_latent_4 etc.
    """
    out = {}

    for key, value in kwargs.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(prefix):
            continue

        m = re.search(r"(\d+)$", key)
        if not m:
            continue

        value = _clean_value(value)
        if value is None:
            continue

        try:
            if not validator(value):
                continue
            idx = int(m.group(1))
        except Exception:
            continue

        out[idx] = value

    return out


def _is_image_tensor(value):
    return torch.is_tensor(value) and value.dim() in (3, 4, 5) and value.numel() > 0


def _is_audio_value(value):
    if isinstance(value, dict):
        wav = value.get("waveform")
        return torch.is_tensor(wav) and wav.numel() > 0
    return torch.is_tensor(value) and value.dim() > 0 and value.numel() > 0


def _is_latent_value(value):
    """
    LATENT может быть словарем {"samples": ...}, NestedTensor-обёрткой
    или обычным тензором.
    LATENT can be a {"samples": ...} dict, NestedTensor wrapper or plain tensor.
    """
    if torch.is_tensor(value):
        return value.numel() > 0

    if isinstance(value, dict) and "samples" in value:
        samples = value["samples"]

        if hasattr(samples, "tensors"):
            try:
                return len(list(samples.tensors)) > 0
            except Exception:
                return False

        return torch.is_tensor(samples) and samples.numel() > 0

    return False


def _split_latent_samples(value):
    """
    Разбирает LATENT на тип оболочки и список частей.
    Splits LATENT into shell type and part list.
    """
    if value is None:
        return None, []

    if torch.is_tensor(value):
        return None, [value]

    if isinstance(value, dict) and "samples" in value:
        samples = value["samples"]

        if hasattr(samples, "tensors"):
            try:
                return type(samples), list(samples.tensors)
            except Exception:
                return None, [samples]

        return None, [samples]

    return None, []


def _get_video_latent_tensor(value):
    """
    Возвращает видео-часть латента (5D), если она есть.
    Returns video latent part (5D), if present.
    """
    _, parts = _split_latent_samples(value)
    return next((t for t in parts if torch.is_tensor(t) and t.dim() == 5), None)


def _get_audio_latent_tensor(value):
    """
    Возвращает аудио-часть латента (не 5D), если она есть.
    Returns audio latent part (non-5D), if present.
    """
    _, parts = _split_latent_samples(value)
    return next((t for t in parts if torch.is_tensor(t) and t.dim() != 5), None)


def _to_frames(img):
    """
    Приводит IMAGE к 4D [F, H, W, C].
    Normalizes IMAGE to 4D [F, H, W, C].
    """
    if img.dim() == 5:
        B, FR, H, W, C = img.shape
        return img.reshape(B * FR, H, W, C)

    if img.dim() == 3:
        return img.unsqueeze(0)

    return img


def _match_image_shape(px, ref):
    """
    Кроп/паддинг до размера первого кадра, чтобы не падать на разном разрешении.
    Crop/padding to first frame size, so mismatched resolutions do not crash.
    """
    if px.shape[1:] == ref.shape[1:]:
        return px

    out = torch.zeros(
        (px.shape[0], ref.shape[1], ref.shape[2], ref.shape[3]),
        dtype=px.dtype,
        device=px.device,
    )

    h = min(px.shape[1], ref.shape[1])
    w = min(px.shape[2], ref.shape[2])
    c = min(px.shape[3], ref.shape[3])

    out[:, :h, :w, :c] = px[:, :h, :w, :c]
    return out


def _stitch_prepared_chunks(chunks):
    """
    Склеивает подготовленные 4D-чанки. Если чанков нет, отдаёт заглушку.
    Stitches prepared 4D chunks. If no chunks, returns dummy placeholder.
    """
    if not chunks:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    ready = []
    ref = chunks[0].float()

    for px in chunks:
        if px.shape[0] == 0:
            continue
        ready.append(_match_image_shape(px.float(), ref))

    if not ready:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    return torch.cat(ready, dim=0)


def _trim_image_chunk(px, order, trim):
    """
    Подрезает первые кадры у сегментов 2+.
    Trims first frames from segments 2+.
    """
    if px is None or px.shape[0] == 0:
        return px

    if order > 0 and trim > 0:
        if px.shape[0] > trim:
            px = px[trim:]
        else:
            px = px[-1:]

    return px


def _black_frames_for_duration(duration, fps):
    """
    Чёрные кадры нужной длительности для заглушки видеоряда.
    Black frames of required duration for video placeholder.
    """
    if fps <= 0 or duration <= 0:
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)

    n = max(1, int(round(duration * fps)))
    return torch.zeros((n, 8, 8, 3), dtype=torch.float32)


def _norm_audio(value):
    """
    Приводит AUDIO к 3D [1, C, T] и клампит в ±1.
    Normalizes AUDIO to 3D [1, C, T] and clamps to ±1.
    """
    if isinstance(value, dict):
        wav = value.get("waveform")
        sr = int(value.get("sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE)
    else:
        wav = value
        sr = DEFAULT_SAMPLE_RATE

    if wav is None or not torch.is_tensor(wav):
        return None

    wav = wav.detach().cpu().float()

    if wav.dim() == 1:
        wav = wav.view(1, 1, -1)

    elif wav.dim() == 2:
        # Чаще всего [C, T]. Если первый размер похож на батч, делаем [B, 1, T].
        # Usually [C, T]. If first dim looks like batch, make [B, 1, T].
        if wav.shape[0] <= 8:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.unsqueeze(1)

    elif wav.dim() > 3:
        wav = wav.reshape(-1, wav.shape[-2], wav.shape[-1])

    if wav.dim() != 3:
        return None

    if wav.shape[-1] == 0:
        return None

    return wav.clamp(-1.0, 1.0), max(1, sr)


def _resample_audio(wav, src_sr, dst_sr):
    """
    Простой ресемплер для согласования разных sample_rate.
    Simple resampler to match different sample rates.
    """
    src_sr = int(src_sr or dst_sr)
    dst_sr = int(dst_sr or src_sr)

    if src_sr == dst_sr or src_sr <= 0 or dst_sr <= 0:
        return wav

    target_len = max(1, int(round(wav.shape[-1] * dst_sr / float(src_sr))))
    if target_len == wav.shape[-1]:
        return wav

    try:
        return F.interpolate(wav, size=target_len, mode="linear", align_corners=False)
    except Exception:
        idx = torch.linspace(0, max(0, wav.shape[-1] - 1), steps=target_len).long()
        return wav[..., idx]


def _normalize_batch_channels(wav, channels):
    """
    Приводит батч к 1 и число каналов к общему целевому.
    Normalizes batch to 1 and channel count to common target.
    """
    if wav.shape[0] != 1:
        wav = wav.mean(dim=0, keepdim=True)

    channels = max(1, int(channels))
    c = wav.shape[1]

    if c == channels:
        return wav

    if channels == 1:
        return wav.mean(dim=1, keepdim=True)

    if c == 1:
        return wav.repeat(1, channels, 1)

    if c < channels:
        rep = (channels + c - 1) // c
        return wav.repeat(1, rep, 1)[:, :channels, :]

    return wav[:, :channels, :]


def _trim_audio_by_fps(wav, sr, trim_frames, fps):
    """
    Подрезка аудио по FPS, когда неизвестна длина в кадрах.
    Audio trim by FPS when frame count is unknown.
    """
    if trim_frames <= 0 or fps <= 0 or sr <= 0:
        return wav

    drop = int(round(trim_frames * sr / float(fps)))
    if drop <= 0:
        return wav

    if wav.shape[-1] > drop:
        return wav[..., drop:]

    return wav[..., :1]


def _trim_audio_by_ratio(wav, trim_frames, full_frames):
    """
    Пропорциональная подрезка аудио под количество кадров.
    Proportional audio trim according to frame count.
    """
    if trim_frames <= 0 or full_frames <= 0:
        return wav

    total = wav.shape[-1]
    if total <= 1:
        return wav

    drop = int(round(trim_frames * total / float(full_frames)))
    drop = max(0, min(drop, total - 1))

    if drop <= 0:
        return wav

    return wav[..., drop:]


def _trim_audio_auto(wav, sr, trim_frames, full_frames, fps):
    """
    Автоматический выбор подрезки: по кадрам, если они известны, иначе по FPS.
    Automatic trim choice: by frames if known, otherwise by FPS.
    """
    if trim_frames <= 0:
        return wav

    if full_frames > 0:
        return _trim_audio_by_ratio(wav, trim_frames, full_frames)

    return _trim_audio_by_fps(wav, sr, trim_frames, fps)


def _concat_audio_norm(norm):
    """
    Склеивает нормализованные AUDIO-чанки.
    Stitches normalized AUDIO chunks.
    """
    if not norm:
        return None

    target_sr = max(1, int(norm[0][1] or DEFAULT_SAMPLE_RATE))
    target_ch = max(1, int(max(w.shape[1] for w, _ in norm)))

    out = []

    for wav, sr in norm:
        wav = _resample_audio(wav, sr, target_sr)
        wav = _normalize_batch_channels(wav, target_ch)

        if wav.shape[-1] > 0:
            out.append(wav)

    if not out:
        return None

    return torch.cat(out, dim=-1), target_sr


def _silent_audio(duration, sr=DEFAULT_SAMPLE_RATE):
    """
    Тишина нужной длительности.
    Silence of required duration.
    """
    sr = max(1, int(sr or DEFAULT_SAMPLE_RATE))
    n = max(1, int(round(max(0.0, float(duration)) * sr)))
    return torch.zeros((1, 1, n), dtype=torch.float32), sr


def _normalize_audio_output(out, fallback_sr):
    """
    Приводит результат декодера аудио к ComfyUI AUDIO-совместимому виду.
    Normalizes audio decoder output to ComfyUI AUDIO-compatible form.
    """
    if out is None:
        return None

    if isinstance(out, dict) and "waveform" in out:
        return out

    if torch.is_tensor(out):
        return {"waveform": out, "sample_rate": fallback_sr}

    if isinstance(out, (list, tuple)) and len(out) > 0:
        first = out[0]

        if isinstance(first, dict) and "waveform" in first:
            return first

        if torch.is_tensor(first):
            sr = fallback_sr
            if len(out) > 1:
                try:
                    sr = int(out[1])
                except Exception:
                    sr = fallback_sr

            return {"waveform": first, "sample_rate": sr}

    return None


def _decode_video_frames(video_vae, video_tensor):
    """
    Пробует декодировать видео-латент через video_vae.
    Tries to decode video latent via video_vae.
    """
    if video_vae is None or video_tensor is None:
        return None

    candidates = (
        video_tensor,
        {"samples": video_tensor},
    )

    for cand in candidates:
        try:
            px = video_vae.decode(cand)
            return _to_frames(px).detach().cpu().float()
        except Exception:
            pass

    return None


def _decode_audio_latent(audio_vae, audio_latent, fallback_sr):
    """
    Пробует декодировать аудио-латент через audio_vae.
    Tries to decode audio latent via audio_vae.
    """
    if audio_vae is None or audio_latent is None:
        return None

    if torch.is_tensor(audio_latent):
        tensor = audio_latent
    else:
        tensor = _get_audio_latent_tensor(audio_latent)

    if tensor is None:
        return None

    candidates = (
        {"samples": tensor},
        tensor,
    )

    for fn_name in ("decode_audio", "decode"):
        fn = getattr(audio_vae, fn_name, None)
        if not callable(fn):
            continue

        for cand in candidates:
            try:
                out = fn(cand)
            except Exception:
                continue

            norm = _normalize_audio_output(out, fallback_sr)
            if norm is not None:
                return norm

    return None


# ==============================================================================
# 🎬🔊AGSoft Stitch Video & Audio
# ==============================================================================

class AGSoftStitchVideoAudio:
    """
    🎬🔊Склейка готовых IMAGE + AUDIO. Без декода.
    Instant stitching of ready IMAGE + AUDIO. No decode.
    """

    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬🔊AGSoft Stitch Video & Audio.\n"
        "Stitches ready IMAGE and AUDIO segments. Dynamic sockets are images_N and audio_N. "
        "inputs_count only controls visible sockets; unconnected/bypassed inputs are ignored. "
        "If an IMAGE segment exists but its AUDIO segment is missing, silence is inserted. "
        "If an AUDIO segment exists but its IMAGE segment is missing, black frames are inserted. "
        "This keeps segment order stable.\n"
        "---\n"
        "🎬🔊AGSoft Stitch Video & Audio.\n"
        "Склеивает готовые IMAGE и AUDIO отрезки. Динамические сокеты: images_N и audio_N. "
        "inputs_count только управляет видимыми сокетами; неподключённые/забайпассенные входы "
        "игнорируются. Если есть IMAGE-отрезок, но нет его AUDIO-отрезка, вставляется тишина. "
        "Если есть AUDIO-отрезок, но нет его IMAGE-отрезка, вставляются чёрные кадры. "
        "Так порядок сегментов остаётся стабильным."
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
                            "Number of visible segment sockets. Unconnected sockets are ignored.\n"
                            "---\n"
                            "Количество видимых сокетов. Неподключённые сокеты игнорируются."
                        ),
                    },
                ),
                "trim_first_frames": (
                    "INT",
                    {
                        "default": DEFAULT_TRIM,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Trim first frames from segments 2+ (anchor-frame duplicate).\n"
                            "---\n"
                            "Подрезать первые кадры у сегментов 2+ (дубль кадра-якоря)."
                        ),
                    },
                ),
            },
            "optional": {
                "fps_float": (
                    "FLOAT",
                    {
                        "default": DEFAULT_FPS,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": (
                            "FPS for silence generation and fallback audio trim.\n"
                            "---\n"
                            "FPS для генерации тишины и резервной подрезки аудио."
                        ),
                    },
                ),
                "trim_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable only if incoming AUDIO is NOT already trimmed.\n"
                            "---\n"
                            "Включайте только если входящее аудио ещё НЕ подрезано."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("stitched_images", "stitched_audio")
    FUNCTION = "stitch"
    CATEGORY = "AGSoft/MiniMaxH3"

    def stitch(
        self,
        inputs_count="2",
        trim_first_frames=DEFAULT_TRIM,
        fps_float=DEFAULT_FPS,
        trim_audio=False,
        **kwargs,
    ):
        trim = int(trim_first_frames or 0)
        fps = float(fps_float or DEFAULT_FPS)
        trim_audio = _as_bool(trim_audio)

        images_map = _collect_connected_map(kwargs, "images_", _is_image_tensor)
        audio_map = _collect_connected_map(kwargs, "audio_", _is_audio_value)

        indexes = sorted(set(images_map.keys()) | set(audio_map.keys()))

        image_chunks = []
        audio_norm = []

        for order, idx in enumerate(indexes):
            img_val = images_map.get(idx)
            aud_val = audio_map.get(idx)

            if img_val is None and aud_val is None:
                continue

            image_chunk = None
            full_frames = 0

            if img_val is not None:
                try:
                    px = _to_frames(img_val).detach().cpu().float()
                except Exception:
                    px = None

                if px is not None and px.shape[0] > 0:
                    full_frames = int(px.shape[0])
                    px = _trim_image_chunk(px, order, trim)

                    if px.shape[0] > 0:
                        image_chunk = px

            audio_chunk = None

            if aud_val is not None:
                res = _norm_audio(aud_val)

                if res is not None:
                    wav, sr = res

                    if order > 0 and trim_audio and trim > 0 and full_frames > 0:
                        wav = _trim_audio_auto(wav, sr, trim, full_frames, fps)

                    if wav.shape[-1] > 0:
                        audio_chunk = (wav, sr)

            # Если нет видео, но есть аудио — чёрные кадры.
            # If no video but audio exists — black frames.
            if image_chunk is None and audio_chunk is not None:
                wav, sr = audio_chunk
                duration = wav.shape[-1] / float(max(1, sr))
                image_chunk = _black_frames_for_duration(duration, fps)

            # Если нет аудио, но есть видео — тишина.
            # If no audio but video exists — silence.
            if audio_chunk is None and image_chunk is not None:
                duration = image_chunk.shape[0] / fps if fps > 0 else 0.0
                audio_chunk = _silent_audio(duration, DEFAULT_SAMPLE_RATE)

            if image_chunk is not None:
                image_chunks.append(image_chunk)

            if audio_chunk is not None:
                audio_norm.append(audio_chunk)

        out_images = _stitch_prepared_chunks(image_chunks)
        audio_res = _concat_audio_norm(audio_norm)

        if audio_res is None:
            duration = out_images.shape[0] / fps if fps > 0 and out_images.shape[0] > 0 else 0.0
            wav, sr = _silent_audio(duration, DEFAULT_SAMPLE_RATE)
        else:
            wav, sr = audio_res

        return (out_images, {"waveform": wav, "sample_rate": sr})


# ==============================================================================
# 🎬🔊AGSoft Latent Stitch Video & Audio
# ==============================================================================

class AGSoftLatentStitchVideoAudio:
    """
    🎬🔊Склейка видео и аудио из разделённых ЛАТЕНТОВ.
    Video/audio stitching from separated LATENTS.

    video_latent_N — видео-латент сегмента / segment video latent.
    audio_latent_N — аудио-латент сегмента / segment audio latent.
    """

    WEB_DIRECTORY = "./web"

    DESCRIPTION = (
        "🎬🔊AGSoft Latent Stitch Video & Audio.\n"
        "Stitches separated segment latents. Dynamic sockets are video_latent_N and audio_latent_N. "
        "video_latent_N is the segment video latent; audio_latent_N is the segment audio latent. "
        "inputs_count only controls visible sockets; unconnected/bypassed inputs are ignored. "
        "Video latents are decoded via video_vae. Audio latents are decoded via audio_vae. "
        "If audio_vae is not connected, stitched_audio becomes silent placeholder. "
        "If audio_latent_N is missing but video_latent_N contains an internal audio part, "
        "that internal part is used as fallback.\n"
        "---\n"
        "🎬🔊AGSoft Latent Stitch Video & Audio.\n"
        "Склеивает разделённые латенты сегментов. Динамические сокеты: video_latent_N и audio_latent_N. "
        "video_latent_N — видео-латент сегмента; audio_latent_N — аудио-латент сегмента. "
        "inputs_count только управляет видимыми сокетами; неподключённые/забайпассенные входы "
        "игнорируются. Видео-латенты декодируются через video_vae. Аудио-латенты декодируются "
        "через audio_vae. Если audio_vae не подключен, stitched_audio становится тихой заглушкой. "
        "Если audio_latent_N не подключён, но внутри video_latent_N есть аудио-часть, "
        "эта внутренняя часть используется как запасной вариант."
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
                            "Number of visible segment sockets. Unconnected sockets are ignored.\n"
                            "---\n"
                            "Количество видимых сокетов. Неподключённые сокеты игнорируются."
                        ),
                    },
                ),
                "trim_first_frames": (
                    "INT",
                    {
                        "default": DEFAULT_TRIM,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Trim first decoded frames from segments 2+. Audio is trimmed proportionally.\n"
                            "---\n"
                            "Подрезать первые декодированные кадры у сегментов 2+. Аудио режется пропорционально."
                        ),
                    },
                ),
            },
            "optional": {
                "video_vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Video VAE used to decode video_latent_N into IMAGE.\n"
                            "---\n"
                            "Video VAE для декодирования video_latent_N в IMAGE."
                        ),
                    },
                ),
                "audio_vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "Audio VAE used to decode audio_latent_N into AUDIO.\n"
                            "---\n"
                            "Audio VAE для декодирования audio_latent_N в AUDIO."
                        ),
                    },
                ),
                "fps_float": (
                    "FLOAT",
                    {
                        "default": DEFAULT_FPS,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.001,
                        "tooltip": (
                            "FPS for timing, silence generation and fallback audio trim.\n"
                            "---\n"
                            "FPS для тайминга, генерации тишины и резервной подрезки аудио."
                        ),
                    },
                ),
                "trim_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Trim decoded audio from segments 2+ proportionally to trimmed frames.\n"
                            "---\n"
                            "Подрезать декодированное аудио у сегментов 2+ пропорционально подрезанным кадрам."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("stitched_images", "stitched_audio")
    FUNCTION = "stitch"
    CATEGORY = "AGSoft/MiniMaxH3"

    def stitch(
        self,
        inputs_count="2",
        trim_first_frames=DEFAULT_TRIM,
        video_vae=None,
        audio_vae=None,
        fps_float=DEFAULT_FPS,
        trim_audio=True,
        **kwargs,
    ):
        trim = int(trim_first_frames or 0)
        fps = float(fps_float or DEFAULT_FPS)
        trim_audio = _as_bool(trim_audio)

        video_map = _collect_connected_map(kwargs, "video_latent_", _is_latent_value)
        audio_map = _collect_connected_map(kwargs, "audio_latent_", _is_latent_value)

        indexes = sorted(set(video_map.keys()) | set(audio_map.keys()))

        image_chunks = []
        audio_norm = []

        for order, idx in enumerate(indexes):
            vid_val = video_map.get(idx)
            aud_val = audio_map.get(idx)

            if vid_val is None and aud_val is None:
                continue

            video_tensor = _get_video_latent_tensor(vid_val) if vid_val is not None else None

            # Основной источник аудио — audio_latent_N.
            # Если он не подключён, пробуем взять аудио-часть из video_latent_N.
            # Primary audio source is audio_latent_N.
            # If not connected, try audio part from video_latent_N as fallback.
            audio_tensor = _get_audio_latent_tensor(aud_val) if aud_val is not None else None
            if audio_tensor is None and vid_val is not None:
                audio_tensor = _get_audio_latent_tensor(vid_val)

            image_chunk = None
            full_frames = 0
            est_frames = 0

            if video_tensor is not None:
                if video_tensor.dim() == 5 and video_tensor.shape[2] > 0:
                    full_frames = int(video_tensor.shape[2])

                px = _decode_video_frames(video_vae, video_tensor)

                if px is not None and px.shape[0] > 0:
                    full_frames = int(px.shape[0])
                    px = _trim_image_chunk(px, order, trim)

                    if px.shape[0] > 0:
                        image_chunk = px

                est_frames = full_frames
                if order > 0 and trim > 0:
                    est_frames = max(1, est_frames - trim)

            audio_chunk = None

            if audio_tensor is not None and audio_vae is not None:
                audio_obj = _decode_audio_latent(audio_vae, audio_tensor, DEFAULT_SAMPLE_RATE)

                if audio_obj is not None:
                    res = _norm_audio(audio_obj)

                    if res is not None:
                        wav, sr = res

                        if order > 0 and trim_audio and trim > 0 and full_frames > 0:
                            wav = _trim_audio_auto(wav, sr, trim, full_frames, fps)

                        if wav.shape[-1] > 0:
                            audio_chunk = (wav, sr)

            # Длительность видео-сегмента: по декоду или по оценке латента.
            # Video segment duration: from decode or latent estimation.
            video_duration = 0.0

            if image_chunk is not None:
                video_duration = image_chunk.shape[0] / fps if fps > 0 else 0.0
            elif est_frames > 0 and fps > 0:
                video_duration = est_frames / fps

            # Если аудио нет, но видео есть — тишина.
            # If no audio but video exists — silence.
            if audio_chunk is None and video_duration > 0:
                audio_chunk = _silent_audio(video_duration, DEFAULT_SAMPLE_RATE)

            # Если видео нет, но аудио есть — чёрные кадры.
            # If no video but audio exists — black frames.
            if image_chunk is None and audio_chunk is not None:
                if video_duration > 0:
                    image_chunk = _black_frames_for_duration(video_duration, fps)
                else:
                    wav, sr = audio_chunk
                    duration = wav.shape[-1] / float(max(1, sr))
                    image_chunk = _black_frames_for_duration(duration, fps)

            if image_chunk is not None:
                image_chunks.append(image_chunk)

            if audio_chunk is not None:
                audio_norm.append(audio_chunk)

        out_images = _stitch_prepared_chunks(image_chunks)
        audio_res = _concat_audio_norm(audio_norm)

        if audio_res is None:
            duration = out_images.shape[0] / fps if fps > 0 and out_images.shape[0] > 0 else 0.0
            wav, sr = _silent_audio(duration, DEFAULT_SAMPLE_RATE)
        else:
            wav, sr = audio_res

        return (out_images, {"waveform": wav, "sample_rate": sr})


# ==============================================================================
# Маппинг для ComfyUI / ComfyUI mappings
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "AGSoftStitchVideoAudio": AGSoftStitchVideoAudio,
    "AGSoftLatentStitchVideoAudio": AGSoftLatentStitchVideoAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftStitchVideoAudio": "🎬🔊AGSoft Stitch Video & Audio",
    "AGSoftLatentStitchVideoAudio": "🎬🔊AGSoft Latent Stitch Video & Audio",
}