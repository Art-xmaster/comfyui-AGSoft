"""
AGSoft_Minimax_H3_Latent_Upscaler.py
Ноды / Nodes:
🎬AGSoft Minimax H3 Latent Upscaler — обученный пространственный апскейл
латентов MiniMax H3 (24 канала, 4D/5D) напрямую в latent-пространстве;
опционально пропускает conditioning и ретаргетит латенты ключевых кадров
minimax_keyframes под апскейленное разрешение.
🎬AGSoft Minimax H3 Latent Upscaler Plus — входной латент проходит обученный
апскейлер, затем refine-семплирование (логика 🧩KSampler Plus) и встроенный
декод VAE в одной ноде; ключевые кадры кондиционирования ретаргетятся
автоматически перед refine-семплированием.
Описание / Description:
Обученный пространственный апскейл латентов MiniMax H3 (24 канала, 4D/5D)
напрямую в latent-пространстве, без дорогого VAE decode/encode.
Вход умный: обычный видеолатент или совместный AV-латент (NestedTensor
[video, audio]) — совместная форма разделяется автоматически, апскейлится
только видеочасть, аудиочасть проходит сквозь ноду без изменений.
Гайды 📍Add Guide живут в conditioning ключом minimax_keyframes и несут
латенты изображений, закодированные под разрешение исходного латента. После
апскейла разрешение генерации другое, и модель MiniMax H3 падает с shape
mismatch при вшивании рядов ключевых кадров. Обе ноды файла приводят латенты
ключевых кадров к целевому разрешению той же обученной сетью (ретаргет):
Plus — внутри себя перед refine-семплированием, база — на сквозном проходе
conditioning через опциональные вход и выход positive.
Версия Plus повторяет цепочку «апскейлер между двумя семплерами»: входной
латент апскейлится обученной сетью, затем уточняется семплером за несколько
шагов (refine) и декодируется встроенным VAE; кэш по ноде хранит фазу
апскейла и фазу семплирования отдельно, поэтому смена виджетов декода
только передекодирует, а смена виджетов семплера не перезапускает апскейл.
---
Learned spatial upscaling of MiniMax H3 (24-channel, 4D/5D) latents directly
in latent space, bypassing costly VAE decode/encode.
Smart input: plain video latent or native joint AV latent (NestedTensor
[video, audio]) — the joint form is split automatically, only the video
member is upscaled while the audio member passes through untouched.
Add Guide guides live in the conditioning under minimax_keyframes and carry
image latents encoded at the source latent resolution. After the upscale the
generation resolution differs and the MiniMax H3 model crashes with a shape
mismatch while splicing keyframe rows. Both nodes of this file re-target the
keyframe latents to the target resolution with the same learned network:
Plus does it internally before refine sampling, the base node does it on a
conditioning passthrough via the optional positive input and output.
The Plus version repeats the "upscaler between two samplers" chain: the input
latent is upscaled by the learned network, then refined by the sampler in a
few steps and decoded by the built-in VAE; the per-node cache keeps the
upscale phase and the sampling phase separate, so changing decode widgets
re-decodes only, and changing sampler widgets does not re-run the upscale.
Возможности / Features:
⚡ Умный вход: plain video latent или joint AV latent с авторазделением
   Smart input: plain video latent or joint AV latent with auto-split
⚡ База: выходы video_latent и собранный обратно video_audio_latent
   Base outputs: video_latent and re-joined video_audio_latent
⚡ Ретаргет латентов minimax_keyframes к апскейленному разрешению в обеих
   нодах; у базы — опциональный сквозной проход conditioning (positive)
   Re-target of minimax_keyframes latents to the upscaled resolution in both
   nodes; the base node gets an optional conditioning passthrough (positive)
⚡ Plus: конвейер апскейл -> refine-семплирование -> декод VAE и аудио
   Plus: pipeline upscale -> refine sampling -> VAE and audio decode
⚡ Три режима размера: множитель, мегапиксели, целевые размеры
   Three sizing modes: scale by multiplier, megapixels, target dimensions
⚡ Двухосевое выравнивание по пиксельной сетке с сохранением пропорций
   Dual-axis pixel-grid alignment with optional aspect-ratio lock
⚡ fp32/fp16/bf16, кэш моделей по (checkpoint, device, precision), offload на CPU
   fp32/fp16/bf16, model cache per (checkpoint, device, precision), CPU offload
⚡ Автоопределение архитектуры чекпоинта (каналы, блоки, temporal-свёртки)
   Auto-detects checkpoint architecture (channels, blocks, temporal convs)
⚡ Кэш фаз по ноде: декод-виджеты не пересемплируют, семплер-виджеты не
   перезапускают апскейл / Per-node phase cache: decode widgets do not
   re-sample, sampler widgets do not re-run the upscale
Автор / Author: AGSoft
Дата / Date: 04.11.2026
"""
import os
import re
import math
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.model_management
import folder_paths
from nodes import common_ksampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Minimax H3 Latent Upscaler] v11.04 loaded (keyframe retarget in both nodes)")

# Регистрация папки моделей / Model folder registration
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER),
    )

# Пространственное сжатие VAE H3 (16x): 1280x704 px -> 80x44 latent
# Spatial compression of the H3 VAE (16x): 1280x704 px -> 80x44 latent
VAE_DOWNSAMPLE = 16
H3_VIDEO_CHANNELS = 24
_PRECISION_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
LOG_PREFIX = "[AGSoft Minimax H3 Latent Upscaler]"
# Плейсхолдер пустых картинок: чётный размер, чтобы видео-кодеры не падали.
# Empty-image placeholder: even size so video encoders do not fail.
PLACEHOLDER_SIZE = 64
DEFAULT_AUDIO_SAMPLE_RATE = 32000
# Кэш фазы апскейла по unique_id: смена виджетов семплера/декода не
# перезапускает обученную сеть, апскейленный латент берётся из кэша.
# Upscale-phase cache per unique_id: sampler/decode widget changes do not
# re-run the learned network, the upscaled latent is taken from the cache.
_UPSCALE_CACHE = {}
# Кэш фазы семплирования по unique_id: смена виджетов декода не
# пересемплирует, а только передекодирует.
# Sampling-phase cache per unique_id: a re-run with changed decode widgets
# re-decodes only, without re-sampling.
_SAMPLE_CACHE = {}
_CACHE_LIMIT = 8

_T = lambda en, ru: en + "\n---\n" + ru


def _alignment_grid(align):
    # Сетка пикселей, совместимая с align и с VAE H3 / Pixel grid compatible with align and the H3 VAE
    return math.lcm(max(1, int(align)), VAE_DOWNSAMPLE)


def _aligned_pixel_size(w_target, h_target, w_in, h_in, align, keep_proportion):
    # Финальные пиксельные размеры на двухосевой сетке, безопасной для VAE H3
    # Final pixel dimensions on a dual-axis H3-safe alignment grid
    grid = _alignment_grid(align)
    w_target = max(float(grid), float(w_target))
    h_target = max(float(grid), float(h_target))
    if not keep_proportion:
        # Независимое округление осей / Independent rounding of both axes
        return (max(grid, int(round(w_target / grid)) * grid),
                max(grid, int(round(h_target / grid)) * grid))
    aspect = float(w_in) / float(h_in)
    ideal_h = w_target / aspect
    w_center = max(1, int(round(w_target / grid)))
    h_center = max(1, int(round(ideal_h / grid)))
    best = None
    for w_units in range(max(1, w_center - 2), w_center + 3):
        for h_units in range(max(1, h_center - 2), h_center + 3):
            width = w_units * grid
            height = h_units * grid
            ratio_error = abs(math.log((width / height) / aspect))
            size_error = abs(width / w_target - 1.0) + abs(height / ideal_h - 1.0)
            score = (ratio_error * 4.0 + size_error, ratio_error, size_error,
                     abs(width - w_target) + abs(height - ideal_h))
            if best is None or score < best[0]:
                best = (score, width, height)
    return best[1], best[2]


# Статистики нормализации латентов H3 (24 канала, из тренировочного кода)
# H3 latent normalization stats (24 channels, from training code)
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.046409729681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# ==========================================
# Совместные AV-латенты (NestedTensor) / Joint AV latents (NestedTensor)
# ==========================================
def _is_nested(value):
    # Comfy NestedTensor или совместимый объект с is_nested/unbind
    # Comfy NestedTensor or a compatible object with is_nested/unbind
    try:
        import comfy.nested_tensor
        if isinstance(value, comfy.nested_tensor.NestedTensor):
            return True
    except Exception:
        pass
    if not hasattr(value, "is_nested") or not hasattr(value, "unbind"):
        return False
    return bool(value.is_nested)


def _nested_members(value):
    members = list(value.unbind())
    for m in members:
        if not torch.is_tensor(m):
            raise TypeError("Joint AV latent members must be torch.Tensor / "
                            "члены совместного AV-латента должны быть torch.Tensor")
    return members


def _wrap_nested(video, audio, template=None):
    # Сборка совместного AV-латента [video, audio] / Rebuild joint AV latent [video, audio]
    try:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor([video, audio])
    except Exception:
        if template is not None:
            try:
                return type(template)([video, audio])
            except Exception:
                pass
    raise ValueError("This ComfyUI build lacks NestedTensor support for joint AV latents / "
                     "эта сборка ComfyUI не поддерживает NestedTensor для совместных AV-латентов")


def _split_input_latent(latent):
    # Умный вход: joint AV -> (video, audio, template), plain -> (video, None, None)
    # Smart input: joint AV -> (video, audio, template), plain -> (video, None, None)
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Input latent must be a LATENT dictionary with 'samples' / "
                         "входной latent должен быть словарём LATENT с ключом 'samples'")
    samples = latent["samples"]
    if _is_nested(samples):
        members = _nested_members(samples)
        if len(members) != 2:
            raise ValueError(f"Joint AV latent must contain exactly [video, audio] members, got {len(members)} / "
                             f"совместный AV-латент должен содержать ровно [video, audio], получено {len(members)}")
        return members[0], members[1], samples
    if torch.is_tensor(samples):
        return samples, None, None
    raise TypeError(f"Unsupported latent samples type: {type(samples).__name__} / "
                    f"неподдерживаемый тип samples: {type(samples).__name__}")


def _validate_video(video):
    if not torch.is_tensor(video) or not video.is_floating_point():
        raise TypeError("MiniMax H3 video latent must be a floating-point torch.Tensor / "
                        "видеолатент MiniMax H3 должен быть float-тензором")
    if video.ndim not in (4, 5) or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"MiniMax H3 video latent must be Bx24xHxW or Bx24xTxHxW, got {tuple(video.shape)} / "
                         f"видеолатент MiniMax H3 должен быть Bx24xHxW или Bx24xTxHxW, получено {tuple(video.shape)}")
    return video


def _output_linked(prompt, unique_id, slot):
    # Подключён ли выход slot куда-то в графе / Whether output slot is linked anywhere in the graph
    if not isinstance(prompt, dict) or unique_id is None:
        return False
    uid = str(unique_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for val in inputs.values():
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == uid:
                try:
                    if int(val[1]) == slot:
                        return True
                except (TypeError, ValueError):
                    continue
    return False


# ==========================================
# Компоненты 3D-сети (как в тренировочном коде) / 3D network components (as in training code)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlockEmb3D(nn.Module):
    # Res-блок с scale-embedding (AdaGN) / Residual block with scale embedding (AdaGN)
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        # Проекция embedding в 2*out_channels ДО chunk: scale и shift по out_channels
        # Project embedding to 2*out_channels BEFORE chunk: scale and shift of out_channels each
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    # Depthwise-свёртка по времени (kernel 5) / Depthwise temporal convolution (kernel 5)
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return x + h


class LatentResizer3D(nn.Module):
    # Чистый 3D-бэкбон: encoder -> trilinear -> decoder, время не трогает
    # Pure 3D backbone: encoder -> trilinear -> decoder, time axis preserved
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=1.0, target_size=None):
        if target_size is None or tuple(target_size) == tuple(x.shape[-3:]):
            return x
        emb = self.embed(torch.tensor([scale - 1.0], dtype=x.dtype, device=x.device).unsqueeze(0))
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = F.interpolate(x, size=tuple(target_size), mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb.expand(x.shape[0], -1)) if isinstance(b, ResBlockEmb3D) else b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


# ==========================================
# Загрузка моделей / Model loading
# ==========================================
MODEL_CACHE = {}


def _detect_arch(sd):
    # Автоопределение конфигурации по ключам state_dict / Auto-detect config from state_dict keys
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512, "dropout": 0.1,
           "temporal_every": 2, "temporal_kernel": 5}
    if "conv_in.weight" in sd:
        cfg["in_channels"] = sd["conv_in.weight"].shape[1]
        cfg["channels"] = sd["conv_in.weight"].shape[0]
    in_ids, out_ids, t_in, t_out = set(), set(), set(), set()
    for k in sd.keys():
        m = re.match(r"in_blocks\.(\d+)\.in_layers\.", k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.in_layers\.", k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r"in_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            t_in.add(int(m.group(1)))
        m = re.match(r"out_blocks\.(\d+)\.dwconv\.weight", k)
        if m:
            t_out.add(int(m.group(1)))
    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)
    if t_in or t_out:
        for k in sd.keys():
            if k.endswith("dwconv.weight"):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0
    return cfg


def _load_sd(path, dtype):
    # Чтение чекпоинта через core-утилиту (без pickle-рисков) + снятие префикса
    # Checkpoint read via core utility (no pickle risk) + prefix stripping
    sd = comfy.utils.load_torch_file(path)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    out = {}
    for k, v in sd.items():
        out[k] = v.to(dtype=dtype) if (torch.is_tensor(v) and v.is_floating_point() and v.dtype != dtype) else v
    return out


def load_model(name, device, precision):
    key = f"{name}::{device}::{precision}"
    if key in MODEL_CACHE:
        # Кэш мог быть offload-нут на CPU; .to() в том же устройстве - no-op
        # Cached net may have been offloaded to CPU; .to() on same device is a no-op
        return MODEL_CACHE[key].to(device)
    path = folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {name}")
    dtype = _PRECISION_DTYPES.get(precision, torch.float32)
    sd = _load_sd(path, dtype)
    cfg = _detect_arch(sd)
    # Meta-конструкция избегает второй FP32-копии модели в памяти
    # Meta construction avoids a second full FP32 copy in memory
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"], dropout=cfg["dropout"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(sd, strict=True, assign=True)
    model.train(False).requires_grad_(False)
    model.to(device)
    del sd
    MODEL_CACHE[key] = model
    logger.info("[AGSoft MinimaxH3] loaded %s | params=%s | temporal=%s | precision=%s | device=%s",
                name, f"{sum(p.numel() for p in model.parameters()):,}",
                "on" if cfg["temporal_every"] > 0 else "off", precision, device)
    return model


def upscale_latent_exact(video, model_name, target_h, target_w, device, precision,
                         offload_after_upscale=False, scale_embedding=None):
    # Обученный пространственный ресайз до точных latent H/W при сохранении B/C/T
    # Learned spatial resize to exact latent H/W while preserving B/C/T
    if not isinstance(video, torch.Tensor) or not video.is_floating_point():
        raise TypeError("Input latent must be a floating-point torch.Tensor / "
                        "входной latent должен быть float-тензором")
    if video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"MiniMax H3 latent must be Bx24xTxHxW, got {tuple(video.shape)} / "
                         f"латент MiniMax H3 должен быть Bx24xTxHxW, получено {tuple(video.shape)}")
    src_h, src_w = int(video.shape[-2]), int(video.shape[-1])
    if target_h < src_h or target_w < src_w:
        raise ValueError("Only upscaling is supported: target must not shrink any spatial axis / "
                         "поддерживается только апскейл: целевой размер не может быть меньше исходного")
    if target_h == src_h and target_w == src_w:
        return video.clone()
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = _PRECISION_DTYPES.get(precision, torch.float32)
    temporal = int(video.shape[2])
    eff = float(scale_embedding) if scale_embedding else ((target_h / src_h) + (target_w / src_w)) / 2.0
    work = video.to(device=dev, dtype=dtype, copy=True)
    model = load_model(model_name, dev, precision)
    mean, std = _make_norm_tensors(dev, dtype)
    with torch.inference_mode():
        work.sub_(mean).div_(std)
        out = model(work, scale=eff, target_size=(temporal, target_h, target_w))
        out.mul_(std).add_(mean)
    del work
    expected = (int(video.shape[0]), H3_VIDEO_CHANNELS, temporal, target_h, target_w)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"Upscaler returned shape {tuple(out.shape)}, expected {expected} / "
                           f"апскейлер вернул форму {tuple(out.shape)}, ожидалась {expected}")
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Upscaler returned NaN or Inf values / апскейлер вернул NaN или Inf")
    out = out.to(device="cpu", dtype=video.dtype)
    if dev.type == "cuda":
        if offload_after_upscale:
            model.to("cpu")
        torch.cuda.empty_cache()
    return out


def _upscale_video(src, model_name, mode, scale, width, height, megapixels, align,
                   keep_proportion, device, precision, offload_after_upscale):
    # Расчёт целевого размера и инференс только видеочасти / Target sizing and inference on the video member only
    was_4d = src.dim() == 4
    s = src.unsqueeze(2) if was_4d else src  # 4D -> 5D с T=1 / 4D -> 5D with T=1
    _b, _c, _t, h_in, w_in = s.shape
    ds = VAE_DOWNSAMPLE
    # 1. Теоретический целевой размер в пикселях / Theoretical pixel-space target
    if mode == "scale by multiplier":
        w_tgt, h_tgt = w_in * ds * scale, h_in * ds * scale
        eff = float(scale)
    elif mode == "target dimensions":
        w_tgt, h_tgt = float(width), float(height)
        eff = (w_tgt / (w_in * ds) + h_tgt / (h_in * ds)) / 2.0
    elif mode == "megapixels":
        target_px = megapixels * 1024 * 1024
        ar = w_in / h_in
        h_tgt = (target_px / ar) ** 0.5
        w_tgt = h_tgt * ar
        eff = (w_tgt / (w_in * ds) + h_tgt / (h_in * ds)) / 2.0
    else:
        raise ValueError(f"Unsupported mode: {mode} / неподдерживаемый режим: {mode}")
    # 2. Привязка к общей сетке выравнивания / Snap to the common alignment grid
    w_fin, h_fin = _aligned_pixel_size(w_tgt, h_tgt, w_in, h_in, align, bool(keep_proportion))
    # 3. Возврат в latent-пространство / Back to latent space
    w_out = max(1, int(w_fin // ds))
    h_out = max(1, int(h_fin // ds))
    if w_out < w_in or h_out < h_in:
        raise ValueError("This model only supports upscaling: target is smaller than source / "
                         "модель поддерживает только апскейл: цель меньше источника")
    if w_out == w_in and h_out == h_in:
        logger.info("[AGSoft MinimaxH3] target equals source, video latent returned unchanged / "
                    "цель равна источнику, видеолатент возвращён без изменений")
        out = s.clone()
    else:
        logger.info("[AGSoft MinimaxH3] latent %dx%d -> %dx%d | pixels %dx%d | scale=%.3f",
                    w_in, h_in, w_out, h_out, w_out * ds, h_out * ds, eff)
        # 4. Инференс целиком по времени: temporal-свёртки не допускают чанкования
        # 4. Full-sequence inference: temporal convs make chunking non-equivalent
        out = upscale_latent_exact(s, model_name, h_out, w_out, device, precision,
                                   offload_after_upscale, eff)
    if was_4d:
        out = out.squeeze(2)
    return out


def _retarget_keyframes(positive, target_h, target_w, model_name, device, precision,
                        offload_after_upscale):
    """Приводит латенты minimax_keyframes (гайды Add Guide) к целевому H/W
    апскейленного видео той же обученной сетью: модель MiniMax H3 вшивает
    ряды ключевых кадров в сетку рядов генерации и требует совпадения
    пространственной сетки, иначе shape mismatch при семплировании.
    Рефы (minimax_refs) и аудио-латенты не трогаются: рефы по дизайну
    кросс-разрешенческие, аудио привязано ко времени, а не к пространству.
    Re-targets minimax_keyframes latents (Add Guide guides) to the target H/W
    of the upscaled video with the same learned network: the MiniMax H3 model
    splices keyframe rows into the generation row grid and requires the
    spatial grids to match, otherwise a shape mismatch hits sampling.
    Refs (minimax_refs) and audio latents are untouched: refs are
    cross-resolution by design, audio is tied to time, not space."""
    if not isinstance(positive, (list, tuple)):
        return positive
    changed = False
    out = []
    for pair in positive:
        if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
            out.append(pair)
            continue
        tensor, extra = pair[0], pair[1]
        kfs = extra.get("minimax_keyframes") if isinstance(extra, dict) else None
        if not kfs:
            out.append(pair)
            continue
        new_kfs = []
        pair_changed = False
        for kf in kfs:
            lat = kf.get("latent") if isinstance(kf, dict) else None
            if not torch.is_tensor(lat):
                new_kfs.append(kf)
                continue
            was_4d = lat.dim() == 4
            s5 = lat.unsqueeze(2) if was_4d else lat
            if s5.dim() != 5 or s5.shape[1] != H3_VIDEO_CHANNELS:
                new_kfs.append(kf)
                continue
            sh, sw = int(s5.shape[-2]), int(s5.shape[-1])
            if (sh, sw) == (target_h, target_w):
                new_kfs.append(kf)
                continue
            if target_h < sh or target_w < sw:
                logger.warning("%s keyframe latent %dx%d kept as is: target %dx%d is smaller",
                               LOG_PREFIX, sh, sw, target_h, target_w)
                new_kfs.append(kf)
                continue
            up = _upscale_video(s5, model_name, "target dimensions", 1.0,
                                target_w * VAE_DOWNSAMPLE, target_h * VAE_DOWNSAMPLE,
                                1.0, VAE_DOWNSAMPLE, False, device, precision,
                                offload_after_upscale)
            if was_4d:
                up = up.squeeze(2)
            new_kf = dict(kf)
            new_kf["latent"] = up
            new_kfs.append(new_kf)
            pair_changed = True
        if pair_changed:
            new_extra = dict(extra)
            new_extra["minimax_keyframes"] = new_kfs
            out.append([tensor, new_extra] + list(pair[2:]))
            changed = True
        else:
            out.append(pair)
    if changed:
        logger.info("%s minimax_keyframes latents retargeted to %dx%d",
                    LOG_PREFIX, target_h, target_w)
    return out if changed else positive


# ==========================================
# Нода ComfyUI (база) / ComfyUI node (base)
# ==========================================
class AGSoftMinimaxH3LatentUpscaler:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "upscale"
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = ("LATENT", "LATENT", "CONDITIONING")
    RETURN_NAMES = ("video_latent", "video_audio_latent", "positive")
    OUTPUT_TOOLTIPS = (
        "Upscaled video-only latent; if the input was joint AV, the audio member is dropped here.\n"
        "---\n"
        "Апскейленный латент только видео; если вход был совместным AV, аудиочасть здесь отброшена.",
        "Joint AV latent: upscaled video re-joined with the original untouched audio member. "
        "Raises an error if connected while the input had no audio part.\n"
        "---\n"
        "Совместный AV-латент: апскейленное видео, собранное обратно с исходной нетронутой аудиочастью. "
        "Выдаёт ошибку, если выход подключён, а на входе не было аудиочасти.",
        "Conditioning passthrough with minimax_keyframes latents retargeted to the upscaled "
        "resolution; None when the positive input is not connected.\n"
        "---\n"
        "Сквозной conditioning с латентами minimax_keyframes, приведёнными к апскейленному "
        "разрешению; None, когда вход positive не подключён.",
    )
    DESCRIPTION = (
        "Learned spatial upscaling of MiniMax H3 latents (24 ch) without VAE decode/encode. "
        "Smart input accepts a plain video latent or a native joint AV latent (NestedTensor [video, audio]): "
        "the joint form is split automatically, only the video member is upscaled, audio passes through untouched. "
        "Optional positive passthrough: connect conditioning with Add Guide keyframes and the node re-targets "
        "their latents to the upscaled resolution, so a downstream sampler never hits the keyframe row mismatch. "
        "Outputs: video_latent (upscaled video only), video_audio_latent (upscaled video re-joined with the "
        "original audio) and positive (retargeted conditioning or None).\n"
        "---\n"
        "Обученный пространственный апскейл латентов MiniMax H3 (24 канала) без VAE decode/encode. "
        "Умный вход принимает обычный видеолатент или совместный AV-латент (NestedTensor [video, audio]): "
        "совместная форма разделяется автоматически, апскейлится только видеочасть, аудио проходит без изменений. "
        "Опциональный сквозной проход positive: подключите conditioning с гайдами Add Guide — нода приведёт их "
        "латенты к апскейленному разрешению, и внешний семплер не поймает рассогласование рядов ключевых кадров. "
        "Выходы: video_latent (только апскейленное видео), video_audio_latent (апскейленное видео с исходным "
        "аудио) и positive (ретаргеченный conditioning или None)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = list(folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER))
        except Exception:
            models = []
        if not models:
            models = [" "]  # пустой combo ломает фронтенд / empty combo breaks the frontend
        required = {
            "latent": ("LATENT", {"tooltip": _T(
                "Smart input: plain 24-channel MiniMax H3 video latent (image or video) OR native joint "
                "audio-video latent (NestedTensor [video, audio]); the joint form is split automatically "
                "and only the video member is upscaled.",
                "Умный вход: обычный видеолатент MiniMax H3 на 24 канала (изображение или видео) ИЛИ совместный "
                "аудио-видео латент (NestedTensor [video, audio]); совместная форма разделяется автоматически, "
                "апскейлится только видеочасть.")}),
            "model_name": (models, {"tooltip": _T(
                "Checkpoint of the learned 3D upscaler from models/latent_upscale_models "
                "(.safetensors or .pth, architecture auto-detected).",
                "Чекпоинт обученного 3D-апскейлера из models/latent_upscale_models "
                "(.safetensors или .pth, архитектура определяется автоматически).")}),
            "mode": (["scale by multiplier", "target dimensions", "megapixels"],
                     {"default": "scale by multiplier", "tooltip": _T(
                "How the target pixel size is computed: source size x scale, exact width/height, "
                "or total area in megapixels at the source aspect ratio.",
                "Как вычисляется целевой размер в пикселях: размер источника x scale, точные width/height "
                "или общая площадь в мегапикселях с пропорцией источника.")}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": _T(
                "Upscale multiplier applied to the source pixel size (mode: scale by multiplier).",
                "Множитель апскейла к пиксельному размеру источника (режим: scale by multiplier).")}),
            "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8, "tooltip": _T(
                "Target pixel width (mode: target dimensions); snapped to the alignment grid.",
                "Целевая ширина в пикселях (режим: target dimensions); приводится к сетке выравнивания.")}),
            "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 8, "tooltip": _T(
                "Target pixel height (mode: target dimensions); snapped to the alignment grid.",
                "Целевая высота в пикселях (режим: target dimensions); приводится к сетке выравнивания.")}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": _T(
                "Target frame area in megapixels at the source aspect ratio (1024x1024 = 1 MP, mode: megapixels).",
                "Целевая площадь кадра в мегапикселях с пропорцией источника (1024x1024 = 1 МП, режим: megapixels).")}),
            "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1, "tooltip": _T(
                "Output pixel grid step: both output axes are rounded to multiples of this value, "
                "kept compatible with the 16-pixel grid of the H3 VAE.",
                "Шаг пиксельной сетки выхода: обе оси результата округляются до кратных этому значению "
                "с сохранением совместимости с сеткой 16 пикселей VAE H3.")}),
            "keep_proportion": ("BOOLEAN", {"default": True, "tooltip": _T(
                "Search nearby grid pairs and pick the one that best preserves the source aspect ratio; "
                "off rounds width and height to the grid independently.",
                "Перебирает соседние пары сетки и выбирает наилучшую по сохранению пропорций источника; "
                "off округляет ширину и высоту по сетке независимо.")}),
            "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": _T(
                "Compute device for the upscaler network; silently falls back to CPU when CUDA is unavailable.",
                "Устройство вычислений сети апскейлера; молча откатывается на CPU, когда CUDA недоступна.")}),
            "precision": (["fp32", "fp16", "bf16"], {"default": "fp16", "tooltip": _T(
                "Inference precision of the upscaler network; fp16/bf16 roughly halve VRAM usage.",
                "Точность инференса сети апскейлера; fp16/bf16 примерно вдвое снижают потребление VRAM.")}),
            "offload_after_upscale": ("BOOLEAN", {"default": False, "tooltip": _T(
                "Move the cached network to CPU after the run to free VRAM; the next run pays the transfer back.",
                "После запуска переносить кэшированную сеть на CPU для освобождения VRAM; следующий запуск платит "
                "обратным переносом.")}),
        }
        optional = {
            "positive": ("CONDITIONING", {"tooltip": _T(
                "Optional conditioning passthrough: keyframe latents of minimax_keyframes (Add Guide) are "
                "retargeted to the upscaled resolution; without it the output positive is None.",
                "Опциональный сквозной conditioning: латенты ключевых кадров minimax_keyframes (Add Guide) "
                "приводятся к апскейленному разрешению; без него выход positive равен None.")}),
        }
        hidden = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return {"required": required, "optional": optional, "hidden": hidden}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        name = kwargs.get("model_name")
        if name and name != " ":
            if folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name) is None:
                return f"Model not found in {_LATENT_UPSCALE_FOLDER}: {name}"
        return True

    def upscale(self, latent, model_name, mode, scale, width, height, megapixels, align,
                keep_proportion, device, precision, offload_after_upscale, positive=None,
                prompt=None, unique_id=None):
        if not model_name or model_name == " ":
            raise ValueError(f"Place .safetensors/.pth checkpoints into ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/ / "
                             f"положите чекпоинты .safetensors/.pth в ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/")
        video_in, audio_in, template = _split_input_latent(latent)
        video_in = _validate_video(video_in)
        orig_dtype = video_in.dtype
        video_out = _upscale_video(video_in, model_name, mode, scale, width, height, megapixels,
                                   align, keep_proportion, device, precision,
                                   offload_after_upscale).to(dtype=orig_dtype)
        video_latent = {"samples": video_out}
        if audio_in is None:
            # Аудиочасти нет: второй выход допустим только неподключённым
            # No audio member: the second output is only valid when unconnected
            if _output_linked(prompt, unique_id, 1):
                raise ValueError(
                    "video_audio_latent output is connected, but the input latent has no audio part (plain video "
                    "latent). Feed a joint AV latent or use the video_latent output. / "
                    "Выход video_audio_latent подключён, но во входном латенте нет аудиочасти (обычный видеолатент). "
                    "Подайте совместный AV-латент или используйте выход video_latent."
                )
        audio = None
        if audio_in is not None:
            audio = audio_in if audio_in.device == video_out.device else audio_in.to(device=video_out.device)
        av_latent = {"samples": _wrap_nested(video_out, audio, template)} if audio is not None else None
        pos_out = None
        if positive is not None:
            target_h, target_w = int(video_out.shape[-2]), int(video_out.shape[-1])
            pos_out = _retarget_keyframes(positive, target_h, target_w, model_name, device,
                                          precision, offload_after_upscale)
        elif _output_linked(prompt, unique_id, 2):
            raise ValueError(
                "positive output is connected, but the positive input is not: connect the conditioning with "
                "guides to retarget it. / выход positive подключён, но вход positive нет: подключите conditioning "
                "с гайдами для ретаргета."
            )
        return (video_latent, av_latent, pos_out)


# ==========================================
# Нода ComfyUI (Plus) / ComfyUI node (Plus)
# ==========================================
class AGSoftMinimaxH3LatentUpscalerPlus:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "sample"
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("video_latent", "video_audio_latent", "images", "audio")
    OUTPUT_TOOLTIPS = (
        "Refined video-only latent after upscale + refine sampling.\n"
        "---\n"
        "Уточнённый латент только видео после апскейла и refine-семплирования.",
        "Refined joint AV latent (upscaled video + audio after sampling); None when the input had no "
        "audio part. Raises an error if connected while the input had no audio.\n"
        "---\n"
        "Уточнённый совместный AV-латент (апскейленное видео + аудио после семплирования); None, если на "
        "входе не было аудиочасти. Выдаёт ошибку, если выход подключён, а на входе не было аудио.",
        "Decoded images/video of the refined latent.\n"
        "---\n"
        "Декодированные изображения/видео уточнённого латента.",
        "The decoded audio of the refined latent.\n"
        "---\n"
        "Декодированное аудио уточнённого латента.",
    )
    DESCRIPTION = (
        "🎬🧊 AGSoft Minimax H3 Latent Upscaler Plus.\n"
        "The input latent (plain video or joint AV) first passes the LEARNED 3D upscaler, then the upscaled "
        "latent is refined by the built-in sampler (few steps, denoise widget) and decoded by the built-in VAE "
        "(tiled/standard/off) with audio decode — the whole 'upscaler between two samplers' chain in one node. "
        "Keyframe latents of minimax_keyframes (Add Guide guides) inside positive are re-targeted to the "
        "upscaled resolution by the same learned network before refine sampling, so guides stay valid and the "
        "model never hits the keyframe row shape mismatch. Per-node phase cache: decode widget changes "
        "re-decode only, sampler widget changes do not re-run the learned upscale.\n"
        "---\n"
        "🎬🧊 AGSoft Minimax H3 Latent Upscaler Plus.\n"
        "Входной латент (обычный видео или совместный AV) сначала проходит ОБУЧЕННЫЙ 3D-апскейлер, затем "
        "апскейленный латент уточняется встроенным семплером (несколько шагов, виджет denoise) и декодируется "
        "встроенным VAE (tiled/standard/off) с декодом аудио — вся цепочка «апскейлер между двумя семплерами» "
        "в одной ноде. Латенты ключевых кадров minimax_keyframes (гайды Add Guide) внутри positive приводятся "
        "к апскейленному разрешению той же обученной сетью перед refine-семплированием, поэтому гайды остаются "
        "валидными и модель не ловит shape mismatch рядов ключевых кадров. Кэш фаз по ноде: смена виджетов "
        "декода только передекодирует, смена виджетов семплера не перезапускает обученный апскейл."
    )

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = list(folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER))
        except Exception:
            models = []
        if not models:
            models = [" "]  # пустой combo ломает фронтенд / empty combo breaks the frontend
        required = {
            # Блок апскейлера как в базовой ноде / Upscaler block as in the base node
            "latent": ("LATENT", {"tooltip": _T(
                "Smart input: plain 24-channel MiniMax H3 video latent (image or video) OR native joint "
                "audio-video latent (NestedTensor [video, audio]); it is upscaled by the learned network "
                "before refine sampling.",
                "Умный вход: обычный видеолатент MiniMax H3 на 24 канала (изображение или видео) ИЛИ совместный "
                "аудио-видео латент (NestedTensor [video, audio]); апскейлится обученной сетью перед "
                "refine-семплированием.")}),
            "model_name": (models, {"tooltip": _T(
                "Checkpoint of the learned 3D upscaler from models/latent_upscale_models "
                "(.safetensors or .pth, architecture auto-detected).",
                "Чекпоинт обученного 3D-апскейлера из models/latent_upscale_models "
                "(.safetensors или .pth, архитектура определяется автоматически).")}),
            "mode": (["scale by multiplier", "target dimensions", "megapixels"],
                     {"default": "scale by multiplier", "tooltip": _T(
                "How the target pixel size is computed: source size x scale, exact width/height, "
                "or total area in megapixels at the source aspect ratio.",
                "Как вычисляется целевой размер в пикселях: размер источника x scale, точные width/height "
                "или общая площадь в мегапикселях с пропорцией источника.")}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": _T(
                "Upscale multiplier applied to the source pixel size (mode: scale by multiplier).",
                "Множитель апскейла к пиксельному размеру источника (режим: scale by multiplier).")}),
            "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8, "tooltip": _T(
                "Target pixel width (mode: target dimensions); snapped to the alignment grid.",
                "Целевая ширина в пикселях (режим: target dimensions); приводится к сетке выравнивания.")}),
            "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 8, "tooltip": _T(
                "Target pixel height (mode: target dimensions); snapped to the alignment grid.",
                "Целевая высота в пикселях (режим: target dimensions); приводится к сетке выравнивания.")}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": _T(
                "Target frame area in megapixels at the source aspect ratio (1024x1024 = 1 MP, mode: megapixels).",
                "Целевая площадь кадра в мегапикселях с пропорцией источника (1024x1024 = 1 МП, режим: megapixels).")}),
            "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1, "tooltip": _T(
                "Output pixel grid step: both output axes are rounded to multiples of this value, "
                "kept compatible with the 16-pixel grid of the H3 VAE.",
                "Шаг пиксельной сетки выхода: обе оси результата округляются до кратных этому значению "
                "с сохранением совместимости с сеткой 16 пикселей VAE H3.")}),
            "keep_proportion": ("BOOLEAN", {"default": True, "tooltip": _T(
                "Search nearby grid pairs and pick the one that best preserves the source aspect ratio; "
                "off rounds width and height to the grid independently.",
                "Перебирает соседние пары сетки и выбирает наилучшую по сохранению пропорций источника; "
                "off округляет ширину и высоту по сетке независимо.")}),
            "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": _T(
                "Compute device for the upscaler network; silently falls back to CPU when CUDA is unavailable.",
                "Устройство вычислений сети апскейлера; молча откатывается на CPU, когда CUDA недоступна.")}),
            "precision": (["fp32", "fp16", "bf16"], {"default": "fp16", "tooltip": _T(
                "Inference precision of the upscaler network; fp16/bf16 roughly halve VRAM usage.",
                "Точность инференса сети апскейлера; fp16/bf16 примерно вдвое снижают потребление VRAM.")}),
            "offload_after_upscale": ("BOOLEAN", {"default": False, "tooltip": _T(
                "Move the cached network to CPU after the run to free VRAM; the next run pays the transfer back.",
                "После запуска переносить кэшированную сеть на CPU для освобождения VRAM; следующий запуск платит "
                "обратным переносом.")}),
            # Блок семплера как в KSampler Plus / Sampler block as in KSampler Plus
            "model": ("MODEL", {"tooltip": _T(
                "The model used for refine sampling of the upscaled latent.",
                "Модель для refine-семплирования апскейленного латента.")}),
            "positive": ("CONDITIONING", {"tooltip": _T(
                "The conditioning describing the attributes you want to include; keyframe latents of "
                "minimax_keyframes inside it are retargeted to the upscaled resolution automatically.",
                "Условие с атрибутами, которые нужно включить; латенты ключевых кадров minimax_keyframes "
                "внутри него приводятся к апскейленному разрешению автоматически.")}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True, "tooltip": _T(
                "The random seed used for creating the refine noise.",
                "Сид для генерации шума уточнения.")}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "tooltip": _T(
                "Refine steps count (3-5 is usually enough), unless sigmas provide their own.",
                "Число шагов уточнения (обычно достаточно 3-5), если сигмы не задали свои.")}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": _T(
                "CFG scale. Usually 1.0 for H3.",
                "Масштаб CFG. Обычно 1.0 для H3.")}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": _T(
                "The algorithm used when refine sampling.",
                "Алгоритм семплирования уточнения.")}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": _T(
                "The scheduler controls how noise is gradually removed.",
                "Планировщик определяет, как удаляется шум.")}),
        }
        optional = {
            "negative": ("CONDITIONING", {"tooltip": _T(
                "Optional negative conditioning. Empty conditioning is created if not connected.",
                "Опциональный негатив. Если не подключён — пустой конд.")}),
            "sigmas": ("SIGMAS", {"tooltip": _T(
                "Optional custom sigmas. If connected, steps/scheduler are ignored.",
                "Опциональные сигмы. Если подключены, steps/scheduler игнорируются.")}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _T(
                "Refine denoise amount (used only when sigmas are NOT connected); low values keep the "
                "upscaled content, high values re-generate more.",
                "Степень денойзинга уточнения (только если сигмы НЕ подключены); низкие значения сохраняют "
                "апскейленное содержимое, высокие генерируют заново.")}),
            "vae": ("VAE", {"tooltip": _T(
                "Optional VAE: decodes the refined latent into images right inside this node (tiled or standard).",
                "Опциональный VAE: декодирует уточнённый латент в картинки прямо внутри ноды (tiled или standard).")}),
            "audio_vae": ("VAE", {"tooltip": _T(
                "Optional Audio VAE for decoding the audio latent.",
                "Опциональный Audio VAE для декодирования аудио-латента.")}),
            "decode_mode": (["tiled", "standard", "off"], {"default": "standard", "tooltip": _T(
                "How the built-in VAE decodes latents: tiled = VAE Decode (Tiled) logic (low VRAM, big "
                "images/video), standard = plain decode (faster, more VRAM), off = no decode.",
                "Как встроенный VAE декодирует латенты: tiled = логика VAE Decode (Tiled) (мало VRAM, большие "
                "картинки/видео), standard = обычный декод (быстрее, больше VRAM), off = без декода.")}),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32, "tooltip": _T(
                "Tiled decode: tile size in pixels (like native VAE Decode (Tiled)).",
                "Плиточный декод: размер плитки в пикселях (как в нативной VAE Decode (Tiled)).")}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": _T(
                "Tiled decode: overlap between tiles in pixels; auto-clamped to tile_size / 4.",
                "Плиточный декод: перекрытие плиток в пикселях; авто-кламп до tile_size / 4.")}),
            "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": _T(
                "Only used for video VAEs: amount of frames to decode at a time.",
                "Только для видео-VAE: сколько кадров декодировать за один проход.")}),
            "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": _T(
                "Only used for video VAEs: amount of frames to overlap; auto-clamped.",
                "Только для видео-VAE: перекрытие в кадрах; авто-кламп.")}),
        }
        hidden = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return {"required": required, "optional": optional, "hidden": hidden}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        name = kwargs.get("model_name")
        if name and name != " ":
            if folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name) is None:
                return f"Model not found in {_LATENT_UPSCALE_FOLDER}: {name}"
        return True

    @staticmethod
    def _log(msg):
        """Единый лог-префикс для консоли ComfyUI.
        Single log prefix for the ComfyUI console."""
        print(f"{LOG_PREFIX} {msg}")

    @staticmethod
    def _vram_stats():
        """Снимок VRAM (MB): использовано и свободно. Возвращает None без CUDA.
        VRAM snapshot (MB): used and free. Returns None without CUDA."""
        if not torch.cuda.is_available():
            return None
        try:
            used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
            return (used_mb, free_mb)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ключ фазы семплирования: идентичность входов сверху + параметры.
    # Виджеты декода в ключ НЕ входят — их смена не инвалидирует кэш.
    # Sampling-phase key: identity of upstream inputs + params.
    # Decode widgets are NOT in the key — changing them keeps the cache.
    # ------------------------------------------------------------------
    @staticmethod
    def _sig_tuple(sigmas):
        if sigmas is None:
            return None
        if isinstance(sigmas, torch.Tensor):
            return tuple(float(x) for x in sigmas.detach().cpu().flatten().tolist())
        return tuple(float(x) for x in sigmas)

    def _sampling_key(self, model, positive, negative, latent_image, sigmas,
                      seed, steps, cfg, sampler_name, scheduler, denoise):
        refs = (model, positive, negative, latent_image, sigmas)
        params = (int(seed), int(steps), float(cfg), str(sampler_name), str(scheduler),
                  float(denoise), self._sig_tuple(sigmas))
        return (refs, params)

    @staticmethod
    def _key_match(entry, key):
        refs, params = key
        erefs, eparams = entry["refs"], entry["params"]
        if len(refs) != len(erefs):
            return False
        for a, b in zip(refs, erefs):
            if a is not b:
                return False
        return params == eparams

    def sample(self, latent, model_name, mode, scale, width, height, megapixels, align,
               keep_proportion, device, precision, offload_after_upscale,
               model, positive, seed, steps, cfg, sampler_name, scheduler,
               negative=None, sigmas=None, denoise=1.0, vae=None, audio_vae=None,
               decode_mode="standard", tile_size=512, overlap=64, temporal_size=64,
               temporal_overlap=8, prompt=None, unique_id=None):
        dec_mode = str(decode_mode)
        uid = str(unique_id) if unique_id is not None else None
        # Фаза 1: обученный апскейл входного латента (с кэшем по ноде).
        # Phase 1: learned upscale of the input latent (per-node cache).
        upscaled = self._upscale_phase(latent, uid, model_name, mode, scale, width, height,
                                       megapixels, align, keep_proportion, device, precision,
                                       offload_after_upscale)
        # Целевое разрешение апскейленного видео для ретаргета ключевых кадров.
        # Target resolution of the upscaled video for keyframe re-targeting.
        video_up, _, _ = _split_input_latent(upscaled)
        target_h, target_w = int(video_up.shape[-2]), int(video_up.shape[-1])
        # Фаза 2: refine-семплирование апскейленного латента (с кэшем по ноде).
        # Ключ строится по ИСХОДНОМУ positive, ретаргет применяется только в miss-ветке.
        # Phase 2: refine sampling of the upscaled latent (per-node cache).
        # The key uses the ORIGINAL positive; the retarget applies in the miss branch only.
        key = self._sampling_key(model, positive, negative, upscaled, sigmas,
                                 seed, steps, cfg, sampler_name, scheduler, denoise)
        entry = _SAMPLE_CACHE.get(uid) if uid is not None else None
        hit = entry is not None and self._key_match(entry, key)
        if hit:
            latent_dict = entry["latent"]
            video_lat = entry["video"]
            audio_lat = entry["audio"]
            self._log("CACHE HIT: refine sampling skipped, VAE decode re-run only")
        else:
            pos_sample = _retarget_keyframes(positive, target_h, target_w, model_name,
                                             device, precision, offload_after_upscale)
            latent_dict, video_lat, audio_lat = self._run_sampling(
                model, pos_sample, negative, upscaled, sigmas, seed, steps, cfg,
                sampler_name, scheduler, denoise)
            if uid is not None:
                _SAMPLE_CACHE[uid] = {"refs": key[0], "params": key[1], "latent": latent_dict,
                                      "video": video_lat, "audio": audio_lat}
                while len(_SAMPLE_CACHE) > _CACHE_LIMIT:
                    _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
                self._log("refine sampling executed, result cached for re-decode")
        # Фаза 3: декод видео и аудио с текущими виджетами декода.
        # Phase 3: video and audio decode with the current decode widgets.
        image = None
        if vae is None or video_lat is None or dec_mode == "off":
            # Декод выключен или нечем декодировать: явное сообщение вместо тишины.
            # Decode disabled or nothing to decode: explicit notice, not silence.
            if vae is None:
                reason = "vae not connected"
            elif video_lat is None:
                reason = "no video latent"
            else:
                reason = "mode=off"
            self._log(f"VAE decode skipped ({reason}): images output = empty "
                      f"{PLACEHOLDER_SIZE}x{PLACEHOLDER_SIZE} placeholder")
        else:
            image = self._decode_and_log(vae, video_lat, dec_mode, tile_size, overlap,
                                         temporal_size, temporal_overlap, "[refine]")
        if image is None:
            image = torch.zeros((1, PLACEHOLDER_SIZE, PLACEHOLDER_SIZE, 3), dtype=torch.float32)
        audio_out = None
        if audio_vae is not None and audio_lat is not None:
            audio_out = self._decode_audio(audio_vae, audio_lat)
        if audio_out is None:
            sr = audio_vae.audio_sample_rate if hasattr(audio_vae, "audio_sample_rate") else None
            audio_out = {
                "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_rate": sr or DEFAULT_AUDIO_SAMPLE_RATE,
            }
        if audio_lat is None and _output_linked(prompt, unique_id, 1):
            raise ValueError(
                "video_audio_latent output is connected, but the input latent has no audio part (plain video "
                "latent). Feed a joint AV latent or use the video_latent output. / "
                "Выход video_audio_latent подключён, но во входном латенте нет аудиочасти (обычный видеолатент). "
                "Подайте совместный AV-латент или используйте выход video_latent."
            )
        video_latent = {"samples": video_lat}
        av_latent = latent_dict if audio_lat is not None else None
        return (video_latent, av_latent, image, audio_out)

    # ------------------------------------------------------------------
    # Фаза апскейла: обученная сеть по входному латенту, результат в кэш.
    # Upscale phase: learned network over the input latent, result cached.
    # ------------------------------------------------------------------
    def _upscale_phase(self, latent, uid, model_name, mode, scale, width, height, megapixels,
                       align, keep_proportion, device, precision, offload_after_upscale):
        if not model_name or model_name == " ":
            raise ValueError(f"Place .safetensors/.pth checkpoints into ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/ / "
                             f"положите чекпоинты .safetensors/.pth в ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/")
        up_key = (str(model_name), str(mode), float(scale), int(width), int(height),
                  float(megapixels), int(align), bool(keep_proportion), str(device),
                  str(precision), bool(offload_after_upscale))
        entry = _UPSCALE_CACHE.get(uid) if uid is not None else None
        if entry is not None and entry["refs"] is latent and entry["params"] == up_key:
            self._log("CACHE HIT: upscale skipped, cached upscaled latent reused")
            return entry["upscaled"]
        video_in, audio_in, template = _split_input_latent(latent)
        video_in = _validate_video(video_in)
        orig_dtype = video_in.dtype
        video_out = _upscale_video(video_in, model_name, mode, scale, width, height, megapixels,
                                   align, keep_proportion, device, precision,
                                   offload_after_upscale).to(dtype=orig_dtype)
        if audio_in is None:
            upscaled = {"samples": video_out}
        else:
            audio = audio_in if audio_in.device == video_out.device else audio_in.to(device=video_out.device)
            upscaled = {"samples": _wrap_nested(video_out, audio, template)}
        if uid is not None:
            _UPSCALE_CACHE[uid] = {"refs": latent, "params": up_key, "upscaled": upscaled}
            while len(_UPSCALE_CACHE) > _CACHE_LIMIT:
                _UPSCALE_CACHE.pop(next(iter(_UPSCALE_CACHE)))
        self._log("upscale phase done, upscaled latent passed to refine sampling")
        return upscaled

    # ------------------------------------------------------------------
    # Один прогон refine-семплирования без декода.
    # One refine sampling pass without decode.
    # ------------------------------------------------------------------
    def _run_sampling(self, model, positive, negative, latent_image, sigmas, seed, steps, cfg,
                      sampler_name, scheduler, denoise):
        # Если негатив не подан, создаём пустой конд.
        # If negative is not provided, create empty conditioning.
        if negative is None:
            negative = self._create_empty_cond(positive)
        # Семплирование: стандартный путь или кастомные сигмы.
        # Sampling: standard path or custom sigmas.
        t0 = time.time()
        if sigmas is None:
            result = common_ksampler(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative=negative,
                latent=latent_image, denoise=denoise
            )
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
                comfy.model_management.get_torch_device(),
                sampler, sigmas,
                model_options=model.model_options if hasattr(model, "model_options") else {},
                latent_image=latent_samples,
                denoise_mask=None, callback=None, disable_pbar=False, seed=seed
            )
            latent_dict = {"samples": samples}
            if isinstance(latent_image, dict):
                for key, value in latent_image.items():
                    if key not in latent_dict:
                        latent_dict[key] = value
            steps_used = max(1, len(sigmas) - 1)
            scheduler_used = "manual sigmas"
            denoise_used = float(sigmas[0])
        dt_sample = time.time() - t0
        self._log(f"[refine] sampling done in {dt_sample:.2f}s | steps={steps_used} cfg={cfg:.2f} "
                  f"sampler={sampler_name} scheduler={scheduler_used} denoise={denoise_used:.2f}")
        video_lat, audio_lat = self._split_av_latent(latent_dict)
        return latent_dict, video_lat, audio_lat

    # ------------------------------------------------------------------
    # Обёртка декода с подробным логом: режим, плитки, время, VRAM.
    # Decode wrapper with verbose log: mode, tiles, time, VRAM.
    # ------------------------------------------------------------------
    def _decode_and_log(self, vae, latent, mode, tile_size, overlap, temporal_size, temporal_overlap, tag):
        samples = latent["samples"] if isinstance(latent, dict) else latent
        if hasattr(samples, "is_nested") and samples.is_nested:
            try:
                samples = samples.unbind()[0]
            except Exception:
                pass
        shape = tuple(int(x) for x in samples.shape) if hasattr(samples, "shape") else "?"
        dtype = str(samples.dtype) if hasattr(samples, "dtype") else "?"
        self._log(f"{tag} VAE decode start | mode={mode} | latent shape={shape} dtype={dtype}")
        vram_before = self._vram_stats()
        t0 = time.time()
        try:
            if mode == "standard":
                images = vae.decode(samples)
                actual_mode = "standard"
                tile_info = None
            else:
                try:
                    images, tile_info = self._decode_tiled_logged(vae, samples, tile_size, overlap,
                                                                  temporal_size, temporal_overlap)
                    actual_mode = "tiled"
                except Exception as e:
                    self._log(f"{tag} tiled decode FAILED, fallback to standard: {type(e).__name__}: {e}")
                    t1 = time.time()
                    images = vae.decode(samples)
                    dt = time.time() - t1
                    actual_mode = "standard (fallback after tiled error)"
                    tile_info = None
                    self._log(f"{tag} fallback standard decode in {dt:.2f}s")
        except Exception as e:
            self._log(f"{tag} VAE decode ERROR: {type(e).__name__}: {e}")
            return None
        dt = time.time() - t0
        if images is None:
            self._log(f"{tag} VAE decode returned None in {dt:.2f}s")
            return None
        if images.dim() == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        out_shape = tuple(int(x) for x in images.shape)
        vram_after = self._vram_stats()
        mem_line = ""
        if vram_before is not None and vram_after is not None:
            used_b, free_b = vram_before
            used_a, free_a = vram_after
            delta = used_a - used_b
            mem_line = f" | VRAM used {used_b:.0f} -> {used_a:.0f} MB (Δ {delta:+.0f}), free {free_a:.0f} MB"
        tiles_line = ""
        if tile_info:
            tiles_line = (f" | tiles px={tile_info['tile_px']}x{tile_info['tile_px']} "
                          f"overlap_px={tile_info['overlap_px']} "
                          f"latent_tiles={tile_info['tile_lat']}x{tile_info['tile_lat']} "
                          f"overlap_lat={tile_info['overlap_lat']} "
                          f"spacial_comp={tile_info['spacial']}")
            if tile_info.get("temporal") is not None:
                tiles_line += (f" | temporal_comp={tile_info['temporal']} "
                               f"t_frames={tile_info['t_frames']} t_overlap={tile_info['t_overlap']}")
        self._log(f"{tag} VAE decode done | mode={actual_mode} | output shape={out_shape} "
                  f"time={dt:.2f}s{tiles_line}{mem_line}")
        return images

    # ------------------------------------------------------------------
    # Плиточный декод с логированием реальных параметров (после компрессии VAE).
    # Tiled decode logging the real parameters (after VAE compression).
    # ------------------------------------------------------------------
    def _decode_tiled_logged(self, vae, latent, tile_size, overlap, temporal_size, temporal_overlap):
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
        tile_info = {
            "tile_px": int(tile_size),
            "overlap_px": int(overlap),
            "tile_lat": tx,
            "overlap_lat": ov,
            "spacial": int(s_comp),
            "temporal": int(t_comp) if t_comp is not None else None,
            "t_frames": ts,
            "t_overlap": to,
        }
        try:
            images = vae.decode_tiled(latent, tile_x=tx, tile_y=tx, overlap=ov, tile_t=ts, overlap_t=to)
        except TypeError:
            # Старое ядро без временных аргументов плиток.
            # Old core without temporal tile arguments.
            images = vae.decode_tiled(latent, tile_x=tx, tile_y=tx, overlap=ov)
            tile_info["t_frames"] = None
            tile_info["t_overlap"] = None
        return images, tile_info

    # ------------------------------------------------------------------
    # H3: {"samples": NestedTensor((video, audio))} -> (video, audio).
    # H3: {"samples": NestedTensor((video, audio))} -> (video, audio).
    # ------------------------------------------------------------------
    def _split_av_latent(self, latent_dict):
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

    def _decode_audio(self, audio_vae, audio_latent):
        decoded = None
        try:
            if hasattr(audio_vae, "decode_audio"):
                decoded = audio_vae.decode_audio(audio_latent)
            elif hasattr(audio_vae, "decode"):
                decoded = audio_vae.decode(audio_latent)
        except Exception as e:
            self._log(f"Audio decode ERROR: {type(e).__name__}: {e}")
            return None
        return self._normalize_audio(decoded, audio_vae)

    # ------------------------------------------------------------------
    # Нормализация аудио в стандарт ComfyUI: {"waveform": [B,C,N], "sample_rate"}.
    # Normalize audio to ComfyUI standard: {"waveform": [B,C,N], "sample_rate"}.
    # ------------------------------------------------------------------
    def _normalize_audio(self, decoded, audio_vae=None):
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
        # MiniMax H3 Audio VAE отдаёт channel-last [B, N, C] -> приводим к [B, C, N].
        # MiniMax H3 Audio VAE returns channel-last [B, N, C] -> convert to [B, C, N].
        if waveform.dim() == 3:
            b, d1, d2 = waveform.shape
            if d2 <= 8 < d1:
                waveform = waveform.movedim(-1, 1)
        if waveform.dim() != 3:
            self._log(f"Unexpected audio waveform dims: {waveform.dim()}")
            return None
        return {
            "waveform": waveform.detach().cpu().float().contiguous(),
            "sample_rate": int(sample_rate),
        }

    # ------------------------------------------------------------------
    # Пустой конд на основе структуры позитивного.
    # Empty conditioning based on positive structure.
    # ------------------------------------------------------------------
    def _create_empty_cond(self, positive):
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


#------------------------------------------------------------------------------
# Регистрация нод.
# Node registration.
#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AGSoftMinimaxH3LatentUpscaler": AGSoftMinimaxH3LatentUpscaler,
    "AGSoftMinimaxH3LatentUpscalerPlus": AGSoftMinimaxH3LatentUpscalerPlus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMinimaxH3LatentUpscaler": "🎬AGSoft Minimax H3 Latent Upscaler",
    "AGSoftMinimaxH3LatentUpscalerPlus": "🎬🧊AGSoft Minimax H3 Latent Upscaler Plus",
}
