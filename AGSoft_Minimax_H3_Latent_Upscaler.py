"""
==============================================================================
AGSoft_Minimax_H3_Latent_Upscaler.py
==============================================================================
Нода: 🎬🧊AGSoft Minimax H3 Latent Upscaler
Описание / Description:
Обученный пространственный апскейл латентов MiniMax H3 (24 канала, 4D/5D)
напрямую в latent-пространстве, без дорогого VAE decode/encode.
Вход умный: обычный видеолатент или совместный AV-латент (NestedTensor
[video, audio]) — совместная форма разделяется автоматически, апскейлится
только видеочасть, аудиочасть проходит сквозь ноду без изменений.
---
Learned spatial upscaling of MiniMax H3 (24-channel, 4D/5D) latents directly
in latent space, bypassing costly VAE decode/encode.
Smart input: plain video latent or native joint AV latent (NestedTensor
[video, audio]) — the joint form is split automatically, only the video
member is upscaled while the audio member passes through untouched.
Возможности / Features:
⚡ Умный вход: plain video latent или joint AV latent с авторазделением
   Smart input: plain video latent or joint AV latent with auto-split
⚡ Два выхода: video_latent и собранный обратно video_audio_latent
   (заменяет отдельные ноды Separate AV Latent и Concat AV Latent)
   Two outputs: video_latent and re-joined video_audio_latent
   (replaces the Separate AV Latent and Concat AV Latent nodes)
⚡ Три режима размера: множитель, мегапиксели, целевые размеры
   Three sizing modes: scale by multiplier, megapixels, target dimensions
⚡ Двухосевое выравнивание по пиксельной сетке с сохранением пропорций
   Dual-axis pixel-grid alignment with optional aspect-ratio lock
⚡ fp32/fp16/bf16, кэш моделей по (checkpoint, device, precision), offload на CPU
   fp32/fp16/bf16, model cache per (checkpoint, device, precision), CPU offload
⚡ Автоопределение архитектуры чекпоинта (каналы, блоки, temporal-свёртки)
   Auto-detects checkpoint architecture (channels, blocks, temporal convs)
Автор / Author: AGSoft
Дата / Date: 19.10.2026
==============================================================================
"""
import os
import re
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
import comfy.utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Minimax H3 Latent Upscaler] v10.19 loaded (smart AV input, video_latent + video_audio_latent outputs)")

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
        return (max(grid, int(round(w_target / grid)) * grid), max(grid, int(round(h_target / grid)) * grid))
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
            score = (ratio_error * 4.0 + size_error, ratio_error, size_error, abs(width - w_target) + abs(height - ideal_h))
            if best is None or score < best[0]:
                best = (score, width, height)
    return best[1], best[2]

# Статистики нормализации латентов H3 (24 канала, из тренировочного кода)
# H3 latent normalization stats (24 channels, from training code)
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
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
    return bool(getattr(value, "is_nested", False)) and hasattr(value, "unbind")

def _nested_members(value):
    members = list(value.unbind())
    for m in members:
        if not torch.is_tensor(m):
            raise TypeError("Joint AV latent members must be torch.Tensor / члены совместного AV-латента должны быть torch.Tensor")
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
    raise ValueError("This ComfyUI build lacks NestedTensor support for joint AV latents / эта сборка ComfyUI не поддерживает NestedTensor для совместных AV-латентов")

def _split_input_latent(latent):
    # Умный вход: joint AV -> (video, audio, template), plain -> (video, None, None)
    # Smart input: joint AV -> (video, audio, template), plain -> (video, None, None)
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Input latent must be a LATENT dictionary with 'samples' / входной latent должен быть словарём LATENT с ключом 'samples'")
    samples = latent["samples"]
    if _is_nested(samples):
        members = _nested_members(samples)
        if len(members) != 2:
            raise ValueError(f"Joint AV latent must contain exactly [video, audio] members, got {len(members)} / совместный AV-латент должен содержать ровно [video, audio], получено {len(members)}")
        return members[0], members[1], samples
    if torch.is_tensor(samples):
        return samples, None, None
    raise TypeError(f"Unsupported latent samples type: {type(samples).__name__} / неподдерживаемый тип samples: {type(samples).__name__}")

def _validate_video(video):
    if not torch.is_tensor(video) or not video.is_floating_point():
        raise TypeError("MiniMax H3 video latent must be a floating-point torch.Tensor / видеолатент MiniMax H3 должен быть float-тензором")
    if video.ndim not in (4, 5) or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"MiniMax H3 video latent must be Bx24xHxW or Bx24xTxHxW, got {tuple(video.shape)} / видеолатент MiniMax H3 должен быть Bx24xHxW или Bx24xTxHxW, получено {tuple(video.shape)}")
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
            nn.SiLU(), nn.Dropout(p=dropout), zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
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
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1), padding=(padding, 0, 0), groups=channels)
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
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5):
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
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512, "dropout": 0.1, "temporal_every": 2, "temporal_kernel": 5}
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
    model.eval().requires_grad_(False)
    model.to(device)
    del sd
    MODEL_CACHE[key] = model
    logger.info("[AGSoft MinimaxH3] loaded %s | params=%s | temporal=%s | precision=%s | device=%s", name, f"{sum(p.numel() for p in model.parameters()):,}", "on" if cfg["temporal_every"] > 0 else "off", precision, device)
    return model

def upscale_latent_exact(video, model_name, target_h, target_w, device, precision, offload_after_upscale=False, scale_embedding=None):
    # Обученный пространственный ресайз до точных latent H/W при сохранении B/C/T
    # Learned spatial resize to exact latent H/W while preserving B/C/T
    if not isinstance(video, torch.Tensor) or not video.is_floating_point():
        raise TypeError("Input latent must be a floating-point torch.Tensor / входной latent должен быть float-тензором")
    if video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(f"MiniMax H3 latent must be Bx24xTxHxW, got {tuple(video.shape)} / латент MiniMax H3 должен быть Bx24xTxHxW, получено {tuple(video.shape)}")
    src_h, src_w = int(video.shape[-2]), int(video.shape[-1])
    if target_h < src_h or target_w < src_w:
        raise ValueError("Only upscaling is supported: target must not shrink any spatial axis / поддерживается только апскейл: целевой размер не может быть меньше исходного")
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
        raise RuntimeError(f"Upscaler returned shape {tuple(out.shape)}, expected {expected} / апскейлер вернул форму {tuple(out.shape)}, ожидалась {expected}")
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Upscaler returned NaN or Inf values / апскейлер вернул NaN или Inf")
    out = out.to(device="cpu", dtype=video.dtype)
    if dev.type == "cuda":
        if offload_after_upscale:
            model.to("cpu")
        torch.cuda.empty_cache()
    return out

# ==========================================
# Нода ComfyUI / ComfyUI node
# ==========================================
class AGSoftMinimaxH3LatentUpscaler:
    CATEGORY = "AGSoft/MiniMaxH3"
    FUNCTION = "upscale"
    WEB_DIRECTORY = "./web"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "video_audio_latent")
    OUTPUT_TOOLTIPS = (
        "Upscaled video-only latent; if the input was joint AV, the audio member is dropped here."
        "\n---\n"
        "Апскейленный латент только видео; если вход был совместным AV, аудиочасть здесь отброшена."
        ,
        "Joint AV latent: upscaled video re-joined with the original untouched audio member. "
        "Raises an error if connected while the input had no audio part."
        "\n---\n"
        "Совместный AV-латент: апскейленное видео, собранное обратно с исходной нетронутой аудиочастью. "
        "Выдаёт ошибку, если выход подключён, а на входе не было аудиочасти."
        ,
    )
    DESCRIPTION = (
        "Learned spatial upscaling of MiniMax H3 latents (24 ch) without VAE decode/encode. "
        "Smart input accepts a plain video latent or a native joint AV latent (NestedTensor [video, audio]): "
        "the joint form is split automatically, only the video member is upscaled, audio passes through untouched. "
        "Outputs: video_latent (upscaled video only) and video_audio_latent (upscaled video re-joined with the original audio), "
        "which replaces the Separate AV Latent / Concat AV Latent node pair."
        "\n---\n"
        "Обученный пространственный апскейл латентов MiniMax H3 (24 канала) без VAE decode/encode. "
        "Умный вход принимает обычный видеолатент или совместный AV-латент (NestedTensor [video, audio]): "
        "совместная форма разделяется автоматически, апскейлится только видеочасть, аудио проходит без изменений. "
        "Выходы: video_latent (только апскейленное видео) и video_audio_latent (апскейленное видео, собранное с исходным аудио), "
        "что заменяет пару нод Separate AV Latent / Concat AV Latent."
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
            "latent": ("LATENT", {"tooltip": (
                "Smart input: plain 24-channel MiniMax H3 video latent (image or video) OR native joint "
                "audio-video latent (NestedTensor [video, audio]); the joint form is split automatically "
                "and only the video member is upscaled."
                "\n---\n"
                "Умный вход: обычный видеолатент MiniMax H3 на 24 канала (изображение или видео) ИЛИ совместный "
                "аудио-видео латент (NestedTensor [video, audio]); совместная форма разделяется автоматически, "
                "апскейлится только видеочасть."
            )}),
            "model_name": (models, {"tooltip": (
                "Checkpoint of the learned 3D upscaler from models/latent_upscale_models "
                "(.safetensors or .pth, architecture auto-detected)."
                "\n---\n"
                "Чекпоинт обученного 3D-апскейлера из models/latent_upscale_models "
                "(.safetensors или .pth, архитектура определяется автоматически)."
            )}),
            "mode": (["scale by multiplier", "target dimensions", "megapixels"], {"default": "scale by multiplier", "tooltip": (
                "How the target pixel size is computed: source size x scale, exact width/height, "
                "or total area in megapixels at the source aspect ratio."
                "\n---\n"
                "Как вычисляется целевой размер в пикселях: размер источника x scale, точные width/height "
                "или общая площадь в мегапикселях с пропорцией источника."
            )}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": (
                "Upscale multiplier applied to the source pixel size (mode: scale by multiplier)."
                "\n---\n"
                "Множитель апскейла к пиксельному размеру источника (режим: scale by multiplier)."
            )}),
            "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8, "tooltip": (
                "Target pixel width (mode: target dimensions); snapped to the alignment grid."
                "\n---\n"
                "Целевая ширина в пикселях (режим: target dimensions); приводится к сетке выравнивания."
            )}),
            "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 8, "tooltip": (
                "Target pixel height (mode: target dimensions); snapped to the alignment grid."
                "\n---\n"
                "Целевая высота в пикселях (режим: target dimensions); приводится к сетке выравнивания."
            )}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1, "tooltip": (
                "Target frame area in megapixels at the source aspect ratio (1024x1024 = 1 MP, mode: megapixels)."
                "\n---\n"
                "Целевая площадь кадра в мегапикселях с пропорцией источника (1024x1024 = 1 МП, режим: megapixels)."
            )}),
            "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1, "tooltip": (
                "Output pixel grid step: both output axes are rounded to multiples of this value, "
                "kept compatible with the 16-pixel grid of the H3 VAE."
                "\n---\n"
                "Шаг пиксельной сетки выхода: обе оси результата округляются до кратных этому значению "
                "с сохранением совместимости с сеткой 16 пикселей VAE H3."
            )}),
            "keep_proportion": ("BOOLEAN", {"default": True, "tooltip": (
                "Search nearby grid pairs and pick the one that best preserves the source aspect ratio; "
                "off rounds width and height to the grid independently."
                "\n---\n"
                "Перебирает соседние пары сетки и выбирает наилучшую по сохранению пропорций источника; "
                "off округляет ширину и высоту по сетке независимо."
            )}),
            "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": (
                "Compute device for the upscaler network; silently falls back to CPU when CUDA is unavailable."
                "\n---\n"
                "Устройство вычислений сети апскейлера; молча откатывается на CPU, если CUDA недоступна."
            )}),
            "precision": (["fp32", "fp16", "bf16"], {"default": "fp16", "tooltip": (
                "Inference precision of the upscaler network; fp16/bf16 roughly halve VRAM usage."
                "\n---\n"
                "Точность инференса сети апскейлера; fp16/bf16 примерно вдвое снижают потребление VRAM."
            )}),
            "offload_after_upscale": ("BOOLEAN", {"default": False, "tooltip": (
                "Move the cached network to CPU after the run to free VRAM; the next run pays the transfer back."
                "\n---\n"
                "После запуска переносить кэшированную сеть на CPU для освобождения VRAM; следующий запуск платит обратным переносом."
            )}),
        }
        hidden = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return {"required": required, "hidden": hidden}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        name = kwargs.get("model_name")
        if name and name != " ":
            if folder_paths.get_full_path(_LATENT_UPSCALE_FOLDER, name) is None:
                return f"Model not found in {_LATENT_UPSCALE_FOLDER}: {name}"
        return True

    def _upscale_video(self, src, model_name, mode, scale, width, height, megapixels, align, keep_proportion, device, precision, offload_after_upscale):
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
            raise ValueError("This model only supports upscaling: target is smaller than source / модель поддерживает только апскейл: цель меньше источника")
        if w_out == w_in and h_out == h_in:
            logger.info("[AGSoft MinimaxH3] target equals source, video latent returned unchanged / цель равна источнику, видеолатент возвращён без изменений")
            out = s.clone()
        else:
            logger.info("[AGSoft MinimaxH3] latent %dx%d -> %dx%d | pixels %dx%d | scale=%.3f", w_in, h_in, w_out, h_out, w_out * ds, h_out * ds, eff)
            # 4. Инференс целиком по времени: temporal-свёртки не допускают чанкования
            # 4. Full-sequence inference: temporal convs make chunking non-equivalent
            out = upscale_latent_exact(s, model_name, h_out, w_out, device, precision, offload_after_upscale, eff)
        if was_4d:
            out = out.squeeze(2)
        return out

    def upscale(self, latent, model_name, mode, scale, width, height, megapixels, align, keep_proportion, device, precision, offload_after_upscale, prompt=None, unique_id=None):
        if not model_name or model_name == " ":
            raise ValueError(f"Place .safetensors/.pth checkpoints into ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/ / положите чекпоинты .safetensors/.pth в ComfyUI/models/{_LATENT_UPSCALE_FOLDER}/")
        video_in, audio_in, template = _split_input_latent(latent)
        video_in = _validate_video(video_in)
        orig_dtype = video_in.dtype
        video_out = self._upscale_video(video_in, model_name, mode, scale, width, height, megapixels, align, keep_proportion, device, precision, offload_after_upscale).to(dtype=orig_dtype)
        video_latent = {"samples": video_out}
        if audio_in is None:
            # Аудиочасти нет: второй выход допустим только неподключённым
            # No audio member: the second output is only valid when unconnected
            if _output_linked(prompt, unique_id, 1):
                raise ValueError(
                    "video_audio_latent output is connected, but the input latent has no audio part (plain video latent). "
                    "Feed a joint AV latent or use the video_latent output. / "
                    "Выход video_audio_latent подключён, но во входном латенте нет аудиочасти (обычный видеолатент). "
                    "Подайте совместный AV-латент или используйте выход video_latent."
                )
            return (video_latent, None)
        audio = audio_in if audio_in.device == video_out.device else audio_in.to(device=video_out.device)
        av_latent = {"samples": _wrap_nested(video_out, audio, template)}
        return (video_latent, av_latent)

NODE_CLASS_MAPPINGS = {
    "AGSoftMinimaxH3LatentUpscaler": AGSoftMinimaxH3LatentUpscaler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMinimaxH3LatentUpscaler": "🎬🧊AGSoft Minimax H3 Latent Upscaler",
}