"""
==============================================================================
AGSoft_KSampler_Plus.py
==============================================================================
Нода: 🧩 AGSoft KSampler Plus
Описание / Description:
Универсальный KSampler со встроенным VAE Decode (Tiled) для изображений:
плиточный декод живёт внутри ноды, отдельная нода VAE Decode не нужна.
---
Universal KSampler with built-in VAE Decode (Tiled) for images: the tiled
decode lives inside the node, no separate VAE Decode node is required.
Возможности / Features:
⚡ Встроенный декод VAE: tiled / standard / off; параметры плитки как у
нативной VAE Decode (Tiled): tile_size, overlap, temporal_size,
temporal_overlap (temporal работает только с видео-VAE).
Built-in VAE decode: tiled / standard / off; tile params mirror the native
VAE Decode (Tiled): tile_size, overlap, temporal_size, temporal_overlap
(temporal works only with video VAEs).
⚡ Кэш семплирования по нодам: повторный Queue с изменёнными виджетами декода
(режим/плитки) НЕ пересемплирует шаги — только передекодирует (CACHE HIT).
Per-node sampling cache: a re-queue with changed decode widgets (mode/tiles)
does NOT re-sample the steps — it only re-decodes (CACHE HIT).
⚡ Подробный лог декода в консоли: режим, размеры плиток после деления на
spacial/temporal compression VAE, время, VRAM. Видно, работает ли tiled.
Verbose decode log in console: mode, tile sizes after VAE spacial/temporal
compression, time, VRAM. Lets you see that tiled decode actually runs.
⚡ Явные сообщения о пропуске декода: mode=off / нет VAE / нет латента
логируются как skip; пустой плейсхолдер 64x64 (чётный) не роняет видео-кодеры.
Explicit decode-skip notices: mode=off / no VAE / no latent are logged as
skip; the empty 64x64 placeholder (even size) does not break video encoders.
⚡ options словарём = одно переопределение; options списком = вся серия за
ОДИН Queue (картинки батчем); LoRA-ключи патчат клон модели на каждый элемент.
options as dict = single override; options as list = whole series in ONE
queue run (images batched); LoRA keys patch a model clone per series item.
⚡ Плиточный декод повторяет логику ядра: клампы overlap, учёт spacial /
temporal compression VAE, фолбэк на обычный декод при ошибке.
Tiled decode mirrors core logic: overlap clamps, VAE spacial / temporal
compression accounted, fallback to standard decode on error.
⚡ MiniMax H3: AV-латент NestedTensor((video, audio)) распаковывается через
.tensors; channel-last аудио приводится к стандарту ComfyUI [B,C,N].
MiniMax H3: AV latent NestedTensor((video, audio)) unpacked via .tensors;
channel-last audio converted to ComfyUI standard [B,C,N].
Автор / Author: AGSoft
Дата / Date: 30.10.2026
==============================================================================
"""
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
from nodes import common_ksampler
import torch
import json
import time
from typing import Dict, Any

# print("[AGSoft KSampler Plus] v10.30 loaded (tiled VAE decode + logs + sampling cache)")

DEFAULT_AUDIO_SAMPLE_RATE = 32000
OPTIONS_TYPE = "KSAMPLER_OPTIONS"
LOG_PREFIX = "[AGSoft KSampler Plus]"
# Плейсхолдер пустых картинок: чётный размер, чтобы видео-кодеры не падали.
# Empty-image placeholder: even size so video encoders do not fail.
PLACEHOLDER_SIZE = 64
# Кэш фазы семплирования по unique_id ноды: повторный прогон с изменёнными
# виджетами декода не пересемплирует, а только передекодирует.
# Sampling-phase cache per node unique_id: a re-run with changed decode
# widgets re-decodes only, without re-sampling.
_SAMPLE_CACHE = {}
_CACHE_LIMIT = 8


#==============================================================================
# 🧩 AGSoft KSampler Plus
#==============================================================================
class AGSoft_KSampler_Plus:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent.\nМодель для денойзинга входного латента."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include.\nУсловие с атрибутами, которые нужно включить."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise.\nЛатентное изображение для денойзинга."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise.\nСид для генерации шума."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Steps count, unless sigmas/options provide their own.\nШаги, если сигмы/options не задали свои."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "CFG scale. Usually 1.0 for H3.\nМасштаб CFG. Обычно 1.0 для H3."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling.\nАлгоритм сэмплирования."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed.\nПланировщик определяет, как удаляется шум."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Optional negative conditioning. Empty conditioning is created if not connected.\nОпциональный негатив. Если не подключён — пустой конд."}),
                "sigmas": ("SIGMAS", {"tooltip": "Optional custom sigmas. If connected, steps/scheduler are ignored.\nОпциональные сигмы. Если подключены, steps/scheduler игнорируются."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise amount (used only when sigmas are NOT connected).\nСтепень денойзинга (только если сигмы НЕ подключены)."}),
                "vae": ("VAE", {"tooltip": "Optional VAE: decodes latents into images right inside this node (tiled or standard).\nОпциональный VAE: декодирует латенты в картинки прямо внутри ноды (tiled или standard)."}),
                "audio_vae": ("VAE", {"tooltip": "Optional Audio VAE for decoding the audio latent.\nОпциональный Audio VAE для декодирования аудио-латента."}),
                "options": (OPTIONS_TYPE, {"tooltip": "Overrides from options_single/options_dual/options_lora. Dict = single override; LIST = whole series in ONE queue run (images batched). LoRA keys patch a model clone per series item.\nПереопределения от options_single/options_dual/options_lora. Словарь = одно переопределение; СПИСОК = вся серия за один прогон (картинки батчем). LoRA-ключи патчат клон модели на каждый элемент серии."}),
                "decode_mode": (["tiled", "standard", "off"], {
                    "default": "standard",
                    "tooltip": (
                        "How the built-in VAE decodes latents: tiled = VAE Decode (Tiled) logic (low VRAM, big "
                        "images/video), standard = plain decode (faster, more VRAM), off = no decode.\n"
                        "---\n"
                        "Как встроенный VAE декодирует латенты: tiled = логика VAE Decode (Tiled) (мало VRAM, "
                        "большие картинки/видео), standard = обычный декод (быстрее, больше VRAM), off = без декода."
                    ),
                }),
                "tile_size": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 32,
                    "tooltip": (
                        "Tiled decode: tile size in pixels (like native VAE Decode (Tiled)).\n"
                        "---\n"
                        "Плиточный декод: размер плитки в пикселях (как в нативной VAE Decode (Tiled))."
                    ),
                }),
                "overlap": ("INT", {
                    "default": 64, "min": 0, "max": 4096, "step": 32,
                    "tooltip": (
                        "Tiled decode: overlap between tiles in pixels; auto-clamped to tile_size / 4.\n"
                        "---\n"
                        "Плиточный декод: перекрытие плиток в пикселях; авто-кламп до tile_size / 4."
                    ),
                }),
                "temporal_size": ("INT", {
                    "default": 64, "min": 8, "max": 4096, "step": 4,
                    "tooltip": (
                        "Only used for video VAEs: amount of frames to decode at a time.\n"
                        "---\n"
                        "Только для видео-VAE: сколько кадров декодировать за один проход."
                    ),
                }),
                "temporal_overlap": ("INT", {
                    "default": 8, "min": 4, "max": 4096, "step": 4,
                    "tooltip": (
                        "Only used for video VAEs: amount of frames to overlap; auto-clamped.\n"
                        "---\n"
                        "Только для видео-VAE: перекрытие в кадрах; авто-кламп."
                    ),
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO", OPTIONS_TYPE)
    RETURN_NAMES = ("latent", "images", "audio", "options")
    OUTPUT_TOOLTIPS = (
        "The denoised latent (last item of the series).\nДеноизированный латент (последний элемент серии).",
        "Decoded images/video; batched when a series runs.\nДекодированные изображения/видео; батчем при серии.",
        "The decoded audio (last item of the series).\nДекодированное аудио (последний элемент серии).",
        "Actual sampling parameters of the last run.\nФактические параметры последнего прогона.",
    )
    FUNCTION = "sample"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🧩 AGSoft KSampler Plus.\n"
        "Universal KSampler with built-in VAE Decode (Tiled) for images: tile params mirror the native "
        "VAE Decode (Tiled); options as dict = single override, options as LIST = whole series in ONE "
        "queue run (images batched); LoRA keys patch a model clone per series item; sampling results are "
        "cached per node, so changing decode widgets re-decodes only.\n"
        "---\n"
        "🧩 AGSoft KSampler Plus.\n"
        "Универсальный KSampler со встроенным VAE Decode (Tiled) для изображений: параметры плитки как у "
        "нативной VAE Decode (Tiled); options словарём = одно переопределение, options СПИСКОМ = вся серия "
        "за ОДИН прогон (картинки батчем); LoRA-ключи патчат клон модели на каждый элемент серии; результат "
        "семплирования кэшируется по ноде, поэтому смена виджетов декода только передекодирует."
    )

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

    @staticmethod
    def _send_step(step, total, rows):
        """Подсветка строк серии в UI.
        Series row highlight in the UI."""
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("agsoft_series_step", {
                "step": int(step), "total": int(total), "rows": list(rows or [])
            })
        except Exception:
            pass

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
                      seed, steps, cfg, sampler_name, scheduler, denoise, options):
        refs = (model, positive, negative, latent_image, sigmas)
        opts_repr = json.dumps(options, sort_keys=True, default=str) if options is not None else ""
        params = (int(seed), int(steps), float(cfg), str(sampler_name), str(scheduler),
                  float(denoise), self._sig_tuple(sigmas), opts_repr)
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

    def sample(self, model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
               negative=None, sigmas=None, denoise=1.0, vae=None, audio_vae=None, options=None,
               decode_mode="tiled", tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
               unique_id=None):
        dec_mode = str(decode_mode)
        uid = str(unique_id) if unique_id is not None else None
        key = self._sampling_key(model, positive, negative, latent_image, sigmas,
                                 seed, steps, cfg, sampler_name, scheduler, denoise, options)
        entry = _SAMPLE_CACHE.get(uid) if uid is not None else None
        hit = entry is not None and self._key_match(entry, key)
        if hit:
            items = entry["items"]
            self._log(f"CACHE HIT: sampling skipped ({len(items)} item(s)), VAE decode re-run only")
        else:
            items = self._run_series(model, positive, latent_image, seed, steps, cfg,
                                     sampler_name, scheduler, negative=negative, sigmas=sigmas,
                                     denoise=denoise, options=options)
            if uid is not None:
                _SAMPLE_CACHE[uid] = {"refs": key[0], "params": key[1], "items": items}
                while len(_SAMPLE_CACHE) > _CACHE_LIMIT:
                    _SAMPLE_CACHE.pop(next(iter(_SAMPLE_CACHE)))
                self._log(f"sampling executed ({len(items)} item(s)), result cached for re-decode")
        # Фаза декода: всегда с текущими виджетами декода.
        # Decode phase: always with the current decode widgets.
        imgs, auds = [], []
        total = len(items)
        for idx, it in enumerate(items):
            tag = f"[{idx + 1}/{total}]" if total > 1 else "[single]"
            self._send_step(idx, total, it["rows"])
            image = None
            if vae is None or it["video"] is None or dec_mode == "off":
                # Декод выключен или нечем декодировать: явное сообщение вместо тишины.
                # Decode disabled or nothing to decode: explicit notice, not silence.
                if vae is None:
                    reason = "vae not connected"
                elif it["video"] is None:
                    reason = "no video latent"
                else:
                    reason = "mode=off"
                self._log(f"{tag} VAE decode skipped ({reason}): images output = empty "
                          f"{PLACEHOLDER_SIZE}x{PLACEHOLDER_SIZE} placeholder")
            else:
                image = self._decode_and_log(vae, it["video"], dec_mode,
                                             tile_size, overlap, temporal_size, temporal_overlap, tag)
            if image is None:
                image = torch.zeros((1, PLACEHOLDER_SIZE, PLACEHOLDER_SIZE, 3), dtype=torch.float32)
            audio = None
            if audio_vae is not None and it["audio_lat"] is not None:
                audio = self._decode_audio(audio_vae, it["audio_lat"])
            if audio is None:
                sr = audio_vae.audio_sample_rate if hasattr(audio_vae, "audio_sample_rate") else None
                audio = {
                    "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
                    "sample_rate": sr or DEFAULT_AUDIO_SAMPLE_RATE,
                }
            imgs.append(image)
            auds.append(audio)
        if imgs:
            images = torch.cat(imgs, dim=0)
        else:
            images = torch.zeros((1, PLACEHOLDER_SIZE, PLACEHOLDER_SIZE, 3), dtype=torch.float32)
        return (items[-1]["latent"], images, auds[-1], items[-1]["meta"])

    # ------------------------------------------------------------------
    # Фаза семплирования: серия или одиночный прогон; элементы для кэша.
    # Sampling phase: series or single run; items for the cache.
    # ------------------------------------------------------------------
    def _run_series(self, model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
                    negative=None, sigmas=None, denoise=1.0, options=None):
        if isinstance(options, (list, tuple)) and options:
            ov_list = [dict(raw_ov or {}) for raw_ov in options]
        else:
            ov_list = [dict(options or {})]
        total = len(ov_list)
        if total > 1:
            self._log(f"SERIES start: {total} items")
        items = []
        for i, ov in enumerate(ov_list):
            rows = ov.pop("_rows", None)
            lat, video, audio_lat, meta = self._run_sampling(
                model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
                negative=negative, sigmas=sigmas, denoise=denoise, overrides=ov,
                series_idx=i if total > 1 else None, series_total=total if total > 1 else None)
            items.append({"latent": lat, "video": video, "audio_lat": audio_lat,
                          "rows": rows, "meta": meta})
        if total > 1:
            self._log(f"SERIES done: {total} items")
        return items

    # ------------------------------------------------------------------
    # Один прогон семплирования с override-словарём (без декода).
    # One sampling pass with an override dict (no decode here).
    # ------------------------------------------------------------------
    def _run_sampling(self, model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
                      negative=None, sigmas=None, denoise=1.0, overrides=None,
                      series_idx=None, series_total=None):
        tag = f"[{series_idx + 1}/{series_total}]" if series_total else "[single]"
        if overrides:
            if "seed" in overrides:
                seed = int(overrides["seed"])
            if "steps" in overrides:
                steps = int(overrides["steps"])
            if "cfg" in overrides:
                cfg = float(overrides["cfg"])
            if "denoise" in overrides:
                denoise = float(overrides["denoise"])
            if overrides.get("sampler"):
                sampler_name = str(overrides["sampler"])
            if overrides.get("scheduler"):
                scheduler = str(overrides["scheduler"])
            if overrides.get("sigmas") is not None:
                sigmas = torch.tensor([float(x) for x in overrides["sigmas"]], dtype=torch.float32)
        # LoRA-оверрайд из options (series-режим options_lora): патчим клон модели.
        # LoRA override from options (options_lora series mode): patch a model clone.
        lora_name_ov = overrides.get("lora_name") if overrides else None
        sm_ov = overrides.get("strength_model") if overrides else None
        if lora_name_ov and sm_ov is not None and str(lora_name_ov) != "__none__":
            path = folder_paths.get_full_path("loras", str(lora_name_ov))
            if path:
                lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
                key_map = comfy.lora.model_lora_keys_unet(model.model, {})
                loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map)
                patched = model.clone()
                patched.add_patches(loaded, float(sm_ov))
                model = patched
                self._log(f"{tag} LoRA patched: {lora_name_ov} x {sm_ov}")
        # Если негатив не подан, создаём пустой конд.
        # If negative is not provided, create empty conditioning.
        if negative is None:
            negative = self._create_empty_cond(positive)
        # Сэмплирование: стандартный путь или кастомные сигмы.
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
            scheduler_used = scheduler
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
        self._log(f"{tag} sampling done in {dt_sample:.2f}s | steps={steps_used} cfg={cfg:.2f} "
                  f"sampler={sampler_name} scheduler={scheduler_used} denoise={denoise_used:.2f}")
        video_latent, audio_latent = self._split_av_latent(latent_dict)
        out_options = {
            "seed": int(seed),
            "steps": int(steps_used),
            "cfg": float(cfg),
            "sampler": sampler_name,
            "scheduler": scheduler_used,
            "denoise": float(denoise_used),
        }
        return (latent_dict, video_latent, audio_latent, out_options)

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
# Регистрация ноды.
# Node registration.
#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AGSoft_KSampler_Plus": AGSoft_KSampler_Plus,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_KSampler_Plus": "🧩AGSoft KSampler Plus",
}