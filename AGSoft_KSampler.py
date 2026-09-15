# ==============================================================================
# AGSoft_KSampler.py
# ==============================================================================
# Ноды: 🧩 AGSoft KSampler, 🛠 AGSoft KSampler Options,
#        🛠 AGSoft KSampler options_single, 🛠 AGSoft KSampler options_dual,
#        🛠 AGSoft KSampler options_lora, 🖼️ AGSoft Contact Sheet
# Описание / Description:
# Файл универсального сэмплера и пакетного тестирования: шесть нод в одном
# месте.
# - 🧩 AGSoft KSampler — универсальный KSampler (SD/SDXL/Flux/Krea2/LTX/H3).
#   options=dict -> одно переопределение; options=list -> вся серия за ОДИН
#   Queue (картинки батчем). LoRA-ключи (lora_name, strength_model) патчат
#   клон модели на каждый элемент серии. Служебный ключ "_rows" несёт номера
#   строк для подсветки (KSampler шлёт agsoft_series_step).
# - 🛠 AGSoft KSampler Options — распаковка options в родные типы, строки
#   "name = value" и общий список без фигурных скобок.
# - 🛠 options_single / options_dual — серии тестовых значений строками ноды.
#   output_mode: series / current; pair_mode в dual: pairs / grid.
# - 🛠 options_lora — серия сил LoRA (model / clip) строками ноды за ОДИН
#   прогон: в series модель проходит без патча, KSampler патчит клон на каждый
#   элемент; clip опционален: если не подключён — ключи clip не создаются и
#   выход clip не передаётся вообще; значения clip отдаются в series в options
#   и в подписи current_str, фактически патчатся в current (кодирование
#   кондинга происходит один раз выше по графу).
# - 🖼️ AGSoft Contact Sheet — вжигает подписи (по строке на кадр) и склеивает
#   батч в сетку NxM одним изображением.
# Universal KSampler (dict = override, list = series in one queue), batch-test
# nodes (single / dual / lora) and a labeled contact-sheet node.
#
# Возможности / Features:
# ⚡ KSampler: options словарём = одно переопределение; options списком = вся
#   серия за ОДИН Queue (картинки батчем); LoRA-ключи патчат клон модели.
#   KSampler: options as dict = single override; options as list = whole
#   series in ONE queue run (images batched); LoRA keys patch a model clone.
# ⚡ MiniMax H3: AV-латент NestedTensor((video, audio)) распаковывается через
#   .tensors; channel-last аудио приводится к стандарту ComfyUI [B,C,N].
#   MiniMax H3: AV latent NestedTensor((video, audio)) unpacked via .tensors;
#   channel-last audio converted to ComfyUI standard [B,C,N].
# ⚡ options_single/dual: output_mode series/current; pair_mode pairs/grid в
#   dual; серверный счётчик итерации в режиме current.
#   options_single/dual: output_mode series/current; pair_mode pairs/grid in
#   dual; server-side iteration counter in current mode.
# ⚡ options_lora: горизонтальные степеры сил (◄/► клик ±0.05, Shift ±0.01),
#   дефолт сил = 1; 30 строк; clip опционален полностью.
#   options_lora: horizontal strength steppers (◄/► click ±0.05, Shift ±0.01),
#   default strengths = 1; 30 rows; clip fully optional.
# ⚡ Подсветка строк серии: сервер шлёт agsoft_series_step, JS подсвечивает
#   активные строки options-ноды.
#   Series row highlighting: the server sends agsoft_series_step, JS highlights
#   the active rows of the options node.
# ⚡ Contact Sheet: подписи из current_str (строка на кадр), палитра из 24
#   шаблонов, раздельная прозрачность текста и плашки, скруглённые углы.
#   Contact Sheet: labels from current_str (one line per frame), 24 preset
#   colors, separate text and box opacity, rounded box corners.
#
# Автор / Author: AGSoft
# Дата / Date: 15.09.2026
# ==============================================================================

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
import math
import time
import os
from typing import Dict, Any


DEFAULT_AUDIO_SAMPLE_RATE = 32000
OPTIONS_TYPE = "KSAMPLER_OPTIONS"
MAX_TEST_ROWS = 30
PARAM_OPTIONS = ["seed", "steps", "cfg", "sampler", "scheduler", "sigmas"]

# Серверное состояние итерации (режим current).
# Server-side iteration state (current mode).
_ITER_STATE = {}


#------------------------------------------------------------------------------
# Хелперы тестовых значений: парсинг, приведение типов, форматирование.
# Helpers for test values: parsing, type coercion and formatting.
#------------------------------------------------------------------------------
def _parse_sigmas_raw(raw):
    """list/tuple/Tensor/'1.0, 0.96, 0.857143, 0.0' -> [float].
    list/tuple/Tensor/'1.0, 0.96, 0.857143, 0.0' -> [float]."""
    if isinstance(raw, torch.Tensor):
        vals = [float(v) for v in raw.detach().cpu().flatten().tolist()]
    elif isinstance(raw, (list, tuple)):
        vals = [float(v) for v in raw]
    else:
        text = str(raw).replace(";", ",").replace("\n", ",")
        vals = [float(p) for p in (x.strip() for x in text.split(",")) if p]
    if len(vals) < 2:
        raise ValueError("sigmas test value needs at least 2 numbers, e.g. '1.0, 0.96, 0.857143, 0.0'")
    return vals


def _coerce_test_value(param, raw):
    if param in ("seed", "steps"):
        return int(round(float(raw)))
    if param == "cfg":
        return float(raw)
    if param == "sampler":
        name = str(raw)
        if name not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f"Unknown sampler '{name}'")
        return name
    if param == "scheduler":
        name = str(raw)
        if name not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(f"Unknown scheduler '{name}'")
        return name
    if param == "sigmas":
        return _parse_sigmas_raw(raw)
    raise ValueError(f"Unknown param '{param}'")


def _fmt_test_value(v):
    if isinstance(v, list):
        return ", ".join(f"{x:g}" for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _lora_short_name(name):
    """Только имя файла лоры: без пути и без расширения .safetensors.
    Only the LoRA file name: no path, no .safetensors extension."""
    base = os.path.basename(str(name).replace("\\", "/"))
    return os.path.splitext(base)[0]


def _control_widgets(required, pair_mode=False):
    """Общие виджеты управления серией: output_mode, active_rows, index, mode,
    loop и шаблоны списков sampler/scheduler для UI.
    Common series control widgets: output_mode, active_rows, index, mode,
    loop plus sampler/scheduler list templates for the UI."""
    required["output_mode"] = (["series", "current"], {
        "default": "series",
        "tooltip": (
            "series: whole series in ONE queue run (KSampler loops, images batched). "
            "current: one item per queue run (server counter, use Auto-Queue).\n"
            "---\n"
            "series: вся серия за ОДИН прогон (KSampler крутит сам, картинки батчем). "
            "current: один элемент за прогон (серверный счётчик, нужен Auto-Queue)."
        )
    })
    if pair_mode:
        required["pair_mode"] = (["pairs", "grid"], {
            "default": "pairs",
            "tooltip": (
                "pairs: each row is a hand-made pair (a+b). grid: column A values × column B values "
                "(Cartesian product; leave cells empty to shorten a column).\n"
                "---\n"
                "pairs: каждая строка — ручная связка (a+b). grid: колонка A × колонка B "
                "(декартово произведение; пустые ячейки укорачивают колонку)."
            )
        })
    required["active_rows"] = ("INT", {
        "default": 2, "min": 1, "max": MAX_TEST_ROWS, "step": 1,
        "tooltip": (
            "Internal: row count, controlled by + Add / – Remove (widget hidden).\n"
            "---\n"
            "Служебный: число строк, управляется + Add / – Remove (виджет скрыт)."
        )
    })
    required["index"] = ("INT", {
        "default": 1, "min": 1, "max": MAX_TEST_ROWS, "step": 1,
        "tooltip": (
            "current mode: start/manual position (1-based). Ignored in series.\n"
            "---\n"
            "current: стартовая/ручная позиция (с 1). В series игнорируется."
        )
    })
    required["mode"] = (["increment", "fixed"], {
        "tooltip": (
            "current mode only: increment advances per queue run; fixed uses index.\n"
            "---\n"
            "Только current: increment двигает элемент каждый прогон; fixed использует index."
        )
    })
    required["loop"] = ("BOOLEAN", {
        "default": True,
        "tooltip": (
            "current mode: wrap to first item at series end.\n"
            "---\n"
            "current: переходить к первому элементу в конце серии."
        )
    })
    # Шаблоны списков для JS-dropdown.
    # List templates for the JS dropdowns.
    required["_sampler_list"] = (comfy.samplers.KSampler.SAMPLERS, {
        "tooltip": (
            "Internal template: sampler list for UI dropdowns (hidden).\n"
            "---\n"
            "Служебный шаблон: список сэмплеров для dropdown в UI (скрыт)."
        )
    })
    required["_scheduler_list"] = (comfy.samplers.KSampler.SCHEDULERS, {
        "tooltip": (
            "Internal template: scheduler list for UI dropdowns (hidden).\n"
            "---\n"
            "Служебный шаблон: список планировщиков для dropdown в UI (скрыт)."
        )
    })


def _series_signature(tag, count, kwargs):
    """Стабильный ключ серии: меняется только при правке строк.
    Stable series key: changes only when rows are edited."""
    parts = [tag, count, str(kwargs.get("pair_mode", "pairs"))]
    for i in range(1, count + 1):
        if tag == "dual":
            parts.append((bool(kwargs.get(f"enabled_{i}", True)),
                          str(kwargs.get(f"param_a_{i}", "")), str(kwargs.get(f"value_a_{i}", "")),
                          str(kwargs.get(f"param_b_{i}", "")), str(kwargs.get(f"value_b_{i}", ""))))
        else:
            parts.append((bool(kwargs.get(f"enabled_{i}", True)),
                          str(kwargs.get(f"param_{i}", "")), str(kwargs.get(f"value_{i}", ""))))
    return json.dumps(parts, sort_keys=True, default=str)


def _next_position(key, series_len, mode, loop, index):
    """Серверная позиция в серии (режим current).
    Server-side position in the series (current mode)."""
    if mode != "increment":
        return max(1, min(int(index), series_len)) - 1
    prev = _ITER_STATE.get(key)
    if prev is None:
        pos = max(1, min(int(index), series_len)) - 1
    else:
        pos = prev + 1
        if pos >= series_len:
            pos = 0 if loop else series_len - 1
    _ITER_STATE[key] = pos
    return pos


#==============================================================================
# 🧩 AGSoft KSampler
#==============================================================================
class AGSoft_KSampler:
    WEB_DIRECTORY = "./web"

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
                "vae": ("VAE", {"tooltip": "Optional VAE for decoding into images/video.\nОпциональный VAE для декодирования в изображения/видео."}),
                "audio_vae": ("VAE", {"tooltip": "Optional Audio VAE for decoding the audio latent.\nОпциональный Audio VAE для декодирования аудио-латента."}),
                "options": (OPTIONS_TYPE, {"tooltip": "Overrides from options_single/options_dual/options_lora. Dict = single override; LIST = whole series in ONE queue run (images batched). LoRA keys (lora_name, strength_model, strength_clip) patch a model clone per series item.\nПереопределения от options_single/options_dual/options_lora. Словарь = одно переопределение; СПИСОК = вся серия за один прогон (картинки батчем). LoRA-ключи (lora_name, strength_model, strength_clip) патчат клон модели на каждый элемент серии."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO", OPTIONS_TYPE)
    RETURN_NAMES = ("latent", "images", "audio", "options")
    OUTPUT_TOOLTIPS = (
        "The denoised latent (last item of the series).\nДеноизированный латент (последний элемент серии).",
        "Decoded images/video; batched when a series runs.\nДекодированные изображения/видео; батчем при серии.",
        "The decoded audio (last item of the series).\nДекодированное аудио (последний элемент серии).",
        "Actual sampling parameters of the last run.\nФактические параметры последнего прогона."
    )
    FUNCTION = "sample"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🧩 AGSoft KSampler.\n"
        "Universal KSampler for all ComfyUI models (SD 1.5, SDXL, Flux, Krea 2, LTX, MiniMax H3).\n"
        "options as dict = single override; options as LIST = whole series in ONE queue run "
        "(images batched); LoRA override keys patch a model clone per series item.\n"
        "---\n"
        "🧩 AGSoft KSampler.\n"
        "Универсальный KSampler для всех моделей ComfyUI (SD 1.5, SDXL, Flux, Krea 2, LTX, MiniMax H3).\n"
        "options словарём = одно переопределение; options СПИСКОМ = вся серия за ОДИН прогон "
        "(картинки батчем); LoRA-ключи патчат клон модели на каждый элемент серии."
    )

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

    def sample(self, model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
               negative=None, sigmas=None, denoise=1.0, vae=None, audio_vae=None, options=None):

        # СЕРИЯ: список override-словарей крутим внутри одного прогона.
        # SERIES: loop the override list inside a single queue run.
        if isinstance(options, (list, tuple)) and options:
            lats, imgs, auds, last_opts = [], [], [], None
            total = len(options)
            for i, raw_ov in enumerate(options):
                ov = dict(raw_ov or {})
                rows = ov.pop("_rows", None)
                self._send_step(i, total, rows)
                lat, img, aud, opts = self._sample_once(
                    model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
                    negative=negative, sigmas=sigmas, denoise=denoise,
                    vae=vae, audio_vae=audio_vae, overrides=ov
                )
                lats.append(lat)
                imgs.append(img)
                auds.append(aud)
                last_opts = opts
            images = torch.cat(imgs, dim=0) if imgs else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            audio = auds[-1] if auds else None
            return (lats[-1], images, audio, last_opts)

        ov = dict(options or {})
        ov.pop("_rows", None)
        return self._sample_once(
            model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
            negative=negative, sigmas=sigmas, denoise=denoise,
            vae=vae, audio_vae=audio_vae, overrides=ov
        )

    # ------------------------------------------------------------------
    # Один прогон семплирования с override-словарём.
    # One sampling pass with an override dict.
    # ------------------------------------------------------------------
    def _sample_once(self, model, positive, latent_image, seed, steps, cfg, sampler_name, scheduler,
                     negative=None, sigmas=None, denoise=1.0, vae=None, audio_vae=None, overrides=None):
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
        # strength_clip несётся в options для подписей/учёта: кондинг кодируется
        # один раз выше по графу, поэтому внутри цикла применяется только model.
        # LoRA override from options (options_lora series mode): patch a model
        # clone. strength_clip rides in options for labels/records: conditioning
        # is encoded once upstream, so only model strength applies inside the loop.
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

        # Если негатив не подан, создаём пустой конд.
        # If negative is not provided, create empty conditioning.
        if negative is None:
            negative = self._create_empty_cond(positive)

        # Сэмплирование: стандартный путь или кастомные сигмы.
        # Sampling: standard path or custom sigmas.
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
                model_options=getattr(model, "model_options", {}),
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

        video_latent, audio_latent = self._split_av_latent(latent_dict)

        images = None
        if vae is not None and video_latent is not None:
            try:
                images = vae.decode(video_latent)
                if images.dim() == 5:
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            except Exception as e:
                print(f"[AGSoft] VAE decode error: {e}")
                images = None

        audio = None
        if audio_vae is not None and audio_latent is not None:
            audio = self._decode_audio(audio_vae, audio_latent)

        # Фолбэки, чтобы ComfyUI не падал на пустых выходах.
        # Fallbacks so ComfyUI doesn't crash on empty outputs.
        if images is None:
            images = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        if audio is None:
            audio = {
                "waveform": torch.zeros((1, 1, 1), dtype=torch.float32),
                "sample_rate": getattr(audio_vae, "audio_sample_rate", None) or DEFAULT_AUDIO_SAMPLE_RATE,
            }

        out_options = {
            "seed": int(seed),
            "steps": int(steps_used),
            "cfg": float(cfg),
            "sampler": sampler_name,
            "scheduler": scheduler_used,
            "denoise": float(denoise_used),
        }
        return (latent_dict, images, audio, out_options)

    # ------------------------------------------------------------------
    # H3: {"samples": NestedTensor((video, audio))} -> (video, audio).
    # H3: {"samples": NestedTensor((video, audio))} -> (video, audio).
    # ------------------------------------------------------------------
    def _split_av_latent(self, latent_dict):
        samples = latent_dict["samples"] if isinstance(latent_dict, dict) else latent_dict

        tensors = getattr(samples, "tensors", None)
        if tensors is None and getattr(samples, "is_nested", False):
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
            print(f"[AGSoft] Audio decode error: {e}")
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
            sample_rate = getattr(audio_vae, "audio_sample_rate", None)
        if sample_rate is None:
            sample_rate = getattr(audio_vae, "sample_rate", None)
        if sample_rate is None:
            sample_rate = DEFAULT_AUDIO_SAMPLE_RATE

        if getattr(waveform, "is_nested", False) or hasattr(waveform, "tensors"):
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
            print(f"[AGSoft] Unexpected audio waveform dims: {waveform.dim()}")
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


#==============================================================================
# 🛠 AGSoft KSampler Options (unpack / распаковка)
#==============================================================================
class AGSoft_KSampler_Options:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "options": (OPTIONS_TYPE, {"tooltip": "Options output of AGSoft KSampler.\nВыход options ноды AGSoft KSampler."}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "FLOAT", "FLOAT", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "seed", "steps", "cfg", "denoise", "sampler", "scheduler",
        "seed_str", "steps_str", "cfg_str", "denoise_str", "sampler_str", "scheduler_str",
        "params_list",
    )
    FUNCTION = "unpack"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🛠 AGSoft KSampler Options.\n"
        "Unpacks AGSoft KSampler options into native-typed values (INT/FLOAT/STRING), "
        "'name = value' strings and a combined parameter list without curly braces.\n"
        "---\n"
        "🛠 AGSoft KSampler Options.\n"
        "Распаковывает options ноды AGSoft KSampler в значения родных типов (INT/FLOAT/STRING), "
        "строки 'name = value' и общий список параметров без фигурных скобок."
    )

    def unpack(self, options):
        seed = int(options.get("seed", 0))
        steps = int(options.get("steps", 0))
        cfg = float(options.get("cfg", 0.0))
        denoise = float(options.get("denoise", 1.0))
        sampler = str(options.get("sampler", ""))
        scheduler = str(options.get("scheduler", ""))

        seed_str = f"seed = {seed}"
        steps_str = f"steps = {steps}"
        cfg_str = f"cfg = {cfg}"
        denoise_str = f"denoise = {denoise}"
        sampler_str = f"sampler = {sampler}"
        scheduler_str = f"scheduler = {scheduler}"

        params_list = "\n".join([seed_str, steps_str, cfg_str, sampler_str, scheduler_str, denoise_str])

        return (
            seed, steps, cfg, denoise, sampler, scheduler,
            seed_str, steps_str, cfg_str, denoise_str, sampler_str, scheduler_str,
            params_list,
        )


#==============================================================================
# 🛠 AGSoft KSampler options_single (series of single test values)
#==============================================================================
class AGSoftKSamplerOptionsSingle:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        required = {}
        _control_widgets(required)
        for i in range(1, MAX_TEST_ROWS + 1):
            required[f"enabled_{i}"] = ("BOOLEAN", {"default": True, "tooltip": f"Row {i}: enable/disable this test value.\nСтрока {i}: включить/выключить это тестовое значение."})
            required[f"param_{i}"] = (PARAM_OPTIONS, {"default": "steps", "tooltip": f"Row {i}: which sampling parameter this value tests.\nСтрока {i}: какой параметр семплинга тестирует значение."})
            required[f"value_{i}"] = ("STRING", {"default": "", "tooltip": f"Row {i}: test value (number, sampler/scheduler name or sigmas list). Edited by the UI.\nСтрока {i}: тестовое значение (число, имя sampler/scheduler или список sigmas). Редактируется через UI."})
        return {"required": required}

    RETURN_TYPES = (OPTIONS_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("options", "current_str", "status")
    OUTPUT_TOOLTIPS = (
        "Override dict (current) or list of override dicts (series) for AGSoft KSampler.\nСловарь переопределений (current) или список словарей (series) для AGSoft KSampler.",
        "Current test value(s) as 'param = value' (one line per series item).\nТекущее значение(я) в виде 'param = value' (по строке на элемент серии).",
        "Progress status.\nСтатус прогресса."
    )
    FUNCTION = "generate"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🛠 AGSoft KSampler options_single.\n"
        "Series of single test values (seed/steps/cfg/sampler/scheduler/sigmas) as node rows.\n"
        "output_mode=series: the whole series runs inside ONE queue run of AGSoft KSampler.\n"
        "output_mode=current: one series item per queue run (server-side counter, Auto-Queue).\n"
        "---\n"
        "🛠 AGSoft KSampler options_single.\n"
        "Серия одиночных тестовых значений (seed/steps/cfg/sampler/scheduler/sigmas) строками ноды.\n"
        "output_mode=series: вся серия крутится внутри ОДНОГО прогона AGSoft KSampler.\n"
        "output_mode=current: один элемент серии за прогон очереди (серверный счётчик, Auto-Queue)."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # current+increment: форсируем пересчёт каждый прогон / force re-execution each run
        if str(kwargs.get("output_mode", "series")) == "current" and str(kwargs.get("mode", "fixed")) == "increment":
            return time.time_ns()
        return 0.0

    def generate(self, active_rows, index, mode, loop, output_mode, **kwargs):
        count = max(1, min(int(active_rows), MAX_TEST_ROWS))
        items = []  # (rows, overrides, label) / (строки, переопределения, подпись)
        for i in range(1, count + 1):
            if not kwargs.get(f"enabled_{i}", True):
                continue
            param = str(kwargs.get(f"param_{i}", "steps"))
            raw = str(kwargs.get(f"value_{i}", "") or "").strip()
            if not raw:
                continue
            val = _coerce_test_value(param, raw)
            items.append(([i], {param: val}, f"{param} = {_fmt_test_value(val)}"))

        if not items:
            return ({}, "<no test values>", "series: 0/0 (add rows)")

        # SERIES: вся серия за один Queue / whole series in one queue run
        if str(output_mode) == "series":
            options = [dict(ov, _rows=rows) for rows, ov, _lab in items]
            current_str = "\n".join(lab for _r, _o, lab in items)
            status = f"series: {len(items)} items / one queue"
            return (options, current_str, status)

        # CURRENT: один элемент за прогон / one item per queue run
        key = _series_signature("single", count, kwargs)
        pos = _next_position(key, len(items), str(mode), bool(loop), index)
        rows, ov, label = items[pos]
        options = dict(ov, _rows=rows)
        status = f"{label.split(' =')[0]}: {pos + 1}/{len(items)}"
        return (options, label, status)


#==============================================================================
# 🛠 AGSoft KSampler options_dual (pairs or grid of two-value combos)
#==============================================================================
class AGSoftKSamplerOptionsDual:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        required = {}
        _control_widgets(required, pair_mode=True)
        for i in range(1, MAX_TEST_ROWS + 1):
            required[f"enabled_{i}"] = ("BOOLEAN", {"default": True, "tooltip": f"Row {i}: enable/disable this pair.\nСтрока {i}: включить/выключить эту связку."})
            required[f"param_a_{i}"] = (PARAM_OPTIONS, {"default": "sampler", "tooltip": f"Row {i}: first parameter (pairs mode) or column A parameter (grid mode).\nСтрока {i}: первый параметр (pairs) или параметр колонки A (grid)."})
            required[f"value_a_{i}"] = ("STRING", {"default": "", "tooltip": f"Row {i}: first value / column A value.\nСтрока {i}: первое значение / значение колонки A."})
            required[f"param_b_{i}"] = (PARAM_OPTIONS, {"default": "scheduler", "tooltip": f"Row {i}: second parameter (pairs mode) or column B parameter (grid mode).\nСтрока {i}: второй параметр (pairs) или параметр колонки B (grid)."})
            required[f"value_b_{i}"] = ("STRING", {"default": "", "tooltip": f"Row {i}: second value / column B value.\nСтрока {i}: второе значение / значение колонки B."})
        return {"required": required}

    RETURN_TYPES = (OPTIONS_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("options", "current_str", "status")
    OUTPUT_TOOLTIPS = (
        "Override dict (current) or list of override dicts (series) for AGSoft KSampler.\nСловарь переопределений (current) или список словарей (series) для AGSoft KSampler.",
        "Current pair(s) as 'a = x, b = y' (one line per series item).\nТекущая связка(и) в виде 'a = x, b = y' (по строке на элемент серии).",
        "Progress status.\nСтатус прогресса."
    )
    FUNCTION = "generate"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🛠 AGSoft KSampler options_dual.\n"
        "Series of two-value combos: hand-made pairs (pair_mode=pairs) or Cartesian grid "
        "column A × column B (pair_mode=grid; empty cells shorten a column).\n"
        "output_mode=series: the whole series runs inside ONE queue run of AGSoft KSampler.\n"
        "---\n"
        "🛠 AGSoft KSampler options_dual.\n"
        "Серия связок из двух значений: ручные пары (pair_mode=pairs) или декартова сетка "
        "колонка A × колонка B (pair_mode=grid; пустые ячейки укорачивают колонку).\n"
        "output_mode=series: вся серия крутится внутри ОДНОГО прогона AGSoft KSampler."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # current+increment: форсируем пересчёт каждый прогон / force re-execution each run
        if str(kwargs.get("output_mode", "series")) == "current" and str(kwargs.get("mode", "fixed")) == "increment":
            return time.time_ns()
        return 0.0

    def generate(self, active_rows, index, mode, loop, output_mode, pair_mode="pairs", **kwargs):
        count = max(1, min(int(active_rows), MAX_TEST_ROWS))
        grid = str(pair_mode) == "grid"
        items = []  # (rows, overrides, label) / (строки, переопределения, подпись)

        if not grid:
            # PAIRS: связка руками в каждой строке.
            # PAIRS: hand-made pair per row.
            for i in range(1, count + 1):
                if not kwargs.get(f"enabled_{i}", True):
                    continue
                pa = str(kwargs.get(f"param_a_{i}", "sampler"))
                pb = str(kwargs.get(f"param_b_{i}", "scheduler"))
                ra = str(kwargs.get(f"value_a_{i}", "") or "").strip()
                rb = str(kwargs.get(f"value_b_{i}", "") or "").strip()
                if not ra or not rb:
                    continue
                if pa == pb:
                    raise ValueError(f"options_dual row {i}: param_a and param_b must be different.")
                va = _coerce_test_value(pa, ra)
                vb = _coerce_test_value(pb, rb)
                items.append(([i], {pa: va, pb: vb},
                              f"{pa} = {_fmt_test_value(va)}, {pb} = {_fmt_test_value(vb)}"))
            tag = "pairs"
        else:
            # GRID: колонка A × колонка B.
            # GRID: column A × column B.
            param_a, param_b = None, None
            col_a, col_b = [], []
            for i in range(1, count + 1):
                if not kwargs.get(f"enabled_{i}", True):
                    continue
                ra = str(kwargs.get(f"value_a_{i}", "") or "").strip()
                rb = str(kwargs.get(f"value_b_{i}", "") or "").strip()
                if ra:
                    if param_a is None:
                        param_a = str(kwargs.get(f"param_a_{i}", "sampler"))
                    col_a.append((i, _coerce_test_value(param_a, ra)))
                if rb:
                    if param_b is None:
                        param_b = str(kwargs.get(f"param_b_{i}", "scheduler"))
                    col_b.append((i, _coerce_test_value(param_b, rb)))
            if not param_a or not param_b:
                return ({}, "<no grid values>", "grid: 0/0 (fill columns)")
            if param_a == param_b:
                raise ValueError("options_dual grid: column A and column B parameters must be different.")
            for ia, va in col_a:
                for ib, vb in col_b:
                    items.append(([ia, ib], {param_a: va, param_b: vb},
                                  f"{param_a} = {_fmt_test_value(va)}, {param_b} = {_fmt_test_value(vb)}"))
            tag = "grid"

        if not items:
            return ({}, "<no pairs>", "series: 0/0 (add rows)")

        # SERIES: вся серия за один Queue / whole series in one queue run
        if str(output_mode) == "series":
            options = [dict(ov, _rows=rows) for rows, ov, _lab in items]
            current_str = "\n".join(lab for _r, _o, lab in items)
            status = f"{tag}: {len(items)} items / one queue"
            return (options, current_str, status)

        # CURRENT: один элемент за прогон / one item per queue run
        key = _series_signature("dual", count, kwargs)
        pos = _next_position(key, len(items), str(mode), bool(loop), index)
        rows, ov, label = items[pos]
        options = dict(ov, _rows=rows)
        status = f"{tag}: {pos + 1}/{len(items)}"
        return (options, label, status)


#==============================================================================
# 🛠 AGSoft KSampler options_lora (LoRA strength series, one Run)
#==============================================================================
class AGSoftKSamplerOptionsLora:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        try:
            loras = folder_paths.get_filename_list("loras")
        except Exception:
            loras = []
        if not loras:
            loras = ["__none__"]
        required = {
            "model": ("MODEL", {
                "tooltip": (
                    "Model under test. In series mode it passes through UNPATCHED: the KSampler patches a clone per series item. In current mode it is patched here at the current pair.\n"
                    "---\n"
                    "Тестируемая модель. В series проходит БЕЗ патча: KSampler патчит клон на каждый элемент серии. В current патчится здесь текущей парой."
                )
            }),
            "lora_name": (loras, {
                "tooltip": (
                    "The LoRA file under test (from models/loras).\n"
                    "---\n"
                    "Тестируемый файл LoRA (из models/loras)."
                )
            }),
            "output_mode": (["series", "current"], {
                "default": "series",
                "tooltip": (
                    "series: the whole strength series runs inside ONE queue run (KSampler loops, images batched); model strength applies per item, clip strength rides in options/labels. current: one pair per queue run, model AND clip are patched here (server counter, Auto-Queue or manual index).\n"
                    "---\n"
                    "series: вся серия сил за ОДИН прогон (KSampler крутит сам, картинки батчем); сила model применяется на каждый элемент, сила clip едет в options/подписях. current: одна пара за прогон, патчатся model И clip (серверный счётчик, Auto-Queue или ручной index)."
                )
            }),
            "mode": (["increment", "fixed"], {
                "tooltip": (
                    "current mode only: increment advances one row per queue run; fixed uses index.\n"
                    "---\n"
                    "Только current: increment двигает строку каждый прогон; fixed использует index."
                )
            }),
            "index": ("INT", {
                "default": 1, "min": 1, "max": MAX_TEST_ROWS, "step": 1,
                "tooltip": (
                    "current mode: start/manual row (1-based). Ignored in series.\n"
                    "---\n"
                    "current: стартовая/ручная строка (с 1). В series игнорируется."
                )
            }),
            "loop": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "current mode: wrap to the first row at series end.\n"
                    "---\n"
                    "current: переходить к первой строке в конце серии."
                )
            }),
            "active_rows": ("INT", {
                "default": 2, "min": 1, "max": MAX_TEST_ROWS, "step": 1,
                "tooltip": (
                    "Internal: row count, controlled by + Add / – Remove (widget hidden).\n"
                    "---\n"
                    "Служебный: число строк, управляется + Add / – Remove (виджет скрыт)."
                )
            }),
        }
        for i in range(1, MAX_TEST_ROWS + 1):
            required[f"enabled_{i}"] = ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    f"Row {i}: enable/disable this strength pair.\n"
                    "---\n"
                    f"Строка {i}: включить/выключить эту пару сил."
                )
            })
            required[f"strength_m_{i}"] = ("STRING", {
                "default": "1",
                "tooltip": (
                    f"Row {i}: LoRA MODEL strength (default 1; clear to skip the row). Horizontal stepper in UI: ◄/► click ±0.05, Shift ±0.01.\n"
                    "---\n"
                    f"Строка {i}: сила LoRA для MODEL (по умолчанию 1; пусто = пропуск строки). Горизонтальный степер в UI: ◄/► клик ±0.05, Shift ±0.01."
                )
            })
            required[f"strength_c_{i}"] = ("STRING", {
                "default": "1",
                "tooltip": (
                    f"Row {i}: LoRA CLIP strength (default 1). Used in current mode; in series it rides in options/labels. Ignored entirely when clip is not connected. Horizontal stepper in UI.\n"
                    "---\n"
                    f"Строка {i}: сила LoRA для CLIP (по умолчанию 1). Применяется в current; в series едет в options/подписях. Полностью игнорируется, если clip не подключён. Горизонтальный степер в UI."
                )
            })
        return {
            "required": required,
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": (
                        "Optional CLIP. If NOT connected: no clip keys are created anywhere and the clip output passes nothing (None). If connected: patched with the clip strength in current mode; in series its strengths ride in options/labels.\n"
                        "---\n"
                        "Опциональный CLIP. Если НЕ подключён: ключи clip не создаются нигде и выход clip не передаёт ничего (None). Если подключён: в current патчится силой clip; в series его силы едут в options/подписях."
                    )
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", OPTIONS_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "options", "current_str", "status")
    OUTPUT_TOOLTIPS = (
        "Series: unpatched model (KSampler patches per item). Current: model patched at the current pair.\n"
        "Series: модель без патча (KSampler патчит сам). Current: модель с патчем текущей пары.",
        "CLIP only when connected: patched at the current pair in current mode, passthrough in series. Not connected = nothing passed (None).\n"
        "CLIP только если подключён: в current с патчем текущей пары, в series проход. Не подключён = ничего не передаётся (None).",
        "Series: list of LoRA overrides for AGSoft KSampler (one per item, strength_clip included only when clip is connected). Current: service dict.\n"
        "Series: список LoRA-переопределений для AGSoft KSampler (по элементу; strength_clip только если clip подключён). Current: служебный словарь.",
        "Current value(s) as 'lora = name, model = x[, clip = y]' (one line per series item; clip part only when clip is connected).\n"
        "Текущее значение(я) в виде 'lora = name, model = x[, clip = y]' (по строке на элемент; часть clip только если clip подключён).",
        "Progress status.\nСтатус прогресса."
    )
    FUNCTION = "apply_lora"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🛠 AGSoft KSampler options_lora.\n"
        "Series of LoRA strengths as node rows, ONE queue run: in series mode the node hands the list of "
        "overrides to AGSoft KSampler, which patches a model clone per item and batches the images; clip is "
        "fully optional — when not connected no clip keys exist and nothing is passed; horizontal strength "
        "steppers (◄/► ±0.05, Shift ±0.01), default strengths = 1, 30 rows.\n"
        "---\n"
        "🛠 AGSoft KSampler options_lora.\n"
        "Серия сил LoRA строками ноды за ОДИН прогон: в series нода передаёт список переопределений в "
        "AGSoft KSampler, который патчит клон модели на каждый элемент и складывает картинки в батч; clip "
        "полностью опционален — если не подключён, ключи clip не создаются и ничего не передаётся; "
        "горизонтальные степеры сил (◄/► ±0.05, Shift ±0.01), дефолт сил = 1, 30 строк."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # current+increment: форсируем пересчёт каждый прогон / force re-execution each run
        if str(kwargs.get("output_mode", "series")) == "current" and str(kwargs.get("mode", "fixed")) == "increment":
            return time.time_ns()
        return 0.0

    def apply_lora(self, model, lora_name, output_mode, mode, index, loop, active_rows, clip=None, **kwargs):
        count = max(1, min(int(active_rows), MAX_TEST_ROWS))
        items = []  # (row, strength_model, strength_clip) / (строка, сила model, сила clip)
        for i in range(1, count + 1):
            if not kwargs.get(f"enabled_{i}", True):
                continue
            sm_raw = str(kwargs.get(f"strength_m_{i}", "") or "").strip()
            if not sm_raw:
                continue
            sm = float(sm_raw)
            sc_raw = str(kwargs.get(f"strength_c_{i}", "") or "").strip()
            sc = float(sc_raw) if sc_raw else 1.0
            items.append((i, sm, sc))

        if not items:
            return (model, clip, {}, "<no test values>", "lora: 0/0 (add rows)")
        if not lora_name or lora_name == "__none__":
            _row, sm, sc = items[0]
            return (model, clip, {}, f"lora = none, model = {sm:g}", "lora: no file")

        # SERIES: список переопределений — KSampler патчит клон модели на каждый элемент.
        # strength_clip добавляется ТОЛЬКО если clip подключён; подписи тоже.
        # SERIES: list of overrides — the KSampler patches a model clone per item.
        # strength_clip is added ONLY when clip is connected; labels too.
        if str(output_mode) == "series":
            options = []
            labels = []
            for row, sm, sc in items:
                ov = {"lora_name": str(lora_name), "strength_model": sm, "_rows": [row]}
                if clip is not None:
                    ov["strength_clip"] = sc
                options.append(ov)
                labels.append(
                    f"lora = {_lora_short_name(lora_name)}, model = {sm:g}" + (f", clip = {sc:g}" if clip is not None else "")
                )
            current_str = "\n".join(labels)
            status = f"lora series: {len(items)} items / one queue"
            return (model, clip, options, current_str, status)

        # CURRENT/FIXED: патчим model и clip здесь текущей парой (clip только если подключён).
        # CURRENT/FIXED: patch model and clip here at the current pair (clip only when connected).
        key = json.dumps(
            ["lora", count, str(lora_name)] +
            [(bool(kwargs.get(f"enabled_{i}", True)),
              str(kwargs.get(f"strength_m_{i}", "")),
              str(kwargs.get(f"strength_c_{i}", "")))
             for i in range(1, count + 1)],
            sort_keys=True, default=str
        )
        pos = _next_position(key, len(items), str(mode), bool(loop), index)
        row, sm, sc = items[pos]

        path = folder_paths.get_full_path("loras", str(lora_name))
        lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora_sd), key_map)

        patched_model = model.clone()
        patched_model.add_patches(loaded, sm)
        patched_clip = clip
        if clip is not None:
            patched_clip = clip.clone()
            patched_clip.add_patches(loaded, sc)

        options = {"_rows": [row]}
        current_str = f"lora = {_lora_short_name(lora_name)}, model = {sm:g}" + (f", clip = {sc:g}" if clip is not None else "")
        status = f"lora: {pos + 1}/{len(items)}"
        return (patched_model, patched_clip, options, current_str, status)


#==============================================================================
# 🖼️ AGSoft Contact Sheet (labeled frames + NxM grid)
#==============================================================================
class AGSoftContactSheet:
    # Готовая палитра цветов (шаблоны).
    # Preset color palette (templates).
    PRESET_COLORS = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "lime": (0, 255, 0),
        "green": (0, 128, 0),
        "cyan": (0, 255, 255),
        "sky_blue": (0, 127, 255),
        "blue": (0, 0, 255),
        "navy": (0, 0, 128),
        "violet": (127, 0, 255),
        "magenta": (255, 0, 255),
        "pink": (255, 105, 180),
        "maroon": (128, 0, 0),
        "brown": (139, 69, 19),
        "gold": (255, 215, 0),
        "olive": (128, 128, 0),
        "teal": (0, 128, 128),
        "turquoise": (64, 224, 208),
        "light_gray": (211, 211, 211),
        "gray": (128, 128, 128),
        "dark_gray": (64, 64, 64),
        "cream": (255, 255, 224),
    }
    COLOR_NAMES = list(PRESET_COLORS.keys())

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": (
                        "Image batch (e.g. series result from AGSoft KSampler).\n"
                        "---\n"
                        "Батч изображений (например результат серии из AGSoft KSampler)."
                    )
                }),
            },
            "optional": {
                "current_str": ("STRING", {
                    "tooltip": (
                        "Connect the current_str output of options_single / options_dual / options_lora here: one line per frame (line N is burned into frame N). If not connected or empty — no labels.\n"
                        "---\n"
                        "Подключи сюда выход current_str нод options_single / options_dual / options_lora: по одной строке на кадр (строка N вжигается в кадр N). Не подключено или пусто — без подписей."
                    )
                }),
                "columns": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": (
                        "Grid columns; 0 = auto (square-ish).\n"
                        "---\n"
                        "Колонок в сетке; 0 = авто (примерно квадрат)."
                    )
                }),
                "font_size": ("INT", {
                    "default": 24, "min": 8, "max": 128, "step": 1,
                    "tooltip": (
                        "Label font size.\n"
                        "---\n"
                        "Размер шрифта подписи."
                    )
                }),
                "padding": ("INT", {
                    "default": 8, "min": 0, "max": 64, "step": 1,
                    "tooltip": (
                        "Grid padding and label inset from the frame edge.\n"
                        "---\n"
                        "Отступы сетки и отступ подписи от края кадра."
                    )
                }),
                "position": (["top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"], {
                    "default": "top_left",
                    "tooltip": (
                        "Label position: top or bottom, left / center / right.\n"
                        "---\n"
                        "Позиция подписи: сверху или снизу, слева / по центру / справа."
                    )
                }),
                "text_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Text opacity (0 = invisible, 1 = solid).\n"
                        "---\n"
                        "Прозрачность текста (0 = невидимый, 1 = сплошной)."
                    )
                }),
                "box_opacity": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Background box opacity behind the text (0 = no box).\n"
                        "---\n"
                        "Прозрачность плашки под текстом (0 = без плашки)."
                    )
                }),
                "text_color": (cls.COLOR_NAMES, {
                    "default": "white",
                    "tooltip": (
                        "Text color preset (24 templates).\n"
                        "---\n"
                        "Шаблон цвета текста (24 варианта)."
                    )
                }),
                "box_color": (cls.COLOR_NAMES, {
                    "default": "black",
                    "tooltip": (
                        "Background box color preset (24 templates).\n"
                        "---\n"
                        "Шаблон цвета плашки (24 варианта)."
                    )
                }),
                "box_radius": ("INT", {
                    "default": 10, "min": 0, "max": 128, "step": 1,
                    "tooltip": (
                        "Box corner radius in pixels; 0 = straight corners. Auto-clamped to half of the box size.\n"
                        "---\n"
                        "Радиус скругления углов плашки в пикселях; 0 = прямые углы. Автоматически ограничивается половиной размера плашки."
                    )
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("sheet", "labeled")
    OUTPUT_TOOLTIPS = (
        "Single NxM contact sheet image.\nОдно изображение-сетка NxM.",
        "Batch with labels burned in (same order).\nБатч с вжжёнными подписями (тот же порядок)."
    )
    FUNCTION = "build"
    CATEGORY = "AGSoft/🧩KSampler"
    DESCRIPTION = (
        "🖼️ AGSoft Contact Sheet.\n"
        "Burns one label per frame (from current_str) into a chosen position with preset colors, "
        "separate text/box opacity and rounded box corners; stitches the batch into an NxM contact sheet.\n"
        "---\n"
        "🖼️ AGSoft Contact Sheet.\n"
        "Вжигает по подписи на кадр (из current_str) в выбранную позицию с шаблонными цветами, "
        "раздельной прозрачностью текста и плашки и скруглёнными углами; склеивает батч в контактный лист NxM."
    )

    @staticmethod
    def _font(size):
        from PIL import ImageFont
        for name in ("arial.ttf", "DejaVuSans.ttf",
                     "C:/Windows/Fonts/arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(name, int(size))
            except Exception:
                continue
        try:
            return ImageFont.load_default(int(size))
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _burn(im, text, font, position, padding, text_opacity, box_opacity,
              text_rgb, box_rgb, box_radius):
        from PIL import Image, ImageDraw

        pad = int(padding)
        m = 6
        box_a = int(round(255 * max(0.0, min(1.0, float(box_opacity)))))
        text_a = int(round(255 * max(0.0, min(1.0, float(text_opacity)))))

        # Габариты плашки меряем на черновом слое.
        # Measure the box on a scratch layer.
        scratch = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(scratch)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_w, box_h = tw + 2 * m, th + 2 * m

        vert, horiz = position.split("_")[0], position.split("_")[1]
        if horiz == "left":
            x = pad
        elif horiz == "center":
            x = max(0, (im.width - box_w) // 2)
        else:
            x = max(0, im.width - box_w - pad)
        if vert == "top":
            y = pad
        else:
            y = max(0, im.height - box_h - pad)

        result = im

        # Слой 1: плашка со скруглёнными углами, своим цветом и прозрачностью.
        # Layer 1: rounded box with its own color and opacity, composited over the frame.
        if box_a > 0:
            box_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
            db = ImageDraw.Draw(box_layer)
            radius = max(0, min(int(box_radius), min(box_w, box_h) // 2))
            fill = (box_rgb[0], box_rgb[1], box_rgb[2], box_a)
            if radius > 0:
                try:
                    db.rounded_rectangle([x, y, x + box_w, y + box_h], radius=radius, fill=fill)
                except Exception:
                    db.rectangle([x, y, x + box_w, y + box_h], fill=fill)
            else:
                db.rectangle([x, y, x + box_w, y + box_h], fill=fill)
            result = Image.alpha_composite(result, box_layer)

        # Слой 2: текст поверх плашки, своим цветом и прозрачностью.
        # Layer 2: text over the box with its own color and opacity.
        if text_a > 0:
            text_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
            dt = ImageDraw.Draw(text_layer)
            dt.text((x + m, y + m - bbox[1]), text, font=font,
                    fill=(text_rgb[0], text_rgb[1], text_rgb[2], text_a))
            result = Image.alpha_composite(result, text_layer)

        return result

    @staticmethod
    def _to_tensor(pils):
        import numpy as np
        arrs = [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in pils]
        return torch.from_numpy(np.stack(arrs, axis=0))

    def build(self, images, current_str="", columns=0, font_size=24, padding=8,
              position="top_left", text_opacity=1.0, box_opacity=0.6,
              text_color="white", box_color="black", box_radius=10):
        from PIL import Image
        import numpy as np

        arr = (images.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        imgs = [Image.fromarray(a).convert("RGBA") for a in arr]
        lines = str(current_str or "").split("\n")
        font = self._font(font_size)
        text_rgb = self.PRESET_COLORS.get(str(text_color), (255, 255, 255))
        box_rgb = self.PRESET_COLORS.get(str(box_color), (0, 0, 0))

        labeled = []
        for idx, im in enumerate(imgs):
            text = lines[idx].strip() if idx < len(lines) else ""
            if text and float(text_opacity) > 0.0:
                im = self._burn(im, text, font, position, padding,
                                text_opacity, box_opacity, text_rgb, box_rgb, box_radius)
            labeled.append(im)

        labeled_t = self._to_tensor(labeled)

        n = len(labeled)
        if n == 0:
            return (labeled_t, labeled_t)
        cols = int(columns) if columns and int(columns) > 0 else max(1, math.ceil(math.sqrt(n)))
        cols = max(1, min(cols, n))
        rows_n = max(1, math.ceil(n / cols))
        cw = max(im.width for im in labeled)
        ch = max(im.height for im in labeled)
        pad = int(padding)
        canvas = Image.new("RGBA", (cols * cw + (cols + 1) * pad, rows_n * ch + (rows_n + 1) * pad), (0, 0, 0, 255))
        for idx, im in enumerate(labeled):
            r, c = divmod(idx, cols)
            canvas.paste(im, (pad + c * (cw + pad), pad + r * (ch + pad)))

        return (self._to_tensor([canvas]), labeled_t)


#------------------------------------------------------------------------------
# Регистрация нод.
# Node registration.
#------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AGSoft_KSampler": AGSoft_KSampler,
    "AGSoft_KSampler_Options": AGSoft_KSampler_Options,
    "AGSoftKSamplerOptionsSingle": AGSoftKSamplerOptionsSingle,
    "AGSoftKSamplerOptionsDual": AGSoftKSamplerOptionsDual,
    "AGSoftKSamplerOptionsLora": AGSoftKSamplerOptionsLora,
    "AGSoftContactSheet": AGSoftContactSheet,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_KSampler": "🧩AGSoft KSampler",
    "AGSoft_KSampler_Options": "🛠AGSoft KSampler Options",
    "AGSoftKSamplerOptionsSingle": "🛠AGSoft KSampler options_single",
    "AGSoftKSamplerOptionsDual": "🛠AGSoft KSampler options_dual",
    "AGSoftKSamplerOptionsLora": "🛠AGSoft KSampler options_lora",
    "AGSoftContactSheet": "🖼️AGSoft Contact Sheet",
}