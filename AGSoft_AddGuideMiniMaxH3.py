"""
==============================================================================
AGSoft_AddGuideMiniMaxH3.py
==============================================================================
Нода: 📍AGSoft Add Guide for MiniMax H3
Описание / Description:
Добавляет направляющие кадры (изображение + опционально аудио) в conditioning
MiniMax H3 через нативный ключ minimax_keyframes. Позиция задаётся в секундах
или в кадрах (-1 = последний кадр), количество направляющих регулируется
Number of Guides.
---
Adds guide frames (image + optional audio) into MiniMax H3 conditioning via
the native minimax_keyframes key. Position is set in seconds or frames
(-1 = last frame), the number of guides is controlled by Number of Guides.
Возможности / Features:
⚡ Гайды пишутся в нативный ключ minimax_keyframes (resolved_frame_index,
   latent, audio_latent) — модель реально их получает
   Guides are written into the native minimax_keyframes key
   (resolved_frame_index, latent, audio_latent) so the model consumes them
⚡ Режим позиции: секунды (FLOAT) или кадры (INT); -1 = последний кадр
   Position mode: seconds (FLOAT) or frames (INT); -1 = last frame
⚡ Ошибка при выходе за пределы видео (как в нативной ноде)
   Error when guide exceeds video bounds (as in native node)
⚡ Динамические слоты направляющих: входы добавляются/удаляются через JS
   Dynamic guide slots: inputs are added/removed via JS
⚡ Аудио-входы опциональны: появляются только при подключённом audio_vae
   Audio inputs are optional: appear only when audio_vae is linked
⚡ Клипы 17k+5 кадров, ресемпл/кроп аудио и spatial comp 16 как в core
   17k+5 frame clips, audio resample/crop and spatial comp 16 as in core
Автор / Author: AGSoft
Дата / Date: 20.09.2026
==============================================================================
"""

import json
import math
import logging
import importlib
import sys
import comfy.utils
import node_helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# print("[AGSoft Add Guide MiniMax H3] loaded (bounds check error instead of silent clamp)")

MAX_GUIDES = 16


def _find_native():
    # locate core module with MiniMaxH3AddGuide / ищем core-модуль с MiniMaxH3AddGuide
    try:
        import nodes as core_nodes
        cls = core_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3AddGuide")
        if cls is not None:
            return importlib.import_module(cls.__module__)
    except Exception:
        pass
    for mod in list(sys.modules.values()):
        if mod is not None and hasattr(mod, "MiniMaxH3AddGuide") and hasattr(mod, "FRAME_PER_TOKEN"):
            return mod
    return None


NATIVE = _find_native()
if NATIVE is None:
    logger.warning("[AGSoft Add Guide MiniMax H3] native MiniMaxH3AddGuide not found in core / нативная MiniMaxH3AddGuide не найдена в core")


class AGSoftAddGuideMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "positive": ("CONDITIONING", {
                "tooltip": ("Input positive conditioning.\n---\nВходной позитивный conditioning."),
            }),
            "latent": ("LATENT", {
                "tooltip": ("MiniMax H3 AV latent: defines size and frame count.\n---\nAV-латент MiniMax H3: задаёт размер и число кадров."),
            }),
            "vae": ("VAE", {
                "tooltip": ("Video VAE for encoding guide images.\n---\nВидео VAE для кодирования направляющих изображений."),
            }),
            "number_of_guides": ("INT", {
                "default": 1, "min": 1, "max": MAX_GUIDES, "step": 1,
                "tooltip": ("How many guide slots are active.\n---\nСколько направляющих слотов активно."),
            }),
            "position_mode": (["Seconds", "Frames"], {
                "default": "Seconds",
                "tooltip": (
                    "How guides are positioned: Seconds uses fps to convert to frame index; "
                    "Frames takes the index directly.\n---\n"
                    "Как позиционируются гайды: Seconds конвертирует через fps в индекс кадра; "
                    "Frames берёт индекс напрямую."
                ),
            }),
            "fps": ("FLOAT", {
                "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                "tooltip": ("FPS for seconds-to-frame conversion (MiniMax H3 native is 24).\n---\nFPS для перевода секунд в кадры (родной для MiniMax H3 — 24)."),
            }),
        }
        optional = {
            "audio_vae": ("VAE", {
                "tooltip": ("Audio VAE: linking it reveals optional audio inputs.\n---\nAudio VAE: подключение открывает опциональные аудио-входы."),
            }),
        }
        for i in range(1, MAX_GUIDES + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": (f"Guide image/clip #{i}; batches >=5 frames become 17k+5 clips.\n---\nНаправляющее изображение/клип #{i}; батчи >=5 кадров становятся клипами 17k+5."),
            })
            optional[f"audio_{i}"] = ("AUDIO", {
                "tooltip": (f"Guide audio #{i} (optional, needs audio_vae).\n---\nНаправляющее аудио #{i} (опционально, нужен audio_vae)."),
            })
            optional[f"seconds_{i}"] = ("FLOAT", {
                "default": 0.0, "min": -1.0, "max": 3600.0, "step": 0.05,
                "tooltip": (f"Position of guide #{i} in seconds, -1 = last frame.\n---\nПозиция направляющей #{i} в секундах, -1 = последний кадр."),
            })
            optional[f"frames_{i}"] = ("INT", {
                "default": 0, "min": -1, "max": 9999, "step": 1,
                "tooltip": (f"Position of guide #{i} in frames, -1 = last frame.\n---\nПозиция направляющей #{i} в кадрах, -1 = последний кадр."),
            })
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "guides_info")
    OUTPUT_TOOLTIPS = (
        "Conditioning with native minimax_keyframes guides.\n---\nConditioning с нативными гайдами minimax_keyframes.",
        "JSON summary of guides (seconds, resolved frame_idx).\n---\nJSON-сводка направляющих (секунды, разрешённый frame_idx).",
    )
    FUNCTION = "add_guides"
    CATEGORY = "AGSoft/MiniMaxH3"
    DESCRIPTION = (
        "Attach guide frames (seconds or frames based) to MiniMax H3 conditioning via native minimax_keyframes.\n---\nПрикрепляет направляющие кадры (в секундах или кадрах) к conditioning MiniMax H3 через нативный minimax_keyframes."
    )
    WEB_DIRECTORY = "./web"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        n = kwargs.get("number_of_guides", 1)
        if n is not None and not (1 <= int(n) <= MAX_GUIDES):
            return f"number_of_guides out of range 1..{MAX_GUIDES}: {n}"
        return True

    def _av_tensors(self, samples):
        # native AV latent: nested, 2 tensors, video 24ch / нативный AV-латент: nested, 2 тензора, видео 24 канала
        tensors = getattr(samples, "tensors", None)
        if not getattr(samples, "is_nested", False) or not tensors or len(tensors) != 2:
            return None
        video = tensors[0]
        if video.ndim != 5 or video.shape[1] != 24:
            return None
        return video, tensors[1]

    def _frame_count(self, video):
        fpt = getattr(NATIVE, "FRAME_PER_TOKEN", None)
        if fpt is None:
            raise RuntimeError("core FRAME_PER_TOKEN not found / core FRAME_PER_TOKEN не найден")
        return sum(fpt[k % 5] for k in range(video.shape[2]))

    def _frame_idx(self, mode, value, fps):
        # -1 = last frame (native resolves from end) / -1 = последний кадр (нативно с конца)
        if mode == "Seconds":
            sec = float(value)
            if sec < 0:
                return -1
            fi = int(round(sec * fps))
        else:
            fi = int(value)
            if fi < 0:
                return -1
        return fi  # no clamp here; bounds check happens later / без clamp; проверка границ дальше

    def _guide_frames(self, image):
        n = image.shape[0]
        if n < 5:
            return 1
        while n % 17 != 5:
            n -= 1
        return n

    def add_guides(self, positive, latent, vae, number_of_guides, position_mode, fps, audio_vae=None, **kwargs):
        if NATIVE is None:
            raise RuntimeError("AGSoft Add Guide needs core MiniMaxH3AddGuide (ComfyUI >= 0.34.0) / нужен core MiniMaxH3AddGuide (ComfyUI >= 0.34.0)")
        av = self._av_tensors(latent.get("samples"))
        if av is None:
            raise ValueError("latent must be a MiniMax H3 AV latent / латент должен быть AV-латентом MiniMax H3")
        video, audio_tok = av
        height = video.shape[3] * 16  # spatial comp 16 as in core / пространственный комп 16 как в core
        width = video.shape[4] * 16
        frame_count = self._frame_count(video)
        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        info = []
        for i in range(1, int(number_of_guides) + 1):
            img = kwargs.get(f"image_{i}")
            aud = kwargs.get(f"audio_{i}")
            if img is None and aud is None:
                continue  # skip empty slot / пропускаем пустой слот
            if position_mode == "Seconds":
                value = float(kwargs.get(f"seconds_{i}") or 0.0)
            else:
                value = int(kwargs.get(f"frames_{i}") or 0)
            fi = self._frame_idx(position_mode, value, fps)
            resolved = fi if fi >= 0 else frame_count + fi
            gf = self._guide_frames(img) if img is not None else 1
            if resolved < 0 or resolved + gf > frame_count:
                raise ValueError(f"guide #{i}: frame {resolved} + {gf} frames outside {frame_count} / гайд #{i}: кадр {resolved} + {gf} вне {frame_count}")
            keyframe = {"resolved_frame_index": resolved}
            if img is not None:
                frames = NATIVE._resize(img[:gf], width, height, "center")
                keyframe["latent"] = vae.encode(frames)
            if aud is not None:
                if audio_vae is None:
                    raise ValueError(f"guide #{i}: audio needs audio_vae / гайд #{i}: аудио требует audio_vae")
                z, rt = NATIVE._encode_ref_audio(audio_vae, aud)
                fr = getattr(NATIVE, "FRAME_RESCALE", 0.25)
                max_rt = math.floor(audio_tok.shape[-1] - fr * resolved)
                if max_rt < 1:
                    raise ValueError(f"guide #{i}: frame {resolved} past audio track end / гайд #{i}: кадр {resolved} за концом аудио")
                if rt > max_rt:
                    z = z[..., :max_rt].clone()
                keyframe["audio_latent"] = z
            keyframes.append(keyframe)
            info.append({
                "slot": i, "mode": position_mode,
                "value": value, "frame_idx": resolved, "guide_frames": gf,
            })
        if not info:
            logger.warning("[AGSoft Add Guide MiniMax H3] no image/audio connected / нет подключённых изображений/аудио")
        out = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        return (out, json.dumps(info, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {
    "AGSoftAddGuideMiniMaxH3": AGSoftAddGuideMiniMaxH3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAddGuideMiniMaxH3": "📍AGSoft Add Guide for MiniMax H3"
}