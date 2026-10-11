"""
==============================================================================
AGSoft_Get_Video_Components.py
==============================================================================
Нода: 🎬AGSoft Get Video Components
Версия / Version: v10.11
Описание / Description:
Клон встроенной ноды GetVideoComponents с расширенными выходами: добавлены
width, height, total_frames и duration_seconds. Принимает объект VIDEO и
разбирает его на кадры (IMAGE), аудио (AUDIO), fps (FLOAT), bit_depth (INT),
color_space (STRING) плюс геометрию и длительность. Поддерживает и новый API
(get_components) и старый (get_pixels) для совместимости.
---
Clone of the built-in GetVideoComponents node with extra outputs: added
width, height, total_frames and duration_seconds. Takes a VIDEO object and
extracts frames (IMAGE), audio (AUDIO), fps (FLOAT), bit_depth (INT),
color_space (STRING) plus geometry and duration. Supports both the new API
(get_components) and the legacy one (get_pixels) for compatibility.
Возможности / Features:
⚡ Один проход по контейнеру через get_components() — кадры, аудио, fps
   извлекаются одновременно; bit_depth и color_space — отдельными методами.
   One pass over the container via get_components() — frames, audio, fps
   extracted together; bit_depth and color_space — via separate methods.
⚡ width/height/total_frames берутся из shape уже декодированного тензора
   кадров (с учётом crop/rotation), а duration_seconds = total_frames/fps.
   width/height/total_frames taken from the shape of the already decoded
   frame tensor (with crop/rotation applied), and duration_seconds = total/fps.
⚡ Поддержка обоих API видео: get_components() (новый) и get_pixels()
   (старый VideoFromFile) + тензорный fallback.
   Both video APIs supported: get_components() (new) and get_pixels()
   (legacy VideoFromFile) + tensor fallback.
⚡ Корректная обработка пустого видео (0 кадров): width=height=total=duration=0.
   Correct handling of empty video (0 frames): width=height=total=duration=0.
⚡ Registry-safe: без urllib/aiohttp/subprocess/eval, без голых https://.
   Registry-safe: no urllib/aiohttp/subprocess/eval, no bare https://.
Автор / Author: AGSoft
Дата / Date: 11.10.2026
==============================================================================
"""
import logging
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#print("[AGSoft Get Video Components] v10.11 loaded (clone + width/height/frames/duration)")

# Двуязычные тултипы / Bilingual tooltips
_T = lambda en, ru: en + "\n---\n" + ru

# ==============================================================================
# Нода: 🎬AGSoft Get Video Components
# ==============================================================================
class AGSoft_GetVideoComponents:
    CATEGORY = "AGSoft/Video"
    FUNCTION = "extract"

    DESCRIPTION = _T(
        "Extracts components from a VIDEO object: images (IMAGE batch), audio "
        "(AUDIO or None), frame rate, bit depth, color space, plus frame width, "
        "height, total frame count and duration in seconds.\n"
        "Width/height come from the already-decoded frame tensor (after any "
        "crop/rotation applied by the source VIDEO). duration_seconds is "
        "total_frames / fps — same rule as AGSoft MiniMax Base.\n"
        "Supports both new (get_components) and legacy (get_pixels) VIDEO APIs.",
        "Извлекает компоненты из объекта VIDEO: images (батч IMAGE), audio "
        "(AUDIO или None), частоту кадров, битность, цветовое пространство, "
        "плюс ширину/высоту кадра, общее число кадров и длительность в секундах.\n"
        "Ширина/высота берутся из уже декодированного тензора кадров (с учётом "
        "crop/rotation, применённых источником VIDEO). duration_seconds = "
        "total_frames / fps — та же логика, что в AGSoft MiniMax Base.\n"
        "Поддерживает и новый (get_components), и старый (get_pixels) VIDEO API."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": _T(
                        "VIDEO object to extract components from. Accepts output "
                        "of AGSoft Load Video, native Load Video or any object "
                        "with get_components()/get_pixels().",
                        "Объект VIDEO для разбора. Принимает выход AGSoft Load Video, "
                        "нативного Load Video или любого объекта с get_components()/get_pixels()."
                    ),
                }),
            },
        }

    RETURN_TYPES = (
        "IMAGE",     # images
        "AUDIO",     # audio
        "INT",       # width
        "INT",       # height
        "INT",       # total_frames
        "FLOAT",     # duration_seconds
        "FLOAT",     # fps
        "INT",       # bit_depth
        "STRING",    # color_space
    )
    RETURN_NAMES = (
        "images",
        "audio",
        "width",
        "height",
        "total_frames",
        "duration_seconds",
        "fps",
        "bit_depth",
        "color_space",
    )
    OUTPUT_TOOLTIPS = (
        _T("All video frames as a (N, H, W, 3) IMAGE batch, float32, range 0..1.",
           "Все кадры видео как батч IMAGE (N, H, W, 3), float32, диапазон 0..1."),
        _T("Audio track as an AUDIO object (waveform + sample_rate), or None if "
           "the video has no decodable audio stream.",
           "Звуковая дорожка как объект AUDIO (waveform + sample_rate), либо None, "
           "если в видео нет декодируемой аудиодорожки."),
        _T("Frame width in pixels — taken from the decoded frame tensor "
           "(after any source-applied crop/rotation). 0 if the video is empty.",
           "Ширина кадра в пикселях — из декодированного тензора кадров "
           "(с учётом crop/rotation от источника). 0, если видео пустое."),
        _T("Frame height in pixels — taken from the decoded frame tensor "
           "(after any source-applied crop/rotation). 0 if the video is empty.",
           "Высота кадра в пикселях — из декодированного тензора кадров "
           "(с учётом crop/rotation от источника). 0, если видео пустое."),
        _T("Total number of frames extracted. Equals images.shape[0]. "
           "0 if the video is empty.",
           "Общее число извлечённых кадров. Равно images.shape[0]. "
           "0, если видео пустое."),
        _T("Exact clip duration in seconds (total_frames / fps). "
           "0.0 if the video is empty or fps is invalid.",
           "Точная длительность клипа в секундах (total_frames / fps). "
           "0.0, если видео пустое или fps некорректен."),
        _T("Frame rate in frames per second (float).",
           "Частота кадров (float)."),
        _T("Bit depth per color component of the source video stream (int). "
           "Falls back to 8 when the stream metadata is missing.",
           "Битность на цветовую компоненту исходного видеопотока (int). "
           "Фоллбэк на 8, если метаданные потока отсутствуют."),
        _T("Color space / transfer of the decoded frames: 'sRGB', 'HDR' or "
           "'HDR PQ'. Falls back to 'sRGB' for unknown transfer.",
           "Цветовое пространство / transfer декодированных кадров: 'sRGB', 'HDR' "
           "или 'HDR PQ'. Фоллбэк на 'sRGB' для неизвестного transfer."),
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # -------------------------------------------------------------------------
    # Основной метод / Main method
    # -------------------------------------------------------------------------
    def extract(self, video):
        images, audio, fps_float = self._get_components(video)

        # Геометрия и длительность / Geometry and duration
        if images is not None and isinstance(images, torch.Tensor) and images.shape[0] > 0:
            total_frames = int(images.shape[0])
            height = int(images.shape[1])
            width = int(images.shape[2])
        else:
            total_frames = 0
            height = 0
            width = 0

        if fps_float and fps_float > 0:
            duration_seconds = float(total_frames) / float(fps_float)
        else:
            duration_seconds = 0.0

        # bit_depth и color_space — отдельными методами (как в нативной ноде)
        bit_depth = self._safe_call(lambda: video.get_bit_depth(), fallback=8,
                                    method="get_bit_depth")
        color_space = self._safe_call(lambda: video.get_color_space(), fallback="sRGB",
                                      method="get_color_space")

        # fps в нативной ноде всегда приводится к float
        fps_out = float(fps_float) if fps_float else 0.0

        return (
            images if images is not None else torch.zeros(0, 0, 0, 3),
            audio,
            width,
            height,
            total_frames,
            duration_seconds,
            fps_out,
            int(bit_depth) if bit_depth is not None else 8,
            str(color_space) if color_space is not None else "sRGB",
        )

    # -------------------------------------------------------------------------
    # Универсальный разбор VIDEO: get_components() → get_pixels() → dict → тензор
    # Universal VIDEO parsing: get_components() -> get_pixels() -> dict -> tensor
    # -------------------------------------------------------------------------
    def _get_components(self, video):
        images = None
        audio = None
        fps_float = 0.0

        # 1) Новый API: get_components() возвращает VideoComponents
        gc = getattr(video, "get_components", None)
        if callable(gc):
            try:
                comp = gc()
                if comp is not None:
                    images = getattr(comp, "images", None)
                    audio = getattr(comp, "audio", None)
                    fr = getattr(comp, "frame_rate", None)
                    if fr is not None:
                        try:
                            fps_float = float(fr)
                        except (TypeError, ValueError):
                            fps_float = 0.0
            except Exception as e:
                logger.warning(f"[AGSoft Get Video Components] get_components failed: {e}")

        # 2) Fallback: get_pixels() (старый VideoFromFile) — только кадры
        if images is None and hasattr(video, "get_pixels"):
            try:
                images = video.get_pixels()
            except Exception as e:
                logger.warning(f"[AGSoft Get Video Components] get_pixels failed: {e}")

        # 3) Fallback: dict (иногда так приходит из кастомных нод)
        if images is None and isinstance(video, dict):
            images = video.get("images") or video.get("pixels")
            audio = video.get("audio")
            fr = video.get("frame_rate") or video.get("fps")
            if fr is not None:
                try:
                    fps_float = float(fr)
                except (TypeError, ValueError):
                    fps_float = 0.0

        # 4) Fallback: сам тензор — трактуем как IMAGE-батч
        if images is None and isinstance(video, torch.Tensor):
            images = video

        # Нормализация изображений
        if images is not None and isinstance(images, torch.Tensor):
            if images.dim() == 5:  # [B, F, H, W, C] → [F, H, W, C]
                images = images.reshape(images.shape[0] * images.shape[1], *images.shape[2:])
            images = images.detach()
            if images.dtype != torch.float32:
                images = images.float()
            # защита от uint8-диапазона
            if images.numel() > 0 and images.max().item() > 1.5:
                images = images / 255.0

        return images, audio, fps_float

    # -------------------------------------------------------------------------
    # Безопасный вызов метода видео-объекта с фоллбэком / Safe call with fallback
    # -------------------------------------------------------------------------
    def _safe_call(self, fn, fallback, method):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"[AGSoft Get Video Components] {method} failed: {e}")
            return fallback


# ==============================================================================
# Регистрация / Registration
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "AGSoft_GetVideoComponents": AGSoft_GetVideoComponents,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_GetVideoComponents": "🎬AGSoft Get Video Components",
}