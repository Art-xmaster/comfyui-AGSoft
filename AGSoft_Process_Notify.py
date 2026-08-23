# ==============================================================================
# AGSoft_Process_Notify.py
# ==============================================================================
# Нода: 🔔AGSoft Process Notify
# Описание / Description:
# Оповещение о завершении процесса. На вход — ЛЮБОЙ тип (*): подключите выход
# любой ноды (сохранение изображения/видео/звука/текста и т.д.) — когда она
# отработает, нода ПРОИГРАЕТ звук в браузере.
# Звуки: встроенные синтезированные пресеты (beep/ding/chime/success/alarm/pop)
# + файлы из папки sounds/ в корне ноды. Имена БЕЗ угловых скобок, чтобы
# фронтенд не выкидывал их из комбо как HTML-теги.
# Есть loop, громкость, задержка, кнопки Test/Stop.
#
# Completion notification node. Input accepts ANY type (*): connect the output
# of any node (image/video/audio/text save, etc.) — when it finishes, this node
# PLAYS a sound in the browser.
# Sounds: built-in WebAudio presets (beep/ding/chime/success/alarm/pop) + files
# from the node's sounds/ folder. Names WITHOUT angle brackets so the frontend
# doesn't drop them from the combo as HTML tags.
# Has loop, volume, delay, Test/Stop buttons.
#
# Возможности / Features:
# ⚡ Вход любого типа (*) — вставляется после любой ноды. / Any-type (*) input.
# ⚡ Встроенные синтезированные звуки (WebAudio) + файлы из sounds/.
#   Built-in synth sounds (WebAudio) + files from sounds/.
# ⚡ loop / громкость / задержка. / loop / volume / delay.
# ⚡ Test — прослушать без запуска; Stop — мгновенно глушит ВСЁ (все аудио и
#   таймеры), независимо от числа нажатий Test.
#   Test — preview without running; Stop — instantly silences EVERYTHING (all
#   audios and timers), regardless of how many times Test was pressed.
# ⚡ Авто-остановка прошлого звука при новом срабатывании.
#   Auto-stops the previous sound on a new trigger.
# ⚡ Кнопки вписываются в ширину ноды (резиновые). / Buttons fit the node width.
# ⚡ OUTPUT_NODE=True, всегда выполняется (IS_CHANGED=nan).
#   OUTPUT_NODE=True, always runs (IS_CHANGED=nan).
#
# ВАЖНО: список комбо строится при импорте. После правки sounds/ или кода —
# полностью перезапусти ComfyUI.
# IMPORTANT: the combo list is built at import. After changing sounds/ or the
# code — fully restart ComfyUI.
#
# Автор / Author: AGSoft
# Дата / Date: 22.08.2026
# ==============================================================================

import os

from aiohttp import web
from server import PromptServer

SOUND_EXTS = (".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".webm")

# Встроенные синтезированные пресеты (WebAudio, без файлов). БЕЗ "<>".
# Built-in synth presets (WebAudio, no files). WITHOUT "<>".
BUILTIN_SOUNDS = ["beep", "ding", "chime", "success", "alarm", "pop"]


def _sounds_dir() -> str:
    """Папка sounds/ в корне ноды. / The sounds/ folder at the node root."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def _list_sounds():
    """Встроенные пресеты + файлы из sounds/. / Built-ins + files from sounds/."""
    files = []
    base = _sounds_dir()
    if os.path.isdir(base):
        for f in sorted(os.listdir(base)):
            if f.lower().endswith(SOUND_EXTS):
                files.append(f)
    return BUILTIN_SOUNDS + files


# print("[AGSoft Process Notify] sounds:", _list_sounds())


# ------------------------------------------------------------------------------
# Endpoint: отдаёт звуковой файл из папки sounds/ (безопасно).
# Endpoint: serves a sound file from the sounds/ folder (safe).
# ------------------------------------------------------------------------------
@PromptServer.instance.routes.get("/agsoft/sound")
async def agsoft_sound(request):
    name = request.query.get("name", "")
    if not name:
        return web.Response(status=400)
    safe = os.path.basename(name)  # защита от выхода из папки / prevent traversal
    path = os.path.join(_sounds_dir(), safe)
    if not os.path.isfile(path):
        return web.Response(status=404)
    return web.FileResponse(path)


class AGSoft_Process_Notify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sound_file": (
                    _list_sounds(),
                    {
                        "default": "beep",
                        "tooltip": (
                            "Sound: built-in synth presets (beep/ding/chime/success/alarm/pop) "
                            "or a file from the sounds/ folder.\n"
                            "---\n"
                            "Звук: встроенные пресеты (beep/ding/chime/success/alarm/pop) "
                            "или файл из папки sounds/."
                        ),
                    },
                ),
                "volume": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "Громкость 0..1. / Volume 0..1.",
                    },
                ),
                "loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Loop the sound until stopped / next trigger.\n"
                            "---\n"
                            "Зациклить звук до остановки / следующего срабатывания."
                        ),
                    },
                ),
                "delay": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                        "tooltip": "Задержка перед воспроизведением (сек). / Playback delay (sec).",
                    },
                ),
            },
            "optional": {
                "any_input": (
                    "*",
                    {
                        "tooltip": (
                            "Any-type input: connect any node output; when it finishes, the sound "
                            "plays. Can stay unconnected.\n"
                            "---\n"
                            "Вход любого типа: подключите выход любой ноды — по её завершении "
                            "проиграется звук. Может оставаться неподключённым."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "notify"
    CATEGORY = "AGSoft/Utils"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "🔔 AGSoft Process Notify.\n"
        "Plays a sound (built-in synth preset or a file from sounds/) in the browser when the "
        "connected node finishes. Any-type input, loop, volume, delay, Test/Stop.\n"
        "---\n"
        "🔔 AGSoft Process Notify.\n"
        "Проигрывает звук (встроенный пресет или файл из sounds/) в браузере по завершении "
        "подключённой ноды. Вход любого типа, loop, громкость, задержка, кнопки Test/Stop."
    )

    def notify(self, sound_file="beep", volume=1.0, loop=False, delay=0.0, any_input=None):
        return {
            "ui": {"agsoft_notify": [{"sound": sound_file, "volume": float(volume),
                                     "loop": bool(loop), "delay": float(delay)}]},
            "result": (any_input,),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {"AGSoft_Process_Notify": AGSoft_Process_Notify}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoft_Process_Notify": "🔔AGSoft Process Notify"}
