# ==============================================================================
# AGSoft_Audio_Frame_Cutter.py
# ==============================================================================
# Ноды / Nodes:
# 🔊✂️AGSoft Audio Frame Cutter Master — ведущая нода: все настройки, первый отрезок.
# 🔊✂️AGSoft Audio Frame Cutter Link — второстепенная нода: один отрезок продолжения.
#
# Описание / Description:
# Цепочечная резка исходной дорожки на отрезки по кадрам. Master держит ВСЕ
# настройки (fps, trim, смещение начала), режет отрезок 1 и создаёт шину chain:
# нормализованная волна источника, sample_rate, fps, trim, курсор (секунды),
# счётчик кадров frames_done и накопленные куски. Каждая Link-нода читает шину,
# режет СВОЙ отрезок ровно с конца предыдущего (курсор), добавляет кусок в шину
# и передаёт дальше. Любое число звеньев без громоздких сокетов/выходов.
# Chain-based cutting of a source track into frame-defined segments. The Master
# holds ALL settings (fps, trim, start offset), cuts segment 1 and creates the
# chain bus: normalized source waveform, sample_rate, fps, trim, cursor
# (seconds), frames_done counter and accumulated pieces. Each Link node reads
# the bus, cuts ITS segment exactly from the end of the previous one (cursor),
# appends its piece to the bus and passes it on. Any number of links without
# bulky sockets/outputs.
#
# Выходы КАЖДОЙ ноды / Outputs of EVERY node:
# ⚡ audio — сэмпл-точный кусок отрезка (аудио-реф сцены).
#    audio — sample-accurate segment piece (scene audio reference).
# ⚡ start_seconds / end_seconds / duration_seconds — точные границы окна
#    (для Trim Audio Duration: start_index+duration; ✂️Audio Split Plus:
#    start_time+end_time).
#    start_seconds / end_seconds / duration_seconds — exact window boundaries
#    (for Trim Audio Duration: start_index+duration; ✂️Audio Split Plus:
#    start_time+end_time).
# ⚡ chain — шина дальше по цепочке (курсор, кадры, куски, настройки).
#    chain — the bus further down the chain (cursor, frames, pieces, settings).
# ⚡ audio_stitched — сборка ВСЕХ отрезков вверх по цепочке, включая текущий:
#    на ПОСЛЕДНЕЙ ноде цепи это готовый трек склеенного видео (ComfyUI не умеет
#    передавать данные против потока, поэтому сборка появляется в хвосте цепи).
#    audio_stitched — assembly of ALL segments up the chain including the current
#    one: on the LAST node of the chain it is the finished stitched-video track
#    (ComfyUI cannot pass data upstream, so the assembly appears at the tail).
# ⚡ stitched_duration_seconds — длительность склеенного видео/аудио вверх по
#    цепи включая текущий отрезок; на ПОСЛЕДНЕЙ ноде = длительность всего ролика.
#    stitched_duration_seconds — duration of the stitched video/audio up the chain
#    including the current segment; on the LAST node = full movie duration.
#
# Математика / Math:
# ⚡ Окна встык по источнику (без пропусков = без щелчков); отрезки 2+ (все
#    Link) короче на trim_first_frames/fps — дубль кадра-якоря first_frame.
#    Windows tile the source contiguously (no gaps = no clicks); segments 2+
#    (all Links) are shorter by trim_first_frames/fps — the anchor-frame duplicate.
# ⚡ Курсор и frames_done передаются по шине: Link всегда начинает с конца
#    предыдущего отрезка, дрейф исключён.
#    Cursor and frames_done travel in the bus: a Link always starts from the end
#    of the previous segment, drift is impossible.
#
# Автор / Author: AGSoft
# Дата / Date: 12.09.2026
# ==============================================================================

DEFAULT_FPS = 24.0
DEFAULT_TRIM = 1
CHAIN_TYPE = "AGS_FRAME_CUT_CHAIN"

import torch


def _norm_waveform(wav):
    """
    Приводит волну к 3D [B, C, T] и клампит в ±1 (защита WAV-заголовка Video Save).
    Normalizes waveform to 3D [B, C, T] and clamps to ±1 (Video Save WAV safety).
    """
    wav = wav.detach().cpu().float()
    if wav.dim() == 1:
        wav = wav.view(1, 1, -1)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(1)
    elif wav.dim() > 3:
        wav = wav.reshape(-1, wav.shape[-2], wav.shape[-1])
    return wav.clamp(-1.0, 1.0)


def _slice_piece(wav, sr, start, length):
    """
    Сэмпл-точный срез волны [start, start+length); паддинг тишиной при нехватке.
    Sample-accurate waveform slice [start, start+length); silence padding if short.
    """
    want = max(1, int(round(length * sr)))
    a = int(round(start * sr))
    b = a + want
    n = wav.shape[-1]
    if a >= n:
        return torch.zeros((*wav.shape[:-1], want), dtype=wav.dtype), True
    piece = wav[..., a:min(b, n)]
    padded = False
    if piece.shape[-1] < want:
        pad = torch.zeros((*piece.shape[:-1], want - piece.shape[-1]), dtype=piece.dtype)
        piece = torch.cat([piece, pad], dim=-1)
        padded = True
    return piece, padded


def _audio_dict(wav, sr):
    """Собирает AUDIO-словарь ComfyUI / Builds a ComfyUI AUDIO dict."""
    return {"waveform": wav, "sample_rate": sr}


class AGSoftAudioFrameCutterMaster:
    """
    🔊️ Ведущая нода цепи: все настройки + первый отрезок + создание шины.
    Chain head node: all settings + first segment + bus creation.
    """

    DESCRIPTION = (
        "🔊✂️AGSoft Audio Frame Cutter Master.\n"
        "Head of the cutting chain. Holds ALL settings: fps (fps_float from 🎬AGSoft MiniMax Base), "
        "trim_first_frames (anchor-frame duplicate, must match the video stitch trim), start offset "
        "(offset_seconds + offset_frames/fps). Cuts segment 1 (frames = its length in frames, link "
        "total_frames from Base) and creates the chain bus carrying the normalized source waveform, "
        "sample_rate, fps, trim, cursor in seconds, frames_done counter and accumulated pieces. "
        "Outputs: audio = segment 1 piece (scene reference); start/end/duration_seconds = exact "
        "window boundaries for external cutters; chain = bus for the next 🔊✂️Link node; "
        "audio_stitched = assembly of all segments up to this node (here: segment 1 only) — the "
        "LAST node of the chain holds the finished full track on this output; "
        "stitched_duration_seconds = stitched video duration up to this node (full movie on the "
        "last node). Windows tile the source contiguously (no gaps = no clicks).\n"
        "---\n"
        "🔊✂️AGSoft Audio Frame Cutter Master.\n"
        "Голова цепи резки. Держит ВСЕ настройки: fps (fps_float из 🎬AGSoft MiniMax Base), "
        "trim_first_frames (дубль кадра-якоря, должен совпадать с trim видео-склейки), смещение "
        "начала (offset_seconds + offset_frames/fps). Режет отрезок 1 (frames = его длина в кадрах, "
        "подключите total_frames из Base) и создаёт шину chain: нормализованная волна источника, "
        "sample_rate, fps, trim, курсор в секундах, счётчик frames_done и накопленные куски. "
        "Выходы: audio = кусок отрезка 1 (реф сцены); start/end/duration_seconds = точные границы "
        "окна для внешних режущих нод; chain = шина для следующей ноды 🔊️Link; audio_stitched = "
        "сборка всех отрезков вверх по цепи (здесь: только отрезок 1) — на ПОСЛЕДНЕЙ ноде цепи на "
        "этом выходе лежит готовый полный трек; stitched_duration_seconds = длительность склеенного "
        "видео вверх по цепи (на последней ноде = весь ролик). Окна идут по источнику встык (без "
        "пропусков = без щелчков)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Source track to cut from (AGSoft Load Audio or any AUDIO output).\n"
                            "---\n"
                            "Исходная дорожка для резки (AGSoft Load Audio или любой выход AUDIO)."
                        )
                    }
                ),
                "fps_float": (
                    "FLOAT",
                    {
                        "default": DEFAULT_FPS, "min": 1.0, "max": 120.0, "step": 0.001,
                        "tooltip": (
                            "Frame rate. Connect fps_float from 🎬AGSoft MiniMax Base (24 MiniMax H3, 25 LTX, etc.).\n"
                            "---\n"
                            "Частота кадров. Подключите fps_float из 🎬AGSoft MiniMax Base (24 MiniMax H3, 25 LTX и т.д.)."
                        )
                    }
                ),
                "frames": (
                    "INT",
                    {
                        "default": 124, "min": 1, "max": 100000, "step": 1,
                        "tooltip": (
                            "Segment 1 length in frames. Link total_frames from 🎬AGSoft MiniMax Base.\n"
                            "---\n"
                            "Длительность отрезка 1 в кадрах. Подключите total_frames из 🎬AGSoft MiniMax Base."
                        )
                    }
                ),
            },
            "optional": {
                "trim_first_frames": (
                    "INT",
                    {
                        "default": DEFAULT_TRIM, "min": 0, "max": 8, "step": 1,
                        "tooltip": (
                            "Anchor-frame duplicate frames removed from segments 2+ (all Link nodes). "
                            "Must match the video stitch trim.\n"
                            "---\n"
                            "Кадры-дубли кадра-якоря, сбрасываемые с отрезков 2+ (все Link-ноды). "
                            "Должно совпадать с trim видео-склейки."
                        )
                    }
                ),
                "offset_frames": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 1000000, "step": 1,
                        "tooltip": (
                            "Start of segment 1 in frames (e.g. 22). Added to offset_seconds.\n"
                            "---\n"
                            "Начало отрезка 1 в кадрах (например 22). Складывается с offset_seconds."
                        )
                    }
                ),
                "offset_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                        "tooltip": (
                            "Start of segment 1 in seconds (e.g. 18.0). Added to offset_frames/fps.\n"
                            "---\n"
                            "Начало отрезка 1 в секундах (например 18.0). Складывается с offset_frames/fps."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = (CHAIN_TYPE, "AUDIO", "FLOAT", "FLOAT", "FLOAT", "AUDIO", "FLOAT")
    RETURN_NAMES = (
        "chain", "audio", "start_seconds", "end_seconds",
        "duration_seconds", "audio_stitched", "stitched_duration_seconds",
    )
    FUNCTION = "cut_first"
    CATEGORY = "AGSoft/Audio"

    def cut_first(self, audio, fps_float, frames,
                  trim_first_frames=DEFAULT_TRIM, offset_frames=0, offset_seconds=0.0):
        fps = float(fps_float or DEFAULT_FPS)
        if fps <= 0:
            raise ValueError(
                f"[AGSoft Audio Frame Cutter Master] fps должен быть > 0, получено {fps}.\n"
                f"fps must be > 0, got {fps}."
            )
        if not isinstance(audio, dict) or "waveform" not in audio:
            raise ValueError(
                "[AGSoft Audio Frame Cutter Master] Вход audio не является валидным AUDIO.\n"
                "Input audio is not a valid AUDIO object."
            )
        sr = int(audio.get("sample_rate", 0) or 44100)
        wav = _norm_waveform(audio["waveform"])

        trim = int(trim_first_frames or 0)
        offset = float(offset_seconds or 0.0) + int(offset_frames or 0) / fps
        length = max(0.0, int(frames) / fps)

        piece, padded = _slice_piece(wav, sr, offset, length)
        if padded:
            print(
                f"[AGSoft Audio Frame Cutter Master] Warning: источник короче окна, хвост дополнен тишиной.\n"
                f"Source is shorter than the window, tail padded with silence."
            )

        chain = {
            "wave": wav,
            "sr": sr,
            "fps": fps,
            "trim_s": trim / fps,
            "cursor": offset + length,
            "frames_done": int(frames),
            "pieces": [piece],
        }
        stitched_dur = piece.shape[-1] / float(sr)
        line = f"seg1: frames={int(frames)} start={offset!r}s end={offset + length!r}s dur={length!r}s"
        print(f"[AGSoft Audio Frame Cutter Master] fps={fps!r} | trim={trim} | {line}")

        return (
            chain,
            _audio_dict(piece, sr),
            offset,
            offset + length,
            length,
            _audio_dict(piece, sr),
            stitched_dur,
        )


class AGSoftAudioFrameCutterLink:
    """
    🔊✂️ Звено цепи: режет свой отрезок с курсора шины и передаёт её дальше.
    Chain link: cuts its segment from the bus cursor and passes the bus on.
    """

    DESCRIPTION = (
        "🔊✂️AGSoft Audio Frame Cutter Link.\n"
        "One continuation segment. Reads the chain bus from the previous node (Master or Link), "
        "cuts ITS segment starting exactly at the bus cursor (the end of the previous segment), "
        "shortened by trim_first_frames/fps (the anchor-frame duplicate, set on the Master), "
        "appends its piece to the bus and passes the bus on. Set frames = this segment's length in "
        "frames (link total_frames from 🎬AGSoft MiniMax Base). Outputs: audio = this segment's "
        "piece (scene reference); start/end/duration_seconds = exact window boundaries for external "
        "cutters; chain = bus for the next Link; audio_stitched = assembly of ALL segments up the "
        "chain including this one — on the LAST node of the chain this output is the finished full "
        "track for AGSoft Video Save.audio; stitched_duration_seconds = stitched video duration up "
        "to this node (full movie on the last node). Add as many Links as you need: 4, 10, 50.\n"
        "---\n"
        "🔊✂️AGSoft Audio Frame Cutter Link.\n"
        "Одно звено продолжения. Читает шину chain от предыдущей ноды (Master или Link), режет СВОЙ "
        "отрезок ровно с курсора шины (конец предыдущего отрезка), укороченный на "
        "trim_first_frames/fps (дубль кадра-якоря, задаётся на Master), добавляет свой кусок в шину "
        "и передаёт её дальше. Задайте frames = длина этого отрезка в кадрах (подключите "
        "total_frames из 🎬AGSoft MiniMax Base). Выходы: audio = кусок этого отрезка (реф сцены); "
        "start/end/duration_seconds = точные границы окна для внешних режущих нод; chain = шина для "
        "следующего Link; audio_stitched = сборка ВСЕХ отрезков вверх по цепи включая текущий — на "
        "ПОСЛЕДНЕЙ ноде цепи на этом выходе лежит готовый полный трек для AGSoft Video Save.audio; "
        "stitched_duration_seconds = длительность склеенного видео вверх по цепи (на последней "
        "ноде = весь ролик). Звеньев может быть сколько угодно: 4, 10, 50."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (
                    CHAIN_TYPE,
                    {
                        "tooltip": (
                            "Chain bus from the previous Master/Link node.\n"
                            "---\n"
                            "Шина chain от предыдущей ноды Master/Link."
                        )
                    }
                ),
                "frames": (
                    "INT",
                    {
                        "default": 124, "min": 1, "max": 100000, "step": 1,
                        "tooltip": (
                            "This segment's length in frames. Link total_frames from 🎬AGSoft MiniMax Base.\n"
                            "---\n"
                            "Длительность этого отрезка в кадрах. Подключите total_frames из 🎬AGSoft MiniMax Base."
                        )
                    }
                ),
            },
        }

    RETURN_TYPES = (CHAIN_TYPE, "AUDIO", "FLOAT", "FLOAT", "FLOAT", "AUDIO", "FLOAT")
    RETURN_NAMES = (
        "chain", "audio", "start_seconds", "end_seconds",
        "duration_seconds", "audio_stitched", "stitched_duration_seconds",
    )
    FUNCTION = "cut_next"
    CATEGORY = "AGSoft/Audio"

    def cut_next(self, chain, frames):
        if not isinstance(chain, dict) or "wave" not in chain:
            raise ValueError(
                "[AGSoft Audio Frame Cutter Link] Вход chain не является шиной цепи.\n"
                "Input chain is not a valid chain bus."
            )
        wav = chain["wave"]
        sr = int(chain["sr"])
        fps = float(chain["fps"])
        trim_s = float(chain["trim_s"])
        cursor = float(chain["cursor"])
        index = len(chain["pieces"]) + 1

        length = max(0.0, int(frames) / fps - trim_s)
        piece, padded = _slice_piece(wav, sr, cursor, length)
        if padded:
            print(
                f"[AGSoft Audio Frame Cutter Link] Warning: источник короче окна seg{index}, хвост дополнен тишиной.\n"
                f"Source is shorter than the window of seg{index}, tail padded with silence."
            )

        pieces = chain["pieces"] + [piece]
        new_chain = {
            "wave": wav,
            "sr": sr,
            "fps": fps,
            "trim_s": trim_s,
            "cursor": cursor + length,
            "frames_done": int(chain.get("frames_done", 0)) + int(frames),
            "pieces": pieces,
        }
        stitched = torch.cat(pieces, dim=-1)
        stitched_dur = stitched.shape[-1] / float(sr)
        line = (
            f"seg{index}: frames={int(frames)} start={cursor!r}s "
            f"end={cursor + length!r}s dur={length!r}s"
        )
        print(f"[AGSoft Audio Frame Cutter Link] {line}")

        return (
            new_chain,
            _audio_dict(piece, sr),
            cursor,
            cursor + length,
            length,
            _audio_dict(stitched, sr),
            stitched_dur,
        )


# Маппинг для ComfyUI / ComfyUI mappings
NODE_CLASS_MAPPINGS = {
    "AGSoftAudioFrameCutterMaster": AGSoftAudioFrameCutterMaster,
    "AGSoftAudioFrameCutterLink": AGSoftAudioFrameCutterLink,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftAudioFrameCutterMaster": "🔊✂️AGSoft Audio Frame Cutter Master",
    "AGSoftAudioFrameCutterLink": "🔊✂️AGSoft Audio Frame Cutter Link",
}