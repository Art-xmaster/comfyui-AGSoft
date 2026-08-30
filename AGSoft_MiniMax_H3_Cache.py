# ==============================================================================
# AGSoft_MiniMax_H3_Cache.py
# ==============================================================================
# Node: 🚀 AGSoft MiniMax H3 Cache
# Version: v1.00
# Accelerates MiniMax H3 sampling by reusing the cached full-step model output
# when both streams (video and audio) change little between sampler steps.
# The change is measured per stream as a sampled relative delta
# mean(|cur-prev|)/(mean(|prev|)+eps) on fp32 strided snapshots; a step is
# reused only when BOTH deltas are below their thresholds (audio veto).
# Profiles: Balanced / Visual Fast / Dialogue Safe / Action Safe / Custom.
# Cache window by start/end percent of the sigma schedule; warmup steps and
# a consecutive-skip limit protect quality.
# SAFE WRAPPER: the node transparently wraps MiniMaxH3Model._forward and
# passes ALL arguments through as-is; internal layers are never called
# manually, so core signature changes cannot break it. Cached outputs are
# cloned and SKIP returns a fresh list copy, because the core forward
# rebinds out[1] (audio velocity scale conversion) of the returned list.
# ---
# Нода: 🚀 AGSoft MiniMax H3 Cache
# Версия: v1.00
# Ускоряет сэмплинг MiniMax H3, переиспользуя закэшированный выход полного
# шага, когда оба потока (видео и аудио) мало меняются между шагами
# сэмплера. Изменение измеряется по каждому потоку как относительная дельта
# по сэмплам mean(|cur-prev|)/(mean(|prev|)+eps) на fp32 strided-снимках; шаг
# переиспользуется, только если ОБЕ дельты ниже своих порогов (вето по аудио).
# Профили: Balanced / Visual Fast / Dialogue Safe / Action Safe / Custom.
# Окно кэша — по start/end процентам расписания сигм; warmup-шаги и лимит
# пропусков подряд защищают качество.
# БЕЗОПАСНАЯ ОБЁРТКА: нода прозрачно оборачивает MiniMaxH3Model._forward и
# передаёт ВСЕ аргументы как есть; внутренние слои никогда не вызываются
# вручную, поэтому смены сигнатур ядра не могут её сломать. Кэш выходов
# клонируется, а SKIP возвращает свежую копию списка, т.к. ядро в forward
# перезаписывает out[1] (конверсия аудио-velocity) возвращённого списка.
# 
# Author / Автор: AGSoft
# Date / Дата: 30.08.2026
# ==============================================================================

import types
import time
import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("[AGSoft MiniMax H3 Cache] v1.00 loaded (dual-stream relative delta metric, audio veto, profiles, safe wrapper)")

# ------------------------------------------------------------------------------
# Versioned profiles with calibrated thresholds on the sampled-relative-delta
# scale this node measures.
# Версионированные профили с калиброванными порогами на шкале относительной
# дельты, которую измеряет нода.
# ------------------------------------------------------------------------------
PROFILES = {
    "Balanced":      {"v": 0.120, "a": 0.100, "start": 0.10, "end": 0.90, "warmup": 2, "max_steps": 1},
    "Visual Fast":   {"v": 0.140, "a": 0.120, "start": 0.06, "end": 0.94, "warmup": 2, "max_steps": 3},
    "Dialogue Safe": {"v": 0.078, "a": 0.065, "start": 0.12, "end": 0.88, "warmup": 3, "max_steps": 1},
    "Action Safe":   {"v": 0.065, "a": 0.052, "start": 0.15, "end": 0.85, "warmup": 3, "max_steps": 1},
}


def _make_snapshot(t, stride, device):
    """
    Detached fp32 strided snapshot of a stream tensor (cheap metric reference).
    Отделённый fp32 strided-снимок тензора потока (дешёвая база для метрики).
    """
    a = t.detach()
    if device == "cpu":
        a = a.to("cpu")
    return a.reshape(-1).float()[::stride].clone()


def _sampled_rel_delta(cur, prev_snap, stride, device):
    """
    Sampled relative delta: mean(|cur-prev|)/(mean(|prev|)+eps), fp32.
    Относительная дельта по сэмплам: mean(|cur-prev|)/(mean(|prev|)+eps), fp32.
    """
    a = cur.detach()
    if device == "cpu":
        a = a.to("cpu")
    a = a.reshape(-1).float()[::stride]
    b = prev_snap
    if b.device != a.device:
        b = b.to(a.device)
    num = (a - b).abs().mean()
    den = b.abs().mean() + 1e-6
    return (num / den).item()


def _log_summary(st):
    """
    Final run summary: total / executed / skipped / speedup / time saved.
    Итоговая сводка: всего / выполнено / пропущено / speedup / сэкономлено.
    """
    done = st["run_total"]
    skipped = st["run_skipped"]
    total = done + skipped
    if total <= 0:
        return
    speedup = total / max(1, done)
    logger.info(
        f"[AGSoft MiniMax H3 Cache] Run summary: {total} steps total | "
        f"executed: {done} | skipped: {skipped} | "
        f"{speedup:.2f}x theoretical speedup | ~{st['time_saved']:.0f}s saved."
    )


def _cached_forward(self, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
    """
    Transparent wrapper over MiniMaxH3Model._forward with dual-stream caching.
    Прозрачная обёртка над MiniMaxH3Model._forward с двухпотоковым кэшем.
    """
    orig = self._agsoft_orig_forward
    st = self._agsoft_state

    if transformer_options is None:
        transformer_options = {}

    def call_original():
        # IMPORTANT: orig is a BOUND method — NO explicit self here.
        # ВАЖНО: orig — bound-метод, self передавать НЕЛЬЗЯ.
        return orig(
            x, timestep, context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # current sigma / step index / percent inside the sigma schedule
    # текущая sigma / номер шага / процент внутри расписания сигм
    # ------------------------------------------------------------------
    sigma = None
    try:
        if torch.is_tensor(timestep):
            sigma = float(timestep.flatten()[0]) / 1000.0
        else:
            sigma = float(timestep) / 1000.0
    except Exception:
        sigma = None

    sample_sigmas = transformer_options.get("sample_sigmas", None)

    # new sampling run detection: sigma grows again or schedule object changed
    # детекция нового запуска: sigma снова растёт или сменился объект расписания
    new_run = False
    if sigma is not None and st["last_sigma"] is not None and sigma > st["last_sigma"] + 1e-4:
        new_run = True
    if sample_sigmas is not None and st["last_sigmas_id"] is not None and id(sample_sigmas) != st["last_sigmas_id"]:
        new_run = True

    if new_run:
        if st["verbose"] and not st["summarized"]:
            _log_summary(st)
        st["run_total"] = 0
        st["run_skipped"] = 0
        st["run_time_total"] = 0.0
        st["time_saved"] = 0.0
        st["avg_run"] = 0.0
        st["consec"] = 0
        st["last_video_snap"] = None
        st["last_audio_snap"] = None
        st["last_out"] = None
        st["summarized"] = False

    step_no = None
    percent = None
    is_last = False
    i = None
    if sample_sigmas is not None and sigma is not None:
        try:
            ss = sample_sigmas.detach().to("cpu", torch.float32)
            i = int((ss - sigma).abs().argmin())
            n = int(ss.shape[0])
            step_no = i + 1
            percent = i / max(1, n - 1)
            # the schedule carries a terminal 0 that is never fed into the
            # model, so the last real call is n-2 in that case
            # в расписании есть терминальный 0, который никогда не подаётся
            # в модель, поэтому последний реальный вызов — n-2
            terminal = bool(n > 1 and float(ss[-1]) <= 1e-6)
            last_idx = (n - 2) if terminal else (n - 1)
            is_last = i >= max(0, last_idx)
            st["last_sigmas_id"] = id(sample_sigmas)
        except Exception:
            step_no = None
            percent = None
    if sigma is not None:
        st["last_sigma"] = sigma
    if step_no is None:
        step_no = st["run_total"] + st["run_skipped"] + 1
        i = step_no - 1
    if percent is None:
        percent = 0.5  # unknown schedule: caching allowed / расписание неизвестно: кэш разрешён

    # ------------------------------------------------------------------
    # passthrough for unexpected input shapes / passthrough для неожиданных входов
    # ------------------------------------------------------------------
    if not isinstance(x, (list, tuple)) or len(x) == 0 or not torch.is_tensor(x[0]):
        return call_original()

    video_x = x[0]
    audio_x = x[1] if len(x) > 1 and torch.is_tensor(x[1]) else None

    # ------------------------------------------------------------------
    # reuse decision: dual-stream relative delta + audio veto
    # решение: относительная дельта по двум потокам + вето аудио
    # ------------------------------------------------------------------
    in_window = st["start_percent"] <= percent <= st["end_percent"]
    warmup_done = (i is None) or (i >= st["warmup_steps"])
    can_reuse = (
        in_window
        and warmup_done
        and st["max_steps"] > 0
        and st["last_video_snap"] is not None
        and st["consec"] < st["max_steps"]
    )
    if can_reuse:
        v_delta = _sampled_rel_delta(video_x, st["last_video_snap"], st["video_metric_stride"], st["device"])
        a_delta = None
        if audio_x is not None and st["last_audio_snap"] is not None:
            a_delta = _sampled_rel_delta(audio_x, st["last_audio_snap"], st["audio_metric_stride"], st["device"])
        v_ok = v_delta < st["video_threshold"]
        a_ok = (a_delta is None) or (a_delta < st["audio_threshold"])
        if v_ok and a_ok:
            st["consec"] += 1
            st["run_skipped"] += 1
            st["time_saved"] += st["avg_run"]
            if st["verbose"]:
                a_txt = f"a {a_delta:.4f} < {st['audio_threshold']:.4f}" if a_delta is not None else "no audio"
                logger.info(
                    f"[AGSoft MiniMax H3 Cache] Step {step_no} SKIP "
                    f"(v {v_delta:.4f} < {st['video_threshold']:.4f}, {a_txt}, "
                    f"{st['consec']}/{st['max_steps']} consecutive, 0.00s, ~{st['avg_run']:.1f}s saved)."
                )
            # Return a FRESH list copy. The core forward REBINDS out[1]
            # (audio velocity scale conversion); returning the cached list
            # itself would let the core mutate (and double-convert) the
            # cache. Cached tensors stay raw and untouched.
            # Возвращаем СВЕЖУЮ копию списка. Ядро в forward ПЕРЕЗАПИСЫВАЕТ
            # out[1] (конверсия аудио-velocity); если вернуть сам кэш-список,
            # ядро замутирует кэш и применит конверсию повторно. Кэш-тензоры
            # остаются сырыми и нетронутыми.
            cached = st["last_out"]
            out = list(cached) if isinstance(cached, (list, tuple)) else cached
            if st["verbose"] and is_last and not st["summarized"]:
                _log_summary(st)
                st["summarized"] = True
            return out
        if not v_ok and not a_ok:
            reason = f"veto v {v_delta:.4f} >= {st['video_threshold']:.4f} & a {a_delta:.4f} >= {st['audio_threshold']:.4f}"
        elif not v_ok:
            reason = f"video veto {v_delta:.4f} >= {st['video_threshold']:.4f}"
        else:
            reason = f"AUDIO veto {a_delta:.4f} >= {st['audio_threshold']:.4f}"
    else:
        if not warmup_done:
            reason = f"warmup ({i + 1}/{st['warmup_steps']})"
        elif st["max_steps"] <= 0:
            reason = "reuse disabled (max_steps=0)"
        elif not in_window:
            reason = "outside cache window"
        elif st["last_video_snap"] is None:
            reason = "initial step"
        else:
            reason = f"max consecutive skips reached ({st['max_steps']})"

    # ------------------------------------------------------------------
    # full run with real wall-time measurement
    # полный проход с замером реального времени
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    out = call_original()
    dt = time.perf_counter() - t0
    st["consec"] = 0
    st["last_video_snap"] = _make_snapshot(video_x, st["video_metric_stride"], st["device"])
    if audio_x is not None:
        st["last_audio_snap"] = _make_snapshot(audio_x, st["audio_metric_stride"], st["device"])
    # Cache CLONED raw outputs. The core forward will rebind out[1] of the
    # returned list right after we return; clones keep the pristine
    # _forward velocities for future reuse.
    # Кэшируем КЛОНИРОВАННЫЕ сырые выходы. Ядро сразу после возврата
    # перезапишет out[1] в возвращённом списке; клоны сохраняют нетронутые
    # velocity из _forward для будущего переиспользования.
    try:
        st["last_out"] = [t.detach().clone() if torch.is_tensor(t) else t for t in out]
    except Exception:
        st["last_out"] = out
    st["run_total"] += 1
    st["run_time_total"] += dt
    st["avg_run"] = st["run_time_total"] / max(1, st["run_total"])
    if st["verbose"]:
        logger.info(f"[AGSoft MiniMax H3 Cache] Step {step_no} RUN ({reason}) in {dt:.2f}s.")
        if is_last and not st["summarized"]:
            _log_summary(st)
            st["summarized"] = True
    return out


class AGSoftMiniMaxH3Cache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Input MiniMax H3 model.\n"
                            "---\n"
                            "Входная модель MiniMax H3."
                        ),
                    },
                ),
                "profile": (
                    ["Balanced", "Visual Fast", "Dialogue Safe", "Action Safe", "Custom"],
                    {
                        "default": "Balanced",
                        "tooltip": (
                            "Versioned preset. Any profile except Custom OVERRIDES the manual widgets "
                            "below:\n"
                            "Balanced: v 0.120 / a 0.100, window 0.10-0.90, warmup 2, max_steps 1;\n"
                            "Visual Fast: v 0.140 / a 0.120, window 0.06-0.94, warmup 2, max_steps 3;\n"
                            "Dialogue Safe: v 0.078 / a 0.065, window 0.12-0.88, warmup 3, max_steps 1;\n"
                            "Action Safe: v 0.065 / a 0.052, window 0.15-0.85, warmup 3, max_steps 1;\n"
                            "Custom: uses the manual widgets.\n"
                            "---\n"
                            "Версионированный пресет. Любой профиль кроме Custom ПЕРЕОПРЕДЕЛЯЕТ ручные "
                            "виджеты ниже:\n"
                            "Balanced: v 0.120 / a 0.100, окно 0.10-0.90, warmup 2, max_steps 1;\n"
                            "Visual Fast: v 0.140 / a 0.120, окно 0.06-0.94, warmup 2, max_steps 3;\n"
                            "Dialogue Safe: v 0.078 / a 0.065, окно 0.12-0.88, warmup 3, max_steps 1;\n"
                            "Action Safe: v 0.065 / a 0.052, окно 0.15-0.85, warmup 3, max_steps 1;\n"
                            "Custom: использует ручные виджеты."
                        ),
                    },
                ),
                "video_threshold": (
                    "FLOAT",
                    {
                        "default": 0.120,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": (
                            "CUSTOM ONLY: sampled relative delta threshold for VIDEO: "
                            "mean(|cur-prev|)/(mean(|prev|)+eps). Recommended range: 0.100-0.140.\n"
                            "---\n"
                            "ТОЛЬКО Custom: порог относительной дельты ВИДЕО: "
                            "mean(|cur-prev|)/(mean(|prev|)+eps). Рекомендуемый диапазон: 0.100-0.140."
                        ),
                    },
                ),
                "audio_threshold": (
                    "FLOAT",
                    {
                        "default": 0.100,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": (
                            "CUSTOM ONLY: sampled relative delta threshold for AUDIO (audio veto). "
                            "Recommended range: 0.080-0.120; lower = safer voice.\n"
                            "---\n"
                            "ТОЛЬКО Custom: порог относительной дельты АУДИО (вето по аудио). "
                            "Рекомендуемый диапазон: 0.080-0.120; ниже = безопаснее голос."
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "CUSTOM ONLY: start of the cache window, fraction of the sigma schedule.\n"
                            "---\n"
                            "ТОЛЬКО Custom: начало окна кэширования, доля расписания сигм."
                        ),
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "CUSTOM ONLY: end of the cache window, fraction of the sigma schedule.\n"
                            "---\n"
                            "ТОЛЬКО Custom: конец окна кэширования, доля расписания сигм."
                        ),
                    },
                ),
                "warmup_steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "CUSTOM ONLY: first N steps of every run are always full.\n"
                            "---\n"
                            "ТОЛЬКО Custom: первые N шагов каждого запуска всегда полные."
                        ),
                    },
                ),
                "max_steps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "CUSTOM ONLY: maximum CONSECUTIVE skips before a forced full run. "
                            "0 disables reuse. Higher = faster but riskier.\n"
                            "---\n"
                            "ТОЛЬКО Custom: максимум пропусков ПОДРЯД перед принудительным полным "
                            "прогоном. 0 отключает переиспользование. Выше = быстрее, но рискованнее."
                        ),
                    },
                ),
                "video_metric_stride": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": (
                            "Strided sampling of the video tensor for the metric (every Nth element). "
                            "Higher = cheaper metric, slightly noisier.\n"
                            "---\n"
                            "Strided-выборка видео-тензора для метрики (каждый N-й элемент). "
                            "Больше = дешевле метрика, чуть шумнее."
                        ),
                    },
                ),
                "audio_metric_stride": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": (
                            "Strided sampling of the audio tensor for the metric (every Nth element). "
                            "Denser than video to protect voice.\n"
                            "---\n"
                            "Strided-выборка аудио-тензора для метрики (каждый N-й элемент). "
                            "Плотнее видео для защиты голоса."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Where the metric snapshots/deltas are computed. auto = on the tensor's device; "
                            "cpu avoids GPU sync but copies data; cuda forces GPU.\n"
                            "---\n"
                            "Где считаются снимки/дельты метрики. auto = на устройстве тензора; cpu "
                            "избегает GPU-синхронизации, но копирует данные; cuda принудительно GPU."
                        ),
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Per-step logging (RUN with real wall time / SKIP with both deltas / final run "
                            "summary).\n"
                            "---\n"
                            "Логирование каждого шага (RUN с реальным временем / SKIP с обеими дельтами / "
                            "итоговая сводка)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_cache"
    CATEGORY = "AGSoft/Model"
    DESCRIPTION = (
        "🚀 AGSoft MiniMax H3 Cache.\n"
        "Accelerates MiniMax H3 sampling by reusing the cached full-step model output when BOTH video "
        "and audio sampled relative deltas are below their thresholds (audio veto).\n"
        "Profiles: Balanced / Visual Fast / Dialogue Safe / Action Safe / Custom; cache window by "
        "start/end percent of the sigma schedule, warmup steps, consecutive-skip limit, strided fp32 "
        "metrics per stream.\n"
        "---\n"
        "🚀 AGSoft MiniMax H3 Cache.\n"
        "Ускоряет сэмплинг MiniMax H3, переиспользуя закэшированный выход полного шага, когда ОБЕ "
        "относительные дельты (видео и аудио) ниже своих порогов (вето по аудио).\n"
        "Профили: Balanced / Visual Fast / Dialogue Safe / Action Safe / Custom; окно кэша по "
        "start/end процентам расписания сигм, warmup-шаги, лимит пропусков подряд, strided fp32-метрики "
        "по каждому потоку.\n"
    )

    def apply_cache(self, model, profile, video_threshold, audio_threshold, start_percent, end_percent,
                    warmup_steps, max_steps, video_metric_stride, audio_metric_stride,
                    device, verbose):
        m = model.clone()
        dm = m.model.diffusion_model

        if not getattr(dm, "_agsoft_patched", False):
            # save the ORIGINAL bound method, then shadow _forward on the instance
            # сохраняем ОРИГИНАЛЬНЫЙ bound-метод и подменяем _forward на инстансе
            dm._agsoft_orig_forward = dm._forward
            dm._forward = types.MethodType(_cached_forward, dm)
            dm._agsoft_patched = True
            logger.info("[AGSoft MiniMax H3 Cache] _forward wrapper installed.")

        if profile in PROFILES:
            p = PROFILES[profile]
            v_thr, a_thr = p["v"], p["a"]
            start_p, end_p = p["start"], p["end"]
            warmup, max_st = p["warmup"], p["max_steps"]
            src = f"profile={profile} (preset values; manual widgets ignored)"
        else:
            v_thr, a_thr = float(video_threshold), float(audio_threshold)
            start_p, end_p = float(start_percent), float(end_percent)
            warmup, max_st = int(warmup_steps), int(max_steps)
            src = "profile=Custom (manual widgets)"

        dm._agsoft_state = {
            "video_threshold": v_thr,
            "audio_threshold": a_thr,
            "start_percent": min(start_p, end_p),
            "end_percent": max(start_p, end_p),
            "warmup_steps": warmup,
            "max_steps": max_st,
            "video_metric_stride": max(1, int(video_metric_stride)),
            "audio_metric_stride": max(1, int(audio_metric_stride)),
            "device": device,
            "verbose": bool(verbose),
            "last_sigma": None,
            "last_sigmas_id": None,
            "last_video_snap": None,
            "last_audio_snap": None,
            "last_out": None,
            "consec": 0,
            "run_total": 0,
            "run_skipped": 0,
            "run_time_total": 0.0,
            "time_saved": 0.0,
            "avg_run": 0.0,
            "summarized": False,
        }
        logger.info(
            f"[AGSoft MiniMax H3 Cache] {src}: v_thr={v_thr:.4f}, a_thr={a_thr:.4f}, "
            f"window=[{min(start_p, end_p):.2f}, {max(start_p, end_p):.2f}], warmup={warmup}, "
            f"max_steps={max_st}, strides=({video_metric_stride},{audio_metric_stride}), "
            f"device={device}, verbose={verbose}"
        )
        return (m,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {
    "AGSoftMiniMaxH3Cache": AGSoftMiniMaxH3Cache
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftMiniMaxH3Cache": "🚀AGSoft MiniMax H3 Cache"
}