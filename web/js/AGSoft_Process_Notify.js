// ==============================================================================
// AGSoft_Process_Notify.js
// ==============================================================================
// JS для ноды 🔔AGSoft Process Notify.
// Проигрывает по onExecuted: встроенный WebAudio-пресет (beep/ding/chime/
// success/alarm/pop) или файл из sounds/ (через /agsoft/sound).
// loop / громкость / задержка. Кнопки Test и Stop.
//
// JS for the 🔔AGSoft Process Notify node.
// Plays on onExecuted: a built-in WebAudio preset or a file from sounds/.
// loop / volume / delay. Test and Stop buttons.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Process Notify] JS loaded (synth presets + robust stop + fitted buttons)");

let audioCtx = null;
let activeAudios = [];   // все играющие файлы / all playing file audios
let activeTimers = [];   // все таймеры (loop + delay) / all timers (loop + delay)

const getCtx = () => {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
};

const tone = (ctx, freq, delay, dur, type, vol) => {
    const t = ctx.currentTime + delay;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, vol), t + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc["connect"](gain)["connect"](ctx.destination);
    osc.start(t);
    osc.stop(t + dur + 0.05);
};

const SYNTHS = {
    "beep":    (v, c) => tone(c, 880, 0, 0.35, "sine", v),
    "ding":    (v, c) => { tone(c, 1318.5, 0, 0.6, "sine", v); tone(c, 1975.5, 0, 0.4, "sine", v * 0.3); },
    "chime":   (v, c) => [880, 1174.7, 1568].forEach((f, i) => tone(c, f, i * 0.12, 0.5, "sine", v)),
    "success": (v, c) => { tone(c, 523.3, 0, 0.15, "triangle", v); tone(c, 784, 0.15, 0.3, "triangle", v); },
    "alarm":   (v, c) => { for (let i = 0; i < 3; i++) { tone(c, 950, i * 0.3, 0.14, "square", v * 0.4); tone(c, 1200, i * 0.3 + 0.15, 0.14, "square", v * 0.4); } },
    "pop":     (v, c) => {
        const t = c.currentTime;
        const osc = c.createOscillator(); const g = c.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(400, t);
        osc.frequency.exponentialRampToValueAtTime(80, t + 0.12);
        g.gain.setValueAtTime(v, t);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
        osc["connect"](g)["connect"](c.destination);
        osc.start(t); osc.stop(t + 0.16);
    },
};

// ------------------------------------------------------------------------------
// Полная остановка ВСЕХ аудио и таймеров. / Full stop of ALL audios and timers.
// ------------------------------------------------------------------------------
const stopAll = () => {
    for (const a of activeAudios) {
        try { a.pause(); a.currentTime = 0; } catch (e) { /* ignore */ }
    }
    activeAudios = [];
    for (const t of activeTimers) {
        try { clearInterval(t); } catch (e) { /* ignore */ }
    }
    activeTimers = [];
};

const playNotify = (cfg) => {
    stopAll();
    const doPlay = () => {
        const vol = (cfg && cfg.volume != null) ? cfg.volume : 1.0;
        const loop = !!(cfg && cfg.loop);
        const sound = (cfg && cfg.sound) || "beep";

        if (SYNTHS[sound]) {
            SYNTHS[sound](vol, getCtx());
            if (loop) activeTimers.push(setInterval(() => SYNTHS[sound](vol, getCtx()), 1200));
            return;
        }

        const url = api.apiURL("/agsoft/sound?name=" + encodeURIComponent(sound));
        const audio = new Audio(url);
        audio.volume = vol;
        audio.loop = loop;
        activeAudios.push(audio);
        audio.play().catch((e) => console.warn("[AGSoft Process Notify] play failed:", e));
    };

    const delay = (cfg && cfg.delay) ? cfg.delay * 1000 : 0;
    if (delay > 0) activeTimers.push(setTimeout(doPlay, delay));
    else doPlay();
};

app.registerExtension({
    name: "AGSoft.ProcessNotify",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoft_Process_Notify") return;

        const readCfg = () => {
            const w = (name) => {
                const widget = node.widgets && node.widgets.find(x => x.name === name);
                return widget ? widget.value : null;
            };
            return {
                sound: w("sound_file") || "beep",
                volume: w("volume") ?? 1.0,
                loop: !!w("loop"),
                delay: w("delay") ?? 0.0,
            };
        };

        // ---- Панель кнопок: резиновая, вписывается в ширину ноды ----
        const wrap = document.createElement("div");
        wrap.style.width = "100%";
        wrap.style.maxWidth = "100%";
        wrap.style.display = "flex";
        wrap.style.gap = "6px";
        wrap.style.height = "28px";
        wrap.style.boxSizing = "border-box";

        const baseStyle = {
            flex: "1 1 0",
            minWidth: "0",
            height: "28px",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "0 6px",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontFamily: "sans-serif",
            fontSize: "12px",
            fontWeight: "600",
            lineHeight: "1",
            boxSizing: "border-box",
            overflow: "hidden",
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
        };

        const testBtn = document.createElement("button");
        testBtn.textContent = "🔔 Test";
        Object.assign(testBtn.style, baseStyle, { background: "#26a" });
        testBtn.onclick = (e) => { e.preventDefault(); playNotify(readCfg()); };

        const stopBtn = document.createElement("button");
        stopBtn.textContent = "⏹ Stop";
        Object.assign(stopBtn.style, baseStyle, { background: "#a33" });
        stopBtn.onclick = (e) => { e.preventDefault(); stopAll(); };

        wrap.appendChild(testBtn);
        wrap.appendChild(stopBtn);

        const btnWidget = node.addDOMWidget("agsoft_notify_buttons", "div", wrap,
            { serialize: false, hideOnZoom: false });
        btnWidget.computeSize = (width) => [Math.max(120, (width || 240) - 8), 34];

        const origExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            const list = output && output.agsoft_notify;
            if (list && list.length) playNotify(list[0]);
        };

        // ---- ФИКС: на создании кнопки вылезают — принудительно пересчитываем
        // ---- размер ноды (то же, что при ручном ресайзе).
        // ---- FIX: on creation buttons overflow — force a node size recompute
        // ---- (same as manual resize).
        const relayout = () => {
            const w = Math.max(node.size?.[0] || 0, 240);
            node.setSize([w, node.computeSize()[1]]);
            node.setDirtyCanvas(true, true);
        };
        relayout();
        setTimeout(relayout, 0);
        requestAnimationFrame(relayout);
    }
});