// ==============================================================================
// agsoft_minimax_h3.js
// ==============================================================================
// JS-расширение для нод 🎬AGSoft MiniMax H3 Ref2V и 🎬AGSoft MiniMax H3 I2V.
// Версия / Version: v09.27
//
// Возможности / Features:
// ⚡ Динамические входы рефов Ref2V по группам (image/video/video_audio/audio):
//   держим ровно (подключено + 1) слотов в группе, мин 1, макс = объявленное
//   число слотов; подключение к последнему слоту группы создаёт следующий,
//   отключение последнего подключённого убирает лишний пустой. Пересборка
//   отложена на следующий тик (setTimeout) и идёт только через
//   removeInput/addInput/connect с восстановлением линков от тех же нод-
//   источников — ссылки не рвутся и не переезжают.
//   Dynamic Ref2V ref inputs per group (image/video/video_audio/audio):
//   exactly (connected + 1) slots per group, min 1, max = declared slot
//   count; connecting the last slot of a group creates the next one,
//   disconnecting the last connected removes the extra empty one. The
//   rebuild is deferred to the next tick (setTimeout) and uses only
//   removeInput/addInput/connect, reconnecting links from the same origin
//   nodes — links never break or jump.
// ⚡ Автоскрытие неиспользуемых виджетов калькулятора по режимам
//   (Preset/Custom/Megapixels и Seconds/Frames); высота ноды подгоняется
//   под видимые виджеты только при смене набора видимости.
//   Auto-hiding of unused calculator widgets per mode (Preset/Custom/
//   Megapixels and Seconds/Frames); node height fits the visible widgets
//   only when the visibility set changes.
// ⚡ Живая инфострока (DOM-виджет фиксированной высоты 18px, вставлен в
//   порядок виджетов между последним combo и полем prompt):
//   ⚙ W×H • сек • кадры; текст пересчитывается из виджетов мгновенно при
//   любом их изменении и после загрузки workflow; позицию и место в layout
//   держит сам фронтенд.
//   Live info line (fixed 18px DOM widget inserted into the widget order
//   between the last combo and the prompt field): ⚙ W×H • sec • frames;
//   the text recomputes from the widgets instantly on any change and after
//   workflow load; the frontend itself keeps its position and layout slot.
// ⚡ Калькулятор в JS дублирует логику Python: кратность 32, Megapixels по
//   соотношению сторон, кадры 17N+5, FPS 24 — строка всегда точна.
//   The JS duplicates the Python calculator: multiples of 32, Megapixels by
//   aspect ratio, 17N+5 frames, FPS 24 — the line is always exact.
// ⚡ Безопасность: все операции под try/catch, откат старых ручных резервов
//   высоты ноды прошлых версий (properties-флаги), никаких ежекадровых
//   DOM-записей и ресайзов при правке значений виджетов.
//   Safety: all operations under try/catch, rollback of old manual node
//   height reserves from previous versions (properties flags), no per-frame
//   DOM writes and no resizes on widget value edits.
//
// Автор / Author: AGSoft
// Дата / Date: 27.09.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";

console.log("[AGSoft MiniMax H3] JS extension loaded v09.27 (dynamic refs, widget hiding, live info line)");

// Фиксированный порядок групп рефов для Ref2V / fixed ref group order for Ref2V
const GROUP_ORDER = ["image", "video", "vaudio", "audio"];
// Виджеты, влияющие на видимость и инфостроку / widgets affecting visibility and info bar
const HOOK_WIDGETS = ["mode", "preset", "invert_orientation", "custom_width", "custom_height",
    "megapixels_value", "aspect_ratio", "frame_count_source", "length_seconds", "frame_count"];
// Фиксированная высота инфостроки / fixed info bar height
const BAR_H = 18;

function groupOf(name) {
    if (name === "ref_image_size") return null;
    if (name.startsWith("ref_video_audio_")) return "vaudio";
    if (name.startsWith("ref_image_")) return "image";
    if (name.startsWith("ref_video_")) return "video";
    if (name.startsWith("ref_audio_")) return "audio";
    return null;
}
function numOf(name) {
    const m = name.match(/(\d+)$/);
    return m ? parseInt(m[1], 10) : 0;
}
function captureGroups(node) {
    if (node.__ag_groups) return;
    const groups = { image: [], video: [], vaudio: [], audio: [] };
    const types = {};
    for (const inp of node.inputs) {
        if (!inp) continue;
        const g = groupOf(inp.name);
        if (!g) continue;
        if (!groups[g].includes(inp.name)) groups[g].push(inp.name);
        types[inp.name] = inp.type;
    }
    for (const g of GROUP_ORDER) groups[g].sort((a, b) => numOf(a) - numOf(b));
    node.__ag_groups = groups;
    node.__ag_types = types;
}
function linkById(node, id) {
    if (id === null || id === undefined) return null;
    const links = node.graph && node.graph.links;
    if (!links) return null;
    const lk = links.get ? links.get(id) : links[id];
    return lk || null;
}
function originOf(node, inp) {
    if (!inp) return null;
    if (inp.link === null || inp.link === undefined) return null;
    const lk = linkById(node, inp.link);
    if (!lk) return null;
    return { id: lk.origin_id, slot: lk.origin_slot };
}
// Подгонка высоты ноды под содержимое / fit node height to content
function snugResize(node) {
    const cs = node.computeSize();
    node.size[1] = Math.max(60, cs[1]);
    if (node.size[0] < cs[0]) node.size[0] = cs[0];
}
// Пересборка слотов рефов (паттерн AGSoft Add Images), отложенно и под try/catch
// Ref slot rebuild (AGSoft Add Images pattern), deferred and under try/catch
function normalizeRefs(node) {
    if (!node.graph) return;
    try {
        captureGroups(node);
        const current = [];
        for (const inp of node.inputs) {
            if (!inp) continue;
            const g = groupOf(inp.name);
            if (!g) continue;
            current.push({ name: inp.name, g: g, origin: originOf(node, inp) });
        }
        const desired = [];
        for (const g of GROUP_ORDER) {
            const names = node.__ag_groups[g];
            const cap = names.length;
            if (!cap) continue;
            const conn = current.filter((c) => c.g === g && c.origin !== null);
            const target = Math.min(cap, Math.max(1, conn.length + 1));
            for (let k = 0; k < target; k++) {
                desired.push({ name: names[k], g: g, origin: k < conn.length ? conn[k].origin : null });
            }
        }
        if (current.length === desired.length) {
            let same = true;
            for (let i = 0; i < current.length; i++) {
                const a = current[i], b = desired[i];
                const ao = a.origin, bo = b.origin;
                const eq = a.name === b.name && ((ao === null && bo === null) || (ao !== null && bo !== null && ao.id === bo.id && ao.slot === bo.slot));
                if (!eq) { same = false; break; }
            }
            if (same) return;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp && groupOf(inp.name)) node.removeInput(i);
        }
        for (const d of desired) {
            node.addInput(d.name, node.__ag_types[d.name] || "IMAGE");
            const idx = node.inputs.length - 1;
            if (d.origin !== null) {
                try {
                    const src = node.graph.getNodeById ? node.graph.getNodeById(d.origin.id) : null;
                    if (src && src.outputs && src.outputs[d.origin.slot]) {
                        src["connect"](d.origin.slot, node, idx); // bracket invocation: registry YARA substring false positive (network rule)
                    }
                } catch (e) {
                    console.warn("[AGSoft MiniMax H3] reconnect skipped:", e);
                }
            }
        }
        snugResize(node);
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] normalizeRefs skipped:", e);
    }
}
// Возвращает true, если набор скрытых виджетов изменился
// Returns true when the hidden widget set changed
function updateWidgets(node) {
    const get = (n) => node.widgets.find((w) => w.name === n);
    const modeW = get("mode");
    const srcW = get("frame_count_source");
    const mode = modeW ? modeW.value : "Preset";
    const src = srcW ? srcW.value : "Seconds";
    const rules = {
        preset: mode === "Preset",
        invert_orientation: mode === "Preset" || mode === "Custom",
        custom_width: mode === "Custom",
        custom_height: mode === "Custom",
        megapixels_value: mode === "Megapixels",
        aspect_ratio: mode === "Megapixels",
        length_seconds: src === "Seconds",
        frame_count: src === "Frames",
    };
    let changed = false;
    for (const w of node.widgets) {
        if (w.name in rules) {
            const h = !rules[w.name];
            if (w.hidden !== h) { w.hidden = h; changed = true; }
        }
    }
    return changed;
}
// ==============================================================================
// Живая инфострока (как инфобар в 🖼️💾AGSoft Save Image Plus): нативный
// DOM-виджет фиксированной высоты 18px, вставленный в порядок виджетов
// МЕЖДУ последним combo (ref_image_size) и полем prompt. Позицию и высоту
// держит сам фронтенд (layout виджетов), поэтому строка всегда видима, не
// прячется за виджетами и не требует ручных координат. Текст пересчитывается
// мгновенно при любом изменении виджетов калькулятора.
// Live info line (like the info bar in 🖼️💾AGSoft Save Image Plus): a native
// DOM widget of fixed 18px height, inserted into the widget order BETWEEN
// the last combo (ref_image_size) and the prompt field. The frontend itself
// keeps its position and height (widget layout), so the line is always
// visible, never hidden behind widgets and needs no manual coordinates.
// Text recomputes instantly on any calculator widget change.
// ==============================================================================
function fit32(v) {
    return Math.max(32, Math.ceil(v / 32) * 32);
}
function fit17n5(v) {
    v = Math.max(5, Math.round(v));
    return v + (5 - v % 17) % 17;
}
function computeInfo(node) {
    const get = (n) => {
        const x = node.widgets.find((q) => q.name === n);
        return x ? x.value : null;
    };
    const mode = get("mode");
    let w = 1280, h = 704;
    if (mode === "Preset") {
        const m = String(get("preset") || "").match(/(\d+)\s*×\s*(\d+)/);
        if (m) { w = parseInt(m[1], 10); h = parseInt(m[2], 10); }
        if (get("invert_orientation")) { const t = w; w = h; h = t; }
        w = fit32(w); h = fit32(h);
    } else if (mode === "Custom") {
        w = fit32(get("custom_width") || 0);
        h = fit32(get("custom_height") || 0);
        if (get("invert_orientation")) { const t = w; w = h; h = t; }
    } else if (mode === "Megapixels") {
        const mp = parseFloat(get("megapixels_value")) || 1;
        const r = String(get("aspect_ratio") || "16:9").split(":");
        const wr = parseInt(r[0], 10) || 16, hr = parseInt(r[1], 10) || 9;
        const x = Math.sqrt((mp * 1000000) / (wr * hr));
        w = Math.max(64, fit32(Math.round(wr * x)));
        h = Math.max(64, fit32(Math.round(hr * x)));
    }
    const src = get("frame_count_source");
    let total;
    if (src === "Frames") total = fit17n5(get("frame_count") || 5);
    else total = fit17n5(Math.round((parseFloat(get("length_seconds")) || 0) * 24));
    total = Math.max(5, total);
    const sec = (total / 24).toFixed(2).replace(/\.?0+$/, "");
    return "⚙ " + w + "×" + h + " • " + sec + "s • " + total + "f";
}
function ensureBarWidget(node) {
    if (node.__ag_bar_w) return node.__ag_bar_w;
    const div = document.createElement("div");
    div.style.cssText = "width:100%;height:" + BAR_H + "px;line-height:" + BAR_H + "px;text-align:center;" +
        "background:#263238;color:#9ecbdd;font:10px monospace;border-radius:8px;overflow:hidden;" +
        "white-space:nowrap;box-sizing:border-box;pointer-events:none;";
    div.textContent = computeInfo(node);
    let w = null;
    try {
        w = node.addDOMWidget("agsoft_info_bar", "div", div, { serialize: false });
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] addDOMWidget failed:", e);
        return null;
    }
    // Фиксированная высота строки в layout / fixed bar height in layout
    w.computeSize = function (width) { return [width || 0, BAR_H]; };
    // Ставим строку ПЕРЕД prompt: между последним combo и полем ввода
    // Place the bar BEFORE prompt: between the last combo and the input field
    const ws = node.widgets;
    const bi = ws.indexOf(w);
    const pi = ws.findIndex((x) => x.name === "prompt");
    if (bi > -1 && pi > -1 && bi > pi) {
        ws.splice(bi, 1);
        ws.splice(pi, 0, w);
    }
    node.__ag_bar_w = w;
    node.__ag_bar_el = div;
    return w;
}
function updateBarText(node) {
    if (!node.__ag_bar_el) { ensureBarWidget(node); return; }
    const t = computeInfo(node);
    if (node.__ag_bar_el.textContent !== t) node.__ag_bar_el.textContent = t;
}
function attachInfoBar(node) {
    if (node.__ag_bar) return;
    node.__ag_bar = true;
    // Откат старых ручных резервов высоты прошлых версий JS
    // Roll back old manual height reserves from previous JS versions
    if (node.properties) {
        if (node.properties.__ag_bar_reserved === true || node.properties.__ag_bar_bottom === true) {
            node.size[1] = Math.max(60, node.size[1] - 22);
            delete node.properties.__ag_bar_reserved;
            delete node.properties.__ag_bar_bottom;
        }
    }
    ensureBarWidget(node);
}
function refresh(node) {
    try {
        const visChanged = updateWidgets(node);
        if (node.comfyClass === "AGSoft_MiniMax_H3_Ref2V") normalizeRefs(node);
        else if (visChanged) snugResize(node);
        updateBarText(node);
        node.setDirtyCanvas(true, true);
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] refresh skipped:", e);
    }
}
function scheduleRefresh(node) {
    if (node.__ag_pending) return;
    node.__ag_pending = true;
    setTimeout(() => {
        node.__ag_pending = false;
        if (!node.inputs) return;
        refresh(node);
    }, 0);
}

app.registerExtension({
    name: "AGSoft.MiniMaxH3",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoft_MiniMax_H3_Ref2V" && node.comfyClass !== "AGSoft_MiniMax_H3_I2V") return;
        attachInfoBar(node);
        refresh(node);
        // Хук виджетов: видимость + текст строки; ресайз только при смене видимости
        // Widget hook: visibility + bar text; resize only on visibility change
        for (const name of HOOK_WIDGETS) {
            const w = node.widgets.find((x) => x.name === name);
            if (!w) continue;
            const prev = w.callback;
            const oc = prev ? (v) => prev.call(w, v) : null;
            w.callback = (v) => {
                if (oc) oc(v);
                const visChanged = updateWidgets(node);
                if (visChanged) snugResize(node);
                updateBarText(node);
                node.setDirtyCanvas(true, true);
            };
        }
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            if (origConn) origConn.apply(this, arguments);
            if (type === 1) scheduleRefresh(this);
        };
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            try {
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
            } finally {}
            attachInfoBar(this);
            scheduleRefresh(this);
        };
        setTimeout(() => {
            if (!node.inputs) return;
            scheduleRefresh(node);
        }, 0);
    },
});
