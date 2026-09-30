// ==============================================================================
// agsoft_minimax_base.js
// Версия / Version: v09.60
// Фронтенд ноды 🎬AGSoft MiniMax Base / Frontend layer for the Base node.
// • Живая инфострока внизу (18px): ⚙ W×H • MP • ~формат • кадры • точные сек;
//   ширина строки = ширине ноды; кадры раньше секунд — обрезается хвост секунд.
//   Live info line at the bottom (18px): ⚙ W×H • MP • ~aspect • frames • exact
//   sec; bar width = node width; frames come before seconds.
// • Автоскрытие неиспользуемых виджетов по режимам; скрытые виджеты выкинуты
//   из карты попаданий (y = -9999), чтобы линки вставали точно в видимые.
//   Unused widgets hide per mode; hidden widgets are removed from the drop
//   hit map (y = -9999) so links land exactly on visible widgets.
// • Ведомая нода (вход copy_options) схлопывает калькулятор и читает значения
//   мастера; изменения мастера обновляют всех ведомых.
//   A follower (copy_options input) collapses the calculator and reads the
//   master's values; master changes refresh all followers.
// • Значения подключённых нод читаются из источника (виджет "value", reroute).
//   Linked values are read from the origin node ("value" widget, reroute).
// • Всё под try/catch; входы и линки не трогаются.
//   Everything under try/catch; inputs and links are never touched.
//
// Автор / Author: AGSoft
// Дата / Date: 30.09.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";

console.log("[AGSoft MiniMax Base] JS extension loaded v09.60");


// Имя сокета цепочки (переименовать здесь, как COPY_SOCKET в py) /
// Chain socket name (rename here, same as COPY_SOCKET in py)
const COPY_SOCKET = "copy_options";

const BAR_H = 18;

// Координата-«небытие» для скрытых виджетов / off-screen coordinate for hidden widgets
const Y_OFF = -9999;

const HOOK_WIDGETS = [
    "mode", "preset", "invert_orientation", "custom_width", "custom_height",
    "megapixels_value", "aspect_ratio", "frame_count_source",
    "length_seconds", "frame_count",
];

// ==============================================================================
// Утилиты / utilities
// ==============================================================================

function fit32(v) {
    return Math.max(32, Math.ceil(v / 32) * 32);
}

function fit17n5(v) {
    let x = Math.max(5, Math.round(Number(v) || 5));
    while (x % 17 !== 5) x += 1;
    return x;
}

function getWidget(node, name) {
    if (!node || !node.widgets) return null;
    return node.widgets.find((w) => w.name === name) || null;
}

function linkOf(node, input) {
    if (!node.graph || !node.graph.links) return null;
    const links = node.graph.links;
    const lk = links.get ? links.get(input.link) : links[input.link];
    return lk || null;
}

function toBool(v) {
    return v === true || v === "true" || v === 1;
}

function isLinkedInput(node, name) {
    const inp = node.inputs ? node.inputs.find((i) => i.name === name) : null;
    return !!(inp && inp.link !== null && inp.link !== undefined);
}

// ==============================================================================
// Чтение значений с учётом линков / value reading with link resolution
// ==============================================================================

function hookSourceWidget(widget, target) {
    if (!widget || !target) return;
    if (!widget.__ag_base_targets) widget.__ag_base_targets = new Set();
    if (widget.__ag_base_targets.has(target)) return;
    widget.__ag_base_targets.add(target);

    const old = widget.callback;
    widget.callback = function (v) {
        try { if (old) old.apply(this, arguments); } catch (e) {}
        for (const n of widget.__ag_base_targets) scheduleRefresh(n);
    };
}

function resolveLinkedValue(node, name, depth, target) {
    if (depth > 6) return undefined;
    const inp = node.inputs ? node.inputs.find((i) => i.name === name) : null;
    if (!inp || inp.link === null || inp.link === undefined) return undefined;

    const lk = linkOf(node, inp);
    if (!lk) return undefined;
    const src = node.graph && node.graph.getNodeById
        ? node.graph.getNodeById(lk.origin_id) : null;
    if (!src) return undefined;

    // Нода-константа с виджетом "value" (AGSoft Float/Integer, Primitive...).
    const w = src.widgets ? src.widgets.find((x) => x.name === "value") : null;
    if (w) {
        hookSourceWidget(w, target);
        return w.value;
    }

    // Сквозная нода (reroute и т.п.) — идём дальше по её входу.
    if (src.inputs && src.inputs.length && src.outputs && src.outputs.length) {
        const pin = src.inputs[0];
        if (pin && pin.link !== null && pin.link !== undefined) {
            return resolveLinkedValue(src, pin.name, depth + 1, target);
        }
    }

    return undefined;
}

function getValue(node, name, target) {
    try {
        const linked = resolveLinkedValue(node, name, 0, target || node);
        if (linked !== undefined) return linked;
    } catch (e) {}
    const w = getWidget(node, name);
    return w ? w.value : null;
}

// ==============================================================================
// Цепочка copy_options / copy_options chain
// ==============================================================================

function chainMaster(node, depth) {
    try {
        if (!node || depth > 8) return null;
        const inp = node.inputs ? node.inputs.find((i) => i.name === COPY_SOCKET) : null;
        if (!inp || inp.link === null || inp.link === undefined) return null;
        const lk = linkOf(node, inp);
        if (!lk) return null;
        const src = node.graph && node.graph.getNodeById
            ? node.graph.getNodeById(lk.origin_id) : null;
        if (!src || src.comfyClass !== "AGSoft_MiniMax_Base") return null;
        return chainMaster(src, depth + 1) || src;
    } catch (e) {
        return null;
    }
}

function refreshDownstream(node, depth) {
    try {
        if (!node || depth > 8) return;
        const out = node.outputs ? node.outputs.find((o) => o.name === COPY_SOCKET) : null;
        if (!out || !out.links) return;
        for (const lid of out.links) {
            const lk = linkOf(node, { link: lid });
            if (!lk) continue;
            const dst = node.graph.getNodeById(lk.target_id);
            if (!dst || dst.comfyClass !== "AGSoft_MiniMax_Base") continue;
            updateWidgets(dst);
            updateBarText(dst);
            syncBarWidth(dst);
            if (dst.setDirtyCanvas) dst.setDirtyCanvas(true, true);
            refreshDownstream(dst, depth + 1);
        }
    } catch (e) {}
}

// ==============================================================================
// Форматы / formats
// ==============================================================================

function parsePreset(value) {
    const m = String(value || "").match(/(\d+)\s*[×x]\s*(\d+)/i);
    return m ? [parseInt(m[1], 10), parseInt(m[2], 10)] : [1280, 704];
}

function parseRatio(value) {
    const m = String(value || "16:9").replace(/\s/g, "").match(/(\d+):(\d+)/);
    if (!m) return [16, 9];
    return [Math.max(1, parseInt(m[1], 10)), Math.max(1, parseInt(m[2], 10))];
}

const RATIO_CANDIDATES = [[1, 1], [5, 4], [4, 3], [3, 2], [16, 9], [21, 9], [2, 1]];

function approximateRatio(w, h) {
    const target = Math.max(w, h) / Math.max(1, Math.min(w, h));
    let best = RATIO_CANDIDATES[0];
    let bestDiff = Infinity;
    for (const [a, b] of RATIO_CANDIDATES) {
        const d = Math.abs(a / b - target);
        if (d < bestDiff) { bestDiff = d; best = [a, b]; }
    }
    const [a, b] = best;
    return w >= h ? `${a}:${b}` : `${b}:${a}`;
}

function num(v) {
    return Number(v.toFixed(2)).toString();
}

// ТОЧНЫЕ секунды: 12 знаков после запятой, хвостовые нули убраны.
// Exact seconds: 12 decimals, trailing zeros stripped.
function exactSec(v) {
    return Number(v.toFixed(12)).toString();
}

// ==============================================================================
// Расчёт инфостроки / info line computation
// ==============================================================================

function computeInfo(node) {
    try {
        const master = chainMaster(node, 0) || node;
        const g = (n) => getValue(master, n, node);

        const mode = String(g("mode") || "Preset");
        const invert = toBool(g("invert_orientation"));
        let w = 1280, h = 704;

        if (mode === "Preset") {
            [w, h] = parsePreset(g("preset"));
            if (invert) { const t = w; w = h; h = t; }
            w = fit32(w); h = fit32(h);
        } else if (mode === "Custom") {
            w = fit32(g("custom_width") || 64);
            h = fit32(g("custom_height") || 64);
            if (invert) { const t = w; w = h; h = t; }
            w = Math.max(64, w); h = Math.max(64, h);
        } else if (mode === "Megapixels") {
            const mp = parseFloat(g("megapixels_value")) || 1.0;
            const [wr, hr] = parseRatio(g("aspect_ratio"));
            const x = Math.sqrt((mp * 1000000) / (wr * hr));
            w = Math.max(64, fit32(Math.round(wr * x)));
            h = Math.max(64, fit32(Math.round(hr * x)));
            if (invert) { const t = w; w = h; h = t; }
        }

        const src = String(g("frame_count_source") || "Seconds");
        let total;
        if (src === "Frames") {
            total = fit17n5(g("frame_count") || 5);
        } else {
            total = fit17n5(Math.round((parseFloat(g("length_seconds")) || 0) * 24));
        }
        total = Math.max(5, total);

        const sec = exactSec(total / 24);
        const mp = num((w * h) / 1000000) + "MP";
        const ratio = "~" + approximateRatio(w, h);

        // Кадры ПЕРЕД секундами: при нехватке места обрежутся секунды.
        // Frames BEFORE seconds: a narrow node truncates seconds, not frames.
        return `⚙ ${w}×${h} • ${mp} • ${ratio} • ${total}f • ${sec}s`;
    } catch (e) {
        return "⚙ AGSoft MiniMax Base";
    }
}

// ==============================================================================
// Инфострока / info bar (v09.59: ширина = ширине ноды, не вылезает)
// ==============================================================================

function syncBarWidth(node) {
    try {
        const el = node.__ag_base_bar_el;
        if (!el) return;
        const nw = node.size && node.size[0] ? node.size[0] : 320;
        el.style.width = Math.max(60, Math.floor(nw) - 16) + "px";
    } catch (e) {}
}

function ensureBarWidget(node) {
    if (!node) return null;
    if (node.__ag_base_bar_w) return node.__ag_base_bar_w;

    const div = document.createElement("div");
    div.style.cssText =
        "height:" + BAR_H + "px;line-height:" + BAR_H + "px;" +
        "text-align:center;background:#263238;color:#9ecbdd;font:10px monospace;" +
        "border-radius:8px;overflow:hidden;" +
        "white-space:nowrap;box-sizing:border-box;pointer-events:none;";
    div.textContent = computeInfo(node);

    let w = null;
    try {
        w = node.addDOMWidget("agsoft_minimax_base_info", "div", div, { serialize: false });
    } catch (e) {
        console.warn("[AGSoft MiniMax Base] addDOMWidget failed:", e);
        return null;
    }
    w.computeSize = function (width) { return [width || 0, BAR_H]; };

    // Инфострока всегда внизу / the info bar is always at the bottom.
    try {
        const idx = node.widgets.indexOf(w);
        if (idx > -1) { node.widgets.splice(idx, 1); node.widgets.push(w); }
    } catch (e) {}

    node.__ag_base_bar_w = w;
    node.__ag_base_bar_el = div;
    syncBarWidth(node);
    return w;
}

function updateBarText(node) {
    if (!node) return;
    if (!node.__ag_base_bar_el) { ensureBarWidget(node); return; }
    const t = computeInfo(node);
    if (node.__ag_base_bar_el.textContent !== t) node.__ag_base_bar_el.textContent = t;
}

// ==============================================================================
// Автоскрытие + восстановление виджетов / auto-hiding AND restoring
// ==============================================================================

function fitNodeHeight(node) {
    try {
        const canvas = app && app.canvas;
        if (canvas && canvas.resizing_node === node) return;
        if (!node.computeSize) return;
        const size = node.computeSize();
        if (!size || size.length < 2) return;
        const curW = node.size && node.size.length === 2 ? node.size[0] : size[0];
        const newW = Math.max(size[0], curW);
        if (node.setSize) node.setSize([newW, size[1]]);
        else node.size = [newW, size[1]];
    } catch (e) {}
}

// v09.59: настоящее скрытие + сброс координат скрытых, чтобы карта попаданий
// дропа не видела скрытые виджеты и линки вставали точно в видимые.
// Real hiding + coordinate reset so the drop hit map never sees hidden
// widgets and links land exactly on visible ones.
function setHidden(w, hidden) {
    try {
        if (w.hidden !== hidden) w.hidden = hidden;
        if (hidden) {
            w.y = Y_OFF;
        } else if (w.y === Y_OFF) {
            w.y = 0;
        }
    } catch (e) {}
}

function updateWidgets(node) {
    try {
        if (!node || !node.widgets) return false;

        const master = chainMaster(node, 0);
        const chained = !!master;
        const eff = master || node;

        const mode = String(getValue(eff, "mode", node) || "Preset");
        const src = String(getValue(eff, "frame_count_source", node) || "Seconds");

        // invert_orientation активен во ВСЕХ режимах.
        // invert_orientation is active in ALL modes.
        let rules;
        if (chained) {
            rules = {};
            for (const n of HOOK_WIDGETS) rules[n] = false;
        } else {
            rules = {
                mode: true,
                preset: mode === "Preset",
                invert_orientation: true,
                custom_width: mode === "Custom",
                custom_height: mode === "Custom",
                megapixels_value: mode === "Megapixels",
                aspect_ratio: mode === "Megapixels",
                frame_count_source: true,
                length_seconds: src === "Seconds",
                frame_count: src === "Frames",
            };
        }

        let changed = false;
        for (const w of node.widgets) {
            if (!w) continue;
            if (!Object.prototype.hasOwnProperty.call(rules, w.name)) continue;
            // Виджеты с подключённым линком не скрываем.
            // Widgets with a connected link are never hidden.
            if (isLinkedInput(node, w.name)) { setHidden(w, false); continue; }

            const hidden = !rules[w.name];
            if (w.hidden !== hidden || (hidden && w.y !== Y_OFF)) {
                setHidden(w, hidden);
                changed = true;
            }
        }
        if (changed) fitNodeHeight(node);
        return changed;
    } catch (e) {
        return false;
    }
}

// ==============================================================================
// Refresh
// ==============================================================================

function refresh(node) {
    try {
        if (!node) return;
        ensureBarWidget(node);
        updateWidgets(node);
        updateBarText(node);
        syncBarWidth(node);
        refreshDownstream(node, 0);
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    } catch (e) {}
}

function scheduleRefresh(node) {
    if (!node || node.__ag_base_pending) return;
    node.__ag_base_pending = true;
    setTimeout(() => {
        node.__ag_base_pending = false;
        refresh(node);
    }, 0);
}

// ==============================================================================
// Extension
// ==============================================================================

app.registerExtension({
    name: "AGSoft.MiniMaxBase",

    async nodeCreated(node) {
        try {
            if (!node || node.comfyClass !== "AGSoft_MiniMax_Base") return;

            ensureBarWidget(node);
            refresh(node);

            // Хуки виджетов калькулятора / calculator widget hooks
            for (const name of HOOK_WIDGETS) {
                const w = getWidget(node, name);
                if (!w) continue;
                const old = w.callback;
                w.callback = function (v) {
                    try { if (old) old.apply(this, arguments); } catch (e) {}
                    refresh(node); // себя + ведомых по copy_options / self + followers
                };
            }

            // Ресайз ноды → подгон ширины инфостроки.
            // Node resize → info bar width sync.
            const origResize = node.onResize;
            node.onResize = function (size) {
                try { if (origResize) origResize.apply(this, arguments); } catch (e) {}
                syncBarWidth(this);
            };

            // Подключение/отключение линков → пересчёт. ВХОДЫ И ЛИНКИ НЕ ТРОГАЕМ.
            // Linking/unlinking → recompute. Inputs and links are NEVER touched.
            const origConn = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info) {
                try { if (origConn) origConn.apply(this, arguments); } catch (e) {}
                scheduleRefresh(this);
            };

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                try { if (origOnConfigure) origOnConfigure.apply(this, arguments); } catch (e) {}
                ensureBarWidget(this);
                scheduleRefresh(this);
            };

            setTimeout(() => refresh(node), 0);
        } catch (e) {
            console.warn("[AGSoft MiniMax Base] nodeCreated skipped:", e);
        }
    },
});