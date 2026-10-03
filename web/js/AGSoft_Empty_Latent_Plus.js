// ==============================================================================
// AGSoft_Empty_Latent_Plus.js
// ==============================================================================
// Frontend extension for the 🧊AGSoft Empty Latent Plus node.
// Version: v1.03
//
// WHAT IT DOES
// • Live info line at the bottom of the node (DOM widget):
//     - before execution: "🧊 W×H (ratio) · VAE auto" (a preview of the final
//       pixel size, computed with the same formulas as the Python side);
//     - after execution: the exact result from Python
//       "🧊 W×H (ratio) → latent lw×lh · ch N".
// • The line is glued to the last visible widget; the node bottom always
//   coincides with the line bottom; the line width follows the node width.
// • The last execution result is stored in the workflow JSON (serialize /
//   onConfigure), so it survives ComfyUI tab switching and workflow reloads.
//   It resets to "VAE auto" only when a parameter is changed manually.
// • Auto-hiding of unused widgets per size_mode (Ratio / Preset / Custom):
//     - Ratio  : ratio_preset, custom_ratio (only for 'custom'), base,
//                base_value / megapixels_value (depending on the anchor);
//     - Preset : size_preset;
//     - Custom : width, height.
//   invert_orientation, multiple, rounding, batch_size are always visible.
// • Widgets converted to inputs are never hidden while a link is connected.
// • Hidden widgets are removed from the drop hit map (y = -9999), so links
//   land exactly on the visible sockets; inputs and links are never modified.
// • The client-side math is a 1:1 copy of the Python side: ratio parsing,
//   banker's rounding, multiple alignment.
//
// ------------------------------------------------------------------------------
//
// Фронтенд-расширение для ноды 🧊AGSoft Empty Latent Plus.
// Версия: v1.03
//
// ЧТО ДЕЛАЕТ
// • Живая инфострока внизу ноды (DOM-виджет):
//     - до выполнения: "🧊 W×H (пропорция) · VAE auto" (превью итогового
//       пиксельного размера по тем же формулам, что и в Python);
//     - после выполнения: точный результат из Python
//       "🧊 W×H (пропорция) → latent lw×lh · ch N".
// • Строка приклеена к последнему видимому виджету; низ ноды всегда совпадает
//   с низом строки; ширина строки следует за шириной ноды.
// • Последний результат сохраняется в JSON воркфлоу (serialize / onConfigure)
//   и переживает переключение вкладок и перезагрузку воркфлоу.
//   Сброс в "VAE auto" — только при ручном изменении параметров.
// • Автоскрытие неиспользуемых виджетов по size_mode (Ratio / Preset / Custom):
//     - Ratio  : ratio_preset, custom_ratio (только для 'custom'), base,
//                base_value / megapixels_value (в зависимости от якоря);
//     - Preset : size_preset;
//     - Custom : width, height.
//   invert_orientation, multiple, rounding, batch_size видны всегда.
// • Виджеты, конвертированные в инпуты, не скрываются, пока подключён линк.
// • Скрытые виджеты выкинуты из карты попаданий (y = -9999), поэтому линки
//   встают точно в видимые сокеты; входы и линки никогда не изменяются.
// • Клиентская математика — копия Python 1:1: парсинг пропорций,
//   банковское округление, выравнивание под кратность.
//
// Author: AGSoft
// Date: 02.10.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS_NAME = "AGSoft_Empty_Latent_Plus";
const BAR_H = 20;
const Y_OFF = -9999;

const HOOK_WIDGETS = [
    "size_mode", "size_preset", "ratio_preset", "custom_ratio",
    "base", "base_value", "megapixels_value", "width", "height",
    "multiple", "rounding", "invert_orientation", "batch_size"
];

// ==============================================================================
// Утилиты
// ==============================================================================

function getWidget(node, name) {
    if (!node || !node.widgets) return null;
    return node.widgets.find((w) => w.name === name) || null;
}

function isLinkedInput(node, name) {
    const inp = node.inputs ? node.inputs.find((i) => i.name === name) : null;
    return !!(inp && inp.link !== null && inp.link !== undefined);
}

function toBool(v) {
    return v === true || v === "true" || v === 1 || v === "True";
}

// ==============================================================================
// Математика (1:1 с Python)
// ==============================================================================

const NUM_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
function parseNum(p) {
    const t = String(p).trim();
    if (!NUM_RE.test(t)) throw new Error("format");
    return parseFloat(t);
}

function parseRatio(text) {
    let s = String(text ?? "").trim();
    for (const sep of ["x", "X", "/", ",", ";"]) s = s.split(sep).join(":");
    const parts = s.split(":").filter(p => p.trim() !== "");
    if (parts.length !== 2) throw new Error("format");
    const w = parseNum(parts[0]), h = parseNum(parts[1]);
    if (w <= 0 || h <= 0) throw new Error("positive");
    return [w, h];
}

function parseSizePreset(name) {
    const s = String(name ?? "").trim();
    if (!s.includes("-") || !s.includes("(")) throw new Error("format");
    const dims = s.split("-")[1].split("(")[0].replace("×", "x").toLowerCase();
    const ratio = s.substring(s.lastIndexOf("(") + 1, s.lastIndexOf(")"));
    const [w, h] = dims.split("x").map(v => parseInt(v, 10));
    if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) throw new Error("positive");
    return [w, h, ratio];
}

function pyRound(v) {
    const f = Math.floor(v), d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
}

function toMultiple(value, multiple, mode) {
    const v = Math.max(1.0, Number(value));
    const m = parseInt(multiple, 10) || 1;
    if (m <= 1) return pyRound(v);
    if (mode === "floor") return Math.max(m, Math.floor(v / m) * m);
    if (mode === "ceil") return Math.max(m, Math.ceil(v / m) * m);
    return Math.max(m, pyRound(v / m) * m);
}

// ==============================================================================
// Расчёт инфостроки (до выполнения)
// ==============================================================================

function computeInfo(node) {
    try {
        const g = (n) => { const w = getWidget(node, n); return w ? w.value : null; };
        const sizeMode = String(g("size_mode") || "Ratio");
        let w = 1024, h = 1024, label = "1:1";

        if (sizeMode === "Custom") {
            w = parseFloat(g("width")) || 1024;
            h = parseFloat(g("height")) || 1024;
            label = "custom";
        } else if (sizeMode === "Preset") {
            const p = parseSizePreset(g("size_preset"));
            w = p[0]; h = p[1]; label = p[2];
        } else {
            const ratioPreset = String(g("ratio_preset") || "1:1");
            const ratioText = ratioPreset === "custom" ? String(g("custom_ratio") || "1:1").trim() : ratioPreset;
            const [rw, rh] = parseRatio(ratioText);
            const base = String(g("base") || "width");
            const val = parseFloat(g("base_value")) || 1024;
            const ratio = rw / rh;

            if (base === "width") { w = val; h = val * rh / rw; }
            else if (base === "height") { h = val; w = val * rw / rh; }
            else if (base === "longest") { if (rw >= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
            else if (base === "shortest") { if (rw <= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
            else {
                const mpp = parseFloat(g("megapixels_value")) || 1.0;
                const t = mpp * 1e6; w = Math.sqrt(t * ratio); h = Math.sqrt(t / ratio);
            }
            label = ratioText;
        }

        if (toBool(g("invert_orientation"))) {
            const t = w; w = h; h = t;
            label += " ↕";
        }

        const multiple = parseInt(g("multiple"), 10) || 64;
        const rounding = String(g("rounding") || "round");

        const W = toMultiple(w, multiple, rounding);
        const H = toMultiple(h, multiple, rounding);

        return `🧊 ${W}×${H} (${label}) · VAE auto`;
    } catch (e) {
        return "🧊 AGSoft Empty Latent Plus";
    }
}

// ==============================================================================
// Инфострока (DOM Widget)
// ==============================================================================

function syncBarWidth(node) {
    try {
        const el = node.__ag_bar_el;
        if (!el) return;
        const nw = node.size && node.size[0] ? node.size[0] : 320;
        el.style.width = Math.max(60, Math.floor(nw) - 16) + "px";
    } catch (e) {}
}

function ensureBarWidget(node) {
    if (!node) return null;
    if (node.__ag_bar_w) return node.__ag_bar_w;

    const div = document.createElement("div");
    div.style.cssText =
        "height:" + BAR_H + "px;line-height:" + BAR_H + "px;" +
        "text-align:center;background:#263238;color:#9ecbdd;font:bold 11px monospace;" +
        "border-radius:8px;overflow:hidden;" +
        "white-space:nowrap;box-sizing:border-box;pointer-events:none;" +
        "margin-top:-2px;";
    div.textContent = computeInfo(node);

    let w = null;
    try {
        w = node.addDOMWidget("agsoft_empty_latent_info", "div", div, { serialize: false });
    } catch (e) {
        return null;
    }
    w.computeSize = function (width) { return [width || 0, BAR_H]; };

    try {
        const idx = node.widgets.indexOf(w);
        if (idx > -1) { node.widgets.splice(idx, 1); node.widgets.push(w); }
    } catch (e) {}

    node.__ag_bar_w = w;
    node.__ag_bar_el = div;
    syncBarWidth(node);
    return w;
}

function updateBarText(node) {
    if (!node) return;
    if (!node.__ag_bar_el) { ensureBarWidget(node); return; }

    const t = node._ag_executed_display
        ? ("🧊 " + node._ag_executed_display)
        : computeInfo(node);

    if (node.__ag_bar_el.textContent !== t) node.__ag_bar_el.textContent = t;
}

// ==============================================================================
// Автоскрытие виджетов
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

        const g = (n) => { const w = getWidget(node, n); return w ? w.value : null; };
        const sizeMode = String(g("size_mode") || "Ratio");
        const ratioPreset = String(g("ratio_preset") || "1:1");
        const base = String(g("base") || "width");

        const rules = {
            size_mode: true,
            size_preset: sizeMode === "Preset",
            ratio_preset: sizeMode === "Ratio",
            custom_ratio: sizeMode === "Ratio" && ratioPreset === "custom",
            base: sizeMode === "Ratio",
            base_value: sizeMode === "Ratio" && base !== "megapixels",
            megapixels_value: sizeMode === "Ratio" && base === "megapixels",
            width: sizeMode === "Custom",
            height: sizeMode === "Custom",
            multiple: true,
            rounding: true,
            invert_orientation: true,
            batch_size: true,
        };

        let changed = false;
        for (const w of node.widgets) {
            if (!w) continue;
            if (!Object.prototype.hasOwnProperty.call(rules, w.name)) continue;

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
// Refresh & Executed Hook
// ==============================================================================

function refresh(node) {
    try {
        if (!node) return;
        ensureBarWidget(node);
        updateWidgets(node);
        updateBarText(node);
        syncBarWidth(node);
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    } catch (e) {}
}

function scheduleRefresh(node) {
    if (!node || node.__ag_pending) return;
    node.__ag_pending = true;
    setTimeout(() => {
        node.__ag_pending = false;
        refresh(node);
    }, 0);
}

// Выполнение ноды: сохраняем точную строку из Python
api.addEventListener("executed", (evt) => {
    try {
        const detail = evt.detail;
        if (!detail) return;
        const node = app.graph.getNodeById(detail.node);
        if (!node || node.comfyClass !== NODE_CLASS_NAME) return;

        const out = detail.output;
        if (!out) return;

        let display = Array.isArray(out.display) ? out.display[0] : out.display;
        if (typeof display === "string" && display.trim() !== "") {
            node._ag_executed_display = display;

            if (!node.__ag_bar_el) ensureBarWidget(node);
            node.__ag_bar_el.textContent = "🧊 " + display;
        }
    } catch (e) {}
});

// ==============================================================================
// Extension
// ==============================================================================

app.registerExtension({
    name: "AGSoft.EmptyLatentPlus",

    async nodeCreated(node) {
        try {
            if (!node || node.comfyClass !== NODE_CLASS_NAME) return;

            node._ag_executed_display = null;

            ensureBarWidget(node);
            refresh(node);

            for (const name of HOOK_WIDGETS) {
                const w = getWidget(node, name);
                if (!w) continue;
                const old = w.callback;
                w.callback = function (v) {
                    try { if (old) old.apply(this, arguments); } catch (e) {}
                    // Параметр изменён вручную -> старая генерация неактуальна
                    node._ag_executed_display = null;
                    scheduleRefresh(node);
                };
            }

            // Персистентность: строка последней генерации в JSON воркфлоу.
            const origSerialize = node.serialize;
            const baseSerialize = () => origSerialize.call(node);
            node.serialize = function () {
                const o = baseSerialize();
                try {
                    if (this._ag_executed_display) {
                        o.ag_executed_display = this._ag_executed_display;
                    }
                } catch (e) {}
                return o;
            };

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                try {
                    if (info && typeof info.ag_executed_display === "string" && info.ag_executed_display.trim() !== "") {
                        this._ag_executed_display = info.ag_executed_display;
                    }
                } catch (e) {}

                try { if (origOnConfigure) origOnConfigure.apply(this, arguments); } catch (e) {}
                ensureBarWidget(this);
                scheduleRefresh(this);
            };

            const origResize = node.onResize;
            node.onResize = function (size) {
                try { if (origResize) origResize.apply(this, arguments); } catch (e) {}
                syncBarWidth(this);
            };

            const origConn = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info) {
                try { if (origConn) origConn.apply(this, arguments); } catch (e) {}
                scheduleRefresh(this);
            };

            setTimeout(() => refresh(node), 0);
        } catch (e) {
            console.warn("[AGSoft Empty Latent Plus] nodeCreated skipped:", e);
        }
    },
});