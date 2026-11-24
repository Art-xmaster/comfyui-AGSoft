// ==============================================================================
// AGSoft_Empty_Latent_Plus.js
// ==============================================================================
// Нода: 🧊AGSoft Empty Latent Plus
// Версия: v11.24
//
// WHAT IT DOES
// • Status line format: "🧊 1344×768 • 1.03MP • ~16:9 • VAE auto" before a run
//   and the exact Python display after it:
//   "🧊 1344×768 • 1.03MP • ~16:9 • latent 168×96 • ch 4".
//   The ratio is the nearest common ratio of the final pixels; "~" marks an
//   approximation. Before a run the VAE factor is unknown on the client, so the
//   preview shows the multiple-aligned size and the "VAE auto" marker.
// • Custom mode: the arrow step of width/height equals 'multiple' (64 -> ±64,
//   1 -> ±1). Both the legacy step scale and the modern step2 are written, so
//   Classic and Nodes 2.0 behave the same. When 'multiple' changes, the current
//   values are pulled to the nearest multiple.
// • Auto-hiding of unused widgets per size_mode (Ratio / Preset / Custom):
//     - Ratio  : ratio_preset, custom_ratio (only for 'custom'), base,
//                base_value / megapixels_value (depending on the anchor);
//     - Preset : size_preset;
//     - Custom : width, height.
//   size_mode, multiple, rounding, invert_orientation, batch_size are always
//   visible.
// • Hiding is applied through four independent channels: widget.hidden,
//   widget.options.hidden, computeSize() -> [0, -4] and y = -9999 (out of the
//   drop hit map). The state is re-applied on every reconciliation pass.
// • Widgets converted to inputs are never hidden while a link is connected.
//   Inputs and links are never modified.
// • Initialization is fault-isolated: hooks and the reconciliation timer are
//   installed before the DOM widget, every step runs in its own guard.
// • The size line is a DOM widget with a hard fixed height (flex:0 0 auto,
//   min-height = max-height, getMinHeight / getMaxHeight), styled like the
//   status lines of AGSoft Image Crop Plus.
// • Node height comes from the native computeSize (visible widgets only);
//   node width is fully user-controlled, it can grow and shrink.
// • The last result is stored in the workflow JSON (serialize / onConfigure)
//   and resets only when a parameter is changed manually.
// • The client math is a 1:1 copy of Python: ratio parsing, preset parsing,
//   banker's rounding, multiple alignment, nearest-ratio label.
//
// ------------------------------------------------------------------------------
//
// ЧТО ДЕЛАЕТ
// • Формат статусной строки: "🧊 1344×768 • 1.03MP • ~16:9 • VAE auto" до запуска
//   и точный display из Python после него:
//   "🧊 1344×768 • 1.03MP • ~16:9 • latent 168×96 • ch 4".
//   Пропорция — ближайшая типовая к итоговым пикселям; "~" означает приближение.
//   До запуска фактор VAE клиенту неизвестен, поэтому превью показывает размер
//   после выравнивания по кратности и маркер "VAE auto".
// • Режим Custom: шаг стрелок width/height равен 'multiple' (64 -> ±64,
//   1 -> ±1). Пишется и legacy-масштаб step, и современный step2, поэтому
//   Classic и Nodes 2.0 ведут себя одинаково. При смене 'multiple' текущие
//   значения подтягиваются к ближайшей кратности.
// • Автоскрытие неиспользуемых виджетов по size_mode (Ratio / Preset / Custom):
//     - Ratio  : ratio_preset, custom_ratio (только для 'custom'), base,
//                base_value / megapixels_value (в зависимости от якоря);
//     - Preset : size_preset;
//     - Custom : width, height.
//   size_mode, multiple, rounding, invert_orientation, batch_size видны всегда.
// • Скрытие применяется по четырём независимым каналам: widget.hidden,
//   widget.options.hidden, computeSize() -> [0, -4] и y = -9999 (вне карты
//   попаданий). Состояние переприменяется на каждом проходе сверки.
// • Виджеты, конвертированные в инпуты, не скрываются, пока подключён линк.
//   Входы и линки никогда не изменяются.
// • Инициализация изолирована по сбоям: хуки и таймер сверки ставятся до
//   DOM-виджета, каждый шаг в своей защите.
// • Строка размера — DOM-виджет с жёстко фиксированной высотой (flex:0 0 auto,
//   min-height = max-height, getMinHeight / getMaxHeight), оформлена как
//   статусные строки AGSoft Image Crop Plus.
// • Высота ноды берётся из нативного computeSize (только видимые виджеты);
//   ширина полностью пользовательская — и вверх, и вниз.
// • Последний результат сохраняется в JSON воркфлоу (serialize / onConfigure)
//   и сбрасывается только при ручном изменении параметров.
// • Клиентская математика — копия Python 1:1: разбор пропорций и пресетов,
//   банковское округление, выравнивание, метка ближайшей пропорции.
//
// Автор / Author: AGSoft
// Дата / Date: 24.11.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Empty Latent Plus] JS extension loaded v11.24 (MP status line, auto-hide, Nodes 2.0)");

const NODE_CLASS_NAME = "AGSoft_Empty_Latent_Plus";
const BAR_H = 20;
const Y_OFF = -9999;
const MIN_W = 160;
const TICK_MS = 200;
const HOOK_WIDGETS = [
    "size_mode", "size_preset", "ratio_preset", "custom_ratio", "base",
    "base_value", "megapixels_value", "width", "height", "multiple",
    "rounding", "invert_orientation", "batch_size",
];
const RATIO_LABELS = [
    [1, 1], [5, 4], [4, 3], [3, 2], [16, 10], [16, 9], [2, 1], [21, 9],
    [4, 5], [3, 4], [2, 3], [5, 8], [9, 16], [1, 2], [9, 21],
];

// ==============================================================================
// Утилиты / Utils
// ==============================================================================
// Каждый шаг изолирован: сбой одного не отключает остальные.
// Every step is isolated: a failure of one never disables the others.
function safe(fn, label) {
    try {
        return fn();
    } catch (e) {
        console.warn("[AGSoft Empty Latent Plus] " + label + " skipped:", e);
        return null;
    }
}

function getWidget(node, name) {
    if (!node || !node.widgets) return null;
    return node.widgets.find((w) => w.name === name) || null;
}

function widgetValue(node, name) {
    const w = getWidget(node, name);
    return w ? w.value : null;
}

function isLinkedInput(node, name) {
    const inp = node.inputs ? node.inputs.find((i) => i.name === name) : null;
    return !!(inp && inp.link !== null && inp.link !== undefined);
}

function toBool(v) {
    return v === true || v === "true" || v === 1 || v === "True";
}

function valuesSignature(node) {
    const parts = HOOK_WIDGETS.map((n) => String(widgetValue(node, n) ?? ""));
    parts.push(node._ag_executed_display || "");
    return parts.join("|");
}

// ==============================================================================
// Математика (1:1 с Python) / Math (1:1 with Python)
// ==============================================================================
const NUM_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;

function parseNum(p) {
    const t = String(p).trim();
    if (!NUM_RE.test(t)) throw new Error("format");
    const v = parseFloat(t);
    if (!isFinite(v)) throw new Error("format");
    return v;
}

function parseRatio(text) {
    let s = String(text ?? "").trim();
    for (const sep of ["x", "X", "/", ",", ";"]) s = s.split(sep).join(":");
    const parts = s.split(":").filter((p) => p.trim() !== "");
    if (parts.length !== 2) throw new Error("format");
    const w = parseNum(parts[0]);
    const h = parseNum(parts[1]);
    if (w <= 0 || h <= 0) throw new Error("positive");
    return [w, h];
}

function parseSizePreset(name) {
    const s = String(name ?? "").trim();
    if (!s.includes("-") || !s.includes("(")) throw new Error("format");
    const dims = s.slice(s.indexOf("-") + 1).split("(")[0]
        .replace("×", "x").toLowerCase().trim();
    const idx = dims.indexOf("x");
    if (idx < 0) throw new Error("format");
    const w = Math.trunc(parseNum(dims.slice(0, idx)));
    const h = Math.trunc(parseNum(dims.slice(idx + 1)));
    if (w <= 0 || h <= 0) throw new Error("positive");
    return [w, h];
}

function pyRound(v) {
    const f = Math.floor(v);
    const d = v - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
}

function toMultiple(value, multiple, mode) {
    const v = Math.max(1.0, Number(value));
    const m = Math.max(1, parseInt(multiple, 10) || 1);
    if (m <= 1) return pyRound(v);
    if (mode === "floor") return Math.max(m, Math.floor(v / m) * m);
    if (mode === "ceil") return Math.max(m, Math.ceil(v / m) * m);
    return Math.max(m, pyRound(v / m) * m);
}

// Ближайшая типовая пропорция, "~" — приближение.
// Nearest common ratio, "~" marks an approximation.
function ratioText(width, height) {
    const target = width / height;
    let best = RATIO_LABELS[0];
    let bestErr = null;
    for (const pair of RATIO_LABELS) {
        const err = Math.abs(pair[0] / pair[1] - target);
        if (bestErr === null || err < bestErr - 1e-12) {
            bestErr = err;
            best = pair;
        }
    }
    const prefix = bestErr < 1e-9 ? "" : "~";
    return `${prefix}${best[0]}:${best[1]}`;
}

// Клиентский расчёт display (те же формулы, что в generate(), без фактора VAE).
// Client-side display (same formulas as generate(), without the VAE factor).
function computeInfo(node) {
    try {
        const sizeMode = String(widgetValue(node, "size_mode") || "Ratio");
        let w = 1024;
        let h = 1024;
        if (sizeMode === "Custom") {
            w = parseFloat(widgetValue(node, "width")) || 1024;
            h = parseFloat(widgetValue(node, "height")) || 1024;
        } else if (sizeMode === "Preset") {
            const p = parseSizePreset(widgetValue(node, "size_preset"));
            w = p[0];
            h = p[1];
        } else {
            const ratioPreset = String(widgetValue(node, "ratio_preset") || "1:1");
            const raw = ratioPreset === "custom"
                ? String(widgetValue(node, "custom_ratio") || "1:1").trim()
                : ratioPreset;
            const [rw, rh] = parseRatio(raw);
            const ratioSrc = rw / rh;
            const base = String(widgetValue(node, "base") || "megapixels");
            const val = parseFloat(widgetValue(node, "base_value")) || 1024;
            if (base === "width") {
                w = val;
                h = val / ratioSrc;
            } else if (base === "height") {
                w = val * ratioSrc;
                h = val;
            } else if (base === "longest") {
                if (rw >= rh) {
                    w = val;
                    h = val / ratioSrc;
                } else {
                    w = val * ratioSrc;
                    h = val;
                }
            } else if (base === "shortest") {
                if (rw <= rh) {
                    w = val;
                    h = val / ratioSrc;
                } else {
                    w = val * ratioSrc;
                    h = val;
                }
            } else {
                const t = (parseFloat(widgetValue(node, "megapixels_value")) || 1.0) * 1e6;
                w = Math.sqrt(t * ratioSrc);
                h = Math.sqrt(t / ratioSrc);
            }
        }
        if (toBool(widgetValue(node, "invert_orientation"))) {
            const t = w;
            w = h;
            h = t;
        }
        const multiple = parseInt(widgetValue(node, "multiple"), 10) || 64;
        const rounding = String(widgetValue(node, "rounding") || "round");
        const W = toMultiple(w, multiple, rounding);
        const H = toMultiple(h, multiple, rounding);
        const mp = ((W * H) / 1e6).toFixed(2);
        return { ok: true, text: `🧊 ${W}×${H} • ${mp}MP • ${ratioText(W, H)} • VAE auto` };
    } catch (e) {
        const msg = (e && e.message === "positive")
            ? "❌ values must be > 0 / значения должны быть > 0"
            : "❌ invalid ratio / неверная пропорция (W:H)";
        return { ok: false, text: msg };
    }
}

// ==============================================================================
// Шаг стрелок width/height = multiple / Arrow step of width/height = multiple
// ==============================================================================
function syncSteps(node) {
    const m = Math.max(1, parseInt(widgetValue(node, "multiple"), 10) || 1);
    if (node.__ag_step === m) return false;
    node.__ag_step = m;
    for (const name of ["width", "height"]) {
        safe(() => {
            const w = getWidget(node, name);
            if (!w || !w.options) return;
            // Legacy-канвас хранит step ×10, Nodes 2.0 читает step2.
            // The legacy canvas stores step ×10, Nodes 2.0 reads step2.
            if (w.__ag_step_scale === undefined) {
                const raw = Number(w.options.step);
                w.__ag_step_scale = raw >= 40 ? 10 : 1;
            }
            w.options.step = m * w.__ag_step_scale;
            w.options.step2 = m;
            w.options.min = Math.max(1, m);
            const cur = Number(w.value);
            if (isFinite(cur)) {
                const snapped = Math.max(m, Math.round(cur / m) * m);
                if (snapped !== cur) w.value = snapped;
            }
        }, "step " + name);
    }
    return true;
}

// ==============================================================================
// Автоскрытие виджетов / Auto-hiding widgets
// ==============================================================================
function zeroSize() {
    return [0, -4];
}

// Скрытый виджет не резервирует место ни в одном из рендереров.
// A hidden widget reserves no space in either renderer.
function applySize(w, hidden) {
    if (hidden) {
        if (!Object.prototype.hasOwnProperty.call(w, "__ag_cs")) {
            w.__ag_cs = typeof w.computeSize === "function" ? w.computeSize : null;
        }
        w.computeSize = zeroSize;
    } else if (Object.prototype.hasOwnProperty.call(w, "__ag_cs")) {
        if (w.__ag_cs) w.computeSize = w.__ag_cs;
        else delete w.computeSize;
        delete w.__ag_cs;
    }
}

function setHidden(w, hidden) {
    safe(() => {
        w.hidden = hidden;
    }, "widget hidden flag");
    safe(() => {
        if (w.options) w.options.hidden = hidden;
    }, "widget options flag");
    safe(() => {
        if (hidden) {
            w.y = Y_OFF;
        } else if (w.y === Y_OFF) {
            w.y = 0;
        }
    }, "widget hit map");
    safe(() => applySize(w, hidden), "widget size");
}

function visibilityRules(node) {
    const sizeMode = String(widgetValue(node, "size_mode") || "Ratio");
    const ratioPreset = String(widgetValue(node, "ratio_preset") || "1:1");
    const base = String(widgetValue(node, "base") || "megapixels");
    return {
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
}

// Состояние переприменяется всегда: внешняя перекладка не может его сбросить.
// The state is always re-applied: an external re-layout cannot reset it.
function updateWidgets(node) {
    if (!node || !node.widgets) return false;
    const rules = visibilityRules(node);
    let changed = false;
    for (const w of node.widgets) {
        if (!w) continue;
        if (!Object.prototype.hasOwnProperty.call(rules, w.name)) continue;
        const want = isLinkedInput(node, w.name) ? false : !rules[w.name];
        const was = !!w.hidden;
        setHidden(w, want);
        if (was !== want) changed = true;
    }
    return changed;
}

// ==============================================================================
// Строка размера (DOM Widget, фиксированная высота)
// Size line (DOM widget, hard fixed height)
// ==============================================================================
function applyTheme(node, ok) {
    const el = node.__ag_bar_el;
    if (!el) return;
    const base = node.bgcolor || "#2b2b2b";
    const accent = ok ? (node.color || "#5b6ee1") : "#a04040";
    el.style.setProperty("--ags-bg", base);
    el.style.setProperty("--ags-accent", accent);
    el.style.setProperty("--ags-text", ok ? "#cdd3ff" : "#ffb4b4");
    el.style.background = "rgba(0,0,0,0.30)";
    el.style.background = "color-mix(in srgb, var(--ags-bg) 60%, #000 40%)";
    el.style.border = "1px solid rgba(255,255,255,0.10)";
    el.style.border = "1px solid color-mix(in srgb, var(--ags-accent) 55%, #000 45%)";
    el.style.color = ok ? "#cdd3ff" : "#ffb4b4";
    el.style.color = "var(--ags-text)";
}

function ensureBar(node) {
    if (!node || node.__ag_bar_w) return node.__ag_bar_w || null;
    if (typeof node.addDOMWidget !== "function") return null;
    const div = document.createElement("div");
    div.style.cssText = [
        "width:100%",
        "height:" + BAR_H + "px",
        "min-height:" + BAR_H + "px",
        "max-height:" + BAR_H + "px",
        "flex:0 0 auto",
        "font:600 11px/" + (BAR_H - 2) + "px system-ui,sans-serif",
        "font-variant-numeric:tabular-nums",
        "text-align:center",
        "color:#cdd3ff",
        "background:rgba(0,0,0,0.30)",
        "border:1px solid rgba(255,255,255,0.10)",
        "border-radius:6px",
        "box-sizing:border-box",
        "padding:0 6px",
        "overflow:hidden",
        "white-space:nowrap",
        "text-overflow:ellipsis",
        "pointer-events:none",
    ].join(";") + ";";
    div.textContent = "🧊 AGSoft Empty Latent Plus";
    const w = safe(() => node.addDOMWidget("agsoft_empty_latent_info", "div", div, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => BAR_H,
        getMaxHeight: () => BAR_H,
    }), "info widget");
    if (!w) return null;
    safe(() => {
        w.height = BAR_H;
    }, "info height");
    safe(() => {
        w.computeSize = function (width) {
            return [width || 0, BAR_H];
        };
    }, "info computeSize");
    safe(() => {
        const idx = node.widgets.indexOf(w);
        if (idx > -1) {
            node.widgets.splice(idx, 1);
            node.widgets.push(w);
        }
    }, "info reorder");
    node.__ag_bar_w = w;
    node.__ag_bar_el = div;
    node.__ag_bar_ok = null;
    safe(() => applyTheme(node, true), "info theme");
    return w;
}

function updateBar(node) {
    const el = node.__ag_bar_el;
    if (!el) return;
    let text;
    let ok = true;
    if (node._ag_executed_display) {
        text = "🧊 " + node._ag_executed_display;
    } else {
        const r = computeInfo(node);
        text = r.text;
        ok = r.ok;
    }
    if (el.textContent !== text) {
        el.textContent = text;
        el.title = text;
    }
    if (node.__ag_bar_ok !== ok) {
        node.__ag_bar_ok = ok;
        safe(() => applyTheme(node, ok), "info theme");
    }
}

// ==============================================================================
// Размеры ноды: ширина пользовательская, высота по содержимому
// Node sizing: user width, content height
// ==============================================================================
function fallbackHeight(node) {
    const lg = window.LiteGraph;
    const slotH = (lg && lg.NODE_SLOT_HEIGHT) || 20;
    const widgetH = (lg && lg.NODE_WIDGET_HEIGHT) || 20;
    const rows = Math.max(
        node.inputs ? node.inputs.length : 0,
        node.outputs ? node.outputs.length : 0,
        1
    );
    let h = rows * slotH + 12;
    for (const w of node.widgets || []) {
        if (!w || w.hidden) continue;
        let wh = widgetH;
        if (typeof w.computeSize === "function") {
            const cs = w.computeSize(node.size ? node.size[0] : 320);
            if (Array.isArray(cs) && isFinite(cs[1])) wh = cs[1];
        } else if (isFinite(w.height)) {
            wh = w.height;
        }
        h += wh + 4;
    }
    return h;
}

function hookComputeSize(node) {
    if (node.__ag_size_hooked) return;
    node.__ag_size_hooked = true;
    const native = node.computeSize;
    node.__ag_native_compute = native;
    node.computeSize = function (out) {
        const s = native ? native.call(this, out) : [MIN_W, 60];
        const w = Array.isArray(s) && isFinite(s[0]) ? s[0] : MIN_W;
        const h = Array.isArray(s) && isFinite(s[1]) ? s[1] : 60;
        const cur = this.size && this.size[0] ? this.size[0] : w;
        // Ширину не форсим вверх: пользователь может и уменьшать ноду.
        // Width is never forced up: the user can shrink the node as well.
        return [Math.max(MIN_W, Math.min(w, cur)), h];
    };
}

function measureHeight(node) {
    const native = node.__ag_native_compute;
    if (typeof native === "function") {
        const s = safe(() => native.call(node), "native size");
        if (Array.isArray(s) && isFinite(s[1]) && s[1] > 0) return Math.ceil(s[1]);
    }
    return Math.ceil(fallbackHeight(node));
}

function fitNode(node) {
    if (!node || node.__ag_fitting || !node.size) return;
    if (node.flags && node.flags.collapsed) return;
    const h = measureHeight(node);
    const w = Math.max(MIN_W, node.size[0] || MIN_W);
    if (Math.abs(node.size[1] - h) < 1 && Math.abs(node.size[0] - w) < 1) return;
    node.__ag_fitting = true;
    safe(() => {
        if (node.setSize) node.setSize([w, h]);
        else node.size = [w, h];
    }, "height fit");
    node.__ag_fitting = false;
}

// ==============================================================================
// Refresh и сверка / Refresh and reconciliation
// ==============================================================================
function refresh(node) {
    if (!node) return;
    // Автоскрытие идёт первым и не зависит от инфостроки.
    // Auto-hiding runs first and does not depend on the info line.
    safe(() => updateWidgets(node), "auto-hide");
    safe(() => syncSteps(node), "step sync");
    safe(() => ensureBar(node), "info create");
    safe(() => updateBar(node), "info text");
    safe(() => fitNode(node), "size fit");
    node.__ag_sig = valuesSignature(node);
    safe(() => {
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, "canvas dirty");
}

function scheduleRefresh(node) {
    if (!node || node.__ag_pending) return;
    node.__ag_pending = true;
    setTimeout(() => {
        node.__ag_pending = false;
        refresh(node);
    }, 0);
}

function tick(node) {
    if (!node || !node.widgets) return;
    const changed = safe(() => updateWidgets(node), "auto-hide tick") === true;
    const stepped = safe(() => syncSteps(node), "step tick") === true;
    const sig = valuesSignature(node);
    if (changed || stepped || sig !== node.__ag_sig) {
        node.__ag_sig = sig;
        safe(() => updateBar(node), "info tick");
        safe(() => {
            if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        }, "canvas dirty");
    }
    safe(() => fitNode(node), "size tick");
}

api.addEventListener("executed", (evt) => {
    safe(() => {
        const detail = evt.detail;
        if (!detail) return;
        const node = app.graph.getNodeById(detail.node);
        if (!node || node.comfyClass !== NODE_CLASS_NAME) return;
        const out = detail.output;
        if (!out) return;
        const display = Array.isArray(out.display) ? out.display[0] : out.display;
        if (typeof display !== "string" || display.trim() === "") return;
        node._ag_executed_display = display;
        updateBar(node);
        node.__ag_sig = valuesSignature(node);
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, "executed hook");
});

// ==============================================================================
// Хуки / Hooks
// ==============================================================================
function hookWidgets(node) {
    for (const name of HOOK_WIDGETS) {
        safe(() => {
            const w = getWidget(node, name);
            if (!w || w.__ag_hooked) return;
            w.__ag_hooked = true;
            const old = w.callback;
            w.callback = function (v) {
                safe(() => {
                    if (old) old.apply(this, arguments);
                }, "widget callback");
                node._ag_executed_display = null;
                scheduleRefresh(node);
            };
        }, "hook " + name);
    }
}

function hookLifecycle(node) {
    safe(() => {
        const orig = node.serialize;
        node.serialize = function () {
            const o = orig ? orig.call(this) : {};
            if (this._ag_executed_display && o) {
                o.ag_executed_display = this._ag_executed_display;
            }
            return o;
        };
    }, "serialize hook");
    safe(() => {
        const orig = node.onConfigure;
        node.onConfigure = function (info) {
            safe(() => {
                if (info && typeof info.ag_executed_display === "string"
                    && info.ag_executed_display.trim() !== "") {
                    this._ag_executed_display = info.ag_executed_display;
                }
            }, "display restore");
            safe(() => {
                if (orig) orig.apply(this, arguments);
            }, "configure");
            scheduleRefresh(this);
        };
    }, "configure hook");
    safe(() => {
        const orig = node.onResize;
        node.onResize = function (size) {
            safe(() => {
                if (orig) orig.apply(this, arguments);
            }, "resize");
            // Горизонталь пользовательская, вертикаль автоматическая.
            // Horizontal is user-owned, vertical is automatic.
            safe(() => fitNode(this), "resize fit");
        };
    }, "resize hook");
    safe(() => {
        const orig = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            safe(() => {
                if (orig) orig.apply(this, arguments);
            }, "connections");
            scheduleRefresh(this);
        };
    }, "connections hook");
}

function startReconcile(node) {
    const timer = safe(() => setInterval(() => safe(() => tick(node), "tick"), TICK_MS), "timer");
    let ro = null;
    safe(() => {
        const host = node.__ag_bar_el ? node.__ag_bar_el.parentElement : null;
        if (host && typeof ResizeObserver === "function") {
            ro = new ResizeObserver(() => scheduleRefresh(node));
            ro.observe(host);
        }
    }, "observer");
    safe(() => {
        const orig = node.onRemoved;
        node.onRemoved = function () {
            safe(() => {
                if (timer) clearInterval(timer);
                if (ro) ro.disconnect();
            }, "cleanup");
            if (orig) return orig.apply(this, arguments);
        };
    }, "removed hook");
}

// ==============================================================================
// Extension / Расширение
// ==============================================================================
app.registerExtension({
    name: "AGSoft.EmptyLatentPlus",
    async nodeCreated(node) {
        if (!node) return;
        if (node.comfyClass !== NODE_CLASS_NAME && node.type !== NODE_CLASS_NAME) return;
        node._ag_executed_display = null;
        // Порядок важен: сначала логика, потом DOM-виджет.
        // Order matters: logic first, DOM widget second.
        safe(() => hookWidgets(node), "widget hooks");
        safe(() => hookLifecycle(node), "lifecycle hooks");
        safe(() => hookComputeSize(node), "computeSize hook");
        safe(() => updateWidgets(node), "initial auto-hide");
        safe(() => syncSteps(node), "initial step sync");
        safe(() => ensureBar(node), "initial info line");
        safe(() => startReconcile(node), "reconcile start");
        refresh(node);
        setTimeout(() => refresh(node), 0);
        setTimeout(() => refresh(node), 150);
    },
});