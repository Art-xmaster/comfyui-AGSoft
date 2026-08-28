// ==============================================================================
// AGSoft_Empty_Latent_Plus.js
// ==============================================================================
// JS-расширение для ноды 🧊AGSoft Empty Latent Plus.
// JS extension for the 🧊AGSoft Empty Latent Plus node.
// (See the RU list above — the same features.)
//
// Возможности / Features:
// ⚡ Оформленная живая строка внизу ноды: W×H (ratio) → latent W×H · ch N.
//   Styled live line at the node bottom: W×H (ratio) → latent W×H · ch N.
// ⚡ Пересчёт при ЛЮБОМ изменении виджетов (механизм сигнатур) и сразу после
//   создания ноды. / Recalculation on ANY widget change (signature mechanism)
//   and right after the node is created.
// ⚡ Ошибка пропорции — красная строка с подсказкой вместо немого сбоя.
//   Ratio error — a red hint line instead of a silent failure.
// ⚡ Математика 1:1 с Python: парсинг пропорций, банковское округление,
//   кратность и фактор латента. / Math 1:1 with Python: ratio parsing,
//   banker's rounding, multiple & latent-factor alignment.
//
// Author: AGSoft
// Date: 28.08.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS_NAME = "AGSoft_Empty_Latent_Plus";
const INFO_H = 30; // высота информационной строки

// Копия MODEL_PRESETS из Python (объединённые группы).
const MODEL_PRESETS = {
    "SD1.5 / SDXL":                     { ch: 4,  factor: 8 },
    "SD3 / FLUX.1 / Krea2 / QwenImage": { ch: 16, factor: 8 },
    "FLUX.2 / Flux2-klein":             { ch: 128, factor: 16 },
};

// Парсер пресета размера: 'Portrait - 896x1152 (3:4)' -> [896, 1152, '3:4']. Понимает и '×'.
function parseSizePreset(name) {
    const s = String(name ?? "").trim();
    if (!s.includes(" - ") || !s.includes(" (")) throw new Error("format");
    const dims = s.split(" - ")[1].split(" (")[0].replace("×", "x").toLowerCase();
    const ratio = s.substring(s.lastIndexOf("(") + 1, s.lastIndexOf(")"));
    const [w, h] = dims.split("x").map(v => parseInt(v, 10));
    if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) throw new Error("positive");
    return [w, h, ratio];
}

function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

// Гарантирует место под инфо-строку (идемпотентно).
function ensureInfoHeight(node) {
    try {
        const base = node.computeSize();
        const need = (Array.isArray(base) ? base[1] : base.height) + INFO_H;
        if (node.size[1] < need) node.size[1] = need;
    } catch (e) { /* ignore */ }
}

// === МАТЕМАТИКА: точная копия Python (_parse_ratio / _to_multiple) ===
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
function pyRound(v) { // банковское округление как round() в Python
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

// Клиентский расчёт строки display (те же формулы, что в generate()).
function computeInfo(node) {
    const get = (n) => { const w = node.widgets.find(x => x.name === n); return w ? w.value : null; };
    try {
        const mt = String(get("model_type") || "SD1.5 / SDXL");
        const pre = MODEL_PRESETS[mt] || Object.values(MODEL_PRESETS)[0];
        const ch = pre.ch;
        const factor = pre.factor;

        let w, h, label;
        if (String(get("size_mode")) === "Custom") {
            w = parseFloat(get("width")); h = parseFloat(get("height"));
            if (!isFinite(w) || !isFinite(h) || w <= 0 || h <= 0) throw new Error("positive");
            label = "custom";
         } else if (String(get("size_mode")) === "Preset") {
             const p = parseSizePreset(get("size_preset"));
             w = p[0]; h = p[1]; label = p[2];
         } else {
            const ratioPreset = String(get("ratio_preset") || "1:1");
            const ratioText = ratioPreset === "custom" ? String(get("custom_ratio") || "").trim() : ratioPreset;
            const [rw, rh] = parseRatio(ratioText);
            const base = String(get("base") || "width");
            const val = parseFloat(get("base_value"));
            if (!isFinite(val)) throw new Error("format");
            const ratio = rw / rh;
            if (base === "width") { w = val; h = val * rh / rw; }
            else if (base === "height") { h = val; w = val * rw / rh; }
            else if (base === "longest") { if (rw >= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
            else if (base === "shortest") { if (rw <= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
            else {
                const mpp = parseFloat(get("megapixels_value"));
                if (!isFinite(mpp)) throw new Error("format");
                const t = mpp * 1e6; w = Math.sqrt(t * ratio); h = Math.sqrt(t / ratio);
            }
            label = ratioText;
        }

        let W = toMultiple(w, get("multiple"), get("rounding"));
        let H = toMultiple(h, get("multiple"), get("rounding"));
        W = Math.max(factor, Math.floor(W / factor) * factor);
        H = Math.max(factor, Math.floor(H / factor) * factor);
        const lw = Math.floor(W / factor), lh = Math.floor(H / factor);
        return { ok: true, text: `${W}×${H} (${label}) → latent ${lw}×${lh} · ch ${ch}` };
    } catch (e) {
        const msg = (e && e.message === "positive")
            ? "❌ значения должны быть > 0"
            : "❌ неверный формат пропорции (ожидалось W:H)";
        return { ok: false, text: msg };
    }
}

app.registerExtension({
    name: "AGSoft.EmptyLatentPlus",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS_NAME && node.type !== NODE_CLASS_NAME) return;
        node._ag_info = "";
        node._ag_info_ok = true;
        node._ag_sig = "";
        ensureInfoHeight(node);
        // Сигнатура ВСЕХ значимых виджетов: ловим ЛЮБОЕ изменение при перерисовке.
        const WIDGET_NAMES = ["model_type", "size_mode", "size_preset",
            "ratio_preset", "custom_ratio", "base", "base_value", "megapixels_value",
            "width", "height", "multiple", "rounding"];
        const sigOf = () => WIDGET_NAMES.map(n => {
            const w = node.widgets.find(x => x.name === n);
            return w ? String(w.value) : "";
        }).join("|");
        const refreshInfo = () => {
            node._ag_sig = sigOf();
            const r = computeInfo(node);
            node._ag_info = r.text;
            node._ag_info_ok = r.ok;
            node.setDirtyCanvas(true, true);
        };
        WIDGET_NAMES.forEach(n => {
            const w = node.widgets.find(x => x.name === n);
            if (w) { const o = w.callback; w.callback = (v) => { if (o) o(v); refreshInfo(); }; }
        });
        refreshInfo(); // значения видны сразу, без запуска воркфлоу
        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            ensureInfoHeight(this); // после загрузки воркфлоу size приходит из файла
            setTimeout(refreshInfo, 0); // сохранённые значения виджетов уже приехали
            return r;
        };
        // Отрисовка оформленной строки в самом низу ноды.
        node.onDrawForeground = function (ctx) {
            try {
                if (this.flags && this.flags.collapsed) return;
                // ГЛАВНЫЙ механизм: сигнатура изменилась -> пересчёт строки.
                const sig = sigOf();
                if (sig !== this._ag_sig) {
                    this._ag_sig = sig;
                    const r = computeInfo(this);
                    this._ag_info = r.text;
                    this._ag_info_ok = r.ok;
                }
                const W = this.size[0], H = this.size[1];
                const x = 6, y = H - INFO_H + 4, w = W - 12, h = INFO_H - 8;
                ctx.save();
                ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
                roundRectPath(ctx, x, y, w, h, 6);
                ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fill();
                ctx.strokeStyle = this._ag_info_ok ? "#5b6ee1" : "#a04040"; ctx.lineWidth = 1; ctx.stroke();
                ctx.fillStyle = this._ag_info_ok ? "#cdd3ff" : "#ffb4b4";
                ctx.font = "bold 12px monospace";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                const txt = this._ag_info_ok ? ("🧊 " + this._ag_info) : this._ag_info;
                ctx.fillText(txt, x + w / 2, y + h / 2 + 1, w - 12);
                ctx.restore();
            } catch (e) { /* ignore */ }
        };
    },
});