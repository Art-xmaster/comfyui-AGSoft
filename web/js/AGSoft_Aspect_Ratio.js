// ==============================================================================
// AGSoft_Aspect_Ratio.js
// ==============================================================================
// JS-расширение для ноды 📐AGSoft Aspect Ratio.
// ⚡ Оформленная информационная строка в самом низу ноды: после выполнения
// показывает полученные значения (та же строка, что в выходе display).
//
// Author: AGSoft
// Date: 27.08.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS_NAME = "AGSoft_Aspect_Ratio";
const INFO_H = 30; // высота информационной строки

function roundRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

// Гарантирует место под инфо-строку (идемпотентно: не растёт при повторных загрузках).
function ensureInfoHeight(node) {
    try {
        const base = node.computeSize();
        const need = (Array.isArray(base) ? base[1] : base.height) + INFO_H;
        if (node.size[1] < need) node.size[1] = need;
    } catch (e) { /* ignore */ }
}

// === МАТЕМАТИКА: точная копия Python (_parse_ratio / _to_multiple) ===
function parseRatio(text) {
    let s = String(text ?? "").trim();
    for (const sep of ["x", "X", "/", ",", ";"]) s = s.split(sep).join(":");
    const parts = s.split(":").filter(p => p.trim() !== "");
    if (parts.length !== 2) throw new Error("format");
    const w = parseNum(parts[0]), h = parseNum(parts[1]);
    if (w <= 0 || h <= 0) throw new Error("positive");
    return [w, h];
}
// Строгое число: ВСЯ строка обязана быть числом (как float() в Python).
// parseFloat("1д") вернул бы 1 — в этом был баг.
const NUM_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
function parseNum(p) {
    const t = String(p).trim();
    if (!NUM_RE.test(t)) throw new Error("format");
    return parseFloat(t);
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
// Клиентский расчёт строки display (те же формулы, что в calculate()).
function computeInfo(node) {
    const get = (n) => { const w = node.widgets.find(x => x.name === n); return w ? w.value : null; };
    try {
        const preset = String(get("preset") || "1:1");
        const ratioText = preset === "custom" ? String(get("custom_ratio") || "").trim() : preset;
        const [rw, rh] = parseRatio(ratioText);
        const base = String(get("base") || "width");
        const val = parseFloat(get("base_value"));
        if (!isFinite(val)) throw new Error("format");
        const ratio = rw / rh;
        let w, h;
        if (base === "width") { w = val; h = val * rh / rw; }
        else if (base === "height") { h = val; w = val * rw / rh; }
        else if (base === "longest") { if (rw >= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
        else if (base === "shortest") { if (rw <= rh) { w = val; h = val * rh / rw; } else { h = val; w = val * rw / rh; } }
        else {
            const mp = parseFloat(get("megapixels_value"));
            if (!isFinite(mp)) throw new Error("format");
            const t = mp * 1e6; w = Math.sqrt(t * ratio); h = Math.sqrt(t / ratio);
        }
        const W = toMultiple(w, get("multiple"), get("rounding"));
        const H = toMultiple(h, get("multiple"), get("rounding"));
        return { ok: true, text: `${W}×${H} (${ratioText})` };
    } catch (e) {
        const msg = (e && e.message === "positive")
            ? "❌ пропорция: значения должны быть > 0"
            : "❌ неверный формат пропорции (ожидалось W:H)";
        return { ok: false, text: msg };
    }
}

app.registerExtension({
    name: "AGSoft.AspectRatio",

    nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS_NAME && node.type !== NODE_CLASS_NAME) return;
        node._ag_info = "";
        node._ag_info_ok = true;
        node._ag_sig = "";
        ensureInfoHeight(node);
        // Сигнатура ВСЕХ входных значений: по ней ловим ЛЮБОЕ изменение при перерисовке.
        const WIDGET_NAMES = ["preset", "custom_ratio", "base", "base_value", "megapixels_value", "multiple", "rounding"];
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
        ["preset", "custom_ratio", "base", "base_value", "megapixels_value", "multiple", "rounding"].forEach(n => {
            const w = node.widgets.find(x => x.name === n);
            if (w) { const o = w.callback; w.callback = (v) => { if (o) o(v); refreshInfo(); }; }
        });
        refreshInfo(); // значения видны сразу, без "выполните ноду"
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
                // ГЛАВНЫЙ механизм: сигнатура значений изменилась -> пересчёт строки.
                // Не зависит от widget.callback: работает с любым виджетом и фронтом.
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
                const txt = this._ag_info_ok ? ("📐 " + this._ag_info) : this._ag_info;
                ctx.fillText(txt, x + w / 2, y + h / 2 + 1, w - 12);
                ctx.restore();
            } catch (e) { /* ignore */ }
        };
    },
});

// Слушатель "executed" удалён: строка теперь считается клиентом (computeInfo)
// и обновляется мгновенно — без выполнения воркфлоу.