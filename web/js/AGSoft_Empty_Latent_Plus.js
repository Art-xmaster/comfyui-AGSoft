// ==============================================================================
// AGSoft_Empty_Latent_Plus.js
// ==============================================================================
// JS-расширение для ноды 🧊AGSoft Empty Latent Plus.
// JS extension for the 🧊AGSoft Empty Latent Plus node.
//
// Возможности / Features:
// ⚡ Живая строка внизу ноды.
//   До выполнения: W×H (ratio) · VAE auto
//   После выполнения: точная строка из Python.
//   Live line at the bottom of the node.
//   Before execution: W×H (ratio) · VAE auto
//   After execution: exact string from Python.
// ⚡ Скрытие неиспользуемых виджетов для size_mode: Ratio / Preset / Custom.
//   Hiding unused widgets for size_mode: Ratio / Preset / Custom.
// ⚡ Дополнительные скрытия внутри Ratio:
//   - custom_ratio виден только если ratio_preset = custom;
//   - base_value виден для width/height/longest/shortest;
//   - megapixels_value виден только для base = megapixels.
//   Extra hiding inside Ratio:
//   - custom_ratio visible only when ratio_preset = custom;
//   - base_value visible for width/height/longest/shortest;
//   - megapixels_value visible only for base = megapixels.
// ⚡ Пересчёт при любом изменении виджетов.
//   Recalculation on any widget change.
// ⚡ Ошибка пропорции — красная строка с подсказкой.
//   Ratio error — red hint line.
//
// Author: AGSoft
// Date: 01.10.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS_NAME = "AGSoft_Empty_Latent_Plus";
const INFO_H = 34;

const WIDGET_NAMES = [
    "size_mode",
    "size_preset",
    "ratio_preset",
    "custom_ratio",
    "base",
    "base_value",
    "megapixels_value",
    "width",
    "height",
    "multiple",
    "rounding",
    "batch_size",
];

let executedHookInstalled = false;

function isTargetNode(node) {
    return !!(
        node &&
        (node.comfyClass === NODE_CLASS_NAME || node.type === NODE_CLASS_NAME)
    );
}

// ------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------

function getWidget(node, name) {
    if (!node || !Array.isArray(node.widgets)) {
        return null;
    }

    return node.widgets.find((w) => w.name === name) || null;
}

function getWidgetValue(node, name, fallback = null) {
    const w = getWidget(node, name);

    if (!w) {
        return fallback;
    }

    return w.value;
}

function setWidgetVisible(node, name, visible) {
    const w = getWidget(node, name);

    if (!w) {
        return false;
    }

    const hidden = !visible;

    if (w.hidden !== hidden) {
        w.hidden = hidden;
        return true;
    }

    return false;
}

function sizeFromCompute(node) {
    try {
        const s = node.computeSize();

        if (Array.isArray(s)) {
            return [s[0] || 0, s[1] || 0];
        }

        return [s.width || 0, s.height || 0];
    } catch (e) {
        return [0, 0];
    }
}

function layoutNode(node, force = false) {
    try {
        const [minW, minHWidgets] = sizeFromCompute(node);
        const minH = minHWidgets + INFO_H;

        if (!node.size) {
            node.size = [minW, minH];
        }

        let targetW;
        let targetH;

        if (force) {
            targetW = Math.max(minW, 260);
            targetH = Math.max(minH, INFO_H + 20);
        } else {
            targetW = Math.max(node.size[0], minW);
            targetH = Math.max(node.size[1], minH);
        }

        if (
            targetW !== node.size[0] ||
            targetH !== node.size[1]
        ) {
            node.setSize([targetW, targetH]);
        }

        node.setDirtyCanvas(true, true);
    } catch (e) {
        // ignore
    }
}

function sigOf(node) {
    try {
        return WIDGET_NAMES
            .map((name) => {
                const w = getWidget(node, name);
                return w ? String(w.value) : "";
            })
            .join("|");
    } catch (e) {
        return "";
    }
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

// ------------------------------------------------------------------------------
// Math: ratio / preset / multiple
// ------------------------------------------------------------------------------

const NUM_RE = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;

function parseNum(p) {
    const t = String(p ?? "").trim();

    if (!NUM_RE.test(t)) {
        throw new Error("format");
    }

    return parseFloat(t);
}

function parseRatio(text) {
    let s = String(text ?? "").trim();

    for (const sep of ["x", "X", "/", ",", ";"]) {
        s = s.split(sep).join(":");
    }

    const parts = s
        .split(":")
        .map((p) => p.trim())
        .filter((p) => p !== "");

    if (parts.length !== 2) {
        throw new Error("format");
    }

    const w = parseNum(parts[0]);
    const h = parseNum(parts[1]);

    if (w <= 0 || h <= 0) {
        throw new Error("positive");
    }

    return [w, h];
}

function parseSizePreset(name) {
    const s = String(name ?? "").trim();

    if (!s.includes("-") || !s.includes("(")) {
        throw new Error("format");
    }

    const dims = s
        .split("-", 2)[1]
        .split("(", 1)[0]
        .replace("×", "x")
        .toLowerCase()
        .trim();

    const ratio = s
        .substring(s.lastIndexOf("(") + 1, s.lastIndexOf(")"))
        .trim();

    const parts = dims
        .split("x", 2)
        .map((v) => parseInt(v.trim(), 10));

    const w = parts[0];
    const h = parts[1];

    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
        throw new Error("positive");
    }

    return [w, h, ratio];
}

function pyRound(v) {
    // Банковское округление, как round() в Python.
    const f = Math.floor(v);
    const d = v - f;

    if (d > 0.5) {
        return f + 1;
    }

    if (d < 0.5) {
        return f;
    }

    return (f % 2 === 0) ? f : f + 1;
}

function toMultiple(value, multiple, mode) {
    const v = Math.max(1.0, Number(value));
    const m = parseInt(multiple, 10) || 1;

    if (m <= 1) {
        return pyRound(v);
    }

    if (mode === "floor") {
        return Math.max(m, Math.floor(v / m) * m);
    }

    if (mode === "ceil") {
        return Math.max(m, Math.ceil(v / m) * m);
    }

    return Math.max(m, pyRound(v / m) * m);
}

// ------------------------------------------------------------------------------
// Visibility
// ------------------------------------------------------------------------------

function updateVisibility(node, forceLayout = false) {
    try {
        const sizeMode = String(getWidgetValue(node, "size_mode", "Ratio"));
        const ratioPreset = String(getWidgetValue(node, "ratio_preset", "1:1"));
        const base = String(getWidgetValue(node, "base", "width"));

        const isRatio = sizeMode === "Ratio";
        const isPreset = sizeMode === "Preset";
        const isCustom = sizeMode === "Custom";

        let changed = false;

        changed = setWidgetVisible(node, "size_preset", isPreset) || changed;

        changed = setWidgetVisible(node, "ratio_preset", isRatio) || changed;

        changed = setWidgetVisible(
            node,
            "custom_ratio",
            isRatio && ratioPreset === "custom"
        ) || changed;

        changed = setWidgetVisible(node, "base", isRatio) || changed;

        changed = setWidgetVisible(
            node,
            "base_value",
            isRatio && base !== "megapixels"
        ) || changed;

        changed = setWidgetVisible(
            node,
            "megapixels_value",
            isRatio && base === "megapixels"
        ) || changed;

        changed = setWidgetVisible(node, "width", isCustom) || changed;
        changed = setWidgetVisible(node, "height", isCustom) || changed;

        if (changed || forceLayout) {
            layoutNode(node, forceLayout);
        }
    } catch (e) {
        // ignore
    }
}

// ------------------------------------------------------------------------------
// Live info before execution
// ------------------------------------------------------------------------------

function computeInfo(node) {
    try {
        let w;
        let h;
        let label;

        const sizeMode = String(getWidgetValue(node, "size_mode", "Ratio"));

        if (sizeMode === "Custom") {
            w = parseFloat(getWidgetValue(node, "width", 1024));
            h = parseFloat(getWidgetValue(node, "height", 1024));

            if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
                throw new Error("positive");
            }

            label = "custom";
        } else if (sizeMode === "Preset") {
            const p = parseSizePreset(getWidgetValue(node, "size_preset", ""));

            w = p[0];
            h = p[1];
            label = p[2];
        } else {
            const ratioPreset = String(getWidgetValue(node, "ratio_preset", "1:1"));

            const ratioText = ratioPreset === "custom"
                ? String(getWidgetValue(node, "custom_ratio", "")).trim()
                : ratioPreset;

            const [rw, rh] = parseRatio(ratioText);

            const base = String(getWidgetValue(node, "base", "width"));
            const val = parseFloat(getWidgetValue(node, "base_value", 1024));

            if (!Number.isFinite(val)) {
                throw new Error("format");
            }

            const ratio = rw / rh;

            if (base === "width") {
                w = val;
                h = val * rh / rw;
            } else if (base === "height") {
                h = val;
                w = val * rw / rh;
            } else if (base === "longest") {
                if (rw >= rh) {
                    w = val;
                    h = val * rh / rw;
                } else {
                    h = val;
                    w = val * rw / rh;
                }
            } else if (base === "shortest") {
                if (rw <= rh) {
                    w = val;
                    h = val * rh / rw;
                } else {
                    h = val;
                    w = val * rw / rh;
                }
            } else {
                const mpp = parseFloat(getWidgetValue(node, "megapixels_value", 1));

                if (!Number.isFinite(mpp)) {
                    throw new Error("format");
                }

                const target = mpp * 1_000_000;

                w = Math.sqrt(target * ratio);
                h = Math.sqrt(target / ratio);
            }

            label = ratioText;
        }

        const multiple = getWidgetValue(node, "multiple", 64);
        const rounding = getWidgetValue(node, "rounding", "round");

        const W = toMultiple(w, multiple, rounding);
        const H = toMultiple(h, multiple, rounding);

        return {
            ok: true,
            text: `${W}×${H} (${label}) · VAE auto`,
        };
    } catch (e) {
        const msg = (e && e.message === "positive")
            ? "❌ значения должны быть > 0"
            : "❌ неверный формат пропорции (ожидалось W:H)";

        return {
            ok: false,
            text: msg,
        };
    }
}

function refreshInfo(node, forceLayout = false) {
    try {
        updateVisibility(node, forceLayout);

        node._ag_sig = sigOf(node);

        const r = computeInfo(node);

        node._ag_info = r.text;
        node._ag_info_ok = r.ok;

        node.setDirtyCanvas(true, true);
    } catch (e) {
        // ignore
    }
}

// ------------------------------------------------------------------------------
// Executed hook: replace "VAE auto" with exact Python display
// ------------------------------------------------------------------------------

function applyExecutedInfo(node, output) {
    try {
        if (!node || !output) {
            return;
        }

        const out = output.ui || output;

        let display = Array.isArray(out.display)
            ? out.display[0]
            : out.display;

        if (typeof display !== "string" || display.trim() === "") {
            const W = Array.isArray(out.width) ? out.width[0] : out.width;
            const H = Array.isArray(out.height) ? out.height[0] : out.height;
            const ch = Array.isArray(out.channels) ? out.channels[0] : out.channels;
            const factor = Array.isArray(out.factor) ? out.factor[0] : out.factor;

            if (
                Number.isFinite(W) &&
                Number.isFinite(H) &&
                Number.isFinite(ch) &&
                Number.isFinite(factor) &&
                factor > 0
            ) {
                const lw = Math.floor(W / factor);
                const lh = Math.floor(H / factor);

                display = `${W}×${H} → latent ${lw}×${lh} · ch ${ch}`;
            }
        }

        if (typeof display === "string" && display.trim() !== "") {
            node._ag_info = display;
            node._ag_info_ok = true;
            node.setDirtyCanvas(true, true);
        }
    } catch (e) {
        // ignore
    }
}

function installExecutedHook() {
    if (executedHookInstalled) {
        return;
    }

    executedHookInstalled = true;

    api.addEventListener("executed", (evt) => {
        try {
            const detail = evt.detail;

            if (!detail) {
                return;
            }

            const node = app.graph.getNodeById(detail.node);

            if (!isTargetNode(node)) {
                return;
            }

            applyExecutedInfo(node, detail.output);
        } catch (e) {
            // ignore
        }
    });
}

// ------------------------------------------------------------------------------
// Extension
// ------------------------------------------------------------------------------

app.registerExtension({
    name: "AGSoft.EmptyLatentPlus",

    nodeCreated(node) {
        if (!isTargetNode(node)) {
            return;
        }

        installExecutedHook();

        node._ag_info = "";
        node._ag_info_ok = true;
        node._ag_sig = "";

        // Wrap widget callbacks.
        WIDGET_NAMES.forEach((name) => {
            const w = getWidget(node, name);

            if (!w || w._ag_bound) {
                return;
            }

            w._ag_bound = true;

            const oldCallback = w.callback;

            w.callback = (value) => {
                if (oldCallback) {
                    oldCallback(value);
                }

                refreshInfo(node, false);
            };
        });

        // After workflow loading.
        const origConfigure = node.onConfigure;

        node.onConfigure = function (info) {
            const result = origConfigure
                ? origConfigure.apply(this, arguments)
                : undefined;

            setTimeout(() => {
                refreshInfo(this, true);
            }, 0);

            return result;
        };

        // Initial layout.
        setTimeout(() => {
            refreshInfo(node, true);
        }, 0);

        // Draw bottom live line.
        node.onDrawForeground = function (ctx) {
            try {
                if (this.flags && this.flags.collapsed) {
                    return;
                }

                const sig = sigOf(this);

                if (sig !== this._ag_sig) {
                    this._ag_sig = sig;

                    const r = computeInfo(this);

                    this._ag_info = r.text;
                    this._ag_info_ok = r.ok;

                    // Update hidden widgets outside draw cycle.
                    setTimeout(() => {
                        updateVisibility(this, false);
                    }, 0);
                }

                const W = this.size[0];
                const H = this.size[1];

                const barH = Math.max(18, INFO_H - 12);
                const x = 8;
                const y = H - barH - 8;
                const w = W - x * 2;

                ctx.save();

                ctx.beginPath();
                ctx.rect(0, 0, W, H);
                ctx.clip();

                roundRectPath(ctx, x, y, w, barH, 7);

                ctx.fillStyle = "rgba(0,0,0,0.35)";
                ctx.fill();

                ctx.strokeStyle = this._ag_info_ok ? "#5b6ee1" : "#a04040";
                ctx.lineWidth = 1;
                ctx.stroke();

                ctx.fillStyle = this._ag_info_ok ? "#cdd3ff" : "#ffb4b4";
                ctx.font = "bold 11px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";

                const text = this._ag_info_ok
                    ? ("🧊 " + this._ag_info)
                    : this._ag_info;

                ctx.fillText(
                    text,
                    x + w / 2,
                    y + barH / 2 + 0.5,
                    w - 16
                );

                ctx.restore();
            } catch (e) {
                // ignore
            }
        };
    },
});