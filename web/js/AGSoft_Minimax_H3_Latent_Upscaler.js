/*
==============================================================================
AGSoft_Minimax_H3_Latent_Upscaler.js
==============================================================================
Расширение: автоскрытие неиспользуемых виджетов размера по режиму (pattern AGSoft).
---
Extension: auto-hides unused size widgets per mode (AGSoft pattern).
Автор / Author: AGSoft
Дата / Date: 19.10.2026
==============================================================================
*/
import { app } from "../../../scripts/app.js";

console.log("[AGSoft Minimax H3 Latent Upscaler] JS extension loaded");

// Скрытые виджеты выкинуты из карты попаданий, линки встают точно в видимые
// Hidden widgets are removed from the hit map so links land on visible ones
const Y_OFF = -9999;

function getWidget(node, name) {
    if (!node || !node.widgets) return null;
    for (const w of node.widgets) if (w && w.name === name) return w;
    return null;
}

function isLinkedInput(node, name) {
    // Виджеты с подключённым линком не скрываем / Widgets with a connected link are never hidden
    if (!node.inputs) return false;
    for (const inp of node.inputs) {
        if (inp && inp.name === name && inp.link != null) return true;
    }
    return false;
}

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
        if (hidden) w.y = Y_OFF;
        else if (w.y === Y_OFF) w.y = 0;
    } catch (e) {}
}

function updateWidgets(node) {
    try {
        if (!node || !node.widgets) return false;
        const modeW = getWidget(node, "mode");
        const mode = String(modeW && modeW.value ? modeW.value : "scale by multiplier");
        const rules = {
            mode: true,
            scale: mode === "scale by multiplier",
            width: mode === "target dimensions",
            height: mode === "target dimensions",
            megapixels: mode === "megapixels",
        };
        let changed = false;
        for (const w of node.widgets) {
            if (!w) continue;
            if (!Object.prototype.hasOwnProperty.call(rules, w.name)) continue;
            if (isLinkedInput(node, w.name)) {
                setHidden(w, false);
                continue;
            }
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

let refreshTimer = 0;
function scheduleRefresh(node) {
    // Дебаунс: несколько событий подряд -> один пересчёт / Debounce: coalesce events into one recompute
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => updateWidgets(node), 0);
}

app.registerExtension({
    name: "AGSoft.MinimaxH3LatentUpscaler",
    async nodeCreated(node) {
        try {
            if (!node || node.comfyClass !== "AGSoftMinimaxH3LatentUpscaler") return;
            updateWidgets(node);
            // Хук виджета режима / Hook the mode widget
            const modeW = getWidget(node, "mode");
            if (modeW) {
                const old = modeW.callback;
                modeW.callback = function (v) {
                    try { if (old) old.apply(this, arguments); } catch (e) {}
                    updateWidgets(node);
                };
            }
            // Подключение/отключение линков -> пересчёт. ВХОДЫ И ЛИНКИ НЕ ТРОГАЕМ.
            // Linking/unlinking -> recompute. Inputs and links are NEVER touched.
            const origConn = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info) {
                try { if (origConn) origConn.apply(this, arguments); } catch (e) {}
                scheduleRefresh(this);
            };
            // Восстановление видимости при загрузке workflow / Restore visibility on workflow load
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                try { if (origOnConfigure) origOnConfigure.apply(this, arguments); } catch (e) {}
                scheduleRefresh(this);
            };
            setTimeout(() => updateWidgets(node), 0);
        } catch (e) {
            console.warn("[AGSoft Minimax H3 Latent Upscaler] nodeCreated skipped:", e);
        }
    },
});