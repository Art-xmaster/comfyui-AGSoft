/*
==============================================================================
AGSoft_AddGuideMiniMaxH3.js
==============================================================================
Нода / Node: 📍AGSoft Add Guide for MiniMax H3
Описание / Description:
Динамические слоты направляющих: входы image/audio создаются и удаляются по
значению Number of Guides; аудио-входы появляются только при подключённом audio_vae;
виджеты seconds_i / frames_i показываются парой к каждому активному слоту и
переключаются по position_mode.
---
Dynamic guide slots: image/audio inputs are created and removed according to
the Number of Guides value; audio inputs appear only when audio_vae is linked;
seconds_i / frames_i widgets are shown paired to each active slot and are
switched by position_mode.
Автор / Author: AGSoft
Дата / Date: 20.09.2026
==============================================================================
*/
import { app } from "../../../scripts/app.js";

console.log("[AGSoft Add Guide MiniMax H3] JS extension loaded");

const MAX_GUIDES = 16;
const INPUT_TYPE = 1; // LiteGraph.INPUT / входной слот

function findInput(node, name) {
    return (node.inputs || []).findIndex((x) => x.name === name);
}
function ensureInput(node, name, type) {
    if (findInput(node, name) < 0) node.addInput(name, type);
}
function dropInput(node, name) {
    const idx = findInput(node, name);
    if (idx < 0) return;
    try {
        if (node.inputs[idx].link != null) node.disconnectInput(idx);
    } catch (e) { /* ignore / игнор */ }
    node.removeInput(idx);
}
function audioActive(node) {
    const idx = findInput(node, "audio_vae");
    return idx >= 0 && node.inputs[idx].link != null;
}
// single source of truth: slots + paired position widgets / единая логика: слоты + парные виджеты позиции
function refresh(node) {
    const counter = (node.widgets || []).find((w) => w.name === "number_of_guides");
    const mode = (node.widgets || []).find((w) => w.name === "position_mode");
    if (!counter || !mode) return;
    const count = Math.max(1, Math.min(MAX_GUIDES, parseInt(counter.value) || 1));
    const showSec = mode.value === "Seconds";
    const useAudio = audioActive(node);
    // inputs: drop extra from tail, add missing in order / входы: убираем лишние с конца, добавляем по порядку
    for (let i = MAX_GUIDES; i > count; i--) {
        dropInput(node, "audio_" + i);
        dropInput(node, "image_" + i);
    }
    if (!useAudio) {
        for (let i = 1; i <= MAX_GUIDES; i++) dropInput(node, "audio_" + i);
    }
    for (let i = 1; i <= count; i++) {
        ensureInput(node, "image_" + i, "IMAGE");
        if (useAudio) ensureInput(node, "audio_" + i, "AUDIO");
    }
    // position widgets: only for active slots, seconds or frames / виджеты позиции: только активные слоты, секунды или кадры
    for (let i = 1; i <= MAX_GUIDES; i++) {
        const active = i <= count;
        const ws = (node.widgets || []).find((x) => x.name === "seconds_" + i);
        const wf = (node.widgets || []).find((x) => x.name === "frames_" + i);
        if (ws) ws.hidden = !active || !showSec;
        if (wf) wf.hidden = !active || showSec;
    }
    if (node.setSize) node.setSize(node.computeSize());
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "AGSoft.AddGuideMiniMaxH3",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftAddGuideMiniMaxH3") return;
        const counter = (node.widgets || []).find((w) => w.name === "number_of_guides");
        const mode = (node.widgets || []).find((w) => w.name === "position_mode");
        if (!counter || !mode) return;
        // wrap callbacks safely, keep old ones / безопасно оборачиваем callback'и, сохраняя старые
        const prevCount = counter.callback;
        const ocCount = prevCount ? (v) => prevCount.call(counter, v) : null;
        counter.callback = (v) => { if (ocCount) ocCount(v); refresh(node); };
        const prevMode = mode.callback;
        const ocMode = prevMode ? (v) => prevMode.call(mode, v) : null;
        mode.callback = (v) => { if (ocMode) ocMode(v); refresh(node); };
        // react to audio_vae link changes / реагируем на подключение/отключение audio_vae
        const origOCC = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, linked, linkInfo) {
            if (origOCC) origOCC.apply(this, arguments);
            if (type === INPUT_TYPE && this.inputs && this.inputs[index] && this.inputs[index].name === "audio_vae") {
                refresh(this);
            }
        };
        // restore visibility on workflow load / восстанавливаем вид при загрузке workflow
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            refresh(this);
        };
        refresh(node);
    },
});