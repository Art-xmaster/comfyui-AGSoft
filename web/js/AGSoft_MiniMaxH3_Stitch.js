// ==============================================================================
// AGSoft_MiniMaxH3_Stitch.js
// ==============================================================================
// Динамические входы для нод 🎬🧊AGSoft MiniMaxH3 Stitch Latent и
// 🎬🖼️AGSoft MiniMaxH3 Stitch Images.
// Dynamic inputs for the 🎬🧊AGSoft MiniMaxH3 Stitch Latent and
// 🎬🖼️AGSoft MiniMaxH3 Stitch Images nodes.
//
// Виджет inputs_count (2-50) управляет количеством сокетов:
//   - Stitch Latent : latent_N  (LATENT) — латенты отрезков (видео+аудио).
//   - Stitch Images : images_N  (IMAGE)  — декодированные кадры отрезков,
//                     latent_N  (LATENT) — опциональный источник АУДИО
//                     (пара к images_N по индексу).
// The inputs_count widget (2-50) controls the number of sockets:
//   - Stitch Latent : latent_N  (LATENT) — segment latents (video+audio).
//   - Stitch Images : images_N  (IMAGE)  — decoded segment frames,
//                     latent_N  (LATENT) — optional AUDIO source
//                     (paired with images_N by index).
//
// Лишние сокеты (index > target) удаляются вместе со ссылками, недостающие
// добавляются по порядку 1..N. Порядок сокетов = порядок склейки.
// Extra sockets (index > target) are removed together with their links,
// missing ones are added in order 1..N. Socket order = stitch order.
//
// Автор / Author: AGSoft
// Дата / Date: 07.09.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";

// Конфигурация динамических сокетов по comfyClass ноды.
// Dynamic socket configuration per node comfyClass.
const SOCKET_SPECS = {
    AGSoftMiniMaxH3StitchLatent: [
        {
            prefix: "latent_",
            type: "LATENT",
            tooltip: (
                "Segment latent (NestedTensor: video+audio). Order defines stitch order.\n" +
                "---\n" +
                "Латент отрезка (NestedTensor: видео+аудио). Порядок определяет порядок склейки."
            ),
        },
    ],
    AGSoftMiniMaxH3StitchImages: [
        {
            prefix: "images_",
            type: "IMAGE",
            tooltip: (
                "Decoded frames of the segment (IMAGE). Order defines stitch order.\n" +
                "---\n" +
                "Декодированные кадры отрезка (IMAGE). Порядок определяет порядок склейки."
            ),
        },
        {
            prefix: "latent_",
            type: "LATENT",
            tooltip: (
                "Segment latent — optional AUDIO source, paired with images_N by index.\n" +
                "---\n" +
                "Латент отрезка — опциональный источник АУДИО, пара к images_N по индексу."
            ),
        },
    ],
};

function manageDynamicInputs(node, specs) {
    const widget = node.widgets?.find(w => w.name === "inputs_count");
    if (!widget) return;

    const updateInputs = () => {
        const target = parseInt(widget.value, 10) || 2;
        if (!node.inputs) {
            node.inputs = [];
        }
        // Удаляем лишние сокеты (index > target) управляемых префиксов.
        // Remove extra sockets (index > target) for managed prefixes.
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const input = node.inputs[i];
            const match = input.name.match(/^(.+?)(\d+)$/);
            if (!match) continue;
            const spec = specs.find(s => s.prefix === match[1]);
            if (!spec) continue;
            const index = parseInt(match[2], 10);
            if (isNaN(index) || index > target) {
                node.removeInput(i);
            }
        }
        // Добавляем недостающие сокеты 1..target (images_N раньше latent_N).
        // Add missing sockets 1..target (images_N before latent_N per index).
        for (let i = 1; i <= target; i++) {
            for (const spec of specs) {
                const name = `${spec.prefix}${i}`;
                if (!node.inputs.some(input => input.name === name)) {
                    node.addInput(name, spec.type);
                    const added = node.inputs.find(input => input.name === name);
                    if (added && spec.tooltip) {
                        added.tooltip = spec.tooltip;
                    }
                }
            }
        }
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true);
    };

    const oldCallback = widget.callback;
    widget.callback = (value) => {
        if (oldCallback) oldCallback(value);
        updateInputs();
    };
    setTimeout(updateInputs, 50);
}

app.registerExtension({
    name: "AGSoft.MiniMaxH3Stitch",
    async nodeCreated(node) {
        const specs = SOCKET_SPECS[node.comfyClass];
        if (!specs) return;
        manageDynamicInputs(node, specs);
    }
});