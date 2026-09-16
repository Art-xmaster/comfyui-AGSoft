// ==============================================================================
// AGSoft_Stitch_Video_Audio.js
// ==============================================================================
// Динамические входы для нод:
// 🎬🔊AGSoft Stitch Video & Audio
// 🎬🔊AGSoft Latent Stitch Video & Audio
//
// Dynamic inputs for the nodes:
// 🎬🔊AGSoft Stitch Video & Audio
// 🎬🔊AGSoft Latent Stitch Video & Audio
//
// Виджет inputs_count (2-50) управляет количеством сокетов:
//   - Stitch Video & Audio       : images_N (IMAGE), audio_N (AUDIO).
//   - Latent Stitch Video & Audio: video_latent_N (LATENT), audio_latent_N (LATENT).
//
// Лишние сокеты (index > target) удаляются вместе со ссылками, недостающие
// добавляются по порядку 1..N. Порядок сокетов = порядок склейки.
// Extra sockets (index > target) are removed together with their links,
// missing ones are added in order 1..N. Socket order = stitch order.
//
// Автор / Author: AGSoft
// Дата / Date: 16.09.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";


// ------------------------------------------------------------------------------
// Конфигурация динамических сокетов по comfyClass ноды.
// Dynamic socket configuration per node comfyClass.
// ------------------------------------------------------------------------------

const SOCKET_SPECS = {
    AGSoftStitchVideoAudio: [
        {
            prefix: "images_",
            type: "IMAGE",
            tooltip: (
                "Decoded segment frames (IMAGE). Order defines stitch order.\n" +
                "---\n" +
                "Декодированные кадры отрезка (IMAGE). Порядок определяет порядок склейки."
            ),
        },
        {
            prefix: "audio_",
            type: "AUDIO",
            tooltip: (
                "Decoded segment audio (AUDIO). Order defines stitch order.\n" +
                "---\n" +
                "Декодированное аудио отрезка (AUDIO). Порядок определяет порядок склейки."
            ),
        },
    ],

    AGSoftLatentStitchVideoAudio: [
        {
            prefix: "video_latent_",
            type: "LATENT",
            tooltip: (
                "Segment video latent (LATENT). Order defines stitch order.\n" +
                "---\n" +
                "Видео-латент отрезка (LATENT). Порядок определяет порядок склейки."
            ),
        },
        {
            prefix: "audio_latent_",
            type: "LATENT",
            tooltip: (
                "Segment audio latent (LATENT). Order defines stitch order.\n" +
                "---\n" +
                "Аудио-латент отрезка (LATENT). Порядок определяет порядок склейки."
            ),
        },
    ],
};


function manageDynamicInputs(node, specs) {
    const widget = node.widgets?.find(w => w.name === "inputs_count");
    if (!widget) return;

    const updateInputs = () => {
        const raw = parseInt(widget.value, 10);
        const target = isNaN(raw) ? 2 : Math.max(2, Math.min(50, raw));

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

        // Добавляем недостающие сокеты 1..target.
        // Add missing sockets 1..target.
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
    name: "AGSoft.StitchVideoAudio",

    async nodeCreated(node) {
        const specs = SOCKET_SPECS[node.comfyClass];
        if (!specs) return;

        manageDynamicInputs(node, specs);
    },
});