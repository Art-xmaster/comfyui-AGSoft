import { app } from "../../../scripts/app.js";

// Dynamic audio_1, audio_2, ... inputs for AGSoft Audio Concatenate Plus.
app.registerExtension({
    name: "AGSoft.AudioConcatenatePlus",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftAudioConcatenatePlus") return;

        const widget = node.widgets?.find(w => w.name === "inputs_count");
        if (!widget) return;

        const updateInputs = () => {
            const target = parseInt(widget.value, 10) || 2;

            if (!node.inputs) {
                node.inputs = [];
            }

            // Remove extra audio_ inputs.
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const input = node.inputs[i];
                if (!input.name.startsWith("audio_")) continue;
                const match = input.name.match(/audio_(\d+)/);
                if (!match) continue;
                const index = parseInt(match[1], 10);
                if (isNaN(index) || index > target) {
                    node.removeInput(i);
                }
            }

            // Add missing audio_ inputs.
            for (let i = 1; i <= target; i++) {
                const name = `audio_${i}`;
                const exists = node.inputs.some(input => input.name === name);
                if (!exists) {
                    node.addInput(name, "STRING");
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
});