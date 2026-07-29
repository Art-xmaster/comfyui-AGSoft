import { app } from "../../../scripts/app.js";

// Dynamic video_1, video_2, ... inputs for AGSoft Video Concatenate Plus.
app.registerExtension({
    name: "AGSoft.VideoConcatenatePlus",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftVideoConcatenatePlus") return;

        const widget = node.widgets?.find(w => w.name === "inputs_count");
        if (!widget) return;

        const updateInputs = () => {
            const target = parseInt(widget.value, 10) || 2;

            if (!node.inputs) {
                node.inputs = [];
            }

            // Remove extra video_ inputs.
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const input = node.inputs[i];

                if (!input.name.startsWith("video_")) continue;

                const match = input.name.match(/video_(\d+)/);
                if (!match) continue;

                const index = parseInt(match[1], 10);

                if (isNaN(index) || index > target) {
                    node.removeInput(i);
                }
            }

            // Add missing video_ inputs.
            for (let i = 1; i <= target; i++) {
                const name = `video_${i}`;

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