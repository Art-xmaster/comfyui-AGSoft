import { app } from "../../../scripts/app.js";

// Динамические входы image_1, image_2, ... (тип IMAGE) для AGSoft Images To Video.
app.registerExtension({
    name: "AGSoft.ImagesToVideo",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftImagesToVideo") return;

        const widget = node.widgets?.find(w => w.name === "inputs_count");
        if (!widget) return;

        const updateInputs = () => {
            const target = parseInt(widget.value, 10) || 2;

            if (!node.inputs) {
                node.inputs = [];
            }

            // Удаляем лишние image_-входы.
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const input = node.inputs[i];
                if (!input.name.startsWith("image_")) continue;
                const match = input.name.match(/image_(\d+)/);
                if (!match) continue;
                const index = parseInt(match[1], 10);
                if (isNaN(index) || index > target) {
                    node.removeInput(i);
                }
            }

            // Добавляем недостающие входы image_1 ... image_N.
            for (let i = 1; i <= target; i++) {
                const name = `image_${i}`;
                const exists = node.inputs.some(input => input.name === name);
                if (!exists) {
                    node.addInput(name, "IMAGE");
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