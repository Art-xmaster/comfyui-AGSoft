import { app } from "../../../scripts/app.js";

// Реализация динамических входов через nodeCreated
app.registerExtension({
    name: "AGSoft.VideoConcatenate",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftVideoConcatenate") return;

        const widget = node.widgets?.find(w => w.name === "inputs_count");
        if (!widget) return;

        const updateInputs = () => {
            const target = parseInt(widget.value, 10) || 2;
            const current = (node.inputs || []).filter(i => i.name.startsWith("video_")).length;

            if (current < target) {
                for (let i = current + 1; i <= target; i++) {
                    node.addInput(`video_${i}`, "STRING");
                }
            } else if (current > target) {
                for (let i = current; i > target; i--) {
                    const idx = node.inputs.findIndex(inp => inp.name === `video_${i}`);
                    if (idx !== -1) node.removeInput(idx);
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

        setTimeout(updateInputs, 50); // Инициализация
    }
});