import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AGSoft.LoopImages",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AGSoft_Loop_Images") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origCreated) origCreated.apply(this, arguments);

                if (!this.inputs) this.inputs = [];
                if (!this.widgets) this.widgets = [];

                const updateInputs = () => {
                    const numWidget = this.widgets.find(w => w.name === "number_of_images");
                    if (!numWidget) return;

                    const num = Math.max(1, parseInt(numWidget.value) || 1);

                    // Сохраняем не-image входы
                    const nonImage = this.inputs.filter(inp => !inp.name?.startsWith?.("image_"));
                    const imageInputs = [];

                    for (let i = 1; i <= num; i++) {
                        const name = `image_${i}`;
                        let existing = this.inputs.find(inp => inp.name === name);
                        if (existing) {
                            imageInputs.push(existing);
                        } else {
                            imageInputs.push({ name, type: "IMAGE", link: null, widget: null });
                        }
                    }

                    this.inputs = [...nonImage, ...imageInputs];
                    this.setSize(this.computeSize());
                    app.graph.setDirtyCanvas(true, true);
                };

                // Перемещаем number_of_images вверх
                const numWidget = this.widgets.find(w => w.name === "number_of_images");
                if (numWidget) {
                    this.widgets = [numWidget, ...this.widgets.filter(w => w !== numWidget)];
                    const origCallback = numWidget.callback;
                    numWidget.callback = function () {
                        if (origCallback) origCallback.apply(this, arguments);
                        updateInputs();
                    };
                }

                // Инициализация
                setTimeout(updateInputs, 0);
            };
        }
    }
});