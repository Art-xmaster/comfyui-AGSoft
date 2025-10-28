import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AGSoft.LoopTexts",
    async nodeCreated(node) {
        if (node.comfyClass === "AGSoft_Loop_Texts") {
            const updateInputs = () => {
                const initialWidth = node.size[0];
                const numInputsWidget = node.widgets.find(w => w.name === "number_of_inputs");
                if (!numInputsWidget) return;
                const numInputs = numInputsWidget.value;

                if (!node.inputs) {
                    node.inputs = [];
                }

                const existingInputs = node.inputs.filter(input => input.name.startsWith('text_'));

                if (existingInputs.length < numInputs) {
                    for (let i = existingInputs.length + 1; i <= numInputs; i++) {
                        const inputName = `text_${i}`;
                        if (!node.inputs.find(input => input.name === inputName)) {
                            node.addInput(inputName, "STRING");
                        }
                    }
                } else {
                    node.inputs = node.inputs.filter(input => !input.name.startsWith('text_') || parseInt(input.name.split('_')[1]) <= numInputs);
                }

                node.setSize(node.computeSize());
                node.size[0] = initialWidth;
            };

            const numInputsWidget = node.widgets.find(w => w.name === "number_of_inputs");
            if (numInputsWidget) {
                node.widgets = [numInputsWidget, ...node.widgets.filter(w => w !== numInputsWidget)];
                numInputsWidget.callback = () => {
                    updateInputs();
                    app.graph.setDirtyCanvas(true);
                };
            }

            setTimeout(updateInputs, 0);
        }
    }
});