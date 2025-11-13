// AGSoft_Text_Split.js
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AGSoft.TextSplit",
    async nodeCreated(node) {
        if (node.comfyClass === "AGSoftTextSplit") {
            const updateOutputs = () => {
                const numOutputsWidget = node.widgets.find(w => w.name === "number_of_outputs");
                if (!numOutputsWidget) return;

                const numOutputs = Math.min(parseInt(numOutputsWidget.value), 50);
                const initialWidth = node.size[0];

                // Удаляем все выходы, кроме служебных
                node.outputs = node.outputs.filter(output => 
                    !output.name.startsWith('string_')
                );

                // Добавляем только нужное количество выходов
                for (let i = 1; i <= numOutputs; i++) {
                    const outputName = `string_${i}`;
                    const outputTooltip = `Output #${i} of the split text. / ${i}-й выход разбитого текста.`;
                    node.addOutput(outputName, "STRING", { tooltip: outputTooltip });
                }

                // Восстанавливаем ширину узла и обновляем размер
                node.setSize(node.computeSize());
                node.size[0] = initialWidth;
                app.graph.setDirtyCanvas(true);
            };

            // Находим виджет number_of_outputs
            const numOutputsWidget = node.widgets.find(w => w.name === "number_of_outputs");
            if (numOutputsWidget) {
                // Перемещаем его в начало списка виджетов
                node.widgets = [numOutputsWidget, ...node.widgets.filter(w => w !== numOutputsWidget)];
                
                // Устанавливаем callback на изменение значения
                numOutputsWidget.callback = () => {
                    updateOutputs();
                    app.graph.setDirtyCanvas(true);
                };
            }

            // Инициализируем выходы при создании узла
            setTimeout(updateOutputs, 0);
        }
    }
});