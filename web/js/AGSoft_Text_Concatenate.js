import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "AGSoft.TextConcatenate",
    async nodeCreated(node) {
        // Проверяем, что это наша нода
        if (node.comfyClass === "AGSoftTextConcatenate") {
            const numInputsWidget = node.widgets.find(w => w.name === "number_of_inputs");
            if (!numInputsWidget) return;

            // Перемещаем виджет количества в самый верх для удобства
            node.widgets = [numInputsWidget, ...node.widgets.filter(w => w !== numInputsWidget)];

            const updateInputs = () => {
                const targetCount = numInputsWidget.value;
                
                // Находим все существующие входы text_X
                const textInputs = node.inputs.filter(i => i.name.startsWith("text_"));
                const currentCount = textInputs.length;

                if (currentCount < targetCount) {
                    // Добавляем недостающие входы
                    for (let i = currentCount + 1; i <= targetCount; i++) {
                        node.addInput(`text_${i}`, "STRING");
                    }
                } else if (currentCount > targetCount) {
                    // Удаляем лишние входы С КОНЦА (безопасно для индексов и связей)
                    for (let i = currentCount; i > targetCount; i--) {
                        const inputName = `text_${i}`;
                        const idx = node.inputs.findIndex(inp => inp.name === inputName);
                        if (idx !== -1) {
                            node.removeInput(idx);
                        }
                    }
                }
                
                // Пересчитываем размер ноды и обновляем канвас
                node.setSize(node.computeSize());
                app.graph.setDirtyCanvas(true);
            };

            // Вешаем callback на изменение виджета
            numInputsWidget.callback = () => {
                updateInputs();
            };

            // Инициализируем входы при первом создании/загрузке ноды
            setTimeout(updateInputs, 50);
        }
    }
});