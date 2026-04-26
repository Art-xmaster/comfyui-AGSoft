// Подключаем основной объект 'app' для взаимодействия с интерфейсом ComfyUI
import { app } from "../../../scripts/app.js";

// Регистрируем расширение для динамического изменения количества входов
app.registerExtension({
    name: "AGSoft.ReferenceToLatent.DynamicInputs",
    
    /**
     * Вызывается каждый раз, когда нода создаётся или загружается из workflow.
     * @param {Object} node - Объект созданной ноды
     */
    async nodeCreated(node) {
        // Проверяем, что это наша нода (по comfyClass)
        if (node.comfyClass === "AGSoftReferenceToLatent") {
            
            /**
             * Основная функция обновления входов.
             * Считывает текущее значение виджета 'mode' и создаёт/удаляет слоты 'imageX'.
             */
            const updateInputs = () => {
                // 1. Находим виджет 'mode'
                const modeWidget = node.widgets?.find(w => w.name === "mode");
                if (!modeWidget) return;
                
                // 2. Получаем количество изображений из значения виджета
                let imageCount = 1;
                const modeValue = modeWidget.value;
                if (typeof modeValue === 'string') {
                    const num = parseInt(modeValue);
                    if (!isNaN(num)) imageCount = num;
                } else if (typeof modeValue === 'number') {
                    imageCount = modeValue;
                }
                
                // Убеждаемся, что у ноды есть массив inputs
                if (!node.inputs) {
                    node.inputs = [];
                }
                
                // 3. Сохраняем существующие связи (линки) перед удалением слотов
                //    Это нужно, чтобы при добавлении слота заново восстановить соединение.
                const savedConnections = {};
                for (const input of node.inputs) {
                    // Ищем слоты вида 'image1', 'image2' и т.д. с активной связью
                    if (input.name && input.name.match(/^image\d+$/) && input.link) {
                        const link = node.graph.links[input.link];
                        if (link) {
                            savedConnections[input.name] = {
                                origin_id: link.origin_id,
                                origin_slot: link.origin_slot
                            };
                        }
                    }
                }
                
                // 4. Удаляем лишние слоты (которых больше, чем нужно по 'mode')
                node.inputs = node.inputs.filter(input => {
                    // Пропускаем все слоты, не являющиеся 'imageX'
                    if (!input.name || !input.name.match(/^image\d+$/)) return true;
                    // Извлекаем номер из имени слота
                    const match = input.name.match(/image(\d+)/);
                    if (match && parseInt(match[1]) <= imageCount) return true;
                    // Если слот лишний и у него есть связь — разрываем её
                    if (input.link) node.disconnectInput(input.name);
                    return false;
                });
                
                // 5. Добавляем недостающие слоты
                for (let i = 1; i <= imageCount; i++) {
                    const inputName = `image${i}`;
                    // Проверяем, существует ли уже такой слот
                    if (!node.inputs.find(input => input.name === inputName)) {
                        // Добавляем новый вход с типом "IMAGE"
                        node.addInput(inputName, "IMAGE");
                        // Если для этого имени была сохранена связь — восстанавливаем её
                        if (savedConnections[inputName]) {
                            const { origin_id, origin_slot } = savedConnections[inputName];
                            const originNode = node.graph.getNodeById(origin_id);
                            if (originNode) {
                                originNode.connect(origin_slot, node, inputName);
                            }
                        }
                    }
                }
                
                // 6. Обновляем размеры ноды, чтобы интерфейс корректно отобразил новые слоты
                node.setSize(node.computeSize());
                // Сообщаем графу, что холст нужно перерисовать
                app.graph.setDirtyCanvas(true);
            };
            
            // --- Навешиваем обработчик на виджет 'mode' ---
            const modeWidget = node.widgets.find(w => w.name === "mode");
            if (modeWidget) {
                // Сохраняем оригинальный callback виджета (если есть)
                const origCallback = modeWidget.callback;
                // Подменяем callback, чтобы при изменении значения вызывалась наша функция обновления
                modeWidget.callback = () => {
                    // Вызываем оригинальный callback, если он был
                    if (origCallback) origCallback();
                    // Обновляем количество входов
                    updateInputs();
                    // Перерисовываем граф
                    app.graph.setDirtyCanvas(true);
                };
            }
            
            // Вызываем updateInputs один раз при создании ноды, чтобы привести всё в соответствие
            // Используем setTimeout, чтобы дать ComfyUI завершить инициализацию ноды.
            setTimeout(updateInputs, 0);
        }
    }
});