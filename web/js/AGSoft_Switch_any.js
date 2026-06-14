// AGSoft_Switch_any.js
// Автор: AGSoft
// Дата: 14 июня 2026 г.
// Описание: JavaScript расширение для динамического создания входов ноды AGSoft Switch Any

import { app } from "../../../scripts/app.js";

// Регистрируем расширение для ComfyUI
app.registerExtension({
    name: "AGSoft.SwitchAny",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AGSoft_Switch_any") {
            
            const origCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                if (origCreated) {
                    origCreated.apply(this, arguments);
                }
                
                if (!this.inputs) this.inputs = [];
                if (!this.widgets) this.widgets = [];
                
                // Функция для обновления динамических входов
                const updateInputs = () => {
                    const numWidget = this.widgets.find(w => w.name === "number_of_inputs");
                    if (!numWidget) return;
                    
                    const num = Math.max(2, parseInt(numWidget.value) || 2);
                    
                    // Сохраняем нединамические входы (selected_input и number_of_inputs)
                    const nonDynamicInputs = this.inputs.filter(inp => 
                        inp.name !== undefined && 
                        !inp.name.startsWith("input_")
                    );
                    
                    // Создаем динамические входы
                    const dynamicInputs = [];
                    for (let i = 1; i <= num; i++) {
                        const name = `input_${i}`;
                        let existing = this.inputs.find(inp => inp.name === name);
                        
                        if (existing) {
                            dynamicInputs.push(existing);
                        } else {
                            dynamicInputs.push({
                                name: name,
                                type: "*",
                                link: null
                            });
                        }
                    }
                    
                    // Обновляем входы
                    this.inputs = [...nonDynamicInputs, ...dynamicInputs];
                    
                    // Обновляем комбо бокс - показываем только доступные опции
                    updateComboBox(num);
                    
                    // Обновляем размер ноды
                    this.setSize(this.computeSize());
                    app.graph.setDirtyCanvas(true, true);
                };
                
                // Функция для обновления комбо бокса
                const updateComboBox = (num) => {
                    const comboWidget = this.widgets.find(w => w.name === "selected_input");
                    if (!comboWidget) return;
                    
                    const currentValue = comboWidget.value;
                    
                    // Генерируем новые опции только до num
                    const newOptions = [];
                    for (let i = 1; i <= num; i++) {
                        newOptions.push(`input_${i}`);
                    }
                    
                    // Обновляем опции
                    comboWidget.options.values = newOptions;
                    
                    // Проверяем текущее значение
                    if (!newOptions.includes(currentValue)) {
                        comboWidget.value = newOptions[0];
                    }
                    
                    if (comboWidget.callback) {
                        comboWidget.callback(comboWidget.value);
                    }
                };
                
                // Находим number_of_inputs и перемещаем наверх
                const numWidget = this.widgets.find(w => w.name === "number_of_inputs");
                if (numWidget) {
                    this.widgets = [numWidget, ...this.widgets.filter(w => w !== numWidget)];
                    
                    const origCallback = numWidget.callback;
                    numWidget.callback = function() {
                        if (origCallback) {
                            origCallback.apply(this, arguments);
                        }
                        updateInputs();
                    };
                }
                
                // Первоначальная инициализация
                setTimeout(updateInputs, 10);
            };
            
            // Обработчик загрузки workflow
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(nodeData) {
                if (origOnConfigure) {
                    origOnConfigure.apply(this, arguments);
                }
                
                setTimeout(() => {
                    const numWidget = this.widgets.find(w => w.name === "number_of_inputs");
                    if (numWidget) {
                        const num = Math.max(2, parseInt(numWidget.value) || 2);
                        
                        const nonDynamicInputs = this.inputs.filter(inp => 
                            inp.name !== undefined && 
                            !inp.name.startsWith("input_")
                        );
                        
                        const dynamicInputs = [];
                        for (let i = 1; i <= num; i++) {
                            const name = `input_${i}`;
                            let existing = this.inputs.find(inp => inp.name === name);
                            
                            if (existing) {
                                dynamicInputs.push(existing);
                            } else {
                                dynamicInputs.push({
                                    name: name,
                                    type: "*",
                                    link: null
                                });
                            }
                        }
                        
                        this.inputs = [...nonDynamicInputs, ...dynamicInputs];
                        
                        // Обновляем комбо бокс
                        const comboWidget = this.widgets.find(w => w.name === "selected_input");
                        if (comboWidget) {
                            const newOptions = [];
                            for (let i = 1; i <= num; i++) {
                                newOptions.push(`input_${i}`);
                            }
                            comboWidget.options.values = newOptions;
                        }
                        
                        this.setSize(this.computeSize());
                        app.graph.setDirtyCanvas(true, true);
                    }
                }, 20);
            };
        }
    }
});