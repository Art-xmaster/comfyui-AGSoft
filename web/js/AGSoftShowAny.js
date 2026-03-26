// AGSoftShowAny.js
// Универсальная нода для отображения значений любого типа
// Полностью копирует механизм AGSoftShowText, но работает с любыми типами данных

import { app } from "../../../../scripts/app.js";
import { ComfyWidgets } from "../../../../scripts/widgets.js";

// Регистрируем расширение для отображения любых значений
app.registerExtension({
    name: "AGSoft.ShowAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Проверяем, что это наша нода AGSoft Show Any
        if (nodeData.name === "AGSoftShowAny") {
            
            // Функция для создания виджетов отображения текста
            function populate(texts) {
                // Удаляем существующие виджеты
                if (this.widgets) {
                    // На старых версиях фронтенда есть скрытый converted-widget
                    const isConvertedWidget = +!!this.inputs?.[0].widget;
                    for (let i = isConvertedWidget; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = isConvertedWidget;
                }
                
                // Создаем массив текстов для отображения
                const v = [...texts];
                if (!v[0]) {
                    v.shift();
                }
                
                // Для каждого текста создаем виджет
                for (let list of v) {
                    // Приводим list к массиву
                    if (!(list instanceof Array)) list = [list];
                    
                    for (const l of list) {
                        // Создаем многострочное текстовое поле
                        const w = ComfyWidgets["STRING"](this, "text_" + this.widgets?.length ?? 0, ["STRING", { multiline: true }], app).widget;
                        w.inputEl.readOnly = true;  // Делаем поле только для чтения
                        w.inputEl.style.opacity = 0.6;  // Делаем полупрозрачным
                        w.value = l;  // Устанавливаем значение
                    }
                }
                
                // Обновляем размер ноды
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // Сохраняем оригинальный метод onExecuted
            const onExecuted = nodeType.prototype.onExecuted;
            // Переопределяем onExecuted для отображения значений
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                // Вызываем populate с полученными текстами
                populate.call(this, message.text);
            };

            // Сохраняем значения для конфигурации
            const VALUES = Symbol();
            const configure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                // Сохраняем значения виджетов перед конфигурацией
                this[VALUES] = arguments[0]?.widgets_values;
                return configure?.apply(this, arguments);
            };

            // Восстанавливаем значения после конфигурации
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                const widgets_values = this[VALUES];
                if (widgets_values?.length) {
                    // Задержка для создания начального виджета на новых версиях
                    requestAnimationFrame(() => {
                        populate.call(this, widgets_values.slice(+(widgets_values.length > 1 && this.inputs?.[0].widget)));
                    });
                }
            };
        }
    },
});