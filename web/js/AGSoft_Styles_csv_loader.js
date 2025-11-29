import { app } from "../../../scripts/app.js";

// Регистрация расширения для ноды AGSoft Styles CSV Loader
app.registerExtension({
    name: "AGSoft.StylesCSVLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Проверяем правильное имя ноды
        if (nodeData.name === "AGSoft_Styles_CSV_Loader") {
            // Сохраняем оригинальную функцию onNodeCreated
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                // Вызываем оригинальную функцию
                const result = originalOnNodeCreated ? 
                               originalOnNodeCreated.apply(this, arguments) : undefined;
                
                // Даем время для полной инициализации виджетов
                setTimeout(() => {
                    this.setupStyleLoader();
                }, 100);
                
                return result;
            };
            
            // Добавляем метод для настройки загрузчика стилей
            nodeType.prototype.setupStyleLoader = function() {
                const styleFileWidget = this.widgets?.find(w => w.name === "style_file");
                const styleNameWidget = this.widgets?.find(w => w.name === "style_name");
                const reloadFilesWidget = this.widgets?.find(w => w.name === "reload_files");
                
                if (!styleFileWidget || !styleNameWidget) {
                    console.warn("AGSoft Styles CSV Loader: Required widgets not found");
                    return;
                }
                
                // Функция для загрузки стилей из выбранного файла
                const loadStylesForFile = async (filename) => {
                    if (!filename || filename.startsWith("No style files found")) {
                        styleNameWidget.options.values = ["No styles available"];
                        styleNameWidget.value = "No styles available";
                        this.trigger("widgetChanged", { 
                            name: "style_name", 
                            value: "No styles available" 
                        });
                        return;
                    }
                    
                    try {
                        const response = await fetch("/agsoft_load_styles", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                            },
                            body: JSON.stringify({ filename: filename })
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            
                            if (data.styles && data.styles.length > 0) {
                                const styleNames = data.styles;
                                const currentValue = styleNameWidget.value;
                                const hasCurrentValue = styleNames.includes(currentValue);
                                
                                // Обновляем список стилей
                                styleNameWidget.options.values = styleNames;
                                
                                // Сохраняем текущее значение, если оно есть в новом списке
                                if (hasCurrentValue && currentValue !== "No styles available") {
                                    styleNameWidget.value = currentValue;
                                } else {
                                    styleNameWidget.value = styleNames[0];
                                }
                                
                                // Уведомляем о изменении значения
                                this.trigger("widgetChanged", { 
                                    name: "style_name", 
                                    value: styleNameWidget.value 
                                });
                            } else {
                                styleNameWidget.options.values = ["No styles found in file"];
                                styleNameWidget.value = "No styles found in file";
                                this.trigger("widgetChanged", { 
                                    name: "style_name", 
                                    value: "No styles found in file" 
                                });
                            }
                        } else {
                            const errorText = await response.text();
                            console.error("AGSoft: Failed to load styles:", errorText);
                            styleNameWidget.options.values = ["Error loading styles"];
                            styleNameWidget.value = "Error loading styles";
                        }
                    } catch (error) {
                        console.error("AGSoft: Error loading styles:", error);
                        styleNameWidget.options.values = ["Error loading styles"];
                        styleNameWidget.value = "Error loading styles";
                    }
                    
                    // Обновляем отображение
                    app.graph.setDirtyCanvas(true, false);
                };
                
                // Функция для перезагрузки списка файлов
                const reloadFileList = async () => {
                    try {
                        const response = await fetch("/agsoft_get_style_files");
                        
                        if (response.ok) {
                            const files = await response.json();
                            
                            if (files && files.length > 0) {
                                const currentValue = styleFileWidget.value;
                                const hasCurrentValue = files.includes(currentValue);
                                
                                styleFileWidget.options.values = files;
                                
                                if (hasCurrentValue) {
                                    styleFileWidget.value = currentValue;
                                    // Обновляем стили для текущего файла
                                    await loadStylesForFile(currentValue);
                                } else {
                                    styleFileWidget.value = files[0];
                                    // Загружаем стили для нового файла
                                    await loadStylesForFile(files[0]);
                                }
                            }
                        } else {
                            console.error("AGSoft: Failed to reload file list");
                        }
                    } catch (error) {
                        console.error("AGSoft: Error reloading file list:", error);
                    }
                    
                    // Сбрасываем кнопку перезагрузки
                    if (reloadFilesWidget) {
                        reloadFilesWidget.value = false;
                    }
                };
                
                // Устанавливаем обработчики событий для виджетов
                styleFileWidget.callback = () => {
                    loadStylesForFile(styleFileWidget.value);
                };
                
                if (reloadFilesWidget) {
                    reloadFilesWidget.callback = () => {
                        if (reloadFilesWidget.value) {
                            reloadFileList();
                        }
                    };
                }
                
                // Инициализация: загружаем стили для текущего файла
                if (styleFileWidget.value && 
                    !styleFileWidget.value.startsWith("No style files found")) {
                    loadStylesForFile(styleFileWidget.value);
                }
            };
        }
    }
});