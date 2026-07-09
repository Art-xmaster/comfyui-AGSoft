// AGSoft Save Workflow Image - JavaScript Extension for ComfyUI
// Файл: ComfyUI/custom_nodes/comfyui-AGSoft/web/js/AGSoft_Save_workflowImage.js
// Версия: 2.1 (Исправленная)
// Описание: Расширение для сохранения рабочей области (workflow) в PNG с возможностью
//           экспорта/импорта workflow и сохранения в JSON.

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

let fileInput = null;
let getDrawTextConfig = null;

/**
 * @class WorkflowImage
 * @description Base class for exporting the workflow canvas as an image.
 *              Базовый класс для экспорта холста рабочего процесса в виде изображения.
 */
class WorkflowImage {
    static accept = "";
    extension = "";

    /**
     * Вычисляет границы всех нод на холсте, чтобы охватить всю рабочую область.
     * @returns {number[]} Массив [x1, y1, x2, y2] представляющий ограничивающий прямоугольник.
     */
    getBounds() {
        const bounds = app.graph._nodes.reduce(
            (p, n) => {
                if (n.pos[0] < p[0]) p[0] = n.pos[0];
                if (n.pos[1] < p[1]) p[1] = n.pos[1];
                const nodeBounds = n.getBounding();
                const r = n.pos[0] + nodeBounds[2];
                const b = n.pos[1] + nodeBounds[3];
                if (r > p[2]) p[2] = r;
                if (b > p[3]) p[3] = b;
                return p;
            },
            [99999, 99999, -99999, -99999]
        );
        
        // Добавляем отступы (padding) со всех сторон, чтобы ноды не прилипали к краям
        bounds[0] -= 150;
        bounds[1] -= 150;
        bounds[2] += 150;
        bounds[3] += 150;
        return bounds;
    }

    /**
     * Сохраняет текущее состояние холста (масштаб, смещение, размеры)
     * для последующего восстановления после экспорта.
     */
    saveState() {
        this.state = {
            scale: app.canvas.ds.scale,
            width: app.canvas.canvas.width,
            height: app.canvas.canvas.height,
            offset: app.canvas.ds.offset,
            transform: app.canvas.canvas.getContext("2d").getTransform(),
        };
    }

    /**
     * Восстанавливает ранее сохраненное состояние холста.
     */
    restoreState() {
        app.canvas.ds.scale = this.state.scale;
        app.canvas.canvas.width = this.state.width;
        app.canvas.canvas.height = this.state.height;
        app.canvas.ds.offset = this.state.offset;
        app.canvas.canvas.getContext("2d").setTransform(this.state.transform);
    }

    /**
     * Обновляет вид холста (масштаб и смещение) так, чтобы все ноды поместились на изображении.
     * @param {number[]} bounds - Границы рабочей области.
     */
    updateView(bounds) {
        const w = bounds[2] - bounds[0];
        const h = bounds[3] - bounds[1];
        let scale = 1.2;
        const MAX_DIM = 8192;
        
        // Ограничение максимального размера изображения для PNG (защита от браузерных лимитов)
        if (this.extension === "png" && (w * scale > MAX_DIM || h * scale > MAX_DIM)) {
            scale = Math.min(MAX_DIM / w, MAX_DIM / h);
        }
        
        app.canvas.ds.scale = scale;
        app.canvas.canvas.width = w * scale;
        app.canvas.canvas.height = h * scale;
        app.canvas.ds.offset = [-bounds[0], -bounds[1]];
    }

    /**
     * Конфигурация для отрисовки текстовых виджетов на холсте.
     * Используется для перехвата отрисовки DOM-элементов в Canvas.
     */
    getDrawTextConfig(_, widget) {
        return {
            x: 10,
            y: widget.last_y + 10,
            resetTransform: false,
        };
    }

    /**
     * Основной метод экспорта изображения.
     * @param {boolean} includeWorkflow - Если true, внедряет JSON workflow в изображение.
     */
    async export(includeWorkflow) {
        this.saveState();
        const bounds = this.getBounds();
        this.updateView(bounds);
        
        // Принудительная перерисовка холста
        app.graph.setDirtyCanvas(true, true);
        app.canvas.setDirty(true, true);
        
        // Перехватываем отрисовку текста (хак для отрисовки DOM-виджетов на Canvas)
        getDrawTextConfig = this.getDrawTextConfig;
        app.canvas.draw(true, true);
        
        // Ждем, чтобы DOM-элементы успели перерисоваться
        await new Promise(r => setTimeout(r, 500));
        app.canvas.draw(true, true);

        // Принудительная отрисовка изображений внутри нод (например, превью)
        const ctx = app.canvas.ctx;
        ctx.save();
        if (ctx.setTransform) {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
        }
        ctx.scale(app.canvas.ds.scale, app.canvas.ds.scale);
        ctx.translate(app.canvas.ds.offset[0], app.canvas.ds.offset[1]);

        for (const node of app.graph._nodes) {
            if (node.imgs && node.imgs.length > 0) {
                const anyComplete = node.imgs.some(img => img.complete && img.width > 0);
                if (anyComplete) {
                    this.drawNodeImages(ctx, node);
                }
            }
        }
        ctx.restore();

        getDrawTextConfig = null;

        // Получаем Blob изображения (с workflow или без)
        const blob = await this.getBlob(includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined);
        
        this.restoreState();
        app.canvas.draw(true, true);
        this.download(blob);
    }

    /**
     * Отрисовывает изображения внутри конкретной ноды.
     * @param {CanvasRenderingContext2D} ctx - Контекст холста.
     * @param {LGraphNode} node - Нода, изображения которой нужно отрисовать.
     */
    drawNodeImages(ctx, node) {
        ctx.save();
        ctx.translate(node.pos[0], node.pos[1]);

        let contentStartY = 26; // Учитываем заголовок ноды
        
        // Расчет высоты виджетов
        if (node.widgets && node.widgets.length > 0) {
            let widgetsHeight = 0;
            for (const w of node.widgets) {
                widgetsHeight += 28; // Примерная высота одного виджета
            }
            contentStartY += widgetsHeight + 5;
        }

        // Специфичная логика для некоторых типов нод ComfyUI (костыль для корректного расчета высоты)
        const isSampler = node.type && node.type.includes("Sampler");
        const isSave = node.type && (node.type.includes("Save") || node.type.includes("Output") || node.type.includes("Preview"));
        
        if (isSampler && contentStartY < 220) contentStartY = 220;
        const footerHeight = isSave ? 55 : 25;
        
        const drawHeight = node.size[1] - contentStartY - footerHeight;
        const nodeWidth = node.size[0];

        if (drawHeight <= 0) {
            ctx.restore();
            return;
        }

        const imgs = node.imgs.filter(img => img.complete && img.width > 0);
        if (imgs.length === 0) {
            ctx.restore();
            return;
        }

        try {
            if (imgs.length === 1) {
                // Если одно изображение — центрируем его
                const img = imgs[0];
                const imgAspect = img.width / img.height;
                const areaAspect = nodeWidth / drawHeight;
                let targetWidth, targetHeight, offsetXImg, offsetYImg;

                if (imgAspect > areaAspect) {
                    targetWidth = nodeWidth;
                    targetHeight = nodeWidth / imgAspect;
                    offsetXImg = 0;
                    offsetYImg = contentStartY + (drawHeight - targetHeight) / 2;
                } else {
                    targetHeight = drawHeight;
                    targetWidth = drawHeight * imgAspect;
                    offsetXImg = (nodeWidth - targetWidth) / 2;
                    offsetYImg = contentStartY;
                }
                ctx.drawImage(img, offsetXImg, offsetYImg, targetWidth, targetHeight);
            } else {
                // Если изображений несколько — рисуем сеткой
                const count = imgs.length;
                const cols = Math.ceil(Math.sqrt(count));
                const rows = Math.ceil(count / cols);
                const cellWidth = nodeWidth / cols;
                const cellHeight = drawHeight / rows;

                // Заливаем область контента цветом фона ноды, чтобы скрыть сетку холста
                ctx.fillStyle = "#353535"; // Темно-серый цвет фона
                ctx.fillRect(0, contentStartY, nodeWidth, drawHeight);

                for (let i = 0; i < count; i++) {
                    const img = imgs[i];
                    if (!img.complete || img.width === 0) continue;

                    const row = Math.floor(i / cols);
                    const col = i % cols;
                    const cellX = col * cellWidth;
                    const cellY = contentStartY + row * cellHeight;

                    // Рисуем изображение с сохранением пропорций по центру ячейки
                    const imgAspect = img.width / img.height;
                    const cellAspect = cellWidth / cellHeight;
                    let targetWidth, targetHeight, offsetXImg, offsetYImg;

                    if (imgAspect > cellAspect) {
                        targetWidth = cellWidth;
                        targetHeight = cellWidth / imgAspect;
                        offsetXImg = cellX;
                        offsetYImg = cellY + (cellHeight - targetHeight) / 2;
                    } else {
                        targetHeight = cellHeight;
                        targetWidth = cellHeight * imgAspect;
                        offsetXImg = cellX + (cellWidth - targetWidth) / 2;
                        offsetYImg = cellY;
                    }
                    ctx.drawImage(img, offsetXImg, offsetYImg, targetWidth, targetHeight);
                }
            }
        } catch (e) {
            console.error("[AGSoft] Failed to draw images for node", node.id, e);
        }
        ctx.restore();
    }

    /**
     * Скачивает Blob как файл.
     * @param {Blob} blob - Данные для скачивания.
     */
    download(blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        Object.assign(a, {
            href: url,
            download: "workflow." + this.extension,
            style: "display: none",
        });
        document.body.append(a);
        a.click();
        setTimeout(() => {
            a.remove();
            window.URL.revokeObjectURL(url);
        }, 0);
    }

    /**
     * Открывает диалог выбора файла для импорта workflow.
     */
    static import() {
        if (!fileInput) {
            fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                onchange: () => {
                    // ИСПРАВЛЕНО: добавлена проверка на наличие выбранного файла
                    if (fileInput.files.length > 0) {
                        app.handleFile(fileInput.files[0]);
                    }
                },
            });
            document.body.append(fileInput);
        }
        fileInput.accept = WorkflowImage.accept;
        fileInput.click();
    }
    
    // Заглушка для переопределения в наследниках
    async getBlob(workflow) {
        return new Blob();
    }
}

/**
 * @class JSONWorkflowSaver
 * @description Class for saving the workflow as a JSON file.
 *              Класс для сохранения рабочего процесса в виде JSON-файла.
 */
class JSONWorkflowSaver {
    /**
     * Сериализует текущий граф и скачивает его как JSON-файл с таймстампом в имени.
     */
    static save() {
        try {
            const workflow = app.graph.serialize();
            const jsonStr = JSON.stringify(workflow, null, 2);
            
            // Формируем имя файла с текущей датой и временем
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            // ИСПРАВЛЕНО: убран лишний символ переноса строки \n в конце имени файла
            const filename = `workflow_${timestamp}.json`; 
            
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename; 
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("[AGSoft] Error saving JSON: ", error);
        }
    }
}

/**
 * @class PngWorkflowImage
 * @extends WorkflowImage
 * @description Class for handling PNG export, including embedding workflow JSON into PNG metadata.
 *              Класс для работы с экспортом в PNG, включая внедрение JSON workflow в метаданные PNG.
 */
class PngWorkflowImage extends WorkflowImage {
    static accept = ".png,image/png";
    extension = "png";

    /**
     * Конвертирует 32-битное число в массив из 4 байт (Big Endian).
     * @param {number} n - Число для конвертации.
     * @returns {Uint8Array} Массив байтов.
     */
    n2b(n) {
        return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
    }

    /**
     * Объединяет несколько ArrayBuffer в один Uint8Array.
     * @param {...ArrayBuffer} bufs - Буферы для объединения.
     * @returns {Uint8Array} Объединенный массив.
     */
    joinArrayBuffer(...bufs) {
        const result = new Uint8Array(bufs.reduce((totalSize, buf) => totalSize + buf.byteLength, 0));
        bufs.reduce((offset, buf) => {
            result.set(buf, offset);
            return offset + buf.byteLength;
        }, 0);
        return result;
    }

    /**
     * Вычисляет CRC32 для массива данных (требуется для формирования валидных PNG chunks).
     * @param {Uint8Array} data - Данные для вычисления CRC.
     * @returns {number} Значение CRC32.
     */
    crc32(data) {
        const crcTable = PngWorkflowImage.crcTable || (PngWorkflowImage.crcTable = (() => {
            let c;
            const crcTable = [];
            for (let n = 0; n < 256; n++) {
                c = n;
                for (let k = 0; k < 8; k++) {
                    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
                }
                crcTable[n] = c;
            }
            return crcTable;
        })());
        
        let crc = 0 ^ -1;
        for (let i = 0; i < data.byteLength; i++) {
            crc = (crc >>> 8) ^ crcTable[(crc ^ data[i]) & 0xff];
        }
        return (crc ^ -1) >>> 0;
    }

    /**
     * Создает Blob изображения PNG. Если передан workflow, внедряет его в PNG как tEXt chunk.
     * @param {string} [workflow] - JSON-строка workflow для внедрения.
     * @returns {Promise<Blob>} Промис с Blob-объектом изображения.
     */
    async getBlob(workflow) {
        return new Promise((r) => {
            // Использование временного холста для устранения белых полос и прозрачности
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = app.canvasEl.width;
            tempCanvas.height = app.canvasEl.height;
            const tempCtx = tempCanvas.getContext("2d");
            
            // Заливаем фон цветом холста ComfyUI
            tempCtx.fillStyle = app.canvas.clear_background_color || "#222222";
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(app.canvasEl, 0, 0);

            tempCanvas.toBlob(async (blob) => {
                if (workflow) {
                    const buffer = await blob.arrayBuffer();
                    const typedArr = new Uint8Array(buffer);
                    const view = new DataView(buffer);
                    
                    // Формируем tEXt chunk: Keyword\0Text
                    // ИСПРАВЛЕНО: убран лишний \n в конце текста
                    const data = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
                    const chunk = this.joinArrayBuffer(
                        this.n2b(data.length - 4), // Длина данных (без учета типа и CRC)
                        data, 
                        this.n2b(this.crc32(data)) // CRC32
                    );
                    
                    // Вставляем chunk сразу после IHDR (первый chunk, размер которого хранится в байтах 8-11)
                    const sz = view.getUint32(8) + 20; 
                    const result = this.joinArrayBuffer(
                        typedArr.subarray(0, sz), 
                        chunk, 
                        typedArr.subarray(sz)
                    );
                    blob = new Blob([result], { type: "image/png" });
                }
                r(blob);
            }, "image/png");
        });
    }
}

const pngSaver = new PngWorkflowImage();

/**
 * Регистрация расширения в ComfyUI.
 * DESCRIPTION / ОПИСАНИЕ:
 * Registers the extension in ComfyUI, adding custom menu options for saving/importing workflows.
 * Регистрирует расширение в ComfyUI, добавляя пользовательские пункты меню для сохранения/импорта рабочих процессов.
 */
app.registerExtension({
    name: "AGSoft.SaveWorkflow",
    
    /**
     * Метод init() вызывается при инициализации расширения.
     * Здесь мы перехватываем отрисовку текстовых виджетов (STRING), чтобы они корректно
     * отображались на экспортируемом изображении (так как в ComfyUI они являются DOM-элементами).
     */
    init() {
        // Вспомогательная функция для переноса текста при отрисовке на Canvas
        function wrapText(context, text, x, y, maxWidth, lineHeight) {
            var words = text.split(" "), line = "", i, test, metrics;
            for (i = 0; i < words.length; i++) {
                test = words[i];
                metrics = context.measureText(test);
                while (metrics.width > maxWidth) {
                    test = test.substring(0, test.length - 1);
                    metrics = context.measureText(test);
                }
                if (words[i] != test) {
                    words.splice(i + 1, 0, words[i].substr(test.length));
                    words[i] = test;
                }
                test = line + words[i] + " ";
                metrics = context.measureText(test);
                if (metrics.width > maxWidth && i > 0) {
                    context.fillText(line, x, y);
                    line = words[i] + " ";
                    y += lineHeight;
                } else {
                    line = test;
                }
            }
            context.fillText(line, x, y);
        }

        // Перехватываем стандартный виджет STRING
        const stringWidget = ComfyWidgets.STRING;
        ComfyWidgets.STRING = function () {
            const w = stringWidget.apply(this, arguments);
            if (w.widget && w.widget.type === "customtext") {
                const draw = w.widget.draw;
                w.widget.draw = function (ctx) {
                    draw.apply(this, arguments);
                    if (this.inputEl.hidden) return;
                    
                    // Если активен режим экспорта, рисуем текст на Canvas
                    if (getDrawTextConfig) {
                        const config = getDrawTextConfig(ctx, this);
                        ctx.save();
                        if (config.resetTransform) {
                            ctx.resetTransform();
                        }
                        const style = document.defaultView.getComputedStyle(this.inputEl, null);
                        const x = config.x;
                        const y = config.y;
                        const domWrapper = this.inputEl.closest(".dom-widget") ?? this.inputEl;
                        
                        let w = parseInt(domWrapper.style.width);
                        if (w === 0) {
                            w = this.node.size[0] - 20;
                        }
                        const h = parseInt(domWrapper.style.height);
                        
                        ctx.beginPath();
                        ctx.rect(x, y, w, h);
                        ctx.clip();
                        
                        ctx.fillStyle = style.getPropertyValue("background-color");
                        ctx.fillRect(x, y, w, h);
                        ctx.fillStyle = style.getPropertyValue("color");
                        ctx.font = style.getPropertyValue("font");
                        
                        const t = ctx.getTransform();
                        const line = t.d * 12;
                        const split = this.inputEl.value.split("\n");
                        let start = y;
                        for (const l of split) {
                            start += line;
                            wrapText(ctx, l, x + 4, start, w, line);
                        }
                        ctx.restore();
                    }
                };
            }
            return w;
        };
    },

    /**
     * Метод setup() вызывается после полной инициализации ComfyUI.
     * Добавляем новые пункты в контекстное меню холста.
     */
    setup() {
        const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
        LGraphCanvas.prototype.getCanvasMenuOptions = function () {
            const options = orig.apply(this, arguments);
            options.push(null, {
                content: "📸 AGSoft Workflow Image",
                submenu: {
                    options: [
                        {
                            content: "📸 Save as PNG (with workflow)",
                            callback: () => pngSaver.export(true)
                        },
                        {
                            content: "🖼️ Save as PNG (image only)",
                            callback: () => pngSaver.export(false)
                        },
                        {
                            content: "💾 Save as JSON",
                            callback: () => JSONWorkflowSaver.save()
                        },
                        {
                            // ИСПРАВЛЕНО: теперь можно импортировать и PNG, и JSON
                            content: "📂 Import from PNG/JSON",
                            callback: () => PngWorkflowImage.import()
                        }
                    ]
                }
            });
            return options;
        };
    }
});