// AGSoft Save Workflow Image - JavaScript Extension for ComfyUI
// Файл: ComfyUI/custom_nodes/comfyui-AGSoft/web/js/AGSoft_Save_workflowImage.js
// Версия: 1.0
// Описание: Расширение для сохранения рабочей области (workflow) в PNG с возможностью
//           экспорта/импорта workflow и сохранения в JSON

import { app } from "../../../scripts/app.js";

// Глобальные переменные для работы с файлами и отрисовкой
let fileInput = null;
let getDrawTextConfig = null;

/**
 * Базовый класс для работы с изображениями workflow
 * Содержит основную логику: расчет границ, сохранение/восстановление состояния канваса,
 * отрисовку и экспорт
 */
class WorkflowImage {
    static accept = "";

    /**
     * Вычисляет границы всего графа (всех нод) с отступами
     * @returns {Array} [left, top, right, bottom] - границы рабочей области
     */
    getBounds() {
        const bounds = app.graph._nodes.reduce(
            (p, n) => {
                // Находим минимальные координаты (левый верхний угол)
                if (n.pos[0] < p[0]) p[0] = n.pos[0];
                if (n.pos[1] < p[1]) p[1] = n.pos[1];
                
                // Находим максимальные координаты (правый нижний угол) с учетом размеров нод
                const nodeBounds = n.getBounding();
                const r = n.pos[0] + nodeBounds[2];
                const b = n.pos[1] + nodeBounds[3];
                if (r > p[2]) p[2] = r;
                if (b > p[3]) p[3] = b;
                return p;
            },
            [99999, 99999, -99999, -99999] // Начальные значения для поиска min/max
        );

        // Добавляем отступы для красивого оформления
        bounds[0] -= 150; // Отступ слева
        bounds[1] -= 150; // Отступ сверху
        bounds[2] += 150; // Отступ справа
        bounds[3] += 150; // Отступ снизу
        return bounds;
    }

    /**
     * Сохраняет текущее состояние канваса (масштаб, размеры, смещение, трансформацию)
     * Нужно для восстановления после экспорта
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
     * Восстанавливает сохраненное состояние канваса
     */
    restoreState() {
        app.canvas.ds.scale = this.state.scale;
        app.canvas.canvas.width = this.state.width;
        app.canvas.canvas.height = this.state.height;
        app.canvas.ds.offset = this.state.offset;
        app.canvas.canvas.getContext("2d").setTransform(this.state.transform);
    }

    /**
     * Обновляет вид канваса для захвата всей рабочей области
     * Устанавливает оптимальный масштаб с учетом максимального размера изображения
     * @param {Array} bounds - границы рабочей области
     */
    updateView(bounds) {
        const w = bounds[2] - bounds[0]; // Ширина рабочей области
        const h = bounds[3] - bounds[1]; // Высота рабочей области
        const MAX_DIM = 8192; // Максимальный размер изображения (ограничение браузера)
        const TARGET_SCALE = 1.2; // Желаемый масштаб для хорошей читаемости
        
        let scale = TARGET_SCALE;

        // Если желаемый масштаб превышает максимальный размер, уменьшаем его
        if (w * scale > MAX_DIM || h * scale > MAX_DIM) {
            scale = Math.min(MAX_DIM / w, MAX_DIM / h);
        }

        // Устанавливаем параметры для захвата
        app.canvas.ds.scale = 1;
        app.canvas.canvas.width = w * scale;
        app.canvas.canvas.height = h * scale;
        app.canvas.ds.offset = [-bounds[0], -bounds[1]];
        app.canvas.canvas.getContext("2d").setTransform(scale, 0, 0, scale, 0, 0);
    }

    /**
     * Основной метод экспорта
     * @param {boolean} includeWorkflow - включать ли workflow в изображение
     */
    async export(includeWorkflow) {
        // Сохраняем текущее состояние канваса
        this.saveState();
        
        // Обновляем вид для захвата всей рабочей области
        const bounds = this.getBounds();
        console.log("[AGSoft] Bounds for export:", bounds);
        this.updateView(bounds);
        
        // Принудительно перерисовываем все ноды
        app.graph.setDirtyCanvas(true, true);
        app.canvas.setDirty(true, true);

        // Устанавливаем callback для отрисовки текста и рендерим канвас
        getDrawTextConfig = this.getDrawTextConfig;
        app.canvas.draw(true, true);
        
        // Ждем загрузки асинхронных текстур и рендерим снова
        await new Promise(r => setTimeout(r, 500));
        app.canvas.draw(true, true);

        // Ручная отрисовка изображений (стандартный рендер может их пропустить)
        const ctx = app.canvas.ctx;
        const offsetX = app.canvas.ds.offset[0];
        const offsetY = app.canvas.ds.offset[1];

        // Проходим по всем нодам и отрисовываем их изображения
        for (const node of app.graph._nodes) {
            if (node.imgs && node.imgs.length > 0) {
                const img = node.imgs[0];
                if (img.complete && img.width > 0) {
                    this.drawNodeImage(ctx, node, img, offsetX, offsetY);
                }
            }
        }
        
        getDrawTextConfig = null;

        // Генерируем blob изображения (с workflow или без)
        const blob = await this.getBlob(
            includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined
        );
        console.log("[AGSoft] Generated blob size:", blob ? blob.size : "null");

        // Восстанавливаем исходное состояние канваса
        this.restoreState();
        app.canvas.draw(true, true);

        // Скачиваем полученное изображение
        this.download(blob);
    }

    /**
     * Отрисовывает изображение внутри ноды с правильным позиционированием
     * @param {CanvasRenderingContext2D} ctx - контекст для рисования
     * @param {Object} node - нода ComfyUI
     * @param {HTMLImageElement} img - изображение для отрисовки
     * @param {number} offsetX - смещение по X
     * @param {number} offsetY - смещение по Y
     */
    drawNodeImage(ctx, node, img, offsetX, offsetY) {
        ctx.save();
        ctx.translate(node.pos[0] + offsetX, node.pos[1] + offsetY);
        
        // Рассчитываем область для изображения внутри ноды
        let contentStartY = 0;
        if (node.widgets && node.widgets.length > 0) {
            const headerHeight = 26; // Высота заголовка ноды
            let widgetsHeight = 0;
            
            for (const w of node.widgets) {
                widgetsHeight += 28; // Высота одного виджета
            }
            
            contentStartY = headerHeight + widgetsHeight + 5;
        }
        
        // Особые случаи для разных типов нод
        const isSampler = node.type && node.type.includes("Sampler");
        const isSave = node.type && (node.type.includes("Save") || node.type.includes("Output") || node.type.includes("Preview"));
        
        // Для сэмплеров изображение обычно ниже
        if (isSampler && contentStartY < 220) contentStartY = 220;

        const footerHeight = isSave ? 55 : 25;
        const drawHeight = node.size[1] - contentStartY - footerHeight;
        
        if (drawHeight > 0) {
            try {
                // Подбираем размер изображения чтобы оно вписалось в область с сохранением пропорций
                const imgAspect = img.width / img.height;
                const areaAspect = node.size[0] / drawHeight;
                
                let targetWidth, targetHeight, offsetXImg, offsetYImg;
                
                if (imgAspect > areaAspect) {
                    // Изображение шире области - подгоняем по ширине
                    targetWidth = node.size[0];
                    targetHeight = node.size[0] / imgAspect;
                    offsetXImg = 0;
                    offsetYImg = contentStartY + (drawHeight - targetHeight) / 2;
                } else {
                    // Изображение выше области - подгоняем по высоте
                    targetHeight = drawHeight;
                    targetWidth = drawHeight * imgAspect;
                    offsetXImg = (node.size[0] - targetWidth) / 2;
                    offsetYImg = contentStartY;
                }
                
                ctx.drawImage(img, offsetXImg, offsetYImg, targetWidth, targetHeight);
            } catch (e) {
                console.error("[AGSoft] Failed to draw image for node", node.id, e);
            }
        }
        ctx.restore();
    }

    /**
     * Скачивает сгенерированный blob как файл
     * @param {Blob} blob - данные для скачивания
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
        setTimeout(function () {
            a.remove();
            window.URL.revokeObjectURL(url);
        }, 0);
    }

    /**
     * Callback для отрисовки текста (используется при рендеринге)
     * @param {CanvasRenderingContext2D} ctx - контекст для рисования
     * @param {Object} widget - виджет для отрисовки
     * @returns {Object} конфигурация отрисовки
     */
    getDrawTextConfig(_, widget) {
        return {
            x: 10,
            y: widget.last_y + 10,
            resetTransform: false,
        };
    }

    /**
     * Импорт изображения с workflow
     */
    static import() {
        if (!fileInput) {
            fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                onchange: () => {
                    app.handleFile(fileInput.files[0]);
                },
            });
            document.body.append(fileInput);
        }
        fileInput.accept = WorkflowImage.accept;
        fileInput.click();
    }
}

/**
 * Класс для работы с PNG форматом
 * Содержит методы для встраивания workflow в PNG чанки
 */
class PngWorkflowImage extends WorkflowImage {
    static accept = ".png,image/png";
    extension = "png";

    /**
     * Конвертирует число в 4-байтовый массив (big-endian)
     * @param {number} n - число для конвертации
     * @returns {Uint8Array} 4-байтовое представление
     */
    n2b(n) {
        return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
    }

    /**
     * Объединяет несколько ArrayBuffer в один
     * @param {...ArrayBuffer} bufs - буферы для объединения
     * @returns {Uint8Array} объединенный буфер
     */
    joinArrayBuffer(...bufs) {
        const result = new Uint8Array(
            bufs.reduce((totalSize, buf) => totalSize + buf.byteLength, 0)
        );
        bufs.reduce((offset, buf) => {
            result.set(buf, offset);
            return offset + buf.byteLength;
        }, 0);
        return result;
    }

    /**
     * Вычисляет CRC32 для данных PNG чанка
     * @param {Uint8Array} data - данные для вычисления CRC
     * @returns {number} CRC32 хеш
     */
    crc32(data) {
        // Кэшируем таблицу CRC для производительности
        const crcTable =
            PngWorkflowImage.crcTable ||
            (PngWorkflowImage.crcTable = (() => {
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
     * Получает blob PNG с внедренным workflow (если указан)
     * @param {string} workflow - JSON строка с workflow
     * @returns {Promise<Blob>} PNG blob
     */
    async getBlob(workflow) {
        return new Promise((r) => {
            app.canvasEl.toBlob(async (blob) => {
                if (workflow) {
                    // Внедряем workflow в PNG как tEXt чанк
                    const buffer = await blob.arrayBuffer();
                    const typedArr = new Uint8Array(buffer);
                    const view = new DataView(buffer);

                    // Создаем tEXt чанк с workflow
                    // Формат: tEXtключ\0значение
                    const data = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
                    
                    // ВАЖНО: data.length - 4, потому что длина чанка не включает 4 байта типа чанка (tEXt)
                    // Это стандарт PNG: длина считается только для данных, тип идет отдельно
                    const chunk = this.joinArrayBuffer(
                        this.n2b(data.length - 4),  // Длина данных (без учета "tEXt")
                        data,                        // Данные (включая "tEXt" и само значение)
                        this.n2b(this.crc32(data))   // CRC32 от данных
                    );

                    // Находим позицию после заголовка PNG и всех чанков до IEND
                    const sz = view.getUint32(8) + 20;
                    
                    // Вставляем наш чанк перед последним чанком IEND
                    const result = this.joinArrayBuffer(
                        typedArr.subarray(0, sz),  // Все до нового чанка
                        chunk,                       // Новый чанк с workflow
                        typedArr.subarray(sz)        // Остаток (должен быть IEND)
                    );

                    blob = new Blob([result], { type: "image/png" });
                }

                r(blob);
            });
        });
    }
}

/**
 * Класс для сохранения workflow в отдельный JSON файл
 */
class JSONWorkflowSaver {
    /**
     * Сохраняет текущий workflow в JSON файл
     */
    static save() {
        try {
            // Сериализуем граф
            const workflow = app.graph.serialize();
            const jsonStr = JSON.stringify(workflow, null, 2);
            
            // Генерируем имя с timestamp
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const filename = `workflow_${timestamp}.json`;
            
            // Создаем и скачиваем файл
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log("[AGSoft] JSON saved:", filename);
            
        } catch (error) {
            console.error("[AGSoft] Error saving JSON:", error);
        }
    }
}

// Создаем экземпляр для работы с PNG
const pngSaver = new PngWorkflowImage();

// Регистрируем расширение в ComfyUI
app.registerExtension({
    name: "AGSoft.SaveWorkflow",
    
    /**
     * Настройка расширения при загрузке
     * Добавляет пункты меню в контекстное меню рабочей области
     */
    setup() {
        // Сохраняем оригинальный метод получения опций меню
        const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
        
        // Переопределяем метод для добавления своих пунктов
        LGraphCanvas.prototype.getCanvasMenuOptions = function() {
            // Получаем стандартные опции
            const options = orig.apply(this, arguments);
            
            // Добавляем разделитель и наше подменю
            options.push(null, {
                content: "📸 AGSoft Workflow Image", // Название в меню
                submenu: {
                    options: [
                        {
                            content: "📸 Save as PNG (with workflow)", // Сохранить с workflow
                            callback: () => {
                                console.log("[AGSoft] Saving PNG with workflow...");
                                pngSaver.export(true);
                            }
                        },
                        {
                            content: "🖼️ Save as PNG (image only)", // Сохранить только изображение
                            callback: () => {
                                console.log("[AGSoft] Saving PNG image only...");
                                pngSaver.export(false);
                            }
                        },
                        {
                            content: "💾 Save as JSON", // Сохранить как JSON
                            callback: () => {
                                console.log("[AGSoft] Saving JSON...");
                                JSONWorkflowSaver.save();
                            }
                        },
                        {
                            content: "📂 Import from PNG", // Импорт из PNG
                            callback: () => {
                                console.log("[AGSoft] Importing workflow from PNG...");
                                PngWorkflowImage.import();
                            }
                        }
                    ]
                }
            });
            
            return options;
        };
        
        console.log("[AGSoft] ✅ Menu extension loaded successfully");
        console.log("[AGSoft] ✅ PNG workflow embedding ready");
    }
});