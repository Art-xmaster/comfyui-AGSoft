import { app } from "../../../scripts/app.js";
const NODE_CLASS_NAME = "AGSoft Image Crop Plus";

function addCustomUI(node) {
    if (node.widgets && node.widgets.some(w => w.name === "agsoft_crop_ui")) return;

    // === КОНТЕЙНЕР ===
    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 10px 0;
        width: 100%;
        box-sizing: border-box;
    `;

    // === КНОПКА ЗАГРУЗКИ ===
    const uploadButton = document.createElement("button");
    uploadButton.textContent = "📁 Загрузить изображение / Upload Image";
    uploadButton.style.cssText = `
        padding: 8px 12px;
        background: #4a6a8a;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        font-weight: bold;
        transition: background 0.2s;
    `;
    uploadButton.onmouseenter = () => uploadButton.style.background = "#5a7a9a";
    uploadButton.onmouseleave = () => uploadButton.style.background = "#4a6a8a";
    container.appendChild(uploadButton);

    // === КОНТЕЙНЕР ИЗОБРАЖЕНИЯ ===
    const imageContainer = document.createElement("div");
    imageContainer.style.position = "relative";
    imageContainer.style.display = "block";
    imageContainer.style.width = "100%";
    imageContainer.style.marginTop = "5px";
    imageContainer.style.backgroundColor = "#1e1e1e";
    imageContainer.style.borderRadius = "4px";
    imageContainer.style.overflow = "hidden";

    const imgElement = document.createElement("img");
    imgElement.style.cssText = `
        width: 100%;
        height: auto;
        display: block;
        cursor: crosshair;
        user-select: none;
    `;
    imgElement.alt = "Нет изображения / No image";
    imgElement.draggable = false;

    const canvas = document.createElement("canvas");
    canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
        width: 100%;
        height: 100%;
    `;

    imageContainer.appendChild(imgElement);
    imageContainer.appendChild(canvas);
    container.appendChild(imageContainer);

    // === КНОПКА СБРОСА ===
    const resetButton = document.createElement("button");
    resetButton.textContent = "🗑️ Сбросить / Reset";
    resetButton.style.cssText = `
        padding: 5px 10px;
        background: #6c6c6c;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
        align-self: center;
    `;
    resetButton.onmouseenter = () => resetButton.style.background = "#7c7c7c";
    resetButton.onmouseleave = () => resetButton.style.background = "#6c6c6c";
    container.appendChild(resetButton);

    // === СТАТУС-БАР ===
    const statusDiv = document.createElement("div");
    statusDiv.style.cssText = `
        font-size: 11px;
        color: #ccc;
        text-align: center;
        padding: 4px;
        background: #333;
        border-radius: 3px;
        margin-top: 5px;
    `;
    container.appendChild(statusDiv);

    // === ИНФОРМАЦИЯ О РАЗМЕРАХ ===
    const sizeDiv = document.createElement("div");
    sizeDiv.style.cssText = `
        font-size: 11px;
        color: #aaa;
        text-align: center;
        padding: 4px;
        background: #2a2a2a;
        border-radius: 3px;
        font-family: monospace;
    `;
    container.appendChild(sizeDiv);

    // === СОСТОЯНИЕ ===
    let points = []; // Для режима Points
    let imageRect = null;
    let cropRect = { x: 0, y: 0, w: 0, h: 0 }; // Для режимов Preset/Manual
    let draggingPoint = null;
    let dragStart = null;
    let isDragging = false;

    // Для прямоугольника
    let rectDragType = null; // 'move', 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w'
    let rectDragStart = null;

    // === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
    function getWidgetValue(name) {
        const w = node.widgets.find(w => w.name === name);
        return w ? w.value : null;
    }

    function parseRatio(ratioStr) {
        const parts = ratioStr.split(':').map(Number);
        return parts[0] / parts[1];
    }

    function updateCropCoords() {
        const coordsWidget = node.widgets.find(w => w.name === "crop_coords");
        if (!coordsWidget) return;
        
        const mode = getWidgetValue("crop_mode");
        let data;
        
        if (mode === "Points (4 clicks)") {
            data = points;
        } else {
            data = {
                x: Math.round(cropRect.x),
                y: Math.round(cropRect.y),
                w: Math.round(cropRect.w),
                h: Math.round(cropRect.h)
            };
        }
        
        coordsWidget.value = JSON.stringify(data);
        if (coordsWidget.callback) coordsWidget.callback(coordsWidget.value);
        node.setDirtyCanvas(true);
    }

    function clampPoint(point) {
        if (!imageRect) return point;
        return {
            x: Math.max(0, Math.min(imageRect.width, point.x)),
            y: Math.max(0, Math.min(imageRect.height, point.y))
        };
    }

    function initCropRect() {
        if (!imageRect) return;
        const mode = getWidgetValue("crop_mode");
        
        if (mode === "Points (4 clicks)") {
            points = [];
            cropRect = { x: 0, y: 0, w: 0, h: 0 };
        } else {
            points = []; // Очищаем точки
            let targetW = imageRect.width * 0.7;
            let targetH = imageRect.height * 0.7;
            
            if (mode === "Preset Ratio") {
                const ratio = parseRatio(getWidgetValue("aspect_ratio") || "1:1");
                if (targetW / targetH > ratio) {
                    targetW = targetH * ratio;
                } else {
                    targetH = targetW / ratio;
                }
            } else if (mode === "Manual Size") {
                const mw = parseInt(getWidgetValue("manual_width")) || 512;
                const mh = parseInt(getWidgetValue("manual_height")) || 512;
                // Ограничиваем рамку вписыванием в изображение с сохранением пропорций
                const scale = Math.min(imageRect.width / mw, imageRect.height / mh, 1);
                targetW = mw * scale;
                targetH = mh * scale;
            }
            
            cropRect = {
                x: (imageRect.width - targetW) / 2,
                y: (imageRect.height - targetH) / 2,
                w: targetW,
                h: targetH
            };
        }
        updateCropCoords();
        drawOverlay();
        updateInfo();
    }

    // === ОТРИСОВКА ===
    function drawOverlay() {
        if (!canvas || !imageRect || !imgElement.complete || imgElement.naturalWidth === 0) return;
        
        const displayWidth = imgElement.clientWidth;
        const displayHeight = imgElement.clientHeight;
        if (displayWidth === 0 || displayHeight === 0) return;
        
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, displayWidth, displayHeight);
        
        const scaleX = displayWidth / imageRect.width;
        const scaleY = displayHeight / imageRect.height;
        const mode = getWidgetValue("crop_mode");

        if (mode === "Points (4 clicks)") {
            // === РЕЖИМ ТОЧЕК (как в оригинале) ===
            if (points.length === 4) {
                const xs = points.map(p => p.x);
                const ys = points.map(p => p.y);
                const minX = Math.min(...xs) * scaleX;
                const minY = Math.min(...ys) * scaleY;
                const maxW = (Math.max(...xs) - Math.min(...xs)) * scaleX; 
                const maxH = (Math.max(...ys) - Math.min(...ys)) * scaleY;
                
                ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
                ctx.fillRect(minX, minY, maxW, maxH);
                ctx.strokeStyle = "#00ff00";
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(minX, minY, maxW, maxH);
                ctx.setLineDash([]);
            }
            
            points.forEach((point, idx) => {
                const displayX = point.x * scaleX;
                const displayY = point.y * scaleY;
                const isDraggingPoint = draggingPoint === idx;
                
                ctx.beginPath();
                ctx.arc(displayX, displayY, isDraggingPoint ? 8 : 6, 0, 2 * Math.PI);
                ctx.fillStyle = idx === points.length - 1 ? "#ffaa00" : "#ff4444";
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = isDraggingPoint ? 3 : 2;
                ctx.stroke();
                
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 12px Arial";
                ctx.shadowBlur = 2;
                ctx.shadowColor = "black";
                ctx.fillText((idx + 1).toString(), displayX - 4, displayY - 8);
                ctx.shadowBlur = 0;
            });
        } else {
            // === РЕЖИМ ПРЯМОУГОЛЬНИКА ===
            if (cropRect.w <= 0 || cropRect.h <= 0) return;
            
            const rx = cropRect.x * scaleX;
            const ry = cropRect.y * scaleY;
            const rw = cropRect.w * scaleX;
            const rh = cropRect.h * scaleY;
            
            // Затемнение вне области
            ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
            ctx.fillRect(0, 0, displayWidth, ry);
            ctx.fillRect(0, ry + rh, displayWidth, displayHeight - (ry + rh));
            ctx.fillRect(0, ry, rx, rh);
            ctx.fillRect(rx + rw, ry, displayWidth - (rx + rw), rh);
            
            // Заполнение области
            ctx.fillStyle = "rgba(0, 170, 255, 0.15)";
            ctx.fillRect(rx, ry, rw, rh);
            
            // Контур
            ctx.strokeStyle = "#00aaff";
            ctx.lineWidth = 2;
            ctx.setLineDash([]);
            ctx.strokeRect(rx, ry, rw, rh);
            
            // Сетка правила третей
            ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(rx + rw / 3, ry); ctx.lineTo(rx + rw / 3, ry + rh);
            ctx.moveTo(rx + 2 * rw / 3, ry); ctx.lineTo(rx + 2 * rw / 3, ry + rh);
            ctx.moveTo(rx, ry + rh / 3); ctx.lineTo(rx + rw, ry + rh / 3);
            ctx.moveTo(rx, ry + 2 * rh / 3); ctx.lineTo(rx + rw, ry + 2 * rh / 3);
            ctx.stroke();
            
            // Маркеры
            const handles = [
                { x: rx, y: ry }, { x: rx + rw/2, y: ry }, { x: rx + rw, y: ry },
                { x: rx, y: ry + rh/2 }, { x: rx + rw, y: ry + rh/2 },
                { x: rx, y: ry + rh }, { x: rx + rw/2, y: ry + rh }, { x: rx + rw, y: ry + rh }
            ];
            handles.forEach(h => {
                ctx.beginPath();
                ctx.arc(h.x, h.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = "#ffffff";
                ctx.fill();
                ctx.strokeStyle = "#00aaff";
                ctx.lineWidth = 2;
                ctx.stroke();
            });
            
            // Текст размера в центре
            const multiple = parseInt(getWidgetValue("multiple")) || 8;
            const alignedW = Math.floor(cropRect.w / multiple) * multiple;
            const alignedH = Math.floor(cropRect.h / multiple) * multiple;
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.shadowBlur = 3;
            ctx.shadowColor = "black";
            ctx.fillText(`${Math.round(alignedW)}×${Math.round(alignedH)}`, rx + rw/2, ry + rh/2);
            ctx.shadowBlur = 0;
            ctx.textAlign = "start";
            ctx.textBaseline = "alphabetic";
        }
    }

    // === ИНФОРМАЦИЯ ===
    function updateInfo() {
        if (!imageRect) {
            statusDiv.innerHTML = "⚠️ Нет изображения. Загрузите файл, перетащите его сюда или выберите из списка.";
            sizeDiv.innerHTML = "📐 Размер: --x-- px";
            return;
        }
        
        const mode = getWidgetValue("crop_mode");
        const multiple = parseInt(getWidgetValue("multiple")) || 8;
        
        if (mode === "Points (4 clicks)") {
            if (points.length === 0) {
                statusDiv.innerHTML = "📌 Кликните на изображении, чтобы добавить точки (нужно 4 точки)";
            } else if (points.length < 4) {
                statusDiv.innerHTML = `✅ Добавлено ${points.length} из 4 точек. Осталось: ${4 - points.length}`;
            } else {
                statusDiv.innerHTML = "✅ Прямоугольник готов! Нажмите Queue Prompt для обрезки.";
            }
            
            if (points.length === 4) {
                const xs = points.map(p => p.x);
                const ys = points.map(p => p.y);
                const w = Math.max(...xs) - Math.min(...xs);
                const h = Math.max(...ys) - Math.min(...ys);
                const alignedW = Math.floor(w / multiple) * multiple;
                const alignedH = Math.floor(h / multiple) * multiple;
                sizeDiv.innerHTML = `📐 Область: ${Math.round(w)}x${Math.round(h)} px <br>🎯 После выравнивания (×${multiple}): ${alignedW}x${alignedH} px`;
            } else {
                sizeDiv.innerHTML = `📐 Изображение: ${imageRect.width}x${imageRect.height} px`;
            }
        } else {
            statusDiv.innerHTML = "🖱️ Перетащите прямоугольник за центр или потяните за углы/грани.";
            const alignedW = Math.floor(cropRect.w / multiple) * multiple;
            const alignedH = Math.floor(cropRect.h / multiple) * multiple;
            sizeDiv.innerHTML = `📐 Область: ${Math.round(cropRect.w)}x${Math.round(cropRect.h)} px <br>🎯 После выравнивания (×${multiple}): ${alignedW}x${alignedH} px`;
        }
    }

    // === ПОИСК ТОЧКИ ПОД КУРСОРОМ (для режима Points) ===
    function findPointUnderCursor(clientX, clientY) {
        if (!imageRect || points.length === 0) return null;
        
        const rect = imgElement.getBoundingClientRect();
        const scaleX = imageRect.width / rect.width;
        const scaleY = imageRect.height / rect.height;
        
        const mouseX = (clientX - rect.left) * scaleX;
        const mouseY = (clientY - rect.top) * scaleY;
        
        const threshold = 20;
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const dx = p.x - mouseX;
            const dy = p.y - mouseY;
            if (Math.sqrt(dx * dx + dy * dy) <= threshold) return i;
        }
        return null;
    }

    // === ПОИСК МАРКЕРА ПРЯМОУГОЛЬНИКА (для режимов Preset/Manual) ===
    function findRectHandleAt(mouseImgX, mouseImgY) {
        const threshold = 20; // в координатах изображения
        const { x, y, w, h } = cropRect;
        
        // Углы
        if (Math.hypot(mouseImgX - x, mouseImgY - y) <= threshold) return 'nw';
        if (Math.hypot(mouseImgX - (x + w), mouseImgY - y) <= threshold) return 'ne';
        if (Math.hypot(mouseImgX - x, mouseImgY - (y + h)) <= threshold) return 'sw';
        if (Math.hypot(mouseImgX - (x + w), mouseImgY - (y + h)) <= threshold) return 'se';
        
        // Середины граней
        if (Math.hypot(mouseImgX - (x + w/2), mouseImgY - y) <= threshold) return 'n';
        if (Math.hypot(mouseImgX - (x + w/2), mouseImgY - (y + h)) <= threshold) return 's';
        if (Math.hypot(mouseImgX - x, mouseImgY - (y + h/2)) <= threshold) return 'w';
        if (Math.hypot(mouseImgX - (x + w), mouseImgY - (y + h/2)) <= threshold) return 'e';
        
        // Внутри прямоугольника - перемещение
        if (mouseImgX >= x && mouseImgX <= x + w && mouseImgY >= y && mouseImgY <= y + h) {
            return 'move';
        }
        
        return null;
    }

    function getMouseImageCoords(clientX, clientY) {
        const rect = imgElement.getBoundingClientRect();
        const scaleX = imageRect.width / rect.width;
        const scaleY = imageRect.height / rect.height;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    // === ОБРАБОТЧИКИ (ОБЩИЕ ДЛЯ ВСЕХ РЕЖИМОВ) ===
    function onMouseDown(e) {
        if (!imageRect) return;
        const mode = getWidgetValue("crop_mode");
        
        if (mode === "Points (4 clicks)") {
            // Логика точек
            const pointIndex = findPointUnderCursor(e.clientX, e.clientY);
            if (pointIndex !== null) {
                draggingPoint = pointIndex;
                isDragging = true;
                dragStart = {
                    x: e.clientX,
                    y: e.clientY,
                    originalPoint: { ...points[pointIndex] }
                };
                imgElement.style.cursor = "grabbing";
                e.preventDefault();
                e.stopPropagation();
            }
        } else {
            // Логика прямоугольника
            const mouse = getMouseImageCoords(e.clientX, e.clientY);
            const handle = findRectHandleAt(mouse.x, mouse.y);
            if (handle) {
                rectDragType = handle;
                rectDragStart = {
                    mouseX: mouse.x,
                    mouseY: mouse.y,
                    rect: { ...cropRect }
                };
                isDragging = true;
                imgElement.style.cursor = handle === 'move' ? "grabbing" : "crosshair";
                e.preventDefault();
                e.stopPropagation();
            }
        }
    }

    function onMouseMove(e) {
        if (!isDragging || !imageRect) return;
        const mode = getWidgetValue("crop_mode");
        
        if (mode === "Points (4 clicks)" && draggingPoint !== null && dragStart) {
            // Перемещение точки
            const rect = imgElement.getBoundingClientRect();
            const scaleX = imageRect.width / rect.width;
            const scaleY = imageRect.height / rect.height;
            const deltaX = (e.clientX - dragStart.x) * scaleX;
            const deltaY = (e.clientY - dragStart.y) * scaleY;
            
            let newPoint = { 
                x: dragStart.originalPoint.x + deltaX,
                y: dragStart.originalPoint.y + deltaY
            };
            newPoint = clampPoint(newPoint);
            points[draggingPoint] = newPoint;
            
            drawOverlay();
            updateInfo();
            updateCropCoords();
            e.preventDefault();
            e.stopPropagation();
        } else if (mode !== "Points (4 clicks)" && rectDragType && rectDragStart) {
            // Перемещение/ресайз прямоугольника
            const mouse = getMouseImageCoords(e.clientX, e.clientY);
            const dx = mouse.x - rectDragStart.mouseX;
            const dy = mouse.y - rectDragStart.mouseY;
            const orig = rectDragStart.rect;
            
            let x = orig.x, y = orig.y, w = orig.w, h = orig.h;
            const isPreset = mode === "Preset Ratio";
            const ratio = isPreset ? parseRatio(getWidgetValue("aspect_ratio")) : null;
            
            if (rectDragType === 'move') {
                x = orig.x + dx;
                y = orig.y + dy;
            } else {
                // Ресайз
                if (rectDragType.includes('e')) w = orig.w + dx;
                if (rectDragType.includes('w')) { x = orig.x + dx; w = orig.w - dx; }
                if (rectDragType.includes('s')) h = orig.h + dy; 
                if (rectDragType.includes('n')) { y = orig.y + dy; h = orig.h - dy; }
                
                // Сохранение пропорций для пресета
                if (isPreset && ratio) {
                    if (rectDragType === 'n' || rectDragType === 's') {
                        const newW = h * ratio;
                        x = orig.x + (orig.w - newW) / 2;
                        w = newW;
                    } else if (rectDragType === 'e' || rectDragType === 'w') {
                        const newH = w / ratio;
                        y = orig.y + (orig.h - newH) / 2;
                        h = newH;
                    } else {
                        const newWFromH = h * ratio;
                        const newHFromW = w / ratio;
                        if (Math.abs(dx) > Math.abs(dy)) {
                            h = w / ratio;
                        } else {
                            w = h * ratio;
                        }
                        if (rectDragType.includes('w')) x = orig.x + orig.w - w;
                        if (rectDragType.includes('n')) y = orig.y + orig.h - h;
                    }
                }
                
                if (w < 32) w = 32;
                if (h < 32) h = 32;
            }
            
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (x + w > imageRect.width) w = imageRect.width - x;
            if (y + h > imageRect.height) h = imageRect.height - y;
            
            cropRect = { x, y, w, h };
            drawOverlay();
            updateInfo();
            updateCropCoords();
            e.preventDefault();
            e.stopPropagation();
        }
    }

    function onMouseUp(e) {
        if (draggingPoint !== null) {
            draggingPoint = null;
            dragStart = null;
            imgElement.style.cursor = "crosshair";
            e.preventDefault();
            e.stopPropagation();
        }
        if (rectDragType !== null) {
            rectDragType = null;
            rectDragStart = null;
            imgElement.style.cursor = "crosshair";
        }
        setTimeout(() => { isDragging = false; }, 10);
    }

    // === КЛИК (только для режима Points) ===
    function onImageClick(e) {
        if (isDragging || draggingPoint !== null) return;
        
        const mode = getWidgetValue("crop_mode");
        if (mode !== "Points (4 clicks)") return;
        
        if (!imageRect) {
            alert("Сначала загрузите изображение!");
            return;
        }
        
        const rect = imgElement.getBoundingClientRect();
        const scaleX = imageRect.width / rect.width;
        const scaleY = imageRect.height / rect.height;
        
        let clickX = (e.clientX - rect.left) * scaleX;
        let clickY = (e.clientY - rect.top) * scaleY;
        
        clickX = Math.max(0, Math.min(imageRect.width, clickX));
        clickY = Math.max(0, Math.min(imageRect.height, clickY));
        
        if (points.length < 4) {
            points.push({ x: clickX, y: clickY });
        } else {
            points = [{ x: clickX, y: clickY }];
        }
        
        drawOverlay();
        updateInfo();
        updateCropCoords();
    }

    // === ОБНОВЛЕНИЕ РАЗМЕРА НОДЫ ===
    function updateNodeSize() {
        if (node.onResize) node.onResize(node.size);
        node.setDirtyCanvas(true);
        setTimeout(() => {
            if (node.graph) node.graph.setDirtyCanvas(true);
        }, 50);
    }

    // === ЗАГРУЗКА ИЗОБРАЖЕНИЯ ===
    function loadImageByName(filename) {
        if (!filename) {
            imgElement.src = "";
            imgElement.alt = "Нет изображения / No image";
            imageRect = null;
            points = [];
            cropRect = { x: 0, y: 0, w: 0, h: 0 };
            drawOverlay();
            updateInfo();
            updateCropCoords();
            updateNodeSize();
            return;
        }
        
        const imageUrl = `/view?filename=${encodeURIComponent(filename)}&type=input`;
        imgElement.src = imageUrl;
        imgElement.alt = "Загрузка...";
        
        imgElement.onload = () => {
            imageRect = {
                width: imgElement.naturalWidth,
                height: imgElement.naturalHeight
            };
            points = [];
            initCropRect();
            
            const imgAspect = imageRect.width / imageRect.height;
            const maxImageWidth = 500;
            let newWidth = Math.min(maxImageWidth + 60, imageRect.width + 60);
            if (newWidth < 300) newWidth = 300;
            
            let imageDisplayHeight = newWidth / imgAspect;
            let widgetsHeight = 250;
            let topPadding = 90;
            let bottomPadding = 50; 
            let newHeight = imageDisplayHeight + widgetsHeight + topPadding + bottomPadding;
            if (newHeight < 450) newHeight = 450;
            
            node.setSize([newWidth, newHeight]);
            
            setTimeout(() => {
                updateNodeSize();
                drawOverlay();
            }, 50);
        };
        
        imgElement.onerror = () => {
            console.error(`Не удалось загрузить: ${imageUrl}`);
            imgElement.alt = `Ошибка загрузки: ${filename}`;
            imageRect = null;
            points = [];
            cropRect = { x: 0, y: 0, w: 0, h: 0 };
            drawOverlay();
            updateInfo();
            updateNodeSize();
        };
    }

    // === ЗАГРУЗКА ФАЙЛА ===
    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('overwrite', 'false');
        statusDiv.innerHTML = "📤 Загрузка...";
        
        try {
            const response = await fetch('/upload/image', {
                method: 'POST',
                body: formData
            });
            
            if (response.status === 200) {
                const data = await response.json();
                if (data && data.name) {
                    const imageNameWidget = node.widgets.find(w => w.name === "image_name");
                    if (imageNameWidget) {
                        imageNameWidget.value = data.name;
                        if (imageNameWidget.callback) imageNameWidget.callback(data.name);
                    }
                    loadImageByName(data.name);
                    statusDiv.innerHTML = "✅ Загружено!";
                    setTimeout(() => updateInfo(), 1000);
                }
            } else {
                statusDiv.innerHTML = "❌ Ошибка загрузки";
            }
        } catch (error) {
            statusDiv.innerHTML = "❌ Ошибка соединения";
            console.error("Upload error:", error);
        }
    }

    // === ПРИВЯЗКА СОБЫТИЙ ===
    uploadButton.onclick = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/jpeg,image/png,image/webp,image/bmp';
        input.onchange = async (event) => {
            const file = event.target.files[0];
            if (file) await uploadImage(file);
        };
        input.click();
    };

    // ==========================================================
    // === ДОБАВЛЕНО: ПОДДЕРЖКА DRAG & DROP ===
    // ==========================================================
    imageContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        imageContainer.style.border = "2px dashed #4a6a8a";
        imageContainer.style.backgroundColor = "#2a3a4a";
    });

    imageContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        imageContainer.style.border = "none";
        imageContainer.style.backgroundColor = "#1e1e1e";
    });

    imageContainer.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        imageContainer.style.border = "none";
        imageContainer.style.backgroundColor = "#1e1e1e";

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                await uploadImage(file);
            } else {
                statusDiv.innerHTML = "❌ Поддерживаются только изображения! / Images only!";
                setTimeout(() => updateInfo(), 2000);
            }
        }
    });
    // ==========================================================

    const imageNameWidget = node.widgets.find(w => w.name === "image_name");
    if (imageNameWidget) {
        const originalCallback = imageNameWidget.callback;
        imageNameWidget.callback = (value) => {
            if (originalCallback) originalCallback(value);
            loadImageByName(value);
        };
        if (imageNameWidget.value) loadImageByName(imageNameWidget.value);
    }

    // Кнопка сброса - прямая привязка
    resetButton.onclick = () => {
        const mode = getWidgetValue("crop_mode");
        if (mode === "Points (4 clicks)") {
            points = [];
        } else {
            initCropRect();
            return;
        }
        drawOverlay();
        updateInfo();
        updateCropCoords();
    };

    // Обработчики смены виджетов - сброс состояния
    ['crop_mode', 'aspect_ratio', 'manual_width', 'manual_height'].forEach(name => {
        const w = node.widgets.find(w => w.name === name);
        if (w) {
            const originalCallback = w.callback;
            w.callback = (value) => {
                if (originalCallback) originalCallback(value);
                if (imageRect) initCropRect();
            };
        }
    });

    const multipleWidget = node.widgets.find(w => w.name === "multiple");
    if (multipleWidget) {
        const originalCallback = multipleWidget.callback;
        multipleWidget.callback = (value) => {
            if (originalCallback) originalCallback(value);
            updateInfo();
            drawOverlay();
        };
    }

    // Обработчики мыши
    imgElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    imgElement.addEventListener('click', onImageClick);

    const resizeObserver = new ResizeObserver(() => {
        drawOverlay();
        updateNodeSize();
    });
    resizeObserver.observe(imgElement);
    window.addEventListener('resize', () => {
        drawOverlay();
        updateNodeSize();
    });

    // === РЕГИСТРАЦИЯ ВИДЖЕТА ===
    node.addDOMWidget("agsoft_crop_ui", "agsoft_crop_ui", container, {
        getValue() { return points; },
        setValue(value) {
            if (value && Array.isArray(value)) {
                points = value;
                drawOverlay();
                updateInfo();
                updateNodeSize();
            }
        }
    });

    setTimeout(() => {
        updateNodeSize();
        if (imageNameWidget && imageNameWidget.value) {
            loadImageByName(imageNameWidget.value);
        }
    }, 100);
}

app.registerExtension({
    name: "AGSoft.ImageCropPlus",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== NODE_CLASS_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            addCustomUI(this);
            return result;
        };
        
        const computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function() {
            let size = computeSize ? computeSize.apply(this, arguments) : [200, 100];
            const customWidget = this.widgets?.find(w => w.name === "agsoft_crop_ui");
            if (customWidget && this.domWidgets) {
                const widgetElement = this.domWidgets.find(w => w.name === "agsoft_crop_ui");
                if (widgetElement && widgetElement.element) {
                    const elementHeight = widgetElement.element.clientHeight;
                    if (elementHeight > 0) size[1] = Math.max(size[1], elementHeight + 50);
                }
            }
            return size;
        };
    }
});