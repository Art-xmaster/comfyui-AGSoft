import { app } from "../../../scripts/app.js";

const NODE_CLASS_NAME = "AGSoft Image Crop Plus";

function addCustomUI(node) {
    if (node.widgets && node.widgets.some(w => w.name === "agsoft_crop_ui")) return;

    const container = document.createElement("div");
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 10px 0;
        width: 100%;
        height: calc(100% - 20px);
        box-sizing: border-box;
    `;

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

    // Контейнер занимает всё свободное место в ноде (flex-grow: 1)
    const imageContainer = document.createElement("div");
    imageContainer.style.cssText = `
        position: relative;
        width: 100%;
        flex-grow: 1;
        min-height: 150px;
        background-color: #1e1e1e;
        border-radius: 4px;
        overflow: hidden;
    `;

    // Изображение центрируется и сжимается до своих реальных пропорций (max-width/height)
    // Это гарантирует, что getBoundingClientRect() будет возвращать ТОЧНЫЕ координаты картинки
    const imgElement = document.createElement("img");
    imgElement.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 100%;
        max-height: 100%;
        display: block;
        cursor: crosshair;
        user-select: none;
    `;
    imgElement.alt = "Нет изображения / No image";
    imgElement.draggable = false;

    // Canvas абсолютно позиционируется и синхронизирует свои размеры с imgElement
    const canvas = document.createElement("canvas");
    canvas.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
    `;

    imageContainer.appendChild(imgElement);
    imageContainer.appendChild(canvas);
    container.appendChild(imageContainer);

    const controlBar = document.createElement("div");
    controlBar.style.cssText = `display: flex; gap: 10px; justify-content: center; margin-top: 5px;`;

    const resetButton = document.createElement("button");
    resetButton.textContent = "🗑️ Сбросить точки / Reset Points";
    resetButton.style.cssText = `
        padding: 5px 10px;
        background: #6c6c6c;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 11px;
    `;
    resetButton.onmouseenter = () => resetButton.style.background = "#7c7c7c";
    resetButton.onmouseleave = () => resetButton.style.background = "#6c6c6c";
    controlBar.appendChild(resetButton);
    container.appendChild(controlBar);

    const statusDiv = document.createElement("div");
    statusDiv.style.cssText = `font-size: 11px; color: #ccc; text-align: center; padding: 4px; background: #333; border-radius: 3px; margin-top: 5px;`;
    container.appendChild(statusDiv);

    const sizeDiv = document.createElement("div");
    sizeDiv.style.cssText = `font-size: 11px; color: #aaa; text-align: center; padding: 4px; background: #2a2a2a; border-radius: 3px; font-family: monospace;`;
    container.appendChild(sizeDiv);

    let points = [];
    let imageRect = null;
    let draggingPoint = null;
    let isDragging = false;

    function updateCropCoords() {
        const coordsWidget = node.widgets.find(w => w.name === "crop_coords");
        if (coordsWidget) {
            coordsWidget.value = JSON.stringify(points);
            if (coordsWidget.callback) coordsWidget.callback(coordsWidget.value);
            node.setDirtyCanvas(true);
        }
    }

    function clampPoint(point) {
        if (!imageRect) return point;
        return {
            x: Math.max(0, Math.min(imageRect.width, point.x)),
            y: Math.max(0, Math.min(imageRect.height, point.y))
        };
    }

    function getMousePos(e) {
        const rect = imgElement.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function screenToImage(screenX, screenY) {
        const rect = imgElement.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;
        return {
            x: screenX * (imageRect.width / rect.width),
            y: screenY * (imageRect.height / rect.height)
        };
    }

    function imageToScreen(imgX, imgY) {
        const rect = imgElement.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
        return {
            x: imgX * (rect.width / imageRect.width),
            y: imgY * (rect.height / imageRect.height)
        };
    }

    function drawOverlay() {
        if (!canvas || !imageRect || !imgElement.complete || imgElement.naturalWidth === 0) return;
        
        const displayWidth = imgElement.clientWidth;
        const displayHeight = imgElement.clientHeight;
        
        if (displayWidth === 0 || displayHeight === 0) return;
        
        canvas.style.width = displayWidth + 'px';
        canvas.style.height = displayHeight + 'px';
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, displayWidth, displayHeight);
        
        if (points.length === 0) return;
        
        const scaleX = displayWidth / imageRect.width;
        const scaleY = displayHeight / imageRect.height;
        
        if (points.length === 4) {
            const xs = points.map(p => p.x);
            const ys = points.map(p => p.y);
            const minX = Math.min(...xs);
            const minY = Math.min(...ys);
            const maxX = Math.max(...xs);
            const maxY = Math.max(...ys);
            
            ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
            ctx.fillRect(minX * scaleX, minY * scaleY, (maxX - minX) * scaleX, (maxY - minY) * scaleY);
            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(minX * scaleX, minY * scaleY, (maxX - minX) * scaleX, (maxY - minY) * scaleY);
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
    }

    function updateInfo() {
        if (!imageRect) {
            statusDiv.innerHTML = "⚠️ Нет изображения.";
            sizeDiv.innerHTML = "📐 Размер: --x-- px";
            return;
        }
        if (points.length === 0) statusDiv.innerHTML = "📌 Кликните на изображении (нужно 4 точки)";
        else if (points.length < 4) statusDiv.innerHTML = `✅ Добавлено ${points.length} из 4.`;
        else statusDiv.innerHTML = "✅ Прямоугольник готов!";
        
        if (points.length === 4) {
            const xs = points.map(p => p.x), ys = points.map(p => p.y);
            const w = Math.max(...xs) - Math.min(...xs);
            const h = Math.max(...ys) - Math.min(...ys);
            const multiple = parseInt(node.widgets.find(w => w.name === "multiple")?.value || 8);
            sizeDiv.innerHTML = `📐 Область: ${Math.round(w)}x${Math.round(h)} px <br>🎯 Выравнено (×${multiple}): ${Math.floor(w/multiple)*multiple}x${Math.floor(h/multiple)*multiple} px`;
        } else {
            sizeDiv.innerHTML = `📐 Изображение: ${imageRect.width}x${imageRect.height} px`;
        }
    }

    function findPointUnderCursor(clientX, clientY) {
        if (!imageRect || points.length === 0) return null;
        
        const rect = imgElement.getBoundingClientRect();
        const mouseX = clientX - rect.left;
        const mouseY = clientY - rect.top;
        
        if (mouseX < 0 || mouseX > rect.width || mouseY < 0 || mouseY > rect.height) return null;
        
        const thresholdScreen = 15;
        
        for (let i = 0; i < points.length; i++) {
            const pScreen = imageToScreen(points[i].x, points[i].y);
            const dx = pScreen.x - mouseX;
            const dy = pScreen.y - mouseY;
            if (Math.sqrt(dx * dx + dy * dy) <= thresholdScreen) return i;
        }
        return null;
    }

    function onMouseDown(e) {
        if (!imageRect) return;
        const pointIndex = findPointUnderCursor(e.clientX, e.clientY);
        if (pointIndex !== null) {
            draggingPoint = pointIndex;
            isDragging = true;
            imgElement.style.cursor = "grabbing";
            e.preventDefault();
            e.stopPropagation();
        }
    }

    function onMouseMove(e) {
        if (draggingPoint !== null && imageRect && isDragging) {
            const rect = imgElement.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const imgCoords = screenToImage(mouseX, mouseY);
            if (imgCoords) {
                points[draggingPoint] = clampPoint(imgCoords);
                drawOverlay();
                updateInfo();
                updateCropCoords();
            }
            e.preventDefault();
            e.stopPropagation();
        }
    }

    function onMouseUp(e) {
        if (draggingPoint !== null) {
            draggingPoint = null;
            imgElement.style.cursor = "crosshair";
            e.preventDefault();
            e.stopPropagation();
        }
        setTimeout(() => { isDragging = false; }, 10);
    }

    function onImageClick(e) {
        if (isDragging || draggingPoint !== null) return;
        if (!imageRect) return;
        
        const mouse = getMousePos(e);
        const imgCoords = screenToImage(mouse.x, mouse.y);
        if (!imgCoords) return;
        
        const clickX = Math.max(0, Math.min(imageRect.width, imgCoords.x));
        const clickY = Math.max(0, Math.min(imageRect.height, imgCoords.y));
        
        if (points.length < 4) points.push({ x: clickX, y: clickY });
        else points = [{ x: clickX, y: clickY }];
        
        drawOverlay();
        updateInfo();
        updateCropCoords();
    }

    function updateNodeSize() {
        if (node.onResize) node.onResize(node.size);
        node.setDirtyCanvas(true);
        setTimeout(() => { if (node.graph) node.graph.setDirtyCanvas(true); }, 50);
    }

    function loadImageByName(filename) {
        if (!filename) {
            imgElement.src = "";
            imgElement.alt = "Нет изображения / No image";
            imageRect = null;
            points = [];
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
            imageRect = { width: imgElement.naturalWidth, height: imgElement.naturalHeight };
            points = [];
            drawOverlay();
            updateInfo();
            updateCropCoords();
            
            const imgAspect = imageRect.width / imageRect.height;
            let newWidth = Math.min(560, imageRect.width + 60);
            if (newWidth < 300) newWidth = 300;
            let newHeight = (newWidth / imgAspect) + 350;
            if (newHeight < 450) newHeight = 450;
            
            node.setSize([newWidth, newHeight]);
            setTimeout(() => { updateNodeSize(); drawOverlay(); }, 50);
        };
        
        imgElement.onerror = () => {
            imgElement.alt = `Ошибка загрузки: ${filename}`;
            imageRect = null;
            points = [];
            drawOverlay();
            updateInfo();
            updateNodeSize();
        };
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('overwrite', 'false');
        statusDiv.innerHTML = "📤 Загрузка...";
        
        try {
            const response = await fetch('/upload/image', { method: 'POST', body: formData });
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
            }
        } catch (error) {
            statusDiv.innerHTML = "❌ Ошибка соединения";
        }
    }

    uploadButton.onclick = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = async (event) => { if (event.target.files[0]) await uploadImage(event.target.files[0]); };
        input.click();
    };

    const imageNameWidget = node.widgets.find(w => w.name === "image_name");
    if (imageNameWidget) {
        const originalCallback = imageNameWidget.callback;
        imageNameWidget.callback = (value) => {
            if (originalCallback) originalCallback(value);
            loadImageByName(value);
        };
        if (imageNameWidget.value) loadImageByName(imageNameWidget.value);
    }

    resetButton.onclick = () => { points = []; drawOverlay(); updateInfo(); updateCropCoords(); };

    const multipleWidget = node.widgets.find(w => w.name === "multiple");
    if (multipleWidget) {
        const originalCallback = multipleWidget.callback;
        multipleWidget.callback = (value) => { if (originalCallback) originalCallback(value); updateInfo(); drawOverlay(); };
    }

    imgElement.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    imgElement.addEventListener('click', onImageClick);

    const resizeObserver = new ResizeObserver(() => { drawOverlay(); updateNodeSize(); });
    resizeObserver.observe(imageContainer);
    window.addEventListener('resize', () => { drawOverlay(); updateNodeSize(); });

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
        if (imageNameWidget && imageNameWidget.value) loadImageByName(imageNameWidget.value);
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