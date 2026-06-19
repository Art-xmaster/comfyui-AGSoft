// agsoft_image_compare.js
// Автор: AGSoft
// Дата: 14 июня 2026 г.
// Описание: JavaScript расширение для динамического создания входов ноды AGSoft Image Compare
import { app } from "../../../scripts/app.js";

function drawImageFit(ctx, img, areaX, areaY, areaW, areaH, zoom, panX, panY) {
    if (!img || !img.complete || img.width === 0 || areaW <= 0 || areaH <= 0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(areaX, areaY, areaW, areaH);
    ctx.clip();

    const scaleX = areaW / img.width;
    const scaleY = areaH / img.height;
    const scale = Math.min(scaleX, scaleY);

    const drawW = img.width * scale * zoom;
    const drawH = img.height * scale * zoom;
    const centerX = areaX + areaW / 2;
    const centerY = areaY + areaH / 2;
    const dx = centerX - (drawW / 2) + panX;
    const dy = centerY - (drawH / 2) + panY;

    ctx.drawImage(img, dx, dy, drawW, drawH);
    ctx.restore();
}

app.registerExtension({
    name: "AGSoft.ImageCompare",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AGSoftImageCompare") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.ag_img1 = null;
                this.ag_img2 = null;
                this.ag_sliderPos = 0.5;
                this.ag_isDragging = false;
                this.ag_hideImg2 = false;
                this.setSize([450, 450]);
                return r;
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                
                const loadImage = (imgArray, imgProp) => {
                    if (imgArray && imgArray.length > 0) {
                        const imgData = imgArray[0];
                        const url = `/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${encodeURIComponent(imgData.subfolder)}&t=${Date.now()}`;
                        const img = new Image();
                        img.onload = () => {
                            this[imgProp] = img;
                            app.graph.setDirtyCanvas(true, true);
                        };
                        img.src = url;
                    } else {
                        this[imgProp] = null;
                    }
                };
                
                loadImage(message.image_1, "ag_img1");
                loadImage(message.image_2, "ag_img2");
                return r;
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
                
                if ((this.flags && this.flags.collapsed) || (!this.ag_img1 && !this.ag_img2)) return r;
                
                const getVal = (name, def) => {
                    const w = this.widgets?.find(w => w.name === name);
                    return w !== undefined ? w.value : def;
                };

                const mode = getVal('mode', 'Slider');
                const zoom = getVal('zoom', 1.0);
                const panX = getVal('pan_x', 0);
                const panY = getVal('pan_y', 0);
                
                const padding = 10;
                
                let yStart = 10;
                if (this.widgets && this.widgets.length > 0) {
                    let maxWidgetBottom = 0;
                    for (let widget of this.widgets) {
                        const widgetBottom = widget.y + (widget.height || 20);
                        if (widgetBottom > maxWidgetBottom) {
                            maxWidgetBottom = widgetBottom;
                        }
                    }
                    yStart = maxWidgetBottom + 10;
                }
                
                const w = Math.max(10, this.size[0] - padding * 2);
                const h = Math.max(10, this.size[1] - yStart - padding);

                if (mode === 'Slider') {
                    if (this.ag_img1) drawImageFit(ctx, this.ag_img1, padding, yStart, w, h, zoom, panX, panY);
                    
                    if (this.ag_img2 && !this.ag_hideImg2) {
                        ctx.save();
                        const clipW = w * this.ag_sliderPos;
                        ctx.beginPath();
                        ctx.rect(padding, yStart, clipW, h);
                        ctx.clip();
                        
                        drawImageFit(ctx, this.ag_img2, padding, yStart, w, h, zoom, panX, panY);
                        ctx.restore();
                        
                        ctx.beginPath();
                        ctx.moveTo(padding + clipW, yStart);
                        ctx.lineTo(padding + clipW, yStart + h);
                        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        
                        ctx.beginPath();
                        ctx.arc(padding + clipW, yStart + h / 2, 8, 0, Math.PI * 2);
                        ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                        ctx.fill();
                        ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }
                } else if (mode === 'Difference') {
                    if (this.ag_img1) drawImageFit(ctx, this.ag_img1, padding, yStart, w, h, zoom, panX, panY);
                    if (this.ag_img2 && !this.ag_hideImg2) {
                        ctx.save();
                        ctx.globalCompositeOperation = 'difference';
                        drawImageFit(ctx, this.ag_img2, padding, yStart, w, h, zoom, panX, panY);
                        ctx.restore();
                    }
                } else if (mode === 'Side-by-Side') {
                    if (this.ag_img1) {
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(padding, yStart, w / 2, h);
                        ctx.clip();
                        drawImageFit(ctx, this.ag_img1, padding, yStart, w, h, zoom, panX, panY);
                        ctx.restore();
                    }
                    if (this.ag_img2 && !this.ag_hideImg2) {
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(padding + w / 2, yStart, w / 2, h);
                        ctx.clip();
                        drawImageFit(ctx, this.ag_img2, padding, yStart, w, h, zoom, panX, panY);
                        ctx.restore();
                    }
                    ctx.beginPath();
                    ctx.moveTo(padding + w / 2, yStart);
                    ctx.lineTo(padding + w / 2, yStart + h);
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
                return r;
            };

            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function (e, pos, graphcanvas) {
                const mode = this.widgets?.find(w => w.name === 'mode')?.value || 'Slider';
                if (mode === 'Slider') {
                    let yStart = 10;
                    if (this.widgets && this.widgets.length > 0) {
                        let maxWidgetBottom = 0;
                        for (let widget of this.widgets) {
                            const widgetBottom = widget.y + (widget.height || 20);
                            if (widgetBottom > maxWidgetBottom) {
                                maxWidgetBottom = widgetBottom;
                            }
                        }
                        yStart = maxWidgetBottom + 10;
                    }
                    
                    const padding = 10;
                    const w = this.size[0] - padding * 2;
                    const h = this.size[1] - yStart - padding;
                    
                    if (pos[0] >= padding && pos[0] <= padding + w && pos[1] >= yStart && pos[1] <= yStart + h) {
                        this.ag_isDragging = true;
                        this.ag_sliderPos = Math.max(0, Math.min(1, (pos[0] - padding) / w));
                        graphcanvas.setDirty(true, true);
                        return true;
                    }
                }
                return onMouseDown ? onMouseDown.apply(this, arguments) : false;
            };

            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function (e, pos, graphcanvas) {
                if (this.ag_isDragging) {
                    const padding = 10;
                    const w = this.size[0] - padding * 2;
                    this.ag_sliderPos = Math.max(0, Math.min(1, (pos[0] - padding) / w));
                    graphcanvas.setDirty(true, true);
                }
                return onMouseMove ? onMouseMove.apply(this, arguments) : false;
            };

            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function (e, pos, graphcanvas) {
                this.ag_isDragging = false;
                return onMouseUp ? onMouseUp.apply(this, arguments) : false;
            };

            const onKeyDown = nodeType.prototype.onKeyDown;
            nodeType.prototype.onKeyDown = function (e, pos, graphcanvas) {
                if (e.code === 'Space') {
                    this.ag_hideImg2 = !this.ag_hideImg2;
                    graphcanvas.setDirty(true, true);
                    return true;
                }
                return onKeyDown ? onKeyDown.apply(this, arguments) : false;
            };
        }
    }
});