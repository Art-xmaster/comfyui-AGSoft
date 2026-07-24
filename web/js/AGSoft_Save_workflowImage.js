// AGSoft Save Workflow Image - JavaScript Extension for ComfyUI
// File(Файл): ComfyUI/custom_nodes/comfyui-AGSoft/web/js/AGSoft_Save_workflowImage.js
//
// Version(Версия): 2.2
//
// Description:
// Extension for saving the ComfyUI workspace as a PNG image.
// Supports:
// - export PNG without workflow;
// - export PNG with embedded JSON workflow;
// - save workflow as JSON;
// - import workflow from PNG/JSON.
//
// Описание:
// Расширение для сохранения рабочей области ComfyUI в PNG.
// Поддерживается:
// - экспорт PNG без workflow;
// - экспорт PNG с внедрённым JSON workflow;
// - сохранение workflow в JSON;
// - импорт workflow из PNG/JSON.
//
// Автор: AGSoft
// Дата: 24.07.2026
//

/* global LGraphCanvas */

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

let fileInput = null;
let getDrawTextConfig = null;

class WorkflowImage {
    static accept = "";
    extension = "";

    getCanvasElement() {
        return app.canvasEl || app.canvas.canvas;
    }

    getBounds() {
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;

        // Nodes
        for (const node of app.graph._nodes) {
            const b = node.getBounding();

            const l = node.pos[0];
            const t = node.pos[1];
            const r = l + b[2];
            const bt = t + b[3];

            if (l < minX) minX = l;
            if (t < minY) minY = t;
            if (r > maxX) maxX = r;
            if (bt > maxY) maxY = bt;
        }

        // Groups
        const groups = app.graph.groups || app.graph._groups || [];

        if (groups && typeof groups[Symbol.iterator] === "function") {
            for (const group of groups) {
                const gb = group?._bounding;

                if (!Array.isArray(gb) || gb.length < 4) {
                    continue;
                }

                const [gx, gy, gw, gh] = gb;

                if (!isFinite(gx) || !isFinite(gy) || !isFinite(gw) || !isFinite(gh)) {
                    continue;
                }

                minX = Math.min(minX, gx);
                minY = Math.min(minY, gy);
                maxX = Math.max(maxX, gx + gw);
                maxY = Math.max(maxY, gy + gh);
            }
        }

        // Links
        const serialized = app.graph.serialize();
        const links = Array.isArray(serialized.links) ? serialized.links : [];

        for (const link of links) {
            const originNode = app.graph.getNodeById(link[0]);
            const targetNode = app.graph.getNodeById(link[2]);

            if (!originNode?.pos || !targetNode?.pos) {
                continue;
            }

            const originSlot = Number(link[1] || 0);
            const targetSlot = Number(link[3] || 0);

            const startX = originNode.pos[0] + (originNode.size?.[0] || 0);
            const startY = originNode.pos[1] + 26 + originSlot * 20;

            const endX = targetNode.pos[0];
            const endY = targetNode.pos[1] + 26 + targetSlot * 20;

            const dist = Math.abs(endX - startX);
            const cpOffset = Math.max(dist * 0.5, 100);

            const xs = [
                startX,
                endX,
                startX + cpOffset,
                endX - cpOffset,
                startX + cpOffset + 50,
                endX - cpOffset - 50
            ];

            const ys = [
                startY,
                endY,
                startY,
                endY,
                startY + 50,
                endY - 50
            ];

            for (const x of xs) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
            }

            for (const y of ys) {
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }

        if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
            return [0, 0, 800, 600];
        }

        const padding = 100;

        return [
            minX - padding,
            minY - padding,
            maxX + padding,
            maxY + padding
        ];
    }

    saveState() {
        const canvas = this.getCanvasElement();

        this.state = {
            scale: app.canvas.ds.scale,
            width: app.canvas.canvas.width,
            height: app.canvas.canvas.height,
            offset: [app.canvas.ds.offset[0], app.canvas.ds.offset[1]],
            transform: app.canvas.canvas.getContext("2d").getTransform(),
            elWidth: canvas.width,
            elHeight: canvas.height,
            elStyleWidth: canvas.style.width,
            elStyleHeight: canvas.style.height
        };
    }

    restoreState() {
        const canvas = this.getCanvasElement();

        app.canvas.ds.scale = this.state.scale;
        app.canvas.canvas.width = this.state.width;
        app.canvas.canvas.height = this.state.height;
        app.canvas.ds.offset = this.state.offset;

        app.canvas.canvas
            .getContext("2d")
            .setTransform(this.state.transform);

        canvas.width = this.state.elWidth;
        canvas.height = this.state.elHeight;

        if (typeof this.state.elStyleWidth === "string") {
            canvas.style.width = this.state.elStyleWidth;
        }

        if (typeof this.state.elStyleHeight === "string") {
            canvas.style.height = this.state.elStyleHeight;
        }
    }

    updateView(bounds) {
        const w = bounds[2] - bounds[0];
        const h = bounds[3] - bounds[1];

        let scale = 1.2;
        const MAX_DIM = 8192;

        if (this.extension === "png" && (w * scale > MAX_DIM || h * scale > MAX_DIM)) {
            scale = Math.min(MAX_DIM / w, MAX_DIM / h);
        }

        app.canvas.ds.scale = scale;

        const canvas = this.getCanvasElement();

        const width = Math.max(1, Math.ceil(w * scale));
        const height = Math.max(1, Math.ceil(h * scale));

        canvas.width = width;
        canvas.height = height;

        canvas.style.width = width + "px";
        canvas.style.height = height + "px";

        app.canvas.ds.offset = [-bounds[0], -bounds[1]];

        if (app.canvas.canvas !== canvas) {
            app.canvas.canvas.width = width;
            app.canvas.canvas.height = height;
            app.canvas.canvas.style.width = canvas.style.width;
            app.canvas.canvas.style.height = canvas.style.height;
        }
    }

    getDrawTextConfig(_, widget) {
        return {
            x: 10,
            y: widget.last_y + 10,
            resetTransform: false
        };
    }

    async export(includeWorkflow) {
        this.saveState();

        let blob = null;

        try {
            const bounds = this.getBounds();
            this.updateView(bounds);

            const canvas = this.getCanvasElement();
            const ctx = canvas.getContext("2d");

            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            app.graph.setDirtyCanvas(true, true);
            app.canvas.setDirty(true, true);

            getDrawTextConfig = this.getDrawTextConfig.bind(this);

            app.canvas.draw(true, true);
            await new Promise((resolve) => setTimeout(resolve, 300));

            app.canvas.draw(true, true);
            await new Promise((resolve) => setTimeout(resolve, 300));

            blob = await this.getBlob(
                includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined
            );
        } finally {
            getDrawTextConfig = null;
            this.restoreState();
            app.canvas.draw(true, true);
        }

        if (blob) {
            this.download(blob);
        }
    }

    download(blob) {
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");

        Object.assign(a, {
            href: url,
            download: "workflow." + this.extension,
            style: "display: none"
        });

        document.body.append(a);
        a.click();

        setTimeout(() => {
            a.remove();
            window.URL.revokeObjectURL(url);
        }, 0);
    }

    static import() {
        if (!fileInput) {
            fileInput = document.createElement("input");

            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                onchange: () => {
                    if (fileInput.files.length > 0) {
                        app.handleFile(fileInput.files[0]);
                    }
                }
            });

            document.body.append(fileInput);
        }

        fileInput.value = "";
        fileInput.accept = this.accept || WorkflowImage.accept || "";
        fileInput.click();
    }

    async getBlob(workflow) {
        return new Promise((resolve) => {
            const canvas = this.getCanvasElement();

            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = Math.max(1, canvas.width);
            tempCanvas.height = Math.max(1, canvas.height);

            const tempCtx = tempCanvas.getContext("2d");

            if (!tempCtx) {
                resolve(null);
                return;
            }

            tempCtx.fillStyle = app.canvas.clear_background_color || "#222222";
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);

            tempCanvas.toBlob(async (blob) => {
                try {
                    if (!blob) {
                        resolve(blob);
                        return;
                    }

                    if (!workflow) {
                        resolve(blob);
                        return;
                    }

                    const buffer = await blob.arrayBuffer();

                    if (buffer.byteLength <= 12) {
                        resolve(blob);
                        return;
                    }

                    const typedArr = new Uint8Array(buffer);
                    const view = new DataView(buffer);

                    // PNG tEXt chunk:
                    // [chunk type][keyword][0x00][text]
                    const type = new Uint8Array([0x74, 0x45, 0x58, 0x74]); // "tEXt"
                    const keyword = new TextEncoder().encode("workflow");
                    const separator = new Uint8Array([0x00]);
                    const text = new TextEncoder().encode(workflow);

                    const chunkData = this.joinArrayBuffer(
                        type,
                        keyword,
                        separator,
                        text
                    );

                    const chunk = this.joinArrayBuffer(
                        this.n2b(chunkData.length - 4),
                        chunkData,
                        this.n2b(this.crc32(chunkData))
                    );

                    // Вставляем после IHDR.
                    const insertPosition = view.getUint32(8) + 20;

                    const result = this.joinArrayBuffer(
                        typedArr.subarray(0, insertPosition),
                        chunk,
                        typedArr.subarray(insertPosition)
                    );

                    resolve(new Blob([result], { type: "image/png" }));
                } catch (error) {
                    console.error("[AGSoft] Failed to embed workflow into PNG:", error);
                    resolve(blob);
                }
            }, "image/png");
        });
    }

    n2b(n) {
        return new Uint8Array([
            (n >> 24) & 0xff,
            (n >> 16) & 0xff,
            (n >> 8) & 0xff,
            n & 0xff
        ]);
    }

    joinArrayBuffer(...bufs) {
        const totalSize = bufs.reduce((sum, buf) => sum + buf.byteLength, 0);
        const result = new Uint8Array(totalSize);

        bufs.reduce((offset, buf) => {
            result.set(buf, offset);
            return offset + buf.byteLength;
        }, 0);

        return result;
    }

    crc32(data) {
        const crcTable = this.constructor.crcTable || (this.constructor.crcTable = (() => {
            let c;
            const table = [];

            for (let n = 0; n < 256; n++) {
                c = n;

                for (let k = 0; k < 8; k++) {
                    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
                }

                table[n] = c;
            }

            return table;
        })());

        let crc = 0 ^ -1;

        for (let i = 0; i < data.byteLength; i++) {
            crc = (crc >>> 8) ^ crcTable[(crc ^ data[i]) & 0xff];
        }

        return (crc ^ -1) >>> 0;
    }
}

class JSONWorkflowSaver {
    static save() {
        try {
            const workflow = app.graph.serialize();
            const jsonStr = JSON.stringify(workflow, null, 2);

            const timestamp = new Date()
                .toISOString()
                .replace(/[:.]/g, "-")
                .slice(0, 19);

            const filename = `workflow_${timestamp}.json`;

            const blob = new Blob([jsonStr], { type: "application/json" });
            const url = URL.createObjectURL(blob);

            const a = document.createElement("a");
            a.href = url;
            a.download = filename;

            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("[AGSoft] Error saving JSON:", error);
        }
    }
}

class PngWorkflowImage extends WorkflowImage {
    static accept = ".png,image/png";
    extension = "png";
}

const pngSaver = new PngWorkflowImage();

app.registerExtension({
    name: "AGSoft.SaveWorkflow",

    init() {
        function wrapText(context, text, x, y, maxWidth, lineHeight) {
            if (maxWidth <= 0) {
                context.fillText(text, x, y);
                return;
            }

            const words = text.split(" ");
            let line = "";

            for (let i = 0; i < words.length; i++) {
                let test = words[i];
                let metrics = context.measureText(test);

                while (metrics.width > maxWidth && test.length > 0) {
                    test = test.slice(0, -1);
                    metrics = context.measureText(test);
                }

                if (words[i] !== test) {
                    words.splice(i + 1, 0, words[i].slice(test.length));
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

        const stringWidget = ComfyWidgets.STRING;

        ComfyWidgets.STRING = function () {
            const w = stringWidget.apply(this, arguments);

            if (w.widget && w.widget.type === "customtext") {
                const draw = w.widget.draw;

                w.widget.draw = function (ctx) {
                    draw.apply(this, arguments);

                    if (!this.inputEl || this.inputEl.hidden) {
                        return;
                    }

                    if (!getDrawTextConfig) {
                        return;
                    }

                    const config = getDrawTextConfig(ctx, this);

                    ctx.save();

                    if (config.resetTransform) {
                        if (ctx.resetTransform) {
                            ctx.resetTransform();
                        } else {
                            ctx.setTransform(1, 0, 0, 1, 0, 0);
                        }
                    }

                    const style = document.defaultView?.getComputedStyle(this.inputEl, null);

                    if (!style) {
                        ctx.restore();
                        return;
                    }

                    const x = config.x;
                    const y = config.y;

                    const domWrapper = this.inputEl.closest(".dom-widget") ?? this.inputEl;

                    let widgetWidth = parseInt(domWrapper.style.width, 10);

                    if (!widgetWidth) {
                        widgetWidth = (this.node?.size?.[0] || 220) - 20;
                    }

                    const transform = ctx.getTransform();
                    const lineHeight = Number.isFinite(transform.d) ? transform.d * 12 : 12;

                    const lines = this.inputEl.value.split("\n");

                    let widgetHeight = parseInt(domWrapper.style.height, 10);

                    if (!widgetHeight) {
                        widgetHeight = Math.max(20, lines.length * lineHeight + 8);
                    }

                    ctx.beginPath();
                    ctx.rect(x, y, widgetWidth, widgetHeight);
                    ctx.clip();

                    ctx.fillStyle = style.getPropertyValue("background-color");
                    ctx.fillRect(x, y, widgetWidth, widgetHeight);

                    ctx.fillStyle = style.getPropertyValue("color");
                    ctx.font = style.getPropertyValue("font");

                    let currentY = y;

                    for (const lineText of lines) {
                        currentY += lineHeight;
                        wrapText(ctx, lineText, x + 4, currentY, widgetWidth, lineHeight);
                    }

                    ctx.restore();
                };
            }

            return w;
        };
    },

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
