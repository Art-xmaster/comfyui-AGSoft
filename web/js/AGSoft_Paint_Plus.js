// ==============================================================================
// AGSoft_Paint_Plus.js
// ==============================================================================
// JS-расширение ноды 🎨AGSoft Paint Plus. Версия: v10.29
// JS extension for the 🎨AGSoft Paint Plus node. Version: v10.29
// Author / Автор: AGSoft
// Date / Дата: 29.10.2026
// ------------------------------------------------------------------------------
// Description / Описание:
//
// EN:
// Frontend companion of the paint node: a DOM widget with a transparent
// stroke canvas over a live CSS background color, a fixed two-row toolbar
// (row 1: Brush, Shape, shape select, Canvas and Brush color pickers;
// row 2 always starts with Undo, then Redo, Eraser, Clear, Upload, Save),
// a bottom info line (18px) and a custom right-click context menu (load,
// copy, save image/mask, undo, redo, clear). Five brushes (circle, square,
// spray, soft, flat), fifteen shapes including four speech-bubble variants,
// opacity and shape fill, upload placement modes (stretch, fit,
// crop_center), undo/redo stacks and drag&drop of images. The stroke canvas
// is saved as a PNG into the temp folder through a loopback route; the
// hidden widget holds only the file key, so workflow drafts stay tiny.
// Node size is strictly controlled: width is master, height snaps to
// toolbar + canvas (aspect ratio) + info line, vertical drag snaps back.
// ---
// RU:
// Фронтенд-спутник ноды рисования: DOM-виджет с прозрачным холстом штрихов
// поверх живого CSS-цвета фона, фиксированный двухрядный тулбар
// (ряд 1: Brush, Shape, селект фигур, палитры Canvas и Brush; ряд 2 всегда
// начинается с Undo, далее Redo, Eraser, Clear, Upload, Save), нижняя
// инфострока (18px) и кастомное меню правого клика (загрузка, копирование,
// сохранение изображения/маски, undo, redo, очистка). Пять кистей (circle,
// square, spray, soft, flat), пятнадцать фигур включая 4 вида речевых
// пузырей, прозрачность и заливка фигур, режимы размещения загрузки
// (stretch, fit, crop_center), стеки undo/redo и drag&drop картинок.
// Холст штрихов сохраняется PNG в temp-папку через loopback-роут; скрытый
// виджет держит только ключ файла, поэтому черновики workflow лёгкие.
// Размер ноды жёстко контролируется: ширина — мастер, высота снапится к
// тулбар + холст (пропорция) + инфострока, вертикальный драг возвращается.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Paint Plus] JS extension loaded v10.29");

const INFO_H = 18;    // высота нижней инфостроки / bottom info line height
const EXTRA_H = 28;   // padding + gap + нижний отступ ноды / paddings + node bottom margin
const INIT_W = 520;   // стартовая ширина ноды / initial node width
const MIN_W = 260;    // минимальная ширина ноды / minimal node width

function base() {
    return (api && api.api_base) ? api.api_base : "";
}

app.registerExtension({
    name: "AGSoft.PaintPlus",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftPaintPlus") return;

        const dataWidget = node.widgets.find(w => w.name === "drawing_key");
        if (!dataWidget) return;
        const wWidget = node.widgets.find(w => w.name === "width");
        const hWidget = node.widgets.find(w => w.name === "height");
        const cColorWidget = node.widgets.find(w => w.name === "canvas_color");
        const bColorWidget = node.widgets.find(w => w.name === "brush_color");
        const bSizeWidget = node.widgets.find(w => w.name === "brush_size");
        const bShapeWidget = node.widgets.find(w => w.name === "brush_shape");
        const opacityWidget = node.widgets.find(w => w.name === "opacity");
        const fillWidget = node.widgets.find(w => w.name === "fill_shapes");
        const sidesWidget = node.widgets.find(w => w.name === "polygon_sides");
        const textWidget = node.widgets.find(w => w.name === "text_value");
        const uploadModeWidget = node.widgets.find(w => w.name === "upload_mode");

        dataWidget.hidden = true;
        dataWidget.y = -9999;

        // Старые workflow могли сохранить opacity как NaN: санитизируем.
        // Old workflows may have saved opacity as NaN: sanitize it.
        const sanitizeOpacity = () => {
            if (opacityWidget && !isFinite(parseFloat(opacityWidget.value))) {
                opacityWidget.value = 1.0;
            }
        };
        sanitizeOpacity();

        // Уникальный ключ новой ноды: node.id при создании ещё равен -1.
        // Unique key for a new node: node.id is still -1 at creation time.
        const token = Date.now().toString(36) +
            Math.floor(Math.random() * 1679616).toString(36);
        const paintKey = `agsoft_paint_${token}.png`;
        const currentKey = () => {
            const v = String(dataWidget.value || "");
            if (v.indexOf("agsoft_paint_") === 0 && v.slice(-4) === ".png") return v;
            return paintKey;
        };

        const container = document.createElement("div");
        container.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:4px;" +
            "box-sizing:border-box;width:100%;overflow:hidden;outline:none;";
        container.tabIndex = 0;

        // Тулбар из двух рядов: ряд 2 (Undo и далее) всегда с новой строки.
        // Two-row toolbar: row 2 (Undo and further) always starts a new line.
        const toolbar = document.createElement("div");
        toolbar.style.cssText = "display:flex;flex-direction:column;gap:4px;";
        container.appendChild(toolbar);
        const toolbarRow1 = document.createElement("div");
        toolbarRow1.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center;";
        toolbar.appendChild(toolbarRow1);
        const toolbarRow2 = document.createElement("div");
        toolbarRow2.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center;";
        toolbar.appendChild(toolbarRow2);

        const canvasWrap = document.createElement("div");
        canvasWrap.style.cssText = "position:relative;width:100%;box-sizing:border-box;" +
            "border:1px solid var(--ags-accent,#555);border-radius:4px;overflow:hidden;" +
            "background:#FFFFFF;";
        container.appendChild(canvasWrap);

        // Холст держит только штрихи с альфой; фон — CSS-цвет под ним.
        // The canvas holds alpha strokes only; background is a CSS color below.
        const canvas = document.createElement("canvas");
        canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;" +
            "display:block;cursor:crosshair;";
        canvasWrap.appendChild(canvas);
        const ctx = canvas.getContext("2d", { willReadFrequently: true });

        const infoLine = document.createElement("div");
        infoLine.style.cssText = "height:18px;line-height:18px;font:10px monospace;" +
            "color:var(--ags-text,#9ecbdd);white-space:nowrap;overflow:hidden;" +
            "text-overflow:ellipsis;width:100%;";
        container.appendChild(infoLine);

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/*";
        fileInput.style.display = "none";
        container.appendChild(fileInput);

        let currentTool = "brush";
        let activeBtn = null;
        let isDrawing = false;
        let dirty = false;
        let startX = 0;
        let startY = 0;
        let snapshot = null;
        let lastDropAt = 0;
        let saveTimer = 0;
        let infoLock = 0;
        let menuEl = null;
        let domWidget = null;
        const undoStack = [];
        const redoStack = [];

        const hexOr = (v, dflt) => {
            const s = String(v || "").trim();
            return /^#[0-9a-fA-F]{6}$/.test(s) ? s : dflt;
        };
        const rgba = (hex, a) => {
            const s = hexOr(hex, "#000000");
            const r = parseInt(s.slice(1, 3), 16);
            const g = parseInt(s.slice(3, 5), 16);
            const b = parseInt(s.slice(5, 7), 16);
            return `rgba(${r},${g},${b},${a})`;
        };
        const brushColor = () => hexOr(bColorWidget?.value, "#000000");
        const opacityValue = () => {
            const v = parseFloat(opacityWidget?.value);
            return isFinite(v) ? Math.min(1, Math.max(0.05, v)) : 1;
        };
        const fillShapes = () => Boolean(fillWidget?.value);

        const applyBg = () => {
            canvasWrap.style.background = hexOr(cColorWidget?.value, "#FFFFFF");
        };

        const flashInfo = (text) => {
            infoLock = Date.now() + 2000;
            infoLine.textContent = text;
            setTimeout(() => {
                if (Date.now() >= infoLock) updateInfo();
            }, 2100);
        };

        // Сохранение в temp через loopback-роут; в виджет пишем только ключ.
        // Save into temp via the loopback route; widget holds only the key.
        const syncData = () => {
            dirty = true;
            const data = canvas.toDataURL("image/png");
            clearTimeout(saveTimer);
            saveTimer = setTimeout(() => {
                fetch(base() + "/agsoft/paint_plus/save", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: currentKey(), data: data })
                }).then(r => r.json()).then(j => {
                    if (j && j.ok) dataWidget.value = j.key;
                }).catch(e => console.warn("[AGSoft Paint Plus] save skipped:", e));
            }, 120);
        };

        const restoreImage = () => {
            const key = String(dataWidget.value || "");
            if (key.indexOf("agsoft_paint_") !== 0 || key.slice(-4) !== ".png") return;
            fetch(base() + "/agsoft/paint_plus/load?key=" + encodeURIComponent(key))
                .then(r => (r.ok ? r.blob() : null))
                .then(blob => {
                    if (!blob) return;
                    const url = URL.createObjectURL(blob);
                    const img = new Image();
                    img.onload = () => {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        URL.revokeObjectURL(url);
                        dirty = true;
                    };
                    img.src = url;
                })
                .catch(e => console.warn("[AGSoft Paint Plus] restore skipped:", e));
        };

        // Композит фона и штрихов / background + strokes composite
        const compositeCanvas = () => {
            const tmp = document.createElement("canvas");
            tmp.width = canvas.width;
            tmp.height = canvas.height;
            const tctx = tmp.getContext("2d");
            tctx.fillStyle = hexOr(cColorWidget?.value, "#FFFFFF");
            tctx.fillRect(0, 0, tmp.width, tmp.height);
            tctx.drawImage(canvas, 0, 0);
            return tmp;
        };
        // Маска из альфы штрихов: белое по чёрному / mask from stroke alpha
        const maskCanvas = () => {
            const src = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const tmp = document.createElement("canvas");
            tmp.width = canvas.width;
            tmp.height = canvas.height;
            const tctx = tmp.getContext("2d");
            const out = tctx.createImageData(tmp.width, tmp.height);
            for (let i = 0; i < src.data.length; i += 4) {
                const a = src.data[i + 3];
                out.data[i] = a;
                out.data[i + 1] = a;
                out.data[i + 2] = a;
                out.data[i + 3] = 255;
            }
            tctx.putImageData(out, 0, 0);
            return tmp;
        };
        const withImageBlob = (fn) => {
            compositeCanvas().toBlob(b => {
                if (b) fn(b);
            }, "image/png");
        };
        const withMaskBlob = (fn) => {
            maskCanvas().toBlob(b => {
                if (b) fn(b);
            }, "image/png");
        };
        const withImageDataURL = (fn) => fn(compositeCanvas().toDataURL("image/png"));
        const withMaskDataURL = (fn) => fn(maskCanvas().toDataURL("image/png"));

        const suggestedName = (kind) => `agsoft_paint_${kind}_${Date.now()}.png`;

        // Диалог сохранения с выбором папки; фолбэк — скачивание браузером.
        // Save dialog with folder pick; fallback is a browser download.
        const saveBlobAs = (blob, name) => {
            if (window.showSaveFilePicker) {
                window.showSaveFilePicker({
                    suggestedName: name,
                    types: [{ description: "PNG image", accept: { "image/png": [".png"] } }]
                }).then(handle => handle.createWritable())
                    .then(w => w.write(blob).then(() => w.close()))
                    .then(() => flashInfo("saved: " + name))
                    .catch(e => console.warn("[AGSoft Paint Plus] picker skipped:", e));
                return;
            }
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = name;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 4000);
            flashInfo("saved: " + name);
        };

        const copyBlob = (blob) => {
            try {
                if (!navigator.clipboard || !window.ClipboardItem) return;
                navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })])
                    .then(() => flashInfo("copied to clipboard"))
                    .catch(e => console.warn("[AGSoft Paint Plus] copy skipped:", e));
            } catch (e) {
                console.warn("[AGSoft Paint Plus] copy skipped:", e);
            }
        };

        const saveToServer = (data, kind) => {
            fetch(base() + "/agsoft/paint_plus/save_png", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ data: data, kind: kind, name: suggestedName(kind) })
            }).then(r => r.json()).then(j => {
                if (j && j.ok) flashInfo("saved: output/agsoft_paint/" + j.name);
            }).catch(e => console.warn("[AGSoft Paint Plus] output save skipped:", e));
        };

        const canvasDispH = (iw) => {
            const cw = Math.max(64, parseInt(wWidget?.value, 10) || 1024);
            const ch = Math.max(64, parseInt(hWidget?.value, 10) || 1024);
            return Math.max(80, Math.round((iw - 2) * ch / cw));
        };

        const updateInfo = () => {
            if (Date.now() < infoLock) return;
            const cw = parseInt(wWidget?.value, 10) || 1024;
            const ch = parseInt(hWidget?.value, 10) || 1024;
            const bs = parseInt(bSizeWidget?.value, 10) || 10;
            const op = opacityValue().toFixed(2);
            const fm = fillShapes() ? "fill" : "line";
            infoLine.textContent = `${cw}×${ch}px · ${currentTool} · ` +
                `${shapeSelect.value} · ${bs}px · op ${op} · ${fm}`;
        };

        const widgetTop = () => {
            if (domWidget && domWidget.y > 4) return domWidget.y;
            let top = 0;
            (node.widgets || []).forEach(w => {
                if (w === domWidget || w.y < 0) return;
                const bottom = w.y + (w.height || 20);
                if (bottom > top) top = bottom;
            });
            return top > 4 ? top + 6 : 120;
        };

        const strictFit = () => {
            const nw = Math.max(MIN_W, Math.round(node.size?.[0] || INIT_W));
            node.size[0] = nw;
            const iw = container.clientWidth || (nw - 20);
            const tbH = toolbar.offsetHeight || 64;
            const viewH = canvasDispH(iw);
            canvasWrap.style.height = viewH + "px";
            const wantH = Math.max(220, Math.round(
                widgetTop() + tbH + viewH + INFO_H + EXTRA_H
            ));
            if (Math.abs((node.size?.[1] || 0) - wantH) > 1) node.size[1] = wantH;
            updateInfo();
            node.setDirtyCanvas(true, true);
        };

        const resizeCanvases = () => {
            const cw = Math.max(64, parseInt(wWidget?.value, 10) || 1024);
            const ch = Math.max(64, parseInt(hWidget?.value, 10) || 1024);
            if (canvas.width === cw && canvas.height === ch) return;
            const tmp = document.createElement("canvas");
            tmp.width = canvas.width || 1;
            tmp.height = canvas.height || 1;
            tmp.getContext("2d").drawImage(canvas, 0, 0);
            canvas.width = cw;
            canvas.height = ch;
            undoStack.length = 0;
            redoStack.length = 0;
            ctx.clearRect(0, 0, cw, ch);
            if (tmp.width > 1 && tmp.height > 1) {
                ctx.drawImage(tmp, 0, 0, tmp.width, tmp.height, 0, 0, cw, ch);
            }
            syncData();
        };

        const pushUndo = () => {
            try {
                undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                if (undoStack.length > 24) undoStack.shift();
                redoStack.length = 0;
            } catch (e) {}
        };
        const undo = () => {
            if (!undoStack.length) return;
            try {
                redoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                ctx.putImageData(undoStack.pop(), 0, 0);
                syncData();
            } catch (e) {}
        };
        const redo = () => {
            if (!redoStack.length) return;
            try {
                undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                ctx.putImageData(redoStack.pop(), 0, 0);
                syncData();
            } catch (e) {}
        };
        const clearCanvas = () => {
            pushUndo();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            syncData();
        };

        // Размещение загруженной картинки по upload_mode: stretch / fit /
        // crop_center. / Uploaded image placement by upload_mode.
        const drawImageByMode = (img) => {
            const W = canvas.width;
            const H = canvas.height;
            const iw = img.naturalWidth || img.width;
            const ih = img.naturalHeight || img.height;
            if (!iw || !ih || !W || !H) return;
            const mode = uploadModeWidget?.value || "stretch";
            if (mode === "fit" || mode === "crop_center") {
                const scale = mode === "fit"
                    ? Math.min(W / iw, H / ih)
                    : Math.max(W / iw, H / ih);
                const dw = iw * scale;
                const dh = ih * scale;
                ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
            } else {
                ctx.drawImage(img, 0, 0, W, H);
            }
        };

        const toolButtons = [];
        const makeBtn = (text, title, onClick, row) => {
            const btn = document.createElement("button");
            btn.textContent = text;
            btn.title = title;
            btn.style.cssText = "padding:4px 8px;cursor:pointer;border-radius:3px;" +
                "border:1px solid var(--ags-accent,#555);" +
                "background:var(--ags-bg,#3a3a3a);color:var(--ags-text,#dddddd);";
            btn.onclick = onClick;
            (row || toolbarRow1).appendChild(btn);
            toolButtons.push(btn);
            return btn;
        };

        const setActive = (btn) => {
            activeBtn = btn;
            toolButtons.forEach(b => {
                const on = b === activeBtn;
                b.style.background = on ? "var(--ags-accent, #4a6da7)" : "var(--ags-bg, #3a3a3a)";
                b.style.color = on ? "#ffffff" : "var(--ags-text, #dddddd)";
            });
            updateInfo();
        };

        const makeColor = (label, title, widget, dflt, onPick) => {
            const lab = document.createElement("span");
            lab.textContent = label;
            lab.style.cssText = "font-size:11px;color:var(--ags-text,#dddddd);";
            toolbarRow1.appendChild(lab);
            const inp = document.createElement("input");
            inp.type = "color";
            inp.title = title;
            inp.value = hexOr(widget?.value, dflt);
            inp.style.cssText = "width:36px;height:22px;padding:1px;border-radius:3px;" +
                "border:1px solid var(--ags-accent,#555);background:transparent;";
            inp.oninput = () => {
                if (widget) widget.value = inp.value;
                if (onPick) onPick(inp.value);
            };
            toolbarRow1.appendChild(inp);
            return inp;
        };

        // Ряд 1: Brush, Shape, селект фигур, палитры Canvas и Brush.
        // Ряд 2: Undo, Redo, Eraser, Clear, Upload, Save (всегда с новой строки).
        // Row 1: Brush, Shape, shape select, Canvas and Brush pickers.
        // Row 2: Undo, Redo, Eraser, Clear, Upload, Save (always a new line).
        const brushBtn = makeBtn("🖌 Brush", "Freehand brush\n---\nСвободное рисование кистью", () => {
            currentTool = "brush";
            setActive(brushBtn);
        }, toolbarRow1);
        const shapeBtn = makeBtn("⬟ Shape", "Draw selected shape by drag\n---\nНарисовать выбранную фигуру протяжкой", () => {
            currentTool = "shape";
            setActive(shapeBtn);
        }, toolbarRow1);

        const shapeSelect = document.createElement("select");
        shapeSelect.title = "Shape type (drag on canvas)\n---\nТип фигуры (протяжкой на холсте)";
        ["line", "circle", "square", "rect", "star", "triangle", "diamond",
            "ellipse", "arrow", "polygon", "text",
            "bubble_round", "bubble_rect", "bubble_cloud", "bubble_spike"].forEach(s => {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s.charAt(0).toUpperCase() + s.slice(1).replace("_", " ");
            shapeSelect.appendChild(opt);
        });
        shapeSelect.style.padding = "3px 4px";
        shapeSelect.onchange = () => {
            currentTool = "shape";
            setActive(shapeBtn);
        };
        toolbarRow1.appendChild(shapeSelect);

        const canvasColorInput = makeColor(
            "Canvas",
            "Canvas color, live background under strokes\n---\nЦвет холста, живой фон под штрихами",
            cColorWidget,
            "#FFFFFF",
            () => applyBg()
        );
        const brushColorInput = makeColor(
            "Brush",
            "Brush and shape color\n---\nЦвет кисти и фигур",
            bColorWidget,
            "#000000",
            null
        );

        makeBtn("⟲ Undo", "Undo last action (Ctrl+Z)\n---\nОтменить действие (Ctrl+Z)", undo, toolbarRow2);
        makeBtn("⟳ Redo", "Redo action (Ctrl+Y)\n---\nВернуть действие (Ctrl+Y)", redo, toolbarRow2);

        const eraserBtn = makeBtn("🧽 Eraser", "Eraser to transparency\n---\nЛастик до прозрачности", () => {
            currentTool = "eraser";
            setActive(eraserBtn);
        }, toolbarRow2);
        makeBtn("🗑 Clear", "Remove all strokes\n---\nУбрать все штрихи", clearCanvas, toolbarRow2);
        makeBtn("📥 Upload", "Load image into strokes, placement by upload_mode\n---\nЗагрузить картинку в штрихи, размещение по upload_mode", () => {
            fileInput.click();
        }, toolbarRow2);
        makeBtn("💾 Save", "Save PNG into a chosen folder\n---\nСохранить PNG в выбранную папку", () => {
            withImageBlob(b => saveBlobAs(b, suggestedName("image")));
        }, toolbarRow2);

        // Меню правого клика на холсте / right-click context menu on canvas
        const onDocDown = (e) => {
            if (menuEl && !menuEl.contains(e.target)) closeMenu();
        };
        const closeMenu = () => {
            if (!menuEl) return;
            menuEl.remove();
            menuEl = null;
            document.removeEventListener("mousedown", onDocDown, true);
        };
        const openMenu = (x, y) => {
            closeMenu();
            menuEl = document.createElement("div");
            menuEl.style.cssText = "position:fixed;z-index:10000;min-width:250px;padding:4px;" +
                "background:var(--ags-bg,#262626);border:1px solid var(--ags-accent,#555);" +
                "border-radius:6px;box-shadow:0 6px 18px rgba(0,0,0,0.5);display:flex;" +
                "flex-direction:column;gap:2px;";
            const item = (label, fn) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.style.cssText = "text-align:left;padding:5px 10px;border:none;" +
                    "border-radius:4px;background:transparent;cursor:pointer;" +
                    "font-size:12px;color:var(--ags-text,#dddddd);";
                b.onmouseenter = () => {
                    b.style.background = "var(--ags-accent,#4a6da7)";
                };
                b.onmouseleave = () => {
                    b.style.background = "transparent";
                };
                b.onclick = () => {
                    closeMenu();
                    fn();
                };
                menuEl.appendChild(b);
            };
            const sep = () => {
                const s = document.createElement("div");
                s.style.cssText = "height:1px;margin:2px 6px;background:var(--ags-accent,#555);opacity:0.5;";
                menuEl.appendChild(s);
            };
            item("📥 Load image… / Загрузить изображение…", () => fileInput.click());
            item("📋 Copy image / Копировать изображение", () => withImageBlob(copyBlob));
            item("📋 Copy mask / Копировать маску", () => withMaskBlob(copyBlob));
            sep();
            item("💾 Save image as… / Сохранить изображение как…", () => {
                withImageBlob(b => saveBlobAs(b, suggestedName("image")));
            });
            item("💾 Save mask as… / Сохранить маску как…", () => {
                withMaskBlob(b => saveBlobAs(b, suggestedName("mask")));
            });
            sep();
            item("📤 Save image to output / Сохранить изображение в output", () => {
                withImageDataURL(d => saveToServer(d, "image"));
            });
            item("📤 Save mask to output / Сохранить маску в output", () => {
                withMaskDataURL(d => saveToServer(d, "mask"));
            });
            sep();
            item("⟲ Undo / Отменить", undo);
            item("⟳ Redo / Вернуть", redo);
            item("🗑 Clear / Очистить", clearCanvas);
            document.body.appendChild(menuEl);
            const mw = menuEl.offsetWidth;
            const mh = menuEl.offsetHeight;
            menuEl.style.left = Math.max(4, Math.min(x, window.innerWidth - mw - 8)) + "px";
            menuEl.style.top = Math.max(4, Math.min(y, window.innerHeight - mh - 8)) + "px";
            setTimeout(() => document.addEventListener("mousedown", onDocDown, true), 0);
        };
        const onContextMenu = (e) => {
            e.preventDefault();
            e.stopPropagation();
            openMenu(e.clientX, e.clientY);
        };
        canvas.addEventListener("contextmenu", onContextMenu);
        canvasWrap.addEventListener("contextmenu", onContextMenu);

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (canvas.width / rect.width),
                y: (e.clientY - rect.top) * (canvas.height / rect.height)
            };
        };

        // Сегмент кисти: line-штрих для circle/square, штампы для spray/soft/flat.
        // Brush segment: line stroke for circle/square, stamps for spray/soft/flat.
        const strokeSegment = (x0, y0, x1, y1) => {
            const shape = bShapeWidget?.value || "circle";
            const size = Math.max(1, parseInt(bSizeWidget?.value, 10) || 10);
            ctx.globalAlpha = opacityValue();
            if (shape === "circle" || shape === "square") {
                ctx.lineWidth = size;
                ctx.lineCap = shape === "square" ? "square" : "round";
                ctx.lineJoin = "round";
                ctx.strokeStyle = brushColor();
                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.stroke();
            } else if (shape === "spray") {
                ctx.fillStyle = brushColor();
                const dots = Math.max(4, Math.round(size / 2));
                for (let i = 0; i < dots; i++) {
                    const a = Math.random() * Math.PI * 2;
                    const rr = Math.random() * size;
                    ctx.fillRect(x1 + Math.cos(a) * rr, y1 + Math.sin(a) * rr, 1.5, 1.5);
                }
            } else {
                const dist = Math.hypot(x1 - x0, y1 - y0);
                const step = Math.max(1, size / 5);
                const n = Math.max(1, Math.ceil(dist / step));
                for (let i = 0; i <= n; i++) {
                    const px = x0 + (x1 - x0) * (i / n);
                    const py = y0 + (y1 - y0) * (i / n);
                    if (shape === "soft") {
                        const g = ctx.createRadialGradient(px, py, 0, px, py, size / 2);
                        g.addColorStop(0, rgba(brushColor(), 0.5));
                        g.addColorStop(1, rgba(brushColor(), 0));
                        ctx.fillStyle = g;
                        ctx.beginPath();
                        ctx.arc(px, py, size / 2, 0, Math.PI * 2);
                        ctx.fill();
                    } else {
                        ctx.save();
                        ctx.translate(px, py);
                        ctx.rotate(Math.PI / 4);
                        ctx.fillStyle = brushColor();
                        ctx.fillRect(-size / 2, -size / 8, size, size / 4);
                        ctx.restore();
                    }
                }
            }
            ctx.globalAlpha = 1;
        };

        const eraseSegment = (x0, y0, x1, y1) => {
            const size = Math.max(1, parseInt(bSizeWidget?.value, 10) || 10);
            ctx.globalCompositeOperation = "destination-out";
            ctx.lineWidth = size;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.strokeStyle = "rgba(0,0,0,1)";
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.globalCompositeOperation = "source-over";
        };

        const starPath = (cx, cy, outer, inner) => {
            const spikes = 5;
            let rot = -Math.PI / 2;
            const step = Math.PI / spikes;
            ctx.beginPath();
            ctx.moveTo(cx + Math.cos(rot) * outer, cy + Math.sin(rot) * outer);
            for (let i = 0; i < spikes; i++) {
                rot += step;
                ctx.lineTo(cx + Math.cos(rot) * inner, cy + Math.sin(rot) * inner);
                rot += step;
                ctx.lineTo(cx + Math.cos(rot) * outer, cy + Math.sin(rot) * outer);
            }
            ctx.closePath();
        };

        // Тело и хвост речевого пузыря; spike — взрыв без хвоста.
        // Speech bubble body and tail; spike is a burst without a tail.
        const bubblePath = (variant, l, t, r, b) => {
            const cx = (l + r) / 2;
            const cy = (t + b) / 2;
            const rx = Math.max(6, (r - l) / 2);
            const ry = Math.max(6, (b - t) / 2);
            ctx.beginPath();
            if (variant === "bubble_round") {
                ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
            } else if (variant === "bubble_rect") {
                const rad = Math.min(16, rx, ry);
                ctx.moveTo(l + rad, t);
                ctx.arcTo(r, t, r, b, rad);
                ctx.arcTo(r, b, l, b, rad);
                ctx.arcTo(l, b, l, t, rad);
                ctx.arcTo(l, t, r, t, rad);
                ctx.closePath();
            } else if (variant === "bubble_cloud") {
                const bumps = 10;
                for (let i = 0; i < bumps; i++) {
                    const a0 = (i / bumps) * Math.PI * 2;
                    const px = cx + Math.cos(a0) * rx;
                    const py = cy + Math.sin(a0) * ry;
                    ctx.moveTo(px + ry * 0.35, py);
                    ctx.arc(px, py, ry * 0.35, 0, Math.PI * 2);
                }
            } else {
                const spikes = 12;
                let rot = -Math.PI / 2;
                const step = Math.PI / spikes;
                ctx.moveTo(cx, cy - ry);
                for (let i = 0; i < spikes; i++) {
                    rot += step;
                    ctx.lineTo(cx + Math.cos(rot) * rx * 0.8, cy + Math.sin(rot) * ry * 0.8);
                    rot += step;
                    ctx.lineTo(cx + Math.cos(rot) * rx, cy + Math.sin(rot) * ry);
                }
                ctx.closePath();
            }
            if (variant !== "bubble_spike") {
                const tx = l + (r - l) * 0.3;
                ctx.moveTo(tx, b - 2);
                ctx.lineTo(tx - ry * 0.35, b + ry * 0.5);
                ctx.lineTo(tx + ry * 0.3, b - 2);
                ctx.closePath();
            }
        };

        const drawShape = (shape, x0, y0, x1, y1) => {
            const left = Math.min(x0, x1);
            const right = Math.max(x0, x1);
            const top = Math.min(y0, y1);
            const bottom = Math.max(y0, y1);
            ctx.globalAlpha = opacityValue();
            if (shape === "line") {
                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.stroke();
            } else if (shape === "arrow") {
                const ang = Math.atan2(y1 - y0, x1 - x0);
                const size = Math.max(1, parseInt(bSizeWidget?.value, 10) || 10);
                const hl = Math.max(10, size * 1.6);
                ctx.beginPath();
                ctx.moveTo(x0, y0);
                ctx.lineTo(x1, y1);
                ctx.moveTo(x1, y1);
                ctx.lineTo(x1 - Math.cos(ang - 0.45) * hl, y1 - Math.sin(ang - 0.45) * hl);
                ctx.moveTo(x1, y1);
                ctx.lineTo(x1 - Math.cos(ang + 0.45) * hl, y1 - Math.sin(ang + 0.45) * hl);
                ctx.stroke();
            } else if (shape === "text") {
                const fs = Math.max(8, Math.round(Math.abs(y1 - y0)));
                ctx.font = `${fs}px sans-serif`;
                ctx.textBaseline = "top";
                ctx.fillText(String(textWidget?.value || ""), x0, y0);
            } else if (shape === "circle") {
                ctx.beginPath();
                ctx.arc(x0, y0, Math.hypot(x1 - x0, y1 - y0), 0, Math.PI * 2);
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape === "ellipse") {
                ctx.beginPath();
                ctx.ellipse(x0, y0, Math.abs(x1 - x0), Math.abs(y1 - y0), 0, 0, Math.PI * 2);
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape === "square") {
                const s = Math.max(right - left, bottom - top);
                const sx = x1 >= x0 ? 1 : -1;
                const sy = y1 >= y0 ? 1 : -1;
                if (fillShapes()) ctx.fillRect(x0, y0, s * sx, s * sy);
                else ctx.strokeRect(x0, y0, s * sx, s * sy);
            } else if (shape === "rect") {
                if (fillShapes()) ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                else ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
            } else if (shape === "triangle") {
                ctx.beginPath();
                ctx.moveTo((left + right) / 2, top);
                ctx.lineTo(right, bottom);
                ctx.lineTo(left, bottom);
                ctx.closePath();
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape === "diamond") {
                ctx.beginPath();
                ctx.moveTo((left + right) / 2, top);
                ctx.lineTo(right, (top + bottom) / 2);
                ctx.lineTo((left + right) / 2, bottom);
                ctx.lineTo(left, (top + bottom) / 2);
                ctx.closePath();
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape === "star") {
                const r = Math.hypot(x1 - x0, y1 - y0);
                starPath(x0, y0, r, r / 2);
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape === "polygon") {
                const n = Math.min(12, Math.max(3, parseInt(sidesWidget?.value, 10) || 6));
                const r = Math.hypot(x1 - x0, y1 - y0);
                ctx.beginPath();
                for (let i = 0; i < n; i++) {
                    const a = -Math.PI / 2 + (i / n) * Math.PI * 2;
                    const px = x0 + Math.cos(a) * r;
                    const py = y0 + Math.sin(a) * r;
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.closePath();
                fillShapes() ? ctx.fill() : ctx.stroke();
            } else if (shape.indexOf("bubble_") === 0) {
                bubblePath(shape, left, top, right, bottom);
                fillShapes() ? ctx.fill() : ctx.stroke();
            }
            ctx.globalAlpha = 1;
        };

        canvas.onmousedown = (e) => {
            if (e.button !== 0) return;
            e.preventDefault();
            container.focus();
            const pos = getPos(e);
            startX = pos.x;
            startY = pos.y;
            isDrawing = true;
            pushUndo();
            if (currentTool === "shape") {
                snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
                return;
            }
            strokeSegment(startX, startY, startX + 0.01, startY + 0.01);
        };

        canvas.onmousemove = (e) => {
            if (!isDrawing) return;
            const pos = getPos(e);
            if (currentTool === "shape") {
                if (snapshot) ctx.putImageData(snapshot, 0, 0);
                ctx.lineWidth = Math.max(1, parseInt(bSizeWidget?.value, 10) || 10);
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                ctx.strokeStyle = brushColor();
                ctx.fillStyle = brushColor();
                drawShape(shapeSelect.value, startX, startY, pos.x, pos.y);
                return;
            }
            if (currentTool === "eraser") eraseSegment(startX, startY, pos.x, pos.y);
            else strokeSegment(startX, startY, pos.x, pos.y);
            startX = pos.x;
            startY = pos.y;
        };

        const endStroke = () => {
            if (!isDrawing) return;
            isDrawing = false;
            snapshot = null;
            syncData();
        };
        canvas.onmouseup = endStroke;
        canvas.onmouseleave = endStroke;

        container.addEventListener("keydown", (e) => {
            if (!(e.ctrlKey || e.metaKey)) return;
            const k = String(e.key || "").toLowerCase();
            if (k === "z" && !e.shiftKey) {
                e.preventDefault();
                undo();
            } else if (k === "y" || (k === "z" && e.shiftKey)) {
                e.preventDefault();
                redo();
            }
        });

        const loadFileToCanvas = (f) => {
            if (!f || !f.type.startsWith("image/")) return;
            const reader = new FileReader();
            reader.onload = ev => {
                const img = new Image();
                img.onload = () => {
                    pushUndo();
                    drawImageByMode(img);
                    syncData();
                };
                img.src = ev.target.result;
            };
            reader.readAsDataURL(f);
        };

        fileInput.onchange = () => {
            loadFileToCanvas(fileInput.files && fileInput.files[0]);
            fileInput.value = "";
        };

        container.addEventListener("dragover", (e) => {
            e.preventDefault();
            canvasWrap.style.outline = "2px dashed #7fd4ff";
        });
        container.addEventListener("dragleave", () => {
            canvasWrap.style.outline = "none";
        });
        container.addEventListener("drop", (e) => {
            e.preventDefault();
            canvasWrap.style.outline = "none";
            const now = Date.now();
            if (now - lastDropAt < 600) return;
            lastDropAt = now;
            loadFileToCanvas(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
        });

        [wWidget, hWidget].forEach(w => {
            if (!w) return;
            const oldCb = w.callback;
            w.callback = (val) => {
                if (oldCb) oldCb.call(w, val);
                resizeCanvases();
                strictFit();
            };
        });
        [opacityWidget, fillWidget].forEach(w => {
            if (!w) return;
            const oldCb = w.callback;
            w.callback = (val) => {
                if (oldCb) oldCb.call(w, val);
                sanitizeOpacity();
                updateInfo();
            };
        });

        domWidget = node.addDOMWidget("paint_canvas_widget", "custom_canvas", container, {
            getValue: () => dataWidget.value,
            setValue: v => {
                if (typeof v === "string" && v.indexOf("agsoft_paint_") === 0) restoreImage();
            }
        });
        domWidget.computeSize = function (width) {
            const iw = Math.max(120, (width || (node.size?.[0] || INIT_W) - 20));
            const tbH = toolbar.offsetHeight || 64;
            return [width || 0, tbH + canvasDispH(iw) + INFO_H + 20];
        };

        const origOnResize = node.onResize;
        node.onResize = function () {
            if (origOnResize) origOnResize.apply(this, arguments);
            strictFit();
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            setTimeout(() => {
                sanitizeOpacity();
                canvasColorInput.value = hexOr(cColorWidget?.value, "#FFFFFF");
                brushColorInput.value = hexOr(bColorWidget?.value, "#000000");
                applyBg();
                resizeCanvases();
                strictFit();
                restoreImage();
                strictFit();
            }, 60);
        };

        if ((node.size?.[0] || 0) < INIT_W) node.size[0] = INIT_W;
        applyBg();
        resizeCanvases();
        setActive(brushBtn);
        strictFit();
        restoreImage();
        setTimeout(() => strictFit(), 0);
        setTimeout(() => strictFit(), 120);
    }
});