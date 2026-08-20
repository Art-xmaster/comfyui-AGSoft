// ==============================================================================
// AGSoft_Save_Image_Plus.js
// ==============================================================================
// JS-расширение для ноды 🖼️💾AGSoft Save Image Plus.
//
// Возможности / Features:
// ⚡ Превью ВСЕГО БАТЧА на канвасе (адаптивная сетка, вписывается в размер
//   ноды; ресайз ноды = новая сетка; клип по границам ноды).
//   Whole-batch preview on the canvas (adaptive grid fitted into the node
//   size; node resize = new grid; clipped to the node bounds).
// ⚡ Кнопка "💾 Save" НАД каждым изображением (hit-test, курсор pointer,
//   после сохранения "✔ Saved" + тост с именем файла).
//   A "💾 Save" button OVER each image (hit-test, pointer cursor,
//   "✔ Saved" + toast with the filename after saving).
// ⚡ Кнопка шлёт воркфлоу из браузера (app.graph.serialize()) — воркфлоу
//   вшивается в сохранённое изображение (tEXt-чанк "workflow").
//   The button sends the workflow from the browser (app.graph.serialize()) —
//   the workflow is embedded into the saved image (tEXt "workflow" chunk).
// ⚡ Стартовая высота ноды = высота виджетов + 320px под превью — изображения
//   видны СРАЗУ после добавления ноды, растягивать вручную не нужно.
//   Initial node height = widgets height + 320px for the preview — images are
//   visible IMMEDIATELY after adding the node, no manual stretching needed.
// ⚡ Превью сохраняется в node.properties → переживает перезагрузку воркфлоу.
//   The preview is stored in node.properties → survives workflow reloads.
// ⚡ Ресайз ноды не тронут (клик перехватывается ТОЛЬКО по кнопкам Save).
//   Node resize untouched (clicks intercepted ONLY over the Save buttons).
//
// JS extension for the 🖼️AGSoft Save Image Plus node.
// (See the RU list above — the same features.)
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// console.log("[AGSoft Save Image Plus] JS extension loaded v20.08 (style header + batch preview + Save now over each image)");

// ------------------------------------------------------------------------------
// Простой toast (для показа пути после сохранения).
// Simple toast (to show the path after saving).
// ------------------------------------------------------------------------------
const showToast = (msg, isError) => {
    const t = document.createElement("div");
    t.textContent = msg;
    Object.assign(t.style, {
        position: "fixed",
        bottom: "24px",
        right: "24px",
        padding: "10px 14px",
        background: isError ? "#a33" : "#2a6",
        color: "#fff",
        borderRadius: "6px",
        fontFamily: "sans-serif",
        fontSize: "13px",
        zIndex: 9999,
        boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
        maxWidth: "70vw",
        wordBreak: "break-word",
        transition: "opacity 0.3s",
    });
    document.body.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        setTimeout(() => t.remove(), 300);
    }, 3000);
};

app.registerExtension({
    name: "AGSoft.SaveImagePlus",

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftSaveImagePlus") return;

        // Элементы превью: { entry, img, aspect, flash }.
        // Preview items: { entry, img, aspect, flash }.
        node._agsoft_items = [];
        // Раскладка ячеек для hit-test: [{ index, btn:[x,y,w,h], imgArea:[...] }].
        // Cell layout for hit-testing.
        node._agsoft_layout = [];

        const BTN_H = 18;

        // ------------------------------------------------------------------
        // Чтение настроек виджетов (для Save).
        // Read widget settings (for Save).
        // ------------------------------------------------------------------
        const readParams = () => {
            const w = (name) => {
                const widget = node.widgets && node.widgets.find(x => x.name === name);
                return widget ? widget.value : null;
            };
            return {
                filename_prefix: w("filename_prefix") || "image",
                output_path: w("output_path") || "",
                create_dated_subfolder: !!w("create_dated_subfolder"),
                image_format: w("image_format") || "png",
                png_compression: w("png_compression") ?? 1,
                jpg_quality: w("jpg_quality") ?? 90,
                webp_quality: w("webp_quality") ?? 90,
                overwrite_existing: !!w("overwrite_existing"),
                embed_workflow: !!w("embed_workflow"),
            };
        };

        // ------------------------------------------------------------------
        // Сохранение конкретного превью (temp-файла) в output.
        // ВАЖНО: шлём воркфлоу из браузера — иначе вшивать будет нечего.
        // Save a specific preview (temp file) to output.
        // IMPORTANT: send the workflow from the browser — otherwise there is
        // nothing to embed.
        // ------------------------------------------------------------------
        const saveEntry = async (item) => {
            const entry = item && item.entry;
            if (!entry || !entry.filename) {
                showToast("No preview to save.", true);
                return;
            }

            try {
                const resp = await fetch(api.apiURL("/agsoft/save_now"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        temp_path: entry.fullpath || "",
                        temp_filename: entry.filename,
                        params: readParams(),
                        // Воркфлоу из браузера — чистый граф.
                        // Workflow from the browser — pure graph.
                        workflow: app.graph.serialize(),
                    }),
                });

                const data = await resp.json();

                if (data && data.ok) {
                    item.flash = Date.now();
                    node.setDirtyCanvas(true, true);
                    showToast(`Saved: ${data.filename}`, false);
                    console.log("[AGSoft Save Image Plus] saved:", data);
                } else {
                    showToast(`Save failed: ${data?.error || "unknown"}`, true);
                }
            } catch (err) {
                console.warn("[AGSoft Save Image Plus] save_now error:", err);
                showToast(`Save failed: ${err.message}`, true);
            }
        };

        // ------------------------------------------------------------------
        // Построение элементов превью из списка (без DOM — только Image()).
        // Build preview items from the list (no DOM — Image() only).
        // ------------------------------------------------------------------
        const buildItems = (list) => {
            node._agsoft_items = [];

            for (const entry of list) {
                const item = { entry, img: null, aspect: null, flash: 0 };

                const im = new Image();
                im.onload = () => {
                    item.img = im;
                    if (im.naturalWidth && im.naturalHeight) {
                        item.aspect = im.naturalWidth / im.naturalHeight;
                    }
                    node.setDirtyCanvas(true, true);
                };

                const params = new URLSearchParams({
                    filename: entry.filename,
                    type: entry.type || "temp",
                    subfolder: entry.subfolder || "",
                });
                im.src = api.apiURL("/view?" + params.toString());

                node._agsoft_items.push(item);
            }

            node.setDirtyCanvas(true, true);
        };

        node.properties = node.properties || {};

        const origExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);

            const list = output && output.agsoft_previews;
            if (list && list.length) {
                node.properties["agsoft_previews"] = list;
                buildItems(list);
            }
        };

        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);

            const list = node.properties && node.properties["agsoft_previews"];
            if (list && list.length) buildItems(list);
        };

        // ------------------------------------------------------------------
        // Раскладка сетки: считается при КАЖДОЙ отрисовке из текущего размера
        // ноды → превью всегда вписано; ресайз ноды = новая сетка.
        // Низ виджетов — по last_y последнего виджета (надёжно).
        // Число колонок — из пропорций свободной области (адаптивно).
        //
        // Grid layout: computed on EVERY draw from the current node size →
        // the preview always fits; node resize = new grid.
        // Widgets bottom — from the last widget's last_y (reliable).
        // Column count — from the free area's aspect (adaptive).
        // ------------------------------------------------------------------
        const computeLayout = () => {
            const layout = [];
            const n = (node._agsoft_items || []).length;
            if (!n) return layout;

            let bottom = 0;
            if (node.widgets && node.widgets.length) {
                const lw = node.widgets[node.widgets.length - 1];
                const lh = lw.computeSize
                    ? lw.computeSize(node.size[0])[1]
                    : (lw.height || 20);
                bottom = (lw.last_y || 0) + lh;
            } else {
                bottom = node.widgets_start_y || 30;
            }

            const areaX = 6;
            const areaY = bottom + 6;
            const areaW = node.size[0] - 12;
            const areaH = node.size[1] - areaY - 6;
            if (areaW < 24 || areaH < 24) return layout;

            // Адаптивная сетка: cols из пропорций области.
            // Adaptive grid: cols from the area's aspect.
            const aspect = areaW / Math.max(1, areaH);
            let cols = Math.round(Math.sqrt(n * aspect));
            cols = Math.max(1, Math.min(n, cols || 1));
            const rows = Math.ceil(n / cols);
            const cw = areaW / cols;
            const ch = areaH / rows;

            for (let i = 0; i < n; i++) {
                const cx = areaX + (i % cols) * cw;
                const cy = areaY + Math.floor(i / cols) * ch;

                layout.push({
                    index: i,
                    btn: [cx + 3, cy + 3, cw - 6, BTN_H],
                    imgArea: [cx + 3, cy + 3 + BTN_H + 2, cw - 6, Math.max(4, ch - BTN_H - 8)],
                });
            }

            return layout;
        };

        // ------------------------------------------------------------------
        // Отрисовка: клип по ноде + кнопка Save + изображение (letterbox).
        // Drawing: clip to the node + Save button + letterboxed image.
        // ------------------------------------------------------------------
        node.onDrawForeground = function (ctx) {
            try {
                if (this.flags && this.flags.collapsed) return;

                const items = this._agsoft_items || [];
                const layout = computeLayout();
                this._agsoft_layout = layout;
                if (!layout.length) return;

                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, this.size[0], this.size[1]);
                ctx.clip();

                const now = Date.now();

                for (const cell of layout) {
                    const item = items[cell.index];
                    if (!item) continue;

                    const [bx, by, bw, bh] = cell.btn;
                    const flashed = item.flash && (now - item.flash < 1500);

                    ctx.fillStyle = flashed ? "#3b7" : "#2a6";
                    ctx.fillRect(bx, by, bw, bh);
                    ctx.fillStyle = "#fff";
                    ctx.font = "10px sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(
                        flashed ? "✔ Saved" : "💾 Save",
                        bx + bw / 2,
                        by + bh / 2 + 0.5
                    );

                    const [ix, iy, iw, ih] = cell.imgArea;

                    ctx.fillStyle = "#111";
                    ctx.fillRect(ix, iy, iw, ih);

                    if (item.img && item.aspect) {
                        const scale = Math.min(
                            iw / item.img.naturalWidth,
                            ih / item.img.naturalHeight
                        );
                        const w = item.img.naturalWidth * scale;
                        const h = item.img.naturalHeight * scale;
                        ctx.drawImage(
                            item.img,
                            ix + (iw - w) / 2,
                            iy + (ih - h) / 2,
                            w,
                            h
                        );
                    }
                }

                ctx.restore();
            } catch (e) {
                // Никогда не ломаем отрисовку/взаимодействие ноды.
                // Never break the node's drawing/interaction.
            }
        };

        // ------------------------------------------------------------------
        // Hit-test кнопок Save. Перехватываем клик ТОЛЬКО по кнопкам.
        // Save button hit-testing. We intercept clicks ONLY over the buttons.
        // ------------------------------------------------------------------
        const hitButton = (pos) => {
            const layout = node._agsoft_layout || [];
            for (const cell of layout) {
                const [bx, by, bw, bh] = cell.btn;
                if (pos[0] >= bx && pos[0] <= bx + bw &&
                    pos[1] >= by && pos[1] <= by + bh) {
                    return cell.index;
                }
            }
            return -1;
        };

        const origMouseDown = node.onMouseDown;
        node.onMouseDown = function (e, pos, canvas) {
            const idx = hitButton(pos);
            if (idx >= 0) {
                const item = (node._agsoft_items || [])[idx];
                if (item) saveEntry(item);
                return true; // не тащим ноду за кнопку / don't drag the node by the button
            }
            if (origMouseDown) return origMouseDown.apply(this, arguments);
            return false;
        };

        // ------------------------------------------------------------------
        // Стартовый размер: ширина 320, высота = виджеты + 320px под превью.
        // Изображения видны СРАЗУ, растягивать вручную не нужно.
        // Initial size: width 320, height = widgets + 320px for the preview.
        // Images are visible IMMEDIATELY, no manual stretching needed.
        // ------------------------------------------------------------------
        let widgetsH = 30;
        if (node.widgets) {
            for (const w of node.widgets) {
                const h = w.computeSize
                    ? w.computeSize(320)[1]
                    : (w.height || 20);
                widgetsH += h + 4;
            }
        }
        node.setSize([320, widgetsH + 320]);
        app.graph.setDirtyCanvas(true);
    }
});