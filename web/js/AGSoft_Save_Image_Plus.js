// ==============================================================================
// AGSoft_Save_Image_Plus.js
// ==============================================================================
// JS-расширение для ноды 🖼️AGSoft Save Image Plus.
// Версия / Version: v25.09.3
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
// ⚡ Ресайз ноды не тронут (клик перехватывается ТОЛЬКО по кнопкам панели).
//   Node resize untouched (clicks intercepted ONLY over the panel buttons).
// ⚡ Строка размера W × H под каждым превью (как в Preview Image).
//   A W × H size line under each preview (like in Preview Image).
// ⚡ Панель кнопок над каждым превью: [💾 Save][📂][][⬇] — сохранить в
//   output / открыть в новой вкладке / копировать в буфер / скачать через
//   браузер (Save As в любую папку).
//   Button row over each preview: save to output / open in a new tab /
//   copy to clipboard / download via browser (Save As to any folder).
// ⚡ Правый клик по превью — меню как в Preview Image:
//   Open Image / Copy Image / Save Image + Save to output.
//   Right-click over a preview — menu like Preview Image.
// ⚡ v20.09.3: спойлер-стрелка — все виджеты настроек сворачиваются; видимыми
//   остаются полоса-спойлер и превью; состояние хранится в node.properties и
//   переживает перезагрузку; НОВАЯ нода создаётся полностью развёрнутой.
//   Spoiler arrow — all settings widgets collapse; only the spoiler bar and
//   the preview stay visible; state is stored in node.properties and survives
//   reloads; a NEW node is created fully EXPANDED.
// ⚡ v25.09.3: информационная полоса над превью
//   [⚙ режим • формат • число превью + подсказка ......... ▼] — заполняет
//   пустое место слева от стрелки; клик по ВСЕЙ полосе сворачивает/
//   разворачивает настройки; надпись живая (обновляется после выполнения и
//   при смене save_image / image_format).
//   Info bar over the preview
//   [⚙ mode • format • preview count + hint ......... ▼] — fills the empty
//   space left of the arrow; clicking ANYWHERE on the bar collapses/expands
//   the settings; the label is live (updates after execution and on
//   save_image / image_format changes).
//
// Автор / Author: AGSoft
// Дата / Date: 25.09.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

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

        const BTN_H = 18;   // высота панели кнопок / button row height
        const ICON_W = 22;  // ширина маленьких кнопок 📂📋⬇ / small buttons width
        const FOOT_H = 14;  // высота строки размера W×H / size label height

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
        // Ссылка /view для entry (превью и все браузерные действия).
        // /view URL for an entry (previews and all browser actions).
        // ------------------------------------------------------------------
        const viewURL = (entry) => {
            const params = new URLSearchParams({
                filename: entry.filename,
                type: entry.type || "temp",
                subfolder: entry.subfolder || "",
            });
            return api.apiURL("/view?" + params.toString());
        };

        // 📂 Open: открыть превью в новой вкладке браузера.
        // Open the preview in a new browser tab.
        const openEntry = (item) => {
            if (!item || !item.entry || !item.entry.filename) {
                showToast("No preview to open.", true);
                return;
            }
            window.open(viewURL(item.entry), "_blank");
        };

        // 📋 Copy: скопировать PNG в буфер обмена (как Copy Image в Preview Image).
        // Copy PNG to clipboard (like Copy Image in Preview Image).
        const copyEntry = async (item) => {
            if (!item || !item.entry || !item.entry.filename) {
                showToast("No preview to copy.", true);
                return;
            }
            try {
                if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
                    throw new Error("Clipboard API unavailable");
                }
                const resp = await fetch(viewURL(item.entry));
                const blob = await resp.blob();
                await navigator.clipboard.write([
                    new ClipboardItem({ [blob.type || "image/png"]: blob }),
                ]);
                showToast("Image copied to clipboard.");
            } catch (err) {
                showToast(`Copy failed: ${err.message}`, true);
            }
        };

        // ⬇ Download: скачать через браузер (Save As в любую папку),
        // как Save Image в Preview Image.
        // Download via browser (Save As to any folder), like Save Image.
        const downloadEntry = async (item) => {
            if (!item || !item.entry || !item.entry.filename) {
                showToast("No preview to download.", true);
                return;
            }
            try {
                const resp = await fetch(viewURL(item.entry));
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = item.entry.filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                setTimeout(() => URL.revokeObjectURL(url), 5000);
                showToast("Download started.");
            } catch (err) {
                showToast(`Download failed: ${err.message}`, true);
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
            refreshBarLabel();
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

        // ----------------------------------------------------------------------
        // Spoiler bar: [⚙ надпись-пояснение ......... [▼]]
        // Вся полоса кликабельна — сворачивает/разворачивает блок настроек.
        // ----------------------------------------------------------------------
        const topBar = document.createElement("div");
        Object.assign(topBar.style, {
            width: "100%",
            height: "24px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "6px",
            padding: "0 4px",
            boxSizing: "border-box",
            background: "rgba(58, 63, 74, 0.35)",
            borderRadius: "4px",
            cursor: "pointer",
        });
        topBar.title =
            "Click anywhere on the bar to collapse/expand the settings block\n" +
            "Клик по полосе — свернуть/развернуть блок настроек";
        const barLabel = document.createElement("span");
        Object.assign(barLabel.style, {
            flex: "1 1 auto",
            minWidth: "0",
            overflow: "hidden",
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
            color: "#9aa4b2",
            fontSize: "10px",
            fontFamily: "sans-serif",
            textAlign: "left",
            userSelect: "none",
            pointerEvents: "none",
        });
        topBar.appendChild(barLabel);
        const toggleBtn = document.createElement("button");
        Object.assign(toggleBtn.style, {
            flex: "none",
            width: "30px",
            height: "20px",
            background: "#3a3f4a",
            color: "#fff",
            border: "none",
            borderRadius: "3px",
            fontSize: "11px",
            cursor: "pointer",
            pointerEvents: "none", // клики обрабатывает вся полоса
        });
        topBar.appendChild(toggleBtn);

        const spoilerWidget = node.addDOMWidget("agsoft_spoiler_bar", "div", topBar, {
            serialize: false,
            hideOnZoom: false,
        });
        if (spoilerWidget) {
            spoilerWidget.computeSize = function(width) {
                return [width || 200, 24];
            };
        }

        const COLLAPSIBLE = new Set([
            "save_image",
            "filename_prefix",
            "output_path",
            "create_dated_subfolder",
            "image_format",
            "png_compression",
            "jpg_quality",
            "webp_quality",
            "overwrite_existing",
            "embed_workflow",
        ]);
        
        let collapsed = false;
        const setWidgetsVisible = (visible) => {
            for (const w of node.widgets || []) {
                if (!COLLAPSIBLE.has(w.name)) continue;
                w.hidden = !visible;
                if (visible) {
                    if (typeof w.show === "function") {
                        try { w.show(); } catch (e) {}
                    }
                } else {
                    if (typeof w.hide === "function") {
                        try { w.hide(); } catch (e) {}
                    }
                }
            }
        };
        
        // Живая надпись: режим • формат • число превью + подсказка.
        function refreshBarLabel() {
            const w = (name) => {
                const widget = node.widgets && node.widgets.find(x => x.name === name);
                return widget ? widget.value : null;
            };
            const fmt = w("image_format") || "png";
            const saveOn = !!w("save_image");
            const n = (node._agsoft_items || []).length;
            const info =
                (saveOn ? "💾 save ON" : "👁 preview only") +
                " • " + fmt +
                " • " + (n === 1 ? "1 preview" : n + " previews");
            barLabel.textContent = collapsed
                ? "⚙ Settings hidden — click ▲ to expand  |  " + info
                : "⚙ " + info + "  —  click ▼ to collapse";
        }
        const updateToggle = () => {
            toggleBtn.textContent = collapsed ? "▲" : "▼";
            toggleBtn.title = collapsed
                ? "Settings hidden — click to expand\nНастройки скрыты — нажмите, чтобы развернуть"
                : "Settings shown — click to collapse\nНастройки видны — нажмите, чтобы свернуть";
            refreshBarLabel();
        };

        // Обновлять надпись при смене режима/формата пользователем.
        for (const nm of ["save_image", "image_format"]) {
            const wd = node.widgets && node.widgets.find(x => x.name === nm);
            if (wd) {
                const origCb = wd.callback;
                wd.callback = function (...args) {
                    if (origCb) origCb.apply(this, args);
                    refreshBarLabel();
                };
            }
        }

        const MIN_PREVIEW_H = 120;
        let previewExtra = 0;
        let baseHeight = null;
        
        const origCompute = node.computeSize;
        const origComputeSize = origCompute
            ? function (...args) { return origCompute.apply(node, args); }
            : null;
            
        node.computeSize = function (...args) {
            const s = origComputeSize ? origComputeSize(...args) : [this.size[0], this.size[1]];
            if (Array.isArray(s)) {
                s[1] = Math.max(0, s[1] - previewExtra);
            }
            return s;
        };

        const measureBase = () => {
            previewExtra = 0;
            const bs = node.computeSize ? node.computeSize() : [node.size[0], 0];
            const b = Math.max(0, Number(bs && bs[1]) || 0);
            baseHeight = b;
            return b;
        };

        function applyState(st, keepExtra, heightHint) {
            collapsed = !!st;
            setWidgetsVisible(!collapsed);
            updateToggle();
            node.properties = node.properties || {};
            node.properties["agsoft_widgets_collapsed"] = collapsed;
            const b = measureBase();
            let extra;
            if (heightHint != null) {
                extra = Math.max(0, heightHint - b);
            } else {
                extra = Math.max(0, keepExtra || 0);
            }
            previewExtra = extra;
            node.setSize([Math.max(220, Number(node.size[0]) || 220), b + extra]);
            node.setDirtyCanvas(true, true);
        }

        topBar.onclick = () => applyState(!collapsed, previewExtra, null);

        const syncLayout = (preserveUserHeight = true) => {
            const currentW = Math.max(220, Number(node.size && node.size[0]) || 220);
            const currentH = Math.max(0, Number(node.size && node.size[1]) || 0);
            const baseH = measureBase();
            let targetH = currentH;
            if (!preserveUserHeight || targetH < baseH) {
                targetH = baseH;
            }
            previewExtra = Math.max(0, targetH - baseH);
            node.setSize([currentW, targetH]);
            node.setDirtyCanvas(true, true);
        };

        const origOnResize = node.onResize;
        node.onResize = function (size) {
            if (baseHeight == null) {
                syncLayout(true);
            }
            previewExtra = Math.max(0, size[1] - (baseHeight || 0));
            if (origOnResize) {
                origOnResize.apply(this, arguments);
            }
            node.setDirtyCanvas(true, true);
        };

        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            const list = node.properties && node.properties["agsoft_previews"];
            if (list && list.length) buildItems(list);
            
            // Restore spoiler state
            const st = !!(node.properties && node.properties["agsoft_widgets_collapsed"]);
            const h = Number(node.size && node.size[1]) || 0;
            setTimeout(() => applyState(st, null, h), 0);
        };

        // ------------------------------------------------------------------
        // Раскладка сетки: считается при КАЖДОЙ отрисовке из текущего размера
        // ноды → превью всегда вписано; ресайз ноды = новая сетка.
        // Низ виджетов — по last_y последнего ВИДИМОГО виджета (надёжно).
        // Число колонок — из пропорций свободной области (адаптивно).
        //
        // Grid layout: computed on EVERY draw from the current node size →
        // the preview always fits; node resize = new grid.
        // Widgets bottom — from the last VISIBLE widget's last_y (reliable).
        // Column count — from the free area's aspect (adaptive).
        // ------------------------------------------------------------------
        const computeLayout = () => {
            const layout = [];
            const n = (node._agsoft_items || []).length;
            if (!n) return layout;
            let bottom = 0;
            if (node.widgets && node.widgets.length) {
                let lw = null;
                for (let i = node.widgets.length - 1; i >= 0; i--) {
                    if (!node.widgets[i].hidden) {
                        lw = node.widgets[i];
                        break;
                    }
                }
                if (lw) {
                    const lh = lw.computeSize
                        ? lw.computeSize(node.size[0])[1]
                        : (lw.height || 20);
                    bottom = (lw.last_y || 0) + lh;
                } else {
                    bottom = node.widgets_start_y || 30;
                }
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
                // В узких ячейках оставляем только [💾 Save] на всю ширину.
                // In narrow cells keep only the full-width [💾 Save].
                const showIcons = cw >= 110;
                const bx = cx + 3, by = cy + 3;
                let x = bx;
                const save = [x, by, showIcons ? Math.max(24, cw - 6 - (3 * ICON_W + 6)) : cw - 6, BTN_H];
                x += save[2] + 2;
                const open = showIcons ? [x, by, ICON_W, BTN_H] : [0, 0, 0, 0];
                if (showIcons) x += ICON_W + 2;
                const copy = showIcons ? [x, by, ICON_W, BTN_H] : [0, 0, 0, 0];
                if (showIcons) x += ICON_W + 2;
                const dl = showIcons ? [x, by, ICON_W, BTN_H] : [0, 0, 0, 0];
                // Строка размера снизу ячейки (как в Preview Image).
                // Size footer at the bottom of the cell (like Preview Image).
                const foot = [cx + 3, cy + ch - 3 - FOOT_H, cw - 6, FOOT_H];
                const imgY = by + BTN_H + 2;
                const imgH = Math.max(4, foot[1] - 2 - imgY);
                layout.push({
                    index: i,
                    btn: save, open, copy, dl,
                    imgArea: [cx + 3, imgY, cw - 6, imgH],
                    foot,
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
                    // [📂][📋][⬇] — маленькие кнопки справа от Save.
                    // [📂][📋][⬇] — small buttons to the right of Save.
                    const icons = [
                        [cell.open, "📂"],
                        [cell.copy, "📋"],
                        [cell.dl, "⬇"],
                    ];
                    for (const [r, glyph] of icons) {
                        if (!r || r[2] <= 0) continue; // скрыты в узких ячейках
                        ctx.fillStyle = "#3a3f4a";
                        ctx.fillRect(r[0], r[1], r[2], r[3]);
                        ctx.fillStyle = "#fff";
                        ctx.fillText(glyph, r[0] + r[2] / 2, r[1] + r[3] / 2 + 0.5);
                    }
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
                    // Строка размера W×H снизу ячейки (как в Preview Image).
                    // W×H size footer at the bottom (like Preview Image).
                    const [fx, fy, fw, fh] = cell.foot;
                    ctx.fillStyle = "#1b1e24";
                    ctx.fillRect(fx, fy, fw, fh);
                    const W = (item.img && item.img.naturalWidth) || item.entry.width || 0;
                    const H = (item.img && item.img.naturalHeight) || item.entry.height || 0;
                    ctx.fillStyle = "#9aa4b2";
                    ctx.font = "9px sans-serif";
                    ctx.fillText(
                        W && H ? `${W} × ${H}` : "—",
                        fx + fw / 2,
                        fy + fh / 2 + 0.5
                    );
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
        const hitAction = (pos) => {
            const layout = node._agsoft_layout || [];
            for (const cell of layout) {
                const zones = [
                    [cell.btn, "save"], [cell.open, "open"],
                    [cell.copy, "copy"], [cell.dl, "dl"],
                ];
                for (const [r, action] of zones) {
                    if (r[2] > 0 &&
                        pos[0] >= r[0] && pos[0] <= r[0] + r[2] &&
                        pos[1] >= r[1] && pos[1] <= r[1] + r[3]) {
                        return { index: cell.index, action };
                    }
                }
            }
            return null;
        };

        // Ячейка целиком (кнопки+превью+строка размера) — для правого клика.
        // Whole cell (buttons+preview+size line) — for the right-click menu.
        const hitCell = (pos) => {
            const layout = node._agsoft_layout || [];
            for (const cell of layout) {
                const left = cell.btn[0];
                const right = cell.imgArea[0] + cell.imgArea[2];
                const top = cell.btn[1];
                const bot = cell.foot[1] + cell.foot[3];
                if (pos[0] >= left && pos[0] <= right && pos[1] >= top && pos[1] <= bot) {
                    return cell.index;
                }
            }
            return -1;
        };

        const origMouseDown = node.onMouseDown;
        node.onMouseDown = function (e, pos, canvas) {
            const hit = hitAction(pos);
            if (hit) {
                const item = (node._agsoft_items || [])[hit.index];
                if (item) {
                    if (hit.action === "save") saveEntry(item);
                    else if (hit.action === "open") openEntry(item);
                    else if (hit.action === "copy") copyEntry(item);
                    else if (hit.action === "dl") downloadEntry(item);
                }
                return true; // не тащим ноду за кнопку / don't drag the node by the button
            }
            if (origMouseDown) return origMouseDown.apply(this, arguments);
            return false;
        };

        // Pointer-курсор над любой кнопкой панели.
        // Pointer cursor over any panel button.
        const origMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, pos, canvas) {
            const over = !!hitAction(pos);
            const el = canvas && canvas.canvas;
            if (el) {
                if (over) { el.style.cursor = "pointer"; node._agsoft_pointer = true; }
                else if (node._agsoft_pointer) { el.style.cursor = ""; node._agsoft_pointer = false; }
            }
            if (origMouseMove) return origMouseMove.apply(this, arguments);
            return false;
        };

        // Контекстное меню как в Preview Image (правый клик по превью).
        // Context menu like Preview Image (right-click over a preview).
        const origMenu = node.getExtraMenuOptions;
        node.getExtraMenuOptions = function (graphcanvas, options) {
            const items = node._agsoft_items || [];
            const gm = (graphcanvas && graphcanvas.graph_mouse) ||
                       (app.canvas && app.canvas.graph_mouse) || null;
            let idx = -1;
            if (gm) idx = hitCell([gm[0] - this.pos[0], gm[1] - this.pos[1]]);
            if (idx >= 0 && items[idx]) {
                const it = items[idx];
                const num = idx + 1;
                options.unshift(
                    { content: `📂 Open Image #${num} (new tab)`, callback: () => openEntry(it) },
                    { content: `📋 Copy Image #${num}`, callback: () => copyEntry(it) },
                    { content: `⬇ Save Image #${num} (browser Save As)`, callback: () => downloadEntry(it) },
                    { content: `💾 Save #${num} to output`, callback: () => saveEntry(it) },
                    null
                );
            } else if (items.length) {
                options.push(null, {
                    content: "ℹ Right-click over a preview: Open/Copy/Save",
                    disabled: true,
                });
            }
            if (origMenu) return origMenu.apply(this, arguments);
            return options;
        };

        // ------------------------------------------------------------------
        // Стартовая высота и синхронизация
        // Initial height and sync
        // ------------------------------------------------------------------
        const fresh = !(
            node.properties &&
            Object.prototype.hasOwnProperty.call(node.properties, "agsoft_widgets_collapsed")
        );
        const initialCollapsed = fresh
            ? false
            : !!node.properties["agsoft_widgets_collapsed"];
            
        setTimeout(() => {
            syncLayout(true);
            applyState(initialCollapsed, fresh ? 320 : previewExtra, null);
            app.graph.setDirtyCanvas(true);
        }, 0);
    }
});