// ==============================================================================
// AGSoft_Save_Image_Plus.js
// ==============================================================================
// JS-расширение для ноды 🖼️💾 AGSoft Save Image Plus.
// Версия / Version: v26.11.11
//
// Описание / Description:
// Фронтенд расширенной ноды сохранения изображений: превью ВСЕГО БАТЧА внутри
// ноды, панель действий над каждым изображением, фокус-режим как в Preview
// Image, сворачивание настроек информационной полосой и поддержка Nodes 2.0.
// Данные превью приходят из Python в ui.agsoft_previews (temp-PNG каждого
// изображения батча); Python-файл ноды этим расширением не изменяется.
// ---
// Frontend of the extended image saver node: a WHOLE-BATCH preview inside the
// node, an action row over every image, a focus mode like Preview Image,
// settings collapsing via the info bar and Nodes 2.0 support.
// Preview data arrives from Python in ui.agsoft_previews (a temp PNG per batch
// image); the node's Python file is not modified by this extension.
//
// Возможности / Features:
// ⚡ Превью всего батча на канвасе: адаптивная сетка вписывается в текущий
//   размер ноды, ресайз ноды = новая сетка, клип по границам ноды, letterbox.
//   Whole-batch canvas preview: adaptive grid fits the current node size,
//   node resize = new grid, clipped to node bounds, letterboxed images.
// ⚡ Стартовая высота ноды = высота виджетов + 320px под превью: изображения
//   видны сразу после выполнения, растягивать ноду вручную не нужно.
//   Initial node height = widgets height + 320px for the preview: images are
//   visible right after execution, no manual stretching required.
// ⚡ Панель кнопок НАД каждым изображением: [💾 Save][][📋][] — сохранить в
//   output / открыть в новой вкладке / копировать в буфер / скачать через
//   браузер (Save As в любую папку); hit-test, курсор pointer, тултипы.
//   Action row OVER each image: [💾 Save][📂][📋][⬇] — save to output / open in
//   a new tab / copy to clipboard / download via browser (Save As anywhere);
//   hit-testing, pointer cursor, tooltips.
// ⚡ После сохранения кнопка показывает "✔ Saved" (1.5 с) + тост с именем файла.
//   After saving the button shows "✔ Saved" (1.5 s) + a toast with the filename.
// ⚡ Кнопка Save шлёт воркфлоу из браузера (app.graph.serialize()) — он вшивается
//   в PNG (tEXt-чанки) или пишется отдельным .json для JPG/WebP/BMP (Python).
//   The Save button sends the workflow from the browser (app.graph.serialize())
//   — embedded into PNG (tEXt chunks) or a separate .json for JPG/WebP/BMP.
// ⚡ Фокус-режим как в Preview Image: клик по изображению открывает только его;
//   ✕ справа сверху — выход, ◀/▶ по бокам — листание по кругу, счётчик i / n
//   и размер W × H снизу области.
//   Focus mode like Preview Image: clicking an image opens only it; ✕ top-right
//   exits, ◀/▶ on the sides navigate cyclically, i / n counter and W × H size
//   are shown at the bottom of the area.
// ⚡ Тултипы (двуязычные) над canvas-кнопками и над изображением: всплывающая
//   DOM-подсказка следует за курсором и скрывается вне зон действий.
//   Bilingual tooltips over the canvas buttons and over the image: a floating
//   DOM hint follows the cursor and hides outside the action zones.
// ⚡ Строка размера W × H под каждым превью (как в Preview Image).
//   A W × H size line under each preview (like in Preview Image).
// ⚡ Правый клик по превью — меню как в Preview Image: Open Image (new tab) /
//   Copy Image / Save Image (browser Save As) / Save to output.
//   Right-click over a preview — a menu like Preview Image: Open Image (new
//   tab) / Copy Image / Save Image (browser Save As) / Save to output.
// ⚡ Информационная полоса [⚙ режим • формат • число превью ......... ▼]: клик по
//   всей полосе сворачивает/разворачивает виджеты настроек; надпись живая
//   (обновляется после выполнения и при смене save_image / image_format).
//   Info bar [⚙ mode • format • preview count ......... ▼]: clicking anywhere on
//   the bar collapses/expands the settings widgets; the label is live (updates
//   after execution and on save_image / image_format changes).
// ⚡ Состояние спойлера хранится в node.properties и переживает перезагрузку;
//   новая нода создаётся полностью развёрнутой.
//   Spoiler state is stored in node.properties and survives reloads; a NEW
//   node is created fully expanded.
// ⚡ Nodes 2.0 (Vue): DOM-зеркало (полоса + сетка + фокус + тултипы) строится из
//   тех же данных; в обычном режиме зеркало — скрытый виджет, поэтому набор
//   виджетов и расчёт высот канвас-версии не меняются; режим определяется
//   автоматически, видим ровно один DOM-виджет на режим.
//   Nodes 2.0 (Vue): a DOM mirror (bar + grid + focus + tooltips) is built from
//   the same data; in legacy mode the mirror is a hidden widget, so the widget
//   set and height math of the canvas version stay unchanged; the mode is
//   detected automatically, exactly one DOM widget is visible per mode.
// ⚡ Ресайз ноды не тронут: клики перехватываются только по кнопкам и зонам
//   изображений; нода свободно сжимается по вертикали (computeSize минус extra).
//   Node resize untouched: clicks are intercepted only over buttons and image
//   zones; the node shrinks freely vertically (computeSize minus extra).
// ⚡ Отрисовка обёрнута в try/catch: сбой превью никогда не ломает отрисовку и
//   взаимодействие ноды.
//   Drawing is wrapped in try/catch: a preview failure never breaks the node's
//   drawing or interaction.
//
// Архитектура / Architecture:
// - Обычный режим / Legacy: канвас-рендер сетки и фокуса + отдельный DOM-виджет
//   полосы-спойлера; раскладка считается при каждой отрисовке из размера ноды.
//   Legacy: canvas rendering of the grid and focus + a separate DOM widget for
//   the spoiler bar; layout is recomputed on every draw from the node size.
// - Nodes 2.0: канвас-хуки не вызываются, поэтому включается DOM-зеркало и
//   скрывается полоса-спойлер; зеркало заполняет область под виджетами.
//   Nodes 2.0: canvas hooks are not called, so the DOM mirror activates and the
//   spoiler bar hides; the mirror fills the area below the widgets.
// - Источник данных / Data source: ui.agsoft_previews из Python (temp-PNG с
//   уникальными uuid-именами — превью корректны при нескольких нодах в графе).
//   ui.agsoft_previews from Python (temp PNGs with unique uuid names — previews
//   stay correct with several saver nodes in one graph).
//
// Автор / Author: AGSoft
// Дата / Date: 19.11.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Save Image Plus] JS extension loaded v26.11.11 (+ focus + tooltips + Vue mirror)");

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

// ------------------------------------------------------------------------------
// Тултип для canvas-кнопок (один на страницу).
// Tooltip for canvas buttons (one per page).
// ------------------------------------------------------------------------------
let tipEl = null;
const tipShow = (text, x, y) => {
    if (!tipEl) {
        tipEl = document.createElement("div");
        Object.assign(tipEl.style, {
            position: "fixed",
            padding: "5px 8px",
            background: "#2a2e37",
            border: "1px solid #3a3f4a",
            borderRadius: "4px",
            color: "#ddd",
            fontSize: "11px",
            fontFamily: "sans-serif",
            whiteSpace: "pre-line",
            pointerEvents: "none",
            zIndex: 10000,
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
        });
        document.body.appendChild(tipEl);
    }
    tipEl.textContent = text;
    tipEl.style.display = "block";
    tipEl.style.left = Math.min(x + 14, window.innerWidth - 260) + "px";
    tipEl.style.top = (y + 18) + "px";
};
const tipHide = () => {
    if (tipEl) tipEl.style.display = "none";
};
const TIP_TEXT = {
    save: "Save to output with current settings\nСохранить в output с текущими настройками",
    open: "Open image in a new tab\nОткрыть изображение в новой вкладке",
    copy: "Copy image to clipboard\nКопировать изображение в буфер обмена",
    dl: "Download via browser (Save As)\nСкачать через браузер (Save As)",
    focus: "Click to focus this image\nКлик — открыть изображение в фокусе",
    close: "Exit focus mode\nВыйти из режима фокуса",
    prev: "Previous image\nПредыдущее изображение",
    next: "Next image\nСледующее изображение",
};

// ------------------------------------------------------------------------------
// Наблюдатель режима: Vue по настройке либо по молчанию canvas-отрисовки.
// Mode watchdog: Vue via setting or via canvas draw silence.
// ------------------------------------------------------------------------------
const sipControllers = new Set();
let sipTimer = null;
const sipTick = () => {
    const now = Date.now();
    for (const ctl of sipControllers) {
        try { ctl.checkMode(now); } catch (e) {}
    }
};
const sipWatch = (ctl) => {
    sipControllers.add(ctl);
    if (!sipTimer) sipTimer = setInterval(sipTick, 500);
};
const sipUnwatch = (ctl) => {
    sipControllers.delete(ctl);
    if (!sipControllers.size && sipTimer) {
        clearInterval(sipTimer);
        sipTimer = null;
    }
};
const vueNodesSetting = () => {
    try {
        const v = app.ui.settings.getSettingValue("Comfy.VueNodes.Enabled");
        if (typeof v === "boolean") return v;
    } catch (e) {}
    return null;
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
        // Фокус-режим: индекс одиночного изображения, -1 = сетка.
        // Focus mode: index of the single image, -1 = grid.
        let focusIndex = -1;
        let canvasAt = Date.now();
        let vueMode = false;
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
                    if (item.domSave) {
                        item.domSave.textContent = "✔ Saved";
                        setTimeout(() => { item.domSave.textContent = "💾 Save"; }, 1500);
                    }
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
            focusIndex = -1;
            for (const entry of list) {
                const item = { entry, img: null, aspect: null, flash: 0, domSave: null };
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
            buildMirror();
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
            spoilerWidget.computeSize = function (width) {
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
            const text = collapsed
                ? "⚙ Settings hidden — click ▲ to expand  |  " + info
                : "⚙ " + info + "  —  click ▼ to collapse";
            barLabel.textContent = text;
            if (mirrorLabel) mirrorLabel.textContent = text;
        }
        const updateToggle = () => {
            toggleBtn.textContent = collapsed ? "▲" : "▼";
            toggleBtn.title = collapsed
                ? "Settings hidden — click to expand\nНастройки скрыты — нажмите, чтобы развернуть"
                : "Settings shown — click to collapse\nНастройки видны — нажмите, чтобы свернуть";
            if (mirrorBtn) mirrorBtn.textContent = collapsed ? "▲" : "▼";
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
            // В Vue высоту не пересчитываем: раскладкой управляет сам рендерер.
            // In Vue we do not recompute height: the renderer owns the layout.
            if (vueMode) {
                node.setDirtyCanvas(true, true);
                return;
            }
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
        // Свободная область превью и раскладка фокус-режима.
        // Free preview area and focus-mode layout.
        // ------------------------------------------------------------------
        const previewArea = () => {
            let bottom = node.widgets_start_y || 30;
            if (node.widgets && node.widgets.length) {
                for (let i = node.widgets.length - 1; i >= 0; i--) {
                    const w = node.widgets[i];
                    if (w.hidden) continue;
                    const h = w.computeSize ? w.computeSize(node.size[0])[1] : (w.height || 20);
                    bottom = (w.last_y || 0) + h;
                    break;
                }
            }
            const ax = 6;
            const ay = bottom + 6;
            const aw = node.size[0] - 12;
            const ah = node.size[1] - ay - 6;
            if (aw < 24 || ah < 24) return null;
            return [ax, ay, aw, ah];
        };
        const focusLayout = (area) => {
            const [ax, ay, aw, ah] = area;
            const by = ay + 3;
            let x = ax + 3;
            const save = [x, by, Math.max(60, Math.min(160, aw * 0.4)), BTN_H];
            x += save[2] + 2;
            const open = [x, by, ICON_W, BTN_H];
            x += ICON_W + 2;
            const copy = [x, by, ICON_W, BTN_H];
            x += ICON_W + 2;
            const dl = [x, by, ICON_W, BTN_H];
            const close = [ax + aw - 3 - ICON_W, by, ICON_W, BTN_H];
            const img = [ax + 3, by + BTN_H + 2, aw - 6, Math.max(8, ah - BTN_H - 20)];
            const prev = [img[0] + 4, img[1] + img[3] / 2 - 17, 24, 34];
            const next = [img[0] + img[2] - 28, img[1] + img[3] / 2 - 17, 24, 34];
            return { save, open, copy, dl, close, img, prev, next };
        };
        const fillBtn = (ctx, r, color, glyph) => {
            ctx.fillStyle = color;
            ctx.fillRect(r[0], r[1], r[2], r[3]);
            ctx.fillStyle = "#fff";
            ctx.font = "10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(glyph, r[0] + r[2] / 2, r[1] + r[3] / 2 + 0.5);
        };
        const drawFocus = (ctx, area) => {
            const items = node._agsoft_items || [];
            const item = items[focusIndex];
            if (!item) {
                focusIndex = -1;
                return;
            }
            const [ax, ay, aw, ah] = area;
            const L = focusLayout(area);
            const now = Date.now();
            const flashed = item.flash && (now - item.flash < 1500);
            fillBtn(ctx, L.save, flashed ? "#3b7" : "#2a6", flashed ? "✔ Saved" : "💾 Save");
            fillBtn(ctx, L.open, "#3a3f4a", "📂");
            fillBtn(ctx, L.copy, "#3a3f4a", "📋");
            fillBtn(ctx, L.dl, "#3a3f4a", "⬇");
            fillBtn(ctx, L.close, "#a33", "✕");
            ctx.fillStyle = "#111";
            ctx.fillRect(L.img[0], L.img[1], L.img[2], L.img[3]);
            if (item.img && item.aspect) {
                const scale = Math.min(
                    L.img[2] / item.img.naturalWidth,
                    L.img[3] / item.img.naturalHeight
                );
                const w = item.img.naturalWidth * scale;
                const h = item.img.naturalHeight * scale;
                ctx.drawImage(
                    item.img,
                    L.img[0] + (L.img[2] - w) / 2,
                    L.img[1] + (L.img[3] - h) / 2,
                    w,
                    h
                );
            }
            if (items.length > 1) {
                fillBtn(ctx, L.prev, "rgba(58,63,74,0.85)", "◀");
                fillBtn(ctx, L.next, "rgba(58,63,74,0.85)", "▶");
            }
            ctx.fillStyle = "#9aa4b2";
            ctx.font = "10px sans-serif";
            ctx.textBaseline = "middle";
            ctx.textAlign = "right";
            ctx.fillText(`${focusIndex + 1} / ${items.length}`, ax + aw - 4, ay + ah - 8);
            ctx.textAlign = "left";
            const W = (item.img && item.img.naturalWidth) || item.entry.width || 0;
            const H = (item.img && item.img.naturalHeight) || item.entry.height || 0;
            ctx.fillText(W && H ? `${W} × ${H}` : "—", ax + 4, ay + ah - 8);
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
                canvasAt = Date.now();
                if (this.flags && this.flags.collapsed) return;
                const items = this._agsoft_items || [];
                const layout = computeLayout();
                this._agsoft_layout = layout;
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, 0, this.size[0], this.size[1]);
                ctx.clip();
                if (focusIndex >= 0) {
                    const area = previewArea();
                    if (area) drawFocus(ctx, area);
                    ctx.restore();
                    return;
                }
                if (!layout.length) {
                    ctx.restore();
                    return;
                }
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
        // Hit-test кнопок Save и зон фокуса. Перехватываем клик ТОЛЬКО по ним.
        // Save button and focus zone hit-testing. Intercept clicks ONLY there.
        // ------------------------------------------------------------------
        const inRect = (pos, r) => (
            r && r[2] > 0 &&
            pos[0] >= r[0] && pos[0] <= r[0] + r[2] &&
            pos[1] >= r[1] && pos[1] <= r[1] + r[3]
        );
        const hitAction = (pos) => {
            if (focusIndex >= 0) {
                const area = previewArea();
                if (!area) return null;
                const L = focusLayout(area);
                const zones = [
                    [L.save, "save"], [L.open, "open"],
                    [L.copy, "copy"], [L.dl, "dl"],
                    [L.close, "close"], [L.prev, "prev"], [L.next, "next"],
                ];
                for (const [r, action] of zones) {
                    if (inRect(pos, r)) return { index: focusIndex, action };
                }
                return null;
            }
            const layout = node._agsoft_layout || [];
            for (const cell of layout) {
                const zones = [
                    [cell.btn, "save"], [cell.open, "open"],
                    [cell.copy, "copy"], [cell.dl, "dl"],
                    [cell.imgArea, "focus"],
                ];
                for (const [r, action] of zones) {
                    if (inRect(pos, r)) return { index: cell.index, action };
                }
            }
            return null;
        };
        // Ячейка целиком (кнопки+превью+строка размера) — для правого клика.
        // Whole cell (buttons+preview+size line) — for the right-click menu.
        const hitCell = (pos) => {
            if (focusIndex >= 0) {
                const area = previewArea();
                return area && inRect(pos, area) ? focusIndex : -1;
            }
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
                const items = node._agsoft_items || [];
                const item = items[hit.index];
                const n = items.length;
                if (hit.action === "focus") {
                    focusIndex = hit.index;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
                if (hit.action === "close") {
                    focusIndex = -1;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
                if (hit.action === "prev" && n) {
                    focusIndex = (focusIndex - 1 + n) % n;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
                if (hit.action === "next" && n) {
                    focusIndex = (focusIndex + 1) % n;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
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
        // Pointer-курсор и тултип над кнопками панели и над изображением.
        // Pointer cursor and tooltip over the panel buttons and the image.
        const origMouseMove = node.onMouseMove;
        node.onMouseMove = function (e, pos, canvas) {
            const hit = hitAction(pos);
            const el = canvas && canvas.canvas;
            if (el) {
                if (hit) { el.style.cursor = "pointer"; node._agsoft_pointer = true; }
                else if (node._agsoft_pointer) { el.style.cursor = ""; node._agsoft_pointer = false; }
            }
            if (hit && TIP_TEXT[hit.action] && e && e.clientX != null) {
                tipShow(TIP_TEXT[hit.action], e.clientX, e.clientY);
            } else {
                tipHide();
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
        // DOM-зеркало для Nodes 2.0: полоса + сетка + фокус из тех же данных.
        // В обычном режиме это hidden-виджет — раскладка v25.09.3 не меняется.
        // DOM mirror for Nodes 2.0: bar + grid + focus from the same data.
        // In legacy mode it is a hidden widget — v25.09.3 layout stays intact.
        // ------------------------------------------------------------------
        const mWrap = document.createElement("div");
        Object.assign(mWrap.style, {
            position: "relative",
            width: "100%",
            height: "348px",
            display: "none",
            flexDirection: "column",
            background: "#14161b",
            borderRadius: "4px",
            overflow: "hidden",
            boxSizing: "border-box",
        });
        const mBar = document.createElement("div");
        Object.assign(mBar.style, {
            flex: "none",
            height: "24px",
            marginBottom: "4px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "6px",
            padding: "0 4px",
            boxSizing: "border-box",
            background: "#262b36",
            border: "1px solid #3a3f4a",
            borderRadius: "4px",
            cursor: "pointer",
        });
        mBar.title = topBar.title;
        const mirrorLabel = document.createElement("span");
        Object.assign(mirrorLabel.style, {
            flex: "1 1 auto",
            minWidth: "0",
            overflow: "hidden",
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
            color: "#9aa4b2",
            fontSize: "10px",
            fontFamily: "sans-serif",
            pointerEvents: "none",
        });
        mBar.appendChild(mirrorLabel);
        const mirrorBtn = document.createElement("button");
        Object.assign(mirrorBtn.style, {
            flex: "none",
            width: "30px",
            height: "20px",
            background: "#3a3f4a",
            color: "#fff",
            border: "none",
            borderRadius: "3px",
            fontSize: "11px",
            pointerEvents: "none",
        });
        mirrorBtn.textContent = "▼";
        mBar.appendChild(mirrorBtn);
        mWrap.appendChild(mBar);
        mBar.onclick = () => applyState(!collapsed, previewExtra, null);
        const mBody = document.createElement("div");
        Object.assign(mBody.style, {
            position: "relative",
            flex: "1 1 auto",
            minHeight: "0",
            overflow: "hidden",
        });
        mWrap.appendChild(mBody);
        const mGrid = document.createElement("div");
        Object.assign(mGrid.style, {
            position: "absolute",
            inset: "0",
            display: "grid",
            gap: "4px",
            padding: "4px",
            overflowY: "auto",
            boxSizing: "border-box",
        });
        mBody.appendChild(mGrid);
        const mEmpty = document.createElement("div");
        Object.assign(mEmpty.style, {
            position: "absolute",
            inset: "0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#6b7280",
            fontSize: "11px",
            fontFamily: "sans-serif",
            pointerEvents: "none",
        });
        mEmpty.textContent = "▶ Run the workflow to see previews / Запустите воркфлоу";
        mBody.appendChild(mEmpty);
        const mFocus = document.createElement("div");
        Object.assign(mFocus.style, {
            position: "absolute",
            inset: "0",
            display: "none",
            background: "#111",
        });
        mBody.appendChild(mFocus);
        const mBtn = (text, bg, tip) => {
            const b = document.createElement("button");
            b.textContent = text;
            b.title = tip;
            Object.assign(b.style, {
                border: "none",
                borderRadius: "3px",
                color: "#fff",
                fontSize: "10px",
                fontFamily: "sans-serif",
                height: "18px",
                cursor: "pointer",
                padding: "0 4px",
                background: bg,
            });
            return b;
        };
        const mFImg = document.createElement("img");
        Object.assign(mFImg.style, {
            position: "absolute",
            left: "8px",
            top: "28px",
            width: "calc(100% - 16px)",
            height: "calc(100% - 50px)",
            objectFit: "contain",
        });
        mFocus.appendChild(mFImg);
        const mFTools = document.createElement("div");
        Object.assign(mFTools.style, {
            position: "absolute", top: "4px", left: "6px", display: "flex", gap: "2px",
        });
        const mFSave = mBtn("💾 Save", "#2a6", TIP_TEXT.save);
        const mFOpen = mBtn("📂", "#3a3f4a", TIP_TEXT.open);
        const mFCopy = mBtn("📋", "#3a3f4a", TIP_TEXT.copy);
        const mFDl = mBtn("⬇", "#3a3f4a", TIP_TEXT.dl);
        mFTools.appendChild(mFSave);
        mFTools.appendChild(mFOpen);
        mFTools.appendChild(mFCopy);
        mFTools.appendChild(mFDl);
        mFocus.appendChild(mFTools);
        const mFClose = mBtn("✕", "#a33", TIP_TEXT.close);
        Object.assign(mFClose.style, { position: "absolute", top: "4px", right: "6px", width: "22px" });
        mFocus.appendChild(mFClose);
        const mFPrev = mBtn("◀", "rgba(58,63,74,0.85)", TIP_TEXT.prev);
        Object.assign(mFPrev.style, {
            position: "absolute", left: "6px", top: "50%", transform: "translateY(-50%)",
            width: "24px", height: "34px",
        });
        mFocus.appendChild(mFPrev);
        const mFNext = mBtn("▶", "rgba(58,63,74,0.85)", TIP_TEXT.next);
        Object.assign(mFNext.style, {
            position: "absolute", right: "6px", top: "50%", transform: "translateY(-50%)",
            width: "24px", height: "34px",
        });
        mFocus.appendChild(mFNext);
        const mFCount = document.createElement("div");
        Object.assign(mFCount.style, {
            position: "absolute", right: "8px", bottom: "4px", color: "#9aa4b2",
            fontSize: "10px", fontFamily: "sans-serif",
        });
        mFocus.appendChild(mFCount);
        const mFSize = document.createElement("div");
        Object.assign(mFSize.style, {
            position: "absolute", left: "8px", bottom: "4px", color: "#9aa4b2",
            fontSize: "10px", fontFamily: "sans-serif",
        });
        mFocus.appendChild(mFSize);
        const mirrorWidget = node.addDOMWidget("agsoft_preview_mirror", "div", mWrap, {
            serialize: false,
            hideOnZoom: false,
        });
        const syncMirrorFocus = () => {
            const items = node._agsoft_items || [];
            const it = focusIndex >= 0 ? items[focusIndex] : null;
            if (!it) {
                mFocus.style.display = "none";
                mGrid.style.display = "grid";
                return;
            }
            mGrid.style.display = "none";
            mFocus.style.display = "block";
            mFImg.src = viewURL(it.entry);
            mFCount.textContent = `${focusIndex + 1} / ${items.length}`;
            const W = (it.img && it.img.naturalWidth) || it.entry.width || 0;
            const H = (it.img && it.img.naturalHeight) || it.entry.height || 0;
            mFSize.textContent = W && H ? `${W} × ${H}` : "—";
            const many = items.length > 1;
            mFPrev.style.display = many ? "block" : "none";
            mFNext.style.display = many ? "block" : "none";
        };
        function buildMirror() {
            const items = node._agsoft_items || [];
            mGrid.textContent = "";
            mEmpty.style.display = items.length ? "none" : "flex";
            const n = items.length;
            if (n) {
                const w = mBody.clientWidth || 220;
                const h = mBody.clientHeight || 320;
                let cols = Math.round(Math.sqrt(n * (w / Math.max(1, h))));
                cols = Math.max(1, Math.min(n, cols || 1));
                const rows = Math.ceil(n / cols);
                const rowH = Math.max(110, Math.floor((h - 8 - (rows - 1) * 4) / rows));
                mGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
                mGrid.style.gridTemplateRows = `repeat(${rows}, ${rowH}px)`;
            }
            items.forEach((item, idx) => {
                const cell = document.createElement("div");
                Object.assign(cell.style, {
                    display: "flex", flexDirection: "column", minHeight: "0", minWidth: "0",
                    background: "#1b1e24", border: "1px solid #2a2e37", borderRadius: "4px",
                    overflow: "hidden",
                });
                const row = document.createElement("div");
                Object.assign(row.style, { display: "flex", gap: "2px", padding: "3px", flex: "none" });
                const save = mBtn("💾 Save", "#2a6", TIP_TEXT.save);
                save.style.flex = "1 1 auto";
                const open = mBtn("📂", "#3a3f4a", TIP_TEXT.open);
                const copy = mBtn("📋", "#3a3f4a", TIP_TEXT.copy);
                const dl = mBtn("⬇", "#3a3f4a", TIP_TEXT.dl);
                row.appendChild(save);
                row.appendChild(open);
                row.appendChild(copy);
                row.appendChild(dl);
                cell.appendChild(row);
                const wrap = document.createElement("div");
                Object.assign(wrap.style, {
                    position: "relative", flex: "1 1 auto", minHeight: "0",
                    background: "#111", cursor: "pointer", overflow: "hidden",
                });
                wrap.title = TIP_TEXT.focus;
                const im = document.createElement("img");
                Object.assign(im.style, {
                    position: "absolute", inset: "0", width: "100%", height: "100%",
                    objectFit: "contain", display: "block",
                });
                im.src = viewURL(item.entry);
                wrap.appendChild(im);
                cell.appendChild(wrap);
                const foot = document.createElement("div");
                Object.assign(foot.style, {
                    flex: "none", height: "14px", lineHeight: "14px", textAlign: "center",
                    fontSize: "9px", fontFamily: "sans-serif", color: "#9aa4b2",
                });
                foot.textContent = item.entry.width && item.entry.height
                    ? `${item.entry.width} × ${item.entry.height}` : "—";
                im.onload = () => {
                    if (im.naturalWidth && im.naturalHeight) {
                        foot.textContent = `${im.naturalWidth} × ${im.naturalHeight}`;
                    }
                };
                cell.appendChild(foot);
                item.domSave = save;
                save.onclick = () => saveEntry(item);
                open.onclick = () => openEntry(item);
                copy.onclick = () => copyEntry(item);
                dl.onclick = () => downloadEntry(item);
                wrap.onclick = () => { focusIndex = idx; syncMirrorFocus(); node.setDirtyCanvas(true, true); };
                mGrid.appendChild(cell);
            });
            syncMirrorFocus();
        }
        mFSave.onclick = () => { const it = (node._agsoft_items || [])[focusIndex]; if (it) saveEntry(it); };
        mFOpen.onclick = () => { const it = (node._agsoft_items || [])[focusIndex]; if (it) openEntry(it); };
        mFCopy.onclick = () => { const it = (node._agsoft_items || [])[focusIndex]; if (it) copyEntry(it); };
        mFDl.onclick = () => { const it = (node._agsoft_items || [])[focusIndex]; if (it) downloadEntry(it); };
        mFClose.onclick = () => { focusIndex = -1; syncMirrorFocus(); node.setDirtyCanvas(true, true); };
        mFPrev.onclick = () => {
            const n = (node._agsoft_items || []).length;
            if (n) { focusIndex = (focusIndex - 1 + n) % n; syncMirrorFocus(); }
        };
        mFNext.onclick = () => {
            const n = (node._agsoft_items || []).length;
            if (n) { focusIndex = (focusIndex + 1) % n; syncMirrorFocus(); }
        };
        const mRO = new ResizeObserver(() => { if (vueMode) buildMirror(); });
        mRO.observe(mBody);
        // Переключение режима: ровно один видимый DOM-виджет на режим.
        // Mode switch: exactly one visible DOM widget per mode.
        let vueSpaceGiven = false;
        const setVueMode = (on) => {
            if (vueMode === on) return;
            vueMode = on;
            if (mirrorWidget) {
                mirrorWidget.hidden = !on;
                mirrorWidget.computeSize = on
                    ? function (width) { return [width || 220, 348]; }
                    : function () { return [0, -4]; };
            }
            mWrap.style.display = on ? "flex" : "none";
            if (spoilerWidget) {
                spoilerWidget.hidden = on;
                spoilerWidget.computeSize = on
                    ? function () { return [0, -4]; }
                    : function (width) { return [width || 200, 24]; };
            }
            topBar.style.display = on ? "none" : "flex";
            if (on) {
                buildMirror();
                if (!vueSpaceGiven) {
                    vueSpaceGiven = true;
                    const saved = Number(node.size && node.size[1]) || 0;
                    const hasPreviewSpace = !!(node.properties && node.properties["agsoft_previews"]);
                    if (!hasPreviewSpace) {
                        node.setSize([Math.max(220, Number(node.size[0]) || 220), saved + 348]);
                    }
                }
            } else {
                baseHeight = null;
                syncLayout(true);
            }
            node.setDirtyCanvas(true, true);
        };
        if (mirrorWidget) {
            mirrorWidget.hidden = true;
            mirrorWidget.computeSize = function () { return [0, -4]; };
        }
        const ctl = {
            checkMode(now) {
                const setting = vueNodesSetting();
                const vue = setting != null ? setting : (now - canvasAt) > 1500;
                setVueMode(vue);
            },
        };
        sipWatch(ctl);
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            sipUnwatch(ctl);
            mRO.disconnect();
            tipHide();
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
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