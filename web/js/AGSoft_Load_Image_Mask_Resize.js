/*
==============================================================================
AGSoft_Load_Image_Mask_Resize.js
==============================================================================
Расширение: 🖼️AGSoft Load Image & Mask Resize
Описание / Description:
    Клиентская часть ноды AGSoft_Load_Image_Mask_Resize.
    1) Превью рисует DOM-виджет ноды, но node.imgs остаётся настоящим:
       от него зависят MaskEditor и пункты меню ноды. Штатный показ
       картинки гасится флагом hideOutputImages и скрытием core-виджета
       превью, поэтому дубля нет, а «Open in MaskEditor» работает.
    2) Правый клик по превью и статус-бару открывает стандартное меню
       ComfyUI (MaskEditor, Open/Copy/Save Image, свойства ноды и т.д.).
       Результат MaskEditor (clipspace) сразу становится источником.
    3) Превью вписывается в текущий размер ноды и НЕ меняет его. Минимум
       высоты (computeSize) не зависит от текущего размера, поэтому ноду
       можно свободно растягивать и сжимать по вертикали в обе стороны.
       При скрытии/показе виджетов область превью сохраняется.
    4) Превью собирается на canvas тем же пайплайном, что и сервер:
       rotate → fit (stretch / crop / pad / proportional) → flip.
       Углы поворота и поля pad заливаются выбранным pad_image_color,
       прозрачность показывается шахматной подложкой, якорь crop_position
       и интерполяция учитываются.
    5) Авто-скрытие виджетов по контексту: resize_mode решает, какие
       размерные виджеты нужны; fit_mode — нужны ли якорь и цвета полей;
       цвета полей видны и при повороте; angle_degrees скрывается,
       когда выбран rotation_preset.
    6) Полное сворачивание: кнопка ▴/▾ прячет параметры, но оставляет
       custom_path, image и кнопку загрузки. Состояние хранится в
       node.properties.agsoft_collapsed и сохраняется в воркфлоу.
    7) Статус-бар в самом низу: реальные МП и Ш×В источника и пересчитанные
       МП и Ш×В результата в реальном времени; после выполнения —
       фактические размеры выхода (метка ✓).
    Виджеты с подключённым линком не скрываются никогда; входы и линки не
    пересобираются — слоты фиксированные.
    ---
    Client part of the AGSoft_Load_Image_Mask_Resize node.
    1) The DOM widget draws the preview, but node.imgs stays real: the mask
       editor and the node menu items depend on it. The built-in picture
       display is disabled via hideOutputImages and by hiding the core
       preview widget, so there is no duplicate and "Open in MaskEditor"
       keeps working.
    2) Right-clicking the preview or the status bar opens the standard
       ComfyUI node menu (MaskEditor, Open/Copy/Save Image, properties...).
       The MaskEditor result (clipspace) immediately becomes the source.
    3) The preview fits the current node size and does NOT change it. The
       height minimum (computeSize) is size-independent, so the node can be
       stretched and shrunk vertically in both directions. Hiding/showing
       widgets preserves the preview area.
    4) The preview is composed on a canvas with the same pipeline as the
       server: rotate → fit (stretch / crop / pad / proportional) → flip.
       Rotation corners and pad bars use pad_image_color, transparency is
       shown over a checkerboard, crop_position and interpolation are kept.
    5) Context-aware widget auto-hiding: resize_mode decides which size
       widgets are needed; fit_mode decides the anchor and pad colors; pad
       colors stay visible while rotating; angle_degrees is hidden when
       rotation_preset is set.
    6) Full collapse: the ▴/▾ button hides the parameters but keeps
       custom_path, image and the upload button. The state is kept in
       node.properties.agsoft_collapsed and saved with the workflow.
    7) Status bar at the very bottom: real source MP and W×H plus recomputed
       output MP and W×H in real time; after execution the actual output
       sizes are shown (✓ mark).
    Widgets with a connected link are never hidden; inputs and links are not
    rebuilt — the slots are fixed.
Возможности / Features:
    ⚡ Правый клик = штатное меню ноды + MaskEditor. / Right-click = the
       standard node menu + MaskEditor.
    ⚡ Одно превью = реальный пайплайн с pad_image_color. / Single preview =
       the real pipeline with pad_image_color.
    ⚡ Свободный ресайз ноды по вертикали и горизонтали. / Free node resize.
    ⚡ Сворачивание параметров с сохранением выбора источника.
       Parameter collapse while keeping the source controls.
    ⚡ Живой статус-бар: МП, ширина, высота. / Live status bar: MP, W, H.
    ⚡ Drag&drop изображения и кнопка загрузки. / Image drag&drop and upload.
Автор / Author: AGSoft
Дата / Date: 14.11.2026
==============================================================================
*/

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Image & Mask Resize] JS extension loaded v11.14 (node context menu + MaskEditor support)");

const NODE_CLASS = "AGSoft_Load_Image_Mask_Resize";
const STYLE_ID = "agsoft-load-image-mask-resize-style";
const CORE_PREVIEW_WIDGET = "$$canvas-image-preview";
const MIN_NODE_WIDTH = 340;
const STATUS_HEIGHT = 26;
const PREVIEW_BOTTOM_GAP = 4;
const MIN_PREVIEW_AREA = 0;
const PREVIEW_DEFAULT_AREA = 240;
const PREVIEW_PADDING = 8;
const PREVIEW_WORK_MAX = 1280;
const PATH_DEBOUNCE_MS = 450;
const DROP_GUARD_MS = 600;

const SIZE_WIDGETS = ["target_megapixels", "target_percentage", "target_width", "target_height"];
const MODE_WIDGETS = {
    target_megapixels: ["target_megapixels"],
    target_percentage: ["target_percentage"],
    target_width: ["target_width"],
    target_height: ["target_height"],
    target_both: ["target_width", "target_height"],
    target_longest: ["target_width"],
    target_shortest: ["target_width"],
};
// Управление источником остаётся доступным в свёрнутом виде.
// Source controls stay available while collapsed.
const COLLAPSE_KEEP = ["custom_path", "image"];
const NEAREST_METHODS = ["nearest", "nearest-exact"];

const IMAGE_COLORS_HEX = {
    black: "#000000", white: "#FFFFFF", gray: "#808080", silver: "#C0C0C0",
    light_gray: "#D3D3D3", dark_gray: "#A9A9A9", red: "#FF0000", green: "#00FF00",
    blue: "#0000FF", yellow: "#FFFF00", cyan: "#00FFFF", magenta: "#FF00FF",
    orange: "#FFA500", pink: "#FFC0CB", brown: "#A52A2A", purple: "#800080",
    violet: "#EE82EE", indigo: "#4B0082", teal: "#008080", navy: "#000080",
    olive: "#808000", maroon: "#800000", dark_blue: "#00008B", light_blue: "#ADD8E6",
    light_green: "#90EE90", dark_green: "#006400", transparent: "transparent",
};

const CROP_POSITIONS = {
    center: [0.5, 0.5],
    "top-left": [0.0, 0.0],
    top: [0.5, 0.0],
    "top-right": [1.0, 0.0],
    left: [0.0, 0.5],
    right: [1.0, 0.5],
    "bottom-left": [0.0, 1.0],
    bottom: [0.5, 1.0],
    "bottom-right": [1.0, 1.0],
};


// =============================================================================
// СТИЛИ / STYLES
// =============================================================================
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.agsoft-resize-preview {
    position: relative;
    width: 100%;
    box-sizing: border-box;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    background-color: #101014;
    background-color: color-mix(in srgb, var(--ags-bg, #101014) 82%, #000000);
    background-image:
        linear-gradient(45deg, rgba(255, 255, 255, 0.06) 25%, transparent 25%,
            transparent 75%, rgba(255, 255, 255, 0.06) 75%),
        linear-gradient(45deg, rgba(255, 255, 255, 0.06) 25%, transparent 25%,
            transparent 75%, rgba(255, 255, 255, 0.06) 75%);
    background-size: 16px 16px;
    background-position: 0 0, 8px 8px;
}
.agsoft-resize-canvas {
    position: absolute;
    left: 50%;
    top: 50%;
    transform-origin: 50% 50%;
    display: none;
    pointer-events: none;
    background: transparent;
}
.agsoft-resize-hint {
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    font: 11px/1.4 system-ui, sans-serif;
    color: #9aa4b2;
    text-align: center;
    padding: 0 10px;
    pointer-events: none;
}
.agsoft-resize-preview.agsoft-drop {
    border: 1px dashed var(--ags-accent, #7fd4ff);
}
.agsoft-resize-status {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    height: ${STATUS_HEIGHT}px;
    box-sizing: border-box;
    padding: 0 6px;
    border-radius: 6px;
    background: #16161c;
    background: color-mix(in srgb, var(--ags-bg, #16161c) 88%, #000000);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.agsoft-resize-status-text {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    font: 11px/1.2 ui-monospace, Consolas, monospace;
    color: var(--ags-text, #cfd6e4);
}
.agsoft-resize-btn {
    flex: 0 0 auto;
    width: 22px;
    height: 20px;
    padding: 0;
    cursor: pointer;
    font: 12px/1 system-ui, sans-serif;
    color: var(--ags-text, #cfd6e4);
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.06);
}
.agsoft-resize-btn:hover {
    border-color: var(--ags-accent, #7fd4ff);
    color: var(--ags-accent, #7fd4ff);
}
`;
    document.head.appendChild(style);
}


// =============================================================================
// УТИЛИТЫ / UTILS
// =============================================================================
function num(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function findWidget(node, name) {
    for (const widget of node.widgets || []) {
        if (widget && widget.name === name) return widget;
    }
    return null;
}

function widgetValue(node, name) {
    const widget = findWidget(node, name);
    return widget ? widget.value : undefined;
}

function readValues(node) {
    const values = {};
    for (const widget of node.widgets || []) {
        if (widget && widget.name) values[widget.name] = widget.value;
    }
    return values;
}

function isLinked(node, widget) {
    for (const input of node.inputs || []) {
        if (input.name === widget.name && input.link !== null && input.link !== undefined) {
            return true;
        }
    }
    return false;
}

function hasTensorLink(node) {
    for (const input of node.inputs || []) {
        if (input.name === "input_image" && input.link !== null && input.link !== undefined) {
            return true;
        }
    }
    return false;
}

function isUploadButton(widget) {
    if (!widget || widget.type !== "button") return false;
    const label = `${widget.name || ""} ${widget.label || ""}`.toLowerCase();
    return label.includes("upload");
}

function viewUrl(name, stamp) {
    let type = "input";
    let value = String(name || "");
    const annotated = value.match(/^(.*)\s\[(input|output|temp)\]$/);
    if (annotated) {
        value = annotated[1];
        type = annotated[2];
    }
    const index = value.lastIndexOf("/");
    const subfolder = index >= 0 ? value.slice(0, index) : "";
    const filename = index >= 0 ? value.slice(index + 1) : value;
    const query = `filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder)}`;
    return api.apiURL(`/view?${query}&type=${type}&rand=${stamp}`);
}

function megapixels(size) {
    return ((size.w * size.h) / 1000000).toFixed(2);
}

function openNodeMenu(node, event) {
    // DOM-виджет перехватывает мышь, поэтому меню ноды открываем сами:
    // так остаются MaskEditor, Open/Copy/Save Image и свойства ноды.
    // The DOM widget captures the mouse, so the node menu is opened here:
    // MaskEditor, Open/Copy/Save Image and node properties stay available.
    const canvas = app.canvas;
    if (!canvas) return;
    if (typeof canvas.selectNode === "function") {
        canvas.selectNode(node);
    }
    if (typeof canvas.convertEventToCanvasOffset === "function") {
        const offset = canvas.convertEventToCanvasOffset(event);
        event.canvasX = offset[0];
        event.canvasY = offset[1];
    }
    if (typeof canvas.processContextMenu === "function") {
        canvas.processContextMenu(node, event);
        return;
    }
    if (canvas.canvas) {
        canvas.canvas.dispatchEvent(new MouseEvent("contextmenu", event));
    }
}

function attachNodeMenu(node, element) {
    element.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openNodeMenu(node, event);
    });
}

function suppressCorePreview(node) {
    // node.imgs не трогаем: MaskEditor и меню ноды читают именно его.
    // Гасим только штатную отрисовку, чтобы картинка не дублировалась.
    // node.imgs is left intact: MaskEditor and the node menu read it.
    // Only the built-in drawing is disabled so the picture is not duplicated.
    node.hideOutputImages = true;
    let changed = false;
    for (const widget of node.widgets || []) {
        if (!widget || widget.name !== CORE_PREVIEW_WIDGET) continue;
        if (widget._agsoftCoreHidden) continue;
        widget._agsoftCoreHidden = true;
        widget.hidden = true;
        widget.y = -9999;
        widget.computeSize = () => [0, -4];
        widget.drawWidget = () => {};
        changed = true;
    }
    if (changed) {
        syncPreviewBox(node);
        layoutPreviewCanvas(node);
        node.setDirtyCanvas(true, true);
    }
}

function adoptClipspace(node, value) {
    // MaskEditor пишет результат в комбо image, поэтому custom_path должен
    // уступить: иначе нарисованная маска не станет источником.
    // MaskEditor writes its result into the image combo, so custom_path must
    // step aside; otherwise the painted mask never becomes the source.
    if (!String(value || "").includes("clipspace")) return;
    const pathWidget = findWidget(node, "custom_path");
    if (pathWidget) pathWidget.value = "";
}

function watchSourceChange(node) {
    // Значение комбо может измениться без колбэка (сохранение из MaskEditor).
    // The combo value can change without a callback (saving from MaskEditor).
    const state = node._agsoft;
    const value = String(widgetValue(node, "image") || "");
    if (value === state.comboWatch) return;
    state.comboWatch = value;
    adoptClipspace(node, value);
    showComboPreview(node);
}


// =============================================================================
// ГЕОМЕТРИЯ: точное повторение серверного пайплайна
// GEOMETRY: exact mirror of the server pipeline
// =============================================================================
function resolveAngle(values) {
    const preset = String(values.rotation_preset || "none");
    if (preset !== "none") return num(preset, 0);
    return num(values.angle_degrees, 0);
}

function rotationFactors(angle) {
    const radians = ((num(angle, 0) % 360) * Math.PI) / 180;
    const rawCos = Math.abs(Math.cos(radians));
    const rawSin = Math.abs(Math.sin(radians));
    return {
        cos: rawCos < 1e-9 ? 0 : rawCos,
        sin: rawSin < 1e-9 ? 0 : rawSin,
    };
}

function rotateDims(width, height, angle) {
    const factors = rotationFactors(angle);
    if (!factors.sin && !factors.cos) return { w: width, h: height };
    return {
        w: Math.max(1, Math.ceil(width * factors.cos + height * factors.sin - 1e-9)),
        h: Math.max(1, Math.ceil(width * factors.sin + height * factors.cos - 1e-9)),
    };
}

function conditionalDims(origW, origH, targetW, targetH, larger, smaller) {
    const big = String(larger || "none");
    const small = String(smaller || "none");
    if (big !== "none") {
        const trig =
            (big === "width" && origW > targetW) ||
            (big === "height" && origH > targetH) ||
            (big === "both" && (origW > targetW || origH > targetH));
        return trig ? { w: targetW, h: targetH } : { w: origW, h: origH };
    }
    if (small !== "none") {
        const trig =
            (small === "width" && origW < targetW) ||
            (small === "height" && origH < targetH) ||
            (small === "both" && (origW < targetW || origH < targetH));
        return trig ? { w: targetW, h: targetH } : { w: origW, h: origH };
    }
    return { w: targetW, h: targetH };
}

function computeFrameTarget(srcW, srcH, values) {
    const mode = String(values.resize_mode || "target_percentage");
    const wantW = Math.max(1, Math.floor(num(values.target_width, 1024)));
    const wantH = Math.max(1, Math.floor(num(values.target_height, 1024)));
    let baseW = 1;
    let baseH = 1;

    if (mode === "target_megapixels") {
        const pixels = Math.max(1, Math.floor(num(values.target_megapixels, 1.0) * 1000000));
        const ratio = Math.sqrt(pixels / Math.max(1, srcW * srcH));
        baseW = Math.max(1, Math.floor(srcW * ratio));
        baseH = Math.max(1, Math.floor(srcH * ratio));
    } else if (mode === "target_percentage") {
        const ratio = Math.abs(num(values.target_percentage, 100)) / 100;
        baseW = Math.max(1, Math.floor(srcW * ratio));
        baseH = Math.max(1, Math.floor(srcH * ratio));
    } else if (mode === "target_width") {
        baseW = wantW;
        baseH = Math.max(1, Math.floor(srcH * (baseW / Math.max(1, srcW))));
    } else if (mode === "target_height") {
        baseH = wantH;
        baseW = Math.max(1, Math.floor(srcW * (baseH / Math.max(1, srcH))));
    } else if (mode === "target_both") {
        baseW = wantW;
        baseH = wantH;
    } else if (mode === "target_longest" || mode === "target_shortest") {
        const side = mode === "target_longest" ? Math.max(srcW, srcH) : Math.min(srcW, srcH);
        const ratio = side > 0 ? wantW / side : 1;
        baseW = Math.max(1, Math.floor(srcW * ratio));
        baseH = Math.max(1, Math.floor(srcH * ratio));
    }

    const cond = conditionalDims(
        srcW, srcH, baseW, baseH,
        values.resize_if_larger, values.resize_if_smaller,
    );
    let targetW = cond.w;
    let targetH = cond.h;
    const div = Math.max(0, Math.floor(num(values.divisible_by, 0)));
    if (div > 0) {
        targetW = Math.max(div, Math.floor(targetW / div) * div);
        targetH = Math.max(div, Math.floor(targetH / div) * div);
    }
    return { w: targetW, h: targetH };
}

function predictOutput(srcW, srcH, values) {
    const rotated = rotateDims(srcW, srcH, resolveAngle(values));
    const target = computeFrameTarget(srcW, srcH, values);
    if (String(values.fit_mode || "crop") === "proportional") {
        const scale = Math.min(target.w / rotated.w, target.h / rotated.h);
        return {
            w: Math.max(1, Math.floor(rotated.w * scale)),
            h: Math.max(1, Math.floor(rotated.h * scale)),
        };
    }
    return target;
}


// =============================================================================
// РАЗМЕРЫ: минимум не зависит от текущего размера, превью берёт остаток
// SIZES: the minimum is size-independent, the preview takes the remainder
// =============================================================================
function baseSize(node) {
    const state = node._agsoft;
    const widget = state.previewWidget;
    const original = widget ? widget.computeSize : null;
    if (widget) widget.computeSize = () => [0, 0];
    let size;
    try {
        size = state.baseComputeSize ? state.baseComputeSize.call(node) : [MIN_NODE_WIDTH, 120];
    } finally {
        if (widget) widget.computeSize = original;
    }
    return [size[0] || MIN_NODE_WIDTH, size[1] || 0];
}

function previewAreaHeight(node) {
    const base = baseSize(node);
    return Math.max(MIN_PREVIEW_AREA, node.size[1] - base[1] - PREVIEW_BOTTOM_GAP);
}

function syncPreviewBox(node) {
    const state = node._agsoft;
    if (!state || !state.previewEl) return;
    state.previewEl.style.height = `${previewAreaHeight(node)}px`;
}

function layoutPreviewCanvas(node) {
    const state = node._agsoft;
    if (!state || !state.canvasEl || !state.frame) return;
    const boxW = Math.max((state.previewEl.clientWidth || node.size[0] - 24) - PREVIEW_PADDING, 1);
    const boxH = Math.max((state.previewEl.clientHeight || previewAreaHeight(node)) - PREVIEW_PADDING, 1);
    const scale = Math.min(boxW / state.frame.w, boxH / state.frame.h);
    const flip = String(widgetValue(node, "flip_mode") || "none");
    const fx = flip === "horizontal" ? -1 : 1;
    const fy = flip === "vertical" ? -1 : 1;
    state.canvasEl.style.width = `${Math.max(1, state.frame.w * scale)}px`;
    state.canvasEl.style.height = `${Math.max(1, state.frame.h * scale)}px`;
    // Отражение — последний шаг пайплайна, поэтому применяется к кадру целиком.
    // Flip is the last pipeline step, so it is applied to the whole frame.
    state.canvasEl.style.transform = `translate(-50%, -50%) scale(${fx}, ${fy})`;
}

function refreshNodeSize(node, area) {
    const base = baseSize(node);
    const width = Math.max(node.size[0], base[0], MIN_NODE_WIDTH);
    const wanted = num(area, previewAreaHeight(node));
    const previewArea = Math.max(MIN_PREVIEW_AREA, wanted);
    const height = base[1] + previewArea + PREVIEW_BOTTOM_GAP;
    if (node.size[0] !== width || Math.abs(node.size[1] - height) > 0.5) {
        node.setSize([width, height]);
    }
    syncPreviewBox(node);
    layoutPreviewCanvas(node);
    node.setDirtyCanvas(true, true);
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

function renderPreview(node) {
    const state = node._agsoft;
    const img = state.sourceImg;
    const canvas = state.canvasEl;
    if (!img || !state.src || !state.src.w || !state.src.h) {
        canvas.style.display = "none";
        return;
    }
    const values = readValues(node);
    const srcW = state.src.w;
    const srcH = state.src.h;
    const angle = resolveAngle(values);
    const factors = rotationFactors(angle);
    const rotated = ((angle % 360) + 360) % 360 !== 0;
    const fit = String(values.fit_mode || "crop");
    const padName = String(values.pad_image_color || "white");
    const padHex = IMAGE_COLORS_HEX[padName] || "#FFFFFF";
    const anchor = CROP_POSITIONS[String(values.crop_position || "center")] || CROP_POSITIONS.center;
    const smooth = !NEAREST_METHODS.includes(String(values.interpolation || "lanczos"));

    const rotW = Math.max(1, Math.round(srcW * factors.cos + srcH * factors.sin));
    const rotH = Math.max(1, Math.round(srcW * factors.sin + srcH * factors.cos));
    const target = computeFrameTarget(srcW, srcH, values);
    let frameW = target.w;
    let frameH = target.h;
    if (fit === "proportional") {
        const scale = Math.min(target.w / rotW, target.h / rotH);
        frameW = Math.max(1, Math.floor(rotW * scale));
        frameH = Math.max(1, Math.floor(rotH * scale));
    }

    // Рабочее разрешение ограничено: превью не должно тратить память полного выхода.
    // Working resolution is capped: the preview must not allocate the full output.
    const work = Math.min(1, PREVIEW_WORK_MAX / Math.max(rotW, rotH, frameW, frameH));
    const stageW = Math.max(1, Math.round(rotW * work));
    const stageH = Math.max(1, Math.round(rotH * work));
    const outW = Math.max(1, Math.round(frameW * work));
    const outH = Math.max(1, Math.round(frameH * work));

    const stage = document.createElement("canvas");
    stage.width = stageW;
    stage.height = stageH;
    const stageCtx = stage.getContext("2d");
    stageCtx.imageSmoothingEnabled = smooth;
    stageCtx.imageSmoothingQuality = "high";
    if (rotated && padHex !== "transparent") {
        stageCtx.fillStyle = padHex;
        stageCtx.fillRect(0, 0, stageW, stageH);
    }
    stageCtx.translate(stageW / 2, stageH / 2);
    stageCtx.rotate((-angle * Math.PI) / 180);
    stageCtx.drawImage(img, (-srcW * work) / 2, (-srcH * work) / 2, srcW * work, srcH * work);

    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, outW, outH);

    if (fit === "stretch") {
        ctx.drawImage(stage, 0, 0, outW, outH);
    } else if (fit === "crop") {
        const scale = Math.max(outW / stageW, outH / stageH);
        const drawW = stageW * scale;
        const drawH = stageH * scale;
        ctx.drawImage(stage, -(drawW - outW) * anchor[0], -(drawH - outH) * anchor[1], drawW, drawH);
    } else {
        if (fit === "pad" && padHex !== "transparent") {
            ctx.fillStyle = padHex;
            ctx.fillRect(0, 0, outW, outH);
        }
        const scale = Math.min(outW / stageW, outH / stageH);
        const drawW = Math.max(1, stageW * scale);
        const drawH = Math.max(1, stageH * scale);
        ctx.drawImage(stage, (outW - drawW) * anchor[0], (outH - drawH) * anchor[1], drawW, drawH);
    }

    state.frame = { w: outW, h: outH };
    canvas.style.display = "block";
    state.hintEl.style.display = "none";
    layoutPreviewCanvas(node);
}


// =============================================================================
// ВИДИМОСТЬ ВИДЖЕТОВ / WIDGET VISIBILITY
// =============================================================================
function wantedVisibility(node) {
    const values = readValues(node);
    const visible = new Map();
    const allowed = MODE_WIDGETS[String(values.resize_mode || "target_percentage")] || SIZE_WIDGETS;
    for (const name of SIZE_WIDGETS) {
        visible.set(name, allowed.includes(name));
    }
    const fit = String(values.fit_mode || "crop");
    const usesPad = fit === "pad";
    const usesAnchor = fit === "pad" || fit === "crop";
    const preset = String(values.rotation_preset || "none");
    // Цвет полей нужен и при повороте: им заливаются пустые углы.
    // Pad colors are also used by rotation: they fill the empty corners.
    const rotated = preset !== "none" || num(values.angle_degrees, 0) !== 0;
    visible.set("crop_position", usesAnchor);
    visible.set("pad_image_color", usesPad || rotated);
    visible.set("pad_mask_color", usesPad || rotated);
    visible.set("angle_degrees", preset === "none");
    return visible;
}

function setWidgetVisible(widget, visible) {
    if (widget._agsoftVisible === visible) return;
    widget._agsoftVisible = visible;
    if (visible) {
        widget.hidden = false;
        widget.y = 0;
        if (widget._agsoftComputeSize) {
            widget.computeSize = widget._agsoftComputeSize;
        } else {
            delete widget.computeSize;
        }
        return;
    }
    if (widget._agsoftComputeSize === undefined) {
        widget._agsoftComputeSize = widget.computeSize || null;
    }
    widget.hidden = true;
    widget.y = -9999;
    widget.computeSize = () => [0, -4];
}

function applyVisibility(node) {
    const state = node._agsoft;
    if (!state) return;
    // Область превью фиксируется ДО смены видимости, чтобы кадр не прыгал.
    // The preview area is captured BEFORE visibility changes so it never jumps.
    const areaBefore = previewAreaHeight(node);
    const collapsed = node.properties.agsoft_collapsed === true;
    const wanted = wantedVisibility(node);
    const shown = [];
    for (const widget of node.widgets || []) {
        if (widget._agsoftAlwaysVisible || widget._agsoftCoreHidden) continue;
        let visible = !collapsed;
        if (collapsed && (COLLAPSE_KEEP.includes(widget.name) || isUploadButton(widget))) {
            visible = true;
        }
        if (visible && wanted.has(widget.name)) {
            visible = wanted.get(widget.name) === true;
        }
        if (isLinked(node, widget)) visible = true;
        setWidgetVisible(widget, visible);
        if (visible) shown.push(widget.name);
    }
    const signature = `${collapsed ? 1 : 0}|${shown.join(",")}`;
    if (signature !== state.signature) {
        state.signature = signature;
        if (!state.restoring) refreshNodeSize(node, areaBefore);
    }
    syncToggle(node);
    updateStatus(node);
    syncPreviewBox(node);
    renderPreview(node);
}


// =============================================================================
// ПРЕВЬЮ / PREVIEW
// =============================================================================
function showHint(node, text) {
    const state = node._agsoft;
    state.pendingUrl = "";
    state.canvasEl.style.display = "none";
    state.hintEl.style.display = "block";
    state.hintEl.textContent = text;
}

function setPreview(node, name, size) {
    const state = node._agsoft;
    if (!name) {
        showHint(node, "no image / нет изображения");
        return;
    }
    const url = viewUrl(name, Date.now());
    state.pendingUrl = url;
    const loader = new Image();
    loader.onload = () => {
        if (state.pendingUrl !== url) return;
        state.sourceImg = loader;
        // Исходник остаётся доступен ядру: MaskEditor берёт картинку отсюда.
        // The source stays available to the core: MaskEditor reads it here.
        node.imgs = [loader];
        node.imageIndex = 0;
        if (size && size.w > 0 && size.h > 0) {
            state.src = { w: size.w, h: size.h };
        } else {
            state.src = { w: loader.naturalWidth, h: loader.naturalHeight };
        }
        // Размер ноды не трогаем: кадр вписывается в отведённое место.
        // The node size stays untouched: the frame fits the allotted space.
        syncPreviewBox(node);
        renderPreview(node);
        updateStatus(node);
        node.setDirtyCanvas(true, true);
    };
    loader.onerror = () => {
        if (state.pendingUrl !== url) return;
        showHint(node, "preview unavailable / превью недоступно");
    };
    loader.src = url;
}

function showComboPreview(node) {
    if (hasTensorLink(node)) {
        showHint(node, "source: input_image / источник: input_image");
        return;
    }
    const custom = String(widgetValue(node, "custom_path") || "").trim();
    if (custom) {
        schedulePathPreview(node);
        return;
    }
    const name = String(widgetValue(node, "image") || "").trim();
    if (!name) {
        node._agsoft.src = null;
        node._agsoft.sourceImg = null;
        node.imgs = [];
        showHint(node, "no image / нет изображения");
        updateStatus(node);
        return;
    }
    setPreview(node, name, null);
}

function schedulePathPreview(node) {
    const state = node._agsoft;
    if (state.pathTimer) clearTimeout(state.pathTimer);
    state.pathTimer = setTimeout(() => {
        state.pathTimer = null;
        requestPathPreview(node);
    }, PATH_DEBOUNCE_MS);
}

async function requestPathPreview(node) {
    const state = node._agsoft;
    const value = String(widgetValue(node, "custom_path") || "").trim();
    if (!value) {
        showComboPreview(node);
        return;
    }
    if (hasTensorLink(node)) {
        showHint(node, "source: input_image / источник: input_image");
        return;
    }
    try {
        const response = await api.fetchApi("/agsoft/image_mask_resize_ensure_preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ custom_path: value }),
        });
        if (response.status !== 200) {
            showHint(node, "not an image or not found / не изображение или не найдено");
            return;
        }
        const data = await response.json();
        state.dirty = true;
        setPreview(node, data.image, { w: num(data.width, 0), h: num(data.height, 0) });
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] preview request failed:", error);
        showHint(node, "preview request failed / запрос превью не удался");
    }
}

async function refreshState(node) {
    const state = node._agsoft;
    if (!state) return;
    try {
        const url = `/agsoft/image_mask_resize_preview_state?node_id=${encodeURIComponent(String(node.id))}`;
        const response = await api.fetchApi(url);
        if (response.status !== 200) return;
        const data = await response.json();
        if (!data || !data.stamp || data.stamp === state.stamp) return;
        state.stamp = data.stamp;
        const srcW = num(data.src_w, 0);
        const srcH = num(data.src_h, 0);
        const outW = num(data.out_w, 0);
        const outH = num(data.out_h, 0);
        if (srcW > 0 && srcH > 0) state.src = { w: srcW, h: srcH };
        state.out = outW > 0 && outH > 0 ? { w: outW, h: outH } : null;
        state.dirty = false;
        if (data.image) {
            setPreview(node, data.image, state.src);
        } else {
            showComboPreview(node);
        }
        updateStatus(node);
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] state request failed:", error);
    }
}


// =============================================================================
// ЗАГРУЗКА ФАЙЛА / FILE UPLOAD
// =============================================================================
async function uploadFile(node, file) {
    const state = node._agsoft;
    if (!file) return;
    if (!String(file.type || "").startsWith("image/")) {
        showHint(node, "image files only / только изображения");
        return;
    }
    showHint(node, "uploading… / загрузка…");
    try {
        const body = new FormData();
        body.append("image", file);
        body.append("overwrite", "true");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (response.status !== 200) {
            throw new Error(`status ${response.status}`);
        }
        const data = await response.json();
        const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
        const combo = findWidget(node, "image");
        if (combo) {
            const options = combo.options || {};
            const values = Array.isArray(options.values) ? options.values : combo.values;
            if (Array.isArray(values) && !values.includes(name)) values.push(name);
            combo.value = name;
            state.comboWatch = String(name);
        }
        // Загруженный файл должен стать активным источником.
        // The uploaded file must become the active source.
        const pathWidget = findWidget(node, "custom_path");
        if (pathWidget) pathWidget.value = "";
        state.dirty = true;
        state.out = null;
        applyVisibility(node);
        showComboPreview(node);
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] upload failed:", error);
        showHint(node, "upload failed / загрузка не удалась");
    }
}

function createPreview(node) {
    const state = node._agsoft;
    const wrap = document.createElement("div");
    wrap.className = "agsoft-resize-preview";
    wrap.title = "Right-click: node menu (MaskEditor). Drop an image here / " +
        "Правый клик: меню ноды (MaskEditor). Перетащите изображение сюда";
    const canvas = document.createElement("canvas");
    canvas.className = "agsoft-resize-canvas";
    const hint = document.createElement("div");
    hint.className = "agsoft-resize-hint";
    hint.textContent = "no image / нет изображения";
    wrap.appendChild(canvas);
    wrap.appendChild(hint);

    wrap.addEventListener("pointerdown", (event) => event.stopPropagation());
    attachNodeMenu(node, wrap);
    wrap.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.stopPropagation();
        wrap.classList.add("agsoft-drop");
    });
    wrap.addEventListener("dragleave", (event) => {
        event.preventDefault();
        event.stopPropagation();
        wrap.classList.remove("agsoft-drop");
    });
    wrap.addEventListener("drop", (event) => {
        event.preventDefault();
        event.stopPropagation();
        wrap.classList.remove("agsoft-drop");
        const now = Date.now();
        if (now - state.lastDropAt < DROP_GUARD_MS) return;
        state.lastDropAt = now;
        const transfer = event.dataTransfer;
        const file = transfer && transfer.files && transfer.files.length ? transfer.files[0] : null;
        if (file) uploadFile(node, file);
    });

    const widget = node.addDOMWidget("agsoft_preview", "agsoft_preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => "",
        setValue: () => {},
    });
    widget.serialize = false;
    widget._agsoftAlwaysVisible = true;
    widget._agsoftVisible = true;
    widget.computeSize = (width) => [width, previewAreaHeight(node)];

    state.previewEl = wrap;
    state.canvasEl = canvas;
    state.hintEl = hint;
    state.previewWidget = widget;
}


// =============================================================================
// СТАТУС-БАР / STATUS BAR
// =============================================================================
function createStatusBar(node) {
    const state = node._agsoft;
    const bar = document.createElement("div");
    bar.className = "agsoft-resize-status";
    bar.title = "Source MP/W/H → output MP/W/H (live) / " +
        "МП/Ш/В источника → МП/Ш/В результата (в реальном времени)";

    const text = document.createElement("div");
    text.className = "agsoft-resize-status-text";
    text.textContent = "— × — · — MP";

    const uploadBtn = document.createElement("button");
    uploadBtn.className = "agsoft-resize-btn";
    uploadBtn.textContent = "⬆";
    uploadBtn.title = "Upload image / Загрузить изображение";

    const toggleBtn = document.createElement("button");
    toggleBtn.className = "agsoft-resize-btn";
    toggleBtn.textContent = "▴";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";

    bar.appendChild(text);
    bar.appendChild(uploadBtn);
    bar.appendChild(toggleBtn);
    bar.appendChild(fileInput);
    bar.addEventListener("pointerdown", (event) => event.stopPropagation());
    attachNodeMenu(node, bar);

    uploadBtn.onclick = () => fileInput.click();
    fileInput.onchange = () => {
        const file = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
        fileInput.value = "";
        if (file) uploadFile(node, file);
    };
    toggleBtn.onclick = () => {
        node.properties.agsoft_collapsed = !(node.properties.agsoft_collapsed === true);
        applyVisibility(node);
    };

    const widget = node.addDOMWidget("agsoft_status", "agsoft_status", bar, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => "",
        setValue: () => {},
    });
    widget.serialize = false;
    widget._agsoftAlwaysVisible = true;
    widget._agsoftVisible = true;
    widget.computeSize = (width) => [width, STATUS_HEIGHT];

    state.statusEl = bar;
    state.statusText = text;
    state.toggleBtn = toggleBtn;
    state.statusWidget = widget;
}

function syncToggle(node) {
    const state = node._agsoft;
    if (!state || !state.toggleBtn) return;
    const collapsed = node.properties.agsoft_collapsed === true;
    state.toggleBtn.textContent = collapsed ? "▾" : "▴";
    state.toggleBtn.title = collapsed
        ? "Show all widgets / Показать все виджеты"
        : "Hide parameters / Скрыть параметры";
}

function updateStatus(node) {
    const state = node._agsoft;
    if (!state || !state.statusText) return;
    const src = state.src;
    if (!src || !src.w || !src.h) {
        state.statusText.textContent = "— × — · — MP  →  — × — · — MP";
        return;
    }
    const values = readValues(node);
    const actual = !state.dirty && state.out ? state.out : null;
    const out = actual || predictOutput(src.w, src.h, values);
    const mark = actual ? " ✓" : "";
    state.statusText.textContent =
        `${src.w}×${src.h} · ${megapixels(src)} MP  →  ` +
        `${out.w}×${out.h} · ${megapixels(out)} MP${mark}`;
}


// =============================================================================
// ТЕМИЗАЦИЯ / THEMING
// =============================================================================
function applyTheme(node) {
    const state = node._agsoft;
    if (!state || !state.previewEl || !state.statusEl) return;
    const bg = node.bgcolor || "#1a1a1a";
    const accent = node.color || "#7fd4ff";
    for (const element of [state.previewEl, state.statusEl]) {
        element.style.setProperty("--ags-bg", bg);
        element.style.setProperty("--ags-accent", accent);
        element.style.setProperty("--ags-text", "#cfd6e4");
    }
}


// =============================================================================
// ХУКИ ВИДЖЕТОВ И НОДЫ / WIDGET AND NODE HOOKS
// =============================================================================
function onWidgetChanged(node, widget, value) {
    const state = node._agsoft;
    state.dirty = true;
    state.out = null;
    if (widget.name === "custom_path") schedulePathPreview(node);
    if (widget.name === "image") {
        state.comboWatch = String(value === undefined ? widget.value : value || "");
        adoptClipspace(node, state.comboWatch);
        showComboPreview(node);
    }
    applyVisibility(node);
}

function hookWidgets(node) {
    for (const widget of node.widgets || []) {
        if (!widget || widget._agsoftAlwaysVisible || widget._agsoftHooked) continue;
        widget._agsoftHooked = true;
        const previous = widget.callback;
        widget.callback = function(value) {
            if (previous) previous.apply(this, arguments);
            onWidgetChanged(node, widget, value);
        };
    }
}

function hookNode(node) {
    node.computeSize = function(out) {
        const base = baseSize(node);
        // Минимум не зависит от текущего размера: ноду можно сжимать обратно.
        // The minimum is size-independent: the node can be shrunk back.
        return [Math.max(base[0], MIN_NODE_WIDTH), base[1] + MIN_PREVIEW_AREA + PREVIEW_BOTTOM_GAP];
    };

    // Штатная отрисовка картинки ядром не нужна: кадр рисует наш виджет.
    // Хук вызывается каждый кадр, поэтому ловит и поздно созданный виджет.
    // The core picture drawing is not needed: our widget draws the frame.
    // The hook runs every frame, so a late-created widget is caught as well.
    node.onDrawBackground = function(ctx) {
        suppressCorePreview(node);
        watchSourceChange(node);
    };

    const baseOnResize = node.onResize;
    node.onResize = function(size) {
        if (baseOnResize) baseOnResize.apply(this, arguments);
        syncPreviewBox(node);
        layoutPreviewCanvas(node);
        updateStatus(node);
    };

    const baseOnConfigure = node.onConfigure;
    node.onConfigure = function(info) {
        if (baseOnConfigure) baseOnConfigure.apply(this, arguments);
        node.properties = node.properties || {};
        if (typeof node.properties.agsoft_collapsed !== "boolean") {
            node.properties.agsoft_collapsed = false;
        }
        const state = node._agsoft;
        state.signature = "";
        state.comboWatch = String(widgetValue(node, "image") || "");
        state.restoring = true;
        applyVisibility(node);
        state.restoring = false;
        // Сохранённый размер ноды важнее значения по умолчанию.
        // The saved node size wins over the default area.
        syncPreviewBox(node);
        layoutPreviewCanvas(node);
        showComboPreview(node);
        refreshState(node);
    };

    const baseOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function(type, index, connected, linkInfo, slot) {
        if (baseOnConnectionsChange) baseOnConnectionsChange.apply(this, arguments);
        node._agsoft.signature = "";
        applyVisibility(node);
        if (!hasTensorLink(node)) showComboPreview(node);
    };

    const baseOnDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        if (baseOnDrawForeground) baseOnDrawForeground.apply(this, arguments);
        const key = `${node.color || ""}|${node.bgcolor || ""}`;
        if (key !== node._agsoft.themeKey) {
            node._agsoft.themeKey = key;
            applyTheme(node);
        }
    };

    const baseOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (baseOnRemoved) baseOnRemoved.apply(this, arguments);
        const state = node._agsoft;
        if (state && state.pathTimer) clearTimeout(state.pathTimer);
    };
}


// =============================================================================
// РЕГИСТРАЦИЯ / REGISTRATION
// =============================================================================
function nodeFromEvent(detail) {
    const raw = detail && detail.node !== undefined ? Number(detail.node) : NaN;
    if (!Number.isFinite(raw) || !app.graph) return null;
    const node = app.graph.getNodeById(raw);
    if (node && node.comfyClass === NODE_CLASS && node._agsoft) return node;
    return null;
}

app.registerExtension({
    name: "AGSoft.LoadImageMaskResize",

    setup() {
        api.addEventListener("executed", (event) => {
            const node = nodeFromEvent(event.detail);
            if (node) refreshState(node);
        });
        api.addEventListener("execution_success", () => {
            const nodes = (app.graph && app.graph._nodes) || [];
            for (const node of nodes) {
                if (node.comfyClass === NODE_CLASS && node._agsoft) refreshState(node);
            }
        });
    },

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        if (typeof node.addDOMWidget !== "function") {
            console.warn("[AGSoft Load Image & Mask Resize] DOM widgets are unsupported here");
            return;
        }
        ensureStyles();
        node.hideOutputImages = true;
        node.properties = node.properties || {};
        if (typeof node.properties.agsoft_collapsed !== "boolean") {
            node.properties.agsoft_collapsed = false;
        }
        node._agsoft = {
            src: null,
            out: null,
            stamp: "",
            dirty: true,
            signature: "",
            themeKey: "",
            comboWatch: "",
            pathTimer: null,
            lastDropAt: 0,
            pendingUrl: "",
            restoring: false,
            baseComputeSize: node.computeSize,
            sourceImg: null,
            frame: null,
        };

        createPreview(node);
        createStatusBar(node);
        hookWidgets(node);
        hookNode(node);
        applyTheme(node);
        applyVisibility(node);
        refreshNodeSize(node, PREVIEW_DEFAULT_AREA);
        node._agsoft.comboWatch = String(widgetValue(node, "image") || "");
        showComboPreview(node);
        refreshState(node);
    },
});