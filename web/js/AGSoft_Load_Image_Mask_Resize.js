/*
==============================================================================
AGSoft_Load_Image_Mask_Resize.js
==============================================================================
Расширение: 🖼️AGSoft Load Image & Mask Resize
Описание / Description:
    Клиентская часть ноды AGSoft_Load_Image_Mask_Resize.
    1) Виджет image: клик по краям (зоны стрелок) листает файлы по кругу,
       клик по центру открывает окно эскизов. Значение и превью обновляются
       сразу, как при обычном выборе в комбо.
    2) Окно эскизов — виртуальная сетка: в DOM живут только видимые строки
       плюс запас. Внизу переключатель папок input / output; выбор из output
       копирует файл в input и делает его источником. Окно открывается на
       текущей картинке, стрелки двигают курсор, Enter выбирает, Esc закрывает.
       Картинки тянутся с серверного роута уменьшенных WebP-эскизов.
    3) Превью рисует DOM-виджет, node.imgs остаётся настоящим: MaskEditor и
       меню ноды работают. Правый клик по превью и статус-бару открывает
       стандартное меню ComfyUI.
    4) Состояние сервера применяется только при совпадении ключа источника.
    5) Drag&drop из проводника, с рабочего стола и из панели Assets: сначала
       ссылка (text/uri-list), затем файлы. Drop ловится и на всей ноде.
    6) Превью собирается на canvas тем же пайплайном, что и сервер:
       rotate → fit (stretch / crop / pad / proportional) → flip, с цветом
       полей pad_image_color и шахматной подложкой для прозрачности.
    7) Превью вписывается в размер ноды и не меняет его; нода свободно
       тянется и сжимается по вертикали и горизонтали.
    8) Авто-скрытие виджетов по режиму; сворачивание параметров кнопкой ▴
       оставляет custom_path, image, загрузку, превью и статус-бар.
    9) Статус-бар: реальные МП и Ш×В источника → пересчитанные МП и Ш×В
       результата; после выполнения — фактический размер выхода (✓).
    ---
    Client part of the AGSoft_Load_Image_Mask_Resize node.
    1) The image widget: clicking its edges (arrow zones) cycles files,
       clicking the middle opens the thumbnail picker. Value and preview
       update immediately, exactly like a normal combo selection.
    2) The picker is a virtual grid: only visible rows plus an overscan live
       in the DOM. A bottom switch selects input / output; picking from
       output copies the file into input and selects it. The window opens on
       the current picture, arrows move the cursor, Enter picks, Esc closes.
       Images come from a server route serving small WebP thumbnails.
    3) The DOM widget draws the preview while node.imgs stays real, so the
       mask editor and the node menu keep working. Right-click opens the
       standard ComfyUI node menu.
    4) Server state is applied only when the source key matches.
    5) Drag&drop from the file manager, the desktop and the Assets panel:
       the link (text/uri-list) first, then files. Drops hit the whole node.
    6) The preview is composed on a canvas with the server pipeline:
       rotate → fit (stretch / crop / pad / proportional) → flip, using
       pad_image_color and a checkerboard for transparency.
    7) The preview fits the node size and never changes it; the node resizes
       freely in both directions.
    8) Mode-aware widget hiding; the ▴ collapse keeps custom_path, image,
       upload, preview and the status bar.
    9) Status bar: real source MP/W×H → recomputed output MP/W×H; after
       execution the actual output size is shown (✓).
Автор / Author: AGSoft
Дата / Date: 16.11.2026
==============================================================================
*/

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Image & Mask Resize] JS extension loaded v11.18 (input/output picker + arrows + original filename characters)");

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
const PICKER_GUARD_MS = 400;
const CLICK_GUARD_MS = 200;
const SEARCH_DEBOUNCE_MS = 120;
const THUMB_GAP = 8;
const THUMB_OVERSCAN = 2;
const COMBO_ARROW_ZONE = 30;
const IMAGE_EXT_RE = /\.(png|jpe?g|jfif|webp|gif|bmp|tiff?|ico|avif)$/i;

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

let thumbDialog = null;


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
.agsoft-thumbs-overlay {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.62);
}
.agsoft-thumbs-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: min(1100px, 92vw);
    height: min(760px, 88vh);
    padding: 12px;
    box-sizing: border-box;
    border-radius: 12px;
    background: #16161c;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.55);
}
.agsoft-thumbs-head {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.agsoft-thumbs-foot {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.agsoft-thumbs-title {
    font: 600 13px system-ui, sans-serif;
    color: #e6ebf5;
}
.agsoft-thumbs-spacer {
    margin-right: auto;
}
.agsoft-thumbs-note {
    margin-left: auto;
    font: 11px system-ui, sans-serif;
    color: #9aa4b2;
}
.agsoft-thumbs-field {
    height: 28px;
    padding: 0 8px;
    box-sizing: border-box;
    border-radius: 6px;
    color: #dfe6f2;
    background: #0f0f14;
    border: 1px solid rgba(255, 255, 255, 0.14);
    font: 12px system-ui, sans-serif;
}
.agsoft-thumbs-tab {
    height: 28px;
    padding: 0 14px;
    cursor: pointer;
    border-radius: 6px;
    font: 12px system-ui, sans-serif;
    color: #cfd6e4;
    background: #0f0f14;
    border: 1px solid rgba(255, 255, 255, 0.14);
}
.agsoft-thumbs-tab:hover {
    border-color: #7fd4ff;
}
.agsoft-thumbs-tab.agsoft-active {
    color: #0b0b0f;
    background: #7fd4ff;
    border-color: #7fd4ff;
}
.agsoft-thumbs-viewport {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    border-radius: 8px;
    background: #0f0f13;
    border: 1px solid rgba(255, 255, 255, 0.06);
    contain: strict;
}
.agsoft-thumbs-sizer {
    position: relative;
    width: 100%;
}
.agsoft-thumbs-card {
    position: absolute;
    box-sizing: border-box;
    overflow: hidden;
    cursor: pointer;
    border-radius: 8px;
    background: #15151b;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.agsoft-thumbs-card img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #0b0b0f;
}
.agsoft-thumbs-card:hover {
    border-color: #7fd4ff;
}
.agsoft-thumbs-card.agsoft-selected {
    border-color: #7fd4ff;
    box-shadow: inset 0 0 0 2px #7fd4ff;
}
.agsoft-thumbs-card.agsoft-cursor {
    border-color: #ffd479;
    box-shadow: inset 0 0 0 2px #ffd479;
}
.agsoft-thumbs-name {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 3px 5px;
    font: 10px/1.2 system-ui, sans-serif;
    color: #dfe6f2;
    background: linear-gradient(transparent, rgba(0, 0, 0, 0.80));
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    pointer-events: none;
}
.agsoft-thumbs-empty {
    position: absolute;
    left: 0;
    right: 0;
    top: 18px;
    text-align: center;
    font: 12px system-ui, sans-serif;
    color: #9aa4b2;
    pointer-events: none;
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

function sourceKey(node) {
    // Ключ источника совпадает с тем, что пишет сервер в состоянии.
    // The source key mirrors what the server stores in the state.
    if (hasTensorLink(node)) return "tensor";
    const custom = String(widgetValue(node, "custom_path") || "").trim();
    if (custom) return custom;
    return String(widgetValue(node, "image") || "");
}

function splitAnnotated(name) {
    let type = "input";
    let value = String(name || "");
    const annotated = value.match(/^(.*)\s\[(input|output|temp)\]$/);
    if (annotated) {
        value = annotated[1];
        type = annotated[2];
    }
    const index = value.lastIndexOf("/");
    return {
        filename: index >= 0 ? value.slice(index + 1) : value,
        subfolder: index >= 0 ? value.slice(0, index) : "",
        type,
    };
}

function viewUrl(name, stamp) {
    const ref = splitAnnotated(name);
    const query = `filename=${encodeURIComponent(ref.filename)}&subfolder=${encodeURIComponent(ref.subfolder)}`;
    return api.apiURL(`/view?${query}&type=${ref.type}&rand=${stamp}`);
}

function thumbUrl(item, px, kind) {
    const query = `filename=${encodeURIComponent(item.name)}&subfolder=${encodeURIComponent(item.subfolder || "")}`;
    return api.apiURL(`/agsoft/image_mask_resize_thumb?${query}&type=${kind}&size=${px}&v=${item.mtime}`);
}

function fullName(item) {
    return item.subfolder ? `${item.subfolder}/${item.name}` : item.name;
}

function megapixels(size) {
    return ((size.w * size.h) / 1000000).toFixed(2);
}

function openNodeMenu(node, event) {
    // DOM-виджет перехватывает мышь, поэтому меню ноды открываем сами.
    // The DOM widget captures the mouse, so the node menu is opened here.
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
    // node.imgs stays intact: MaskEditor and the node menu read it.
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
    // MaskEditor пишет результат в комбо image: custom_path должен уступить.
    // MaskEditor writes into the image combo: custom_path must step aside.
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
// РАЗМЕРЫ / SIZES
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

    // Рабочее разрешение ограничено: превью не должно тратить память выхода.
    // Working resolution is capped so the preview never allocates the output.
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
        // Состояние прошлого запуска не должно перекрывать новый источник.
        // A previous run must never override a newly chosen source.
        if (String(data.source || "") !== sourceKey(node)) return;
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
// ИСТОЧНИК: выбор, загрузка, импорт / SOURCE: select, upload, import
// =============================================================================
function selectImage(node, name, size) {
    const state = node._agsoft;
    const combo = findWidget(node, "image");
    if (combo) {
        const options = combo.options || {};
        const values = Array.isArray(options.values) ? options.values : combo.values;
        if (Array.isArray(values) && !values.includes(name)) values.push(name);
        combo.value = name;
        state.comboWatch = String(name);
    }
    const pathWidget = findWidget(node, "custom_path");
    if (pathWidget) pathWidget.value = "";
    state.dirty = true;
    state.out = null;
    state.stamp = "";
    applyVisibility(node);
    setPreview(node, name, size || null);
}

function setCustomPath(node, path) {
    const widget = findWidget(node, "custom_path");
    if (!widget) return;
    widget.value = path;
    node._agsoft.dirty = true;
    node._agsoft.out = null;
    node._agsoft.stamp = "";
    applyVisibility(node);
    schedulePathPreview(node);
}

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
        selectImage(node, name, null);
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] upload failed:", error);
        showHint(node, "upload failed / загрузка не удалась");
    }
}

async function importServerFile(node, ref) {
    showHint(node, "importing… / импорт…");
    try {
        const response = await api.fetchApi("/agsoft/image_mask_resize_import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: ref.filename,
                subfolder: ref.subfolder || "",
                type: ref.type || "input",
            }),
        });
        if (response.status !== 200) {
            showHint(node, "import failed / импорт не удался");
            return;
        }
        const data = await response.json();
        selectImage(node, data.image, { w: num(data.width, 0), h: num(data.height, 0) });
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] import failed:", error);
        showHint(node, "import failed / импорт не удался");
    }
}

function parseViewRef(text) {
    // Панель Assets отдаёт ссылку /view, а не файл: разбираем параметры.
    // The Assets panel hands over a /view link, not a file: parse its params.
    let url = null;
    try {
        url = new URL(String(text), window.location.origin);
    } catch (error) {
        return null;
    }
    if (!/\/view\/?$/.test(url.pathname)) return null;
    const filename = url.searchParams.get("filename") || "";
    if (!filename) return null;
    return {
        filename,
        subfolder: url.searchParams.get("subfolder") || "",
        type: url.searchParams.get("type") || "input",
    };
}

function fileUrlToPath(text) {
    let value = decodeURIComponent(String(text).replace(/^file:\/\/+/i, ""));
    if (/^\/[a-zA-Z]:/.test(value)) value = value.slice(1);
    return value;
}

function looksLikePath(text) {
    const value = String(text || "").trim();
    if (!IMAGE_EXT_RE.test(value)) return false;
    return /^[a-zA-Z]:[\\/]/.test(value) || value.startsWith("/") || value.startsWith("\\\\");
}

async function applyLink(node, link) {
    const ref = parseViewRef(link);
    if (ref) {
        await importServerFile(node, ref);
        return;
    }
    if (/^file:/i.test(link)) {
        setCustomPath(node, fileUrlToPath(link));
        return;
    }
    if (looksLikePath(link)) {
        setCustomPath(node, link.trim());
        return;
    }
    if (/^https?:/i.test(link)) {
        try {
            const response = await fetch(link);
            if (!response.ok) throw new Error(`status ${response.status}`);
            const blob = await response.blob();
            const tail = link.split("?")[0].split("/").pop() || "image.png";
            await uploadFile(node, new File([blob], tail, { type: blob.type }));
        } catch (error) {
            console.warn("[AGSoft Load Image & Mask Resize] link fetch failed:", error);
            showHint(node, "link unavailable / ссылка недоступна");
        }
        return;
    }
    if (IMAGE_EXT_RE.test(link.trim())) {
        await importServerFile(node, { filename: link.trim(), subfolder: "", type: "input" });
        return;
    }
    showHint(node, "unsupported drop / неподдерживаемый объект");
}

function readTransfer(transfer) {
    // Данные читаются синхронно: после события dataTransfer уже пуст.
    // Data is read synchronously: dataTransfer is empty after the event.
    const data = { files: [], texts: [] };
    if (!transfer) return data;
    for (const file of Array.from(transfer.files || [])) {
        data.files.push(file);
    }
    for (const type of Array.from(transfer.types || [])) {
        if (type === "Files") continue;
        let value = "";
        try {
            value = transfer.getData(type);
        } catch (error) {
            value = "";
        }
        if (value) data.texts.push(value);
    }
    return data;
}

function pickLink(texts) {
    for (const block of texts) {
        for (const rawLine of String(block).split(/\r?\n/)) {
            const line = rawLine.trim();
            if (!line || line.startsWith("#")) continue;
            const html = line.match(/src\s*=\s*["']([^"']+)["']/i);
            const candidate = html ? html[1] : line.split("\t")[0];
            if (parseViewRef(candidate)) return candidate;
            if (/^(file|https?):/i.test(candidate)) return candidate;
            if (looksLikePath(candidate)) return candidate;
            if (IMAGE_EXT_RE.test(candidate)) return candidate;
        }
    }
    return "";
}

async function applyTransfer(node, data) {
    const state = node._agsoft;
    const now = Date.now();
    if (now - state.lastDropAt < DROP_GUARD_MS) return;
    state.lastDropAt = now;
    // Ссылка важнее файла: браузер подкладывает пережатую копию картинки.
    // The link wins over files: the browser synthesizes a re-encoded copy.
    const link = pickLink(data.texts);
    if (link) {
        await applyLink(node, link);
        return;
    }
    if (data.files.length) {
        await uploadFile(node, data.files[0]);
        return;
    }
    showHint(node, "nothing to load / нечего загружать");
}


// =============================================================================
// ОКНО ЭСКИЗОВ: виртуальная сетка / THUMBNAIL PICKER: virtual grid
// =============================================================================
function closeThumbnails() {
    if (!thumbDialog) return;
    const dialog = thumbDialog;
    thumbDialog = null;
    document.removeEventListener("keydown", dialog.onKey, true);
    if (dialog.observer) dialog.observer.disconnect();
    if (dialog.raf) cancelAnimationFrame(dialog.raf);
    if (dialog.searchTimer) clearTimeout(dialog.searchTimer);
    dialog.cards.clear();
    dialog.overlay.remove();
}

function createThumbCard(dialog, index) {
    const item = dialog.view[index];
    const full = fullName(item);
    const card = document.createElement("div");
    card.className = "agsoft-thumbs-card";
    if (full === dialog.current) card.classList.add("agsoft-selected");
    if (index === dialog.cursor) card.classList.add("agsoft-cursor");
    card.title = full;
    card.style.left = `${(index % dialog.cols) * dialog.rowH}px`;
    card.style.top = `${Math.floor(index / dialog.cols) * dialog.rowH}px`;
    card.style.width = `${dialog.cell}px`;
    card.style.height = `${dialog.cell}px`;

    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.alt = item.name;
    img.src = thumbUrl(item, dialog.px, dialog.kind);
    img.onerror = () => img.remove();

    const name = document.createElement("div");
    name.className = "agsoft-thumbs-name";
    name.textContent = item.name;

    card.appendChild(img);
    card.appendChild(name);
    card.onclick = () => pickThumbIndex(index);
    return card;
}

function renderThumbWindow() {
    const dialog = thumbDialog;
    if (!dialog || !dialog.view.length) return;
    const rows = Math.ceil(dialog.view.length / dialog.cols);
    const top = dialog.viewport.scrollTop;
    const height = dialog.viewport.clientHeight;
    const first = Math.max(0, Math.floor(top / dialog.rowH) - THUMB_OVERSCAN);
    const last = Math.min(rows, Math.ceil((top + height) / dialog.rowH) + THUMB_OVERSCAN);

    const needed = new Set();
    for (let row = first; row < last; row++) {
        for (let col = 0; col < dialog.cols; col++) {
            const index = row * dialog.cols + col;
            if (index >= dialog.view.length) break;
            needed.add(index);
        }
    }
    for (const [index, card] of dialog.cards) {
        if (needed.has(index)) continue;
        card.remove();
        dialog.cards.delete(index);
    }
    for (const index of needed) {
        if (dialog.cards.has(index)) continue;
        const card = createThumbCard(dialog, index);
        dialog.sizer.appendChild(card);
        dialog.cards.set(index, card);
    }
    for (const [index, card] of dialog.cards) {
        card.classList.toggle("agsoft-cursor", index === dialog.cursor);
    }
}

function layoutThumbnails() {
    const dialog = thumbDialog;
    if (!dialog) return;
    const wanted = num(dialog.size.value, 140);
    const width = Math.max(dialog.viewport.clientWidth, wanted);
    const cols = Math.max(1, Math.floor((width + THUMB_GAP) / (wanted + THUMB_GAP)));
    dialog.cols = cols;
    dialog.cell = Math.max(48, Math.floor((width - THUMB_GAP * (cols - 1)) / cols));
    dialog.rowH = dialog.cell + THUMB_GAP;
    // Три фиксированных размера эскизов дают максимум попаданий в кэш.
    // Three fixed thumbnail sizes maximize cache hits.
    dialog.px = dialog.cell <= 130 ? 128 : (dialog.cell <= 260 ? 256 : 384);

    const rows = Math.ceil(dialog.view.length / cols);
    dialog.sizer.style.height = `${Math.max(0, rows * dialog.rowH - THUMB_GAP)}px`;
    dialog.empty.style.display = dialog.view.length ? "none" : "block";
    for (const card of dialog.cards.values()) {
        card.remove();
    }
    dialog.cards.clear();
    renderThumbWindow();
}

function scrollThumbIntoView(index) {
    const dialog = thumbDialog;
    if (!dialog) return;
    const top = Math.floor(index / dialog.cols) * dialog.rowH;
    const viewTop = dialog.viewport.scrollTop;
    const viewBottom = viewTop + dialog.viewport.clientHeight;
    if (top < viewTop) {
        dialog.viewport.scrollTop = top;
    } else if (top + dialog.cell > viewBottom) {
        dialog.viewport.scrollTop = top + dialog.cell - dialog.viewport.clientHeight;
    }
    renderThumbWindow();
}

function centerThumbOn(index) {
    const dialog = thumbDialog;
    if (!dialog) return;
    const row = Math.floor(index / dialog.cols);
    const target = row * dialog.rowH - (dialog.viewport.clientHeight - dialog.cell) / 2;
    dialog.viewport.scrollTop = Math.max(0, target);
    renderThumbWindow();
}

function moveThumbCursor(delta) {
    const dialog = thumbDialog;
    if (!dialog || !dialog.view.length) return;
    const next = Math.min(dialog.view.length - 1, Math.max(0, dialog.cursor + delta));
    if (next === dialog.cursor) return;
    dialog.cursor = next;
    scrollThumbIntoView(next);
}

function pickThumbIndex(index) {
    const dialog = thumbDialog;
    if (!dialog) return;
    const item = dialog.view[index];
    if (!item) return;
    const node = dialog.node;
    const kind = dialog.kind;
    const full = fullName(item);
    closeThumbnails();
    if (kind === "output") {
        // Файл из output копируется в input: только так он станет источником.
        // An output file is copied into input: that is how it becomes a source.
        importServerFile(node, { filename: item.name, subfolder: item.subfolder, type: "output" });
        return;
    }
    selectImage(node, full, null);
}

function filterThumbnails() {
    const dialog = thumbDialog;
    if (!dialog) return;
    const term = dialog.search.value.trim().toLowerCase();
    const folder = dialog.folder.value;
    dialog.current = dialog.kind === "input" ? String(widgetValue(dialog.node, "image") || "") : "";
    dialog.view = dialog.files.filter((item) => {
        if (folder !== "*" && item.subfolder !== folder) return false;
        if (!term) return true;
        return fullName(item).toLowerCase().includes(term);
    });
    dialog.count.textContent = `${dialog.view.length} / ${dialog.files.length}`;
    const index = dialog.current
        ? dialog.view.findIndex((item) => fullName(item) === dialog.current)
        : -1;
    dialog.cursor = index >= 0 ? index : 0;
    dialog.viewport.scrollTop = 0;
    layoutThumbnails();
    // Окно должно открываться на выбранной картинке, а не в начале списка.
    // The window must open on the selected picture, not at the list start.
    if (index >= 0) centerThumbOn(index);
}

async function loadThumbnails(force) {
    const dialog = thumbDialog;
    if (!dialog) return;
    dialog.count.textContent = "…";
    try {
        const query = `type=${dialog.kind}&force=${force ? 1 : 0}`;
        const response = await api.fetchApi(`/agsoft/image_mask_resize_thumbs?${query}`);
        if (response.status !== 200) throw new Error(`status ${response.status}`);
        const data = await response.json();
        if (thumbDialog !== dialog) return;
        dialog.files = Array.isArray(data.files) ? data.files : [];
        const folders = Array.isArray(data.folders) ? data.folders : [];
        const selected = dialog.folder.value || "*";
        dialog.folder.replaceChildren();
        const all = document.createElement("option");
        all.value = "*";
        all.textContent = "all folders / все папки";
        dialog.folder.appendChild(all);
        for (const name of folders) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            dialog.folder.appendChild(option);
        }
        dialog.folder.value = folders.includes(selected) ? selected : "*";
        filterThumbnails();
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] thumbnails failed:", error);
        if (thumbDialog === dialog) dialog.count.textContent = "error / ошибка";
    }
}

function setPickerKind(kind) {
    const dialog = thumbDialog;
    if (!dialog || dialog.kind === kind) return;
    dialog.kind = kind;
    dialog.files = [];
    dialog.view = [];
    dialog.cursor = 0;
    dialog.folder.value = "*";
    dialog.inputTab.classList.toggle("agsoft-active", kind === "input");
    dialog.outputTab.classList.toggle("agsoft-active", kind === "output");
    dialog.title.textContent = kind === "output"
        ? "🖼 Output thumbnails / Эскизы output"
        : "🖼 Input thumbnails / Эскизы input";
    dialog.note.textContent = kind === "output"
        ? "output → copy into input / output → копия в input"
        : "arrows + Enter / стрелки + Enter";
    loadThumbnails(false);
}

function pickerKeyDown(event) {
    const dialog = thumbDialog;
    if (!dialog) return;
    if (event.key === "Escape") {
        event.stopPropagation();
        closeThumbnails();
        return;
    }
    const inSearch = document.activeElement === dialog.search;
    if (event.key === "Enter") {
        event.preventDefault();
        event.stopPropagation();
        pickThumbIndex(dialog.cursor);
        return;
    }
    // В поле поиска стрелки влево/вправо двигают каретку текста.
    // Inside the search field left/right move the text caret.
    if (inSearch && (event.key === "ArrowLeft" || event.key === "ArrowRight")) return;
    const deltas = {
        ArrowLeft: -1,
        ArrowRight: 1,
        ArrowUp: -dialog.cols,
        ArrowDown: dialog.cols,
    };
    const delta = deltas[event.key];
    if (delta === undefined) return;
    event.preventDefault();
    event.stopPropagation();
    moveThumbCursor(delta);
}

function openThumbnails(node) {
    if (thumbDialog && thumbDialog.node === node && Date.now() - thumbDialog.openedAt < PICKER_GUARD_MS) {
        return true;
    }
    closeThumbnails();
    try {
        const overlay = document.createElement("div");
        overlay.className = "agsoft-thumbs-overlay";
        const panel = document.createElement("div");
        panel.className = "agsoft-thumbs-panel";
        const head = document.createElement("div");
        head.className = "agsoft-thumbs-head";
        const foot = document.createElement("div");
        foot.className = "agsoft-thumbs-foot";

        const title = document.createElement("div");
        title.className = "agsoft-thumbs-title agsoft-thumbs-spacer";
        title.textContent = "🖼 Input thumbnails / Эскизы input";

        const count = document.createElement("div");
        count.className = "agsoft-thumbs-title";
        count.textContent = "…";

        const search = document.createElement("input");
        search.className = "agsoft-thumbs-field";
        search.type = "text";
        search.placeholder = "search / поиск";

        const folder = document.createElement("select");
        folder.className = "agsoft-thumbs-field";

        const size = document.createElement("input");
        size.className = "agsoft-thumbs-field";
        size.type = "range";
        size.min = "90";
        size.max = "260";
        size.step = "10";
        size.value = "140";
        size.title = "Thumbnail size / Размер эскизов";

        const refresh = document.createElement("button");
        refresh.className = "agsoft-resize-btn";
        refresh.textContent = "⟳";
        refresh.title = "Refresh / Обновить";

        const close = document.createElement("button");
        close.className = "agsoft-resize-btn";
        close.textContent = "✕";
        close.title = "Close / Закрыть";

        const inputTab = document.createElement("button");
        inputTab.className = "agsoft-thumbs-tab agsoft-active";
        inputTab.textContent = "📥 input";
        inputTab.title = "Input folder / Папка input";

        const outputTab = document.createElement("button");
        outputTab.className = "agsoft-thumbs-tab";
        outputTab.textContent = "📤 output";
        outputTab.title = "Output folder / Папка output";

        const note = document.createElement("div");
        note.className = "agsoft-thumbs-note";
        note.textContent = "arrows + Enter / стрелки + Enter";

        const viewport = document.createElement("div");
        viewport.className = "agsoft-thumbs-viewport";
        const sizer = document.createElement("div");
        sizer.className = "agsoft-thumbs-sizer";
        const empty = document.createElement("div");
        empty.className = "agsoft-thumbs-empty";
        empty.textContent = "no images / нет изображений";
        empty.style.display = "none";
        viewport.appendChild(sizer);
        viewport.appendChild(empty);

        head.appendChild(title);
        head.appendChild(count);
        head.appendChild(search);
        head.appendChild(folder);
        head.appendChild(size);
        head.appendChild(refresh);
        head.appendChild(close);
        foot.appendChild(inputTab);
        foot.appendChild(outputTab);
        foot.appendChild(note);
        panel.appendChild(head);
        panel.appendChild(viewport);
        panel.appendChild(foot);
        overlay.appendChild(panel);

        document.addEventListener("keydown", pickerKeyDown, true);
        overlay.addEventListener("pointerdown", (event) => {
            event.stopPropagation();
            if (event.target === overlay) closeThumbnails();
        });

        thumbDialog = {
            node,
            overlay,
            viewport,
            sizer,
            empty,
            search,
            folder,
            size,
            count,
            title,
            note,
            inputTab,
            outputTab,
            onKey: pickerKeyDown,
            kind: "input",
            files: [],
            view: [],
            cards: new Map(),
            cols: 1,
            cell: 140,
            rowH: 148,
            px: 256,
            current: "",
            cursor: 0,
            raf: 0,
            searchTimer: null,
            observer: null,
            openedAt: Date.now(),
        };
        const dialog = thumbDialog;

        viewport.addEventListener("scroll", () => {
            if (dialog.raf) return;
            dialog.raf = requestAnimationFrame(() => {
                dialog.raf = 0;
                renderThumbWindow();
            });
        });
        search.oninput = () => {
            if (dialog.searchTimer) clearTimeout(dialog.searchTimer);
            dialog.searchTimer = setTimeout(() => {
                dialog.searchTimer = null;
                filterThumbnails();
            }, SEARCH_DEBOUNCE_MS);
        };
        folder.onchange = () => filterThumbnails();
        size.oninput = () => layoutThumbnails();
        refresh.onclick = () => loadThumbnails(true);
        close.onclick = () => closeThumbnails();
        inputTab.onclick = () => setPickerKind("input");
        outputTab.onclick = () => setPickerKind("output");

        document.body.appendChild(overlay);
        if (typeof ResizeObserver === "function") {
            dialog.observer = new ResizeObserver(() => layoutThumbnails());
            dialog.observer.observe(viewport);
        }
        search.focus();
        layoutThumbnails();
        loadThumbnails(false);
        return true;
    } catch (error) {
        console.warn("[AGSoft Load Image & Mask Resize] picker failed:", error);
        closeThumbnails();
        return false;
    }
}


// =============================================================================
// ВИДЖЕТ image: стрелки по краям, эскизы по центру
// image widget: arrows at the edges, picker in the middle
// =============================================================================
function comboLocalX(node, options) {
    const canvas = app.canvas;
    const event = options && options.e ? options.e : (options && options.event ? options.event : options);
    if (canvas && event && typeof event.clientX === "number" &&
        typeof canvas.convertEventToCanvasOffset === "function") {
        const offset = canvas.convertEventToCanvasOffset(event);
        return offset[0] - node.pos[0];
    }
    if (canvas && canvas.graph_mouse) {
        return canvas.graph_mouse[0] - node.pos[0];
    }
    return node.size[0] / 2;
}

function stepComboValue(node, widget, direction) {
    const options = widget.options || {};
    const values = Array.isArray(options.values) ? options.values : widget.values;
    if (!Array.isArray(values) || !values.length) return;
    const current = values.indexOf(widget.value);
    const start = current < 0 ? 0 : current;
    const index = (start + direction + values.length) % values.length;
    const next = values[index];
    widget.value = next;
    if (widget.callback) widget.callback(next, app.canvas, node);
    node.setDirtyCanvas(true, true);
}

function handleImageClick(node, widget, localX) {
    const now = Date.now();
    if (now - num(widget._agsoftClickAt, 0) < CLICK_GUARD_MS) return true;
    widget._agsoftClickAt = now;
    if (localX < COMBO_ARROW_ZONE) {
        stepComboValue(node, widget, -1);
        return true;
    }
    if (localX > node.size[0] - COMBO_ARROW_ZONE) {
        stepComboValue(node, widget, 1);
        return true;
    }
    return openThumbnails(node);
}

function hookImagePicker(node) {
    const widget = findWidget(node, "image");
    if (!widget || widget._agsoftPickerHooked) return;
    widget._agsoftPickerHooked = true;
    const previousClick = widget.onClick;
    widget.onClick = function(options) {
        if (handleImageClick(node, widget, comboLocalX(node, options))) return true;
        return previousClick ? previousClick.apply(this, arguments) : false;
    };
    const previousMouse = widget.mouse;
    widget.mouse = function(event, pos) {
        const localX = Array.isArray(pos) ? num(pos[0], 0) : comboLocalX(node, event);
        if (handleImageClick(node, widget, localX)) return true;
        return previousMouse ? previousMouse.apply(this, arguments) : false;
    };
}


// =============================================================================
// ВИДЖЕТЫ ПРЕВЬЮ И СТАТУС-БАРА / PREVIEW AND STATUS BAR WIDGETS
// =============================================================================
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
        applyTransfer(node, readTransfer(event.dataTransfer));
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

function createStatusBar(node) {
    const state = node._agsoft;
    const bar = document.createElement("div");
    bar.className = "agsoft-resize-status";
    bar.title = "Source MP/W/H → output MP/W/H (live) / " +
        "МП/Ш/В источника → МП/Ш/В результата (в реальном времени)";

    const text = document.createElement("div");
    text.className = "agsoft-resize-status-text";
    text.textContent = "— × — · — MP";

    const thumbsBtn = document.createElement("button");
    thumbsBtn.className = "agsoft-resize-btn";
    thumbsBtn.textContent = "🖼";
    thumbsBtn.title = "Thumbnails / Эскизы";

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
    bar.appendChild(thumbsBtn);
    bar.appendChild(uploadBtn);
    bar.appendChild(toggleBtn);
    bar.appendChild(fileInput);
    bar.addEventListener("pointerdown", (event) => event.stopPropagation());
    attachNodeMenu(node, bar);

    thumbsBtn.onclick = () => openThumbnails(node);
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
// ХУКИ НОДЫ / NODE HOOKS
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
        state.stamp = "";
        state.comboWatch = String(widgetValue(node, "image") || "");
        state.restoring = true;
        applyVisibility(node);
        state.restoring = false;
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
        const state = node._agsoft;
        const key = `${node.color || ""}|${node.bgcolor || ""}`;
        if (key !== state.themeKey) {
            state.themeKey = key;
            applyTheme(node);
        }
        if (state.dropHover && ctx) {
            ctx.save();
            ctx.strokeStyle = "#7fd4ff";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(1, 1, node.size[0] - 2, node.size[1] - 2);
            ctx.restore();
        }
    };

    const baseOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (baseOnRemoved) baseOnRemoved.apply(this, arguments);
        const state = node._agsoft;
        if (state && state.pathTimer) clearTimeout(state.pathTimer);
        if (thumbDialog && thumbDialog.node === node) closeThumbnails();
    };
}


// =============================================================================
// DROP НА ВСЮ НОДУ / DROP OVER THE WHOLE NODE
// =============================================================================
function nodeAtPointer(event) {
    const canvas = app.canvas;
    if (!canvas || !app.graph) return null;
    if (typeof canvas.convertEventToCanvasOffset !== "function") return null;
    const offset = canvas.convertEventToCanvasOffset(event);
    const nodes = app.graph._nodes || [];
    for (let i = nodes.length - 1; i >= 0; i--) {
        const node = nodes[i];
        if (node.comfyClass !== NODE_CLASS || !node._agsoft) continue;
        if (offset[0] >= node.pos[0] && offset[0] <= node.pos[0] + node.size[0] &&
            offset[1] >= node.pos[1] && offset[1] <= node.pos[1] + node.size[1]) {
            return node;
        }
    }
    return null;
}

function clearDropHover(except) {
    const nodes = (app.graph && app.graph._nodes) || [];
    for (const node of nodes) {
        if (node.comfyClass !== NODE_CLASS || !node._agsoft) continue;
        if (node === except || !node._agsoft.dropHover) continue;
        node._agsoft.dropHover = false;
        node.setDirtyCanvas(true, false);
    }
}

function isGraphCanvasEvent(event) {
    return !!(app.canvas && app.canvas.canvas && event.target === app.canvas.canvas);
}

function onCanvasDragOver(event) {
    if (!isGraphCanvasEvent(event)) return;
    const node = nodeAtPointer(event);
    clearDropHover(node);
    if (!node) return;
    event.preventDefault();
    event.stopPropagation();
    if (!node._agsoft.dropHover) {
        node._agsoft.dropHover = true;
        node.setDirtyCanvas(true, false);
    }
}

function onCanvasDragLeave(event) {
    if (!isGraphCanvasEvent(event)) return;
    clearDropHover(null);
}

function onCanvasDrop(event) {
    if (!isGraphCanvasEvent(event)) return;
    const node = nodeAtPointer(event);
    clearDropHover(null);
    if (!node) return;
    const data = readTransfer(event.dataTransfer);
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    applyTransfer(node, data);
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
        document.addEventListener("dragover", onCanvasDragOver, true);
        document.addEventListener("dragleave", onCanvasDragLeave, true);
        document.addEventListener("drop", onCanvasDrop, true);
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
            dropHover: false,
            baseComputeSize: node.computeSize,
            sourceImg: null,
            frame: null,
        };

        createPreview(node);
        createStatusBar(node);
        hookWidgets(node);
        hookImagePicker(node);
        hookNode(node);
        applyTheme(node);
        applyVisibility(node);
        refreshNodeSize(node, PREVIEW_DEFAULT_AREA);
        node._agsoft.comboWatch = String(widgetValue(node, "image") || "");
        showComboPreview(node);
        refreshState(node);
    },
});