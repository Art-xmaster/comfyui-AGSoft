// ==============================================================================
// agsoft_image_compare.js
// ==============================================================================
// JS extension for the 🖼️↔️AGSoft Image Compare node.
// Версия / Version: v11.28
// Автор / Author: AGSoft
// Дата / Date: 28.11.2026
// ------------------------------------------------------------------------------
// Описание / Description:
//
// EN:
// Frontend companion of the comparison node: a single DOM widget inside the node
// body shows the server-cached PNG previews and lets the user judge the
// difference visually. Three view modes are supported. In wipe mode two <img>
// elements are stacked (B underneath, A on top), the top one is clipped with
// clip-path inset by the wipe percentage, and a 2px divider line without a
// handle circle marks the boundary. In side_by_side mode a single server image S
// shows both full frames B|A, so nothing is composited on the client. In
// difference mode a single server image D shows the amplified difference.
//
// Layout. The widget is one flex column: the preview takes all the free space
// (flex:1 1 auto, min-height:0), while the controls row and the stats line have
// fixed heights. In the legacy canvas the panel gets an explicit pixel height;
// in Nodes 2.0 it gets 100% and adopts the space allocated by the renderer, so an
// empty band between the preview and the bars is impossible in both renderers.
//
// Sizing. The widget-free base height is measured by temporarily zeroing the
// panel's own computeSize. node.computeSize reports honest content height in
// Nodes 2.0 and a size-independent minimum in the legacy canvas, so the node can
// always be shrunk back. onResize absorbs the user's height into the panel
// height, and the resulting node size is stored in node.properties and restored
// after a workflow reload or a tab switch through a series of deferred ticks,
// which survives late layout passes.
//
// Render mode. The Vue node renderer is detected from the frontend setting or
// from canvas draw silence and re-checked on a reconciliation tick; the same
// tick unconditionally reapplies widget visibility, so an external relayout
// cannot leave stuck visible widgets.
//
// Interaction. pointermove, pointerup, pointercancel and wheel are listened on
// document in the capture phase and hit-tested against the preview rectangle:
// capture runs before any renderer handler, so drag and wheel work in both
// renderers even when lower layers swallow events. The preview and every visual
// layer are pointer-events:none, therefore all events fall through to the panel,
// which is the single interactive target. pointerdown on the panel starts a drag
// and takes pointer capture. The drag moves the wipe divider when the press is
// near the line, when the view is not zoomed in wipe mode, or when Shift is
// held; otherwise it pans. The wheel zooms the preview instead of the graph,
// double click resets zoom and pan, and holding Space while the cursor is over
// the preview temporarily shows side A only.
//
// Widget spoiler. The first button of the controls row collapses every parameter
// widget through four independent channels (the hidden property, the options
// property, a zero computeSize and an off-screen y), leaving only the preview,
// its controls and the stats line. Widgets with a connected link are never
// hidden, inputs and links are never modified, all values survive, and the state
// is stored in node.properties: a new node starts expanded, a loaded node
// restores its own state.
//
// Controls and stats. The row holds the spoiler arrow, save frame, minus/plus
// one frame, reset view, a frame slider over the cached frames and an "f i/n"
// label; it shrinks gracefully to any node width. The fixed-height stats line
// below shows the frame size, the frame index, the mean and maximum difference
// and the share of changed pixels.
//
// Save. The save button composites the currently displayed comparison frame into
// a canvas per active mode (wipe draws B fully and then A clipped at the divider;
// the other modes draw the single server image as is), converts it to a PNG blob
// and posts it to the loopback-only endpoint; on success the frame label is
// replaced by "saved: <file name>" for two seconds.
//
// Metadata and lifecycle. Cache metadata is fetched by an independent poll per
// node and additionally on graph execution events and on workflow configure;
// until a key arrives the preview shows a bilingual placeholder. Widget
// callbacks are wrapped with the previous callback preserved and invoked first.
// Every initialization step is isolated, so a failure of one step cannot disable
// the others; timers, observers, global listeners and the live-panel entry are
// released when the node is removed.
//
// ---
//
// RU:
// Фронтенд-спутник ноды сравнения: один DOM-виджет внутри тела ноды показывает
// кэшированные сервером PNG-превью и даёт визуально оценить разницу.
// Поддерживаются три режима просмотра. В режиме wipe два элемента <img> наложены
// друг на друга (B снизу, A сверху), верхний обрезан через clip-path inset по
// проценту wipe, а границу отмечает линия-разделитель 2px без кружка. В режиме
// side_by_side одно серверное изображение S показывает оба полных кадра B|A,
// поэтому на клиенте ничего не склеивается. В режиме difference одно серверное
// изображение D показывает усиленную разницу.
//
// Раскладка. Виджет — одна flex-колонка: превью забирает всё свободное место
// (flex:1 1 auto, min-height:0), а строка управления и строка статистики имеют
// фиксированную высоту. В Classic-канвасе панель получает явную пиксельную
// высоту; в Nodes 2.0 она получает 100% и подхватывает место, выделенное
// рендерером, поэтому пустая полоса между превью и полосами невозможна в обоих
// рендерерах.
//
// Размеры. Базовая высота без виджета измеряется временным обнулением
// собственного computeSize панели. node.computeSize в Nodes 2.0 отдаёт честную
// высоту контента, а в Classic-канвасе — минимум, не зависящий от текущего
// размера, поэтому ноду всегда можно сжать обратно. onResize впитывает
// пользовательскую высоту в высоту панели, а итоговый размер ноды пишется в
// node.properties и восстанавливается после перезагрузки workflow или смены
// вкладки серией отложенных тиков, что переживает поздние проходы раскладки.
//
// Режим рендера. Vue-рендерер нод определяется по настройке фронтенда либо по
// тишине canvas-отрисовки и перепроверяется в тике реконсиляции; тот же тик
// безусловно переприменяет видимость виджетов, поэтому внешняя перекладка не
// может оставить залипшие видимые виджеты.
//
// Взаимодействие. pointermove, pointerup, pointercancel и wheel слушаются на
// document в capture-фазе с hit-test по rect превью: capture срабатывает раньше
// любых обработчиков рендерера, поэтому драг и колесо работают в обоих
// рендерерах, даже если нижние слои гасят события. Превью и все визуальные слои
// имеют pointer-events:none, поэтому все события проваливаются на панель — она
// единственная интерактивная цель. pointerdown на панели начинает драг и берёт
// pointer capture. Драг двигает wipe-разделитель, если нажатие рядом с линией,
// если вид не увеличен в режиме wipe или если удерживается Shift; иначе он
// панорамирует. Колесо масштабирует превью, а не граф, двойной клик сбрасывает
// зум и панорамирование, а удержание Space при курсоре над превью временно
// показывает только сторону A.
//
// Спойлер виджетов. Первая кнопка строки управления сворачивает все виджеты
// параметров через четыре независимых канала (свойство hidden, свойство options,
// нулевой computeSize и вынос y за экран), оставляя только превью, его панель и
// строку статистики. Виджеты с подключённым линком не скрываются никогда, входы
// и линки не модифицируются, все значения сохраняются, состояние пишется в
// node.properties: новая нода создаётся развёрнутой, загруженная восстанавливает
// своё состояние.
//
// Панель и статистика. Строка содержит стрелку-спойлер, сохранение кадра,
// минус/плюс один кадр, сброс вида, слайдер по кэшированным кадрам и метку
// «f i/n»; она gracefully сжимается под любую ширину ноды. Строка статистики
// фиксированной высоты под ней показывает размер кадра, индекс кадра, среднюю и
// максимальную разницу и долю изменённых пикселей.
//
// Сохранение. Кнопка сохранения собирает текущий отображаемый кадр сравнения в
// canvas по активному режиму (wipe рисует B целиком, затем A, обрезанный по
// разделителю; остальные режимы рисуют одно серверное изображение как есть),
// конвертирует его в PNG-blob и отправляет на loopback-only эндпоинт; при успехе
// метка кадра на две секунды заменяется на «saved: <имя файла>».
//
// Метаданные и жизненный цикл. Метаданные кэша запрашиваются независимым
// поллингом на ноду, дополнительно по событиям выполнения графа и при
// конфигурировании workflow; пока ключ не получен, в превью висит двуязычная
// заглушка. Колбэки виджетов обёрнуты с сохранением и первоочередным вызовом
// предыдущего колбэка. Каждый шаг инициализации изолирован, поэтому сбой одного
// шага не отключает остальные; таймеры, наблюдатели, глобальные слушатели и
// запись в списке живых панелей освобождаются при удалении ноды.
// ------------------------------------------------------------------------------
// Возможности / Features:
//
// ⚡ Three view modes from the server PNG cache: wipe with a draggable 2px line
// and no handle circle, side_by_side (single image S), difference (image D).
// Три режима просмотра из серверного PNG-кэша: wipe с перетаскиваемой линией 2px
// без кружка, side_by_side (одно изображение S), difference (изображение D).
//
// ⚡ One flex-column DOM widget: preview flexes, controls and stats have fixed
// heights, no empty band in Classic or Nodes 2.0.
// Один flex-колоночный DOM-виджет: превью тянется, панель и статистика
// фиксированы, пустой полосы нет ни в Classic, ни в Nodes 2.0.
//
// ⚡ Free resize in both directions: honest content height in Nodes 2.0, a
// size-independent minimum in Classic, user size persisted and restored.
// Свободный ресайз в обе стороны: честная высота контента в Nodes 2.0, минимум не
// зависит от размера в Classic, пользовательский размер сохраняется и
// восстанавливается.
//
// ⚡ Document-capture interaction with rect hit-testing: divider drag, pan, wheel
// zoom and double-click reset work in both renderers.
// Взаимодействие через document-capture с hit-test по rect: драг разделителя,
// панорамирование, зум колесом и сброс двойным кликом работают в обоих
// рендерерах.
//
// ⚡ Widget spoiler through four independent channels: only the preview panel
// stays, linked widgets are never hidden, values and links survive.
// Спойлер виджетов через четыре независимых канала: остаётся только панель
// превью, виджеты с линком не скрываются, значения и линки сохраняются.
//
// ⚡ Batch frame slider with step buttons, Space hold to peek side A, and a reset
// button for zoom and pan.
// Слайдер кадра батча с кнопками шага, удержание Space для показа стороны A и
// кнопка сброса зума и панорамирования.
//
// ⚡ Live stats line: frame size, frame index, mean and maximum difference and
// the share of changed pixels.
// Живая строка статистики: размер кадра, индекс кадра, средняя и максимальная
// разница и доля изменённых пикселей.
//
// ⚡ Save button composites the current comparison frame per mode and stores it
// as PNG into the output folder, with a two-second label confirmation.
// Кнопка сохранения собирает текущий кадр сравнения по режиму и пишет его в PNG
// в папку output с двухсекундным подтверждением в метке.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Image Compare] JS extension loaded v11.28 (widget spoiler, capture interaction)");

const NODE_CLASS = "AGSoftImageCompare";
const MIN_NODE_WIDTH = 320;
const CTRL_H = 30;                 // высота панели управления / controls row height
const BAR_H = 20;                  // высота строки статистики / stats line height
const GAP = 6;                     // зазор flex-колонки / flex column gap
const FIXED_H = CTRL_H + BAR_H + GAP * 2; // фиксированная часть панели / fixed panel part
const MIN_PREVIEW_H = 60;          // минимум превью / minimum preview height
const MIN_PANEL_H = MIN_PREVIEW_H + FIXED_H;
const PANEL_DEFAULT_H = 260 + FIXED_H;
const BOTTOM_GAP = 4;              // зазор под панелью в Classic / gap below panel in Classic
const POLL_MS = 1500;              // шаг поллинга meta / meta polling step
const MODE_TICK_MS = 500;          // шаг реконсиляции / reconciliation step
const VUE_SILENCE_MS = 1500;       // тишина canvas = Vue / canvas silence means Vue
const RESTORE_TICKS = [0, 60, 180, 420];
const RESTORE_LOCK_MS = 1200;
const Y_OFF = -9999;               // вынос из карты попаданий / off the hit map

const modeWatchers = new Set();
let modeTimer = null;

// Реестр живых панелей и состояние активного драга (одно на страницу).
// Registry of live panels and the active drag state (one per page).
const livePanels = new Set();
const drag = { st: null, kind: "wipe", x: 0, y: 0, px: 0, py: 0 };
let globalBound = false;

function safe(fn, label) {
    try {
        return fn();
    } catch (e) {
        console.warn("[AGSoft Image Compare] " + label + " skipped:", e);
        return null;
    }
}

function num(v, d) { const p = Number(v); return Number.isFinite(p) ? p : d; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function base() { return (api && api.api_base) ? api.api_base : ""; }

function el(tag, css, parent) {
    const d = document.createElement(tag);
    if (css) d.style.cssText = css;
    if (parent) parent.appendChild(d);
    return d;
}

function findWidget(node, name) {
    for (const w of node.widgets || []) { if (w && w.name === name) return w; }
    return null;
}
function widgetValue(node, name, dflt) {
    const w = findWidget(node, name);
    return w ? w.value : dflt;
}
function setWidgetValue(node, name, value) {
    const w = findWidget(node, name);
    if (w) w.value = value;
}

function imgUrl(key, side, frame) {
    return base() + `/agsoft/image_compare/image?key=${key}&side=${side}&frame=${frame}`;
}
function metaUrl(node) {
    return base() + `/agsoft/image_compare/meta?node=${String(node.id)}`;
}

function vueNodesSetting() {
    if (!app.ui || !app.ui.settings || typeof app.ui.settings.getSettingValue !== "function") return null;
    try {
        const v = app.ui.settings.getSettingValue("Comfy.VueNodes.Enabled");
        return typeof v === "boolean" ? v : null;
    } catch (e) { return null; }
}
function watchMode(c) {
    modeWatchers.add(c);
    if (!modeTimer) modeTimer = setInterval(() => {
        const now = Date.now();
        for (const ctl of modeWatchers) {
            try { ctl.check(now); } catch (e) {
                console.warn("[AGSoft Image Compare] mode check skipped:", e);
            }
        }
    }, MODE_TICK_MS);
}
function unwatchMode(c) {
    modeWatchers.delete(c);
    if (!modeWatchers.size && modeTimer) { clearInterval(modeTimer); modeTimer = null; }
}

//------------------------------------------------------------------------------
// Спойлер виджетов: четыре независимых канала скрытия
// Widget spoiler: four independent hiding channels
//------------------------------------------------------------------------------
function isLinkedInput(node, name) {
    if (!name) return false;
    for (const inp of node.inputs || []) {
        if (inp && inp.name === name && inp.link !== null && inp.link !== undefined) return true;
    }
    return false;
}

function zeroSize() { return [0, -4]; }

function applySize(w, hidden) {
    if (hidden) {
        if (!Object.prototype.hasOwnProperty.call(w, "__ag_cs")) {
            w.__ag_cs = typeof w.computeSize === "function" ? w.computeSize : null;
        }
        w.computeSize = zeroSize;
    } else if (Object.prototype.hasOwnProperty.call(w, "__ag_cs")) {
        if (w.__ag_cs) w.computeSize = w.__ag_cs;
        else delete w.computeSize;
        delete w.__ag_cs;
    }
}

function setHidden(w, hidden) {
    safe(() => { w.hidden = hidden; }, "hidden flag");
    safe(() => { if (w.options) w.options.hidden = hidden; }, "options flag");
    safe(() => {
        if (hidden) w.y = Y_OFF;
        else if (w.y === Y_OFF) w.y = 0;
    }, "hit map");
    safe(() => applySize(w, hidden), "size");
}

// Переприменяется безусловно: внешняя перекладка не оставит залипших виджетов.
// Reapplied unconditionally so an external relayout cannot leave stuck widgets.
function applyVisibility(node) {
    const st = node.__ag;
    if (!st) return;
    for (const w of node.widgets || []) {
        if (!w || w === st.panelWidget || w.name === "agsoft_panel") continue;
        const want = isLinkedInput(node, w.name) ? false : !!st.collapsed;
        setHidden(w, want);
    }
}

function updateToggle(node) {
    const st = node.__ag;
    if (!st || !st.toggleBtn) return;
    st.toggleBtn.textContent = st.collapsed ? "▾" : "▴";
    st.toggleBtn.title = st.collapsed
        ? "Show all parameters / Показать все параметры"
        : "Hide all parameters / Скрыть все параметры";
}

//------------------------------------------------------------------------------
// Размеры: панель + пользовательский размер / sizes: panel + user size
//------------------------------------------------------------------------------
function baseSize(node) {
    const st = node.__ag;
    const w = st.panelWidget;
    const orig = w ? w.computeSize : null;
    if (w) w.computeSize = () => [0, 0];
    let s;
    try {
        s = st.baseComputeSize ? st.baseComputeSize.call(node) : [MIN_NODE_WIDTH, 120];
    } finally {
        if (w) w.computeSize = orig;
    }
    return [s[0] || MIN_NODE_WIDTH, Math.max(0, s[1] || 0)];
}
function panelHeight(node) {
    const st = node.__ag;
    return Math.max(MIN_PANEL_H, num(st.panelH, PANEL_DEFAULT_H));
}
function bottomGap(node) { return node.__ag.vueMode ? 0 : BOTTOM_GAP; }

function syncPreviewBox(node) {
    const st = node.__ag;
    if (!st || !st.panelEl) return;
    st.panelEl.style.height = st.vueMode ? "100%" : `${panelHeight(node)}px`;
}
function persistUserSize(node) {
    const st = node.__ag;
    if (!st || !st.ready || !node.properties) return;
    const w = Math.round(num(node.size[0], MIN_NODE_WIDTH));
    const h = Math.round(num(node.size[1], 0));
    if (h <= 0) return;
    const prev = node.properties.agsoft_node_size;
    if (Array.isArray(prev) && prev[0] === w && prev[1] === h) return;
    node.properties.agsoft_node_size = [w, h];
}
function savedNodeSize(node) {
    const raw = node.properties ? node.properties.agsoft_node_size : null;
    if (!Array.isArray(raw) || raw.length < 2) return null;
    const w = num(raw[0], 0), h = num(raw[1], 0);
    if (w < MIN_NODE_WIDTH || h <= 0) return null;
    return [w, h];
}
function refreshNodeSize(node) {
    const st = node.__ag;
    if (!st) return;
    const base = baseSize(node);
    const width = Math.max(node.size[0], base[0], MIN_NODE_WIDTH);
    const height = base[1] + panelHeight(node) + bottomGap(node);
    if (node.size[0] !== width || Math.abs(node.size[1] - height) > 0.5) node.setSize([width, height]);
    st.lastHeight = node.size[1];
    persistUserSize(node);
    syncPreviewBox(node);
    node.setDirtyCanvas(true, true);
}
function applyUserSize(node) {
    const st = node.__ag;
    if (!st) return;
    const saved = savedNodeSize(node);
    const base = baseSize(node);
    if (saved) {
        st.panelH = Math.max(MIN_PANEL_H, saved[1] - base[1] - bottomGap(node));
        if (node.size[0] !== saved[0] || Math.abs(node.size[1] - saved[1]) > 0.5) {
            node.setSize([saved[0], saved[1]]);
        }
        st.lastHeight = node.size[1];
        syncPreviewBox(node);
        node.setDirtyCanvas(true, true);
        return;
    }
    refreshNodeSize(node);
}
function scheduleRestore(node) {
    const st = node.__ag;
    if (!st) return;
    st.ready = false;
    for (const t of st.restoreTimers) clearTimeout(t);
    st.restoreTimers = [];
    for (const d of RESTORE_TICKS) {
        st.restoreTimers.push(setTimeout(() => { if (node.__ag) applyUserSize(node); }, d));
    }
    st.restoreTimers.push(setTimeout(() => {
        if (!node.__ag) return;
        applyUserSize(node);
        node.__ag.ready = true;
        persistUserSize(node);
    }, RESTORE_LOCK_MS));
}
function setVueMode(node, on) {
    const st = node.__ag;
    if (!st || st.vueMode === on) return;
    st.vueMode = on;
    syncPreviewBox(node);
    if (st.ready) refreshNodeSize(node);
    node.setDirtyCanvas(true, true);
}

// Спойлер: высота превью сохраняется, нода сжимается ровно на строки виджетов.
// Spoiler: the preview height is kept, the node shrinks by the widget rows only.
function applyCollapsed(node, collapsed) {
    const st = node.__ag;
    if (!st) return;
    st.collapsed = !!collapsed;
    node.properties = node.properties || {};
    node.properties.agsoft_collapsed = st.collapsed;
    safe(() => applyVisibility(node), "visibility");
    safe(() => updateToggle(node), "toggle");
    safe(() => {
        // В Vue раскладкой владеет рендерер: высоту не пересчитываем.
        // In Vue the renderer owns the layout: do not recompute the height.
        if (st.vueMode) {
            syncPreviewBox(node);
            node.setDirtyCanvas(true, true);
            return;
        }
        refreshNodeSize(node);
    }, "collapse size");
}

//------------------------------------------------------------------------------
// Вид области просмотра / viewport rendering
//------------------------------------------------------------------------------
function setWipe(st) {
    if (st.layerA) st.layerA.style.clipPath = `inset(0 0 0 ${st.peek ? 0 : st.wipe}%)`;
    if (st.handle) {
        st.handle.style.left = `${st.wipe}%`;
        st.handle.style.opacity = st.peek ? "0" : "1";
    }
}
function applyTransform(st) {
    const tr = `translate(${st.panX}px, ${st.panY}px) scale(${st.zoom})`;
    for (const img of st.imgs) img.style.transform = tr;
}
function updateSrcs(st) {
    if (!st.key) return;
    if (st.mode === "wipe") {
        if (st.imgA) st.imgA.src = imgUrl(st.key, "A", st.frame);
        if (st.imgB) st.imgB.src = imgUrl(st.key, "B", st.frame);
    } else if (st.master) {
        const side = st.peek ? "A" : (st.mode === "side_by_side" ? "S" : "D");
        st.master.src = imgUrl(st.key, side, st.frame);
    }
}
function makeImg(st, css, parent) {
    const img = el("img", css, parent);
    img.draggable = false;
    img.decoding = "async";
    img.onerror = () => console.warn("[AGSoft Image Compare] preview image failed to load");
    st.imgs.push(img);
    return img;
}

function saveFrame(node, st) {
    if (!st.key) return;
    const mode = st.mode;
    let canvas = null;
    try {
        if (mode === "wipe" && st.imgA && st.imgB) {
            const w = st.imgB.naturalWidth, h = st.imgB.naturalHeight;
            if (!w || !h) return;
            canvas = document.createElement("canvas");
            canvas.width = w; canvas.height = h;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(st.imgB, 0, 0, w, h);
            const x = Math.round(w * st.wipe / 100);
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, 0, w - x, h);
            ctx.clip();
            ctx.drawImage(st.imgA, 0, 0, w, h);
            ctx.restore();
        } else if (st.master && st.master.naturalWidth) {
            const w = st.master.naturalWidth, h = st.master.naturalHeight;
            canvas = document.createElement("canvas");
            canvas.width = w; canvas.height = h;
            canvas.getContext("2d").drawImage(st.master, 0, 0, w, h);
        }
    } catch (e) {
        console.warn("[AGSoft Image Compare] save frame canvas skipped:", e);
        return;
    }
    if (!canvas) return;
    canvas.toBlob((blob) => {
        if (!blob) return;
        fetch(base() + `/agsoft/image_compare/save_frame?mode=${mode}&node=${String(node.id)}`,
            { method: "POST", body: blob })
            .then((r) => r.json())
            .then((j) => {
                if (st.label && j && j.name) {
                    st.label.textContent = "saved: " + j.name;
                    st.labelLock = Date.now() + 2000;
                }
            })
            .catch((e) => console.warn("[AGSoft Image Compare] save frame failed:", e));
    }, "image/png");
}

function setFrame(node, st, i) {
    const pf = st.meta && st.meta.preview_frames ? st.meta.preview_frames : 1;
    st.frame = clamp(Math.round(i), 0, Math.max(0, pf - 1));
    if (st.range) st.range.value = String(st.frame);
    updateSrcs(st);
    updateStats(node);
}
function updateStats(node) {
    const st = node.__ag;
    if (!st || !st.stats) return;
    let txt = "run the graph to preview / запустите граф для превью";
    if (st.key && st.meta) {
        const pf = st.meta.preview_frames || 1;
        const s = (st.meta.stats && st.meta.stats[st.frame]) || {};
        txt = `${st.meta.w}×${st.meta.h} · f${st.frame + 1}/${pf} · ` +
            `mean ${s.mean !== undefined ? s.mean : "—"} · max ${s.max !== undefined ? s.max : "—"} · ` +
            `Δ ${s.pct !== undefined ? s.pct : "—"}%`;
        if (st.label && (!st.labelLock || Date.now() > st.labelLock)) {
            const lt = `f ${st.frame + 1}/${pf}`;
            if (st.label.textContent !== lt) st.label.textContent = lt;
        }
    }
    if (st.stats.textContent !== txt) st.stats.textContent = txt;
}

function render(node) {
    const st = node.__ag;
    if (!st || !st.previewEl) return;
    st.previewEl.textContent = "";
    st.imgs = []; st.imgA = null; st.imgB = null;
    st.layerA = null; st.master = null; st.handle = null;
    st.mode = widgetValue(node, "view_mode", "wipe");
    st.wipe = num(widgetValue(node, "wipe_position", 50), 50);
    st.zoom = num(widgetValue(node, "zoom", 1), 1);
    st.panX = num(widgetValue(node, "pan_x", 0), 0);
    st.panY = num(widgetValue(node, "pan_y", 0), 0);
    if (!st.key) {
        const ph = el("div", "position:absolute;inset:0;display:flex;align-items:center;" +
            "justify-content:center;font:11px monospace;color:#78909c;pointer-events:none;",
            st.previewEl);
        ph.textContent = "run the graph to preview / запустите граф для превью";
        return;
    }
    const fill = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;" +
        "transform-origin:center center;user-select:none;pointer-events:none;";
    if (st.mode === "wipe") {
        const layerB = el("div", "position:absolute;inset:0;z-index:1;pointer-events:none;",
            st.previewEl);
        st.imgB = makeImg(st, fill, layerB);
        st.layerA = el("div", "position:absolute;inset:0;z-index:2;pointer-events:none;",
            st.previewEl);
        st.imgA = makeImg(st, fill, st.layerA);
        st.handle = el("div", "position:absolute;top:0;bottom:0;width:2px;background:#7fd4ff;" +
            "z-index:3;pointer-events:none;", st.previewEl);
        setWipe(st);
    } else {
        st.master = makeImg(st, fill + "z-index:1;", st.previewEl);
    }
    updateSrcs(st);
    applyTransform(st);
}

//------------------------------------------------------------------------------
// Глобальный перехват: capture на document раньше любых обработчиков рендерера
// Global interception: document capture runs before any renderer handler
//------------------------------------------------------------------------------
function previewRect(st) {
    if (!st || !st.previewEl || !st.previewEl.isConnected) return null;
    const r = st.previewEl.getBoundingClientRect();
    return (r.width > 0 && r.height > 0) ? r : null;
}
function insideRect(r, x, y) {
    return !!r && x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
}
function hitPreview(x, y) {
    for (const st of livePanels) {
        const r = previewRect(st);
        if (insideRect(r, x, y)) return st;
    }
    return null;
}

function applyDragMove(st, clientX, clientY) {
    const r = previewRect(st);
    if (!r) return;
    const node = st.node;
    if (drag.kind === "wipe") {
        st.wipe = Math.round(clamp(((clientX - r.left) / r.width) * 100, 0, 100));
        setWipe(st);
        setWidgetValue(node, "wipe_position", st.wipe);
    } else {
        st.panX = clamp(Math.round(drag.px + clientX - drag.x), -5000, 5000);
        st.panY = clamp(Math.round(drag.py + clientY - drag.y), -5000, 5000);
        setWidgetValue(node, "pan_x", st.panX);
        setWidgetValue(node, "pan_y", st.panY);
        applyTransform(st);
    }
    node.setDirtyCanvas(true, true);
}

function onGlobalMove(e) {
    if (drag.st) {
        e.preventDefault();
        e.stopPropagation();
        applyDragMove(drag.st, e.clientX, e.clientY);
        return;
    }
    for (const st of livePanels) {
        st.hover = insideRect(previewRect(st), e.clientX, e.clientY);
    }
}
function onGlobalUp(e) {
    if (!drag.st) return;
    const st = drag.st;
    drag.st = null;
    try {
        if (st.panelEl && st.pointerId !== undefined && st.panelEl.hasPointerCapture &&
            st.panelEl.hasPointerCapture(st.pointerId)) {
            st.panelEl.releasePointerCapture(st.pointerId);
        }
    } catch (err) {
        console.warn("[AGSoft Image Compare] release capture skipped:", err);
    }
    st.pointerId = undefined;
    e.stopPropagation();
}
function onGlobalWheel(e) {
    if (drag.st) { e.preventDefault(); e.stopPropagation(); return; }
    const st = hitPreview(e.clientX, e.clientY);
    if (!st || !st.key) return;
    e.preventDefault();
    e.stopPropagation();
    if (typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
    const z = clamp(st.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1), 0.1, 10);
    st.zoom = Math.round(z * 100) / 100;
    setWidgetValue(st.node, "zoom", st.zoom);
    applyTransform(st);
    st.node.setDirtyCanvas(true, true);
}
function bindGlobal() {
    if (globalBound) return;
    globalBound = true;
    document.addEventListener("pointermove", onGlobalMove, true);
    document.addEventListener("pointerup", onGlobalUp, true);
    document.addEventListener("pointercancel", onGlobalUp, true);
    document.addEventListener("wheel", onGlobalWheel, { capture: true, passive: false });
}

//------------------------------------------------------------------------------
// Панель: превью + управление + статистика в одном виджете
// panel: preview + controls + stats inside a single widget
//------------------------------------------------------------------------------
function buildControls(node, st, parent) {
    const row = el("div", `flex:0 0 ${CTRL_H}px;display:flex;align-items:center;gap:4px;` +
        `height:${CTRL_H}px;width:100%;box-sizing:border-box;padding:0 2px;flex-wrap:nowrap;` +
        `overflow:hidden;`, parent);
    const btn = (txt, fn, title) => {
        const b = el("button", "flex:0 0 auto;font:10px monospace;color:#9ecbdd;background:#263238;" +
            "border:1px solid #37474f;border-radius:6px;padding:1px 6px;cursor:pointer;", row);
        b.textContent = txt;
        if (title) b.title = title;
        b.onclick = () => safe(fn, "control");
        return b;
    };
    st.toggleBtn = btn("▴", () => applyCollapsed(node, !st.collapsed),
        "Hide all parameters / Скрыть все параметры");
    btn("💾", () => saveFrame(node, st),
        "save current comparison frame to output / сохранить текущий кадр сравнения в output");
    btn("−1f", () => setFrame(node, st, st.frame - 1), "previous frame / предыдущий кадр");
    btn("+1f", () => setFrame(node, st, st.frame + 1), "next frame / следующий кадр");
    btn("⟲", () => {
        setWidgetValue(node, "zoom", 1.0);
        setWidgetValue(node, "pan_x", 0);
        setWidgetValue(node, "pan_y", 0);
        st.zoom = 1; st.panX = 0; st.panY = 0;
        applyTransform(st);
    }, "reset zoom and pan / сбросить зум и панорамирование");
    const range = el("input", "flex:1 1 20px;min-width:20px;width:auto;margin:0;", row);
    range.type = "range"; range.min = "0"; range.max = "0"; range.value = "0";
    st.range = range;
    range.addEventListener("input", () => setFrame(node, st, Number(range.value)));
    st.label = el("span", "flex:0 1 auto;min-width:0;font:9px monospace;color:#9ecbdd;" +
        "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", row);
    st.label.textContent = "f 1/1";
    st.stats = el("div", `flex:0 0 ${BAR_H}px;width:100%;height:${BAR_H}px;min-height:${BAR_H}px;` +
        `max-height:${BAR_H}px;box-sizing:border-box;overflow:hidden;white-space:nowrap;` +
        `text-overflow:ellipsis;pointer-events:none;text-align:center;font:600 11px system-ui;` +
        `font-variant-numeric:tabular-nums;line-height:${BAR_H}px;color:#9ecbdd;`, parent);
    updateStats(node);
    updateToggle(node);
}

function attachPointer(node, st, panel) {
    const isControl = (t) => !!(t && t.closest && t.closest("button,input,select"));
    panel.addEventListener("pointerdown", (e) => {
        if (isControl(e.target)) return;
        if (e.button !== 0 && e.button !== 1) return;
        if (!st.key) return;
        const r = previewRect(st);
        if (!insideRect(r, e.clientX, e.clientY)) return;
        e.preventDefault();
        e.stopPropagation();
        const x = e.clientX - r.left;
        const lineX = r.width * st.wipe / 100;
        const nearLine = st.mode === "wipe" && Math.abs(x - lineX) <= 14;
        const zoomed = st.zoom > 1.001;
        let kind = "pan";
        if (e.button === 0 && (nearLine || (st.mode === "wipe" && (!zoomed || e.shiftKey)))) kind = "wipe";
        drag.st = st; drag.kind = kind;
        drag.x = e.clientX; drag.y = e.clientY; drag.px = st.panX; drag.py = st.panY;
        st.pointerId = e.pointerId;
        try { panel.setPointerCapture(e.pointerId); } catch (err) {
            console.warn("[AGSoft Image Compare] capture skipped:", err);
        }
        if (kind === "wipe") applyDragMove(st, e.clientX, e.clientY);
    });
    panel.addEventListener("dblclick", (e) => {
        if (isControl(e.target)) return;
        if (!insideRect(previewRect(st), e.clientX, e.clientY)) return;
        e.stopPropagation();
        st.zoom = 1; st.panX = 0; st.panY = 0;
        setWidgetValue(node, "zoom", 1.0);
        setWidgetValue(node, "pan_x", 0);
        setWidgetValue(node, "pan_y", 0);
        applyTransform(st);
    });
}

function createPanel(node) {
    const st = node.__ag;
    const panel = el("div", "display:flex;flex-direction:column;gap:" + GAP + "px;width:100%;" +
        "height:100%;min-height:" + MIN_PANEL_H + "px;box-sizing:border-box;touch-action:none;");
    panel.addEventListener("pointerdown", (e) => e.stopPropagation());
    const preview = el("div", "position:relative;flex:1 1 auto;min-height:0;width:100%;" +
        "box-sizing:border-box;overflow:hidden;border-radius:8px;background:#000;" +
        "pointer-events:none;touch-action:none;", panel);
    buildControls(node, st, panel);
    const widget = node.addDOMWidget("agsoft_panel", "agsoft_panel", panel, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => panelHeight(node),
        getMinHeight: () => MIN_PANEL_H,
        getValue: () => "",
        setValue: () => {},
    });
    widget.serialize = false;
    widget.computeSize = (width) => [width, panelHeight(node)];
    st.panelEl = panel;
    st.previewEl = preview;
    st.panelWidget = widget;
    attachPointer(node, st, panel);
    livePanels.add(st);
}

//------------------------------------------------------------------------------
// Метаданные / metadata
//------------------------------------------------------------------------------
async function refreshMeta(node) {
    const st = node.__ag;
    if (!st || st.fetching) return;
    st.fetching = true;
    try {
        const resp = await fetch(metaUrl(node));
        const meta = await resp.json();
        st.fetching = false;
        const key = meta && meta.key ? meta.key : null;
        st.meta = meta || {};
        if (key !== st.key) {
            st.key = key;
            const pf = st.meta.preview_frames || 1;
            st.frame = clamp(st.frame, 0, Math.max(0, pf - 1));
            if (st.range) { st.range.max = String(Math.max(0, pf - 1)); st.range.value = String(st.frame); }
            render(node);
            updateStats(node);
        }
    } catch (e) {
        st.fetching = false;
        console.warn("[AGSoft Image Compare] meta refresh skipped:", e);
    }
}

//------------------------------------------------------------------------------
// Хуки ноды / node hooks
//------------------------------------------------------------------------------
function hookWidgets(node) {
    for (const name of ["view_mode", "wipe_position", "zoom", "pan_x", "pan_y"]) {
        const wd = findWidget(node, name);
        if (!wd || wd.__agHooked) continue;
        wd.__agHooked = true;
        const prev = wd.callback;
        const oc = prev ? (v) => prev.call(wd, v) : null;
        wd.callback = (v) => {
            if (oc) oc(v);
            safe(() => {
                const st = node.__ag;
                if (name === "wipe_position") { st.wipe = num(v, 50); setWipe(st); }
                else if (name === "zoom" || name === "pan_x" || name === "pan_y") {
                    st.zoom = num(widgetValue(node, "zoom", 1), 1);
                    st.panX = num(widgetValue(node, "pan_x", 0), 0);
                    st.panY = num(widgetValue(node, "pan_y", 0), 0);
                    applyTransform(st);
                } else { render(node); }
            }, "widget callback");
        };
    }
}

function hookNode(node) {
    const st = node.__ag;
    st.baseComputeSize = node.computeSize;
    node.computeSize = function (out) {
        const base = baseSize(node);
        const width = Math.max(base[0], MIN_NODE_WIDTH);
        if (st.vueMode) return [width, base[1] + panelHeight(node)];
        return [width, base[1] + MIN_PANEL_H + BOTTOM_GAP];
    };
    const baseDraw = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
        if (baseDraw) baseDraw.apply(this, arguments);
        st.canvasAt = Date.now();
    };
    const baseResize = node.onResize;
    node.onResize = function (size) {
        if (baseResize) baseResize.apply(this, arguments);
        const height = num(size && size[1], num(node.size[1], 0));
        const base = baseSize(node)[1];
        if (st.vueMode) {
            const prev = num(st.lastHeight, height);
            const delta = height - prev;
            if (Math.abs(delta) > 0.5) st.panelH = Math.max(MIN_PANEL_H, panelHeight(node) + delta);
            st.lastHeight = base + panelHeight(node);
        } else {
            st.panelH = Math.max(MIN_PANEL_H, height - base - BOTTOM_GAP);
            st.lastHeight = height;
        }
        persistUserSize(node);
        syncPreviewBox(node);
    };
    const baseConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        if (baseConfigure) baseConfigure.apply(this, arguments);
        node.properties = node.properties || {};
        st.collapsed = node.properties.agsoft_collapsed === true;
        safe(() => applyVisibility(this), "configure visibility");
        safe(() => updateToggle(this), "configure toggle");
        safe(() => refreshMeta(this), "configure meta");
        safe(() => scheduleRestore(this), "configure restore");
    };
    const baseRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (baseRemoved) baseRemoved.apply(this, arguments);
        safe(() => {
            if (drag.st === st) drag.st = null;
            livePanels.delete(st);
            if (st.poll) clearInterval(st.poll);
            for (const t of st.restoreTimers) clearTimeout(t);
            if (st.controller) unwatchMode(st.controller);
            window.removeEventListener("keydown", st.onKeyDown);
            window.removeEventListener("keyup", st.onKeyUp);
        }, "cleanup");
    };
}

function attach(node) {
    if (node.__ag) return;
    bindGlobal();
    node.properties = node.properties || {};
    const savedPanel = num(node.properties.agsoft_panel_h, 0);
    node.__ag = {
        node: node, panelEl: null, previewEl: null, panelWidget: null, toggleBtn: null,
        imgs: [], imgA: null, imgB: null, layerA: null, master: null, handle: null,
        key: null, meta: {}, mode: "wipe", wipe: 50, zoom: 1, panX: 0, panY: 0, frame: 0,
        peek: false, hover: false, range: null, label: null, stats: null, labelLock: 0,
        fetching: false, poll: null, vueMode: false, ready: false, canvasAt: Date.now(),
        collapsed: node.properties.agsoft_collapsed === true,
        panelH: savedPanel > 0 ? savedPanel : PANEL_DEFAULT_H,
        lastHeight: num(node.size && node.size[1], 0),
        baseComputeSize: null, controller: null, restoreTimers: [], pointerId: undefined,
    };
    const st = node.__ag;
    safe(() => createPanel(node), "panel");
    safe(() => hookWidgets(node), "widget hooks");
    safe(() => hookNode(node), "node hooks");
    safe(() => applyVisibility(node), "visibility");
    safe(() => render(node), "render");
    safe(() => refreshMeta(node), "meta");
    safe(() => scheduleRestore(node), "restore");
    st.poll = setInterval(() => {
        if (!node.graph) { clearInterval(st.poll); return; }
        safe(() => refreshMeta(node), "poll meta");
    }, POLL_MS);
    st.onKeyDown = (e) => {
        if (e.code !== "Space" || !st.hover || e.repeat || !st.key) return;
        e.preventDefault();
        st.peek = true;
        setWipe(st);
        updateSrcs(st);
    };
    st.onKeyUp = (e) => {
        if (e.code !== "Space" || !st.peek) return;
        st.peek = false;
        setWipe(st);
        updateSrcs(st);
    };
    window.addEventListener("keydown", st.onKeyDown);
    window.addEventListener("keyup", st.onKeyUp);
    st.controller = {
        check(now) {
            const setting = vueNodesSetting();
            const vue = setting !== null ? setting : (now - st.canvasAt) > VUE_SILENCE_MS;
            safe(() => setVueMode(node, vue), "mode");
            safe(() => applyVisibility(node), "tick visibility");
        },
    };
    watchMode(st.controller);
}

app.registerExtension({
    name: "AGSoft.ImageCompare",
    setup() {
        bindGlobal();
        api.addEventListener("executed", (e) => {
            safe(() => {
                const id = e.detail ? e.detail.node : null;
                if (id === null || id === undefined) return;
                const n = app.graph && app.graph.getNodeById ? app.graph.getNodeById(Number(id)) : null;
                if (n && n.__ag) refreshMeta(n);
            }, "executed");
        });
        api.addEventListener("execution_success", () => {
            safe(() => {
                const nodes = app.graph && app.graph._nodes ? app.graph._nodes : [];
                for (const n of nodes) { if (n && n.__ag) refreshMeta(n); }
            }, "execution_success");
        });
    },
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS) return;
        if (typeof node.addDOMWidget !== "function") return;
        safe(() => attach(node), "attach");
    },
});