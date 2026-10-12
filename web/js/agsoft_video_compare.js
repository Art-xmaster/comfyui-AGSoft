// ==============================================================================
// agsoft_video_compare.js
// ==============================================================================
// JS extension for the 🎬AGSoft Video Compare node. Version: v10.28
// Author / Автор: AGSoft
// Date / Дата: 12.10.2026
// ------------------------------------------------------------------------------
// Description / Описание:
//
// EN:
// Frontend companion of the comparison node: a DOM widget embedded into the
// node body that plays the server-encoded preview streams and lets the user
// judge the difference visually. Three view modes are supported. In wipe mode
// two <video> elements (stream B underneath, stream A on top) are stacked and
// the top one is clipped with clip-path inset by the wipe percentage, while a
// vertical divider line can be dragged with mouse or touch directly over the
// viewport: the clip and the divider follow the pointer in real time without
// any node recompute, and on release the value is written back into the
// wipe_position widget. In side_by_side mode a single <video> plays stream S
// (the server-side horizontal concat of full frames B|A), so both clips are
// always visible in full with no client-side compositing. In difference mode
// a single <video> plays stream D (the amplified per-frame difference).
// Below the viewport sits one shared controls row: play/pause for all loaded
// streams at once, minus/plus one frame buttons stepping by the metadata fps,
// a 0..1000 range timeline that seeks every stream together, a time label in
// "current / duration s" format, and a save button. The save button captures
// the currently displayed comparison frame into a canvas (in wipe mode it
// draws stream B fully and then stream A clipped at the divider position; in
// the other modes it draws the master stream as is), converts the canvas to a
// PNG blob and posts it to the loopback-only server endpoint; on success the
// time label is temporarily replaced by "saved: <file name>" for two seconds
// (label lock), after which the normal time readout resumes.
// Playback synchronization keeps one master stream and snaps every follower
// whose currentTime drifts more than 0.06 s back to the master; the sync, the
// time label and the timeline thumb run inside a single requestAnimationFrame
// loop shared by all comparison nodes, so no per-frame DOM writes happen
// besides the label text. Cache metadata (key, fps, frames, width, height,
// duration) is fetched from the server by an independent 1.5 s poll per node,
// additionally triggered by graph execution events and by workflow configure;
// until a key arrives the viewport shows a bilingual "run the graph"
// placeholder. Every re-render stops and detaches the previous streams first.
// Layout: the viewport and the controls row have fixed pixel heights, the DOM
// widget reports their sum through computeSize, and after creation and after
// workflow load the node height is recomputed and expanded (never shrunk) so
// the controls row is never clipped by the node border; the controls row
// itself shrinks gracefully (slider down to 20 px, label with ellipsis) to fit
// any node width. Widget callbacks are wrapped with the previous callback
// preserved and invoked first, so view_mode switches rebuild the viewport and
// wipe_position changes move the divider without a rebuild.
// ---
// RU:
// Фронтенд-спутник ноды сравнения: DOM-виджет внутри тела ноды, который
// проигрывает закодированные сервером превью-потоки и даёт пользователю
// визуально оценить разницу. Поддерживаются три режима просмотра. В режиме
// wipe два элемента <video> (поток B снизу, поток A сверху) наложены друг на
// друга, верхний обрезан через clip-path inset по проценту wipe, а
// вертикальную линию-разделитель можно тянуть мышью или тачем прямо по
// области просмотра: обрезка и линия следуют за указателем в реальном времени
// без пересчёта ноды, а при отпускании значение записывается обратно в виджет
// wipe_position. В режиме side_by_side один <video> проигрывает поток S
// (серверная горизонтальная склейка полных кадров B|A), поэтому оба ролика
// всегда видны целиком без какой-либо клиентской склейки. В режиме difference
// один <video> проигрывает поток D (усиленная покадровая разница). Под
// областью просмотра расположена одна общая строка управления: play/pause
// сразу для всех загруженных потоков, кнопки минус/плюс один кадр с шагом по
// fps из метаданных, таймлайн-range 0..1000, ищущий все потоки одновременно,
// метка времени в формате «текущее / длительность s» и кнопка сохранения.
// Кнопка сохранения снимает текущий отображаемый кадр сравнения в canvas (в
// режиме wipe рисуется поток B целиком, затем поток A, обрезанный по позиции
// разделителя; в остальных режимах рисуется master-поток как есть),
// конвертирует canvas в PNG-blob и отправляет его на loopback-only эндпоинт
// сервера; при успехе метка времени на две секунды заменяется на
// «saved: <имя файла>» (блокировка метки), после чего обычный отсчёт времени
// возвращается. Синхронизация воспроизведения держит один master-поток и
// подтягивает к нему каждый follower, чей currentTime ушёл более чем на
// 0.06 s; синхронизация, метка времени и ползунок таймлайна работают внутри
// одного цикла requestAnimationFrame, общего для всех нод сравнения, поэтому
// покадровых записей в DOM, кроме текста метки, не происходит. Метаданные
// кэша (key, fps, frames, width, height, duration) запрашиваются с сервера
// независимым поллингом раз в 1.5 s на ноду, дополнительно по событиям
// выполнения графа и при конфигурировании workflow; пока ключ не получен, в
// области просмотра висит двуязычная заглушка «запустите граф». Каждый
// рендер сначала останавливает и отключает предыдущие потоки. Раскладка:
// область просмотра и строка управления имеют фиксированные пиксельные
// высоты, DOM-виджет сообщает их сумму через computeSize, а после создания и
// после загрузки workflow высота ноды пересчитывается и расширяется (никогда
// не сжимается), чтобы строка управления не обрезалась границей ноды; сама
// строка управления gracefully сжимается (слайдер до 20 px, метка с
// многоточием), помещаясь в любую ширину ноды. Колбэки виджетов обёрнуты с
// сохранением и первоочередным вызовом предыдущего колбэка, поэтому смена
// view_mode пересобирает область просмотра, а изменение wipe_position
// двигает разделитель без пересборки.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
console.log("[AGSoft Video Compare] JS extension loaded v10.28 (3 modes, save-frame button)");

const VIEW_H = 240;   // высота области просмотра / view area height
const CTRL_H = 30;    // высота панели управления / controls row height
const POLL_MS = 1500; // шаг поллинга meta / meta polling step

function base() { return (api && api.api_base) ? api.api_base : ""; }
function el(tag, css, parent) {
    const d = document.createElement(tag);
    if (css) d.style.cssText = css;
    if (parent) parent.appendChild(d);
    return d;
}
function streamUrl(key, side) {
    return base() + `/agsoft/video_compare/stream?key=${key}&side=${side}`;
}
function metaUrl(node) {
    return base() + `/agsoft/video_compare/meta?node=${String(node.id)}`;
}
function widgetValue(node, name, dflt) {
    const w = node.widgets ? node.widgets.find((x) => x.name === name) : null;
    return w ? w.value : dflt;
}
// Подгонка высоты ноды под виджет: только расширение, никогда не сжатие.
// Fit the node height to the widget: expand only, never shrink.
function refit(node) {
    try {
        if (!node.computeSize || !node.setSize) return;
        const cs = node.computeSize();
        if (!cs || cs.length !== 2 || !isFinite(cs[1])) return;
        const w = (node.size && node.size[0]) ? node.size[0] : cs[0];
        const h = Math.max(cs[1], (node.size && node.size[1]) ? node.size[1] : 0);
        if (!node.size || Math.abs(node.size[1] - h) > 1 || Math.abs(node.size[0] - w) > 1) {
            node.setSize([w, h]);
            node.setDirtyCanvas(true, true);
        }
    } catch (e) {}
}
function stretchHeight(node) {
    const st = node.__ag;
    if (!st || !node.size || !node.computeSize) return;
    try {
        const baseW = VIEW_H + CTRL_H + 2;
        const cs = node.computeSize();
        if (!cs || cs.length !== 2 || !isFinite(cs[1])) return;
        const other = cs[1] - baseW;
        const extra = Math.max(0, Math.round(node.size[1] - other - baseW));
        if (extra !== st.extra) {
            st.extra = extra;
            if (st.div) st.div.style.height = (baseW + extra) + "px";
            if (st.viewEl) st.viewEl.style.height = (VIEW_H + extra) + "px";
            node.setDirtyCanvas(true, true);
        }
    } catch (e) {}
}
function setWipeClip(state) {
    if (state.vA) state.vA.style.clipPath = `inset(0 0 0 ${state.wipe}%)`;
    if (state.handle) state.handle.style.left = `${state.wipe}%`;
}
// Сохранение текущего кадра сравнения: canvas по активному режиму, PNG в output.
// Save the current comparison frame: canvas per active mode, PNG into output.
function saveFrame(node, state) {
    const mode = widgetValue(node, "view_mode", "wipe");
    let canvas = null;
    try {
        if (mode === "wipe" && state.vA && state.vB) {
            const w = state.vB.videoWidth, h = state.vB.videoHeight;
            if (!w || !h) return;
            canvas = document.createElement("canvas");
            canvas.width = w; canvas.height = h;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(state.vB, 0, 0, w, h);
            const x = Math.round(w * state.wipe / 100);
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, 0, w - x, h);
            ctx.clip();
            ctx.drawImage(state.vA, 0, 0, w, h);
            ctx.restore();
        } else if (state.master && state.master.videoWidth) {
            const w = state.master.videoWidth, h = state.master.videoHeight;
            if (!w || !h) return;
            canvas = document.createElement("canvas");
            canvas.width = w; canvas.height = h;
            canvas.getContext("2d").drawImage(state.master, 0, 0, w, h);
        }
    } catch (e) {
        console.warn("[AGSoft Video Compare] save frame canvas skipped:", e);
        return;
    }
    if (!canvas) return;
    canvas.toBlob((blob) => {
        if (!blob) return;
        fetch(base() + `/agsoft/video_compare/save_frame?mode=${mode}&node=${String(node.id)}`,
            { method: "POST", body: blob })
            .then((r) => r.json())
            .then((j) => {
                if (state.label && j && j.name) {
                    state.label.textContent = "saved: " + j.name;
                    state.labelLock = Date.now() + 2000;
                }
            })
            .catch((e) => console.warn("[AGSoft Video Compare] save frame failed:", e));
    }, "image/png");
}
function syncFollowers(state) {
    // Подгоняем все видео к master без ежекадровых DOM-записей кроме label.
    // Snap all videos to master; no per-frame DOM writes except the label.
    if (!state.master) return;
    const t = state.master.currentTime;
    for (const v of state.vids) {
        if (v !== state.master && Math.abs(v.currentTime - t) > 0.06) v.currentTime = t;
    }
    if (state.label && (!state.labelLock || Date.now() > state.labelLock)) {
        const dur = state.meta && state.meta.duration ? state.meta.duration : (state.master.duration || 0);
        const txt = `${t.toFixed(2)} / ${dur.toFixed(2)}s`;
        if (state.label.textContent !== txt) state.label.textContent = txt;
    }
    if (state.range && !state.rangeDrag && state.master.duration) {
        state.range.value = String(Math.round((t / state.master.duration) * 1000));
    }
}
function seekAll(state, t) {
    for (const v of state.vids) { try { v.currentTime = Math.max(0, t); } catch (e) {} }
}
function playToggle(state) {
    if (!state.master) return;
    if (state.master.paused) {
        for (const v of state.vids) { try { v.play(); } catch (e) {} }
    } else {
        for (const v of state.vids) { try { v.pause(); } catch (e) {} }
    }
}
function stepFrames(state, k) {
    if (!state.master) return;
    const fps = state.meta && state.meta.fps ? state.meta.fps : 24;
    for (const v of state.vids) { try { v.pause(); } catch (e) {} }
    seekAll(state, state.master.currentTime + k / fps);
}
function makeVideo(state, side, css, parent) {
    const v = el("video", css, parent);
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.src = streamUrl(state.key, side);
    state.vids.push(v);
    return v;
}
function buildControls(node, state, parent) {
    // Строка умещается в любую ширину ноды: слайдер и метка сжимаются,
    // кнопки компактные, box-sizing и padding по краям.
    // The row fits any node width: slider and label shrink, buttons are
    // compact, box-sizing and side padding.
    const row = el("div", `display:flex;align-items:center;gap:4px;height:${CTRL_H}px;` +
        `width:100%;box-sizing:border-box;padding:0 2px;flex-wrap:nowrap;overflow:hidden;`, parent);
    const btn = (txt, fn, title) => {
        const b = el("button", `flex:0 0 auto;font:10px monospace;color:#9ecbdd;background:#263238;` +
            `border:1px solid #37474f;border-radius:6px;padding:1px 6px;cursor:pointer;`, row);
        b.textContent = txt;
        if (title) b.title = title;
        b.onclick = () => { try { fn(); } catch (e) {} };
        return b;
    };
    btn("▶/❚❚", () => playToggle(state));
    btn("−1f", () => stepFrames(state, -1));
    btn("+1f", () => stepFrames(state, 1));
    btn("💾", () => saveFrame(node, state),
        "save current comparison frame to output / сохранить текущий кадр сравнения в output");
    const range = el("input", `flex:1 1 20px;min-width:20px;width:auto;margin:0;`, row);
    range.type = "range"; range.min = "0"; range.max = "1000"; range.value = "0";
    state.range = range;
    range.addEventListener("pointerdown", () => { state.rangeDrag = true; });
    window.addEventListener("pointerup", () => { state.rangeDrag = false; });
    range.addEventListener("input", () => {
        if (!state.master || !state.master.duration) return;
        seekAll(state, (Number(range.value) / 1000) * state.master.duration);
    });
    state.label = el("span", `flex:0 1 auto;min-width:0;font:9px monospace;color:#9ecbdd;` +
        `white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`, row);
    state.label.textContent = "0.00 / 0.00s";
}
function render(node) {
    const state = node.__ag;
    if (!state || !state.host) return;
    // Остановить старые потоки / stop old streams
    for (const v of state.vids) { try { v.pause(); v.removeAttribute("src"); v.load(); } catch (e) {} }
    state.vids = []; state.master = null; state.vA = null; state.vB = null; state.handle = null;
    state.host.textContent = "";
    if (!state.key) {
        const ph = el("div", `width:100%;height:${VIEW_H + state.extra}px;display:flex;align-items:center;justify-content:center;` +
            `font:11px monospace;color:#78909c;`, state.host);
        ph.textContent = "run the graph to preview / запустите граф для превью";
        state.viewEl = ph;
        return;
    }
    const mode = widgetValue(node, "view_mode", "wipe");
    const view = el("div", `position:relative;width:100%;height:${VIEW_H + state.extra}px;background:#000;overflow:hidden;`, state.host);
    state.viewEl = view;
    const fill = `position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;`;
    if (mode === "wipe") {
        const vB = makeVideo(state, "B", fill + "z-index:1;", view);
        const vA = makeVideo(state, "A", fill + "z-index:2;", view);
        state.vB = vB; state.vA = vA; state.master = vB;
        state.wipe = Number(widgetValue(node, "wipe_position", 50));
        const handle = el("div", `position:absolute;top:0;bottom:0;width:2px;background:#7fd4ff;` +
            `z-index:3;pointer-events:none;`, view);
        state.handle = handle;
        setWipeClip(state);
        // Клиентский драг разделителя без пересчёта / client-side divider drag
        view.addEventListener("pointerdown", (e) => {
            const move = (ev) => {
                const r = view.getBoundingClientRect();
                const pct = Math.max(0, Math.min(100, ((ev.clientX - r.left) / r.width) * 100));
                state.wipe = Math.round(pct);
                setWipeClip(state);
                const w = node.widgets ? node.widgets.find((x) => x.name === "wipe_position") : null;
                if (w) w.value = state.wipe;
            };
            move(e);
            const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up);
        });
    } else if (mode === "side_by_side") {
        // ОДИН серверный поток S: полные кадры B|A, ничего не режется на клиенте.
        // ONE server stream S: full frames B|A, nothing is cut client-side.
        state.master = makeVideo(state, "S", fill + "z-index:1;", view);
    } else {
        state.master = makeVideo(state, "D", fill + "z-index:1;", view);
    }
    buildControls(node, state, state.host);
}
async function refreshMeta(node) {
    const state = node.__ag;
    if (!state || state.fetching) return;
    state.fetching = true;
    try {
        const resp = await fetch(metaUrl(node));
        const meta = await resp.json();
        state.fetching = false;
        const key = meta && meta.key ? meta.key : null;
        state.meta = meta || {};
        if (key !== state.key) {
            state.key = key;
            render(node);
        }
    } catch (e) {
        state.fetching = false;
        console.warn("[AGSoft Video Compare] meta refresh skipped:", e);
    }
}
function attach(node) {
    if (node.__ag) return;
    const div = document.createElement("div");
    div.style.cssText = "width:100%;background:#101418;border-radius:8px;overflow:hidden;";
    const host = el("div", "", div);
    let w = null;
    try {
        w = node.addDOMWidget("agsoft_preview", "div", div, { serialize: false });
    } catch (e) {
        console.warn("[AGSoft Video Compare] addDOMWidget failed:", e);
        return;
    }
    w.computeSize = function (width) { return [width || 0, VIEW_H + CTRL_H + 6]; };
    node.__ag = { host: host, vids: [], master: null, vA: null, vB: null, key: null, meta: {},
                  wipe: 50, fetching: false, poll: null, labelLock: 0, extra: 0, viewEl: null, div: div };
    render(node);
    refreshMeta(node);
    refit(node);
    stretchHeight(node);
    setTimeout(() => { refit(node); stretchHeight(node); }, 0);
    // Независимый поллинг meta: не полагаемся на WS-события фронтенда.
    // Independent meta polling: do not rely on frontend WS events.
    node.__ag.poll = setInterval(() => {
        if (!node.graph) { clearInterval(node.__ag.poll); return; }
        refreshMeta(node);
    }, POLL_MS);
    // Хуки виджетов / widget hooks
    for (const name of ["view_mode", "wipe_position"]) {
        const wd = node.widgets ? node.widgets.find((x) => x.name === name) : null;
        if (!wd) continue;
        const prev = wd.callback;
        const oc = prev ? (v) => prev.call(wd, v) : null;
        wd.callback = (v) => {
            if (oc) oc(v);
            try {
                const st = node.__ag;
                if (name === "wipe_position" && st && st.vA) {
                    st.wipe = Number(v);
                    setWipeClip(st);
                } else {
                    render(node);
                }
            } catch (e) {}
        };
    }
    const origOnResize = node.onResize;
    node.onResize = function () {
        if (origOnResize) origOnResize.apply(this, arguments);
        stretchHeight(this);
    };
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        try {
            refreshMeta(this);
            refit(this);
            stretchHeight(this);
            setTimeout(() => { refit(this); stretchHeight(this); }, 0);
        } catch (e) {}
    };
}
app.registerExtension({
    name: "AGSoft.VideoCompare",
    setup() {
        // После выполнения графа ключ кэша мог смениться / after exec the cache key may change
        api.addEventListener("executed", (e) => {
            try {
                const id = e.detail ? e.detail.node : null;
                if (id === null || id === undefined) return;
                const n = app.graph && app.graph.getNodeById ? app.graph.getNodeById(Number(id)) : null;
                if (n && n.__ag) refreshMeta(n);
            } catch (err) {}
        });
        api.addEventListener("execution_success", () => {
            try {
                const nodes = app.graph && app.graph._nodes ? app.graph._nodes : [];
                for (const n of nodes) {
                    if (n && n.__ag) refreshMeta(n);
                }
            } catch (err) {}
        });
        // Единый rAF-цикл синхронизации для всех нод / single rAF sync loop for all nodes
        const tick = () => {
            try {
                const nodes = app.graph && app.graph._nodes ? app.graph._nodes : [];
                for (const n of nodes) {
                    if (n && n.__ag && n.__ag.master) syncFollowers(n.__ag);
                }
            } catch (e) {}
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoft_VideoCompare") return;
        attach(node);
    },
});