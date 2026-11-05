// ==============================================================================
// AGSoft_H3_Movie_Chain.js
// ==============================================================================
// JS-расширение нод 🎬AGSoft H3 Ref2V Movie Master / 🎬AGSoft H3 Ref2V Movie Link.
// Версия / Version: v11.05
//
// НАЗНАЧЕНИЕ / PURPOSE
// Фронтенд-слой цепи фильма: живая инфострока (только Master), автоскрытие
// виджетов, динамические сокеты рефов и гайдов, схлопывание виджетов
// семплера/декода на Link при подключённой chain_advanced и одноразовая
// санация высоты ноды после скрытия виджетов. Размер ноды после санации
// полностью нативный: JS его больше не пишет.
// Frontend layer of the movie chain: live info line (Master only), widget
// auto-hiding, dynamic ref and guide sockets, sampler/decode widget collapse
// on Link while chain_advanced is linked, and a one-shot node height sanitize
// after widget hiding. After the sanitize the node size is fully native: JS
// never writes it again.
//
// ЧТО ДЕЛАЕТ / WHAT IT DOES
// 1. Живая инфострока / live info line (Master only).
//    DOM-виджет 18 px ставится между последней группой виджетов и полем
//    prompt; формат 1:1 с 🎬AGSoft MiniMax Base: ⚙ W×H • MP • ~формат •
//    кадры • точные секунды (кадры ПЕРЕД секундами, секунды с полной
//    точностью). На Link строки нет: размер и кадры приходят из шины
//    chain_base, виджеты Link их не пересчитывают.
//    An 18 px DOM widget placed between the last widget group and the prompt
//    field; format 1:1 with 🎬AGSoft MiniMax Base: ⚙ W×H • MP • ~aspect •
//    frames • exact seconds (frames BEFORE seconds, seconds at full
//    precision). Link has no line: size and frames come from the chain_base
//    bus, Link widgets do not recompute them.
// 2. Автоскрытие виджетов / widget auto-hiding.
//    Калькулятор: preset / custom_width / custom_height / megapixels_value /
//    aspect_ratio по mode; length_seconds / frame_count по frame_count_source;
//    invert_orientation виден всегда. Гайды: пары seconds_i / frames_i видны
//    только для активных слотов (i <= number_of_guides) и только своего
//    режима позиции. source_audio_start_seconds виден только при подключённом
//    source_audio. Link: steps / cfg / sampler_name / scheduler / denoise /
//    decode_mode / tile_size / overlap / temporal_size / temporal_overlap
//    скрываются, когда подключена chain_advanced — значения едут из шины;
//    отключил шину — виджеты возвращаются для ручной правки сцены.
//    Скрытые виджеты уходят за карту попаданий (y = -9999), поэтому линки
//    встают точно в видимые сокеты; виджеты с подключённым линком не
//    скрываются никогда (как в 🎬AGSoft MiniMax Base).
//    Calculator: preset / custom_width / custom_height / megapixels_value /
//    aspect_ratio by mode; length_seconds / frame_count by frame_count_source;
//    invert_orientation always visible. Guides: seconds_i / frames_i pairs
//    visible only for active slots (i <= number_of_guides) and only for their
//    position mode. source_audio_start_seconds visible only while
//    source_audio is linked. Link: steps / cfg / sampler_name / scheduler /
//    denoise / decode_mode / tile_size / overlap / temporal_size /
//    temporal_overlap hide while chain_advanced is linked — values come from
//    the bus; unlink and the widgets return for manual scene tuning. Hidden
//    widgets leave the drop hit map (y = -9999) so links land on visible
//    sockets; linked widgets are never hidden (same as 🎬AGSoft MiniMax Base).
// 3. Динамические сокеты / dynamic sockets.
//    Рефы (Master): группы image / video / video_audio / audio держат ровно
//    (подключено + 1) слотов, минимум 1, максимум объявленного количества
//    (image 9, video 3, video_audio 3, audio 3 — 1:1 с нативом); пересборка
//    отложена на следующий тик и идёт только через removeInput/addInput,
//    линки восстанавливаются от тех же нод-источников сохранённой ссылкой на
//    метод и call(src, ...). Гайды (обе ноды): image_i существуют для
//    активных слотов; audio_i на Master появляются только при подключённом
//    audio_vae, а на Link доступны ВСЕГДА для активных слотов, потому что
//    audio_vae приходит из шины chain_base и JS его не видит — отсутствие
//    audio_vae в шине ловит Python понятной ошибкой при выполнении.
//    Refs (Master): image / video / video_audio / audio groups keep exactly
//    (connected + 1) slots, min 1, max declared (image 9, video 3,
//    video_audio 3, audio 3 — 1:1 with native); the rebuild is deferred to
//    the next tick and runs only through removeInput/addInput, links are
//    restored from the same origin nodes via a saved method reference and
//    call(src, ...). Guides (both nodes): image_i exist for active slots;
//    audio_i on Master appear only while audio_vae is linked, while on Link
//    they are ALWAYS available for active slots because audio_vae arrives
//    through the chain_base bus and JS cannot see it — a missing audio_vae in
//    the bus is caught by Python with a clear runtime error.
// 4. Высота ноды / node height.
//    Фронтенд считает стартовую высоту ДО скрытия виджетов, и освободившееся
//    место съедает prompt. Поэтому один раз после первого прохода скрытия
//    высота санится: стопка видимых виджетов + 200 px под prompt; сохранённые
//    раздутые высоты (часть под prompt > 600 px) сбрасываются к норме,
//    заниженные (< 60 px) поднимаются до минимума. Дальше размер живёт
//    нативно: ресайз в обе стороны, сохранение и загрузка как у обычных нод,
//    JS его не трогает.
//    The frontend computes the start height BEFORE widgets hide, and the
//    freed space is eaten by prompt. So once, after the first hiding pass,
//    the height is sanitized: visible widget stack + 200 px for prompt; saved
//    inflated heights (prompt part > 600 px) reset to normal, too small ones
//    (< 60 px) raise to the minimum. After that the size lives natively:
//    resize both ways, save and load like plain nodes, JS never touches it.
// 5. Безопасность / safety.
//    Все операции под try/catch; пересборка сокетов не запускается синхронно
//    при создании ноды (линков в этот момент ещё нет); никаких ежекадровых
//    записей размера и DOM-обновлений вне событий.
//    All operations under try/catch; socket rebuild never runs synchronously
//    on node creation (links are not there yet); no per-frame size writes or
//    DOM updates outside events.
//
// Автор / Author: AGSoft
// Дата / Date: 05.11.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";

console.log("[AGSoft H3 Movie Chain] JS extension loaded v11.05 (Link guide audio slots always on)");

const MASTER = "AGSoft_H3_Ref2V_Movie_Master";
const LINK = "AGSoft_H3_Ref2V_Movie_Link";
const MAX_GUIDES = 16;
const GROUP_ORDER = ["image", "video", "vaudio", "audio"];
const HOOK_WIDGETS = ["mode", "preset", "invert_orientation", "custom_width",
    "custom_height", "megapixels_value", "aspect_ratio", "frame_count_source",
    "length_seconds", "frame_count"];
const GUIDE_HOOKS = ["number_of_guides", "position_mode"];
const ADV_WIDGETS = ["steps", "cfg", "sampler_name", "scheduler", "denoise",
    "decode_mode", "tile_size", "overlap", "temporal_size", "temporal_overlap"];
const BAR_H = 18;
const Y_OFF = -9999;
const INPUT_TYPE = 1;
const PROMPT_H0 = 200;
const PROMPT_MIN = 60;
const PROMPT_MAX = 600;
const REF_NAMES = { image: [], video: [], vaudio: [], audio: [] };
for (let i = 0; i < 9; i++) REF_NAMES.image.push("ref_image_" + i);
for (let i = 0; i < 3; i++) REF_NAMES.video.push("ref_video_" + i);
for (let i = 0; i < 3; i++) REF_NAMES.vaudio.push("ref_video_audio_" + i);
for (let i = 0; i < 3; i++) REF_NAMES.audio.push("ref_audio_" + i);

function wget(node, name) {
    const w = (node.widgets || []).find((x) => x.name === name);
    return w ? w.value : null;
}

function wobj(node, name) {
    return (node.widgets || []).find((x) => x.name === name) || null;
}

function inputLinked(node, name) {
    const inp = (node.inputs || []).find((i) => i.name === name);
    return !!(inp && inp.link !== null && inp.link !== undefined);
}

function setHidden(w, hidden) {
    try {
        if (w.hidden !== hidden) w.hidden = hidden;
        if (hidden) w.y = Y_OFF;
        else if (w.y === Y_OFF) w.y = 0;
    } catch (e) { /* ignore / игнор */ }
}

function groupOf(name) {
    if (!name) return null;
    if (name.indexOf("ref_video_audio_") === 0) return "vaudio";
    if (name.indexOf("ref_image_") === 0) return "image";
    if (name.indexOf("ref_video_") === 0) return "video";
    if (name.indexOf("ref_audio_") === 0) return "audio";
    return null;
}

function numOf(name) {
    const m = name.match(/(\d+)$/);
    return m ? parseInt(m[1], 10) : 0;
}

function captureGroups(node) {
    if (node.__ag_groups) return;
    const groups = { image: [], video: [], vaudio: [], audio: [] };
    const types = {};
    for (const inp of node.inputs || []) {
        if (!inp) continue;
        const g = groupOf(inp.name);
        if (!g) continue;
        if (!groups[g].includes(inp.name)) groups[g].push(inp.name);
        types[inp.name] = inp.type;
    }
    for (const g of GROUP_ORDER) groups[g].sort((a, b) => numOf(a) - numOf(b));
    node.__ag_groups = groups;
    node.__ag_types = types;
}

function linkById(node, id) {
    if (id === null || id === undefined) return null;
    const links = node.graph ? node.graph.links : null;
    if (!links) return null;
    return links.get ? (links.get(id) || null) : (links[id] || null);
}

function originOf(node, inp) {
    if (!inp || inp.link === null || inp.link === undefined) return null;
    const lk = linkById(node, inp.link);
    return lk ? { id: lk.origin_id, slot: lk.origin_slot } : null;
}

function findInput(node, name) {
    return (node.inputs || []).findIndex((x) => x.name === name);
}

function ensureInput(node, name, type) {
    if (findInput(node, name) < 0) node.addInput(name, type);
}

function dropInput(node, name) {
    const idx = findInput(node, name);
    if (idx < 0) return;
    try {
        if (node.inputs[idx].link != null) node.disconnectInput(idx);
    } catch (e) { /* ignore / игнор */ }
    node.removeInput(idx);
}

function guideCount(node) {
    return Math.max(0, Math.min(MAX_GUIDES,
        parseInt(wget(node, "number_of_guides"), 10) || 0));
}

// Слоты гайдов как в 📍AGSoft Add Guide: image всегда; audio на Master только
// с подключённым audio_vae, на Link всегда (audio_vae едет из шины chain_base,
// JS его не видит; отсутствие audio_vae в шине ловит Python ошибкой при run).
// Guide slots like 📍AGSoft Add Guide: image always; audio on Master only
// with audio_vae linked, on Link always (audio_vae travels in the chain_base
// bus, JS cannot see it; a missing audio_vae in the bus is caught by Python
// with a runtime error).
function manageGuides(node) {
    const count = guideCount(node);
    const useAudio = node.comfyClass === LINK || inputLinked(node, "audio_vae");
    for (let i = MAX_GUIDES; i > count; i--) {
        dropInput(node, "audio_" + i);
        dropInput(node, "image_" + i);
    }
    if (!useAudio) {
        for (let i = 1; i <= MAX_GUIDES; i++) dropInput(node, "audio_" + i);
    }
    for (let i = 1; i <= count; i++) {
        ensureInput(node, "image_" + i, "IMAGE");
        if (useAudio) ensureInput(node, "audio_" + i, "AUDIO");
    }
}

// Группы рефов держат (подключено+1) слотов; линки восстанавливаются
// сохранённой ссылкой на метод источника и call(src, ...).
// Ref groups keep (connected+1) slots; links restored via a saved method
// reference of the origin node plus call(src, ...).
function normalizeRefs(node) {
    if (!node.graph) return;
    try {
        captureGroups(node);
        let declared = 0;
        for (const g of GROUP_ORDER) declared += (node.__ag_groups[g] || []).length;
        if (!declared) return;
        const current = [];
        for (const inp of node.inputs || []) {
            if (!inp) continue;
            const g = groupOf(inp.name);
            if (!g) continue;
            current.push({ name: inp.name, g: g, origin: originOf(node, inp) });
        }
        const desired = [];
        for (const g of GROUP_ORDER) {
            const names = REF_NAMES[g];
            const cap = names.length;
            if (!cap) continue;
            const conn = current.filter((c) => c.g === g && c.origin !== null);
            const target = Math.min(cap, Math.max(1, conn.length + 1));
            for (let k = 0; k < target; k++) {
                desired.push({ name: names[k], g: g,
                    origin: k < conn.length ? conn[k].origin : null });
            }
        }
        if (current.length === desired.length) {
            let same = true;
            for (let i = 0; i < current.length; i++) {
                const a = current[i], b = desired[i];
                const ao = a.origin, bo = b.origin;
                const eq = a.name === b.name &&
                    ((ao === null && bo === null) ||
                     (ao !== null && bo !== null && ao.id === bo.id && ao.slot === bo.slot));
                if (!eq) { same = false; break; }
            }
            if (same) return;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp && groupOf(inp.name)) node.removeInput(i);
        }
        for (const d of desired) {
            node.addInput(d.name, (node.__ag_types || {})[d.name] ||
                (d.g === "audio" || d.g === "vaudio" ? "AUDIO" : "IMAGE"));
            const idx = node.inputs.length - 1;
            if (d.origin !== null) {
                try {
                    const src = node.graph.getNodeById ?
                        node.graph.getNodeById(d.origin.id) : null;
                    if (src && src.outputs && src.outputs[d.origin.slot]) {
                        const linkFn = src.connect;
                        if (typeof linkFn === "function") {
                            linkFn.call(src, d.origin.slot, node, idx);
                        }
                    }
                } catch (e) {
                    console.warn("[AGSoft H3 Movie Chain] reconnect skipped:", e);
                }
            }
        }
    } catch (e) {
        console.warn("[AGSoft H3 Movie Chain] normalizeRefs skipped:", e);
    }
}

function updateWidgets(node) {
    const mode = wget(node, "mode") || "Preset";
    const src = wget(node, "frame_count_source") || "Seconds";
    const pos = wget(node, "position_mode") || "Seconds";
    const count = guideCount(node);
    const rules = {
        preset: mode === "Preset",
        custom_width: mode === "Custom",
        custom_height: mode === "Custom",
        megapixels_value: mode === "Megapixels",
        aspect_ratio: mode === "Megapixels",
        length_seconds: src === "Seconds",
        frame_count: src === "Frames",
    };
    for (const w of node.widgets || []) {
        if (w.name in rules) {
            if (inputLinked(node, w.name)) { setHidden(w, false); continue; }
            setHidden(w, !rules[w.name]);
            continue;
        }
        const gm = (w.name || "").match(/^(seconds|frames)_(\d+)$/);
        if (gm) {
            const i = parseInt(gm[2], 10);
            const on = i <= count &&
                (gm[1] === "seconds" ? pos === "Seconds" : pos === "Frames");
            setHidden(w, !on);
            continue;
        }
        if (w.name === "source_audio_start_seconds") {
            setHidden(w, !inputLinked(node, "source_audio"));
        }
    }
    // Link: шина chain_advanced подключена — виджеты семплера/декода схлопнуты.
    // Link: chain_advanced linked — sampler/decode widgets collapse to bus.
    if (node.comfyClass === LINK) {
        const advLinked = inputLinked(node, "chain_advanced");
        for (const name of ADV_WIDGETS) {
            const w = wobj(node, name);
            if (w) setHidden(w, advLinked);
        }
    }
}

// Инфострока 1:1 с 🎬AGSoft MiniMax Base / info line 1:1 with 🎬AGSoft MiniMax Base
function fit32(v) {
    return Math.max(32, Math.ceil(v / 32) * 32);
}

function fit17n5(v) {
    let x = Math.max(5, Math.round(Number(v) || 5));
    while (x % 17 !== 5) x += 1;
    return x;
}

const RATIO_CANDIDATES = [[1, 1], [5, 4], [4, 3], [3, 2], [16, 9], [21, 9], [2, 1]];

function approximateRatio(w, h) {
    const target = Math.max(w, h) / Math.max(1, Math.min(w, h));
    let best = RATIO_CANDIDATES[0];
    let bestDiff = Infinity;
    for (const [a, b] of RATIO_CANDIDATES) {
        const d = Math.abs(a / b - target);
        if (d < bestDiff) { bestDiff = d; best = [a, b]; }
    }
    const [a, b] = best;
    return w >= h ? `${a}:${b}` : `${b}:${a}`;
}

function num(v) {
    return Number(v.toFixed(2)).toString();
}

function exactSec(v) {
    return Number(v.toFixed(12)).toString();
}

function computeInfo(node) {
    const mode = wget(node, "mode");
    const invert = wget(node, "invert_orientation");
    let w = 1280, h = 704;
    if (mode === "Preset") {
        const m = String(wget(node, "preset") || "").match(/(\d+)\s*×\s*(\d+)/);
        if (m) { w = parseInt(m[1], 10); h = parseInt(m[2], 10); }
        if (invert) { const t = w; w = h; h = t; }
        w = fit32(w); h = fit32(h);
    } else if (mode === "Custom") {
        w = fit32(wget(node, "custom_width") || 0);
        h = fit32(wget(node, "custom_height") || 0);
        if (invert) { const t = w; w = h; h = t; }
    } else if (mode === "Megapixels") {
        const mp = parseFloat(wget(node, "megapixels_value")) || 1;
        const r = String(wget(node, "aspect_ratio") || "16:9").split(":");
        const wr = parseInt(r[0], 10) || 16, hr = parseInt(r[1], 10) || 9;
        const x = Math.sqrt((mp * 1000000) / (wr * hr));
        w = Math.max(64, fit32(Math.round(wr * x)));
        h = Math.max(64, fit32(Math.round(hr * x)));
        if (invert) { const t = w; w = h; h = t; }
    }
    const src = wget(node, "frame_count_source");
    let total;
    if (src === "Frames") total = fit17n5(wget(node, "frame_count") || 5);
    else total = fit17n5(Math.round((parseFloat(wget(node, "length_seconds")) || 0) * 24));
    total = Math.max(5, total);
    const sec = exactSec(total / 24);
    const mp = num((w * h) / 1000000) + "MP";
    return `⚙ ${w}×${h} • ${mp} • ~${approximateRatio(w, h)} • ${total}f • ${sec}s`;
}

// Строка ставится ПЕРЕД prompt и только на Master / bar BEFORE prompt, Master only.
function ensureBarWidget(node) {
    if (node.__ag_bar_w) return node.__ag_bar_w;
    const div = document.createElement("div");
    div.style.cssText = "width:100%;height:" + BAR_H + "px;line-height:" + BAR_H +
        "px;text-align:center;background:#263238;color:#9ecbdd;font:10px monospace;" +
        "border-radius:8px;overflow:hidden;white-space:nowrap;box-sizing:border-box;" +
        "pointer-events:none;";
    div.textContent = computeInfo(node);
    let w = null;
    try {
        w = node.addDOMWidget("agsoft_info_bar", "div", div, { serialize: false });
    } catch (e) {
        console.warn("[AGSoft H3 Movie Chain] addDOMWidget failed:", e);
        return null;
    }
    w.computeSize = function (width) { return [width || 0, BAR_H]; };
    node.__ag_bar_w = w;
    node.__ag_bar_el = div;
    ensureBarBeforePrompt(node);
    return w;
}

function ensureBarBeforePrompt(node) {
    const w = node.__ag_bar_w;
    if (!w) return;
    const ws = node.widgets || [];
    const bi = ws.indexOf(w);
    const pi = ws.findIndex((x) => x.name === "prompt");
    if (bi > -1 && pi > -1 && bi > pi) {
        ws.splice(bi, 1);
        ws.splice(pi, 0, w);
    }
}

function updateBarText(node) {
    if (!node.__ag_bar_el) return;
    const t = computeInfo(node);
    if (node.__ag_bar_el.textContent !== t) node.__ag_bar_el.textContent = t;
}

function attachInfoBar(node) {
    if (node.__ag_bar) return;
    node.__ag_bar = true;
    if (node.properties) {
        delete node.properties.__ag_bar_reserved;
        delete node.properties.__ag_bar_bottom;
        delete node.properties.agsoft_user_h;
        delete node.properties.agsoft_prompt_h;
        delete node.properties.agsoft_node_h;
        delete node.properties.agsoft_node_w;
        delete node.properties.agsoft_size;
    }
    ensureBarWidget(node);
}

// Текущая высота самого поля prompt по его computeSize / current prompt height.
function promptHeightOf(node) {
    const pw = wobj(node, "prompt");
    if (!pw) return 0;
    try {
        const cs = pw.computeSize ? pw.computeSize(100) : null;
        if (cs && cs[1]) return cs[1];
    } catch (e) { /* ignore / игнор */ }
    return pw.computedHeight || pw.height || 0;
}

// Одноразовая нормализация высоты после первого прохода скрытия виджетов.
// One-shot height normalize after the first widget-hiding pass.
function fixHeightOnce(node) {
    if (node.__ag_sized) return;
    node.__ag_sized = true;
    try {
        const cs = node.computeSize();
        if (!cs || !cs[1]) return;
        const h0 = promptHeightOf(node);
        const stack = Math.max(0, cs[1] - h0);
        const normal = stack + PROMPT_H0;
        const cur = node.size && node.size[1] ? node.size[1] : 0;
        let target = cur;
        if (!cur) target = normal;
        else if (cur > stack + PROMPT_MAX) target = normal;
        else if (cur < stack + PROMPT_MIN) target = stack + PROMPT_MIN;
        if (target !== cur && node.size) {
            node.size[0] = node.size[0] || cs[0];
            node.size[1] = target;
            node.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.warn("[AGSoft H3 Movie Chain] fixHeightOnce skipped:", e);
    }
}

function refresh(node, allowNormalize) {
    try {
        updateWidgets(node);
        if (allowNormalize !== false) {
            manageGuides(node);
            normalizeRefs(node);
        }
        if (node.comfyClass === MASTER) {
            ensureBarBeforePrompt(node);
            updateBarText(node);
        }
        node.setDirtyCanvas(true, true);
    } catch (e) {
        console.warn("[AGSoft H3 Movie Chain] refresh skipped:", e);
    }
}

function scheduleRefresh(node) {
    if (node.__ag_pending) return;
    node.__ag_pending = true;
    setTimeout(() => {
        node.__ag_pending = false;
        if (!node.inputs) return;
        refresh(node, true);
    }, 0);
}

app.registerExtension({
    name: "AGSoft.H3MovieChain",
    async nodeCreated(node) {
        if (node.comfyClass !== MASTER && node.comfyClass !== LINK) return;
        if (node.comfyClass === MASTER) attachInfoBar(node);
        refresh(node, false);
        for (const name of HOOK_WIDGETS) {
            const w = wobj(node, name);
            if (!w) continue;
            const prev = w.callback;
            const oc = prev ? (v) => prev.call(w, v) : null;
            w.callback = (v) => {
                if (oc) oc(v);
                updateWidgets(node);
                if (node.comfyClass === MASTER) updateBarText(node);
                node.setDirtyCanvas(true, true);
            };
        }
        for (const name of GUIDE_HOOKS) {
            const w = wobj(node, name);
            if (!w) continue;
            const prev = w.callback;
            const oc = prev ? (v) => prev.call(w, v) : null;
            w.callback = (v) => {
                if (oc) oc(v);
                scheduleRefresh(node);
            };
        }
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            if (origConn) origConn.apply(this, arguments);
            if (type === INPUT_TYPE) scheduleRefresh(this);
        };
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            if (this.comfyClass === MASTER) attachInfoBar(this);
            scheduleRefresh(this);
        };
        setTimeout(() => {
            if (!node.inputs) return;
            scheduleRefresh(node);
            setTimeout(() => fixHeightOnce(node), 0);
        }, 0);
    },
});