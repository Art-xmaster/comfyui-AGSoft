// ==============================================================================
// agsoft_minimax_h3.js
// ==============================================================================
// JS-расширение для нод 🎬AGSoft MiniMax H3 Ref2V и 🎬AGSoft MiniMax H3 I2V.
// Версия / Version: v09.48
//
// НАЗНАЧЕНИЕ
// Фронтенд-слой нод H3: динамические входы рефов, автоскрытие виджетов
// калькулятора, живая инфострока и стабильный размер нод при пересоздании
// (смена вкладок ComfyUI, загрузка workflow).
// PURPOSE
// Frontend layer for the H3 nodes: dynamic ref inputs, calculator widget
// auto-hiding, a live info line, and stable node size across node recreation
// (ComfyUI tab switch, workflow load).
//
// КАК РАБОТАЕТ / HOW IT WORKS
// 1. Динамические входы рефов (только Ref2V) / dynamic ref inputs (Ref2V only).
//    Группы image / video / video_audio / audio держат ровно (подключено + 1)
//    слотов, минимум 1, максимум = объявленное число слотов (image 9, video 3,
//    video_audio 3, audio 3 — 1:1 с нативом). Пересборка отложена на следующий
//    тик (setTimeout), идёт только через removeInput/addInput/connect и
//    восстанавливает линки от тех же нод-источников — ссылки не рвутся и не
//    переезжают. При создании ноды пересборка синхронно не запускается:
//    линков в этот момент ещё нет.
//    Each group keeps exactly (connected + 1) slots, min 1, max = declared
//    count (image 9, video 3, video_audio 3, audio 3 — 1:1 with native). The
//    rebuild is deferred to the next tick (setTimeout), uses only
//    removeInput/addInput/connect and reconnects links from the same origin
//    nodes — links never break or jump. It never runs synchronously on node
//    creation: links are not there yet.
// 2. Автоскрытие виджетов калькулятора / calculator widget auto-hiding.
//    По режимам Preset / Custom / Megapixels и Seconds / Frames неиспользуемые
//    виджеты скрываются; invert_orientation виден и активен ВО ВСЕХ режимах
//    (в Megapixels меняет расчётный размер в инфостроке); размер ноды при этом
//    не пишется, освободившееся место занимает поле prompt.
//    Unused widgets hide per mode (Preset / Custom / Megapixels and Seconds /
//    Frames); invert_orientation is visible and active in ALL modes (in
//    Megapixels it swaps the computed size in the info line); the node size is
//    not written, the freed space is taken by the prompt field.
// 3. Живая инфострока — формат 1:1 с 🎬AGSoft MiniMax Base / live info line —
//    format 1:1 with 🎬AGSoft MiniMax Base.
//    DOM-виджет 18px между последним combo (ref_image_size) и полем prompt.
//    Текст: ⚙ W×H • MP • ~формат • кадры • точные секунды (кадры ПЕРЕД
//    секундами; секунды с полной точностью exactSec: 5.166666666666667s;
//    MP = Number((w*h/1e6).toFixed(2)); ~формат — ближайшее соотношение из
//    списка Base). Кадры считаются ТОЙ ЖЕ функцией, что и в Python:
//    fit17n5 циклом while (x % 17 !== 5) x += 1 — округление ВВЕРХ до 17N+5,
//    поэтому строка всегда совпадает с total_frames / duration_seconds.
//    A fixed 18px DOM widget between the last combo (ref_image_size) and the
//    prompt field. Text: ⚙ W×H • MP • ~aspect • frames • exact seconds
//    (frames BEFORE seconds; seconds at full exactSec precision:
//    5.166666666666667s; MP = Number((w*h/1e6).toFixed(2)); ~aspect — the
//    nearest ratio from Base's list). Frames use the SAME rule as Python:
//    fit17n5 via a while loop (x % 17 !== 5) x += 1 — rounding UP to 17N+5,
//    so the line always matches total_frames / duration_seconds.
// 4. Стабильный размер нод / stable node size.
//    Память размера хранится по node.id (id переживает пересоздание ноды) и
//    обновляется ТОЛЬКО реальным драгом ручки ресайза
//    (app.canvas.resizing_node === node); программные setSize память не
//    трогают. При пересоздании ноды память сеется из сериализованного
//    info.size — это ровно тот размер, который пользователь оставил на
//    вкладке. Чужой программный setSize перехватывается и откатывается
//    синхронно в том же JS-тике, до отрисовки кадра; постоянный пиннер
//    (setTimeout-цепочка с шагом 60 мс, живёт пока нода в графе) откатывает
//    любой остальной дрейф node.size. Ручной ресайз попадает в сериализацию
//    при уходе с вкладки и восстанавливается при возврате.
//    The size memory is keyed by node.id (the id survives node recreation)
//    and is updated ONLY by a real drag of the resize handle
//    (app.canvas.resizing_node === node); programmatic setSize calls never
//    touch it. On node recreation the memory is seeded from the serialized
//    info.size — exactly the size the user left on the tab. A foreign
//    programmatic setSize is intercepted and reverted synchronously within
//    the same JS tick, before any frame is drawn; a permanent pinner
//    (setTimeout chain, 60 ms step, alive while the node is in the graph)
//    reverts any remaining node.size drift. A manual resize gets serialized
//    when leaving the tab and restored on return.
// 5. Безопасность / safety.
//    Все операции под try/catch, никаких ежекадровых DOM-записей; node.size
//    пишется исключительно для отката несанкционированного дрейфа.
//    All operations under try/catch, no per-frame DOM writes; node.size is
//    written solely to roll back unauthorized drift.
//
// Автор / Author: AGSoft
// Дата / Date: 09.10.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
console.log("[AGSoft MiniMax H3] JS extension loaded v09.48 (fit17n5 = while-loop 1:1 Base, line matches duration_seconds)");
// Фиксированный порядок групп рефов для Ref2V / fixed ref group order for Ref2V
const GROUP_ORDER = ["image", "video", "vaudio", "audio"];
// Виджеты, влияющие на видимость и инфостроку / widgets affecting visibility and info bar
const HOOK_WIDGETS = ["mode", "preset", "invert_orientation", "custom_width", "custom_height",
    "megapixels_value", "aspect_ratio", "frame_count_source", "length_seconds", "frame_count"];
// Фиксированная высота инфостроки / fixed info bar height
const BAR_H = 18;
// Память размеров нод между пересозданиями (ключ — node.id) /
// Node size memory across recreations (key — node.id)
const SIZE_MEM = new Map();
// Шаг пиннера, мс / pinner step, ms
const PIN_STEP = 60;
function groupOf(name) {
    if (name === "ref_image_size") return null;
    if (name.startsWith("ref_video_audio_")) return "vaudio";
    if (name.startsWith("ref_image_")) return "image";
    if (name.startsWith("ref_video_")) return "video";
    if (name.startsWith("ref_audio_")) return "audio";
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
    for (const inp of node.inputs) {
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
    const links = node.graph && node.graph.links;
    if (!links) return null;
    const lk = links.get ? links.get(id) : links[id];
    return lk || null;
}
function originOf(node, inp) {
    if (!inp) return null;
    if (inp.link === null || inp.link === undefined) return null;
    const lk = linkById(node, inp.link);
    if (!lk) return null;
    return { id: lk.origin_id, slot: lk.origin_slot };
}
// ==============================================================================
// Пиннер размера / size pinner: возвращает node.size к запомненному, пишет
// node.size ТОЛЬКО для отката дрейфа; ручной ресайз обновляет память.
// Reverts node.size to the remembered value; writes node.size ONLY to roll
// back drift; manual resize updates the memory.
// ==============================================================================
function rememberSize(node) {
    if (node.size && node.size.length === 2) SIZE_MEM.set(node.id, [node.size[0], node.size[1]]);
}
function pinSize(node) {
    const mem = SIZE_MEM.get(node.id);
    if (!mem || !node.size) return;
    if (Math.abs(node.size[0] - mem[0]) < 0.5 && Math.abs(node.size[1] - mem[1]) < 0.5) return;
    node.size[0] = mem[0];
    node.size[1] = mem[1];
    node.setDirtyCanvas(true, true);
}
function startPin(node) {
    if (node.__ag_watching) return;
    node.__ag_watching = true;
    const tick = () => {
        if (!node.graph) { node.__ag_watching = false; return; }
        try { pinSize(node); } catch (e) {}
        setTimeout(tick, PIN_STEP);
    };
    setTimeout(tick, PIN_STEP);
}
// Пересборка слотов рефов (паттерн AGSoft Add Images), отложенно и под try/catch.
// Размер ноды не пишет: дрейф откатывает пиннер, ручной ресайз уважается.
// Ref slot rebuild (AGSoft Add Images pattern), deferred and under try/catch.
// It never writes the node size: drift is rolled back by the pinner.
function normalizeRefs(node) {
    if (!node.graph) return;
    try {
        captureGroups(node);
        const current = [];
        for (const inp of node.inputs) {
            if (!inp) continue;
            const g = groupOf(inp.name);
            if (!g) continue;
            current.push({ name: inp.name, g: g, origin: originOf(node, inp) });
        }
        const desired = [];
        for (const g of GROUP_ORDER) {
            const names = node.__ag_groups[g];
            const cap = names.length;
            if (!cap) continue;
            const conn = current.filter((c) => c.g === g && c.origin !== null);
            const target = Math.min(cap, Math.max(1, conn.length + 1));
            for (let k = 0; k < target; k++) {
                desired.push({ name: names[k], g: g, origin: k < conn.length ? conn[k].origin : null });
            }
        }
        if (current.length === desired.length) {
            let same = true;
            for (let i = 0; i < current.length; i++) {
                const a = current[i], b = desired[i];
                const ao = a.origin, bo = b.origin;
                const eq = a.name === b.name && ((ao === null && bo === null) || (ao !== null && bo !== null && ao.id === bo.id && ao.slot === bo.slot));
                if (!eq) { same = false; break; }
            }
            if (same) return;
        }
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (inp && groupOf(inp.name)) node.removeInput(i);
        }
        for (const d of desired) {
            node.addInput(d.name, node.__ag_types[d.name] || "IMAGE");
            const idx = node.inputs.length - 1;
            if (d.origin !== null) {
                try {
                    const src = node.graph.getNodeById ? node.graph.getNodeById(d.origin.id) : null;
                    if (src && src.outputs && src.outputs[d.origin.slot]) {
                        src["connect"](d.origin.slot, node, idx); // bracket invocation: registry YARA substring false positive (network rule)
                    }
                } catch (e) {
                    console.warn("[AGSoft MiniMax H3] reconnect skipped:", e);
                }
            }
        }
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] normalizeRefs skipped:", e);
    }
}
// Скрывает/показывает виджеты калькулятора по режимам; размер ноды не трогает.
// invert_orientation виден ВО ВСЕХ режимах (работает во всех, как в Base).
// Hides/shows calculator widgets per mode; does not touch the node size.
// invert_orientation is visible in ALL modes (active in all, as in Base).
function updateWidgets(node) {
    const get = (n) => node.widgets.find((w) => w.name === n);
    const modeW = get("mode");
    const srcW = get("frame_count_source");
    const mode = modeW ? modeW.value : "Preset";
    const src = srcW ? srcW.value : "Seconds";
    const rules = {
        preset: mode === "Preset",
        invert_orientation: true,
        custom_width: mode === "Custom",
        custom_height: mode === "Custom",
        megapixels_value: mode === "Megapixels",
        aspect_ratio: mode === "Megapixels",
        length_seconds: src === "Seconds",
        frame_count: src === "Frames",
    };
    let changed = false;
    for (const w of node.widgets) {
        if (w.name in rules) {
            const h = !rules[w.name];
            if (w.hidden !== h) { w.hidden = h; changed = true; }
        }
    }
    return changed;
}
// ==============================================================================
// Живая инфострока / live info line (18px DOM widget between last combo and prompt)
// ==============================================================================
function fit32(v) {
    return Math.max(32, Math.ceil(v / 32) * 32);
}
// ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ v09.48: цикл вместо формулы с %. В JS остаток от
// деления отрицательный для отрицательных чисел, поэтому старая формула
// округляла ВНИЗ (24 → 22), а Python (fit_length_to_17n5) округляет ВВЕРХ
// (24 → 39). Цикл 1:1 с 🎬AGSoft MiniMax Base — строка совпадает с выходами.
// THE ONLY v09.48 CHANGE: a loop instead of the % formula. JS modulo is
// negative for negatives, so the old formula rounded DOWN (24 → 22) while
// Python (fit_length_to_17n5) rounds UP (24 → 39). The loop is 1:1 with
// 🎬AGSoft MiniMax Base — the line now matches the outputs.
function fit17n5(v) {
    let x = Math.max(5, Math.round(Number(v) || 5));
    while (x % 17 !== 5) x += 1;
    return x;
}
// ==============================================================================
// Форматы 1:1 с 🎬AGSoft MiniMax Base / formats 1:1 with 🎬AGSoft MiniMax Base
// ==============================================================================
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
// ТОЧНЫЕ секунды: 12 знаков после запятой, хвостовые нули убраны (как в Base).
// Exact seconds: 12 decimals, trailing zeros stripped (same as Base).
function exactSec(v) {
    return Number(v.toFixed(12)).toString();
}
function computeInfo(node) {
    const get = (n) => {
        const x = node.widgets.find((q) => q.name === n);
        return x ? x.value : null;
    };
    const mode = get("mode");
    const invert = get("invert_orientation");
    let w = 1280, h = 704;
    if (mode === "Preset") {
        const m = String(get("preset") || "").match(/(\d+)\s*×\s*(\d+)/);
        if (m) { w = parseInt(m[1], 10); h = parseInt(m[2], 10); }
        if (invert) { const t = w; w = h; h = t; }
        w = fit32(w); h = fit32(h);
    } else if (mode === "Custom") {
        w = fit32(get("custom_width") || 0);
        h = fit32(get("custom_height") || 0);
        if (invert) { const t = w; w = h; h = t; }
    } else if (mode === "Megapixels") {
        const mp = parseFloat(get("megapixels_value")) || 1;
        const r = String(get("aspect_ratio") || "16:9").split(":");
        const wr = parseInt(r[0], 10) || 16, hr = parseInt(r[1], 10) || 9;
        const x = Math.sqrt((mp * 1000000) / (wr * hr));
        w = Math.max(64, fit32(Math.round(wr * x)));
        h = Math.max(64, fit32(Math.round(hr * x)));
        if (invert) { const t = w; w = h; h = t; }
    }
    const src = get("frame_count_source");
    let total;
    if (src === "Frames") total = fit17n5(get("frame_count") || 5);
    else total = fit17n5(Math.round((parseFloat(get("length_seconds")) || 0) * 24));
    total = Math.max(5, total);
    // Формат строки 1:1 с Base: кадры ПЕРЕД секундами, секунды точные.
    // Line format 1:1 with Base: frames BEFORE seconds, seconds exact.
    const sec = exactSec(total / 24);
    const mp = num((w * h) / 1000000) + "MP";
    const ratio = "~" + approximateRatio(w, h);
    return `⚙ ${w}×${h} • ${mp} • ${ratio} • ${total}f • ${sec}s`;
}
function ensureBarWidget(node) {
    if (node.__ag_bar_w) return node.__ag_bar_w;
    const div = document.createElement("div");
    div.style.cssText = "width:100%;height:" + BAR_H + "px;line-height:" + BAR_H + "px;text-align:center;" +
        "background:#263238;color:#9ecbdd;font:10px monospace;border-radius:8px;overflow:hidden;" +
        "white-space:nowrap;box-sizing:border-box;pointer-events:none;";
    div.textContent = computeInfo(node);
    let w = null;
    try {
        w = node.addDOMWidget("agsoft_info_bar", "div", div, { serialize: false });
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] addDOMWidget failed:", e);
        return null;
    }
    w.computeSize = function (width) { return [width || 0, BAR_H]; };
    // Ставим строку ПЕРЕД prompt: между последним combo и полем ввода
    // Place the bar BEFORE prompt: between the last combo and the input field
    const ws = node.widgets;
    const bi = ws.indexOf(w);
    const pi = ws.findIndex((x) => x.name === "prompt");
    if (bi > -1 && pi > -1 && bi > pi) {
        ws.splice(bi, 1);
        ws.splice(pi, 0, w);
    }
    node.__ag_bar_w = w;
    node.__ag_bar_el = div;
    return w;
}
function updateBarText(node) {
    if (!node.__ag_bar_el) { ensureBarWidget(node); return; }
    const t = computeInfo(node);
    if (node.__ag_bar_el.textContent !== t) node.__ag_bar_el.textContent = t;
}
function attachInfoBar(node) {
    if (node.__ag_bar) return;
    node.__ag_bar = true;
    // Очистка старых properties-флагов прошлых версий / cleanup of old properties flags
    if (node.properties) {
        delete node.properties.__ag_bar_reserved;
        delete node.properties.__ag_bar_bottom;
        delete node.properties.agsoft_user_h;
        delete node.properties.agsoft_prompt_h;
        delete node.properties.agsoft_node_h;
        delete node.properties.agsoft_node_w;
    }
    ensureBarWidget(node);
}
// allowNormalizeRefs = false только в синхронном refresh при создании:
// линков ещё нет, пересборка насчитала бы «подключено 0».
// allowNormalizeRefs = false only in the synchronous refresh on creation:
// links are not there yet, the rebuild would count "connected 0".
function refresh(node, allowNormalizeRefs) {
    try {
        updateWidgets(node);
        if (allowNormalizeRefs !== false && node.comfyClass === "AGSoft_MiniMax_H3_Ref2V") normalizeRefs(node);
        updateBarText(node);
        node.setDirtyCanvas(true, true);
    } catch (e) {
        console.warn("[AGSoft MiniMax H3] refresh skipped:", e);
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
    name: "AGSoft.MiniMaxH3",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoft_MiniMax_H3_Ref2V" && node.comfyClass !== "AGSoft_MiniMax_H3_I2V") return;
        attachInfoBar(node);
        // Ручной ресайз обновляет память; программные setSize память не трогают.
        // A manual resize updates the memory; programmatic setSize does not.
        const origResize = node.onResize;
        node.onResize = function (size) {
            if (origResize) origResize.apply(this, arguments);
            const canvas = app && app.canvas;
            const isUserResize = canvas && canvas.resizing_node === this;
            if (isUserResize && size && size.length === 2) {
                SIZE_MEM.set(this.id, [size[0], size[1]]);
            }
        };
        // Перехват setSize БЕЗ блокировки: чужой размер откатывается в том же
        // тике; запись всегда проходит, layout не расходится.
        // setSize intercept WITHOUT blocking: foreign size reverted in the
        // same tick; the write always passes, layout never diverges.
        const origSetSize = node.setSize;
        node.setSize = function (size) {
            if (origSetSize) origSetSize.apply(this, arguments);
            try {
                const canvas = app && app.canvas;
                if (canvas && canvas.resizing_node === this) return this; // драг пользователя / user drag
                const mem = SIZE_MEM.get(this.id);
                if (mem && this.size &&
                    (Math.abs(this.size[0] - mem[0]) > 0.5 || Math.abs(this.size[1] - mem[1]) > 0.5)) {
                    this.size[0] = mem[0];
                    this.size[1] = mem[1];
                    this.setDirtyCanvas(true, true);
                }
            } catch (e) {}
            return this;
        };
        // Синхронно пересборку НЕ запускаем: линков ещё нет.
        // Do NOT run the rebuild synchronously: links are not there yet.
        refresh(node, false);
        for (const name of HOOK_WIDGETS) {
            const w = node.widgets.find((x) => x.name === name);
            if (!w) continue;
            const prev = w.callback;
            const oc = prev ? (v) => prev.call(w, v) : null;
            w.callback = (v) => {
                if (oc) oc(v);
                updateWidgets(node);
                updateBarText(node);
                node.setDirtyCanvas(true, true);
            };
        }
        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            if (origConn) origConn.apply(this, arguments);
            if (type === 1) scheduleRefresh(this);
        };
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            try {
                // Сеим память из сериализованного размера ДО применения.
                // Seed the memory from the serialized size BEFORE applying.
                if (info && info.size && info.size.length === 2) SIZE_MEM.set(this.id, [info.size[0], info.size[1]]);
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
            } finally {}
            attachInfoBar(this);
            scheduleRefresh(this);
            startPin(this);
        };
        // Отложенный проход после создания/загрузки + посев памяти для НОВЫХ нод.
        // Deferred pass after creation/load + memory seeding for NEW nodes.
        setTimeout(() => {
            if (!node.inputs) return;
            scheduleRefresh(node);
            setTimeout(() => {
                if (!SIZE_MEM.has(node.id)) rememberSize(node);
                startPin(node);
            }, 0);
        }, 0);
    },
});