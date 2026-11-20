/*
==============================================================================
AGSoft_Pipe.js
==============================================================================
Нода: 🧩 AGSoft Pipe Master / 🔀 AGSoft Pipe Link
Описание / Description:
    Фронтенд шины параметров: динамические слоты, редактор имён с шаблонами,
    выбор записи на каждый выход Link и детерминированный размер нод.
    ---
    Parameter bus frontend: dynamic slots, name editor with presets,
    per-output entry selection for Link and deterministic node sizing.

Возможности / Features:
    ⚡ Master: рост входов только добавлением хвоста, один свободный слот
      Master: inputs grow by appending the tail only, one free slot is kept
    ⚡ Редактор имён: Enter фиксирует, Escape откатывает, фокус не теряется
      Name editor: Enter commits, Escape reverts, typing focus is preserved
    ⚡ Шаблоны имён по группам, повторы получают номер, активная строка видна
      Grouped name presets, duplicates get a number, active row is highlighted
    ⚡ Link: список записей тянется из вышестоящего Master обходом графа
      Link: the entry list is pulled from the upstream Master by a graph walk
    ⚡ Тип и подпись выхода повторяют выбранную запись (нативные цвета типов)
      Output type and label follow the selected entry (native type colors)
    ⚡ Размер ноды считается от содержимого: растёт и сжимается, ширина своя
      Node size is content-driven: grows and shrinks, user width is preserved

Замечания / Notes:
    - Ключи входов in_N и индекс сокета bus неизменны: линки не пересобираются.
      in_N input keys and the bus socket index are immutable: links are kept.
    - Высота панелей — арифметика по числу строк фиксированной высоты; замер
      высоты ноды выполняется без опоры на текущий размер, чтобы она сжималась.
      Panel height is arithmetic over fixed-height rows; node height is
      measured without the current size as a floor, so it can shrink back.
    - Все выборы хранятся в ags_meta, поэтому список выходов восстанавливается
      при загрузке workflow без обращения к значениям шины.
      All selections live in ags_meta, so the output list is restored on
      workflow load without touching bus values.

Автор / Author: AGSoft
Дата / Date: 20.11.2026
==============================================================================
*/

import { app } from "../../../scripts/app.js";

console.log("[AGSoft Pipe] JS extension loaded v11.20");

const MAX_SLOTS = 32;
const MAX_OUT = 32;
const MASTER_CLASS = "AGSoftPipeMaster";
const LINK_CLASS = "AGSoftPipeLink";
const BUS_TYPE = "AGS_PIPE";
const EMPTY = " ";
const NONE_LABEL = "— none —";
const PRESET_LABEL = "— template —";
const ROW_H = 24;
const ROW_GAP = 4;
const PANEL_PAD = 6;
const PANEL_BORDER = 1;
const WIDGET_GAP = 6;
const NODE_BOTTOM_PAD = 10;
const TYPE_ORDER = {
    MODEL: 0, CLIP: 1, VAE: 2, CONDITIONING: 3, LATENT: 4,
    IMAGE: 5, MASK: 6, AUDIO: 7, VIDEO: 8, STRING: 9, INT: 10, FLOAT: 11,
};
const NAME_PRESETS = [
    ["Models", ["model", "clip", "vae", "video_vae", "audio_vae", "clip_vision",
        "controlnet", "upscale_model", "style_model"]],
    ["Conditioning", ["positive", "negative", "conditioning"]],
    ["Latent & media", ["latent", "image", "mask", "video", "audio"]],
    ["Params", ["text", "seed", "steps", "cfg", "denoise", "width", "height",
        "sigmas", "guider", "noise"]],
];

let refreshTimer = null;

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name) || null;
}

function readMeta(node) {
    const w = getWidget(node, "ags_meta");
    try {
        const data = JSON.parse((w && w.value) || "{}");
        return data && typeof data === "object" ? data : {};
    } catch {
        return {};
    }
}

function writeMeta(node, meta) {
    const w = getWidget(node, "ags_meta");
    if (w) w.value = JSON.stringify(meta);
}

function collapseWidget(w) {
    w.hidden = true;
    w.y = -9999;
    if (w._agsCompute === undefined) w._agsCompute = w.computeSize;
    w.computeSize = () => [0, -4];
}

function dirty() {
    app.graph.setDirtyCanvas(true, true);
}

// Rows have fixed heights, so the panel size is pure arithmetic and never drifts.
// Строки фиксированной высоты, поэтому размер панели — арифметика без дрейфа.
function panelHeight(rows) {
    const pad = PANEL_PAD * 2 + PANEL_BORDER * 2;
    if (rows <= 0) return pad;
    return rows * ROW_H + (rows - 1) * ROW_GAP + pad;
}

// Content height must be measured without the current size floor, else it ratchets.
// Высоту меряем без опоры на текущий размер, иначе она только растёт.
function fitNodeHeight(node) {
    if (!node.size || (node.flags && node.flags.collapsed)) return;
    const keep = node.size[1];
    let content;
    node.size[1] = 0;
    try {
        content = node.computeSize();
    } finally {
        node.size[1] = keep;
    }
    const width = Math.max(node.size[0], content[0]);
    const height = content[1] + NODE_BOTTOM_PAD;
    if (node.size[0] !== width || node.size[1] !== height) {
        node.setSize([width, height]);
    }
}

function upstreamNode(node, slot) {
    const input = node.inputs && node.inputs[slot];
    if (!input || input.link == null) return null;
    const link = app.graph.links[input.link];
    if (!link) return null;
    return app.graph.getNodeById(link.origin_id);
}

function slotType(node, slot) {
    const input = node.inputs && node.inputs[slot];
    if (!input || input.link == null) return "*";
    const link = app.graph.links[input.link];
    if (!link) return "*";
    const src = app.graph.getNodeById(link.origin_id);
    const out = src && src.outputs && src.outputs[link.origin_slot];
    return (out && out.type) || "*";
}

function downstreamType(node, outIndex) {
    const out = node.outputs && node.outputs[outIndex];
    if (!out || !out.links || !out.links.length) return "";
    const link = app.graph.links[out.links[0]];
    if (!link) return "";
    const target = app.graph.getNodeById(link.target_id);
    const input = target && target.inputs && target.inputs[link.target_slot];
    return (input && input.type) || "";
}

// Full bus entry list visible at this point of the chain (model-only walk).
// Полный список записей шины в этой точке цепочки (обход только по модели).
function resolveEntries(node, depth) {
    if (!node || depth > 16) return [];
    if (node.comfyClass === MASTER_CLASS) {
        const entries = readMeta(node).entries;
        return Array.isArray(entries) ? entries : [];
    }
    if (node.comfyClass === LINK_CLASS) {
        return resolveEntries(upstreamNode(node, 0), depth + 1);
    }
    return [];
}

function displayOf(entry) {
    return `${entry.name} [${entry.type || "*"}]`;
}

function sortEntries(entries) {
    return entries.slice().sort((a, b) => {
        const ta = TYPE_ORDER[a.type] ?? 99;
        const tb = TYPE_ORDER[b.type] ?? 99;
        if (ta !== tb) return ta - tb;
        return String(a.id).localeCompare(String(b.id));
    });
}

function uniqueName(name, taken) {
    const base = (name || "entry").replace(/\s+/g, " ").trim().slice(0, 48) || "entry";
    let candidate = base;
    let n = 2;
    while (taken.has(candidate)) {
        candidate = `${base}_${n}`;
        n += 1;
    }
    return candidate;
}

function scheduleRefresh() {
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
        refreshTimer = null;
        for (const n of app.graph._nodes || []) {
            if (n.comfyClass === LINK_CLASS && n._agsRefresh) n._agsRefresh();
        }
        dirty();
    }, 60);
}

function panelStyle(el) {
    el.style.cssText =
        "display:flex;flex-direction:column;flex:0 0 auto;gap:4px;padding:6px;" +
        "box-sizing:border-box;width:100%;height:auto;overflow:visible;" +
        "font:11px sans-serif;color:#d6d6d6;background:rgba(20,20,22,0.92);" +
        "border:1px solid rgba(255,255,255,0.06);border-radius:4px;";
}

// Dark scheme forces the native popup to stay readable.
// Тёмная схема держит нативный выпадающий список читаемым.
function selectStyle(el) {
    el.style.cssText =
        "flex:1 1 auto;min-width:0;height:22px;background:#121214;color:#e6e6e6;" +
        "border:1px solid #4a4a52;border-radius:3px;font:11px sans-serif;" +
        "padding:0 4px;box-sizing:border-box;color-scheme:dark;";
}

function optionStyle(opt) {
    opt.style.backgroundColor = "#1b1b1f";
    opt.style.color = "#e6e6e6";
}

function groupStyle(og) {
    og.style.backgroundColor = "#121214";
    og.style.color = "#7fd4ff";
    og.style.fontStyle = "normal";
    og.style.fontWeight = "600";
}

function rowStyle(extra) {
    return "display:flex;flex:0 0 auto;gap:6px;align-items:center;height:24px;" +
        "box-sizing:border-box;" + (extra || "");
}

// ---------------------------------------------------------------- Master ---

function masterEnsureSlots(node) {
    // Append-only growth; trim keeps exactly one free tail slot.
    // Рост только добавлением; усадка оставляет один свободный хвостовой слот.
    const inputs = node.inputs || [];
    const last = inputs[inputs.length - 1];
    if (last && last.link != null && inputs.length - 1 < MAX_SLOTS) {
        const n = inputs.length;
        node.addInput(`in_${n}`, "*");
    }
    while (node.inputs.length > 2) {
        const a = node.inputs[node.inputs.length - 1];
        const b = node.inputs[node.inputs.length - 2];
        if (a.link == null && b.link == null) {
            node.removeInput(node.inputs.length - 1);
        } else {
            break;
        }
    }
}

function masterLabels(node) {
    const meta = readMeta(node);
    const names = meta.names || {};
    for (let i = 1; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        const name = names[input.name] || input.name;
        input.label = `${String(i).padStart(2, "0")} · ${name}`;
    }
}

// Names already carried by the upstream bus also block duplicates.
// Имена из вышестоящей шины тоже участвуют в проверке повторов.
function masterTaken(node, excludeKey) {
    const taken = new Set();
    for (const e of resolveEntries(upstreamNode(node, 0), 0)) {
        taken.add(String(e.name));
    }
    const names = readMeta(node).names || {};
    for (const k of Object.keys(names)) {
        if (k !== excludeKey && names[k]) taken.add(String(names[k]));
    }
    return taken;
}

function masterLocalKeys(node) {
    const keys = [];
    for (let i = 1; i < node.inputs.length; i++) keys.push(node.inputs[i].name);
    return keys;
}

function masterActiveKey(node) {
    const keys = masterLocalKeys(node);
    if (!keys.length) return "";
    if (node._agsActiveKey && keys.includes(node._agsActiveKey)) return node._agsActiveKey;
    const names = readMeta(node).names || {};
    for (const k of keys) {
        if (!names[k] || names[k] === k) return k;
    }
    return keys[0];
}

function masterSetActive(node, key) {
    node._agsActiveKey = key;
    const rows = node._agsRows || {};
    for (const k of Object.keys(rows)) {
        const on = k === key;
        rows[k].row.style.borderLeft = on ? "2px solid #7fd4ff" : "2px solid transparent";
        rows[k].num.style.color = on ? "#7fd4ff" : "#8a8a8a";
    }
}

// After a preset lands, move the marker to the next still-default slot.
// После применения пресета маркер уходит к следующему свободному имени слоту.
function masterAdvanceActive(node, afterKey) {
    const keys = masterLocalKeys(node);
    const start = keys.indexOf(afterKey);
    const names = readMeta(node).names || {};
    for (let i = start + 1; i < keys.length; i++) {
        if (!names[keys[i]] || names[keys[i]] === keys[i]) {
            masterSetActive(node, keys[i]);
            return;
        }
    }
    masterSetActive(node, afterKey);
}

function masterApplyPreset(node) {
    const preset = node._agsPreset;
    if (!preset || preset === EMPTY) return;
    const key = masterActiveKey(node);
    if (!key) return;
    const meta = readMeta(node);
    meta.names = meta.names || {};
    meta.names[key] = uniqueName(preset, masterTaken(node, key));
    writeMeta(node, meta);
    const row = node._agsRows && node._agsRows[key];
    if (row) row.field.value = meta.names[key];
    masterLabels(node);
    masterBuildMeta(node, false);
    masterAdvanceActive(node, key);
}

function masterBuildMeta(node, rebuildEditor) {
    const meta = readMeta(node);
    meta.names = meta.names || {};
    meta.types = meta.types || {};
    const taken = new Set();
    const upstream = resolveEntries(upstreamNode(node, 0), 0);
    const merged = upstream.map((e) => {
        const eid = String(e.id || "");
        const name = uniqueName(String(e.name || "entry"), taken);
        taken.add(name);
        return {
            id: eid.startsWith("U:") ? eid : "U:" + eid,
            key: String(e.key || ""),
            name,
            type: String(e.type || "*"),
        };
    });
    const local = [];
    for (let i = 1; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        const key = input.name;
        const name = uniqueName(String(meta.names[key] || key), taken);
        meta.names[key] = name;
        taken.add(name);
        if (input.link == null) continue;
        local.push({
            id: `L:${i}`,
            key,
            name,
            type: meta.types[key] || slotType(node, i),
        });
    }
    meta.entries = merged.concat(local);
    writeMeta(node, meta);
    masterLabels(node);
    if (rebuildEditor && node._agsEditorRebuild) {
        node._agsEditorRebuild();
    } else if (node._agsRows) {
        // In-place badge refresh keeps the typing focus alive.
        // Обновление бейджей на месте сохраняет фокус ввода.
        for (let i = 1; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            const row = node._agsRows[input.name];
            if (!row) continue;
            row.badge.textContent = meta.types[input.name] ||
                (input.link != null ? slotType(node, i) : "—");
            row.dot.textContent = input.link != null ? "●" : "○";
            row.dot.style.color = input.link != null ? "#7fd47f" : "#666";
        }
        fitNodeHeight(node);
    }
    scheduleRefresh();
}

function setupMaster(node) {
    const inputs = node.inputs || [];
    if (inputs[0] && inputs[0].name !== "bus_in" && inputs.every((s) => s.link == null)) {
        const idx = inputs.findIndex((s) => s.name === "bus_in");
        if (idx > 0) {
            const slot = inputs.splice(idx, 1)[0];
            inputs.unshift(slot);
        }
    }
    while (node.inputs.length > 2 && node.inputs[node.inputs.length - 1].link == null) {
        node.removeInput(node.inputs.length - 1);
    }
    if (node.inputs.length < 2) node.addInput("in_1", "*");
    node.inputs[0].label = "bus ⟵";
    const metaW = getWidget(node, "ags_meta");
    if (metaW) collapseWidget(metaW);
    node._agsPreset = node._agsPreset || EMPTY;

    const editor = document.createElement("div");
    panelStyle(editor);
    editor._agsCount = 2;
    const ew = node.addDOMWidget("ags_editor", "agseditor", editor, { serialize: false });
    ew.computeSize = (width) => [width, panelHeight(editor._agsCount) + WIDGET_GAP];

    node._agsEditorRebuild = () => {
        editor.innerHTML = "";
        node._agsRows = {};
        const meta = readMeta(node);
        const names = meta.names || {};
        const types = meta.types || {};
        let rows = 0;

        const bar = document.createElement("div");
        bar.style.cssText = rowStyle();
        const presetSel = document.createElement("select");
        selectStyle(presetSel);
        presetSel.title = "Preset name for the highlighted row --- " +
            "Шаблонное имя для подсвеченной строки";
        const optNone = document.createElement("option");
        optNone.value = EMPTY;
        optNone.textContent = PRESET_LABEL;
        optionStyle(optNone);
        presetSel.appendChild(optNone);
        for (const group of NAME_PRESETS) {
            const og = document.createElement("optgroup");
            og.label = group[0];
            groupStyle(og);
            for (const name of group[1]) {
                const opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                optionStyle(opt);
                og.appendChild(opt);
            }
            presetSel.appendChild(og);
        }
        presetSel.value = node._agsPreset || EMPTY;
        const applyBtn = document.createElement("button");
        applyBtn.textContent = "＋";
        applyBtn.title = "Apply preset, duplicates get a number --- " +
            "Применить шаблон, при повторе добавляется номер";
        applyBtn.style.cssText =
            "flex:0 0 auto;width:30px;height:22px;background:rgba(127,212,255,0.15);" +
            "border:1px solid #7fd4ff;border-radius:3px;color:#7fd4ff;" +
            "font:12px sans-serif;cursor:pointer;padding:0;box-sizing:border-box;";
        const syncBtn = () => {
            const on = !!presetSel.value && presetSel.value !== EMPTY;
            applyBtn.disabled = !on;
            applyBtn.style.opacity = on ? "1" : "0.4";
            applyBtn.style.cursor = on ? "pointer" : "default";
        };
        presetSel.onchange = () => {
            node._agsPreset = presetSel.value;
            syncBtn();
        };
        presetSel.onkeydown = (ev) => ev.stopPropagation();
        applyBtn.onkeydown = (ev) => ev.stopPropagation();
        applyBtn.onclick = () => masterApplyPreset(node);
        syncBtn();
        bar.append(presetSel, applyBtn);
        editor.appendChild(bar);
        rows += 1;

        const upstream = resolveEntries(upstreamNode(node, 0), 0);
        for (const e of upstream) {
            const row = document.createElement("div");
            row.style.cssText = rowStyle(
                "opacity:0.6;border-left:2px solid transparent;padding-left:2px;"
            );
            const num = document.createElement("span");
            num.style.cssText = "width:24px;text-align:right;";
            num.textContent = "↑";
            const label = document.createElement("span");
            label.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;";
            label.textContent = String(e.name || "");
            const badge = document.createElement("span");
            badge.style.cssText = "color:#7fd4ff;min-width:52px;text-align:right;";
            badge.textContent = String(e.type || "*");
            row.append(num, label, badge);
            editor.appendChild(row);
            rows += 1;
        }

        for (let i = 1; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            const key = input.name;
            const row = document.createElement("div");
            row.style.cssText = rowStyle(
                "border-left:2px solid transparent;padding-left:2px;"
            );
            const num = document.createElement("span");
            num.style.cssText = "width:24px;text-align:right;color:#8a8a8a;";
            num.textContent = String(i).padStart(2, "0");
            const field = document.createElement("input");
            field.type = "text";
            field.value = names[key] || key;
            field.title = "Enter commits, Escape reverts --- " +
                "Enter фиксирует, Escape откатывает";
            field.style.cssText =
                "flex:1 1 auto;min-width:0;height:22px;background:#121214;" +
                "border:1px solid #4a4a52;border-radius:3px;color:#e6e6e6;" +
                "font:11px sans-serif;padding:0 5px;box-sizing:border-box;" +
                "color-scheme:dark;";
            const badge = document.createElement("span");
            badge.style.cssText = "color:#7fd4ff;min-width:52px;text-align:right;";
            badge.textContent = types[key] || (input.link != null ? slotType(node, i) : "—");
            const dot = document.createElement("span");
            dot.style.cssText = "width:10px;text-align:center;";
            dot.textContent = input.link != null ? "●" : "○";
            dot.style.color = input.link != null ? "#7fd47f" : "#666";
            field.onfocus = () => {
                field._agsPrev = (readMeta(node).names || {})[key] || key;
                masterSetActive(node, key);
            };
            field.oninput = () => {
                const m = readMeta(node);
                m.names = m.names || {};
                m.names[key] = field.value;
                writeMeta(node, m);
                masterLabels(node);
                masterBuildMeta(node, false);
            };
            field.onkeydown = (ev) => {
                ev.stopPropagation();
                if (ev.key === "Enter") {
                    ev.preventDefault();
                    field.blur();
                } else if (ev.key === "Escape") {
                    ev.preventDefault();
                    const m = readMeta(node);
                    m.names = m.names || {};
                    m.names[key] = field._agsPrev || key;
                    writeMeta(node, m);
                    field.value = m.names[key];
                    masterLabels(node);
                    masterBuildMeta(node, false);
                    field.blur();
                }
            };
            field.onblur = () => {
                const m = readMeta(node);
                m.names = m.names || {};
                m.names[key] = uniqueName(field.value || key, masterTaken(node, key));
                writeMeta(node, m);
                field.value = m.names[key];
                masterLabels(node);
                masterBuildMeta(node, false);
            };
            row.onmousedown = () => masterSetActive(node, key);
            row.append(num, field, badge, dot);
            editor.appendChild(row);
            node._agsRows[key] = { row, num, field, badge, dot };
            rows += 1;
        }
        editor._agsCount = rows;
        masterSetActive(node, masterActiveKey(node));
        fitNodeHeight(node);
    };

    const prevConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slot, connected, linkInfo) {
        if (prevConn) prevConn.apply(this, arguments);
        if (type !== 1) return;
        if (slot > 0) {
            const meta = readMeta(node);
            meta.types = meta.types || {};
            const input = node.inputs[slot];
            if (input) meta.types[input.name] = connected ? slotType(node, slot) : "*";
            writeMeta(node, meta);
        }
        masterEnsureSlots(node);
        masterBuildMeta(node, true);
    };

    const prevCfg = node.onConfigure;
    node.onConfigure = function (info) {
        if (prevCfg) prevCfg.apply(this, arguments);
        masterEnsureSlots(node);
        masterBuildMeta(node, true);
    };

    masterEnsureSlots(node);
    masterBuildMeta(node, true);
}

// ------------------------------------------------------------------ Link ---

function linkResizeOutputs(node, target) {
    // Bus stays at index 0; only the unconnected tail may shrink.
    // Bus фиксирован на индексе 0; усадка снимает только неподключённый хвост.
    target = Math.max(1, Math.min(MAX_OUT, target));
    let current = node.outputs.length - 1;
    while (current < target) {
        current += 1;
        node.addOutput(`out_${current}`, "*");
    }
    while (current > target) {
        const last = node.outputs[node.outputs.length - 1];
        if (last.links && last.links.length) break;
        node.removeOutput(node.outputs.length - 1);
        current -= 1;
    }
    const countW = getWidget(node, "count");
    if (countW && countW.value !== current) countW.value = current;
    return current;
}

function linkApplyOutput(node, i, entry) {
    const out = node.outputs[i];
    if (!out) return;
    if (entry) {
        out.type = entry.type || "*";
        out.label = entry.name;
    } else {
        out.type = "*";
        out.label = `out_${i}`;
    }
}

function linkRenderRows(node, entries, meta) {
    const box = node._agsOutBox;
    const count = node.outputs.length - 1;
    const sig = count + "|" + entries.map((e) => `${e.id}:${e.name}:${e.type}`).join(",");
    const byId = new Map(entries.map((e) => [String(e.id), e]));
    if (sig !== node._agsOutSig) {
        node._agsOutSig = sig;
        box.innerHTML = "";
        node._agsOutRows = {};
        for (let i = 1; i <= count; i++) {
            const row = document.createElement("div");
            row.style.cssText = rowStyle();
            const num = document.createElement("span");
            num.style.cssText = "width:26px;text-align:right;color:#8a8a8a;";
            num.textContent = String(i).padStart(2, "0");
            const sel = document.createElement("select");
            selectStyle(sel);
            const optNone = document.createElement("option");
            optNone.value = EMPTY;
            optNone.textContent = NONE_LABEL;
            optionStyle(optNone);
            sel.appendChild(optNone);
            for (const e of entries) {
                const opt = document.createElement("option");
                opt.value = displayOf(e);
                opt.textContent = displayOf(e);
                optionStyle(opt);
                sel.appendChild(opt);
            }
            const badge = document.createElement("span");
            badge.style.cssText = "color:#7fd4ff;min-width:52px;text-align:right;";
            const dot = document.createElement("span");
            dot.style.cssText = "width:10px;text-align:center;";
            sel.onkeydown = (ev) => ev.stopPropagation();
            sel.onchange = () => {
                const m = readMeta(node);
                const entry = entries.find((e) => displayOf(e) === sel.value) || null;
                m[`out_${i}`] = entry ?
                    { id: String(entry.id), name: String(entry.name) } :
                    { id: "", name: "" };
                writeMeta(node, m);
                linkApplyOutput(node, i, entry);
                node._agsRefresh();
            };
            row.append(num, sel, badge, dot);
            box.appendChild(row);
            node._agsOutRows[i] = { sel, badge, dot };
        }
        box._agsCount = count;
    }
    for (let i = 1; i <= count; i++) {
        const row = node._agsOutRows[i];
        if (!row) continue;
        const rec = meta[`out_${i}`] || {};
        const entry = byId.get(String(rec.id)) || null;
        row.sel.value = entry ? displayOf(entry) : EMPTY;
        row.badge.textContent = entry ? String(entry.type || "*") : "—";
        const out = node.outputs[i];
        const linked = !!(out && out.links && out.links.length);
        row.dot.textContent = linked ? "●" : "○";
        row.dot.style.color = linked ? "#7fd47f" : "#666";
    }
    fitNodeHeight(node);
}

function setupLink(node) {
    const countW = getWidget(node, "count");
    const initial = countW ? Number(countW.value) || 1 : 1;
    while (node.outputs.length > 1 + initial) {
        node.removeOutput(node.outputs.length - 1);
    }
    if (node.outputs.length < 2) node.addOutput("out_1", "*");
    node.outputs[0].label = "bus ⟶";
    const metaW = getWidget(node, "ags_meta");
    if (metaW) collapseWidget(metaW);
    node._agsOutSig = "";

    const box = document.createElement("div");
    panelStyle(box);
    box._agsCount = 1;
    node._agsOutBox = box;
    const ow = node.addDOMWidget("ags_outputs", "agsoutputs", box, { serialize: false });
    ow.computeSize = (width) => [width, panelHeight(box._agsCount) + WIDGET_GAP];

    node._agsRefresh = () => {
        const entries = sortEntries(resolveEntries(upstreamNode(node, 0), 0));
        const byId = new Map(entries.map((e) => [String(e.id), e]));
        const byName = new Map(entries.map((e) => [String(e.name), e]));
        const meta = readMeta(node);
        const count = node.outputs.length - 1;
        for (let i = 1; i <= count; i++) {
            const rec = meta[`out_${i}`];
            let entry = null;
            if (rec && rec.id) {
                entry = byId.get(String(rec.id)) || byName.get(String(rec.name));
            } else if (!rec) {
                // First run: prefer the type the downstream node expects.
                // Первый запуск: предпочитаем тип, который ждёт принимающая нода.
                const wanted = downstreamType(node, i);
                entry = entries.find((e) => wanted && e.type === wanted) ||
                    entries[0] || null;
            }
            if (entry) {
                meta[`out_${i}`] = { id: String(entry.id), name: String(entry.name) };
            }
            linkApplyOutput(node, i, entry);
        }
        writeMeta(node, meta);
        linkRenderRows(node, entries, meta);
        dirty();
    };

    if (countW) {
        const prevCount = countW.callback;
        countW.callback = (v) => {
            if (prevCount) prevCount.call(countW, v);
            linkResizeOutputs(node, Number(v) || 1);
            node._agsRefresh();
        };
    }

    const prevConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slot, connected, linkInfo) {
        if (prevConn) prevConn.apply(this, arguments);
        if (type === 1 && slot === 0) node._agsRefresh();
        if (type === 2) node._agsRefresh();
    };

    const prevCfg = node.onConfigure;
    node.onConfigure = function (info) {
        if (prevCfg) prevCfg.apply(this, arguments);
        const cw = getWidget(node, "count");
        linkResizeOutputs(node, Number(cw && cw.value) || 1);
        node._agsRefresh();
    };

    linkResizeOutputs(node, initial);
    node._agsRefresh();
}

app.registerExtension({
    name: "AGSoft.Pipe",
    async nodeCreated(node) {
        if (node.comfyClass === MASTER_CLASS) setupMaster(node);
        else if (node.comfyClass === LINK_CLASS) setupLink(node);
    },
});