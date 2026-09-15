// ==============================================================================
// AGSoft_KSampler_Options.js
// ==============================================================================
// JS-расширение для нод 🛠 AGSoft KSampler options_single / options_dual /
// options_lora: UI-строки поверх скрытых нативных виджетов + подсветка.
// JS extension for the 🛠 AGSoft KSampler options_single / options_dual /
// options_lora nodes: row UI over hidden native widgets + highlighting.
//
// Возможности / Features:
// ⚡ Строки single/dual: [тумблер][param dropdown][value control]; тип контрола
//   следует за param (int / float / dropdown sampler|scheduler / текст sigmas).
//   Rows single/dual: [toggle][param dropdown][value control]; the control
//   follows the param type (int / float / sampler|scheduler dropdown / sigmas).
// ⚡ Строки lora: [тумблер][◄ model ►][◄ clip ►] — ГОРИЗОНТАЛЬНЫЕ степеры как
//   в 🧩 AGSoft Multi LoRA Loader: клик ±0.05, Shift ±0.01, ручной ввод;
//   дефолт значений = 1 (берётся из нативных виджетов).
//   Rows lora: [toggle][◄ model ►][◄ clip ►] — HORIZONTAL steppers like in
//   🧩 AGSoft Multi LoRA Loader: click ±0.05, Shift ±0.01, manual typing;
//   default values = 1 (taken from native widgets).
// ⚡ 30 строк максимум (MAX_ROWS = 30).
//   Up to 30 rows (MAX_ROWS = 30).
// ⚡ Контекстное меню строки (правый клик): Move Up / Move Down / Duplicate /
//   Remove — операции над скрытыми нативными виджетами строки.
//   Row context menu (right-click): Move Up / Move Down / Duplicate / Remove —
//   operations over the row's hidden native widgets.
// ⚡ Подсветка: series-режим (single/dual/lora) — серверный agsoft_series_step
//   из KSampler; current-режим — позиция из executed-события options-ноды
//   (парсинг status "…: i/N").
//   Highlighting: series mode (single/dual/lora) — server agsoft_series_step
//   from the KSampler; current mode — position from the options node's own
//   executed event (parsing status "…: i/N").
// ⚡ Пустое значение строки single/dual коммитится дефолтом в нативный виджет;
//   в lora пустые ячейки остаются пустыми (пусто = пропуск строки).
//   Empty single/dual rows commit defaults into native widgets; lora cells
//   stay empty (empty = skip row).
//
// JS extension for the 🛠 AGSoft KSampler options_single / options_dual /
// options_lora nodes. (See the RU list above — the same features.)
// 
// Автор / Author: AGSoft
// Дата / Date: 16.09.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const MAX_ROWS = 30;
const PARAMS = ["seed", "steps", "cfg", "sampler", "scheduler", "sigmas"];
const SINGLE = "AGSoftKSamplerOptionsSingle";
const DUAL = "AGSoftKSamplerOptionsDual";
const LORA = "AGSoftKSamplerOptionsLora";
const KSAMPLER = "AGSoft_KSampler";
const ROW_H = 26;
const BTN_H = 26;
const GAP = 4;

const CSS = `
.agsopt-root{display:flex;flex-direction:column;gap:${GAP}px;width:100%;pointer-events:none;}
.agsopt-row{display:flex;gap:4px;align-items:center;height:${ROW_H}px;pointer-events:auto;border-radius:4px;}
.agsopt-row.agsopt-active{outline:1px solid var(--ags-accent,#4a7dba);
  background:color-mix(in srgb, var(--ags-accent,#4a7dba) 14%, transparent);}
.agsopt-row select,.agsopt-row input{
  background:#353535;background:color-mix(in srgb, var(--ags-bg,#353535) 70%, black);
  color:#ddd;color:var(--ags-text,#ddd);
  border:1px solid #555;border:1px solid color-mix(in srgb, var(--ags-bg,#353535) 42%, white);
  border-radius:4px;height:22px;font-size:11px;padding:0 4px;box-sizing:border-box;}
.agsopt-row select.param{flex:0 0 84px;}
.agsopt-row input.val,.agsopt-row select.val{flex:1;min-width:0;}
.agsopt-step{display:flex;align-items:center;flex:1;min-width:0;height:22px;
  background:#353535;background:color-mix(in srgb, var(--ags-bg,#353535) 70%, black);
  color:#ddd;color:var(--ags-text,#ddd);
  border:1px solid #555;border:1px solid color-mix(in srgb, var(--ags-bg,#353535) 42%, white);
  border-radius:4px;box-sizing:border-box;overflow:hidden;}
.agsopt-step button{flex:0 0 16px;height:100%;border:none;background:transparent;color:inherit;
  cursor:pointer;font-size:8px;line-height:1;padding:0;}
.agsopt-step button:hover{background:rgba(128,128,128,.25);}
.agsopt-step input[type=number]{flex:1;min-width:0;width:auto;height:100%;background:transparent;
  border:none;color:inherit;text-align:center;font-size:11px;padding:0;
  appearance:textfield;-moz-appearance:textfield;}
.agsopt-step input[type=number]::-webkit-outer-spin-button,
.agsopt-step input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0;}
.agsopt-row input[type=checkbox]{
  appearance:none;-webkit-appearance:none;width:28px;height:14px;border-radius:8px;
  background:#555;background:color-mix(in srgb, var(--ags-bg,#555) 55%, black);
  position:relative;cursor:pointer;outline:none;border:none;flex:0 0 auto;margin:0;}
.agsopt-row input[type=checkbox]::after{
  content:"";position:absolute;top:2px;left:2px;width:10px;height:10px;border-radius:50%;
  background:#999;background:color-mix(in srgb, var(--ags-text,#999) 70%, transparent);transition:.15s;}
.agsopt-row input[type=checkbox]:checked{background:#4a7dba;background:var(--ags-accent,#4a7dba);}
.agsopt-row input[type=checkbox]:checked::after{left:16px;background:#fff;}
.agsopt-btns{display:flex;gap:6px;height:${BTN_H}px;pointer-events:auto;}
.agsopt-btns button{flex:1;background:#4a4a4a;
  background:color-mix(in srgb, var(--ags-bg,#4a4a4a) 80%, white);
  color:#eee;color:var(--ags-text,#eee);
  border:1px solid #5a5a5a;border:1px solid color-mix(in srgb, var(--ags-bg,#4a4a4a) 48%, white);
  border-radius:4px;height:${BTN_H}px;cursor:pointer;font-size:11px;}
.agsopt-btns button:hover{background:#5a5a5a;background:color-mix(in srgb, var(--ags-bg,#4a4a4a) 62%, white);}
.agsopt-menu{position:fixed;z-index:10001;background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:4px;min-width:150px;box-shadow:0 4px 12px rgba(0,0,0,.5);}
.agsopt-menu-item{padding:5px 10px;color:#ddd;color:var(--ags-text,#ddd);font-size:11px;border-radius:4px;cursor:pointer;}
.agsopt-menu-item:hover{background:#3a3a3a;background:color-mix(in srgb, var(--ags-bg,#3a3a3a) 80%, white);}
.agsopt-menu-item.disabled{opacity:.4;pointer-events:none;}`;

let cssDone = false;
const injectCss = () => {
    if (cssDone) return;
    cssDone = true;
    const el = document.createElement("style");
    el.textContent = CSS;
    document.head.appendChild(el);
};

const nw = (node, name) => (node.widgets || []).find(w => w.name === name) || null;

const collapse = (w) => {
    if (!w || w._ags_collapsed) return;
    w.computeSize = () => [0, -4];
    w.hidden = true;
    w._ags_collapsed = true;
};

const hookCallback = (w, fn) => {
    if (!w) return;
    const oc = w.callback ? w.callback.bind(w) : null;
    w.callback = (v) => {
        if (oc) oc(v);
        fn(v);
    };
};

const KIND_BY_PARAM = {};
KIND_BY_PARAM["se" + "ed"] = "int";
KIND_BY_PARAM["st" + "eps"] = "int";
KIND_BY_PARAM["cfg"] = "float";
KIND_BY_PARAM["samp" + "ler"] = "list";
KIND_BY_PARAM["sched" + "uler"] = "list";
KIND_BY_PARAM["sigmas"] = "text";

const kindOf = (p) => {
  const kind = KIND_BY_PARAM[p];
  if (kind === undefined) return "text";
  return kind;
};

const DEFAULT_BY_PARAM = {};
DEFAULT_BY_PARAM["se" + "ed"] = "0";
DEFAULT_BY_PARAM["st" + "eps"] = "20";
DEFAULT_BY_PARAM["cfg"] = "1";

const LIST_PARAMS = ["samp" + "ler", "sched" + "uler"];

const defaultFor = (p, list) => {
  const def = DEFAULT_BY_PARAM[p];
  if (def !== undefined) return def;

  if (LIST_PARAMS.includes(p)) {
    let arr = [];
    if (Array.isArray(list)) arr = list;
    if (arr.length > 0) return String(arr[0]);
  }

  return "";
};

// ------------------------------------------------------------------------------
// Контекстное меню: глобальное закрытие по клику вне и по Escape.
// Context menu: global dismissal on outside press and Escape.
// ------------------------------------------------------------------------------
let ctxMenu = null;
const hideMenu = () => { if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; } };
document.addEventListener("pointerdown", (e) => {
    if (ctxMenu && !ctxMenu.contains(e.target)) hideMenu();
}, true);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideMenu(); }, true);

const showMenu = (x, y, items) => {
    hideMenu();
    ctxMenu = document.createElement("div");
    ctxMenu.className = "agsopt-menu";
    for (const it of items) {
        const b = document.createElement("div");
        b.className = "agsopt-menu-item" + (it.disabled ? " disabled" : "");
        b.textContent = it.label;
        if (!it.disabled) {
            b.addEventListener("click", () => { hideMenu(); it.cb(); });
        }
        ctxMenu.appendChild(b);
    }
    ctxMenu.style.left = x + "px";
    ctxMenu.style.top = y + "px";
    document.body.appendChild(ctxMenu);
    const r = ctxMenu.getBoundingClientRect();
    if (r.right > window.innerWidth) ctxMenu.style.left = Math.max(4, window.innerWidth - r.width - 8) + "px";
    if (r.bottom > window.innerHeight) ctxMenu.style.top = Math.max(4, window.innerHeight - r.height - 8) + "px";
};

// ------------------------------------------------------------------------------
// Ячейка param+value (single/dual): dropdown параметра и контрол по типу;
// источник списка (sampler/scheduler) следует за выбранным param.
// One param+value cell (single/dual): param dropdown plus a per-type control;
// the list source (sampler/scheduler) follows the selected param.
// ------------------------------------------------------------------------------
const makeCell = (node, paramW, valueW) => {
    const wrap = document.createElement("div");
    wrap.style.display = "contents";

    const paramSel = document.createElement("select");
    paramSel.className = "param";
    for (const p of PARAMS) {
        const o = document.createElement("option");
        o.value = p;
        o.textContent = p;
        paramSel.appendChild(o);
    }

    let valCtl = null;
    let syncing = false;

    const listWFor = (p) =>
        p === "scheduler" ? nw(node, "_scheduler_list") : nw(node, "_sampler_list");
    const listValuesFor = (p) => {
        const w = listWFor(p);
        return (w && w.options && w.options.values) || [];
    };

    const commitNative = (v) => {
        if (!valueW) return;
        valueW.value = String(v);
        if (valueW.callback) valueW.callback(valueW.value);
    };

    const buildValueCtl = (param) => {
        if (valCtl) valCtl.remove();
        const kind = kindOf(param);
        if (kind === "list") {
            valCtl = document.createElement("select");
            // Метка источника списка: пересоздать контрол при смене param.
            // Tag the list source: rebuild the control when param changes.
            valCtl.dataset.src = param;
            for (const v of listValuesFor(param)) {
                const o = document.createElement("option");
                o.value = v;
                o.textContent = v;
                valCtl.appendChild(o);
            }
        } else {
            valCtl = document.createElement("input");
            if (kind === "int") { valCtl.type = "number"; valCtl.step = "1"; }
            else if (kind === "float") { valCtl.type = "number"; valCtl.step = "0.05"; }
            else { valCtl.type = "text"; valCtl.placeholder = "1.0, 0.96, 0.857143, 0.0"; }
        }
        valCtl.className = "val";
        valCtl.addEventListener("change", () => commitNative(valCtl.value));
        wrap.appendChild(valCtl);
    };

    const sync = () => {
        if (syncing) return;
        syncing = true;
        try {
            const p = String(paramW ? paramW.value : "steps");
            if (paramSel.value !== p) paramSel.value = p;
            const kind = kindOf(p);
            const needRebuild =
                !valCtl ||
                (kind === "list" && (valCtl.tagName !== "SELECT" || valCtl.dataset.src !== p)) ||
                (kind !== "list" && valCtl.tagName === "SELECT") ||
                (kind === "int" && valCtl.type !== "number") ||
                (kind === "float" && valCtl.type !== "number") ||
                (kind === "text" && valCtl.type !== "text");
            if (needRebuild) buildValueCtl(p);
            const raw = String(valueW ? valueW.value : "");
            if (valCtl) {
                let shown;
                if (kind === "list") shown = listValuesFor(p).includes(raw) ? raw : (listValuesFor(p)[0] || "");
                else if (raw !== "") shown = raw;
                else shown = defaultFor(p, listValuesFor(p));
                valCtl.value = shown;
                if (raw === "") commitNative(shown);
            }
        } finally {
            syncing = false;
        }
    };

    paramSel.addEventListener("change", () => {
        if (paramW) {
            paramW.value = paramSel.value;
            if (paramW.callback) paramW.callback(paramW.value);
        }
        commitNative(defaultFor(paramSel.value, listValuesFor(paramSel.value)));
        sync();
    });

    wrap.append(paramSel);
    hookCallback(paramW, () => sync());
    hookCallback(valueW, () => sync());
    sync();
    return { wrap, sync };
};

// ------------------------------------------------------------------------------
// Горизонтальный степер (lora): [◄][значение][►] как в 🧩 AGSoft Multi LoRA
// Loader. Клик ±0.05, Shift ±0.01, ручной ввод сохранён. Значение живёт в
// скрытом нативном виджете; пусто = пропуск строки.
// Horizontal stepper (lora): [◄][value][►] like in 🧩 AGSoft Multi LoRA
// Loader. Click ±0.05, Shift ±0.01, manual typing kept. The value lives in
// the hidden native widget; empty = skip row.
// ------------------------------------------------------------------------------
const makeStepCell = (node, valueW, titleText) => {
    const wrap = document.createElement("div");
    wrap.style.display = "contents";

    const box = document.createElement("div");
    box.className = "agsopt-step";
    box.title = titleText;

    const dec = document.createElement("button");
    dec.textContent = "◀";
    dec.title = "−0.05 (Shift: −0.01)";

    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "0.01";
    inp.min = "-10";
    inp.max = "10";
    inp.title = titleText;

    const inc = document.createElement("button");
    inc.textContent = "▶";
    inc.title = "+0.05 (Shift: +0.01)";

    const commit = () => {
        if (!valueW) return;
        valueW.value = String(inp.value);
        if (valueW.callback) valueW.callback(valueW.value);
    };

    const bump = (dir, ev) => {
        const d = ev && ev.shiftKey ? 0.01 : 0.05;
        const cur = parseFloat(inp.value);
        let v = (Number.isFinite(cur) ? cur : 1) + dir * d;
        const mn = parseFloat(inp.min);
        const mx = parseFloat(inp.max);
        if (Number.isFinite(mn)) v = Math.max(mn, v);
        if (Number.isFinite(mx)) v = Math.min(mx, v);
        inp.value = Math.round(v * 100) / 100;
        commit();
    };
    dec.addEventListener("click", (e) => { e.preventDefault(); bump(-1, e); });
    inc.addEventListener("click", (e) => { e.preventDefault(); bump(1, e); });
    inp.addEventListener("change", commit);

    const sync = () => {
        inp.value = String(valueW ? valueW.value : "");
    };

    hookCallback(valueW, () => sync());
    box.append(dec, inp, inc);
    wrap.append(box);
    sync();
    return { wrap, sync };
};

// ------------------------------------------------------------------------------
// Сборка UI ноды: строки поверх скрытых нативных виджетов, кнопки
// + Add / – Remove, контекстное меню строки. kind: single | dual | lora.
// Node UI builder: rows over hidden native widgets, + Add / – Remove buttons,
// row context menu. kind: single | dual | lora.
// ------------------------------------------------------------------------------
const buildUI = (node, kind) => {
    const root = document.createElement("div");
    root.className = "agsopt-root";

    const activeW = nw(node, "active_rows");
    const PREFIXES =
        kind === "dual" ? ["enabled_", "param_a_", "value_a_", "param_b_", "value_b_"]
        : kind === "lora" ? ["enabled_", "strength_m_", "strength_c_"]
        : ["enabled_", "param_", "value_"];

    const getActive = () => Math.min(MAX_ROWS, Math.max(1, parseInt(activeW ? activeW.value : 1, 10) || 1));
    const setActive = (v) => {
        if (!activeW) return;
        activeW.value = Math.min(MAX_ROWS, Math.max(1, v));
        if (activeW.callback) activeW.callback(activeW.value);
        refresh();
    };

    // ---- Значения строк через скрытые нативные виджеты ----
    // ---- Row value transport over hidden native widgets ----
    const getRowVals = (i) => PREFIXES.map(p => { const w = nw(node, p + i); return w ? w.value : null; });
    const setRowVals = (i, vals) => PREFIXES.forEach((p, k) => {
        const w = nw(node, p + i);
        if (w) { w.value = vals[k]; if (w.callback) w.callback(w.value); }
    });
    const blankRow = () =>
        kind === "dual" ? [true, "sampler", "", "scheduler", ""]
        : kind === "lora" ? [true, "1", "1"]
        : [true, "steps", ""];

    const syncAll = () => { for (const r of rows) r.sync(); };

    const moveRow = (i, dir) => {
        const j = i + dir;
        if (j < 1 || j > getActive()) return;
        const a = getRowVals(i), b = getRowVals(j);
        setRowVals(i, b); setRowVals(j, a);
        syncAll();
    };
    const dupRow = (i) => {
        const a = getActive();
        if (a >= MAX_ROWS) return;
        for (let j = a; j >= i; j--) setRowVals(j + 1, getRowVals(j));
        setActive(a + 1);
        syncAll();
    };
    const removeRow = (i) => {
        const a = getActive();
        if (i < 1 || i > a) return;
        for (let j = i; j < a; j++) setRowVals(j, getRowVals(j + 1));
        setRowVals(a, blankRow());
        setActive(a - 1);
        syncAll();
    };

    const rows = [];
    for (let i = 1; i <= MAX_ROWS; i++) {
        const enW = nw(node, `enabled_${i}`);
        const el = document.createElement("div");
        el.className = "agsopt-row";

        const en = document.createElement("input");
        en.type = "checkbox";
        en.title = "on / off";
        en.addEventListener("change", () => {
            if (enW) { enW.value = en.checked; if (enW.callback) enW.callback(enW.value); }
        });
        hookCallback(enW, (v) => { en.checked = !!v; });
        el.appendChild(en);

        const cells = [];
        if (kind === "dual") {
            cells.push(makeCell(node, nw(node, `param_a_${i}`), nw(node, `value_a_${i}`)));
            cells.push(makeCell(node, nw(node, `param_b_${i}`), nw(node, `value_b_${i}`)));
        } else if (kind === "lora") {
            cells.push(makeStepCell(node, nw(node, `strength_m_${i}`), "model strength"));
            cells.push(makeStepCell(node, nw(node, `strength_c_${i}`), "clip strength"));
        } else {
            cells.push(makeCell(node, nw(node, `param_${i}`), nw(node, `value_${i}`)));
        }
        for (const cell of cells) el.appendChild(cell.wrap);

        // Контекстное меню строки: Move Up / Move Down / Duplicate / Remove.
        // Row context menu: Move Up / Move Down / Duplicate / Remove.
        el.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            const a = getActive();
            showMenu(e.clientX, e.clientY, [
                { label: "⬆️ Move Up", disabled: i <= 1, cb: () => moveRow(i, -1) },
                { label: "⬇️ Move Down", disabled: i >= a, cb: () => moveRow(i, +1) },
                { label: "⧉ Duplicate", disabled: a >= MAX_ROWS, cb: () => dupRow(i) },
                { label: "🗑️ Remove", disabled: a <= 1, cb: () => removeRow(i) },
            ]);
        });

        rows.push({ el, i, cells, sync: () => { en.checked = !!(enW && enW.value); for (const c of cells) c.sync(); } });
        root.appendChild(el);
    }

    const btns = document.createElement("div");
    btns.className = "agsopt-btns";
    const addBtn = document.createElement("button");
    addBtn.textContent = "+ Add";
    addBtn.addEventListener("click", (e) => { e.preventDefault(); setActive(getActive() + 1); });
    const delBtn = document.createElement("button");
    delBtn.textContent = "– Remove";
    delBtn.addEventListener("click", (e) => { e.preventDefault(); setActive(getActive() - 1); });
    btns.append(addBtn, delBtn);
    root.appendChild(btns);

    const uiHeight = (a) => BTN_H + a * ROW_H + (a + 1) * GAP;
    const dw = node.addDOMWidget("agsoft_ksampler_options_ui", "div", root, { serialize: false });
    dw.computeSize = (w) => [w || 220, uiHeight(getActive())];

    const refresh = () => {
        const a = getActive();
        for (const r of rows) r.el.style.display = r.i <= a ? "flex" : "none";
        const s = node.computeSize ? node.computeSize() : null;
        if (s && Array.isArray(s)) node.size[1] = s[1];
        node.setDirtyCanvas(true, true);
        if (app.graph && app.graph.setDirtyCanvas) app.graph.setDirtyCanvas(true);
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => { syncAll(); refresh(); }, 0);
    };

    node._ags_rows = rows;
    node._ags_root = root;

    refresh();
    setTimeout(() => { syncAll(); refresh(); }, 0);
    return root;
};

// ------------------------------------------------------------------------------
// Хелперы подсветки: строки, участвующие в текущем шаге серии.
// Highlight helpers: rows participating in the current series step.
// ------------------------------------------------------------------------------
const clearActive = (node) => {
    const root = node._ags_root;
    if (!root) return;
    root.querySelectorAll(".agsopt-row.agsopt-active").forEach(el => el.classList.remove("agsopt-active"));
};
const clearAllActive = () => {
    for (const n of app.graph.nodes || []) {
        if (n.comfyClass === SINGLE || n.comfyClass === DUAL || n.comfyClass === LORA) clearActive(n);
    }
};
const highlightRows = (node, rowsArr) => {
    clearActive(node);
    for (const r of rowsArr || []) {
        const row = (node._ags_rows || [])[r - 1];
        if (row) row.el.classList.add("agsopt-active");
    }
};

// Строки, участвующие в серии (зеркалит серверную логику).
// Rows participating in the series (mirrors server logic).
const seriesRows = (node) => {
    const dual = node.comfyClass === DUAL;
    const lora = node.comfyClass === LORA;
    const grid = dual && nw(node, "pair_mode") && nw(node, "pair_mode").value === "grid";
    const active = Math.min(MAX_ROWS, Math.max(1, parseInt((nw(node, "active_rows") || {}).value, 10) || 1));
    const has = (name) => String((nw(node, name) || {}).value || "").trim() !== "";

    if (lora) {
        const list = [];
        for (let i = 1; i <= active; i++) {
            const en = (nw(node, `enabled_${i}`) || {}).value;
            if (en === false) continue;
            if (has(`strength_m_${i}`)) list.push(i);
        }
        return { rows: list, grid: false, colA: list, colB: list };
    }
    if (!dual) {
        const list = [];
        for (let i = 1; i <= active; i++) {
            const en = (nw(node, `enabled_${i}`) || {}).value;
            if (en === false) continue;
            if (has(`value_${i}`)) list.push(i);
        }
        return { rows: list, grid: false, colA: list, colB: list };
    }
    const colA = [], colB = [];
    for (let i = 1; i <= active; i++) {
        const en = (nw(node, `enabled_${i}`) || {}).value;
        if (en === false) continue;
        if (has(`value_a_${i}`)) colA.push(i);
        if (has(`value_b_${i}`)) colB.push(i);
    }
    if (!grid) {
        const rows = [];
        for (let i = 1; i <= active; i++) {
            if (colA.includes(i) && colB.includes(i)) rows.push(i);
        }
        return { rows, grid: false, colA, colB };
    }
    return { rows: [...new Set([...colA, ...colB])], grid: true, colA, colB };
};

const rowsForPos = (node, pos) => {
    const s = seriesRows(node);
    if (!s.grid) return [s.rows[pos]];
    const kb = pos % Math.max(1, s.colB.length);
    const ka = Math.floor(pos / Math.max(1, s.colB.length));
    return [s.colA[ka], s.colB[kb]];
};

// ------------------------------------------------------------------------------
// Регистрация расширения: построение UI нод и подсветка строк серии.
// Extension registration: node UI building and series row highlighting.
// ------------------------------------------------------------------------------
let armedSeriesNode = null;

const optionsSourceOf = (ksNode) => {
    const inp = (ksNode.inputs || []).find(i => i.name === "options");
    if (!inp || inp.link == null) return null;
    const link = app.graph.links[inp.link];
    if (!link) return null;
    return app.graph.getNodeById(link.origin_id) || null;
};

app.registerExtension({
    name: "AGSoft.KSamplerOptions",

    async nodeCreated(node) {
        const dual = node.comfyClass === DUAL;
        const lora = node.comfyClass === LORA;
        if (node.comfyClass !== SINGLE && !dual && !lora) return;
        injectCss();

        // Скрываем служебные и строчные виджеты.
        // Hide service and row widgets.
        for (const w of node.widgets || []) {
            if (/^(enabled_|param_|value_|param_a_|value_a_|param_b_|value_b_|strength_m_|strength_c_)\d+$/.test(w.name)
                || w.name === "active_rows"
                || w.name === "_sampler_list"
                || w.name === "_scheduler_list") {
                collapse(w);
            }
        }
        buildUI(node, dual ? "dual" : lora ? "lora" : "single");
    },
});

// Серверный шаг серии (series-режим KSampler): подсветка строк.
// Server series step (KSampler series mode): highlight the rows.
api.addEventListener("agsoft_series_step", (e) => {
    const d = e.detail || {};
    if (armedSeriesNode) highlightRows(armedSeriesNode, d.rows);
});

// executing: вооружаем options-ноду серии (single/dual/lora) при входе в KSampler.
// executing: arm the series options node (single/dual/lora) when KSampler starts.
api.addEventListener("executing", (e) => {
    const id = e.detail;
    if (id == null) { clearAllActive(); armedSeriesNode = null; return; }
    const node = app.graph.getNodeById(id);
    if (node && node.comfyClass === KSAMPLER) {
        const src = optionsSourceOf(node);
        if (src && (src.comfyClass === SINGLE || src.comfyClass === DUAL || src.comfyClass === LORA)
            && (nw(src, "output_mode") || {}).value === "series") {
            armedSeriesNode = src;
            // Превью всей серии: подсветить все участвующие строки сразу.
            // Whole-series preview: highlight all participating rows at once.
            highlightRows(src, seriesRows(src).rows);
        } else {
            armedSeriesNode = null;
            clearAllActive();
        }
    }
});

// executed: current-режим — позиция из status; KSampler finished — сброс.
// executed: current mode — position from status; KSampler finished: clear.
api.addEventListener("executed", (e) => {
    const d = e.detail || {};
    const node = app.graph.getNodeById(d.node);
    if (!node) return;
    if (node.comfyClass === KSAMPLER) {
        clearAllActive();
        armedSeriesNode = null;
        return;
    }
    if (node.comfyClass !== SINGLE && node.comfyClass !== DUAL && node.comfyClass !== LORA) return;
    const outMode = (nw(node, "output_mode") || {}).value;
    if (outMode === "series") return; // подсветкой управляет agsoft_series_step
    const status = String((d.output || {}).status || "");
    const m = status.match(/:\s*(\d+)\/(\d+)/);
    if (!m) return;
    const pos = Math.max(0, parseInt(m[1], 10) - 1);
    highlightRows(node, rowsForPos(node, pos));
});