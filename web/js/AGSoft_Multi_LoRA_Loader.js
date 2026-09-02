// ==============================================================================
// AGSoft_Multi_LoRA_Loader.js
// ==============================================================================
// JS-расширение для ноды 🧩 AGSoft Multi LoRA Loader.
//
// ⚡ Компактные строки слотов в одном DOM-контейнере (без невидимых зон
//   перехвата мыши): тумблер, выбор лоры, ◄ сила model ►, ◄ сила clip ►, ℹ.
// ⚡ Выбор лор деревом папок с фильтром (как файловый менеджер).
// ⚡ Горизонтальные степеры силы: ◄/► всегда видны, клик ±0.05, Shift ±0.01,
//   ручной ввод сохранён.
// ⚡ Контекстное меню строки (правый клик): Show Info / Toggle On-Off /
//   Move Up / Move Down / Remove.
// ⚡ Toggle All — простое действие: поставил = все вкл, снял = все выкл.
// ⚡ Инфо-диалог CivitAI: fetch по SHA256 → страница / триггеры / примеры
//   (img + video, 📝 промпт с chips steps/cfg/sampler); редактируемые
//   локальные заметки; Strength Min/Max ограничивают степеры строки.
// ⚡ Контролы следуют цвету ноды через CSS-переменные (--ags-*).
//
// Author: AGSoft
// Date: 02.09.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const CLASS_ID = "AGSoftMultiLoraLoader";
const MAX_SLOTS = 20;
const ROW_H = 24;
const HEAD_H = 20;
const BTNS_H = 26;
const GAP = 4;
const STEP_W = 76; // horizontal stepper width (Model / CLIP fields)

console.log("[AGSoft Multi LoRA Loader] JS extension loaded v1.08 (horizontal ◄ ► steppers, theme sync, single-container UI, folder-tree chooser)");

// ------------------------------------------------------------------------------
// CSS (row controls painted from --ags-* variables via color-mix, with
// plain fallbacks for old browsers)
// ------------------------------------------------------------------------------
const EXT_CSS = `
.agsoft-lora-root{display:flex;flex-direction:column;gap:${GAP}px;width:100%;}
.agsoft-lora-headrow{display:flex;gap:4px;align-items:center;height:${HEAD_H}px;font-size:11px;
  color:#999;color:color-mix(in srgb, var(--ags-text,#999) 62%, transparent);}
.agsoft-lora-headlabel{flex:1;min-width:0;}
.agsoft-lora-row{display:flex;gap:4px;align-items:center;height:${ROW_H}px;}
.agsoft-lora-row select,.agsoft-lora-row input[type=number]{
  background:#353535;
  background:color-mix(in srgb, var(--ags-bg,#353535) 70%, black);
  color:#ddd;color:var(--ags-text,#ddd);
  border:1px solid #555;
  border:1px solid color-mix(in srgb, var(--ags-bg,#353535) 42%, white);
  border-radius:4px;height:22px;font-size:11px;padding:0 4px;box-sizing:border-box;}
.agsoft-lora-row select{flex:1;min-width:0;text-overflow:ellipsis;}
.agsoft-lora-row input[type=number]{width:62px;}
.agsoft-lora-step{display:flex;align-items:center;width:${STEP_W}px;flex:0 0 auto;height:22px;
  background:#353535;
  background:color-mix(in srgb, var(--ags-bg,#353535) 70%, black);
  color:#ddd;color:var(--ags-text,#ddd);
  border:1px solid #555;
  border:1px solid color-mix(in srgb, var(--ags-bg,#353535) 42%, white);
  border-radius:4px;box-sizing:border-box;overflow:hidden;}
.agsoft-lora-step button{flex:0 0 16px;height:100%;border:none;background:transparent;color:inherit;
  cursor:pointer;font-size:8px;line-height:1;padding:0;}
.agsoft-lora-step button:hover{background:rgba(128,128,128,.25);}
.agsoft-lora-step input[type=number]{flex:1;min-width:0;width:auto;height:100%;background:transparent;
  border:none;color:inherit;text-align:center;font-size:11px;padding:0;
  appearance:textfield;-moz-appearance:textfield;}
.agsoft-lora-step input[type=number]::-webkit-outer-spin-button,
.agsoft-lora-step input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0;}
.agsoft-lora-row input[type=checkbox],.agsoft-lora-headrow input[type=checkbox]{
  appearance:none;-webkit-appearance:none;width:28px;height:14px;border-radius:8px;
  background:#555;
  background:color-mix(in srgb, var(--ags-bg,#555) 55%, black);
  position:relative;cursor:pointer;outline:none;border:none;flex:0 0 auto;margin:0;}
.agsoft-lora-row input[type=checkbox]::after,.agsoft-lora-headrow input[type=checkbox]::after{
  content:"";position:absolute;top:2px;left:2px;width:10px;height:10px;border-radius:50%;
  background:#999;background:color-mix(in srgb, var(--ags-text,#999) 70%, transparent);transition:.15s;}
.agsoft-lora-row input[type=checkbox]:checked,.agsoft-lora-headrow input[type=checkbox]:checked{
  background:#4a7dba;background:var(--ags-accent,#4a7dba);}
.agsoft-lora-row input[type=checkbox]:checked::after,.agsoft-lora-headrow input[type=checkbox]:checked::after{left:16px;background:#fff;}
.agsoft-lora-row button,.agsoft-lora-btns button{
  background:#4a4a4a;
  background:color-mix(in srgb, var(--ags-bg,#4a4a4a) 80%, white);
  color:#eee;color:var(--ags-text,#eee);
  border:1px solid #5a5a5a;
  border:1px solid color-mix(in srgb, var(--ags-bg,#4a4a4a) 48%, white);
  border-radius:4px;height:22px;cursor:pointer;font-size:11px;}
.agsoft-lora-row button:hover,.agsoft-lora-btns button:hover{
  background:#5a5a5a;
  background:color-mix(in srgb, var(--ags-bg,#4a4a4a) 62%, white);}
.agsoft-lora-chooser-btn{flex:1;min-width:0;text-align:left;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.agsoft-lora-btns{display:flex;gap:6px;height:${BTNS_H}px;}
.agsoft-lora-btns button{flex:1;height:${BTNS_H}px;}
.agsoft-lora-menu{position:fixed;z-index:10001;background:#2a2a2a;border:1px solid #555;border-radius:6px;padding:4px;min-width:160px;box-shadow:0 4px 12px rgba(0,0,0,.5);}
.agsoft-lora-menu-item{padding:5px 10px;color:#ddd;font-size:12px;cursor:pointer;border-radius:4px;}
.agsoft-lora-menu-item:hover{background:#3a3a3a;}
.agsoft-lora-menu-item.disabled{color:#777;cursor:default;}
.agsoft-lora-menu-item.disabled:hover{background:transparent;}
.agsoft-lora-chooser{position:fixed;z-index:10002;background:#2a2a2a;border:1px solid #555;border-radius:6px;width:460px;max-width:92vw;display:flex;flex-direction:column;box-shadow:0 8px 20px rgba(0,0,0,.6);}
.agsoft-lora-chooser-filter{margin:6px;padding:5px 8px;background:#222;color:#ddd;border:1px solid #555;border-radius:4px;font-size:12px;outline:none;}
.agsoft-lora-chooser-filter:focus{border-color:#7cb7ff;}
.agsoft-lora-chooser-list{overflow:auto;padding:4px;font-size:12px;color:#ddd;max-height:420px;}
.agsoft-lora-dir,.agsoft-lora-file{padding:3px 8px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.agsoft-lora-dir:hover,.agsoft-lora-file:hover{background:#3a3a3a;}
.agsoft-lora-file.current{background:#33507a;}
.agsoft-lora-arrow{display:inline-block;width:12px;color:#9cf;}
.agsoft-lora-none{color:#999;font-style:italic;}
.agsoft-d-overlay{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:10000;display:flex;align-items:center;justify-content:center;}
.agsoft-d-card{background:#3d3d3d;color:#ddd;border:1px solid #555;border-radius:8px;max-width:900px;width:94%;max-height:88vh;overflow:auto;padding:20px 24px;font-size:13px;}
.agsoft-d-title{color:#fff;font-size:17px;margin:0 0 14px;}
.agsoft-d-badges{margin:0 0 10px;display:flex;gap:6px;}
.agsoft-d-badge{background:#5a4a6a;color:#e6d9f2;border-radius:4px;padding:2px 8px;font-size:11px;}
.agsoft-d-badge.base{background:#4a5a4a;color:#d9f2d9;}
.agsoft-d-table{width:100%;border-collapse:collapse;margin-bottom:12px;}
.agsoft-d-table th,.agsoft-d-table td{border:1px solid #666;padding:6px 10px;text-align:left;vertical-align:top;font-weight:normal;}
.agsoft-d-table th{width:120px;background:#464646;color:#eee;}
.agsoft-d-table td.ags-val{word-break:break-all;}
.agsoft-d-table td.ags-pencil{width:34px;text-align:center;cursor:pointer;}
.agsoft-d-table td.ags-pencil:hover{background:#4a4a4a;}
.agsoft-d-btn{background:#2a2a2a;color:#eee;border:1px solid #666;border-radius:4px;padding:4px 12px;cursor:pointer;}
.agsoft-d-btn:hover{background:#3a3a3a;}
.agsoft-d-link{color:#8ab4f8;}
.agsoft-d-sec{margin:12px 0 6px;color:#bbb;text-transform:uppercase;font-size:11px;letter-spacing:.06em;}
.agsoft-d-chips{display:flex;flex-wrap:wrap;gap:6px;}
.agsoft-d-chip{background:#333;border:1px solid #555;border-radius:12px;padding:2px 10px;cursor:pointer;}
.agsoft-d-chip:hover{background:#444;}
.agsoft-d-strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:8px;}
.agsoft-d-media{position:relative;flex:0 0 auto;}
.agsoft-d-media img,.agsoft-d-media video{height:340px;border-radius:4px;background:#111;display:block;}
.agsoft-d-media video{width:auto;max-width:500px;}
.agsoft-d-media-btns{position:absolute;top:6px;right:6px;display:flex;gap:4px;z-index:2;}
.agsoft-d-media-btns button{width:26px;height:26px;border-radius:4px;border:1px solid #555;background:rgba(30,30,30,.85);color:#eee;cursor:pointer;font-size:12px;}
.agsoft-d-media-btns button:hover{background:rgba(60,60,60,.9);}
.agsoft-d-media-pop{position:absolute;left:0;right:0;bottom:0;max-height:70%;overflow:auto;background:rgba(20,20,20,.92);color:#ddd;font-size:11px;line-height:1.45;padding:8px;border-radius:0 0 4px 4px;display:none;z-index:1;}
.agsoft-d-media-pop.on{display:block;}
.agsoft-d-media-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;}
.agsoft-d-media-chips span{background:#333;border:1px solid #555;border-radius:4px;padding:1px 6px;}
.agsoft-d-edit{width:100%;background:#2a2a2a;color:#eee;border:1px solid #666;border-radius:4px;padding:4px;font-size:12px;}
.agsoft-d-closewrap{text-align:center;margin-top:14px;}
.agsoft-d-err{color:#ff8080;}
.agsoft-d-load{color:#9c9;}
`;
let cssInjected = false;
const injectCss = () => {
    if (cssInjected) return;
    cssInjected = true;
    const st = document.createElement("style");
    st.textContent = EXT_CSS;
    document.head.appendChild(st);
};

const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));

// ------------------------------------------------------------------------------
// Theme sync: node canvas color -> CSS variables on the UI container.
// Синхронизация темы: цвет ноды на канвасе -> CSS-переменные на контейнере.
// ------------------------------------------------------------------------------
const parseColor = (c) => {
    if (!c || typeof c !== "string") return null;
    let m = c.trim();
    if (m.startsWith("#")) {
        let h = m.slice(1);
        if (h.length === 3) h = h.split("").map((x) => x + x).join("");
        if (h.length >= 6) {
            return [
                parseInt(h.slice(0, 2), 16),
                parseInt(h.slice(2, 4), 16),
                parseInt(h.slice(4, 6), 16),
            ];
        }
        return null;
    }
    m = m.match(/rgba?\(([^)]+)\)/);
    if (m) {
        const p = m[1].split(",").map(parseFloat);
        if (p.length >= 3 && p.every((v) => Number.isFinite(v))) return [p[0], p[1], p[2]];
    }
    return null;
};

const luminance = (rgb) => (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;

const applyTheme = (node, root) => {
    const bg = node.bgcolor || "";
    const ac = node.color || "";
    const key = bg + "|" + ac;
    if (root._agsThemeKey === key) return;
    root._agsThemeKey = key;
    root.style.setProperty("--ags-bg", bg || "#353535");
    root.style.setProperty("--ags-accent", ac || "#4a7dba");
    const rgb = parseColor(bg);
    root.style.setProperty("--ags-text", rgb && luminance(rgb) > 0.6 ? "#222" : "#eee");
};

// ------------------------------------------------------------------------------
// Global dismissal: hide menu / chooser on outside press; Escape closes both.
// ------------------------------------------------------------------------------
let ctxMenuEl = null;
let chooserEl = null;

const hideRowMenu = () => {
    if (ctxMenuEl) {
        ctxMenuEl.remove();
        ctxMenuEl = null;
    }
};
const hideChooser = () => {
    if (chooserEl) {
        chooserEl.remove();
        chooserEl = null;
    }
};
document.addEventListener("pointerdown", (e) => {
    if (ctxMenuEl && !ctxMenuEl.contains(e.target)) hideRowMenu();
    if (chooserEl && !chooserEl.contains(e.target) && !e.target.closest(".agsoft-lora-chooser-btn")) hideChooser();
}, true);
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        hideRowMenu();
        hideChooser();
    }
}, true);

const showRowMenu = (x, y, items) => {
    hideRowMenu();
    ctxMenuEl = document.createElement("div");
    ctxMenuEl.className = "agsoft-lora-menu";
    for (const it of items) {
        const b = document.createElement("div");
        b.className = "agsoft-lora-menu-item" + (it.disabled ? " disabled" : "");
        b.textContent = it.label;
        if (!it.disabled) {
            b.addEventListener("click", () => {
                hideRowMenu();
                it.cb();
            });
        }
        ctxMenuEl.appendChild(b);
    }
    ctxMenuEl.style.left = x + "px";
    ctxMenuEl.style.top = y + "px";
    document.body.appendChild(ctxMenuEl);
    const r = ctxMenuEl.getBoundingClientRect();
    if (r.right > window.innerWidth) ctxMenuEl.style.left = Math.max(4, window.innerWidth - r.width - 8) + "px";
    if (r.bottom > window.innerHeight) ctxMenuEl.style.top = Math.max(4, window.innerHeight - r.height - 8) + "px";
};

// ------------------------------------------------------------------------------
// Folder-tree LoRA chooser
// ------------------------------------------------------------------------------
const buildTree = (paths) => {
    const root = { dirs: new Map(), files: [] };
    for (const p of paths) {
        const parts = String(p).split(/[\\/]/);
        let node = root;
        for (let i = 0; i < parts.length - 1; i++) {
            const d = parts[i];
            if (!node.dirs.has(d)) node.dirs.set(d, { dirs: new Map(), files: [] });
            node = node.dirs.get(d);
        }
        node.files.push({ full: String(p), name: parts[parts.length - 1] });
    }
    return root;
};

const showChooser = (anchorEl, paths, noneVal, current, onPick) => {
    hideChooser();
    chooserEl = document.createElement("div");
    chooserEl.className = "agsoft-lora-chooser";

    const filter = document.createElement("input");
    filter.className = "agsoft-lora-chooser-filter";
    filter.placeholder = "Filter list";

    const list = document.createElement("div");
    list.className = "agsoft-lora-chooser-list";

    chooserEl.append(filter, list);
    document.body.appendChild(chooserEl);

    const r = anchorEl.getBoundingClientRect();
    chooserEl.style.left = Math.max(4, r.left) + "px";
    chooserEl.style.top = (r.bottom + 4) + "px";
    requestAnimationFrame(() => {
        const cr = chooserEl.getBoundingClientRect();
        if (cr.right > window.innerWidth) chooserEl.style.left = Math.max(4, window.innerWidth - cr.width - 8) + "px";
        if (cr.bottom > window.innerHeight) {
            const above = r.top - cr.height - 4;
            chooserEl.style.top = (above > 4 ? above : Math.max(4, window.innerHeight - cr.height - 8)) + "px";
        }
    });

    const root = buildTree(paths);

    const fileRow = (full, label, depth, isNone) => {
        const row = document.createElement("div");
        row.className = "agsoft-lora-file" + (isNone ? " agsoft-lora-none" : "");
        row.style.paddingLeft = 8 + depth * 14 + "px";
        row.textContent = label;
        row.title = full;
        if (!isNone && full === current) row.classList.add("current");
        row.addEventListener("click", () => {
            hideChooser();
            onPick(full);
        });
        return row;
    };

    const renderNode = (node, depth) => {
        const frag = document.createDocumentFragment();
        const dirs = [...node.dirs.entries()].sort((a, b) => a[0].localeCompare(b[0]));
        for (const [name, child] of dirs) {
            const row = document.createElement("div");
            row.className = "agsoft-lora-dir";
            row.style.paddingLeft = 8 + depth * 14 + "px";
            row.innerHTML = `<span class="agsoft-lora-arrow">▸</span> 📁 ${esc(name)}`;
            const kids = document.createElement("div");
            kids.style.display = "none";
            kids.appendChild(renderNode(child, depth + 1));
            row.addEventListener("click", () => {
                const open = kids.style.display !== "none";
                kids.style.display = open ? "none" : "block";
                const ar = row.querySelector(".agsoft-lora-arrow");
                if (ar) ar.textContent = open ? "▸" : "▾";
            });
            frag.append(row, kids);
        }
        const files = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
        for (const f of files) frag.appendChild(fileRow(f.full, f.name, depth, false));
        return frag;
    };

    const render = (q) => {
        list.innerHTML = "";
        list.appendChild(fileRow(noneVal, "None", 0, true));
        if (q) {
            const matches = paths.filter((p) => String(p).toLowerCase().includes(q)).sort();
            for (const p of matches) list.appendChild(fileRow(p, p, 0, false));
        } else {
            list.appendChild(renderNode(root, 0));
        }
    };

    filter.addEventListener("input", () => render(filter.value.trim().toLowerCase()));
    render("");
    filter.focus();
};

// ------------------------------------------------------------------------------
// Info dialog (canonical civitai.com URLs, video examples,
// prompt popups on media)
// ------------------------------------------------------------------------------
let overlayEl = null;
let cardEl = null;

const ensureModal = () => {
    injectCss();
    if (overlayEl) return;
    overlayEl = document.createElement("div");
    overlayEl.className = "agsoft-d-overlay";
    cardEl = document.createElement("div");
    cardEl.className = "agsoft-d-card";
    overlayEl.appendChild(cardEl);
    overlayEl.addEventListener("click", (e) => {
        if (e.target === overlayEl) overlayEl.style.display = "none";
    });
    document.body.appendChild(overlayEl);
};

const metaCache = {};

const getMetaLight = async (name) => {
    if (metaCache[name]) return metaCache[name];
    try {
        const resp = await fetch(api.apiURL(`/agsoft/lora_meta?name=${encodeURIComponent(name)}`));
        const data = await resp.json();
        if (data && data.ok) {
            metaCache[name] = data;
            return data;
        }
    } catch (e) {}
    return { ok: false, file: "", meta: { name: "", strength_min: "", strength_max: "", notes: "", hash: "" } };
};

const saveMetaField = async (name, field, value) => {
    try {
        const resp = await fetch(api.apiURL("/agsoft/lora_meta_save"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, [field]: value }),
        });
        const data = await resp.json();
        if (data && data.ok) {
            delete metaCache[name];
            return true;
        }
    } catch (e) {}
    return false;
};

const mediaItemHtml = (im) => {
    const t = String(im.type || "image").toLowerCase();
    const media = t === "video"
        ? `<video src="${esc(im.url)}" muted loop controls playsinline title="${esc(im.prompt || "")}"></video>`
        : `<img src="${esc(im.url)}" data-ags-full="${esc(im.url)}" title="${esc(im.prompt || "")}" loading="lazy">`;
    const hasPop = !!(im.prompt || im.steps || im.cfg || im.sampler);
    const chips = [];
    if (im.steps !== undefined && im.steps !== null && im.steps !== "") chips.push(`steps ${esc(im.steps)}`);
    if (im.cfg !== undefined && im.cfg !== null && im.cfg !== "") chips.push(`cfg ${esc(im.cfg)}`);
    if (im.sampler) chips.push(`sampler ${esc(im.sampler)}`);
    const pop = hasPop
        ? `<div class="agsoft-d-media-pop">` +
          (chips.length ? `<div class="agsoft-d-media-chips">${chips.map((c) => `<span>${c}</span>`).join("")}</div>` : "") +
          `<div>${esc(im.prompt || "")}</div></div>`
        : "";
    const btn = hasPop
        ? `<div class="agsoft-d-media-btns"><button data-ags-pop="1" title="show / hide prompt">📝</button></div>`
        : "";
    return `<div class="agsoft-d-media">${media}${btn}${pop}</div>`;
};

const renderDialog = (ctx) => {
    const m = ctx.meta || {};
    const info = ctx.info || null;
    let html = `<h2 class="agsoft-d-title">${esc(m.name || ctx.name)}</h2>`;
    if (info && info.type) {
        html += `<div class="agsoft-d-badges"><span class="agsoft-d-badge">${esc(info.type)}</span>` +
            (info.version_name ? `<span class="agsoft-d-badge base">${esc(info.version_name)}</span>` : ``) +
            `</div>`;
    }
    const editRow = (label, field) =>
        `<tr><th>${label}</th><td class="ags-val" data-ags-val="${field}">${esc(m[field] || "")}</td>` +
        `<td class="ags-pencil" data-ags-edit="${field}" title="edit">✏️</td></tr>`;
    const civCell = info
        ? `<a class="agsoft-d-link" href="${esc(info.url)}" target="_blank" rel="noopener">ⓒ View on Civitai</a>`
        : `<button class="agsoft-d-btn" data-ags-fetch="1">Fetch info from civitai</button>` +
          (ctx.fetchError
              ? `<div class="agsoft-d-err" style="margin-top:6px">${esc(ctx.fetchError)}` +
                (ctx.searchUrl ? ` — <a class="agsoft-d-link" href="${esc(ctx.searchUrl)}" target="_blank" rel="noopener">search</a>` : "") +
                `</div>`
              : "");
    html += `<table class="agsoft-d-table">`;
    html += `<tr><th>File</th><td class="ags-val">${esc(ctx.file || ctx.name)}</td><td></td></tr>`;
    html += `<tr><th>Hash (sha256)</th><td class="ags-val">${esc(m.hash || "")}</td><td></td></tr>`;
    html += `<tr><th>Civitai</th><td class="ags-val">${civCell}</td><td></td></tr>`;
    html += editRow("Name", "name");
    html += editRow("Strength Min", "strength_min");
    html += editRow("Strength Max", "strength_max");
    html += editRow("Additional Notes", "notes");
    html += `</table>`;
    const words = (info && info.trained_words) || [];
    if (words.length) {
        html += `<div class="agsoft-d-sec">Trigger words (click to copy)</div><div class="agsoft-d-chips">` +
            words.map((w) => `<span class="agsoft-d-chip" data-ags-word="${esc(w)}">${esc(w)}</span>`).join("") +
            `</div>`;
    }
    const imgs = (info && info.images) || [];
    if (imgs.length) {
        html += `<div class="agsoft-d-sec">Examples (📝 = prompt, click image = full size)</div><div class="agsoft-d-strip">` +
            imgs.filter((im) => im && im.url).map(mediaItemHtml).join("") +
            `</div>`;
    }
    if (info && info.description) {
        html += `<div class="agsoft-d-sec">Description</div><div>${esc(info.description)}</div>`;
    }
    html += `<div class="agsoft-d-closewrap"><button class="agsoft-d-btn" data-ags-close="1">Close</button></div>`;
    cardEl.innerHTML = html;
};

const openInfoDialog = async (name, onClamp) => {
    ensureModal();
    overlayEl.style.display = "flex";
    if (!name) {
        cardEl.innerHTML = `<div class="agsoft-d-err">No LoRA selected in this slot.</div>
            <div class="agsoft-d-closewrap"><button class="agsoft-d-btn" data-ags-close="1">Close</button></div>`;
        return;
    }
    const light = await getMetaLight(name);
    const ctx = { name, file: light.file || "", meta: light.meta || {}, info: null, fetchError: "", searchUrl: "", onClamp };
    if (metaCache[name] && metaCache[name].info) ctx.info = metaCache[name].info;
    renderDialog(ctx);

    cardEl.onclick = async (e) => {
        const closeBtn = e.target.closest("[data-ags-close]");
        if (closeBtn) {
            overlayEl.style.display = "none";
            return;
        }
        const chip = e.target.closest("[data-ags-word]");
        if (chip) {
            if (navigator.clipboard) navigator.clipboard.writeText(chip.dataset.agsWord || "").catch(() => {});
            return;
        }
        const popBtn = e.target.closest("[data-ags-pop]");
        if (popBtn) {
            const box = popBtn.closest(".agsoft-d-media");
            const pop = box && box.querySelector(".agsoft-d-media-pop");
            if (pop) pop.classList.toggle("on");
            return;
        }
        const img = e.target.closest("[data-ags-full]");
        if (img) {
            window.open(img.dataset.agsFull, "_blank", "noopener");
            return;
        }
        const fetchBtn = e.target.closest("[data-ags-fetch]");
        if (fetchBtn) {
            fetchBtn.outerHTML = `<span class="agsoft-d-load">Loading… (first fetch computes SHA256)</span>`;
            try {
                const resp = await fetch(api.apiURL(`/agsoft/lora_info?name=${encodeURIComponent(name)}`));
                const data = await resp.json();
                if (data && data.ok) {
                    metaCache[name] = { ok: true, file: data.file || ctx.file, meta: data.meta || ctx.meta, info: data };
                    ctx.info = data;
                    ctx.meta = data.meta || ctx.meta;
                    ctx.file = data.file || ctx.file;
                    ctx.fetchError = "";
                    renderDialog(ctx);
                    if (onClamp) onClamp(ctx.meta);
                } else {
                    ctx.info = null;
                    ctx.fetchError = (data && data.error) || "fetch failed";
                    ctx.searchUrl = (data && data.search_url) || "";
                    renderDialog(ctx);
                }
            } catch (err) {
                ctx.info = null;
                ctx.fetchError = String(err);
                renderDialog(ctx);
            }
            return;
        }
        const pencil = e.target.closest("[data-ags-edit]");
        if (pencil) {
            const field = pencil.dataset.agsEdit;
            const valTd = cardEl.querySelector(`[data-ags-val="${field}"]`);
            if (!valTd || valTd.querySelector("input,textarea")) return;
            const old = ctx.meta[field] || "";
            const isNotes = field === "notes";
            valTd.innerHTML = isNotes
                ? `<textarea class="agsoft-d-edit" rows="3">${esc(old)}</textarea>`
                : `<input class="agsoft-d-edit" value="${esc(old)}">`;
            const ctl = valTd.firstChild;
            ctl.focus();
            const commit = async () => {
                const v = ctl.value;
                const ok = await saveMetaField(name, field, v);
                if (ok) ctx.meta[field] = v;
                metaCache[name] = Object.assign(metaCache[name] || {}, { meta: ctx.meta, file: ctx.file });
                renderDialog(ctx);
                if (onClamp) onClamp(ctx.meta);
            };
            ctl.onblur = commit;
            ctl.onkeydown = (ev) => {
                if (ev.key === "Enter" && !isNotes) {
                    ev.preventDefault();
                    ctl.blur();
                }
            };
            return;
        }
    };
};

// ------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------
const SLOT_PREFIXES = ["enabled_", "lora_", "model_strength_", "clip_strength_"];

const slotIndexOf = (name) => {
    for (const p of SLOT_PREFIXES) {
        if (name && name.startsWith(p)) {
            const i = parseInt(name.slice(p.length), 10);
            if (i >= 1 && i <= MAX_SLOTS) return i;
        }
    }
    return 0;
};

const nw = (node, name) => (node.widgets || []).find((w) => w.name === name) || null;

const collapse = (w) => {
    if (!w || w._ag_collapsed) return;
    w.computeSize = () => [0, -4];
    w.hidden = true;
    w._ag_collapsed = true;
};

const hookCallback = (w, fn) => {
    if (!w) return;
    const oc = w.callback ? w.callback.bind(w) : null;
    w.callback = (v) => {
        if (oc) oc(v);
        fn(v);
    };
};

const applyClampToRow = (row, meta) => {
    if (!row || !meta) return;
    const mn = parseFloat(meta.strength_min);
    const mx = parseFloat(meta.strength_max);
    for (const inp of [row.ms, row.cs]) {
        if (Number.isFinite(mn)) inp.min = String(mn);
        if (Number.isFinite(mx)) inp.max = String(mx);
    }
};

// Horizontal stepper: [◄][value][►]. click = ±0.05, Shift+click = ±0.01.
// Горизонтальный степер: [◄][значение][►]. клик = ±0.05, Shift+клик = ±0.01.
const makeStep = (titleText) => {
    const box = document.createElement("div");
    box.className = "agsoft-lora-step";
    box.title = titleText;

    const dec = document.createElement("button");
    dec.textContent = "◀";
    dec.title = "−0.05 (Shift: −0.01)";

    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "0.01";
    inp.title = titleText;

    const inc = document.createElement("button");
    inc.textContent = "▶";
    inc.title = "+0.05 (Shift: +0.01)";

    const bump = (dir, ev) => {
        const d = ev && ev.shiftKey ? 0.01 : 0.05;
        const cur = parseFloat(inp.value);
        let v = (Number.isFinite(cur) ? cur : 1) + dir * d;
        const mn = parseFloat(inp.min);
        const mx = parseFloat(inp.max);
        if (Number.isFinite(mn)) v = Math.max(mn, v);
        if (Number.isFinite(mx)) v = Math.min(mx, v);
        inp.value = Math.round(v * 100) / 100;
        inp.dispatchEvent(new Event("change"));
    };
    dec.addEventListener("click", (e) => {
        e.preventDefault();
        bump(-1, e);
    });
    inc.addEventListener("click", (e) => {
        e.preventDefault();
        bump(1, e);
    });

    box.append(dec, inp, inc);
    return { box, inp };
};

// ------------------------------------------------------------------------------
// Extension
// ------------------------------------------------------------------------------
app.registerExtension({
    name: "AGSoft.MultiLoraLoader",
    async nodeCreated(node) {
        if (node.comfyClass !== CLASS_ID) return;
        injectCss();

        const aw = nw(node, "active_loras");
        const tw = nw(node, "toggle_all");
        collapse(aw); // hidden; controlled by + Add LoRA
        collapse(tw); // hidden; driven by the custom header switch
        for (const w of node.widgets || []) {
            if (slotIndexOf(w.name) > 0) collapse(w);
        }

        const getActive = () => {
            const v = parseInt(aw ? aw.value : 0, 10);
            if (!Number.isFinite(v)) return 0;
            return Math.min(MAX_SLOTS, Math.max(0, v));
        };

        // ------------------------------------------------------------------
        // ONE single DOM widget container for the whole UI.
        // pointer-events:none on the container: empty space never captures
        // wheel/drag; interactive children re-enable pointer-events:auto.
        // ------------------------------------------------------------------
        const root = document.createElement("div");
        root.className = "agsoft-lora-root";
        root.style.pointerEvents = "none";

        // Theme sync: follow node bgcolor/color on every canvas draw.
        // Синхронизация темы: следуем за bgcolor/color ноды при каждой отрисовке.
        applyTheme(node, root);
        const origDrawBg = node.onDrawBackground;
        node.onDrawBackground = function (ctx, graphcanvas) {
            if (origDrawBg) origDrawBg.apply(this, arguments);
            applyTheme(this, root);
        };

        // Header row: [Toggle All switch][label] ... [Model][CLIP][ℹ gap]
        const head = document.createElement("div");
        head.className = "agsoft-lora-headrow";
        head.style.pointerEvents = "auto";
        const headToggle = document.createElement("input");
        headToggle.type = "checkbox";
        headToggle.checked = true;
        headToggle.title = "set = all on, unset = all off";
        const headLabel = document.createElement("span");
        headLabel.className = "agsoft-lora-headlabel";
        headLabel.textContent = "Toggle All";
        const hm = document.createElement("span");
        hm.style.width = STEP_W + "px";
        hm.style.textAlign = "center";
        hm.textContent = "Model";
        const hc = document.createElement("span");
        hc.style.width = STEP_W + "px";
        hc.style.textAlign = "center";
        hc.textContent = "CLIP";
        const hsp = document.createElement("span");
        hsp.style.width = "26px"; // one icon button (ℹ)
        head.append(headToggle, headLabel, hm, hc, hsp);
        root.appendChild(head);

        const rows = [];

        function refresh() {
            const active = getActive();
            for (const row of rows) {
                row.el.style.display = row.i <= active ? "flex" : "none";
            }
            const s = node.computeSize ? node.computeSize() : null;
            if (s && Array.isArray(s)) {
                node.size[1] = s[1];
            }
            node.setDirtyCanvas(true, true);
            if (app.graph && app.graph.setDirtyCanvas) app.graph.setDirtyCanvas(true);
        }

        function setActive(v) {
            if (!aw) return;
            aw.value = Math.min(MAX_SLOTS, Math.max(0, v));
            if (aw.callback) aw.callback(aw.value);
            refresh();
        }

        function resetSlot(idx) {
            const lEn = nw(node, `enabled_${idx}`);
            const lLo = nw(node, `lora_${idx}`);
            const lMs = nw(node, `model_strength_${idx}`);
            const lCs = nw(node, `clip_strength_${idx}`);
            if (lEn) lEn.value = true;
            if (lLo) lLo.value = (lLo.options && lLo.options.values && lLo.options.values[0]) || "None";
            if (lMs) lMs.value = 1;
            if (lCs) lCs.value = 1;
            if (rows[idx - 1]) {
                rows[idx - 1].sync();
                rows[idx - 1].refreshClamp();
            }
        }

        function deleteRow(idx) {
            const active = getActive();
            if (idx < 1 || idx > active) return;
            for (let j = idx; j < active; j++) {
                const sEn = nw(node, `enabled_${j + 1}`);
                const dEn = nw(node, `enabled_${j}`);
                const sLo = nw(node, `lora_${j + 1}`);
                const dLo = nw(node, `lora_${j}`);
                const sMs = nw(node, `model_strength_${j + 1}`);
                const dMs = nw(node, `model_strength_${j}`);
                const sCs = nw(node, `clip_strength_${j + 1}`);
                const dCs = nw(node, `clip_strength_${j}`);
                if (dEn && sEn) dEn.value = sEn.value;
                if (dLo && sLo) dLo.value = sLo.value;
                if (dMs && sMs) dMs.value = sMs.value;
                if (dCs && sCs) dCs.value = sCs.value;
                if (rows[j - 1]) {
                    rows[j - 1].sync();
                    rows[j - 1].refreshClamp();
                }
            }
            resetSlot(active);
            setActive(active - 1);
        }

        function swapSlots(a, b) {
            const fields = ["enabled_", "lora_", "model_strength_", "clip_strength_"];
            for (const p of fields) {
                const wa = nw(node, p + a);
                const wb = nw(node, p + b);
                if (wa && wb) {
                    const t = wa.value;
                    wa.value = wb.value;
                    wb.value = t;
                }
            }
            if (rows[a - 1]) {
                rows[a - 1].sync();
                rows[a - 1].refreshClamp();
            }
            if (rows[b - 1]) {
                rows[b - 1].sync();
                rows[b - 1].refreshClamp();
            }
        }

        function moveRow(idx, dir) {
            const active = getActive();
            const target = idx + dir;
            if (idx < 1 || idx > active || target < 1 || target > active) return;
            swapSlots(idx, target);
        }

        // Toggle All = SIMPLE ACTION: set -> all on, unset -> all off.
        headToggle.addEventListener("change", () => {
            const target = headToggle.checked;
            const active = getActive();
            for (let r = 1; r <= active; r++) {
                const nEn = nw(node, `enabled_${r}`);
                if (nEn) {
                    nEn.value = target;
                    if (nEn.callback) nEn.callback(nEn.value);
                }
                if (rows[r - 1]) rows[r - 1].sync();
            }
            if (tw) tw.value = target;
        });

        // ------------------------------------------------------------------
        // Slot rows: [switch][chooser][◄model►][◄clip►][ℹ] + right-click menu
        // ------------------------------------------------------------------
        function makeRow(i) {
            const nEn = nw(node, `enabled_${i}`);
            const nLo = nw(node, `lora_${i}`);
            const nMs = nw(node, `model_strength_${i}`);
            const nCs = nw(node, `clip_strength_${i}`);

            const opts = (nLo && nLo.options && nLo.options.values) || [];
            const noneVal = opts[0] || "None";
            const paths = opts.filter((o) => o && o !== noneVal);

            const el = document.createElement("div");
            el.className = "agsoft-lora-row";
            el.style.pointerEvents = "auto";

            const en = document.createElement("input");
            en.type = "checkbox";
            en.title = "on / off";

            const selBtn = document.createElement("button");
            selBtn.className = "agsoft-lora-chooser-btn";
            selBtn.title = "LoRA file (click to choose)";

            // horizontal steppers instead of native vertical spinners
            // горизонтальные степеры вместо нативных вертикальных стрелок
            const msStep = makeStep("model strength");
            const csStep = makeStep("clip strength");
            const ms = msStep.inp;
            const cs = csStep.inp;

            const info = document.createElement("button");
            info.textContent = "ℹ";
            info.style.width = "26px";
            info.title = "LoRA info (CivitAI)";

            const row = { el, en, selBtn, ms, cs, i };

            const syncFromNative = () => {
                if (nEn) en.checked = !!nEn.value;
                if (nLo) {
                    const v = String(nLo.value);
                    selBtn.textContent = (v === noneVal || v === "None") ? "None" : v;
                }
                if (nMs) ms.value = nMs.value;
                if (nCs) cs.value = nCs.value;
            };
            syncFromNative();

            const refreshClamp = async () => {
                const name = nLo ? String(nLo.value) : "";
                if (!name || name === noneVal || name === "None") return;
                const light = await getMetaLight(name);
                applyClampToRow(row, light.meta);
            };

            const openInfo = () => {
                const name = nLo ? String(nLo.value) : "";
                if (!name || name === noneVal || name === "None") {
                    openInfoDialog("", null);
                    return;
                }
                openInfoDialog(name, (meta) => applyClampToRow(row, meta));
            };

            const pick = (v) => {
                if (!nLo) return;
                nLo.value = v;
                if (nLo.callback) nLo.callback(nLo.value);
                row.sync();
                row.refreshClamp();
            };

            selBtn.addEventListener("click", (e) => {
                e.preventDefault();
                showChooser(selBtn, paths, noneVal, nLo ? String(nLo.value) : "", pick);
            });
            en.addEventListener("change", () => {
                if (nEn) {
                    nEn.value = en.checked;
                    if (nEn.callback) nEn.callback(nEn.value);
                }
            });
            ms.addEventListener("change", () => {
                if (nMs) {
                    nMs.value = parseFloat(ms.value) || 0;
                    if (nMs.callback) nMs.callback(nMs.value);
                }
            });
            cs.addEventListener("change", () => {
                if (nCs) {
                    nCs.value = parseFloat(cs.value) || 0;
                    if (nCs.callback) nCs.callback(nCs.value);
                }
            });
            info.addEventListener("click", (e) => {
                e.preventDefault();
                openInfo();
            });

            // Context menu: right-click on the row
            el.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                const active = getActive();
                const isOn = !!(nEn && nEn.value);
                showRowMenu(e.clientX, e.clientY, [
                    { label: "ℹ️ Show Info", cb: openInfo },
                    { label: isOn ? "⚫ Toggle Off" : "🟢 Toggle On", cb: () => {
                        const v = !isOn;
                        if (nEn) {
                            nEn.value = v;
                            if (nEn.callback) nEn.callback(v);
                        }
                        row.sync();
                    } },
                    { label: "⬆️ Move Up", disabled: i <= 1, cb: () => moveRow(i, -1) },
                    { label: "⬇️ Move Down", disabled: i >= active, cb: () => moveRow(i, +1) },
                    { label: "🗑️ Remove", cb: () => deleteRow(i) },
                ]);
            });

            hookCallback(nEn, (v) => { en.checked = !!v; });
            hookCallback(nLo, (v) => {
                const s = String(v);
                selBtn.textContent = (s === noneVal || s === "None") ? "None" : s;
            });
            hookCallback(nMs, (v) => { ms.value = v; });
            hookCallback(nCs, (v) => { cs.value = v; });

            el.append(en, selBtn, msStep.box, csStep.box, info);
            row.sync = syncFromNative;
            row.refreshClamp = refreshClamp;
            return row;
        }

        for (let i = 1; i <= MAX_SLOTS; i++) {
            const row = makeRow(i);
            rows.push(row);
            root.appendChild(row.el);
        }

        // Bottom: only "+ Add LoRA"
        const btns = document.createElement("div");
        btns.className = "agsoft-lora-btns";
        btns.style.pointerEvents = "auto";
        const addBtn = document.createElement("button");
        addBtn.textContent = "+ Add LoRA";
        addBtn.addEventListener("click", (e) => {
            e.preventDefault();
            setActive(getActive() + 1);
        });
        btns.appendChild(addBtn);
        root.appendChild(btns);

        // ------------------------------------------------------------------
        // Single DOM widget; height = exact visible content height.
        // ------------------------------------------------------------------
        const uiHeight = (a) => HEAD_H + BTNS_H + a * ROW_H + (a + 1) * GAP;
        const dw = node.addDOMWidget("agsoft_lora_ui", "div", root, { serialize: false });
        dw.computeSize = (w) => [w || 200, uiHeight(getActive())];

        // ------------------------------------------------------------------
        // Restore / init
        // ------------------------------------------------------------------
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            setTimeout(() => {
                applyTheme(node, root);
                for (const row of rows) {
                    row.sync();
                    row.refreshClamp();
                }
                refresh();
            }, 0);
        };

        refresh();
        setTimeout(() => {
            applyTheme(node, root);
            for (const row of rows) {
                row.sync();
                row.refreshClamp();
            }
            refresh();
        }, 0);
    },
});