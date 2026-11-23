// ==============================================================================
// AGSoft_Video_Concatenate.js
// ==============================================================================
// JS extension for 🎬AGSoft Video Concatenate / 🎬AGSoft Video Concatenate Plus.
// Preview architecture mirrors 🎬AGSoft Video Save.
//
// Features:
// - one extension serves both node classes (AGSoftVideoConcatenate,
//   AGSoftVideoConcatenatePlus);
// - preview container: [top bar: spoiler arrow + green Save button] /
//   [3px progress strip] / [media flex area] / [16px info footer];
// - info footer UNDER the preview: real W×H • fps • duration • codec •
//   audio track • file size (from the server probe in ui.gifs);
// - serialized preview (DOM widget serialize:true, getValue/setValue):
//   survives page reload; output-type previews survive restarts too;
// - onExecuted reads ui.gifs and shows the result (no custom result events);
// - render progress via WS agsoft_concat_progress: strip fill + stage text;
// - Save button OVER the preview: POST /agsoft/concat_save copies the temp
//   result into output (NO re-encode), then the preview switches to the
//   saved output copy and the button disables; toast on success/error;
// - spoiler arrow collapses ALL settings widgets (except the preview and
//   the always-hidden clip_N_path); state lives in
//   node.properties["agsoft_widgets_collapsed"] and survives reloads;
//   ▲ = collapsed, ▼ = expanded; a NEW node is created fully EXPANDED
//   with +300px of preview stretch;
// - Load Video-style vertical resize: measureBase / previewExtra /
//   syncLayout / onResize keep the user's node width and height when
//   widget values change (no collapse to the computeSize minimum);
// - socket trim under inputs_count: only UNCONNECTED excess video_N inputs
//   are removed, missing ones re-added, link target_slot renumbered;
//   connected links are never broken;
// - clip_N_path widgets are always hidden (w.hidden + w.y = -9999) and are
//   filled only by drag&drop uploads into the first free clip slot;
// - drag&drop of video files onto the node: canvas dragover/dragleave/drop,
//   node hit-test via convertEventToCanvasOffset + bounding-box fallback,
//   dashed #7fd4ff highlight, 600ms double-drop guard, video/* validation,
//   upload via /upload/image into ComfyUI input;
// - preview audio: unmuted on mouse enter, muted on mouse leave (as in
//   Video Save); loop + native controls;
// - onConfigure restores spoiler state, socket set and workflow height.
//
// ---
//
// JS-расширение для 🎬AGSoft Video Concatenate / 🎬AGSoft Video Concatenate Plus.
// Архитектура превью повторяет 🎬AGSoft Video Save.
//
// Возможности:
// - одно расширение обслуживает оба класса нод (AGSoftVideoConcatenate,
//   AGSoftVideoConcatenatePlus);
// - контейнер превью: [панель: стрелка-спойлер + зелёная кнопка Save] /
//   [полоса прогресса 3px] / [медиа-область flex] / [инфо-футер 16px];
// - инфо-футер ПОД превью: реальные W×H • fps • длительность • кодек •
//   звук • размер файла (из серверного probe в ui.gifs);
// - сериализуемое превью (DOM-виджет serialize:true, getValue/setValue):
//   переживает перезагрузку страницы; превью типа output — и рестарт;
// - onExecuted читает ui.gifs и показывает результат (без кастомных событий);
// - прогресс рендера через WS agsoft_concat_progress: заполнение полосы и
//   текст стадии;
// - кнопка Save НАД превью: POST /agsoft/concat_save копирует temp-результат
//   в output (БЕЗ перекодирования), затем превью переключается на сохранённую
//   копию и кнопка блокируется; toast при успехе/ошибке;
// - стрелка-спойлер сворачивает ВСЕ виджеты настроек (кроме превью и всегда
//   скрытых clip_N_path); состояние хранится в
//   node.properties["agsoft_widgets_collapsed"] и переживает перезагрузку;
//   ▲ = свёрнуто, ▼ = развёрнуто; НОВАЯ нода создаётся полностью РАЗВЁРНУТОЙ
//   с +300px растяжки превью;
// - вертикальный ресайз как в Load Video: measureBase / previewExtra /
//   syncLayout / onResize сохраняют пользовательские ширину и высоту ноды
//   при изменении значений виджетов (без схлопывания до минимума);
// - обрезка сокетов под inputs_count: удаляются только НЕподключённые лишние
//   входы video_N, недостающие добавляются, target_slot линков
//   перенумеровывается; подключённые линки не рвутся;
// - виджеты clip_N_path всегда скрыты (w.hidden + w.y = -9999) и заполняются
//   только drag&drop-загрузкой в первый свободный слот клипа;
// - drag&drop видеофайлов на ноду: canvas dragover/dragleave/drop, поиск ноды
//   через convertEventToCanvasOffset + fallback по bounding-box, пунктирная
//   подсветка #7fd4ff, защита от двойного drop 600 мс, валидация video/*,
//   загрузка через /upload/image в input ComfyUI;
// - звук превью: включается при наведении мыши, выключается при уходе
//   (как в Video Save); loop + нативные контролы;
// - onConfigure восстанавливает состояние спойлера, набор сокетов и высоту
//   из воркфлоу.
//
// Author: AGSoft
// Date: 23.11.2026
// ==============================================================================


import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Video Concatenate] JS extension loaded v11.23 " +
    "(Video Save-style preview, footer, spoiler, stable user size)");

const CLIP_MAX = 10;
const CLASSES = new Set(["AGSoftVideoConcatenate", "AGSoftVideoConcatenatePlus"]);
const MIN_PREVIEW_H = 120;

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

const fmtSize = (b) => {
    if (!b) return "";
    if (b > 1024 * 1024 * 1024) return (b / 1024 / 1024 / 1024).toFixed(2) + " GB";
    if (b > 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + " MB";
    return Math.round(b / 1024) + " KB";
};

const footerText = (p) => {
    if (!p) return " ";
    const parts = [];
    if (p.width && p.height) parts.push(`${p.width} × ${p.height}`);
    if (p.frame_rate) parts.push(`${Number(p.frame_rate).toFixed(1)} fps`);
    if (p.duration) parts.push(`${Number(p.duration).toFixed(1)} s`);
    if (p.codec) parts.push(String(p.codec));
    if (p.has_audio) parts.push("audio: " + (p.audio_codec || "yes"));
    const sz = fmtSize(p.size_bytes);
    if (sz) parts.push(sz);
    return parts.join("  •  ");
};

const clipWidget = (node, i) => node.widgets?.find((w) => w.name === `clip_${i}_path`);

const socketLinked = (node, i) => {
    const inp = node.inputs?.find((s) => s.name === `video_${i}`);
    return inp && inp.link != null;
};

const hideClipWidgets = (node) => {
    for (let i = 1; i <= CLIP_MAX; i++) {
        const w = clipWidget(node, i);
        if (!w) continue;
        w.hidden = true;
        w.y = -9999;
    }
};

const nodeAtEvent = (e) => {
    try {
        const canvas = app.canvas;
        if (!canvas || !canvas.graph) return null;
        let pos = null;
        try {
            if (canvas.convertEventToCanvasOffset) {
                pos = canvas.convertEventToCanvasOffset(e);
            } else if (canvas.convertEventToCanvas) {
                pos = canvas.convertEventToCanvas(e);
            }
        } catch (err) {
            pos = null;
        }
        if (!pos) return null;
        const nodes = canvas.graph.nodes || canvas.graph._nodes || [];
        for (let i = nodes.length - 1; i >= 0; i--) {
            const n = nodes[i];
            if (
                n && n.pos && n.size &&
                pos[0] >= n.pos[0] && pos[0] <= n.pos[0] + n.size[0] &&
                pos[1] >= n.pos[1] && pos[1] <= n.pos[1] + n.size[1]
            ) {
                return CLASSES.has(n.comfyClass) ? n : null;
            }
        }
        return null;
    } catch (err) {
        return null;
    }
};

let lastDropAt = 0;

app.registerExtension({
    name: "AGSoft.VideoConcatenate",
    setup() {
        api.addEventListener("agsoft_concat_progress", (ev) => {
            const d = ev.detail || {};
            const node = app.graph.getNodeById(Number(d.node));
            if (!node || !node._agsUI) return;
            node._agsUI.prog.style.display = "block";
            node._agsUI.fill.style.width = `${d.percent || 0}%`;
            node._agsUI.foot.textContent = `${d.stage || "render"} ${d.percent || 0}%`;
        });
        const canvasEl = app.canvasEl || (app.canvas && app.canvas.canvas);
        if (!canvasEl) return;
        canvasEl.addEventListener("dragover", (e) => {
            const node = nodeAtEvent(e);
            if (node) {
                e.preventDefault();
                node._agsDragOver = true;
                app.canvas.setDirty(true, true);
            }
        });
        canvasEl.addEventListener("dragleave", () => {
            for (const n of app.graph._nodes || []) n._agsDragOver = false;
            app.canvas.setDirty(true, true);
        });
        canvasEl.addEventListener("drop", async (e) => {
            const now = Date.now();
            if (now - lastDropAt < 600) return;
            const node = nodeAtEvent(e);
            if (!node) return;
            const file = e.dataTransfer?.files?.[0];
            if (!file || !file.type.startsWith("video/")) return;
            e.preventDefault();
            e.stopPropagation();
            lastDropAt = now;
            node._agsDragOver = false;
            const ui = node._agsUI;
            if (ui) ui.foot.textContent = "Uploading… / Загрузка…";
            try {
                const fd = new FormData();
                fd.append("image", file);
                const resp = await fetch(api.apiURL("/upload/image"), {
                    method: "POST",
                    body: fd,
                });
                const data = await resp.json();
                const countW = node.widgets?.find((w) => w.name === "inputs_count");
                const count = parseInt(countW?.value ?? "2", 10) || 2;
                const rel = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                for (let i = 1; i <= count; i++) {
                    const w = clipWidget(node, i);
                    if (w && !String(w.value || "").trim() && !socketLinked(node, i)) {
                        w.value = rel;
                        break;
                    }
                }
                if (ui) ui.foot.textContent = `Loaded / Загружено: ${rel}`;
            } catch (err) {
                if (ui) ui.foot.textContent = "Upload error: " + err.message;
            }
        });
    },
    async nodeCreated(node) {
        if (!CLASSES.has(node.comfyClass)) return;
        // ------------------------------------------------------------------
        // Preview container: [button row] / [progress] / [media] / [footer]
        // Контейнер превью: [панель кнопок] / [прогресс] / [медиа] / [футер]
        // ------------------------------------------------------------------
        const wrap = document.createElement("div");
        wrap.className = "agsoft-concat-preview";
        Object.assign(wrap.style, {
            width: "100%",
            height: "100%",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
        });
        const topBar = document.createElement("div");
        Object.assign(topBar.style, {
            flex: "none",
            display: "flex",
            gap: "4px",
            paddingBottom: "4px",
        });
        const toggleBtn = document.createElement("button");
        Object.assign(toggleBtn.style, {
            flex: "none",
            width: "30px",
            height: "22px",
            background: "#3a3f4a",
            color: "#fff",
            border: "none",
            borderRadius: "3px",
            fontSize: "11px",
            cursor: "pointer",
        });
        const saveBtn = document.createElement("button");
        saveBtn.textContent = "💾 Save";
        saveBtn.title =
            "Copy the current temp result into output (NO re-encode).\n" +
            "Копировать текущий temp-результат в output (БЕЗ перекодирования).";
        Object.assign(saveBtn.style, {
            flex: "1",
            height: "22px",
            background: "#2a6",
            color: "#fff",
            border: "none",
            borderRadius: "3px",
            fontSize: "11px",
            cursor: "pointer",
        });
        saveBtn.disabled = true;
        topBar.appendChild(toggleBtn);
        topBar.appendChild(saveBtn);
        const prog = document.createElement("div");
        Object.assign(prog.style, {
            flex: "none",
            display: "none",
            height: "3px",
            marginBottom: "4px",
            background: "#1b1e24",
            borderRadius: "2px",
            overflow: "hidden",
        });
        const fill = document.createElement("div");
        Object.assign(fill.style, {
            height: "100%",
            width: "0%",
            background: "#7fd4ff",
        });
        prog.appendChild(fill);
        const media = document.createElement("div");
        Object.assign(media.style, {
            flex: "1 1 0",
            minHeight: "0",
            position: "relative",
            background: "#000",
            overflow: "hidden",
        });
        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = false;
        Object.assign(videoEl.style, {
            width: "100%",
            height: "100%",
            objectFit: "contain",
            backgroundColor: "#000",
        });
        videoEl.onmouseenter = () => {
            videoEl.muted = false;
        };
        videoEl.onmouseleave = () => {
            videoEl.muted = true;
        };
        media.appendChild(videoEl);
        const foot = document.createElement("div");
        Object.assign(foot.style, {
            flex: "none",
            height: "16px",
            background: "#1b1e24",
            color: "#9aa4b2",
            fontSize: "10px",
            fontFamily: "sans-serif",
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-start",
            paddingLeft: "6px",
            overflow: "hidden",
            whiteSpace: "nowrap",
        });
        foot.textContent = " ";
        wrap.appendChild(topBar);
        wrap.appendChild(prog);
        wrap.appendChild(media);
        wrap.appendChild(foot);
        node._agsUI = { prog, fill, foot, saveBtn, videoEl };
        // ------------------------------------------------------------------
        // Serialized preview widget / Сериализуемый виджет превью
        // ------------------------------------------------------------------
        let previewWidget = null;
        const showPreview = (p, fromRestore) => {
            if (!p || !p.filename) return;
            const params = new URLSearchParams({
                filename: p.filename,
                type: p.type || "temp",
                subfolder: p.subfolder || "",
            });
            if (!fromRestore) params.set("timestamp", String(Date.now()));
            videoEl.src = api.apiURL("/view?" + params.toString());
            videoEl.load();
            try {
                videoEl.pause();
            } catch (e) {}
            previewWidget._saved = {
                filename: p.filename || "",
                subfolder: p.subfolder || "",
                type: p.type || "temp",
                fullpath: p.fullpath || "",
                width: p.width || 0,
                height: p.height || 0,
                duration: p.duration || 0,
                codec: p.codec || "",
                has_audio: !!p.has_audio,
                audio_codec: p.audio_codec || "",
                size_bytes: p.size_bytes || 0,
                frame_rate: p.frame_rate || 0,
            };
            foot.textContent = footerText(previewWidget._saved);
            saveBtn.disabled = previewWidget._saved.type !== "temp";
            prog.style.display = "none";
            fill.style.width = "0%";
        };
        previewWidget = node.addDOMWidget("agsoft_concat_preview", "div", wrap, {
            serialize: true,
            hideOnZoom: false,
            getValue() {
                return previewWidget._saved || null;
            },
            setValue(v) {
                previewWidget._saved = v || null;
                if (v && v.filename) showPreview(v, true);
                else if (v) foot.textContent = footerText(v);
            },
        });
        // ------------------------------------------------------------------
        // Save button: temp -> output copy / Кнопка Save: копия temp -> output
        // ------------------------------------------------------------------
        saveBtn.onclick = async () => {
            const saved = previewWidget._saved;
            if (!saved || saved.type !== "temp") return;
            saveBtn.disabled = true;
            foot.textContent = "Saving… / Сохранение…";
            try {
                const nameW = node.widgets?.find((w) => w.name === "output_name");
                const subW = node.widgets?.find((w) => w.name === "output_path");
                const resp = await fetch(api.apiURL("/agsoft/concat_save"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        temp_path: saved.fullpath,
                        name: nameW?.value || saved.filename,
                        subfolder: subW?.value || "",
                    }),
                });
                const data = await resp.json();
                if (data.error) throw new Error(data.error);
                showToast(`Saved: ${data.name}`, false);
                showPreview({
                    ...saved,
                    filename: data.name,
                    subfolder: data.subfolder,
                    type: "output",
                    fullpath: data.path,
                }, false);
            } catch (err) {
                showToast(`Save failed: ${err.message}`, true);
                foot.textContent = footerText(saved);
                saveBtn.disabled = false;
            }
        };
        // ------------------------------------------------------------------
        // Spoiler: collapse settings widgets / Спойлер: сворачивание настроек
        // ------------------------------------------------------------------
        const collapsible = (node.widgets || [])
            .filter((w) => w !== previewWidget && !/^clip_\d+_path$/.test(w.name))
            .map((w) => w.name);
        let collapsed = false;
        const setWidgetsVisible = (visible) => {
            for (const w of node.widgets || []) {
                if (w === previewWidget) continue;
                if (/^clip_\d+_path$/.test(w.name)) {
                    w.hidden = true;
                    w.y = -9999;
                    continue;
                }
                if (!collapsible.includes(w.name)) continue;
                w.hidden = !visible;
                w.y = visible ? 0 : -9999;
            }
        };
        const updateToggle = () => {
            toggleBtn.textContent = collapsed ? "▲" : "▼";
            toggleBtn.title = collapsed
                ? "Settings hidden — click to expand\nНастройки скрыты — нажмите, чтобы развернуть"
                : "Settings shown — click to collapse\nНастройки видны — нажмите, чтобы свернуть";
        };
        // ------------------------------------------------------------------
        // Resize logic (Load Video style) / Ресайз как в Load Video
        // ------------------------------------------------------------------
        let previewExtra = 0;
        let baseHeight = null;
        previewWidget.computeSize = function (width) {
            return [width || 200, MIN_PREVIEW_H + previewExtra];
        };
        const origCompute = node.computeSize;
        const origComputeSize = origCompute
            ? function (...args) {
                  return origCompute.apply(node, args);
              }
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
        toggleBtn.onclick = () => applyState(!collapsed, previewExtra, null);
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
        // ------------------------------------------------------------------
        // Socket trim keeps user size / Обрезка сокетов сохраняет размер
        // ------------------------------------------------------------------
        const syncSockets = () => {
            const countW = node.widgets?.find((w) => w.name === "inputs_count");
            const count = parseInt(countW?.value ?? "2", 10) || 2;
            for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--) {
                const inp = node.inputs[i];
                const m = /^video_(\d+)$/.exec(inp.name);
                if (!m) continue;
                if (parseInt(m[1], 10) > count && inp.link == null) node.removeInput(i);
            }
            for (let i = 1; i <= count; i++) {
                if (!node.inputs?.some((s) => s.name === `video_${i}`)) {
                    node.addInput(`video_${i}`, "*", {});
                }
            }
            (node.inputs || []).forEach((inp, pos) => {
                if (inp.link != null) {
                    const l = app.graph.links[inp.link];
                    if (l) l.target_slot = pos;
                }
            });
            syncLayout(true);
        };
        hideClipWidgets(node);
        syncSockets();
        syncLayout(true);
        const fresh = !(
            node.properties &&
            Object.prototype.hasOwnProperty.call(node.properties, "agsoft_widgets_collapsed")
        );
        const initialCollapsed = fresh ? false : !!node.properties["agsoft_widgets_collapsed"];
        applyState(initialCollapsed, fresh ? 300 : previewExtra, null);
        const countW = node.widgets?.find((w) => w.name === "inputs_count");
        if (countW) {
            const oldCb = countW.callback;
            countW.callback = function (val) {
                if (oldCb) oldCb.call(this, val);
                syncSockets();
            };
        }
        // ------------------------------------------------------------------
        // onExecuted: preview from ui.gifs / Превью из ui.gifs
        // ------------------------------------------------------------------
        const origExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            const gifs = output && output.gifs;
            if (!gifs || !gifs.length) return;
            showPreview(gifs[0], false);
            syncLayout(true);
        };
        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            hideClipWidgets(node);
            syncSockets();
            const st = !!(node.properties && node.properties["agsoft_widgets_collapsed"]);
            const h = Number(node.size && node.size[1]) || 0;
            applyState(st, null, h);
        };
        const origOnResize = node.onResize;
        node.onResize = function (size) {
            if (baseHeight == null) syncLayout(true);
            previewExtra = Math.max(0, size[1] - (baseHeight || 0));
            if (origOnResize) origOnResize.apply(this, arguments);
            node.setDirtyCanvas(true, true);
        };
        const origDraw = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origDraw) origDraw.apply(this, arguments);
            if (this._agsDragOver) {
                ctx.save();
                ctx.strokeStyle = "#7fd4ff";
                ctx.setLineDash([6, 4]);
                ctx.lineWidth = 2;
                ctx.strokeRect(0, 0, this.size[0], this.size[1]);
                ctx.restore();
            }
        };
        setTimeout(() => {
            syncLayout(true);
            app.graph.setDirtyCanvas(true);
        }, 0);
    },
});