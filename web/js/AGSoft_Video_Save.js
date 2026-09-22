// ==============================================================================
// AGSoft_Video_Save.js
// ==============================================================================
// JS extension for the 🎬AGSoft Video Save node.
//
// Features:
// - in-node result preview;
// - serialized preview that survives page reload;
// - Load Video-style vertical resize;
// - workflow embedding from browser;
// - workflow restore by dragging saved video;
// - safe passthrough for files without embedded workflow;
// - preview revive only for AGSoft preview containers;
// - v3.02: button row OVER the preview [💾 Save + Meta] / [💾 Save] (copy /
//   -c copy remux, NO re-encode) + info footer under the preview;
// - v3.04: spoiler arrow — all settings widgets collapse; only the save
//   buttons, the preview and the info footer stay visible; state is stored
//   in node.properties and survives reloads;
// - v3.05: a NEW node is created fully EXPANDED (users must see the
//   settings); the arrow is a state indicator (▲ = collapsed, ▼ = expanded);
//   save buttons got the video icon: [💾🎬 Save + Meta] / [💾🎬 Save].
//
// ---
//
// JS-расширение для ноды 🎬AGSoft Video Save.
//
// Возможности:
// - превью результата внутри ноды;
// - сериализуемое превью, переживающее перезагрузку страницы;
// - вертикальный ресайз как в Load Video;
// - вшивка воркфлоу из браузера;
// - восстановление воркфлоу перетаскиванием сохранённого видео;
// - безопасная передача файла дальше, если воркфлоу внутри нет;
// - оживление превью только для превью-контейнеров AGSoft;
// - v3.02: панель кнопок НАД превью [💾 Save + Meta] / [💾 Save] (копия /
//   ремукс -c copy БЕЗ перекодирования) + строка информации под превью;
// - v3.04: спойлер-стрелка — все виджеты настроек сворачиваются; видимыми
//   остаются кнопки сохранения, превью и строка информации; состояние
//   хранится в node.properties и переживает перезагрузку;
// - v3.05: НОВАЯ нода создаётся полностью РАЗВЁРНУТОЙ (пользователь должен
//   видеть настройки); стрелка — индикатор состояния (▲ = свёрнуто,
//   ▼ = развернуто); кнопки сохранения получили иконку видео:
//   [💾🎬 Save + Meta] / [💾 Save].
//
// Author: AGSoft
// Date: 22.09.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Video Save] JS extension loaded v3.05 (new node expanded, arrow-state spoiler, video icons on save buttons)");

// ------------------------------------------------------------------------------
// Простой toast (как в Save Image Plus).
// Simple toast (like in Save Image Plus).
// ------------------------------------------------------------------------------
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

// ------------------------------------------------------------------------------
// Контейнеры, где браузер может не сыграть звук — используем живой транскод.
// ------------------------------------------------------------------------------
const TRANSCODE_EXT = ["mkv", "avi", "ts", "m2ts", "vob", "flv", "wmv", "mpg", "mpeg"];
const ANY_VIDEO_RE = /\.(mp4|mkv|webm|mov|m4v|avi)$/i;
const META_VIDEO_RE = /\.(mp4|m4v|mov|mkv|webm)$/i;
// Не пытаемся вытаскивать воркфлоу из огромных файлов.
// Слишком большие файлы передаются дальше в ComfyUI без анализа.
const MAX_DROP_RESTORE_BYTES = 2 * 1024 * 1024 * 1024;

// ------------------------------------------------------------------------------
// Revive теперь работает только внутри превью AGSoft, а не для всех медиа.
// ------------------------------------------------------------------------------
(function bindPreviewRevive() {
    const MEDIA_SEL = ".agsoft-save-preview video, .agsoft-save-preview audio";
    document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") {
            for (const el of document.querySelectorAll(MEDIA_SEL)) {
                if (el.dataset) {
                    el.dataset.agsoftWasPlaying = el.paused ? "0" : "1";
                }
            }
            return;
        }
        setTimeout(() => {
            for (const el of document.querySelectorAll(MEDIA_SEL)) {
                if (!(el.currentSrc || el.src)) continue;
                const dead =
                    el.error ||
                    el.ended ||
                    el.readyState <= 1 ||
                    el.networkState === 3;
                if (dead) {
                    const count = parseInt(el.dataset.agsoftReviveCount || "0", 10);
                    if (count >= 3) {
                        continue;
                    }
                    el.dataset.agsoftReviveCount = String(count + 1);
                    el.load();
                }
                if (el.dataset && el.dataset.agsoftWasPlaying === "1") {
                    el.play().catch(() => {});
                }
            }
        }, 100);
    });
    document.addEventListener(
        "playing",
        (e) => {
            const el = e.target;
            if (el && el.dataset) {
                el.dataset.agsoftReviveCount = "0";
            }
        },
        true
    );
})();

// ------------------------------------------------------------------------------
// Разворот обёртки { "workflow": <graph> } → чистый граф.
// ------------------------------------------------------------------------------
const unwrapWorkflow = (wf) => {
    if (wf && typeof wf === "object" && wf.workflow && !wf.nodes) {
        return wf.workflow;
    }
    return wf;
};

// ------------------------------------------------------------------------------
// Node under cursor.
// ------------------------------------------------------------------------------
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
        const nodes = canvas.graph.nodes || [];
        for (let i = nodes.length - 1; i >= 0; i--) {
            const n = nodes[i];
            if (
                n &&
                n.pos &&
                n.size &&
                pos[0] >= n.pos[0] &&
                pos[0] <= n.pos[0] + n.size[0] &&
                pos[1] >= n.pos[1] &&
                pos[1] <= n.pos[1] + n.size[1]
            ) {
                return n;
            }
        }
        return null;
    } catch (err) {
        return null;
    }
};

let workflowDropBound = false;
const redispatchedDrops = new WeakSet();
const passthroughDrop = (e, file) => {
    try {
        const dt = new DataTransfer();
        dt.items.add(file);
        const evt = new DragEvent("drop", {
            bubbles: true,
            cancelable: true,
            clientX: e.clientX,
            clientY: e.clientY,
            dataTransfer: dt,
        });
        redispatchedDrops.add(evt);
        (e.target || document.body).dispatchEvent(evt);
    } catch (err) {
        console.warn("[AGSoft Video Save] drop passthrough failed:", err);
    }
};

const bindWorkflowDrop = () => {
    if (workflowDropBound) return;
    workflowDropBound = true;
    window.addEventListener(
        "drop",
        (e) => {
            if (redispatchedDrops.has(e)) return;
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file) return;
            const n = nodeAtEvent(e);
            const cls = ((n && n.comfyClass) || "") + " " + ((n && n.title) || "");
            // Если тащат на LoadVideo / LoadAudio — не мешаем штатной загрузке.
            if (n && /LoadVideo|LoadAudio/i.test(cls)) {
                return;
            }
            if (!ANY_VIDEO_RE.test(file.name)) {
                return;
            }
            // Для не-мета контейнеров и слишком больших файлов — просто отдаём
            // drop обратно в ComfyUI, не пытаясь читать воркфлоу.
            if (!META_VIDEO_RE.test(file.name) || file.size > MAX_DROP_RESTORE_BYTES) {
                e.preventDefault();
                e.stopPropagation();
                passthroughDrop(e, file);
                return;
            }
            e.preventDefault();
            e.stopPropagation();
            console.log("[AGSoft Video Save] video drop intercepted:", file.name);
            (async () => {
                let wf = null;
                try {
                    const body = new FormData();
                    body.append("file", file, file.name);
                    const resp = await fetch(api.apiURL("/agsoft/extract_workflow"), {
                        method: "POST",
                        body,
                    });
                    if (resp.ok) {
                        const data = await resp.json();
                        wf = unwrapWorkflow(data && data.workflow);
                    } else {
                        console.warn("[AGSoft Video Save] extract_workflow HTTP", resp.status);
                    }
                } catch (err) {
                    console.warn("[AGSoft Video Save] workflow restore failed:", err);
                }
                console.log("[AGSoft Video Save] extracted workflow:", !!wf);
                if (wf && app.loadGraphData) {
                    await app.loadGraphData(wf);
                    console.log("[AGSoft Video Save] workflow loaded from", file.name);
                } else {
                    console.warn(
                        "[AGSoft Video Save] no embedded workflow in",
                        file.name,
                        "-> passthrough to ComfyUI"
                    );
                    passthroughDrop(e, file);
                }
            })();
        },
        true
    );
};

// ------------------------------------------------------------------------------
// Формат размера файла для строки информации.
// File size formatting for the info footer.
// ------------------------------------------------------------------------------
const fmtSize = (b) => {
    if (!b) return "";
    if (b > 1024 * 1024 * 1024) return (b / 1024 / 1024 / 1024).toFixed(2) + " GB";
    if (b > 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + " MB";
    return Math.round(b / 1024) + " KB";
};

// ------------------------------------------------------------------------------
// Текст строки информации: W×H • fps • длительность • кодек • звук • размер.
// Info footer text: W×H • fps • duration • codec • audio • size.
// ------------------------------------------------------------------------------
const footerText = (p) => {
    if (!p) return "";
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

app.registerExtension({
    name: "AGSoft.VideoSave",
    setup() {
        bindWorkflowDrop();
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftVideoSave") return;
        // ----------------------------------------------------------------------
        // Preview container: [button row] / [media] / [info footer]
        // Контейнер превью: [панель кнопок] / [медиа] / [строка информации]
        // ----------------------------------------------------------------------
        const wrap = document.createElement("div");
        wrap.className = "agsoft-save-preview";
        Object.assign(wrap.style, {
            width: "100%",
            height: "100%",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
        });
        // Панель кнопок над превью (как в Save Image Plus).
        // Button row over the preview (like in Save Image Plus).
        const topBar = document.createElement("div");
        Object.assign(topBar.style, {
            flex: "none",
            display: "flex",
            gap: "4px",
            paddingBottom: "4px",
        });
        const mkBtn = (label, title) => {
            const b = document.createElement("button");
            b.textContent = label;
            b.title = title;
            Object.assign(b.style, {
                flex: "1",
                height: "22px",
                background: "#2a6",
                color: "#fff",
                border: "none",
                borderRadius: "3px",
                fontSize: "11px",
                cursor: "pointer",
            });
            return b;
        };
        // Спойлер-стрелка: индикатор состояния виджетов настроек.
        // ▲ = свёрнуто (виджеты скрыты), ▼ = развернуто (виджеты видны).
        // Spoiler arrow: state indicator of the settings widgets.
        // ▲ = collapsed (widgets hidden), ▼ = expanded (widgets visible).
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
        const btnMeta = mkBtn(
            "💾🎬 Save + Meta",
            "Save the current video into output (copy / -c copy remux, NO re-encode) WITH the workflow embedded.\n" +
            "Сохранить текущее видео в output (копия / ремукс -c copy, БЕЗ перекодирования) СО вшивкой воркфлоу."
        );
        const btnPlain = mkBtn(
            "💾🎬 Save",
            "Save the current video into output (plain copy, NO re-encode) WITHOUT metadata.\n" +
            "Сохранить текущее видео в output (обычная копия, БЕЗ перекодирования) БЕЗ метаданных."
        );
        topBar.appendChild(toggleBtn);
        topBar.appendChild(btnMeta);
        topBar.appendChild(btnPlain);
        // Медиа-область (видео / картинка).
        // Media area (video / image).
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
        videoEl.style.width = "100%";
        videoEl.style.height = "100%";
        videoEl.style.objectFit = "contain";
        videoEl.style.backgroundColor = "#000";
        videoEl.style.display = "none";
        videoEl.onmouseenter = () => {
            videoEl.muted = false;
        };
        videoEl.onmouseleave = () => {
            videoEl.muted = true;
        };
        const imgEl = document.createElement("img");
        imgEl.style.width = "100%";
        imgEl.style.height = "100%";
        imgEl.style.objectFit = "contain";
        imgEl.style.display = "none";
        media.appendChild(videoEl);
        media.appendChild(imgEl);
        // Строка информации под превью (реальные W×H и т.д.).
        // Info footer under the preview (real W×H etc.).
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
        foot.textContent = "";
        wrap.appendChild(topBar);
        wrap.appendChild(media);
        wrap.appendChild(foot);

        let previewWidget = null;

        // ----------------------------------------------------------------------
        // Спойлер виджетов: имена сворачиваемых виджетов настроек.
        // Widget spoiler: names of the collapsible settings widgets.
        // ----------------------------------------------------------------------
        const COLLAPSIBLE = new Set([
            "filename_prefix",
            "subfolder",
            "video_path",
            "frame_rate",
            "format",
            "preset",
            "crf",
            "save_video",
            "save_video_with_audio",
            "save_audio",
            "save_image",
            "save_metadata",
            "save_output",
        ]);
        let collapsed = false;
        const setWidgetsVisible = (visible) => {
            for (const w of node.widgets || []) {
                if (!COLLAPSIBLE.has(w.name)) continue;
                w.hidden = !visible;
                if (visible) {
                    if (typeof w.show === "function") {
                        try { w.show(); } catch (e) {}
                    }
                } else {
                    if (typeof w.hide === "function") {
                        try { w.hide(); } catch (e) {}
                    }
                }
            }
        };
        const updateToggle = () => {
            // Стрелка = состояние: ▲ свёрнуто, ▼ развернуто.
            // Arrow = state: ▲ collapsed, ▼ expanded.
            toggleBtn.textContent = collapsed ? "▲" : "▼";
            toggleBtn.title = collapsed
                ? "Settings hidden — click to expand\nНастройки скрыты — нажмите, чтобы развернуть"
                : "Settings shown — click to collapse\nНастройки видны — нажмите, чтобы свернуть";
        };

        // ----------------------------------------------------------------------
        // Чтение настроек виджетов (имя/подпапка) для кнопок сохранения.
        // Read widget settings (prefix/subfolder) for the save buttons.
        // ----------------------------------------------------------------------
        const readParams = () => {
            const w = (name) => {
                const widget = node.widgets && node.widgets.find((x) => x.name === name);
                return widget ? widget.value : null;
            };
            return {
                filename_prefix: w("filename_prefix") || "AGSoft_Video",
                subfolder: w("subfolder") || "",
            };
        };

        // ----------------------------------------------------------------------
        // Сохранение текущего превью/результата в output БЕЗ перекодирования:
        // сервер копирует файл или делает быстрый ремукс -c copy с метаданными.
        // Save the current preview/result into output WITHOUT re-encoding:
        // the server copies the file or makes a fast -c copy remux with meta.
        // ----------------------------------------------------------------------
        const saveNow = async (withMeta) => {
            const saved = previewWidget && previewWidget._saved;
            if (!saved || !(saved.path || saved.fullpath || saved.filename)) {
                showToast("No video preview to save yet.", true);
                return;
            }
            try {
                const resp = await fetch(api.apiURL("/agsoft/video_save_now"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        path: saved.path || saved.fullpath || "",
                        filename: saved.filename || "",
                        type: saved.type || "temp",
                        subfolder: saved.subfolder || "",
                        params: readParams(),
                        embed_metadata: !!withMeta,
                        // Воркфлоу из браузера — чистый граф.
                        // Workflow from the browser — pure graph.
                        workflow: withMeta ? app.graph.serialize() : null,
                    }),
                });
                const data = await resp.json();
                if (data && data.ok) {
                    showToast(`Saved: ${data.filename}`, false);
                    console.log("[AGSoft Video Save] saved:", data);
                } else {
                    showToast(`Save failed: ${(data && data.error) || "unknown"}`, true);
                }
            } catch (err) {
                console.warn("[AGSoft Video Save] video_save_now error:", err);
                showToast(`Save failed: ${err.message}`, true);
            }
        };
        btnMeta.onclick = () => saveNow(true);
        btnPlain.onclick = () => saveNow(false);

        // ----------------------------------------------------------------------
        // Если живой транскод не поднялся, пробуем прямой стрим файла.
        // ----------------------------------------------------------------------
        const fallbackToStream = () => {
            try {
                if (!previewWidget || !previewWidget._saved) return;
                const saved = previewWidget._saved;
                if (saved.kind !== "path" || !saved.path) return;
                if (!videoEl.src.includes("/agsoft/preview_path")) return;
                if (videoEl.dataset.agsoftFallbackDone === "1") return;
                videoEl.dataset.agsoftFallbackDone = "1";
                const url = api.apiURL(`/agsoft/stream_path?path=${encodeURIComponent(saved.path)}`);
                videoEl.src = url;
                videoEl.load();
            } catch (err) {
                console.warn("[AGSoft Video Save] preview fallback failed:", err);
            }
        };
        videoEl.addEventListener("error", fallbackToStream);

        // ----------------------------------------------------------------------
        // Show preview + info footer
        // ----------------------------------------------------------------------
        const showPreview = (p, fromRestore) => {
            if (!p) return;
            let url = null;
            if (p.kind === "path" && p.path) {
                const ext = (p.ext || (p.path.split(".").pop() || "")).toLowerCase();
                url = TRANSCODE_EXT.includes(ext)
                    ? api.apiURL(`/agsoft/preview_path?path=${encodeURIComponent(p.path)}`)
                    : api.apiURL(`/agsoft/stream_path?path=${encodeURIComponent(p.path)}`);
            } else if (p.filename) {
                const params = new URLSearchParams({
                    filename: p.filename,
                    type: p.type || "output",
                    subfolder: p.subfolder || "",
                });
                if (p.format) params.set("format", p.format);
                if (!fromRestore) {
                    params.set("timestamp", String(Date.now()));
                }
                url = api.apiURL("/view?" + params.toString());
            } else {
                return;
            }
            if ((p.format || "").startsWith("image")) {
                videoEl.style.display = "none";
                videoEl.removeAttribute("src");
                imgEl.style.display = "block";
                imgEl.src = url;
            } else {
                imgEl.style.display = "none";
                imgEl.removeAttribute("src");
                videoEl.style.display = "block";
                videoEl.dataset.agsoftFallbackDone = "";
                videoEl.src = url;
                videoEl.load();
                try {
                    videoEl.pause();
                } catch (e) {}
            }
            previewWidget._saved = {
                kind: p.kind || "file",
                filename: p.filename || "",
                subfolder: p.subfolder || "",
                type: p.type || "output",
                format: p.format || "",
                path: p.path || "",
                ext: p.ext || "",
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
        };

        // ----------------------------------------------------------------------
        // Preview widget
        // ----------------------------------------------------------------------
        previewWidget = node.addDOMWidget(
            "agsoft_save_preview",
            "div",
            wrap,
            {
                serialize: true,
                hideOnZoom: false,
                getValue() {
                    return previewWidget._saved || null;
                },
                setValue(v) {
                    previewWidget._saved = v || null;
                    if (v && (v.filename || v.path)) {
                        showPreview(v, true);
                    } else if (v) {
                        foot.textContent = footerText(v);
                    }
                },
            }
        );

        // ----------------------------------------------------------------------
        // Resize logic
        // ----------------------------------------------------------------------
        const MIN_PREVIEW_H = 120;
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
        // Базовая высота при ТЕКУЩЕЙ видимости виджетов (previewExtra = 0).
        // Base height for the CURRENT widget visibility (previewExtra = 0).
        const measureBase = () => {
            previewExtra = 0;
            const bs = node.computeSize ? node.computeSize() : [node.size[0], 0];
            const b = Math.max(0, Number(bs && bs[1]) || 0);
            baseHeight = b;
            return b;
        };
        // Применение состояния спойлера + пересчёт высоты ноды.
        // keepExtra: сохранить растяжку превью; heightHint: высота из workflow.
        // Apply spoiler state + recompute the node height.
        // keepExtra: preserve the preview stretch; heightHint: workflow height.
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

        // ----------------------------------------------------------------------
        // onExecuted: preview + embed workflow only if needed
        // ----------------------------------------------------------------------
        const origExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            const gifs = output && output.gifs;
            if (!gifs || !gifs.length) return;
            const p = gifs[0];
            showPreview(p, false);
            const w = node.widgets && node.widgets.find((x) => x.name === "save_metadata");
            const canEmbed =
                w &&
                w.value &&
                p &&
                p.fullpath &&
                p.kind !== "path" &&
                !p.noop &&
                !p.workflow_embedded &&
                p.supports_workflow !== false;
            if (canEmbed) {
                fetch(api.apiURL("/agsoft/embed_workflow"), {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        path: p.fullpath,
                        workflow: app.graph.serialize(),
                        prompt: null,
                    }),
                })
                    .then((r) => r.json())
                    .then((d) => console.log("[AGSoft Video Save] workflow embedded into", p.filename, d))
                    .catch((e) => console.warn("[AGSoft Video Save] workflow embed failed:", e));
            }
        };

        // ----------------------------------------------------------------------
        // Vertical resize / layout sync (Load Video style)
        // Вертикальный ресайз / синхронизация раскладки (как в Load Video)
        // ----------------------------------------------------------------------
        const syncLayout = (preserveUserHeight = true) => {
            const currentW = Math.max(220, Number(node.size && node.size[0]) || 220);
            const currentH = Math.max(0, Number(node.size && node.size[1]) || 0);
            // Temporarily reset extra height to measure the true base height.
            // Временно сбрасываем extra, чтобы измерить реальную базовую высоту.
            const baseH = measureBase();
            let targetH = currentH;
            // If the restored/current height is too small for the new widget set,
            // force the correct minimum base height.
            // Если восстановленной/текущей высоты мало для нового набора виджетов,
            // принудительно ставим правильную минимальную базовую высоту.
            if (!preserveUserHeight || targetH < baseH) {
                targetH = baseH;
            }
            previewExtra = Math.max(0, targetH - baseH);
            node.setSize([currentW, targetH]);
            node.setDirtyCanvas(true, true);
        };
        syncLayout(true);
        // Стартовое состояние спойлера: НОВАЯ нода = полностью развёрнута
        // (пользователь должен видеть настройки); загруженная нода = состояние
        // из node.properties. Превью новой ноды получает +300px растяжки.
        // Initial spoiler state: a NEW node is fully EXPANDED (the user must
        // see the settings); a loaded node = state from node.properties.
        // The new node's preview gets +300px of stretch.
        const fresh = !(
            node.properties &&
            Object.prototype.hasOwnProperty.call(node.properties, "agsoft_widgets_collapsed")
        );
        const initialCollapsed = fresh
            ? false
            : !!node.properties["agsoft_widgets_collapsed"];
        applyState(initialCollapsed, fresh ? 300 : previewExtra, null);
        // Восстановление состояния спойлера и высоты после загрузки workflow.
        // Restore spoiler state and height after the workflow is loaded.
        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            const st = !!(node.properties && node.properties["agsoft_widgets_collapsed"]);
            const h = Number(node.size && node.size[1]) || 0;
            applyState(st, null, h);
        };
        const origOnResize = node.onResize;
        node.onResize = function (size) {
            if (baseHeight == null) {
                syncLayout(true);
            }
            previewExtra = Math.max(0, size[1] - (baseHeight || 0));
            if (origOnResize) {
                origOnResize.apply(this, arguments);
            }
            node.setDirtyCanvas(true, true);
        };
        // Recalculate after all widgets are fully measured/restored.
        // This fixes the preview shifting upward after adding a new widget.
        // Пересчитываем после полного измерения/восстановления всех виджетов.
        // Это фиксит смещение превью вверх после добавления нового виджета.
        setTimeout(() => {
            syncLayout(true);
            app.graph.setDirtyCanvas(true);
        }, 0);
    },
});