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
// - preview revive only for AGSoft preview containers.
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
// - оживление превью только для превью-контейнеров AGSoft.
//
// Author: AGSoft
// Date: 30.08.2026
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Video Save] JS extension loaded v32.01 (preview layout sync, scoped revive, no double embed, safer drop restore)");

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

app.registerExtension({
    name: "AGSoft.VideoSave",

    setup() {
        bindWorkflowDrop();
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftVideoSave") return;

        // ----------------------------------------------------------------------
        // Preview container
        // ----------------------------------------------------------------------
        const wrap = document.createElement("div");
        wrap.className = "agsoft-save-preview";
        wrap.style.width = "100%";
        wrap.style.height = "100%";
        wrap.style.overflow = "hidden";

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

        wrap.appendChild(videoEl);
        wrap.appendChild(imgEl);

        let previewWidget = null;

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
        // Show preview
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
            };
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

        const origComputeSize = node.computeSize ? node.computeSize.bind(node) : null;

        node.computeSize = function (...args) {
            const s = origComputeSize ? origComputeSize(...args) : [this.size[0], this.size[1]];

            if (Array.isArray(s)) {
                s[1] = Math.max(0, s[1] - previewExtra);
            }

            return s;
        };

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
            previewExtra = 0;

            const baseSize = node.computeSize ? node.computeSize() : [currentW, 0];
            const baseH = Math.max(0, Number(baseSize && baseSize[1]) || 0);

            baseHeight = baseH;

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