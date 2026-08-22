// ==============================================================================
// AGSoft_Video_Save.js
// ==============================================================================
// JS-расширение для ноды 🎬AGSoft Video Save.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Video Save] JS extension loaded v30.08 (no-op preview WITH sound + serialized preview + PURE-graph embed + video workflow restore + LoadVideo-style resize + global revive)");

// ------------------------------------------------------------------------------
// Контейнеры, где браузер скорее всего не сможет играть звук (AC3/DTS) —
// для них используем живой транскод /agsoft/preview_path.
// Containers where the browser most likely cannot play audio (AC3/DTS) —
// use live transcode /agsoft/preview_path for them.
// ------------------------------------------------------------------------------
const TRANSCODE_EXT = ["mkv", "avi", "ts", "m2ts", "vob", "flv", "wmv", "mpg", "mpeg"];

// ------------------------------------------------------------------------------
// Авто-«оживление» превью (video/audio) после возврата на вкладку,
// разворачивания браузера и после обрыва фоновых потоков.
// Работает для ВСЕХ нод AGSoft: при скрытии запоминает, что играло;
// при возврате — перезапускает умершее медиа (load()) и возобновляет игру.
//
// Universal preview "revive" (video/audio) after returning to the tab,
// restoring the browser, or after background streams were dropped.
// Works for ALL AGSoft nodes. 
// ------------------------------------------------------------------------------
(function bindPreviewRevive() {
    const MEDIA_SEL = "video, audio";

    document.addEventListener("visibilitychange", () => {
        // Скрылись: запоминаем, что играло.
        // Hidden: remember what was playing.
        if (document.visibilityState === "hidden") {
            for (const el of document.querySelectorAll(MEDIA_SEL)) {
                if (el.dataset) el.dataset.agsoftWasPlaying = el.paused ? "0" : "1";
            }
            return;
        }

        // Вернулись: оживляем умершие превью.
        // Visible: revive dead previews.
        setTimeout(() => {
            for (const el of document.querySelectorAll(MEDIA_SEL)) {
                if (!(el.currentSrc || el.src)) continue;

                const dead =
                    el.error ||
                    el.ended ||
                    el.readyState <= 1 ||
                    el.networkState === 3; // NETWORK_NO_SOURCE

                if (dead) el.load(); // для живых потоков это рестарт

                if (el.dataset && el.dataset.agsoftWasPlaying === "1") {
                    el.play().catch(() => { });
                }
            }
        }, 100);
    });
})();

// ------------------------------------------------------------------------------
// Разворот обёртки {"workflow": <граф>} → чистый граф (см. _unwrap_workflow в py).
// Unwrap the {"workflow": <graph>} wrapper → pure graph (see _unwrap_workflow in py).
// ------------------------------------------------------------------------------
const unwrapWorkflow = (wf) => {
    if (wf && typeof wf === "object" && wf.workflow && !wf.nodes) {
        return wf.workflow;
    }
    return wf;
};

// ------------------------------------------------------------------------------
// Восстановление воркфлоу перетаскиванием сохранённого ВИДЕО.
// Workflow restore by dragging a saved VIDEO.
// ------------------------------------------------------------------------------
const VIDEO_RE = /\.(mp4|mkv|webm|mov|m4v|avi)$/i;

// Нода под курсором (fallback-перебор по габаритам).
// Node under the cursor (fallback bounding-box scan).
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
                n && n.pos && n.size &&
                pos[0] >= n.pos[0] && pos[0] <= n.pos[0] + n.size[0] &&
                pos[1] >= n.pos[1] && pos[1] <= n.pos[1] + n.size[1]
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

const bindWorkflowDrop = () => {
    if (workflowDropBound) return;
    workflowDropBound = true;

    // window + capture=true: перехватываем РАНЬШЕ всех обработчиков ComfyUI.
    // window + capture=true: intercept BEFORE all ComfyUI handlers.
    window.addEventListener("drop", (e) => {
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file) return;

        // PNG не трогаем: с чистым графом в чанке его штатно откроет ComfyUI.
        // Do NOT touch PNGs: with the pure graph in the chunk ComfyUI opens
        // them natively.

        // Тащат на ноду загрузки — пусть работает её drag&drop-загрузка.
        // Dropped onto a load node — let its drag&drop upload handle it.
        const n = nodeAtEvent(e);
        if (n && (n.comfyClass === "AGSoftLoadVideo" || n.comfyClass === "AGSoftLoadAudio")) {
            return;
        }

        if (!VIDEO_RE.test(file.name)) return;

        // Гасим дефолт СИНХРОННО (иначе браузер откроет файл), решение — асинхронно.
        // Swallow the default SYNCHRONOUSLY (otherwise the browser opens the
        // file); the decision is made asynchronously.
        e.preventDefault();
        e.stopPropagation();

        console.log("[AGSoft Video Save] video drop intercepted:", file.name);

        (async () => {
            try {
                const body = new FormData();
                body.append("file", file, file.name);

                const resp = await fetch(api.apiURL("/agsoft/extract_workflow"), {
                    method: "POST",
                    body,
                });

                if (!resp.ok) {
                    console.warn("[AGSoft Video Save] extract_workflow HTTP", resp.status);
                    return;
                }

                const data = await resp.json();
                const wf = unwrapWorkflow(data && data.workflow);
                console.log("[AGSoft Video Save] extracted workflow:", !!wf);

                if (wf && app.loadGraphData) {
                    await app.loadGraphData(wf);
                    console.log("[AGSoft Video Save] workflow loaded from", file.name);
                }
            } catch (err) {
                console.warn("[AGSoft Video Save] workflow restore failed:", err);
            }
        })();
    }, true);
};

app.registerExtension({
    name: "AGSoft.VideoSave",

    setup() {
        bindWorkflowDrop();
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftVideoSave") return;

        // ---------- Контейнер превью ----------
        const wrap = document.createElement("div");
        wrap.style.width = "100%";
        wrap.style.height = "100%";
        wrap.style.overflow = "hidden";

        // Видео: звук — сначала пробуем играть БЕЗ muted (разрешится, если был
        // клик по странице); если браузер запретил — fallback в muted + анмут
        // по наведению (как в VHS).
        // Video: sound — first try UNMUTED (allowed after a page click); if the
        // browser blocks it — fall back to muted + unmute on hover (like VHS).
        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = false;
        videoEl.style.width = "100%";
        videoEl.style.height = "100%";
        videoEl.style.objectFit = "contain";
        videoEl.style.backgroundColor = "#000";
        videoEl.style.display = "none";

        videoEl.onmouseenter = () => { videoEl.muted = false; };
        videoEl.onmouseleave = () => { videoEl.muted = true; };

        // Картинка (PNG / GIF / WebP / если сохранён только первый кадр).
        const imgEl = document.createElement("img");
        imgEl.style.width = "100%";
        imgEl.style.height = "100%";
        imgEl.style.objectFit = "contain";
        imgEl.style.display = "none";

        wrap.appendChild(videoEl);
        wrap.appendChild(imgEl);

        // ---------- Показ превью (file → /view, path → stream/preview_path) ----------
        // Show preview (file → /view, path → stream/preview_path).
        const showPreview = (p, fromRestore) => {
            if (!p) return;

            let url = null;

            if (p.kind === "path" && p.path) {
                // NO-OP превью файла по абсолютному пути (без записи).
                // NO-OP preview of a file by absolute path (no writing).
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

                // Защита от кэша браузера при повторных запусках
                // (при восстановлении timestamp НЕ меняем — файл тот же).
                // Browser cache busting on re-runs
                // (on restore we do NOT change timestamp — same file).
                if (!fromRestore) params.set("timestamp", String(Date.now()));

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
                
                videoEl.src = url;
                videoEl.load();
                try {
                    videoEl.pause();
                } catch (e) {} 
            }

            // Запоминаем для сериализации в воркфлоу.
            // Remember for serialization into the workflow.
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

        // Виджет превью.
        // Preview widget.
        const previewWidget = node.addDOMWidget(
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
                    if (v && (v.filename || v.path)) showPreview(v, true);
                },
            }
        );

        // ---------- Ресайз: ОДИН-В-ОДИН как в 🎬AGSoft Load Video ----------
        // Высота превью для заполнения = MIN + extra.
        // Preview fill height = MIN + extra.
        const MIN_PREVIEW_H = 120;
        let previewExtra = 0;   // добавочная высота (только от действия пользователя)
        let baseHeight = null;  // базовая высота ноды (превью = MIN)

        previewWidget.computeSize = function (width) {
            return [width || 200, MIN_PREVIEW_H + previewExtra];
        };

        // Минимальная высота ноды НЕ включает extra → ноду можно сжимать вверх.
        // Minimum node height does NOT include extra → node can shrink upward.
        const origComputeSize = node.computeSize ? node.computeSize.bind(node) : null;
        node.computeSize = function (...args) {
            const s = origComputeSize ? origComputeSize(...args) : [this.size[0], this.size[1]];
            if (Array.isArray(s)) s[1] = Math.max(0, s[1] - previewExtra);
            return s;
        };

        // ---------- Хук onExecuted: превью + вшивка воркфлоу из браузера ----------
        const origExecuted = node.onExecuted;

        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);

            const gifs = output && output.gifs;
            if (!gifs || !gifs.length) return;

            showPreview(gifs[0], false);

            // Воркфлоу берётся ИЗ БРАУЗЕРА (app.graph.serialize() = чистый граф,
            // как в AGSoft_Save_workflowImage.js) и дошивается в сохранённый файл.
            // Работает даже если extra_pnginfo не доехал до ноды.
            // ВАЖНО: в no-op режиме (kind="path") файл не сохраняется — вшивку
            // пропускаем (вшивать нечего и некуда).
            // The workflow is taken FROM THE BROWSER (app.graph.serialize() =
            // pure graph, like in AGSoft_Save_workflowImage.js) and embedded
            // into the saved file. IMPORTANT: in no-op mode (kind="path")
            // nothing is saved — skip embedding.
            const w = node.widgets && node.widgets.find((x) => x.name === "save_metadata");
            const p = gifs[0];

            if (w && w.value && p && p.fullpath && p.kind !== "path") {
                fetch(api.apiURL("/agsoft/embed_workflow"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
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

        // ---------- Двусторонний ресайз по вертикали (как в Load Video) ----------
        const computeBase = () => {
            previewExtra = 0;
            baseHeight = node.computeSize()[1]; // минимум без extra
        };

        computeBase();

        const origOnResize = node.onResize;

        node.onResize = function (size) {
            if (baseHeight == null) computeBase();

            previewExtra = Math.max(0, size[1] - baseHeight);

            if (origOnResize) origOnResize.apply(this, arguments);

            node.setDirtyCanvas(true, true);
        };

        // ---------- Первичная раскладка ----------
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true);
    }
});
