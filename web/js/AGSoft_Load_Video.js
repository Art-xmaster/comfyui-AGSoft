// ==============================================================================
// AGSoft_Load_Video.js
// ==============================================================================
// JS-расширение для ноды 🎬AGSoft Load Video.
// Добавляет:
// - превью-плеер;
// - СВОЮ полосу перемотки для MKV/AVI/TS: мгновенные прыжки БЕЗ кэша
//   (каждый прыжок = новый живой поток с ffmpeg -ss);
// - кнопку загрузки ОДНИМ запросом (streaming multipart) с прогрессом;
// - строку информации: ширина×высота • длительность • кодек;
// - drag&drop видеофайла из проводника прямо на ноду (вся область ноды,
//   включая плеер и кнопку), с подсветкой рамки и защитой от двойного drop;
// - двусторонний вертикальный ресайз плеера.
//
// JS extension for 🎬AGSoft Load Video node.
// Adds:
// - preview player;
// - CUSTOM seek bar for MKV/AVI/TS: instant jumps with NO cache
//   (every jump = a new live stream via ffmpeg -ss);
// - SINGLE-request (streaming multipart) upload button with progress;
// - info line: width×height • duration • codec;
// - drag&drop of a video file from the OS explorer straight onto the node
//   (whole node area, including player and button), with border highlight
//   and double-drop protection;
// - bidirectional vertical player resize.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Video] JS extension loaded (instant seek NO cache + single upload + metadata + drag&drop + resizable player)");

// ------------------------------------------------------------------------------
// Защита от двойного срабатывания drop (canvas-обработчик + DOM-обработчики).
// Protection against double drop handling (canvas handler + DOM handlers).
// ------------------------------------------------------------------------------
let lastDropAt = 0;

const guardDrop = () => {
    const now = Date.now();
    if (now - lastDropAt < 600) return false;
    lastDropAt = now;
    return true;
};

// ------------------------------------------------------------------------------
// Поиск ноды под курсором на canvas (для drag&drop на canvas-области ноды).
// Find the node under the cursor on canvas (for drag&drop on canvas area).
// ------------------------------------------------------------------------------
const getNodeAtEvent = (e) => {
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

        // Способ 1: штатный getNodeOnPos.
        // Method 1: built-in getNodeOnPos.
        let node = null;

        try {
            if (canvas.graph.getNodeOnPos) {
                node = canvas.graph.getNodeOnPos(pos[0], pos[1], canvas.graph.nodes) || null;
            }
        } catch (err) {
            node = null;
        }

        // Способ 2: ручной перебор по габаритам (fallback).
        // Method 2: manual bounding-box scan (fallback).
        if (!node && canvas.graph.nodes) {
            for (let i = canvas.graph.nodes.length - 1; i >= 0; i--) {
                const n = canvas.graph.nodes[i];
                if (
                    n && n.pos && n.size &&
                    pos[0] >= n.pos[0] && pos[0] <= n.pos[0] + n.size[0] &&
                    pos[1] >= n.pos[1] && pos[1] <= n.pos[1] + n.size[1]
                ) {
                    node = n;
                    break;
                }
            }
        }

        return node;
    } catch (err) {
        return null;
    }
};

// ------------------------------------------------------------------------------
// Canvas-обработчики drag&drop вешаются ОДИН раз на весь canvas.
// Canvas drag&drop handlers are bound ONCE for the whole canvas.
// ------------------------------------------------------------------------------
let canvasDropBound = false;

const bindCanvasDrag = () => {
    if (canvasDropBound) return;

    const canvasEl = app.canvasEl || (app.canvas && app.canvas.canvas);
    if (!canvasEl) return;

    canvasDropBound = true;

    const clearAllDrag = () => {
        const graph = app.canvas && app.canvas.graph;
        if (!graph || !graph.nodes) return;

        for (const n of graph.nodes) {
            if (n && n.comfyClass === "AGSoftLoadVideo" && n.__agsoftDrag) {
                n.__agsoftDrag = false;
                n.setDirtyCanvas(true, true);
            }
        }
    };

    // dragover: разрешаем drop только над нашей нодой + подсветка.
    // dragover: allow drop only over our node + highlight.
    canvasEl.addEventListener("dragover", (e) => {
        const n = getNodeAtEvent(e);

        if (n && n.comfyClass === "AGSoftLoadVideo") {
            e.preventDefault();

            if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";

            if (!n.__agsoftDrag) {
                n.__agsoftDrag = true;
                n.setDirtyCanvas(true, true);
            }
        }
    });

    canvasEl.addEventListener("dragleave", clearAllDrag);

    // drop: забираем файл и отдаём в загрузку ноды.
    // drop: take the file and pass it to the node's upload.
    canvasEl.addEventListener("drop", (e) => {
        const n = getNodeAtEvent(e);

        if (n && n.comfyClass === "AGSoftLoadVideo") {
            e.preventDefault();

            if (n.__agsoftDrag) {
                n.__agsoftDrag = false;
                n.setDirtyCanvas(true, true);
            }

            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];

            if (f && n.__agsoftUpload && n.__agsoftIsVideo && n.__agsoftIsVideo(f) && guardDrop()) {
                n.__agsoftUpload(f);
            }
        }
    });
};

app.registerExtension({
    name: "AGSoft.LoadVideo",

    setup() {
        bindCanvasDrag();
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftLoadVideo") return;

        const combo = node.widgets?.find(w => w.name === "video");
        if (!combo) return;

        // ---------- Превью-плеер ----------
        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.preload = "auto";
        videoEl.style.width = "100%";
        videoEl.style.height = "100%";
        videoEl.style.objectFit = "contain";
        videoEl.style.backgroundColor = "#000";
        videoEl.style.display = "block";

        const MIN_PLAYER_H = 120;

        // Добавочная высота плеера (только от действия пользователя).
        // Extra player height (only from user action).
        let playerExtra = 0;

        // Базовая высота ноды (плеер = MIN).
        // Base node height (player = MIN).
        let baseHeight = null;

        const playerWidget = node.addDOMWidget(
            "agsoft_video_player",
            "div",
            videoEl,
            {
                serialize: false,
                hideOnZoom: false
            }
        );

        // Высота плеера для заполнения = MIN + extra.
        // Player fill height = MIN + extra.
        playerWidget.computeSize = function (width) {
            return [width || 200, MIN_PLAYER_H + playerExtra];
        };

        // Минимальная высота ноды НЕ включает extra → ноду можно сжимать вверх.
        // Minimum node height does NOT include extra → node can be compressed upward.
        const origComputeSize = node.computeSize ? node.computeSize.bind(node) : null;

        node.computeSize = function (...args) {
            const s = origComputeSize ? origComputeSize(...args) : [this.size[0], this.size[1]];
            if (Array.isArray(s)) s[1] = Math.max(0, s[1] - playerExtra);
            return s;
        };

        // ---------- Своя полоса перемотки (live-режим, БЕЗ кэша) ----------
        // Custom seek bar (live mode, NO cache).
        const seekWrap = document.createElement("div");
        seekWrap.style.display = "none";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "100";
        slider.step = "1";
        slider.value = "0";
        slider.style.width = "100%";
        slider.disabled = true;

        const timeLbl = document.createElement("div");
        timeLbl.style.fontSize = "11px";
        timeLbl.style.opacity = "0.8";
        timeLbl.style.marginTop = "2px";
        timeLbl.textContent = "--:-- / --:--";

        seekWrap.appendChild(slider);
        seekWrap.appendChild(timeLbl);

        const seekWidget = node.addDOMWidget(
            "agsoft_seek_bar",
            "div",
            seekWrap,
            {
                serialize: false,
                hideOnZoom: false
            }
        );

        seekWidget.computeSize = () => [200, 44];

        // ---------- Состояние live-режима ----------
        let liveMode = false;      // MKV/AVI/TS: живой поток + своя перемотка
        let seekOffset = 0;        // глобальное время, с которого стартовал поток
        let duration = 0;          // длительность файла (из video_info)
        let dragging = false;      // пользователь тащит слайдер
        let currentFilename = "";

        const fmtTime = (sec) => {
            if (!isFinite(sec) || sec < 0) return "--:--";

            sec = Math.round(sec);

            const h = Math.floor(sec / 3600);
            const m = Math.floor((sec % 3600) / 60);
            const s = sec % 60;

            const mm = String(m).padStart(2, "0");
            const ss = String(s).padStart(2, "0");

            return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
        };

        const updateTimeUI = () => {
            const cur = liveMode
                ? seekOffset + (videoEl.currentTime || 0)
                : (videoEl.currentTime || 0);

            if (!dragging && duration > 0) {
                slider.value = String(Math.min(cur, duration));
            }

            timeLbl.textContent = `${fmtTime(cur)} / ${fmtTime(duration)}`;
        };

        videoEl.addEventListener("timeupdate", updateTimeUI);
        videoEl.addEventListener("loadedmetadata", updateTimeUI);
        videoEl.addEventListener("ended", updateTimeUI);

        // Перемотка: новый живой поток с ffmpeg -ss. Мгновенно, без кэша.
        // Seek: a new live stream via ffmpeg -ss. Instant, no cache.
        const seekTo = (t, autoplay) => {
            if (!currentFilename) return;

            seekOffset = t;

            videoEl.src = api.apiURL(
                `/agsoft/preview?filename=${encodeURIComponent(currentFilename)}` +
                `&start=${t.toFixed(3)}`
            );
            videoEl.load();

            if (autoplay) videoEl.play().catch(() => { });

            updateTimeUI();
        };

        slider.addEventListener("input", () => {
            dragging = true;
            timeLbl.textContent = `${fmtTime(parseFloat(slider.value))} / ${fmtTime(duration)}`;
        });

        slider.addEventListener("change", () => {
            dragging = false;
            seekTo(parseFloat(slider.value) || 0, true);
        });

        // ---------- Кнопка загрузки (один запрос, прогресс через XHR) ----------
        const wrap = document.createElement("div");

        const btn = document.createElement("button");
        btn.textContent = "choose file to upload";
        btn.style.width = "100%";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*";
        fileInput.style.display = "none";

        btn.onclick = (e) => {
            e.preventDefault();
            fileInput.click();
        };

        // Строка информации: ширина×высота • длительность • кодек.
        // Info line: width×height • duration • codec.
        const infoEl = document.createElement("div");
        infoEl.style.fontSize = "11px";
        infoEl.style.opacity = "0.7";
        infoEl.style.marginTop = "2px";
        infoEl.style.whiteSpace = "normal";

        let infoToken = 0;

        const updateInfo = async (filename) => {
            const token = ++infoToken;

            if (!filename) {
                infoEl.textContent = "";
                return;
            }

            try {
                const r = await api.fetchApi(
                    `/agsoft/video_info?filename=${encodeURIComponent(filename)}`
                );

                if (token !== infoToken) return;

                if (!r.ok) {
                    infoEl.textContent = "";
                    return;
                }

                const d = await r.json();

                if (token !== infoToken) return;

                const parts = [];

                if (d.width && d.height) parts.push(`${d.width}×${d.height}`);
                if (d.duration) parts.push(fmtTime(d.duration));
                if (d.codec) parts.push(d.codec);

                infoEl.textContent = parts.join(" • ");
            } catch (e) {
                if (token === infoToken) infoEl.textContent = "";
            }
        };

        // Загрузка ОДНИМ запросом: сервер стримит multipart на диск.
        // Single-request upload: the server streams multipart to disk.
        const handleUpload = (file) => {
            return new Promise((resolve) => {
                const resetBtn = () => {
                    setTimeout(() => {
                        btn.disabled = false;
                        btn.textContent = "choose file to upload";
                    }, 1500);
                };

                try {
                    btn.disabled = true;
                    btn.textContent = "Uploading... 0%";

                    const xhr = new XMLHttpRequest();
                    xhr.open("POST", api.apiURL("/agsoft/upload"));

                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable) {
                            btn.textContent =
                                `Uploading... ${Math.round((e.loaded / e.total) * 100)}%`;
                        }
                    };

                    xhr.onload = () => {
                        try {
                            if (xhr.status >= 200 && xhr.status < 300) {
                                const data = JSON.parse(xhr.responseText || "{}");
                                const name = data.name || file.name;

                                const vals = combo.options?.values || combo.values || [];
                                if (Array.isArray(vals) && !vals.includes(name)) vals.push(name);

                                combo.value = name;

                                if (combo.callback) combo.callback(name);

                                updateSrc();
                            } else {
                                console.error(
                                    "[AGSoft Load Video] upload failed:",
                                    xhr.status, xhr.responseText
                                );
                                btn.textContent = "Upload Error! See F12";
                            }
                        } finally {
                            resetBtn();
                            resolve();
                        }
                    };

                    xhr.onerror = () => {
                        console.error("[AGSoft Load Video] upload network error");
                        btn.textContent = "Upload Error! See F12";
                        resetBtn();
                        resolve();
                    };

                    const body = new FormData();
                    body.append("file", file, file.name);
                    xhr.send(body);
                } catch (e) {
                    console.error("[AGSoft Load Video] upload error:", e);
                    btn.textContent = "Upload Error! See F12";
                    resetBtn();
                    resolve();
                }
            });
        };

        fileInput.onchange = () => {
            if (fileInput.files && fileInput.files[0]) {
                handleUpload(fileInput.files[0]);
                fileInput.value = "";
            }
        };

        wrap.appendChild(btn);
        wrap.appendChild(fileInput);
        wrap.appendChild(infoEl);

        const uploadWidget = node.addDOMWidget(
            "agsoft_video_upload",
            "div",
            wrap,
            {
                serialize: false,
                hideOnZoom: false
            }
        );

        uploadWidget.computeSize = () => [200, 58];

        // ---------- Drag&drop из проводника (DOM-область: плеер + кнопка) ----------
        // Проверка, что перетащили именно видеофайл.
        // Check that the dropped file is actually a video.
        const isVideoFile = (f) => {
            if (!f) return false;
            if (f.type && f.type.startsWith("video/")) return true;
            return /\.(mp4|mov|webm|mkv|avi|ts|m2ts|vob|flv|wmv|mpg|mpeg)$/i.test(f.name || "");
        };

        // Доступно и canvas-обработчику (drop на canvas-области ноды).
        // Also exposed to the canvas handler (drop on node's canvas area).
        node.__agsoftUpload = handleUpload;
        node.__agsoftIsVideo = isVideoFile;

        const setDrag = (on) => {
            node.__agsoftDrag = on;
            node.setDirtyCanvas(true, true);
        };

        const domDragOver = (e) => {
            if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;

            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            setDrag(true);
        };

        const domDrop = (e) => {
            setDrag(false);

            // Всегда гасим drop над нодой, чтобы браузер не открыл файл.
            // Always swallow drop over the node so the browser won't open the file.
            e.preventDefault();
            e.stopPropagation();

            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];

            if (f && isVideoFile(f) && guardDrop()) {
                handleUpload(f);
            }
        };

        const domDragLeave = () => {
            setDrag(false);
        };

        for (const el of [videoEl, wrap]) {
            el.addEventListener("dragover", domDragOver);
            el.addEventListener("drop", domDrop);
            el.addEventListener("dragleave", domDragLeave);
        }

        // Подсветка рамки ноды во время перетаскивания.
        // Node border highlight while dragging.
        const origDrawForeground = node.onDrawForeground;

        node.onDrawForeground = function (ctx) {
            if (origDrawForeground) origDrawForeground.apply(this, arguments);

            if (this.__agsoftDrag) {
                const w = this.size[0];
                const h = this.size[1];

                ctx.save();
                ctx.strokeStyle = "#7fd4ff";
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(1, 1, w - 2, h - 2);
                ctx.restore();
            }
        };

        // ---------- Умное обновление src ----------
        // Контейнеры, где браузер скорее всего не сможет играть звук напрямую.
        // Containers where browser most likely cannot play audio directly.
        const TRANSCODE_EXT = [
            "mkv",
            "avi",
            "ts",
            "m2ts",
            "vob",
            "flv",
            "wmv",
            "mpg",
            "mpeg"
        ];

        const updateSrc = () => {
            const file = combo.value;

            if (!file || !String(file).trim()) {
                videoEl.removeAttribute("src");
                videoEl.load();

                liveMode = false;
                seekWrap.style.display = "none";
                updateInfo("");
                return;
            }

            const filename = String(file).split(/[/\\]/).pop();
            const ext = (filename.split(".").pop() || "").toLowerCase();

            currentFilename = filename;
            updateInfo(filename);

            // Обычные mp4/mov/webm — как есть, нативная перемотка через Range.
            // Normal mp4/mov/webm — as-is, native seeking via Range.
            if (!TRANSCODE_EXT.includes(ext)) {
                liveMode = false;
                seekWrap.style.display = "none";

                videoEl.src = api.apiURL(
                    `/agsoft/stream?filename=${encodeURIComponent(filename)}`
                );
                videoEl.load();
                return;
            }

            // MKV/AVI/TS: живой поток + СВОЯ мгновенная перемотка, БЕЗ кэша.
            // MKV/AVI/TS: live stream + CUSTOM instant seeking, NO cache.
            liveMode = true;
            seekWrap.style.display = "";
            seekOffset = 0;
            duration = 0;
            slider.value = "0";
            slider.disabled = true;
            timeLbl.textContent = "--:-- / --:--";

            // Длительность для полосы перемотки.
            // Duration for the seek bar.
            api.fetchApi(`/agsoft/video_info?filename=${encodeURIComponent(filename)}`)
                .then((r) => (r.ok ? r.json() : null))
                .then((d) => {
                    if (!d || currentFilename !== filename) return;

                    duration = d.duration || 0;

                    if (duration > 0) {
                        slider.max = String(duration);
                        slider.disabled = false;
                    }

                    updateTimeUI();
                })
                .catch(() => { });

            seekTo(0, false);
        };

        const oldCb = combo.callback;
        combo.callback = (v) => {
            if (oldCb) oldCb(v);
            updateSrc();
        };

        const origConfigure = node.onConfigure;

        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            updateSrc();
        };

        // ---------- Двусторонний ресайз по вертикали ----------
        const computeBase = () => {
            playerExtra = 0;
            baseHeight = node.computeSize()[1]; // минимум без extra
        };

        computeBase();

        const origOnResize = node.onResize;

        node.onResize = function (size) {
            if (baseHeight == null) computeBase();

            playerExtra = Math.max(0, size[1] - baseHeight);

            if (origOnResize) origOnResize.apply(this, arguments);

            node.setDirtyCanvas(true, true);
        };

        // Canvas мог быть создан до расширения — страховка.
        // Canvas could be created before the extension — safety net.
        bindCanvasDrag();

        updateSrc();
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true);
    }
});