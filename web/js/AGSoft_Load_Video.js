// ==============================================================================
// AGSoft_Load_Video.js
// ==============================================================================
// JS extension for 🎬AGSoft Load Video.
// Preview architecture mirrors 🎬AGSoft Video Save (no spoiler here):
// [Open / Upload button row] / [media flex] / [seek bar in live mode] /
// [16px info footer]. The preview is serialized and the node keeps the
// user's width and height when widget values or the preview change.
// The absolute source path comes from /agsoft/video_info, so the Open button
// opens files from input exactly like files from output.
// The upload keeps a strong XHR reference on the node and clears the file
// input only after the request finishes, so the transfer is never cut off.
// ---
// JS-расширение для 🎬AGSoft Load Video.
// Архитектура превью повторяет 🎬AGSoft Video Save (здесь без спойлера):
// [панель Open / Upload] / [медиа flex] / [полоса перемотки в live-режиме] /
// [инфо-футер 16px]. Превью сериализуется, нода сохраняет пользовательские
// ширину и высоту при смене значений виджетов и превью.
// Абсолютный путь источника приходит из /agsoft/video_info, поэтому кнопка
// Open открывает файлы из input так же, как файлы из output.
// Загрузка держит сильную ссылку на XHR у ноды и очищает файловый input
// только после завершения запроса, поэтому передача не обрывается.
// Author: AGSoft
// Date: 23.11.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Video] JS extension loaded v11.23 " +
    "(Open uses absolute path for input files, Video Save-style preview, stable size)");

const TRANSCODE_EXT = ["mkv", "avi", "ts", "m2ts", "vob", "flv", "wmv", "mpg", "mpeg"];
const MIN_PREVIEW_H = 162;
const SEEK_H = 44;

const fmtSize = (b) => {
    if (!b) return "";
    if (b > 1024 * 1024 * 1024) return (b / 1024 / 1024 / 1024).toFixed(2) + " GB";
    if (b > 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + " MB";
    return Math.round(b / 1024) + " KB";
};

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

const extOf = (p) => String(
    p.ext || (p.path || p.filename || "").split(".").pop() || ""
).toLowerCase();

const isVideoFile = (f) => {
    if (!f) return false;
    if (f.type && f.type.startsWith("video/")) return true;
    return /\.(mp4|mov|webm|mkv|avi|ts|m2ts|vob|flv|wmv|mpg|mpeg)$/i.test(f.name || "");
};

let lastDropAt = 0;
const guardDrop = () => {
    const now = Date.now();
    if (now - lastDropAt < 600) return false;
    lastDropAt = now;
    return true;
};

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
        const nodes = canvas.graph.nodes || canvas.graph._nodes || [];
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

let canvasDropBound = false;
const bindCanvasDrag = () => {
    if (canvasDropBound) return;
    const canvasEl = app.canvasEl || (app.canvas && app.canvas.canvas);
    if (!canvasEl) return;
    canvasDropBound = true;
    const clearAllDrag = () => {
        const graph = app.canvas && app.canvas.graph;
        const nodes = (graph && (graph.nodes || graph._nodes)) || [];
        for (const n of nodes) {
            if (n && n.comfyClass === "AGSoftLoadVideo" && n.__agsoftDrag) {
                n.__agsoftDrag = false;
                n.setDirtyCanvas(true, true);
            }
        }
    };
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
    canvasEl.addEventListener("drop", (e) => {
        const n = getNodeAtEvent(e);
        if (!n || n.comfyClass !== "AGSoftLoadVideo") return;
        e.preventDefault();
        e.stopPropagation();
        n.__agsoftDrag = false;
        n.setDirtyCanvas(true, true);
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f && n.__agsoftUpload && isVideoFile(f) && guardDrop()) n.__agsoftUpload(f);
    });
};

app.registerExtension({
    name: "AGSoft.LoadVideo",
    setup() {
        bindCanvasDrag();
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftLoadVideo") return;
        const combo = node.widgets?.find((w) => w.name === "video");
        const custom = node.widgets?.find((w) => w.name === "custom_path");
        // ------------------------------------------------------------------
        // Container: [buttons] / [media] / [seek] / [footer]
        // Контейнер: [кнопки] / [медиа] / [перемотка] / [футер]
        // ------------------------------------------------------------------
        const wrap = document.createElement("div");
        wrap.className = "agsoft-load-preview";
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
        const btnOpen = mkBtn(
            "📂🎬 Open",
            "Open the current video in a new browser tab (playable stream).\n" +
            "Открыть текущее видео в новой вкладке браузера (играбельный поток)."
        );
        const btnUpload = mkBtn(
            "⬆️🎬 Upload",
            "Upload a video into the input folder (single request, with progress).\n" +
            "Загрузить видео в папку input (один запрос, с прогрессом)."
        );
        topBar.appendChild(btnOpen);
        topBar.appendChild(btnUpload);
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
        videoEl.preload = "auto";
        videoEl.muted = false;
        Object.assign(videoEl.style, {
            width: "100%",
            height: "100%",
            objectFit: "contain",
            backgroundColor: "#000",
            display: "block",
        });
        videoEl.onmouseenter = () => {
            videoEl.muted = false;
        };
        videoEl.onmouseleave = () => {
            videoEl.muted = true;
        };
        media.appendChild(videoEl);
        const seekWrap = document.createElement("div");
        Object.assign(seekWrap.style, {
            flex: "none",
            height: `${SEEK_H}px`,
            display: "none",
            paddingTop: "2px",
        });
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "100";
        slider.step = "1";
        slider.value = "0";
        slider.style.width = "100%";
        slider.disabled = true;
        const timeLbl = document.createElement("div");
        Object.assign(timeLbl.style, {
            fontSize: "11px",
            opacity: "0.8",
            marginTop: "2px",
            color: "#9aa4b2",
            fontFamily: "sans-serif",
        });
        timeLbl.textContent = "--:-- / --:--";
        seekWrap.appendChild(slider);
        seekWrap.appendChild(timeLbl);
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
            paddingLeft: "6px",
            overflow: "hidden",
            whiteSpace: "nowrap",
        });
        foot.textContent = " ";
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*";
        fileInput.style.display = "none";
        wrap.appendChild(topBar);
        wrap.appendChild(media);
        wrap.appendChild(seekWrap);
        wrap.appendChild(foot);
        wrap.appendChild(fileInput);
        // ------------------------------------------------------------------
        // State / Состояние
        // ------------------------------------------------------------------
        let previewWidget = null;
        let liveMode = false;
        let seekOffset = 0;
        let duration = 0;
        let dragging = false;
        let current = { path: "", filename: "", ext: "" };
        let infoToken = 0;
        let previewExtra = 0;
        let baseHeight = null;
        // ------------------------------------------------------------------
        // Resize: keep the user's size / Ресайз: держим размер пользователя
        // ------------------------------------------------------------------
        const measureBase = () => {
            previewExtra = 0;
            const bs = node.computeSize ? node.computeSize() : [node.size[0], 0];
            baseHeight = Math.max(0, Number(bs && bs[1]) || 0);
            return baseHeight;
        };
        const syncLayout = (preserveUserHeight = true) => {
            const currentW = Math.max(220, Number(node.size && node.size[0]) || 220);
            const currentH = Math.max(0, Number(node.size && node.size[1]) || 0);
            const baseH = measureBase();
            let targetH = currentH;
            if (!preserveUserHeight || targetH < baseH) targetH = baseH;
            previewExtra = Math.max(0, targetH - baseH);
            node.setSize([currentW, targetH]);
            node.setDirtyCanvas(true, true);
        };
        previewWidget = node.addDOMWidget("agsoft_load_preview", "div", wrap, {
            serialize: true,
            hideOnZoom: false,
            getValue() {
                return previewWidget._saved || null;
            },
            setValue(v) {
                previewWidget._saved = v || null;
                if (v && (v.path || v.filename)) showPreview(v, true);
                else if (v) foot.textContent = footerText(v);
            },
        });
        previewWidget.computeSize = function (width) {
            return [width || 200, MIN_PREVIEW_H + (liveMode ? SEEK_H : 0) + previewExtra];
        };
        const origCompute = node.computeSize;
        const origComputeSize = origCompute
            ? function (...args) {
                  return origCompute.apply(node, args);
              }
            : null;
        node.computeSize = function (...args) {
            const s = origComputeSize ? origComputeSize(...args) : [this.size[0], this.size[1]];
            if (Array.isArray(s)) s[1] = Math.max(0, s[1] - previewExtra);
            return s;
        };
        // ------------------------------------------------------------------
        // URLs and preview / Адреса потоков и превью
        // ------------------------------------------------------------------
        const streamURL = (src, start) => {
            const params = new URLSearchParams();
            if (src.path) params.set("path", src.path);
            else params.set("filename", src.filename || "");
            if (TRANSCODE_EXT.includes(extOf(src))) {
                params.set("start", Number(start || 0).toFixed(3));
                return api.apiURL("/agsoft/preview?" + params.toString());
            }
            return api.apiURL("/agsoft/stream?" + params.toString());
        };
        const infoURL = (src) => {
            const params = new URLSearchParams();
            if (src.path) params.set("path", src.path);
            else params.set("filename", src.filename || "");
            return api.apiURL("/agsoft/video_info?" + params.toString());
        };
        const updateTimeUI = () => {
            const cur = liveMode
                ? seekOffset + (videoEl.currentTime || 0)
                : videoEl.currentTime || 0;
            if (!dragging && duration > 0) slider.value = String(Math.min(cur, duration));
            timeLbl.textContent = `${fmtTime(cur)} / ${fmtTime(duration)}`;
        };
        const refreshInfo = async () => {
            const token = ++infoToken;
            if (!current.path && !current.filename) return;
            try {
                const r = await api.fetchApi(infoURL(current));
                if (token !== infoToken || !r.ok) return;
                const d = await r.json();
                if (token !== infoToken) return;
                // Абсолютный путь из input: Open работает так же, как для output.
                // Absolute path for input: Open behaves exactly as for output.
                if (d.path) current.path = d.path;
                const saved = previewWidget._saved || {};
                previewWidget._saved = {
                    ...saved,
                    ...d,
                    path: current.path,
                    fullpath: current.path || saved.fullpath || "",
                    frame_rate: d.fps || 0,
                };
                foot.textContent = footerText(previewWidget._saved);
                duration = Number(d.duration) || 0;
                if (duration > 0) {
                    slider.max = String(duration);
                    slider.disabled = false;
                }
                updateTimeUI();
            } catch (e) {
                /* метаданные не критичны для превью / metadata is not critical */
            }
        };
        function showPreview(p, fromRestore) {
            if (!p || (!p.path && !p.filename)) return;
            current = {
                path: p.path || "",
                filename: p.filename || "",
                ext: extOf(p),
            };
            liveMode = TRANSCODE_EXT.includes(current.ext);
            seekOffset = 0;
            seekWrap.style.display = liveMode ? "block" : "none";
            slider.value = "0";
            slider.disabled = true;
            duration = Number(p.duration) || 0;
            if (duration > 0) {
                slider.max = String(duration);
                slider.disabled = false;
            }
            videoEl.src = streamURL(current, 0);
            videoEl.load();
            try {
                videoEl.pause();
            } catch (e) {}
            previewWidget._saved = {
                kind: p.kind || "path",
                path: current.path,
                filename: current.filename,
                ext: current.ext,
                type: p.type || "path",
                subfolder: p.subfolder || "",
                fullpath: p.fullpath || current.path,
                width: p.width || 0,
                height: p.height || 0,
                duration: p.duration || 0,
                frame_rate: p.frame_rate || 0,
                codec: p.codec || "",
                has_audio: !!p.has_audio,
                audio_codec: p.audio_codec || "",
                size_bytes: p.size_bytes || 0,
            };
            foot.textContent = footerText(previewWidget._saved);
            updateTimeUI();
            syncLayout(true);
            if (!fromRestore || !p.width || !p.duration || !current.path) refreshInfo();
        }
        const clearPreview = () => {
            current = { path: "", filename: "", ext: "" };
            liveMode = false;
            seekWrap.style.display = "none";
            videoEl.removeAttribute("src");
            videoEl.load();
            previewWidget._saved = null;
            foot.textContent = " ";
            syncLayout(true);
        };
        const updateFromWidgets = () => {
            const customValue = custom ? String(custom.value || "").trim() : "";
            if (customValue) {
                showPreview({ kind: "path", path: customValue }, false);
                return;
            }
            const name = combo ? String(combo.value || "").trim() : "";
            if (name) {
                showPreview({ kind: "file", filename: name.split(/[/\\]/).pop() }, false);
                return;
            }
            clearPreview();
        };
        // ------------------------------------------------------------------
        // Seek (live containers) / Перемотка (live-контейнеры)
        // ------------------------------------------------------------------
        const seekTo = (t, autoplay) => {
            if (!current.path && !current.filename) return;
            seekOffset = t;
            videoEl.src = streamURL(current, t);
            videoEl.load();
            if (autoplay) videoEl.play().catch(() => {});
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
        videoEl.addEventListener("timeupdate", updateTimeUI);
        videoEl.addEventListener("loadedmetadata", updateTimeUI);
        // ------------------------------------------------------------------
        // Open / Upload
        // ------------------------------------------------------------------
        btnOpen.onclick = () => {
            if (!current.path && !current.filename) return;
            window.open(streamURL(current, 0), "_blank");
        };
        const handleUpload = (file) => {
            return new Promise((resolve) => {
                // Ссылка на запрос хранится у ноды: он не исчезнет до ответа.
                // The request is referenced by the node: it lives until the response.
                const reset = () => {
                    node.__agsoftXhr = null;
                    setTimeout(() => {
                        btnUpload.disabled = false;
                        btnUpload.textContent = "⬆️🎬 Upload";
                    }, 1200);
                };
                const fail = (label, reason) => {
                    btnUpload.textContent = label;
                    foot.textContent = `Upload interrupted / Загрузка прервана: ${reason}`;
                    console.error("[AGSoft Load Video] upload interrupted:", reason);
                    reset();
                    resolve();
                };
                try {
                    btnUpload.disabled = true;
                    btnUpload.textContent = "Uploading… 0%";
                    const xhr = new XMLHttpRequest();
                    node.__agsoftXhr = xhr;
                    xhr.open("POST", api.apiURL("/agsoft/upload"));
                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable) {
                            const pct = Math.round((e.loaded / e.total) * 100);
                            btnUpload.textContent = `Uploading… ${pct}%`;
                        }
                    };
                    xhr.onload = () => {
                        try {
                            if (xhr.status >= 200 && xhr.status < 300) {
                                const data = JSON.parse(xhr.responseText || "{}");
                                const name = data.name || file.name;
                                const vals = combo && (combo.options?.values || combo.values);
                                if (Array.isArray(vals) && !vals.includes(name)) vals.push(name);
                                if (custom) custom.value = "";
                                if (combo) {
                                    combo.value = name;
                                    if (combo.callback) combo.callback.call(combo, name);
                                }
                                updateFromWidgets();
                            } else {
                                btnUpload.textContent = "Upload error";
                                foot.textContent =
                                    `Upload failed / Ошибка загрузки: HTTP ${xhr.status}`;
                                console.error(
                                    "[AGSoft Load Video] upload failed:",
                                    xhr.status, xhr.responseText
                                );
                            }
                        } finally {
                            reset();
                            resolve();
                        }
                    };
                    xhr.onerror = () => fail("Upload error", "network");
                    xhr.onabort = () => fail("Upload aborted", "aborted");
                    xhr.ontimeout = () => fail("Upload timeout", "timeout");
                    const body = new FormData();
                    body.append("file", file, file.name);
                    const dispatchBody = xhr.send;
                    dispatchBody.call(xhr, body);
                } catch (e) {
                    console.error("[AGSoft Load Video] upload error:", e);
                    btnUpload.textContent = "Upload error";
                    reset();
                    resolve();
                }
            });
        };
        btnUpload.onclick = (e) => {
            e.preventDefault();
            fileInput.click();
        };
        // Input очищается только после ответа сервера, не во время передачи.
        // The input is cleared only after the server response, not mid-transfer.
        fileInput.onchange = async () => {
            const f = fileInput.files && fileInput.files[0];
            if (!f) return;
            await handleUpload(f);
            fileInput.value = "";
        };
        node.__agsoftUpload = handleUpload;
        // ------------------------------------------------------------------
        // Drag&drop on the DOM area / Drag&drop по DOM-области
        // ------------------------------------------------------------------
        const setDrag = (on) => {
            node.__agsoftDrag = on;
            node.setDirtyCanvas(true, true);
        };
        wrap.addEventListener("dragover", (e) => {
            if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) {
                return;
            }
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            setDrag(true);
        });
        wrap.addEventListener("dragleave", () => setDrag(false));
        wrap.addEventListener("drop", (e) => {
            setDrag(false);
            e.preventDefault();
            e.stopPropagation();
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (f && isVideoFile(f) && guardDrop()) handleUpload(f);
        });
        const origDraw = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origDraw) origDraw.apply(this, arguments);
            if (this.__agsoftDrag) {
                ctx.save();
                ctx.strokeStyle = "#7fd4ff";
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.strokeRect(1, 1, this.size[0] - 2, this.size[1] - 2);
                ctx.restore();
            }
        };
        // ------------------------------------------------------------------
        // Hooks: executed / configure / resize / widget callbacks
        // Хуки: выполнение / загрузка / ресайз / колбэки виджетов
        // ------------------------------------------------------------------
        const origExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            const gifs = output && output.gifs;
            if (gifs && gifs.length) showPreview(gifs[0], false);
        };
        const origConfigure = node.onConfigure;
        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            if (!previewWidget._saved) updateFromWidgets();
            const h = Number(node.size && node.size[1]) || 0;
            const baseH = measureBase();
            previewExtra = Math.max(0, h - baseH);
            node.setSize([Math.max(220, Number(node.size[0]) || 220), baseH + previewExtra]);
            node.setDirtyCanvas(true, true);
        };
        const origOnResize = node.onResize;
        node.onResize = function (size) {
            if (baseHeight == null) syncLayout(true);
            previewExtra = Math.max(0, size[1] - (baseHeight || 0));
            if (origOnResize) origOnResize.apply(this, arguments);
            node.setDirtyCanvas(true, true);
        };
        if (combo) {
            const oldCb = combo.callback;
            combo.callback = function (v) {
                if (oldCb) oldCb.call(this, v);
                if (custom) custom.value = "";
                updateFromWidgets();
            };
        }
        if (custom) {
            const oldCustomCb = custom.callback;
            custom.callback = function (v) {
                if (oldCustomCb) oldCustomCb.call(this, v);
                updateFromWidgets();
            };
        }
        bindCanvasDrag();
        updateFromWidgets();
        const fresh = !(node.properties &&
            Object.prototype.hasOwnProperty.call(node.properties, "agsoft_load_sized"));
        if (fresh) {
            node.properties = node.properties || {};
            node.properties["agsoft_load_sized"] = true;
            const baseH = measureBase();
            previewExtra = 300;
            node.setSize([Math.max(220, Number(node.size[0]) || 220), baseH + previewExtra]);
        }
        syncLayout(true);
        setTimeout(() => {
            syncLayout(true);
            app.graph.setDirtyCanvas(true);
        }, 0);
    },
});