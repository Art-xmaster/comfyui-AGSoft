// ==============================================================================
// AGSoft_Load_Audio.js
// ==============================================================================
// JS-расширение для ноды 🔊AGSoft Load Audio.
// Добавляет:
// - превью-плеер;
// - кнопку загрузки файла;
// - drag&drop аудиофайла из проводника прямо на ноду (вся область ноды,
//   включая плеер и кнопку), с подсветкой рамки и защитой от двойного drop;
// - умное обновление src.
//
// JS extension for 🔊AGSoft Load Audio node.
// Adds:
// - preview player;
// - upload button;
// - drag&drop of an audio file from the OS explorer straight onto the node
//   (whole node area, including player and button), with border highlight
//   and double-drop protection;
// - smart src update.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Audio] JS extension loaded (preview + drag&drop)");

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
            if (n && n.comfyClass === "AGSoftLoadAudio" && n.__agsoftDrag) {
                n.__agsoftDrag = false;
                n.setDirtyCanvas(true, true);
            }
        }
    };

    // dragover: разрешаем drop только над нашей нодой + подсветка.
    // dragover: allow drop only over our node + highlight.
    canvasEl.addEventListener("dragover", (e) => {
        const n = getNodeAtEvent(e);

        if (n && n.comfyClass === "AGSoftLoadAudio") {
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

        if (n && n.comfyClass === "AGSoftLoadAudio") {
            e.preventDefault();

            if (n.__agsoftDrag) {
                n.__agsoftDrag = false;
                n.setDirtyCanvas(true, true);
            }

            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];

            if (f && n.__agsoftUpload && n.__agsoftIsAudio && n.__agsoftIsAudio(f) && guardDrop()) {
                n.__agsoftUpload(f);
            }
        }
    });
};

app.registerExtension({
    name: "AGSoft.LoadAudio",

    setup() {
        bindCanvasDrag();
    },

    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftLoadAudio") return;

        const combo = node.widgets?.find(w => w.name === "audio");
        if (!combo) return;

        // ---------- Плеер (скрытый <audio> + controls) ----------
        const audioEl = document.createElement("audio");
        audioEl.controls = true;
        audioEl.style.width = "100%";

        const playerWidget = node.addDOMWidget(
            "agsoft_audio_player",
            "div",
            audioEl,
            { serialize: false, hideOnZoom: false }
        );

        playerWidget.computeSize = () => [200, 40];

        const updateSrc = () => {
            const file = combo.value;

            if (!file || !String(file).trim()) {
                audioEl.removeAttribute("src");
                audioEl.load();
                return;
            }

            const filename = String(file).split(/[/\\]/).pop();

            audioEl.src = api.apiURL(
                `/view?filename=${encodeURIComponent(filename)}&type=input`
            );
            audioEl.load();
        };

        // ---------- Кнопка загрузки ----------
        const wrap = document.createElement("div");

        const btn = document.createElement("button");
        btn.textContent = "choose file to upload";
        btn.style.width = "100%";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "audio/*";
        fileInput.style.display = "none";

        btn.onclick = (e) => {
            e.preventDefault();
            fileInput.click();
        };

        const handleUpload = async (file) => {
            try {
                btn.disabled = true;
                btn.textContent = "Uploading...";

                const body = new FormData();
                body.append("image", file, file.name);
                body.append("type", "input");
                body.append("subfolder", "");

                const resp = await api.fetchApi("/upload/image", { method: "POST", body });

                if (!resp.ok) throw new Error("upload failed: " + resp.status);

                const data = await resp.json();
                const name = data.name || file.name;

                const vals = combo.options?.values || combo.values || [];
                if (Array.isArray(vals) && !vals.includes(name)) vals.push(name);

                combo.value = name;

                if (combo.callback) combo.callback(name);

                updateSrc();
            } catch (e) {
                console.error("[AGSoft Load Audio] upload error:", e);
                btn.textContent = "Upload Error! See F12";
            } finally {
                setTimeout(() => {
                    btn.disabled = false;
                    btn.textContent = "choose file to upload";
                }, 1500);
            }
        };

        fileInput.onchange = () => {
            if (fileInput.files && fileInput.files[0]) {
                handleUpload(fileInput.files[0]);
                fileInput.value = "";
            }
        };

        wrap.appendChild(btn);
        wrap.appendChild(fileInput);

        const uploadWidget = node.addDOMWidget(
            "agsoft_audio_upload",
            "div",
            wrap,
            { serialize: false, hideOnZoom: false }
        );

        uploadWidget.computeSize = () => [200, 30];

        // ---------- Drag&drop из проводника (DOM-область: плеер + кнопка) ----------
        // Проверка, что перетащили именно аудиофайл.
        // Check that the dropped file is actually audio.
        const isAudioFile = (f) => {
            if (!f) return false;
            if (f.type && f.type.startsWith("audio/")) return true;
            return /\.(mp3|wav|flac|ogg|oga|opus|m4a|aac|wma|amr|aiff|aif|mid|midi|mp2)$/i.test(f.name || "");
        };

        // Доступно и canvas-обработчику (drop на canvas-области ноды).
        // Also exposed to the canvas handler (drop on node's canvas area).
        node.__agsoftUpload = handleUpload;
        node.__agsoftIsAudio = isAudioFile;

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

            if (f && isAudioFile(f) && guardDrop()) {
                handleUpload(f);
            }
        };

        const domDragLeave = () => {
            setDrag(false);
        };

        for (const el of [audioEl, wrap]) {
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

        // Canvas мог быть создан до расширения — страховка.
        // Canvas could be created before the extension — safety net.
        bindCanvasDrag();

        updateSrc();
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true);
    }
});