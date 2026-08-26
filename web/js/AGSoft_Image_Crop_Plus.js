// ==============================================================================
// AGSoft_Image_Crop_Plus.js  (FINAL)
// ==============================================================================
// JS-расширение для ноды 🖼️✂️AGSoft Image Crop Plus.
//
// ⚡ Канвас-превью (letterbox, клип) + кнопки 📁 Upload / 🗑️ Reset / ▶️ Resume.
// ⚡ Рамка Preset Ratio / Manual Size; alignRect = математика Python.
// ⚡ Drag & Drop с подсветкой ноды (window-capture, ИСПРАВЛЕНО: корректный
//   перевод экранных координат в координаты канваса через ds.scale/ds.offset).
// ⚡ АВТО-ВЫБОР последнего изображения из image_name, если комбо пустое.
// ⚡ ПАУЗА при тензоре: waiting поллингом 1 с; живые coords; ▶️ Resume.
// ⚡ Синхронизация источника: 🖼️ image_name / 📂 custom_path / 🧠 input_image.
//
// Author: AGSoft
// Date: 26.08.2026 (final)
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS_NAME = "AGSoft Image Crop Plus";
const BTN_H = 26;

let fileInput = null;
function getFileInput(onPick) {
    if (!fileInput) {
        fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "image/jpeg,image/png,image/webp,image/bmp";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);
        fileInput.onchange = (ev) => {
            const f = ev.target.files && ev.target.files[0];
            if (f && fileInput._pick) fileInput._pick(f);
            fileInput.value = "";
        };
    }
    fileInput._pick = onPick;
    return fileInput;
}

app.registerExtension({
    name: "AGSoft.ImageCropPlusCanvas",

    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS_NAME) return;

        const S = {
            img: null, imageRect: null,
            cropRect: { x: 0, y: 0, w: 0, h: 0 },
            disp: null,
            rectDrag: null, rectStart: null,
            btns: {},
            status: "⚠️ Нет изображения.",
            size: "📐 --x--",
            srcLabel: "",
            waiting: false,
            dragOver: false,
        };
        node._crop = S;

        const getW = (n) => { const w = node.widgets.find(w => w.name === n); return w ? w.value : null; };
        const parseRatio = (s) => {
            const p = String(s).split(":").map(Number);
            const r = (p[0] || 1) / (p[1] || 1);
            return isFinite(r) && r > 0 ? r : 1;
        };
        const mult = () => Math.max(1, parseInt(getW("multiple")) || 8);

        function alignRect(r) {
            if (!S.imageRect) return r;
            const m = mult();
            let w = Math.max(m, Math.floor(r.w / m) * m);
            let h = Math.max(m, Math.floor(r.h / m) * m);
            let x = Math.max(0, Math.min(Math.round(r.x), S.imageRect.width - w));
            let y = Math.max(0, Math.min(Math.round(r.y), S.imageRect.height - h));
            return { x, y, w, h };
        }

        function sendLiveCoords() {
            const cw = node.widgets.find(w => w.name === "crop_coords");
            if (!cw) return;
            fetch(api.apiURL("/agsoft/crop_live_coords"), {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id), crop_coords: String(cw.value || "[]") })
            }).catch(() => {});
        }

        function pushCoords() {
            const cw = node.widgets.find(w => w.name === "crop_coords");
            if (!cw) return;
            cw.value = JSON.stringify({
                x: Math.round(S.cropRect.x), y: Math.round(S.cropRect.y),
                w: Math.round(S.cropRect.w), h: Math.round(S.cropRect.h)
            });
            if (cw.callback) cw.callback(cw.value);
            if (S.waiting) { clearTimeout(pushCoords._t); pushCoords._t = setTimeout(sendLiveCoords, 120); }
        }

        function resumeNow() {
            S.waiting = false;
            updateInfo(); node.setDirtyCanvas(true, true);
            fetch(api.apiURL("/agsoft/crop_resume"), {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id) })
            }).catch(() => {});
        }

        function updateInfo() {
            const src = S.srcLabel ? S.srcLabel + " · " : "";
            if (!S.imageRect) { S.status = src + "⚠️ Нет изображения."; S.size = "📐 --x--"; return; }
            const m = mult();
            if (S.dragOver) S.status = src + "📥 Отпустите файл для загрузки...";
            else if (S.waiting) S.status = src + "⏸️ Пауза: настройте кроп и нажмите ▶️ Продолжить.";
            else S.status = src + "🖱️ Тяни за центр/углы/грани. 📥 D&D поддерживается.";
            S.size = `📐 ${Math.round(S.cropRect.w)}x${Math.round(S.cropRect.h)} → ×${m}: ${Math.floor(S.cropRect.w / m) * m}x${Math.floor(S.cropRect.h / m) * m}`;
        }

        function initRect() {
            if (!S.imageRect) return;
            const mode = getW("crop_mode");
            let tw = S.imageRect.width * 0.7, th = S.imageRect.height * 0.7;
            if (mode === "Preset Ratio") {
                const r = parseRatio(getW("aspect_ratio") || "1:1");
                if (tw / th > r) tw = th * r; else th = tw / r;
            } else if (mode === "Manual Size") {
                const mw = parseInt(getW("manual_width")) || 512, mh = parseInt(getW("manual_height")) || 512;
                const s = Math.min(S.imageRect.width / mw, S.imageRect.height / mh, 1);
                tw = mw * s; th = mh * s;
            }
            S.cropRect = alignRect({ x: (S.imageRect.width - tw) / 2, y: (S.imageRect.height - th) / 2, w: tw, h: th });
            pushCoords(); updateInfo();
        }

        function layout() {
            let bottom = node.widgets_start_y || 30;
            if (node.widgets && node.widgets.length) {
                for (const w of node.widgets) {
                    const wh = w.computeSize ? w.computeSize(node.size[0])[1] : (w.height || 20);
                    if (typeof w.last_y === "number" && w.last_y > 0) {
                        bottom = Math.max(bottom, w.last_y + wh);
                    }
                }
            }
            const W = node.size[0];
            const lineH = 16;
            const nBtn = S.waiting ? 3 : 2;
            const btnsH = nBtn * BTN_H + (nBtn - 1) * 6;
            const controlsH = btnsH + 12 + lineH * 2 + 14;
            const cTop = node.size[1] - controlsH;
            const areaH = Math.max(40, cTop - 8 - (bottom + 8));
            const area = { x: 8, y: bottom + 8, w: Math.max(24, W - 16), h: areaH };
            const up = [8, cTop, W - 16, BTN_H];
            const rs = [8, cTop + BTN_H + 6, W - 16, BTN_H];
            const go = [8, cTop + 2 * (BTN_H + 6), W - 16, BTN_H];
            const statusY = cTop + btnsH + 24;
            const sizeY = statusY + lineH;
            return { area, up, rs, go, statusY, sizeY };
        }

        node.onDrawForeground = function (ctx) {
            try {
                if (this.flags && this.flags.collapsed) return;
                const L = layout();
                this._crop.btns = { up: L.up, rs: L.rs, go: L.go };
                ctx.save();
                ctx.beginPath(); ctx.rect(0, 0, this.size[0], this.size[1]); ctx.clip();

                // 1) Превью.
                S.disp = null;
                if (S.img && S.imageRect && L.area.h >= 40) {
                    const a = L.area;
                    const scale = Math.min(a.w / S.imageRect.width, a.h / S.imageRect.height);
                    const dw = S.imageRect.width * scale, dh = S.imageRect.height * scale;
                    const dx = a.x + (a.w - dw) / 2, dy = a.y + (a.h - dh) / 2;
                    S.disp = { dx, dy, dw, dh, scale };
                    ctx.save();
                    ctx.beginPath(); ctx.rect(a.x, a.y, a.w, a.h); ctx.clip();
                    ctx.fillStyle = "#1e1e1e"; ctx.fillRect(a.x, a.y, a.w, a.h);
                    ctx.drawImage(S.img, dx, dy, dw, dh);
                    drawOverlay(ctx);
                    ctx.restore();
                }

                // 2) D&D подсветка — поверх превью, под кнопками.
                if (S.dragOver) {
                    ctx.save();
                    ctx.fillStyle = "rgba(70,130,180,0.25)";
                    ctx.fillRect(0, 0, this.size[0], this.size[1]);
                    ctx.strokeStyle = "#4a9eff"; ctx.lineWidth = 4; ctx.setLineDash([8, 4]);
                    ctx.strokeRect(2, 2, this.size[0] - 4, this.size[1] - 4);
                    ctx.setLineDash([]);
                    ctx.fillStyle = "#fff"; ctx.font = "bold 24px Arial";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.shadowBlur = 6; ctx.shadowColor = "rgba(0,0,0,0.8)";
                    ctx.fillText("📥 DROP IMAGE HERE", this.size[0] / 2, this.size[1] / 2 - 14);
                    ctx.font = "13px sans-serif";
                    ctx.fillText("Отпустите файл для загрузки", this.size[0] / 2, this.size[1] / 2 + 12);
                    ctx.shadowBlur = 0;
                    ctx.restore();
                }

                // 3) Кнопки.
                const drawBtn = (r, label, color) => {
                    ctx.fillStyle = color; ctx.fillRect(r[0], r[1], r[2], r[3]);
                    ctx.fillStyle = "#fff"; ctx.font = "bold 12px sans-serif";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText(label, r[0] + r[2] / 2, r[1] + r[3] / 2 + 1);
                };
                drawBtn(L.up, "📁 Загрузить / Upload", "#4a6a8a");
                drawBtn(L.rs, "🗑️ Сбросить / Reset", "#6c6c6c");
                if (S.waiting) drawBtn(L.go, "▶️ Продолжить / Resume", "#3a7a3a");

                // 4) Текст.
                ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
                ctx.fillStyle = "#ccc"; ctx.font = "11px sans-serif";
                ctx.fillText(S.status, this.size[0] / 2, L.statusY);
                ctx.fillStyle = "#aaa";
                ctx.fillText(S.size, this.size[0] / 2, L.sizeY);
                ctx.textAlign = "start";
                ctx.restore();
            } catch (e) { /* не ломаем отрисовку */ }
        };

        function drawOverlay(ctx) {
            const d = S.disp; if (!d) return;
            const sc = d.scale;
            if (S.cropRect.w <= 0 || S.cropRect.h <= 0) return;
            const r = S.cropRect;
            const rx = r.x * sc + d.dx, ry = r.y * sc + d.dy, rw = r.w * sc, rh = r.h * sc;
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            ctx.fillRect(d.dx, d.dy, d.dw, ry - d.dy);
            ctx.fillRect(d.dx, ry + rh, d.dw, d.dy + d.dh - (ry + rh));
            ctx.fillRect(d.dx, ry, rx - d.dx, rh);
            ctx.fillRect(rx + rw, ry, d.dx + d.dw - (rx + rw), rh);
            ctx.fillStyle = "rgba(0,170,255,0.15)"; ctx.fillRect(rx, ry, rw, rh);
            ctx.strokeStyle = "#00aaff"; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, rw, rh);
            const hs = [[rx, ry], [rx + rw / 2, ry], [rx + rw, ry], [rx, ry + rh / 2],
                        [rx + rw, ry + rh / 2], [rx, ry + rh], [rx + rw / 2, ry + rh], [rx + rw, ry + rh]];
            hs.forEach(h => {
                ctx.beginPath(); ctx.arc(h[0], h[1], 5, 0, 2 * Math.PI);
                ctx.fillStyle = "#fff"; ctx.fill();
                ctx.strokeStyle = "#00aaff"; ctx.stroke();
            });
            const m = mult();
            ctx.fillStyle = "#fff"; ctx.font = "bold 14px Arial";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.shadowBlur = 3; ctx.shadowColor = "black";
            ctx.fillText(`${Math.floor(r.w / m) * m}×${Math.floor(r.h / m) * m}`, rx + rw / 2, ry + rh / 2);
            ctx.shadowBlur = 0; ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
        }

        // === МЫШЬ ===
        const toImg = (pos) => { const d = S.disp; return d ? { x: (pos[0] - d.dx) / d.scale, y: (pos[1] - d.dy) / d.scale } : null; };
        const inRect = (pos, r) => r && pos[0] >= r[0] && pos[0] <= r[0] + r[2] && pos[1] >= r[1] && pos[1] <= r[1] + r[3];

        function handleAt(p) {
            const t = 12, r = S.cropRect, d = S.disp; if (!d) return null;
            const X = (v) => v * d.scale + d.dx, Y = (v) => v * d.scale + d.dy;
            if (Math.hypot(p[0] - X(r.x), p[1] - Y(r.y)) <= t) return "nw";
            if (Math.hypot(p[0] - X(r.x + r.w), p[1] - Y(r.y)) <= t) return "ne";
            if (Math.hypot(p[0] - X(r.x), p[1] - Y(r.y + r.h)) <= t) return "sw";
            if (Math.hypot(p[0] - X(r.x + r.w), p[1] - Y(r.y + r.h)) <= t) return "se";
            if (Math.hypot(p[0] - X(r.x + r.w / 2), p[1] - Y(r.y)) <= t) return "n";
            if (Math.hypot(p[0] - X(r.x + r.w / 2), p[1] - Y(r.y + r.h)) <= t) return "s";
            if (Math.hypot(p[0] - X(r.x), p[1] - Y(r.y + r.h / 2)) <= t) return "w";
            if (Math.hypot(p[0] - X(r.x + r.w), p[1] - Y(r.y + r.h / 2)) <= t) return "e";
            const m = toImg(p);
            if (m && m.x >= r.x && m.x <= r.x + r.w && m.y >= r.y && m.y <= r.y + r.h) return "move";
            return null;
        }

        const origDown = node.onMouseDown;
        node.onMouseDown = function (e, pos) {
            if (inRect(pos, S.btns.up)) { getFileInput((f) => upload(f)).click(); return true; }
            if (inRect(pos, S.btns.rs)) { initRect(); updateInfo(); pushCoords(); node.setDirtyCanvas(true, true); return true; }
            if (S.waiting && inRect(pos, S.btns.go)) { resumeNow(); return true; }
            if (!S.imageRect || !S.disp) return false;
            const h = handleAt(pos);
            if (h) { const m = toImg(pos); S.rectDrag = h; S.rectStart = { mx: m.x, my: m.y, r: { ...S.cropRect } }; return true; }
            if (origDown) return origDown.apply(this, arguments);
            return false;
        };

        const origMove = node.onMouseMove;
        node.onMouseMove = function (e, pos) {
            if (S.rectDrag && S.rectStart) {
                const m = toImg(pos), o = S.rectStart.r, dx = m.x - S.rectStart.mx, dy = m.y - S.rectStart.my;
                let x = o.x, y = o.y, w = o.w, h = o.h;
                const isP = getW("crop_mode") === "Preset Ratio";
                const ratio = isP ? parseRatio(getW("aspect_ratio")) : null;
                if (S.rectDrag === "move") { x = o.x + dx; y = o.y + dy; }
                else if (isP && ratio) {
                    if (S.rectDrag === "n" || S.rectDrag === "s") {
                        const ay = S.rectDrag === "s" ? o.y : o.y + o.h;
                        h = Math.abs(m.y - ay); w = h * ratio;
                        x = o.x + (o.w - w) / 2;
                        y = S.rectDrag === "s" ? ay : ay - h;
                    } else if (S.rectDrag === "e" || S.rectDrag === "w") {
                        const ax = S.rectDrag === "e" ? o.x : o.x + o.w;
                        w = Math.abs(m.x - ax); h = w / ratio;
                        y = o.y + (o.h - h) / 2;
                        x = S.rectDrag === "e" ? ax : ax - w;
                    } else {
                        const ax = S.rectDrag.includes("w") ? o.x + o.w : o.x;
                        const ay = S.rectDrag.includes("n") ? o.y + o.h : o.y;
                        let dw = Math.abs(m.x - ax), dh = Math.abs(m.y - ay);
                        if (dw / ratio > dh) dw = dh * ratio; else dh = dw / ratio;
                        w = dw; h = dh;
                        x = S.rectDrag.includes("w") ? ax - w : ax;
                        y = S.rectDrag.includes("n") ? ay - h : ay;
                    }
                    if (w < 32) { w = 32; h = w / ratio; }
                    if (h < 32) { h = 32; w = h * ratio; }
                    if (w > S.imageRect.width) { const s = S.imageRect.width / w; w *= s; h *= s; }
                    if (h > S.imageRect.height) { const s = S.imageRect.height / h; w *= s; h *= s; }
                    if (x < 0) x = 0; if (y < 0) y = 0;
                    if (x + w > S.imageRect.width) x = S.imageRect.width - w;
                    if (y + h > S.imageRect.height) y = S.imageRect.height - h;
                } else {
                    if (S.rectDrag.includes("e")) w = o.w + dx;
                    if (S.rectDrag.includes("w")) { x = o.x + dx; w = o.w - dx; }
                    if (S.rectDrag.includes("s")) h = o.h + dy;
                    if (S.rectDrag.includes("n")) { y = o.y + dy; h = o.h - dy; }
                    if (w < 32) w = 32; if (h < 32) h = 32;
                    if (x < 0) x = 0; if (y < 0) y = 0;
                    if (x + w > S.imageRect.width) w = S.imageRect.width - x;
                    if (y + h > S.imageRect.height) h = S.imageRect.height - y;
                }
                S.cropRect = alignRect({ x, y, w, h });
                updateInfo(); pushCoords(); node.setDirtyCanvas(true, true); return;
            }
            if (origMove) return origMove.apply(this, arguments);
        };

        const origUp = node.onMouseUp;
        node.onMouseUp = function (e, pos) {
            S.rectDrag = null; S.rectStart = null;
            if (origUp) return origUp.apply(this, arguments);
        };

        // === ЗАГРУЗКА ===
        async function upload(file) {
            const fd = new FormData(); fd.append("image", file); fd.append("overwrite", "false");
            S.status = "📤 Загрузка...";
            node.setDirtyCanvas(true, true);
            try {
                const r = await fetch("/upload/image", { method: "POST", body: fd });
                if (r.status === 200) {
                    const d = await r.json();
                    if (d && d.name) {
                        const w = node.widgets.find(w => w.name === "image_name");
                        if (w) { w.value = d.name; if (w.callback) w.callback(d.name); }
                        S.srcLabel = "🖼️ image_name";
                        loadByName(d.name);
                    }
                } else { S.status = "❌ Ошибка загрузки"; node.setDirtyCanvas(true, true); }
            } catch (e) { S.status = "❌ Ошибка соединения"; node.setDirtyCanvas(true, true); }
        }

        function loadByName(name, reset = true) {
            if (!name) { S.img = null; S.imageRect = null; updateInfo(); node.setDirtyCanvas(true, true); return; }
            const im = new Image();
            im.onload = () => {
                S.img = im; S.imageRect = { width: im.naturalWidth, height: im.naturalHeight };
                if (reset || !S.cropRect.w) initRect();
                else { S.cropRect = alignRect(S.cropRect); pushCoords(); updateInfo(); }
                node.setDirtyCanvas(true, true);
            };
            im.onerror = () => { S.img = null; S.imageRect = null; updateInfo(); node.setDirtyCanvas(true, true); };
            im.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input`);
        }

        // === DRAG & DROP (ИСПРАВЛЕНО) ===
        // Корректный перевод экранных координат в координаты канваса.
        function canvasPosFromEvent(e) {
            const canvas = app.canvas;
            if (!canvas || !canvas.canvas) return null;
            const rect = canvas.canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            const ds = canvas.ds;
            if (ds) {
                const scale = ds.scale || 1;
                const off = ds.offset || [0, 0];
                const offX = Array.isArray(off) ? off[0] : (off.x || 0);
                const offY = Array.isArray(off) ? off[1] : (off.y || 0);
                return [sx / scale - offX, sy / scale - offY];
            }
            return [sx, sy];
        }
        function isMouseInNode(e) {
            const pos = canvasPosFromEvent(e);
            if (!pos || !isFinite(pos[0]) || !isFinite(pos[1])) return false;
            return node.pos && node.size &&
                   pos[0] >= node.pos[0] && pos[0] <= node.pos[0] + node.size[0] &&
                   pos[1] >= node.pos[1] && pos[1] <= node.pos[1] + node.size[1];
        }

        if (!node._agsoft_dnd_bound) {
            node._agsoft_dnd_bound = true;
            const onDragOver = (e) => {
                if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes("Files")) return;
                if (isMouseInNode(e)) {
                    if (!S.dragOver) { S.dragOver = true; updateInfo(); node.setDirtyCanvas(true, true); }
                    e.preventDefault(); e.stopPropagation();
                    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
                } else if (S.dragOver) {
                    S.dragOver = false; updateInfo(); node.setDirtyCanvas(true, true);
                }
            };
            const onDragLeave = (e) => {
                if (S.dragOver && !isMouseInNode(e)) {
                    S.dragOver = false; updateInfo(); node.setDirtyCanvas(true, true);
                }
            };
            const onDrop = (e) => {
                if (!S.dragOver) return; // не наша нода — не мешаем ComfyUI
                S.dragOver = false;
                const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
                if (f && f.type && f.type.startsWith("image/")) {
                    e.preventDefault(); e.stopPropagation();
                    upload(f);
                } else {
                    S.status = "❌ Только изображения (PNG/JPG/WebP/BMP)";
                    updateInfo(); node.setDirtyCanvas(true, true);
                }
            };
            window.addEventListener("dragover", onDragOver, true);
            window.addEventListener("dragleave", onDragLeave, true);
            window.addEventListener("drop", onDrop, true);
            node._agsoft_dnd_cleanup = () => {
                window.removeEventListener("dragover", onDragOver, true);
                window.removeEventListener("dragleave", onDragLeave, true);
                window.removeEventListener("drop", onDrop, true);
            };
        }

        // === СИНХРОНИЗАЦИЯ ИСТОЧНИКА + ПАУЗА ===
        async function pollState() {
            try {
                const r = await fetch(api.apiURL(`/agsoft/crop_preview_state?node_id=${encodeURIComponent(node.id)}`));
                if (!r.ok) return;
                const st = await r.json();
                const w = !!st.waiting;
                if (w !== S.waiting) { S.waiting = w; updateInfo(); node.setDirtyCanvas(true, true); }
                if (st.custom && st.image && node.__crop_last_image !== st.image) {
                    node.__crop_last_image = st.image;
                    S.srcLabel = st.kind === "tensor" ? "🧠 input_image" : "📂 custom_path";
                    loadByName(st.image, false);
                }
            } catch (e) { /* ignore */ }
        }

        async function liveSync() {
            const custom = node.widgets.find(w => w.name === "custom_path");
            if (!custom) return;
            const p = String(custom.value || "").trim();
            if (!p) return;
            try {
                const r = await fetch(api.apiURL("/agsoft/crop_ensure_preview"), {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ custom_path: p })
                });
                if (r.ok) {
                    const d = await r.json();
                    if (d.image) { S.srcLabel = "📂 custom_path"; loadByName(d.image); }
                } else {
                    S.srcLabel = "📂 custom_path"; S.img = null; S.imageRect = null;
                    updateInfo(); S.status = "📂 custom_path: файл не распознан как изображение.";
                    node.setDirtyCanvas(true, true);
                }
            } catch (e) { /* ignore */ }
        }

        const pollTimer = setInterval(pollState, 1000);
        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            clearInterval(pollTimer);
            if (this._agsoft_dnd_cleanup) this._agsoft_dnd_cleanup();
            if (origRemoved) return origRemoved.apply(this, arguments);
        };

        api.addEventListener("agsoft_crop_waiting", (e) => {
            if (String((e.detail || {}).node_id) === String(node.id)) pollState();
        });
        api.addEventListener("agsoft_crop_resumed", (e) => {
            if (String((e.detail || {}).node_id) === String(node.id)) pollState();
        });

        // === КОМБО image_name + АВТО-ВЫБОР ПОСЛЕДНЕГО ИЗОБРАЖЕНИЯ ===
        const iw = node.widgets.find(w => w.name === "image_name");
        if (iw) {
            const o = iw.callback;
            iw.callback = (v) => { if (o) o(v); if (v) { S.srcLabel = "🖼️ image_name"; loadByName(v); } };
            const autoPickLast = () => {
                const vals = (iw.options && iw.options.values) || [];
                if (!vals.length) return;
                if (iw.value && vals.includes(iw.value)) return;
                const last = vals[vals.length - 1];
                if (!last) return;
                iw.value = last;
                S.srcLabel = "🖼️ image_name";
                loadByName(last);
                node.setDirtyCanvas(true, true);
            };
            // Читает сохранённый crop_coords из виджета в S.cropRect (не теряем рамку при загрузке).
            const pullCoords = () => {
                const cw = node.widgets.find(w => w.name === "crop_coords");
                if (!cw) return false;
                try {
                    const d = JSON.parse(String(cw.value || "[]"));
                    if (d && typeof d === "object" && !Array.isArray(d) && "x" in d && "w" in d) {
                        S.cropRect = { x: +d.x, y: +d.y, w: +d.w, h: +d.h };
                        return true;
                    }
                } catch (e) { /* ignore */ }
                return false;
            };
            let configuredOnce = false; // флаг: нода создана загрузкой воркфлоу (onConfigure был)
            const origConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
                configuredOnce = true;
                setTimeout(() => {
                    if (iw.value) {
                        S.srcLabel = "🖼️ image_name";
                        const had = pullCoords();     // восстанавливаем сохранённую рамку
                        loadByName(iw.value, !had);   // есть coords -> НЕ сбрасывать рамку
                    } else {
                        autoPickLast();
                    }
                    node.setDirtyCanvas(true, true);
                }, 0);
                return r;
            };
            // Нова нода из меню: onConfigure не вызывается -> авто-выбор как раньше.
            setTimeout(() => { if (!configuredOnce) autoPickLast(); }, 0);
        }

        ["crop_mode", "aspect_ratio", "manual_width", "manual_height"].forEach(n => {
            const w = node.widgets.find(w => w.name === n);
            if (w) { const o = w.callback; w.callback = (v) => { if (o) o(v); if (S.imageRect) initRect(); node.setDirtyCanvas(true, true); }; }
        });
        const mw = node.widgets.find(w => w.name === "multiple");
        if (mw) { const o = mw.callback; mw.callback = (v) => { if (o) o(v); if (S.imageRect) { S.cropRect = alignRect(S.cropRect); } updateInfo(); pushCoords(); node.setDirtyCanvas(true, true); }; }
        const cw2 = node.widgets.find(w => w.name === "custom_path");
        if (cw2) { const o = cw2.callback; cw2.callback = (v) => { if (o) o(v); setTimeout(liveSync, 200); }; }

        api.addEventListener("executed", (e) => {
            const id = e.detail?.display_node || e.detail?.node;
            if (id == node.id) pollState();
        });

        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, linked, link_info) {
            const r = origConn ? origConn.apply(this, arguments) : undefined;
            if (type === 0 && this.inputs && this.inputs[index]) {
                const nm = this.inputs[index].name;
                if (nm === "input_image" || nm === "mask") {
                    const imgIn = this.inputs.find(i => i.name === "input_image");
                    if (imgIn && imgIn.link) {
                        S.srcLabel = "🧠 input_image";
                        S.status = "🧠 input_image: нажмите Queue — будет пауза для кропа.";
                        node.setDirtyCanvas(true, true);
                    } else if (!imgIn || !imgIn.link) {
                        const hasCustom = cw2 && String(cw2.value || "").trim();
                        if (!hasCustom) { S.srcLabel = "🖼️ image_name"; updateInfo(); node.setDirtyCanvas(true, true); }
                    }
                }
            }
            return r;
        };

        node.setSize([Math.max(node.size[0], 380), Math.max(node.size[1], 700)]);
        pollState();
        if (cw2 && String(cw2.value || "").trim()) setTimeout(liveSync, 300);
        node.setDirtyCanvas(true, true);
    }
});