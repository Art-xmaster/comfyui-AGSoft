// ==============================================================================
// AGSoft_Image_Crop_Plus.js  (v3 — responsive DOM widget: Classic + Nodes 2.0)
// ==============================================================================
// JS extension for the 🖼️✂️AGSoft Image Crop Plus node.
// JS-расширение для ноды 🖼️✂️AGSoft Image Crop Plus.
//
// ⚡ Responsive preview: the DOM widget fills all free node space; the image
//   is letterboxed into the stage on any node resize (vertical + horizontal).
//   Адаптивное превью: DOM-виджет занимает всё свободное место ноды; изображение
//   вписывается letterbox'ом при любом ресайзе ноды (вертикаль + горизонталь).
// ⚡ Controls (Upload / Reset / Resume + info lines) are pinned to node bottom.
//   Кнопки и инфостроки прижаты к низу ноды.
// ⚡ English UI with bilingual tooltips (EN --- RU).
//   Английский интерфейс с двуязычными тултипами (EN --- RU).
// ⚡ Pointer Capture: drag never sticks, even outside the node bounds.
//   Pointer Capture: drag не залипает даже за пределами ноды.
// ⚡ Anchored edge math (no jitter at image borders), nearest-handle picking.
//   Якорная математика граней (без джиттера у краёв), выбор ближайшего хэндла.
// ⚡ Sources sync: 🖼️ image_name / 📂 custom_path / 🧠 input_image; pause + Resume.
//   Синхронизация источников; пауза при тензоре + ▶️ Resume.
//
// Author: AGSoft
// Date: 21.11.2026 (v3)
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Image Crop Plus] JS extension loaded v11.21");

const NODE_CLASS_NAME = "AGSoft Image Crop Plus";
const MIN_STAGE_H = 140;
const CONTROLS_H = 66;   // buttons 30 + gap 6 + status 15 + size 15
const WRAP_PAD = 16;     // wrap padding 8 * 2
const STAGE_GAP = 8;
const MIN_WIDGET_H = MIN_STAGE_H + STAGE_GAP + CONTROLS_H + WRAP_PAD;
const HANDLE_T = 12;
const MIN_CROP = 32;

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

let stylesInjected = false;
function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const st = document.createElement("style");
    st.textContent = `
.agc-wrap{box-sizing:border-box;width:100%;padding:8px;display:flex;flex-direction:column;gap:8px;
font:12px/1.35 system-ui,sans-serif;color:#ccc;user-select:none;-webkit-user-select:none;overflow:hidden;}
.agc-wrap *{box-sizing:border-box;}
.agc-stage{position:relative;flex:1 1 auto;min-height:${MIN_STAGE_H}px;background:#141414;
border:1px solid #333;border-radius:6px;overflow:hidden;}
.agc-stage canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;
cursor:crosshair;}
.agc-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
text-align:center;padding:16px;color:#9a9a9a;font-size:12px;white-space:pre-line;pointer-events:none;}
.agc-drop{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
flex-direction:column;gap:6px;background:rgba(70,130,180,.28);border:2px dashed #4a9eff;
border-radius:6px;color:#fff;font-weight:600;pointer-events:none;text-shadow:0 1px 3px rgba(0,0,0,.8);}
.agc-wrap.agc-dragover .agc-drop{display:flex;}
.agc-wrap.agc-waiting .agc-stage{border-color:#d9a441;animation:agcPulse 1.6s ease-in-out infinite;}
@keyframes agcPulse{0%,100%{box-shadow:0 0 0 1px rgba(217,164,65,.45);}
50%{box-shadow:0 0 0 2px rgba(217,164,65,.9),0 0 14px rgba(217,164,65,.35);}}
.agc-ctrl{flex:0 0 auto;display:flex;flex-direction:column;gap:6px;margin-top:auto;}
.agc-btns{display:flex;gap:6px;height:30px;}
.agc-btn{flex:1;min-width:0;border:1px solid #444;border-radius:5px;background:#3a3a3a;color:#eee;
font:600 12px system-ui,sans-serif;cursor:pointer;padding:0 6px;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis;transition:background .12s,border-color .12s,transform .05s;}
.agc-btn:hover{background:#4a4a4a;border-color:#666;}
.agc-btn:active{transform:translateY(1px);}
.agc-btn.agc-up{background:#31506e;border-color:#3f6a94;}
.agc-btn.agc-up:hover{background:#3b6287;border-color:#5588b8;}
.agc-btn.agc-go{background:#2f6b2f;border-color:#3f8a3f;display:none;}
.agc-btn.agc-go:hover{background:#387d38;border-color:#57a857;}
.agc-wrap.agc-waiting .agc-btn.agc-go{display:block;}
.agc-status{height:15px;line-height:15px;text-align:center;color:#bbb;font-size:11px;overflow:hidden;
white-space:nowrap;text-overflow:ellipsis;}
.agc-size{height:15px;line-height:15px;text-align:center;color:#8f8f8f;font-size:11px;
font-variant-numeric:tabular-nums;overflow:hidden;white-space:nowrap;}
`;
    document.head.appendChild(st);
}

app.registerExtension({
    name: "AGSoft.ImageCropPlusCanvas",
    async nodeCreated(node) {
        if (node.comfyClass !== NODE_CLASS_NAME) return;
        if (typeof node.addDOMWidget !== "function") {
            console.warn("[AGSoft Image Crop Plus] addDOMWidget unavailable — update ComfyUI frontend.");
            return;
        }
        injectStyles();

        const S = {
            img: null, imageRect: null,
            cropRect: { x: 0, y: 0, w: 0, h: 0 },
            disp: null, drag: null, start: null,
            srcLabel: "", waiting: false, dragOver: false,
            tensorHint: false, flash: null,
            _liveT: null, _flashT: null,
        };
        node._crop = S;

        const getW = (n) => { const w = node.widgets.find(w => w.name === n); return w ? w.value : null; };
        const clamp = (v, a, b) => Math.max(a, Math.min(v, b));
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

        // === DOM WIDGET / DOM-ВИДЖЕТ ===
        const wrap = document.createElement("div");
        wrap.className = "agc-wrap";

        const stage = document.createElement("div");
        stage.className = "agc-stage";
        stage.title = "Crop preview: drag the frame by center / corners / edges.\n---\n" +
            "Превью кропа: тяните рамку за центр / углы / грани.";
        const cv = document.createElement("canvas");
        const empty = document.createElement("div");
        empty.className = "agc-empty";
        const drop = document.createElement("div");
        drop.className = "agc-drop";
        drop.innerHTML = '<div style="font-size:22px">📥</div><div>DROP IMAGE HERE</div>' +
            '<div style="font-weight:400;font-size:11px">Release to upload / Отпустите для загрузки</div>';
        stage.append(cv, empty, drop);

        const ctrl = document.createElement("div");
        ctrl.className = "agc-ctrl";
        const btns = document.createElement("div");
        btns.className = "agc-btns";
        const mkBtn = (label, cls, title) => {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "agc-btn" + (cls ? " " + cls : "");
            b.textContent = label;
            b.title = title;
            return b;
        };
        const bUp = mkBtn("📁 Upload", "agc-up",
            "Upload an image into the input folder.\n---\nЗагрузить изображение в папку input.");
        const bRs = mkBtn("🗑️ Reset", "",
            "Reset the crop frame to the default position.\n---\nСбросить рамку кропа в исходное положение.");
        const bGo = mkBtn("▶️ Resume", "agc-go",
            "Resume graph execution with the current crop.\n---\nПродолжить выполнение графа с текущим кропом.");
        btns.append(bUp, bRs, bGo);
        const stLine = document.createElement("div");
        stLine.className = "agc-status";
        const szLine = document.createElement("div");
        szLine.className = "agc-size";
        ctrl.append(btns, stLine, szLine);
        wrap.append(stage, ctrl);

        // Keep node dragging away from preview interactions.
        // Не даём ноде таскаться при работе с превью.
        wrap.addEventListener("pointerdown", (e) => e.stopPropagation());

        const ctx = cv.getContext("2d");
        let dpr = 1;

        function resizeCanvas() {
            const w = stage.clientWidth || 300, h = stage.clientHeight || MIN_STAGE_H;
            dpr = Math.min(2, window.devicePixelRatio || 1);
            const bw = Math.max(1, Math.round(w * dpr)), bh = Math.max(1, Math.round(h * dpr));
            if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
            draw();
        }
        const ro = new ResizeObserver(resizeCanvas);
        ro.observe(stage);

        function draw() {
            const w = cv.width / dpr, h = cv.height / dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            S.disp = null;
            if (!S.img || !S.imageRect) { empty.style.display = "flex"; return; }
            empty.style.display = "none";
            const scale = Math.min(w / S.imageRect.width, h / S.imageRect.height);
            const dw = S.imageRect.width * scale, dh = S.imageRect.height * scale;
            const dx = (w - dw) / 2, dy = (h - dh) / 2;
            S.disp = { dx, dy, dw, dh, scale };
            ctx.fillStyle = "#161616"; ctx.fillRect(0, 0, w, h);
            ctx.drawImage(S.img, dx, dy, dw, dh);
            drawOverlay();
        }

        function drawOverlay() {
            const d = S.disp; if (!d) return;
            const r = S.cropRect;
            if (r.w <= 0 || r.h <= 0) return;
            const sc = d.scale;
            const rx = r.x * sc + d.dx, ry = r.y * sc + d.dy, rw = r.w * sc, rh = r.h * sc;
            ctx.fillStyle = "rgba(0,0,0,0.6)";
            ctx.fillRect(d.dx, d.dy, d.dw, ry - d.dy);
            ctx.fillRect(d.dx, ry + rh, d.dw, d.dy + d.dh - (ry + rh));
            ctx.fillRect(d.dx, ry, rx - d.dx, rh);
            ctx.fillRect(rx + rw, ry, d.dx + d.dw - (rx + rw), rh);
            ctx.fillStyle = "rgba(0,170,255,0.10)"; ctx.fillRect(rx, ry, rw, rh);
            ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 1; i < 3; i++) {
                ctx.moveTo(rx + rw * i / 3, ry); ctx.lineTo(rx + rw * i / 3, ry + rh);
                ctx.moveTo(rx, ry + rh * i / 3); ctx.lineTo(rx + rw, ry + rh * i / 3);
            }
            ctx.stroke();
            ctx.strokeStyle = "#00aaff"; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, rw, rh);
            const hs = [[rx, ry], [rx + rw / 2, ry], [rx + rw, ry], [rx, ry + rh / 2],
                        [rx + rw, ry + rh / 2], [rx, ry + rh], [rx + rw / 2, ry + rh], [rx + rw, ry + rh]];
            hs.forEach(h => {
                ctx.beginPath(); ctx.arc(h[0], h[1], 5, 0, 2 * Math.PI);
                ctx.fillStyle = "#fff"; ctx.fill();
                ctx.strokeStyle = "#00aaff"; ctx.lineWidth = 1.5; ctx.stroke();
            });
            const m = mult();
            ctx.fillStyle = "#fff"; ctx.font = "bold 14px Arial";
            ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.shadowBlur = 3; ctx.shadowColor = "black";
            ctx.fillText(`${Math.floor(r.w / m) * m}×${Math.floor(r.h / m) * m}`, rx + rw / 2, ry + rh / 2);
            ctx.shadowBlur = 0; ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
        }

        // === RESPONSIVE HEIGHT / АДАПТИВНАЯ ВЫСОТА ===
        let previewWidget = null;
        let currentH = MIN_WIDGET_H + 180;

        function estimateTop() {
            let y = node.widgets_start_y || 30;
            for (const w of node.widgets || []) {
                if (w === previewWidget) break;
                const h = w.computeSize ? w.computeSize(node.size[0])[1] : (w.height || 20);
                y += h + 4;
            }
            return y;
        }
        function widgetTop() {
            if (previewWidget && typeof previewWidget.last_y === "number" && previewWidget.last_y > 0) {
                return previewWidget.last_y;
            }
            // Vue (Nodes 2.0): measure real offset inside the node DOM root.
            // Vue (Nodes 2.0): замеряем реальное смещение внутри DOM-корня ноды.
            const root = wrap.closest(".lg-node") || wrap.closest("[data-node-id]");
            if (root) {
                const scale = (app.canvas && app.canvas.ds && app.canvas.ds.scale) || 1;
                const dr = wrap.getBoundingClientRect();
                const nr = root.getBoundingClientRect();
                if (nr.height > 0 && scale > 0) return (dr.top - nr.top) / scale;
            }
            return estimateTop();
        }
        function applyLayout() {
            if (node.flags && node.flags.collapsed) return;
            const avail = Math.max(MIN_WIDGET_H, Math.round(node.size[1] - widgetTop() - 6));
            if (Math.abs(avail - currentH) > 1) {
                currentH = avail;
                wrap.style.height = avail + "px";
                node.setDirtyCanvas(true, true);
            }
        }

        // === COORDS / STATE ===
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
            if (S.waiting) { clearTimeout(S._liveT); S._liveT = setTimeout(sendLiveCoords, 120); }
        }
        function resumeNow() {
            S.waiting = false;
            updateInfo();
            fetch(api.apiURL("/agsoft/crop_resume"), {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id) })
            }).catch(() => {});
        }
        function flash(msg) {
            S.flash = msg; updateInfo();
            clearTimeout(S._flashT);
            S._flashT = setTimeout(() => { S.flash = null; updateInfo(); }, 2600);
        }
        function updateInfo() {
            const src = S.srcLabel ? S.srcLabel + " · " : "";
            let status;
            if (S.flash) status = S.flash;
            else if (S.dragOver) status = src + "📥 Drop the file to upload...";
            else if (S.waiting) status = src + "⏸️ Paused: adjust the crop, then press ▶️ Resume.";
            else if (!S.imageRect) status = src + "⚠️ No image.";
            else status = src + "🖱️ Drag center/corners/edges. 📥 D&D supported.";
            stLine.textContent = status;
            stLine.title = status;
            if (S.imageRect) {
                const m = mult();
                const aw = Math.floor(S.cropRect.w / m) * m, ah = Math.floor(S.cropRect.h / m) * m;
                szLine.textContent =
                    `📐 ${Math.round(S.cropRect.w)}x${Math.round(S.cropRect.h)} → ×${m}: ${aw}x${ah}`;
            } else szLine.textContent = "📐 --x--";
            empty.textContent = S.tensorHint
                ? "🧠 input_image connected.\nPress Queue — the node will pause for cropping."
                : "⚠️ No image.\nPick a file, enter custom_path or connect input_image.";
            wrap.classList.toggle("agc-waiting", !!S.waiting);
            wrap.classList.toggle("agc-dragover", !!S.dragOver);
        }
        function initRect() {
            if (!S.imageRect) return;
            const mode = getW("crop_mode");
            let tw = S.imageRect.width * 0.7, th = S.imageRect.height * 0.7;
            if (mode === "Preset Ratio") {
                const r = parseRatio(getW("aspect_ratio") || "1:1");
                if (tw / th > r) tw = th * r; else th = tw / r;
            } else if (mode === "Manual Size") {
                const mw = parseInt(getW("manual_width")) || 512;
                const mh = parseInt(getW("manual_height")) || 512;
                const s = Math.min(S.imageRect.width / mw, S.imageRect.height / mh, 1);
                tw = mw * s; th = mh * s;
            }
            S.cropRect = alignRect({
                x: (S.imageRect.width - tw) / 2, y: (S.imageRect.height - th) / 2, w: tw, h: th
            });
            pushCoords(); updateInfo();
        }

        // === POINTER: capture fixes sticky drag / захват чинит залипание ===
        function localPos(e) {
            const r = cv.getBoundingClientRect();
            const kx = cv.clientWidth / (r.width || 1), ky = cv.clientHeight / (r.height || 1);
            return [(e.clientX - r.left) * kx, (e.clientY - r.top) * ky];
        }
        const toImg = (p) => {
            const d = S.disp;
            return d ? { x: (p[0] - d.dx) / d.scale, y: (p[1] - d.dy) / d.scale } : null;
        };
        function handleAt(p) {
            const d = S.disp; if (!d) return null;
            const r = S.cropRect;
            const X = v => v * d.scale + d.dx, Y = v => v * d.scale + d.dy;
            const pts = {
                nw: [r.x, r.y], n: [r.x + r.w / 2, r.y], ne: [r.x + r.w, r.y],
                e: [r.x + r.w, r.y + r.h / 2], se: [r.x + r.w, r.y + r.h],
                s: [r.x + r.w / 2, r.y + r.h], sw: [r.x, r.y + r.h], w: [r.x, r.y + r.h / 2]
            };
            let best = null, bd = HANDLE_T;
            for (const k in pts) {
                const dist = Math.hypot(p[0] - X(pts[k][0]), p[1] - Y(pts[k][1]));
                if (dist <= bd) { bd = dist; best = k; }
            }
            if (best) return best;
            const m = toImg(p);
            if (m && m.x >= r.x && m.x <= r.x + r.w && m.y >= r.y && m.y <= r.y + r.h) return "move";
            return null;
        }
        const cursorFor = (h) => ({
            nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
            n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", move: "move"
        }[h] || "crosshair");
        function endDrag() { S.drag = null; S.start = null; }

        cv.addEventListener("pointerdown", (e) => {
            if (e.button !== 0) return;
            e.preventDefault(); e.stopPropagation();
            if (!S.imageRect || !S.disp) return;
            const p = localPos(e);
            const h = handleAt(p);
            if (!h) return;
            const m = toImg(p);
            S.drag = h;
            S.start = { mx: m.x, my: m.y, r: { ...S.cropRect } };
            try { cv.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
            cv.style.cursor = cursorFor(h);
        });
        cv.addEventListener("pointermove", (e) => {
            const p = localPos(e);
            if (S.drag && S.start) {
                if (e.buttons === 0) { endDrag(); return; }
                e.preventDefault(); e.stopPropagation();
                applyDrag(p);
                return;
            }
            cv.style.cursor = cursorFor(handleAt(p));
        });
        cv.addEventListener("pointerup", (e) => { if (S.drag) { e.stopPropagation(); endDrag(); } });
        cv.addEventListener("pointercancel", endDrag);
        cv.addEventListener("lostpointercapture", endDrag);

        function applyDrag(p) {
            const m = toImg(p); if (!m) return;
            const o = S.start.r, dx = m.x - S.start.mx, dy = m.y - S.start.my;
            const IW = S.imageRect.width, IH = S.imageRect.height;
            const dir = S.drag;
            let x = o.x, y = o.y, w = o.w, h = o.h;
            if (dir === "move") {
                x = clamp(o.x + dx, 0, IW - w);
                y = clamp(o.y + dy, 0, IH - h);
            } else if (getW("crop_mode") === "Preset Ratio") {
                const ratio = parseRatio(getW("aspect_ratio") || "1:1");
                if (dir === "n" || dir === "s") {
                    const ay = dir === "s" ? o.y : o.y + o.h;
                    h = Math.abs(m.y - ay); w = h * ratio;
                    x = o.x + (o.w - w) / 2;
                    y = dir === "s" ? ay : ay - h;
                } else if (dir === "e" || dir === "w") {
                    const ax = dir === "e" ? o.x : o.x + o.w;
                    w = Math.abs(m.x - ax); h = w / ratio;
                    y = o.y + (o.h - h) / 2;
                    x = dir === "e" ? ax : ax - w;
                } else {
                    const ax = dir.includes("w") ? o.x + o.w : o.x;
                    const ay = dir.includes("n") ? o.y + o.h : o.y;
                    let dw = Math.abs(m.x - ax), dh = Math.abs(m.y - ay);
                    if (dw / ratio > dh) dw = dh * ratio; else dh = dw / ratio;
                    w = dw; h = dh;
                    x = dir.includes("w") ? ax - w : ax;
                    y = dir.includes("n") ? ay - h : ay;
                }
                if (w < MIN_CROP) { w = MIN_CROP; h = w / ratio; }
                if (h < MIN_CROP) { h = MIN_CROP; w = h * ratio; }
                if (w > IW) { const s = IW / w; w *= s; h *= s; }
                if (h > IH) { const s = IH / h; w *= s; h *= s; }
                x = clamp(x, 0, IW - w); y = clamp(y, 0, IH - h);
            } else {
                // Free mode: anchor the opposite edge (no jitter at borders).
                // Свободный режим: якорь — противоположная грань (без джиттера).
                let L = o.x, T = o.y, R = o.x + o.w, B = o.y + o.h;
                if (dir.includes("e")) R = clamp(o.x + o.w + dx, o.x + MIN_CROP, IW);
                if (dir.includes("w")) L = clamp(o.x + dx, 0, o.x + o.w - MIN_CROP);
                if (dir.includes("s")) B = clamp(o.y + o.h + dy, o.y + MIN_CROP, IH);
                if (dir.includes("n")) T = clamp(o.y + dy, 0, o.y + o.h - MIN_CROP);
                x = L; y = T; w = R - L; h = B - T;
            }
            S.cropRect = alignRect({ x, y, w, h });
            updateInfo(); pushCoords(); draw();
        }

        // === BUTTONS / КНОПКИ ===
        bUp.addEventListener("click", () => getFileInput((f) => upload(f)).click());
        bRs.addEventListener("click", () => { initRect(); draw(); });
        bGo.addEventListener("click", () => resumeNow());

        // === UPLOAD / ЗАГРУЗКА ===
        async function upload(file) {
            const fd = new FormData();
            fd.append("image", file); fd.append("overwrite", "false");
            flash("📤 Uploading...");
            try {
                const r = await fetch(api.apiURL("/upload/image"), { method: "POST", body: fd });
                if (!r.ok) throw new Error("http " + r.status);
                const d = await r.json();
                if (d && d.name) {
                    S.flash = null;
                    S.srcLabel = "🖼️ image_name";
                    const w = node.widgets.find(w => w.name === "image_name");
                    if (w) {
                        const vals = w.options && w.options.values;
                        if (Array.isArray(vals) && !vals.includes(d.name)) vals.push(d.name);
                        w.value = d.name;
                        if (w.callback) w.callback(d.name);
                        else loadByName(d.name, true, d.subfolder || "");
                    } else loadByName(d.name, true, d.subfolder || "");
                    updateInfo();
                } else flash("❌ Upload failed");
            } catch (e) { flash("❌ Connection error"); }
        }
        function loadByName(name, reset = true, subfolder = "") {
            if (!name) { S.img = null; S.imageRect = null; updateInfo(); draw(); return; }
            const im = new Image();
            im.onload = () => {
                S.img = im;
                S.imageRect = { width: im.naturalWidth, height: im.naturalHeight };
                if (reset || !S.cropRect.w) initRect();
                else { S.cropRect = alignRect(S.cropRect); pushCoords(); updateInfo(); }
                draw();
            };
            im.onerror = () => { S.img = null; S.imageRect = null; updateInfo(); draw(); };
            const sf = subfolder ? `&subfolder=${encodeURIComponent(subfolder)}` : "";
            im.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input${sf}`);
        }

        // === DRAG & DROP: widget (both renderers) + whole node ===
        const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
        function setDragOver(v) {
            if (S.dragOver === v) return;
            S.dragOver = v; updateInfo();
        }
        let dragDepth = 0;
        wrap.addEventListener("dragenter", (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault(); e.stopPropagation();
            dragDepth++; setDragOver(true);
        });
        wrap.addEventListener("dragover", (e) => {
            if (!hasFiles(e)) return;
            e.preventDefault(); e.stopPropagation();
            e.dataTransfer.dropEffect = "copy";
            setDragOver(true);
        });
        wrap.addEventListener("dragleave", () => {
            dragDepth = Math.max(0, dragDepth - 1);
            if (!dragDepth) setDragOver(false);
        });
        wrap.addEventListener("drop", (e) => {
            dragDepth = 0; setDragOver(false);
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!f) return;
            e.preventDefault(); e.stopPropagation();
            if (f.type && f.type.startsWith("image/")) upload(f);
            else flash("❌ Images only (PNG/JPG/WebP/BMP)");
        });
        function canvasPosFromEvent(e) {
            const canvas = app.canvas;
            if (!canvas || !canvas.canvas) return null;
            const rect = canvas.canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
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
        const onDragOverW = (e) => {
            if (!hasFiles(e) || wrap.contains(e.target)) return;
            if (isMouseInNode(e)) {
                setDragOver(true);
                e.preventDefault(); e.stopPropagation();
                e.dataTransfer.dropEffect = "copy";
            } else if (S.dragOver) setDragOver(false);
        };
        const onDragLeaveW = (e) => {
            if (S.dragOver && !wrap.contains(e.target) && !isMouseInNode(e)) setDragOver(false);
        };
        const onDropW = (e) => {
            if (wrap.contains(e.target) || !S.dragOver) return;
            setDragOver(false);
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!f) return;
            e.preventDefault(); e.stopPropagation();
            if (f.type && f.type.startsWith("image/")) upload(f);
            else flash("❌ Images only (PNG/JPG/WebP/BMP)");
        };
        window.addEventListener("dragover", onDragOverW, true);
        window.addEventListener("dragleave", onDragLeaveW, true);
        window.addEventListener("drop", onDropW, true);

        // === REGISTER WIDGET + SIZE HOOKS / РЕГИСТРАЦИЯ ВИДЖЕТА + ХУКИ РАЗМЕРА ===
        wrap.style.height = currentH + "px";
        previewWidget = node.addDOMWidget("agsoft_preview", "preview", wrap, {
            getValue: () => "",
            setValue: () => {},
            getMinHeight: () => MIN_WIDGET_H,
            getMaxHeight: () => 4096,
            hideOnZoom: false,
        });
        previewWidget.computeSize = function (width) { return [width, currentH]; };

        const origCompute = node.computeSize;
        node.computeSize = function () {
            // Min size uses the minimal widget height so the node can shrink.
            // Минимальный размер считает виджет на минимальной высоте (нода сжимается).
            const saved = currentH;
            currentH = MIN_WIDGET_H;
            const s = origCompute ? origCompute.apply(this, arguments) : [node.size[0], 640];
            currentH = saved;
            s[0] = Math.max(s[0], 380);
            s[1] = Math.max(s[1], estimateTop() + MIN_WIDGET_H + 8);
            return s;
        };
        const origResize = node.onResize;
        node.onResize = function () {
            const r = origResize ? origResize.apply(this, arguments) : undefined;
            applyLayout();
            return r;
        };
        const roWrap = new ResizeObserver(() => applyLayout());
        roWrap.observe(wrap);
        const layoutTimer = setInterval(applyLayout, 250);

        // === SOURCE SYNC + PAUSE / СИНХРОНИЗАЦИЯ ИСТОЧНИКА + ПАУЗА ===
        async function pollState() {
            try {
                const url = `/agsoft/crop_preview_state?node_id=${encodeURIComponent(node.id)}`;
                const r = await fetch(api.apiURL(url));
                if (!r.ok) return;
                const st = await r.json();
                const w = !!st.waiting;
                if (w !== S.waiting) { S.waiting = w; updateInfo(); }
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
                S.srcLabel = "📂 custom_path";
                if (r.ok) {
                    const d = await r.json();
                    if (d.image) { loadByName(d.image); return; }
                }
                S.img = null; S.imageRect = null;
                flash("📂 custom_path: file is not recognized as an image.");
                draw();
            } catch (e) { /* ignore */ }
        }
        const pollTimer = setInterval(pollState, 1000);
        const onWaiting = (e) => { if (String((e.detail || {}).node_id) === String(node.id)) pollState(); };
        const onResumed = (e) => { if (String((e.detail || {}).node_id) === String(node.id)) pollState(); };
        const onExecuted = (e) => {
            const id = e.detail?.display_node || e.detail?.node;
            if (id == node.id) pollState();
        };
        api.addEventListener("agsoft_crop_waiting", onWaiting);
        api.addEventListener("agsoft_crop_resumed", onResumed);
        api.addEventListener("executed", onExecuted);

        const origRemoved = node.onRemoved;
        node.onRemoved = function () {
            clearInterval(pollTimer); clearInterval(layoutTimer);
            clearTimeout(S._liveT); clearTimeout(S._flashT);
            ro.disconnect(); roWrap.disconnect();
            window.removeEventListener("dragover", onDragOverW, true);
            window.removeEventListener("dragleave", onDragLeaveW, true);
            window.removeEventListener("drop", onDropW, true);
            api.removeEventListener("agsoft_crop_waiting", onWaiting);
            api.removeEventListener("agsoft_crop_resumed", onResumed);
            api.removeEventListener("executed", onExecuted);
            if (origRemoved) return origRemoved.apply(this, arguments);
        };

        // === image_name COMBO + AUTO-PICK LAST + FRAME RESTORE ===
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
            };
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
            let configuredOnce = false;
            const origConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
                configuredOnce = true;
                setTimeout(() => {
                    if (iw.value) {
                        S.srcLabel = "🖼️ image_name";
                        const had = pullCoords();
                        loadByName(iw.value, !had);
                    } else autoPickLast();
                    applyLayout();
                    node.setDirtyCanvas(true, true);
                }, 0);
                return r;
            };
            setTimeout(() => { if (!configuredOnce) autoPickLast(); }, 0);
        }
        ["crop_mode", "aspect_ratio", "manual_width", "manual_height"].forEach(n => {
            const w = node.widgets.find(w => w.name === n);
            if (w) {
                const o = w.callback;
                w.callback = (v) => { if (o) o(v); if (S.imageRect) { initRect(); draw(); } };
            }
        });
        const mw = node.widgets.find(w => w.name === "multiple");
        if (mw) {
            const o = mw.callback;
            mw.callback = (v) => {
                if (o) o(v);
                if (S.imageRect) S.cropRect = alignRect(S.cropRect);
                updateInfo(); pushCoords(); draw();
            };
        }
        const cw2 = node.widgets.find(w => w.name === "custom_path");
        if (cw2) { const o = cw2.callback; cw2.callback = (v) => { if (o) o(v); setTimeout(liveSync, 200); }; }

        const origConn = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, linked, link_info) {
            const r = origConn ? origConn.apply(this, arguments) : undefined;
            if (type === 0 && this.inputs && this.inputs[index]) {
                const nm = this.inputs[index].name;
                if (nm === "input_image" || nm === "mask") {
                    const imgIn = this.inputs.find(i => i.name === "input_image");
                    S.tensorHint = !!(imgIn && imgIn.link != null);
                    if (S.tensorHint) S.srcLabel = "🧠 input_image";
                    else {
                        const hasCustom = cw2 && String(cw2.value || "").trim();
                        if (!hasCustom) S.srcLabel = "🖼️ image_name";
                    }
                    updateInfo();
                }
            }
            return r;
        };

        // === START / СТАРТ ===
        node.setSize([Math.max(node.size[0], 400), Math.max(node.size[1], 760)]);
        applyLayout();
        resizeCanvas();
        updateInfo();
        pollState();
        if (cw2 && String(cw2.value || "").trim()) setTimeout(liveSync, 300);
        node.setDirtyCanvas(true, true);
    }
});