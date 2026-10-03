// ==============================================================================
// AGSoft_Seed.js
// ==============================================================================
// Frontend extension for the 🎲AGSoft Seed node.
// Version: v10.03
//
// WHAT IT DOES
// • Live "Used Seed: N" line (DOM widget) at the bottom of the node:
//    - shows the actual first seed of the cycle, i.e. the base seed mapped
//      into [min_seed, max_seed] with the same modulo math as the Python side;
//    - the mapped value is written back into the seed widget, so the widget,
//      the line and output[0] always show the same number;
//    - refreshes on any change of seed / offset / min_seed / max_seed.
// • 🎲 Randomize button: sets a new random base seed (0..2^53-1).
// • 📋 Copy Used Seed button: copies the shown seed to the clipboard.
// • Hard-fixed node height: title + widget rows + DOM block; the node bottom
//   always coincides with the bottom of the buttons; resize changes width only.
// • All UI code is wrapped in try/catch: the extension can never crash the app.
// ------------------------------------------------------------------------------
// Фронтенд-расширение для ноды 🎲AGSoft Seed.
// Версия: v10.03
//
// ЧТО ДЕЛАЕТ
// • Живая строка "Used Seed: N" (DOM-виджет) внизу ноды:
//    - показывает фактический первый сид цикла — базу, отображённую
//      в [min_seed, max_seed] по той же modulo-математике, что и в Python;
//    - отображённое значение пишется обратно в виджет seed, поэтому виджет,
//      строка и output[0] всегда показывают одно и то же число;
//    - обновляется при изменении seed / offset / min_seed / max_seed.
// • Кнопка 🎲 Randomize: ставит новую случайную базу (0..2^53-1).
// • Кнопка 📋 Copy Used Seed: копирует показанный сид в буфер обмена.
// • Жёстко фиксированная высота ноды: заголовок + строки виджетов + DOM-блок;
//   низ ноды всегда совпадает с низом кнопок; ресайз меняет только ширину.
// • Весь UI-код обёрнут в try/catch: расширение не может уронить приложение.
//
// Author: AGSoft
// Date: 03.10.2026
// ==============================================================================
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Seed] JS extension loaded");

// Fixed geometry constants / Константы фиксированной геометрии
const DOM_H = 84;   // display + buttons block with margin / блок дисплея и кнопок с отступом
const ROW_H = 22;   // one widget row height / высота одной строки виджета
const TITLE_H = 32; // node title height / высота заголовка ноды
const PAD = 4;      // bottom padding / нижний отступ
const MIN_W = 300;  // minimal node width / минимальная ширина ноды

app.registerExtension({
    name: "AGSoft.Seed",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftSeed") return;
        try {
            const wrapper = document.createElement("div");
            wrapper.style.display = "flex";
            wrapper.style.flexDirection = "column";
            wrapper.style.alignItems = "center";
            wrapper.style.gap = "6px";
            wrapper.style.marginTop = "4px";
            wrapper.style.width = "100%";
            wrapper.style.height = DOM_H + "px";
            wrapper.style.boxSizing = "border-box";
            wrapper.style.overflow = "hidden";

            const display = document.createElement("div");
            display.style.color = "#fff";
            display.style.fontFamily = "monospace";
            display.style.fontSize = "15px";
            display.style.fontWeight = "bold";
            display.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
            display.style.padding = "5px 10px";
            display.style.borderRadius = "4px";
            display.style.textAlign = "center";
            display.style.width = "100%";
            display.style.boxSizing = "border-box";
            display.textContent = "Used Seed: -";

            const btnRow = document.createElement("div");
            btnRow.style.display = "flex";
            btnRow.style.gap = "6px";
            btnRow.style.width = "100%";

            const randomBtn = document.createElement("button");
            randomBtn.textContent = "🎲 Randomize";
            randomBtn.style.flex = "1";
            randomBtn.style.padding = "5px";
            randomBtn.style.cursor = "pointer";
            randomBtn.style.backgroundColor = "#233";
            randomBtn.style.color = "#fff";
            randomBtn.style.border = "1px solid #555";
            randomBtn.style.borderRadius = "4px";
            randomBtn.title = "Set a new random base seed (0..2^53-1).\n---\nПоставить новую случайную базу (0..2^53-1).";

            const copyBtn = document.createElement("button");
            copyBtn.textContent = "📋 Copy Used Seed";
            copyBtn.style.flex = "1";
            copyBtn.style.padding = "5px";
            copyBtn.style.cursor = "pointer";
            copyBtn.style.backgroundColor = "#323";
            copyBtn.style.color = "#fff";
            copyBtn.style.border = "1px solid #555";
            copyBtn.style.borderRadius = "4px";
            copyBtn.title = "Copy the shown used seed to the clipboard.\n---\nСкопировать показанный использованный сид в буфер обмена.";
            copyBtn.onclick = () => {
                const text = display.textContent.replace("Used Seed: ", "");
                if (text !== "-") {
                    navigator.clipboard.writeText(text);
                    copyBtn.textContent = "✅ Copied!";
                    setTimeout(() => { copyBtn.textContent = "📋 Copy Used Seed"; }, 1500);
                }
            };

            btnRow.appendChild(randomBtn);
            btnRow.appendChild(copyBtn);
            wrapper.appendChild(display);
            wrapper.appendChild(btnRow);

            // DOM widget with fixed height / DOM-виджет с фиксированной высотой
            const domWidget = node.addDOMWidget("seed_display", "div", wrapper);
            domWidget.computeSize = (w) => [w || MIN_W, DOM_H];

            // Hard-fixed node height: title + widget rows + DOM block; resize changes width only
            // Жестко фиксированная высота ноды: заголовок + строки виджетов + DOM-блок; ресайз меняет только ширину
            const calcH = () => TITLE_H + (node.widgets ? node.widgets.length : 0) * ROW_H + DOM_H + PAD;
            node.computeSize = (w) => [Math.max(w || 0, MIN_W), calcH()];
            node.onResize = (s) => { node.size = [Math.max(s[0], MIN_W), calcH()]; };
            const fixSize = () => {
                node.size = [Math.max(node.size[0], MIN_W), calcH()];
                if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
            };
            setTimeout(fixSize, 60);

            // Actual seed = base mapped into [min_seed, max_seed]; write it back to the widget,
            // so widget, Used Seed and output[0] always match
            // Фактический сид = база, отображённая в [min_seed, max_seed]; пишем его обратно в виджет,
            // чтобы виджет, Used Seed и output[0] всегда совпадали
            const getW = (n) => (node.widgets ? node.widgets.find((x) => x.name === n) : null);
            const num = (n) => { const w = getW(n); return w ? Number(w.value) : 0; };
            const clamp = (v, lo, hi) => {
                if (hi < lo) { const t = lo; lo = hi; hi = t; }
                const span = (hi - lo) + 1;
                return lo + ((((v - lo) % span) + span) % span);
            };
            const usedNow = () => clamp(num("seed") + num("offset"), num("min_seed"), num("max_seed"));
            const sync = () => {
                const u = usedNow();
                display.textContent = "Used Seed: " + u;
                const w = getW("seed");
                if (w && Number(w.value) !== u) w.value = u;
            };
            setTimeout(() => {
                sync();
                const names = ["seed", "offset", "min_seed", "max_seed"];
                for (let i = 0; i < names.length; i++) {
                    const w = getW(names[i]);
                    if (!w) continue;
                    const prev = w.callback;
                    const oc = prev ? (v) => prev.call(w, v) : null;
                    w.callback = (v) => {
                        if (oc) oc(v);
                        sync();
                    };
                }
                randomBtn.onclick = () => {
                    const rw = getW("seed");
                    if (!rw) return;
                    rw.value = Math.floor(Math.random() * 9007199254740991);
                    if (rw.callback) rw.callback(rw.value);
                };
            }, 60);

            // Restore size and display on workflow load / Восстановление размера и дисплея при загрузке workflow
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
                setTimeout(fixSize, 60);
                setTimeout(sync, 60);
            };
        } catch (e) {
            console.warn("[AGSoft Seed] UI init skipped:", e);
        }
    }
});