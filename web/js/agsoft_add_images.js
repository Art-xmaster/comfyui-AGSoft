// ==============================================================================
// AGSoft_Add_Images.js
// ==============================================================================
// JS-расширение для ноды 🖼️AGSoft Add Images.
//
// Возможности / Features:
// ⚡ Динамические опциональные входы: держим ровно (подключено + 1) слотов,
//   мин 1, макс 50; подключение к последнему слоту создаёт следующий,
//   отключение последнего подключённого убирает лишний пустой.
//   Dynamic optional inputs: exactly (connected + 1) slots kept, min 1,
//   max 50; connecting the last slot creates the next, disconnecting the
//   last connected removes the extra empty one.
// ⚡ Синхронный трим 50 объявленных слотов при создании ноды (нода не
//   "мелькает" длинной), пересчёт высоты ноды после добавления/удаления.
//   Synchronous trim of the 50 declared slots on node creation (no tall
//   flash), node height recompute after add/remove.
// ⚡ Страховка после загрузки workflow (setTimeout): повторный трим, если
//   подключений нет.
//   Safety net after workflow load (setTimeout): re-trim when no links.
//
// JS extension for the 🖼️ AGSoft Add Images node.
// (See the RU list above — the same features.)
// ==============================================================================
import { app } from "../../../scripts/app.js";

console.log("[AGSoft Add Images] JS extension loaded v09.25");

const MAX_IN = 50;

function agsoftNormalize(node) {
    // Keep exactly (connected + 1) inputs, min 1, max 50, and fix node height / Держим ровно (подключено + 1) входов, мин 1, макс 50, и чиним высоту ноды
    const connected = node.inputs.filter((i) => i.link !== null && i.link !== undefined).length;
    const target = Math.min(MAX_IN, connected + 1);
    while (node.inputs.length > target) node.removeInput(node.inputs.length - 1);
    while (node.inputs.length < target) node.addInput("image_" + (node.inputs.length + 1), "IMAGE");
    node.inputs.forEach((inp, i) => { inp.name = "image_" + (i + 1); });
    const cs = node.computeSize();
    node.size[1] = cs[1];
    if (node.size[0] < cs[0]) node.size[0] = cs[0];
    if (app.graph) app.graph.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "AGSoft.AGSoft_Add_Images",
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoft_Add_Images") return;
        // Synchronous trim: new node must not flash with 50 slots / Синхронный трим: новая нода не должна мелькать с 50 слотами
        agsoftNormalize(node);
        const orig = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            if (orig) orig.apply(this, arguments);
            if (type === 1) agsoftNormalize(this); // LiteGraph.INPUT = 1 / входы
        };
        // Safety net after workflow load / Страховка после загрузки workflow
        setTimeout(() => {
            if (!node.inputs) return;
            const linked = node.inputs.some((i) => i.link !== null && i.link !== undefined);
            if (!linked && node.inputs.length > 1) agsoftNormalize(node);
        }, 0);
    }
});