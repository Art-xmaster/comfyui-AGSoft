/*
==============================================================================
AGSoft_SetGet.js
==============================================================================
Нода: 🧩 AGSoft Set / 🧩 AGSoft Get
Описание / Description:
    Виртуальная беспроводная пара: ядро резолвит выход Get в источник Set,
    значения не копируются, кэш и частичный прогон работают как с проводом.
    ---
    Virtual wireless pair: the core resolves a Get output to the Set source,
    values are never copied, caching and partial runs behave like a wire.

Возможности / Features:
    ⚡ Тип выхода Get зеркалит тип источника, цвет сокета нативный
      Get output type mirrors the source type, native socket colors
    ⚡ Пустое имя заполняется само при подключении: model_01, clip_01, bus_01
      An empty name auto-fills on connection: model_01, clip_01, bus_01
    ⚡ Без DOM-оверлеев: мышь, колесо и drag работают поверх ноды везде
      No DOM overlays: mouse, wheel and drag work over the node everywhere
    ⚡ Жёсткая проверка перед запуском: дубли, сироты, пустой вход, mute
      Strict pre-run checks: duplicates, orphans, empty input, mute
    ⚡ Pipe Link видит записи шины через беспроводную пару
      Pipe Link sees bus entries through the wireless pair
    ⚡ Пунктирная воздушная линия по выделению или через меню ноды
      Dashed air link on selection or via the node menu

Замечания / Notes:
    - Входы и линки не пересобираются: меняются только типы слотов и подписи.
      Inputs and links are never rebuilt: only slot types and labels change.
    - Размер ноды считает нативный computeSize:无任何 ручных оверрайдов.
      Node size comes from native computeSize: no manual overrides at all.
    - Автоимя ставится только в пустое поле: ручной ввод не перезаписывается.
      The auto name only fills an empty field: manual input is never overwritten.

Автор / Author: AGSoft
Дата / Date: 26.11.2026
==============================================================================
*/

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft SetGet] JS extension loaded v11.26.7");

const SET_CLASS = "AGSoftSet";
const GET_CLASS = "AGSoftGet";
const LINK_CLASS = "AGSoftPipeLink";
const EMPTY = " ";

let resolving = 0;
let tickTimer = null;
const liveNodes = new Set();
const flagged = new Set();

function safe(fn, label) {
    try {
        return fn();
    } catch (e) {
        console.warn("[AGSoft SetGet] " + label + " skipped:", e);
        return null;
    }
}

function dirty() {
    safe(() => app.graph.setDirtyCanvas(true, true), "dirty");
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name) || null;
}

function nameOf(node) {
    const w = getWidget(node, "name");
    const v = w ? String(w.value ?? "").trim() : "";
    return v === EMPTY ? "" : v;
}

function setWidgetValue(node, w, value) {
    const old = w.value;
    w.value = value;
    if (node.onWidgetChanged) {
        safe(() => node.onWidgetChanged(w.name, value, old, w), "widget notify");
    }
}

function rootOf(graph) {
    return (graph && graph.rootGraph) || graph || null;
}

function allGraphs(graph) {
    const root = rootOf(graph);
    if (!root) return [];
    const seen = new Set([root]);
    const queue = [root];
    const subs = root._subgraphs || root.subgraphs;
    if (subs) {
        for (const sg of subs.values()) {
            if (sg && !seen.has(sg)) {
                seen.add(sg);
                queue.push(sg);
            }
        }
    }
    for (let i = 0; i < queue.length; i++) {
        for (const n of queue[i]._nodes || []) {
            if (n.subgraph && !seen.has(n.subgraph)) {
                seen.add(n.subgraph);
                queue.push(n.subgraph);
            }
        }
    }
    return queue;
}

function ancestorsOf(graph) {
    const chain = [graph];
    const root = rootOf(graph);
    if (!graph || graph === root) return chain;
    const graphs = allGraphs(graph);
    const seen = new Set([graph]);
    let cur = graph;
    while (cur !== root) {
        let parent = null;
        for (const g of graphs) {
            if (g === cur) continue;
            for (const n of g._nodes || []) {
                if (n.subgraph === cur) {
                    parent = g;
                    break;
                }
            }
            if (parent) break;
        }
        if (!parent || seen.has(parent)) break;
        seen.add(parent);
        chain.push(parent);
        cur = parent;
    }
    if (!chain.includes(root)) chain.push(root);
    return chain;
}

function setsIn(graph) {
    return ((graph && graph._nodes) || []).filter((n) => n.comfyClass === SET_CLASS);
}

function findSetter(graph, name) {
    if (!graph || !name) return null;
    for (const g of ancestorsOf(graph)) {
        for (const n of setsIn(g)) {
            if (nameOf(n) === name) return n;
        }
    }
    return null;
}

function findWrapper(root, inner) {
    for (const g of allGraphs(root)) {
        for (const n of g._nodes || []) {
            if (n.subgraph === inner) return n;
        }
    }
    return null;
}

function gettersOf(setNode) {
    const name = nameOf(setNode);
    const sg = setNode.graph;
    if (!name || !sg) return [];
    const out = [];
    for (const g of allGraphs(sg)) {
        if (!ancestorsOf(g).includes(sg)) continue;
        for (const n of g._nodes || []) {
            if (n.comfyClass === GET_CLASS && nameOf(n) === name &&
                findSetter(g, name) === setNode) {
                out.push(n);
            }
        }
    }
    return out;
}

function linkOf(graph, id) {
    if (id == null || !graph) return null;
    if (graph.getLink) return graph.getLink(id);
    const store = graph._links || graph.links;
    if (!store) return null;
    return store instanceof Map ? store.get(id) : store[id] || null;
}

function sourceOf(setNode) {
    const g = setNode.graph;
    const inp = setNode.inputs && setNode.inputs[0];
    const link = inp ? linkOf(g, inp.link) : null;
    if (!link) return null;
    const src = g.getNodeById(link.origin_id);
    if (!src) return null;
    const out = src.outputs && src.outputs[link.origin_slot];
    return { node: src, slot: link.origin_slot, type: (out && out.type) || "*" };
}

function applySetType(node) {
    const src = sourceOf(node);
    const t = src ? src.type : "*";
    const inp = node.inputs && node.inputs[0];
    const out = node.outputs && node.outputs[0];
    if (inp) inp.type = t;
    if (out) {
        out.type = t;
        out.label = nameOf(node) || "value";
    }
}

function applyGetType(node) {
    const setter = findSetter(node.graph, nameOf(node));
    const inp = setter && setter.inputs && setter.inputs[0];
    const out = node.outputs && node.outputs[0];
    if (out) {
        out.type = (inp && inp.type) || "*";
        out.label = nameOf(node) || "value";
    }
}

function comboValuesFor(graph) {
    const names = [];
    const seen = new Set();
    for (const g of ancestorsOf(graph)) {
        for (const n of setsIn(g)) {
            const nm = nameOf(n);
            if (nm && !seen.has(nm)) {
                seen.add(nm);
                names.push(nm);
            }
        }
    }
    names.sort();
    return names;
}

// In-place values refresh plus a fresh options reference: the renderer
// re-reads the list without any widget re-registration or DOM overlay.
// Обновление values на месте плюс новая ссылка options: рендерер перечитывает
// список без перерегистрации виджета и без DOM-оверлея.
function refreshCombo(node) {
    const w = getWidget(node, "name");
    if (!w) return;
    const vals = comboValuesFor(node.graph);
    const cur = String(w.value ?? EMPTY);
    if (cur !== EMPTY && !vals.includes(cur)) vals.unshift(cur);
    if (!vals.length) vals.push(EMPTY);
    const sig = vals.join("|");
    if (w.__agSig === sig) return;
    w.__agSig = sig;
    const opts = w.options || (w.options = {});
    if (Array.isArray(opts.values)) {
        opts.values.length = 0;
        for (const v of vals) opts.values.push(v);
    }
    w.options = { ...opts, values: vals.slice() };
    if (!vals.includes(String(w.value ?? EMPTY))) w.value = vals[0];
}

function refreshAllCombos() {
    for (const g of allGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            if (n.comfyClass === GET_CLASS) safe(() => refreshCombo(n), "combo");
        }
    }
}

function autoName(node) {
    const w = getWidget(node, "name");
    if (!w || nameOf(node)) return;
    const src = sourceOf(node);
    if (!src) return;
    const out = src.node.outputs && src.node.outputs[src.slot];
    const virtual = src.node.comfyClass === SET_CLASS || src.node.comfyClass === GET_CLASS;
    const raw = (!virtual && out && (out.label || out.name)) || src.type || "value";
    let base = String(raw).toLowerCase().replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "").replace(/_\d+$/, "").slice(0, 32);
    if (!base) base = "value";
    const taken = new Set(comboValuesFor(node.graph));
    let n = 1;
    let candidate = base + "_" + String(n).padStart(2, "0");
    while (taken.has(candidate)) {
        n += 1;
        candidate = base + "_" + String(n).padStart(2, "0");
    }
    setWidgetValue(node, w, candidate);
    safe(() => refreshAllCombos(), "combos");
}

function notifyPipeLinks() {
    for (const g of allGraphs(app.graph)) {
        for (const n of g._nodes || []) {
            if (n.comfyClass === LINK_CLASS && n._agsRefresh) {
                safe(() => n._agsRefresh(), "pipe refresh");
            }
        }
    }
}

function nodeSig(node) {
    const nm = nameOf(node);
    if (node.comfyClass === SET_CLASS) {
        const src = sourceOf(node);
        return nm + "|" + (src ? src.type + "|" + src.node.id : "*");
    }
    const setter = findSetter(node.graph, nm);
    const inp = setter && setter.inputs && setter.inputs[0];
    return nm + "|" + ((inp && inp.type) || "*") + "|" + (setter ? setter.id : "-");
}

function refreshNode(node) {
    if (!node.graph) return;
    if (node.comfyClass === GET_CLASS) safe(() => refreshCombo(node), "combo");
    if (node.comfyClass === SET_CLASS && !nameOf(node) && !app.configuringGraph) {
        safe(() => autoName(node), "auto name");
    }
    const sig = nodeSig(node);
    if (node.__agSig !== sig) {
        node.__agSig = sig;
        if (node.comfyClass === SET_CLASS) {
            safe(() => applySetType(node), "set type");
            safe(() => refreshAllCombos(), "combos");
            for (const getter of gettersOf(node)) {
                safe(() => refreshNode(getter), "getter sync");
            }
        } else {
            safe(() => applyGetType(node), "get type");
        }
        dirty();
        safe(() => notifyPipeLinks(), "pipe notify");
    }
}

function scheduleRefresh(node) {
    if (node.__agPending) return;
    node.__agPending = true;
    setTimeout(() => {
        node.__agPending = false;
        safe(() => refreshNode(node), "scheduled");
    }, 0);
}

function markDuplicates() {
    for (const g of allGraphs(app.graph)) {
        const counts = new Map();
        for (const n of setsIn(g)) {
            const nm = nameOf(n);
            if (nm) counts.set(nm, (counts.get(nm) || 0) + 1);
        }
        for (const n of setsIn(g)) {
            const nm = nameOf(n);
            const bad = !!nm && counts.get(nm) > 1;
            if (!!n.has_errors !== bad) {
                n.has_errors = bad;
                dirty();
            }
        }
    }
}

function ensureTick() {
    if (tickTimer) return;
    tickTimer = setInterval(() => {
        safe(() => {
            for (const node of Array.from(liveNodes)) {
                if (!node.graph) {
                    liveNodes.delete(node);
                    continue;
                }
                safe(() => refreshNode(node), "tick");
            }
            markDuplicates();
        }, "tick pass");
    }, 250);
}

function setupNode(node) {
    if (node.__agReady) return;
    node.__agReady = true;
    const isSet = node.comfyClass === SET_CLASS;

    safe(() => {
        const w = getWidget(node, "name");
        if (w && !w.__agHooked) {
            w.__agHooked = true;
            const prev = w.callback;
            w.callback = (v) => {
                if (prev) prev.call(w, v);
                scheduleRefresh(node);
                if (isSet) {
                    safe(() => refreshAllCombos(), "combos");
                    for (const getter of gettersOf(node)) scheduleRefresh(getter);
                }
            };
        }
    }, "widget hook");

    safe(() => {
        const prevConn = node.onConnectionsChange;
        node.onConnectionsChange = function () {
            if (prevConn) prevConn.apply(this, arguments);
            if (app.configuringGraph) return;
            scheduleRefresh(node);
            if (isSet) {
                for (const getter of gettersOf(node)) scheduleRefresh(getter);
            }
        };
    }, "conn hook");

    safe(() => {
        const prevCfg = node.onConfigure;
        node.onConfigure = function () {
            if (prevCfg) prevCfg.apply(this, arguments);
            scheduleRefresh(node);
            setTimeout(() => scheduleRefresh(node), 0);
        };
    }, "configure hook");

    safe(() => {
        const prevAdded = node.onAdded;
        node.onAdded = function () {
            if (prevAdded) prevAdded.apply(this, arguments);
            if (isSet) safe(() => refreshAllCombos(), "combos");
        };
    }, "added hook");

    safe(() => {
        const prevRemoved = node.onRemoved;
        node.onRemoved = function () {
            liveNodes.delete(node);
            if (prevRemoved) prevRemoved.apply(this, arguments);
            if (isSet) safe(() => refreshAllCombos(), "combos");
        };
    }, "removed hook");

    liveNodes.add(node);
    ensureTick();
    safe(() => refreshNode(node), "init");
    setTimeout(() => safe(() => refreshNode(node), "init0"), 0);
    setTimeout(() => safe(() => refreshNode(node), "init150"), 150);
}

function goToPair(node) {
    const target = node.comfyClass === SET_CLASS
        ? gettersOf(node)[0]
        : findSetter(node.graph, nameOf(node));
    if (!target) return;
    const canvas = app.canvas;
    if (target.graph && target.graph !== canvas.graph) {
        const wrapper = findWrapper(canvas.graph, target.graph);
        if (wrapper) {
            canvas.centerOnNode(wrapper);
            canvas.selectNode(wrapper, false);
        }
        return;
    }
    canvas.centerOnNode(target);
    canvas.selectNode(target, false);
    dirty();
}

function drawAirLinks(canvas, ctx) {
    const graph = canvas.graph || app.graph;
    if (!graph || !graph._nodes) return;
    for (const setNode of setsIn(graph)) {
        for (const getter of gettersOf(setNode)) {
            const on = setNode.__agAir || getter.__agAir ||
                setNode.is_selected || getter.is_selected;
            if (!on) continue;
            let target = getter;
            if (getter.graph !== graph) {
                target = findWrapper(graph, getter.graph);
                if (!target) continue;
            }
            const a = setNode.getConnectionPos(false, 0);
            const b = target.getConnectionPos(true, 0);
            const t = (setNode.inputs && setNode.inputs[0] &&
                setNode.inputs[0].type) || "*";
            const colors = canvas.default_connection_color_byType || {};
            const color = colors[t] || "#7fd4ff";
            ctx.save();
            ctx.setLineDash([6, 4]);
            canvas.renderLink(ctx, a, b, null, false, null, color,
                LiteGraph.RIGHT, LiteGraph.LEFT);
            ctx.restore();
        }
    }
}

function validateWireless() {
    for (const n of flagged) n.has_errors = false;
    flagged.clear();
    const problems = [];
    const badNodes = [];
    const graphs = allGraphs(app.graph);
    for (const g of graphs) {
        const counts = new Map();
        for (const n of setsIn(g)) {
            const nm = nameOf(n);
            if (nm) counts.set(nm, (counts.get(nm) || 0) + 1);
        }
        for (const n of setsIn(g)) {
            const nm = nameOf(n);
            if (nm && counts.get(nm) > 1) {
                problems.push(`Duplicate Set name "${nm}" / Дубликат имени Set "${nm}"`);
                badNodes.push(n);
            }
        }
    }
    for (const g of graphs) {
        for (const n of g._nodes || []) {
            if (n.comfyClass !== GET_CLASS || n.mode === 2) continue;
            const out = n.outputs && n.outputs[0];
            if (!out || !out.links || !out.links.length) continue;
            const nm = nameOf(n);
            const setter = nm ? findSetter(g, nm) : null;
            if (!setter) {
                problems.push(`Get "${nm || "?"}" has no Set / у Get нет Set`);
                badNodes.push(n);
            } else if (!sourceOf(setter)) {
                problems.push(`Set "${nm}" has no input / у Set нет входа`);
                badNodes.push(n, setter);
            } else if (setter.mode === 2) {
                problems.push(`Set "${nm}" is muted / Set "${nm}" заглушен`);
                badNodes.push(n, setter);
            }
        }
    }
    if (!problems.length) return;
    for (const n of badNodes) {
        n.has_errors = true;
        flagged.add(n);
    }
    dirty();
    const detail = problems.slice(0, 4).join("; ");
    safe(() => app.extensionManager.toast.add({
        severity: "error",
        summary: "AGSoft Set/Get",
        detail,
        life: 6000,
    }), "toast");
    throw new Error("AGSoft Set/Get: " + detail);
}

app.registerExtension({
    name: "AGSoft.SetGet",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== SET_CLASS && nodeData.name !== GET_CLASS) return;
        nodeType.prototype.isVirtualNode = true;

        if (nodeData.name === SET_CLASS) {
            nodeType.prototype.getInputLink = function () {
                const inp = this.inputs && this.inputs[0];
                return inp ? linkOf(this.graph, inp.link) : null;
            };
        } else {
            nodeType.prototype.getInputLink = function () {
                const setter = findSetter(this.graph, nameOf(this));
                if (!setter || setter.graph !== this.graph) return null;
                const inp = setter.inputs && setter.inputs[0];
                return inp ? linkOf(setter.graph, inp.link) : null;
            };
            nodeType.prototype.resolveVirtualOutput = function () {
                const setter = findSetter(this.graph, nameOf(this));
                if (!setter || setter.graph === this.graph) return undefined;
                const src = sourceOf(setter);
                return src ? { node: src.node, slot: src.slot } : undefined;
            };
            Object.defineProperty(nodeType.prototype, "mode", {
                configurable: true,
                get() {
                    const m = this.__agMode || 0;
                    return resolving > 0 && m === 4 ? 0 : m;
                },
                set(v) {
                    this.__agMode = v;
                },
            });
        }

        const prevMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (prevMenu) prevMenu.apply(this, arguments);
            const node = this;
            options.push({
                content: node.__agAir
                    ? "Hide air link / Скрыть линию"
                    : "Show air link / Показать линию",
                callback: () => {
                    node.__agAir = !node.__agAir;
                    dirty();
                },
            });
            options.push({
                content: "Go to pair / Перейти к паре",
                callback: () => goToPair(node),
            });
        };
    },

    async nodeCreated(node) {
        if (node.comfyClass !== SET_CLASS && node.comfyClass !== GET_CLASS) return;
        safe(() => setupNode(node), "setup");
    },

    setup() {
        const prevG2P = app.graphToPrompt;
        app.graphToPrompt = async function () {
            resolving += 1;
            try {
                return await prevG2P.apply(this, arguments);
            } finally {
                resolving -= 1;
            }
        };

        const prevQueue = api.queuePrompt;
        api.queuePrompt = async function () {
            validateWireless();
            return prevQueue.apply(this, arguments);
        };

        const prevDraw = app.canvas.onDrawBackground;
        app.canvas.onDrawBackground = function (ctx, area) {
            if (prevDraw) prevDraw.call(this, ctx, area);
            safe(() => drawAirLinks(this, ctx), "air draw");
        };

        const lgProto = window.LGraphCanvas && window.LGraphCanvas.prototype;
        if (lgProto) {
            const prevVisible = lgProto.computeVisibleNodes;
            lgProto.computeVisibleNodes = function () {
                const visible = prevVisible.apply(this, arguments);
                for (const n of (this.graph && this.graph._nodes) || []) {
                    if ((n.comfyClass === SET_CLASS || n.comfyClass === GET_CLASS) &&
                        (n.__agAir || n.is_selected) && !visible.includes(n)) {
                        visible.push(n);
                    }
                }
                return visible;
            };
        }

        document.addEventListener("dblclick", () => {
            if (document.querySelector(".litecontextmenu")) return;
            const sel = Object.values(app.canvas.selected_nodes || {});
            if (sel.length !== 1) return;
            if (sel[0].comfyClass === GET_CLASS) safe(() => goToPair(sel[0]), "jump");
        });
    },
});