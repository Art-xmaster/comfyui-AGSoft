// ==============================================================================
// AGSoft_Load_Image_Mask_Resize.js
// ==============================================================================
// JS-расширение для ноды 🖼️AGSoft Load Image & Mask Resize.
//
// НЕ рисует своё превью (иначе дубль со встроенным). Переключает САМ комбо
// image + node.imgs, чтобы нативное превью показывало реально использованный
// источник:
//
// 1) подключён input_image (тензор) — после выполнения превью показывает его
//    первый кадр (сервер сохранил превью-PNG);
// 2) custom_path вводится руками — живой апдейт:
//    POST /agsoft/image_mask_resize_ensure_preview → имя в input → комбо;
// 3) custom_path подключён ЛИНКОМ — значение видно только на сервере во время
//    выполнения, поэтому апдейт по событию executed + БЫСТРЫЙ поллинг
//    (300 мс, GET /agsoft/image_mask_resize_preview_state, защита штампом);
// 4) источник — комбо — восстанавливаем прежнее значение комбо.
//
// JS extension for 🖼️AGSoft Load Image & Mask Resize node.
//
// It does NOT draw its own preview (that would duplicate the built-in one).
// It switches the image combo ITSELF + node.imgs so the native preview shows
// the actually used source:
//
// 1) input_image tensor linked — after execution the preview shows its first
//    frame (server saved a preview PNG);
// 2) typed custom_path — live update via POST ensure_preview;
// 3) LINKED custom_path — value only exists server-side at execution, so we
//    update on the executed event + FAST polling (300 ms, GET preview_state,
//    stamp-guarded);
// 4) combo source — restore the previous combo value.
// ==============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[AGSoft Load Image & Mask Resize] JS v2.2 loaded (fast preview: 300ms poll + stamp)");

const ENSURE_ROUTE = "/agsoft/image_mask_resize_ensure_preview";
const STATE_ROUTE = "/agsoft/image_mask_resize_preview_state";
const PREFIX = "__agsoft_preview__";
const POLL_MS = 300;
const CLASSES = ["AGSoft_Load_Image_Mask_Resize", "AGSoft_Image_Resize_Plus"];

const isOurNode = (node) => node && CLASSES.includes(node.comfyClass);

// ------------------------------------------------------------------------------
// Гарантируем, что значение есть в списке комбо.
// Make sure the value exists in the combo options list.
// ------------------------------------------------------------------------------
function ensureComboOption(combo, name) {
    combo.options = combo.options || {};

    if (!Array.isArray(combo.options.values)) {
        combo.options.values = [];
    }

    if (!combo.options.values.includes(name)) {
        combo.options.values.push(name);
    }

    if (Array.isArray(combo.values) && !combo.values.includes(name)) {
        combo.values.push(name);
    }
}

// ------------------------------------------------------------------------------
// Страховка: грузим картинку прямо в node.imgs (canvas-превью рисует node.imgs).
// Safety net: load the picture straight into node.imgs (canvas preview draws it).
// ------------------------------------------------------------------------------
function loadPreviewIntoNode(node, name) {
    try {
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            node.setDirtyCanvas?.(true, true);
        };
        img.src = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input`);
    } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------------------
// Переключение комбо image + запуск нативного превью.
// Switch the image combo + trigger the native preview.
// ------------------------------------------------------------------------------
function applyComboValue(node, combo, name) {
    ensureComboOption(combo, name);

    combo.value = name;

    // Штатный callback комбо (фронтенд может сам обновить превью).
    // Standard combo callback (the frontend may refresh the preview itself).
    if (typeof combo.callback === "function") {
        combo.callback(name);
    }

    loadPreviewIntoNode(node, name);

    node.setDirtyCanvas?.(true, true);
    if (app.graph && app.graph.setDirtyCanvas) {
        app.graph.setDirtyCanvas(true, true);
    }
}

// ------------------------------------------------------------------------------
// Применение серверного состояния. Благодаря stamp состояние применяется
// ровно ОДИН раз на каждое выполнение ноды — поллинг безопасен и не
// перезаписывает ручной выбор комбо между выполнениями.
// Apply the server-side state. Thanks to stamp the state is applied exactly
// ONCE per node execution — polling is safe and does not overwrite a manual
// combo selection between executions.
// ------------------------------------------------------------------------------
async function applyState(node) {
    const combo = node.widgets?.find(w => w.name === "image");
    if (!combo) return;

    try {
        const r = await fetch(
            api.apiURL(`${STATE_ROUTE}?node_id=${encodeURIComponent(node.id)}`)
        );
        if (!r.ok) return;

        const st = await r.json();
        const stamp = st.stamp || 0;

        // Ничего нового с прошлого применения — не трогаем.
        // Nothing new since the last apply — leave untouched.
        if (stamp && node.__agsoft_last_stamp === stamp) return;
        node.__agsoft_last_stamp = stamp;

        const current = String(combo.value || "");

        node.properties = node.properties || {};

        if (st.custom && st.image) {
            // Запоминаем исходное значение комбо ОДИН раз (до первой подмены).
            // Remember the original combo value ONCE (before first override).
            if (
                node.properties.__agsoft_orig_image === undefined &&
                !current.startsWith(PREFIX)
            ) {
                node.properties.__agsoft_orig_image = current;
            }

            if (current !== st.image) {
                applyComboValue(node, combo, st.image);
            } else {
                loadPreviewIntoNode(node, st.image);
            }
        } else {
            // Источник — комбо: возвращаем прежнее значение, если подменяли.
            // Combo source: restore the previous value if we overrode it.
            const orig = node.properties.__agsoft_orig_image;

            if (orig !== undefined) {
                delete node.properties.__agsoft_orig_image;

                if (current.startsWith(PREFIX)) {
                    applyComboValue(node, combo, orig);
                }
            }
        }
    } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------------------
// Живой апдейт при вводе пути руками (до очереди).
// Live update while typing a path (before queue).
// ------------------------------------------------------------------------------
async function liveSync(node) {
    const custom = node.widgets?.find(w => w.name === "custom_path");
    const combo = node.widgets?.find(w => w.name === "image");
    if (!custom || !combo) return;

    const p = String(custom.value || "").trim();
    if (!p) return;

    try {
        const resp = await fetch(api.apiURL(ENSURE_ROUTE), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ custom_path: p, node_id: node.id }),
        });

        if (!resp.ok) return;

        const data = await resp.json();

        if (data.image && String(combo.value || "") !== data.image) {
            node.properties = node.properties || {};
            const current = String(combo.value || "");
            if (
                node.properties.__agsoft_orig_image === undefined &&
                !current.startsWith(PREFIX)
            ) {
                node.properties.__agsoft_orig_image = current;
            }
            applyComboValue(node, combo, data.image);
        }
    } catch (e) { /* ignore */ }
}

function scheduleLiveSync(node, ms = 300) {
    if (node.__agsoft_live_timer) {
        clearTimeout(node.__agsoft_live_timer);
    }
    node.__agsoft_live_timer = setTimeout(() => liveSync(node), ms);
}

// ------------------------------------------------------------------------------
// Слушатель executed — вешается ОДИН раз на всё приложение.
// Executed listener — bound ONCE for the whole app.
// ------------------------------------------------------------------------------
let executedBound = false;

function bindExecuted() {
    if (executedBound) return;
    executedBound = true;

    api.addEventListener("executed", (e) => {
        try {
            const id = e.detail?.display_node || e.detail?.node;
            if (id === undefined || id === null) return;

            const node = app.graph && app.graph.getNodeById
                ? app.graph.getNodeById(id)
                : null;

            if (!isOurNode(node)) return;

            applyState(node);
        } catch (err) { /* ignore */ }
    });
}

app.registerExtension({
    name: "AGSoft.LoadImageMaskResize",

    setup() {
        bindExecuted();
    },

    async nodeCreated(node) {
        if (!isOurNode(node)) return;

        bindExecuted();

        // Живой апдейт при вводе custom_path руками.
        // Live update when custom_path is typed manually.
        const custom = node.widgets?.find(w => w.name === "custom_path");

        if (custom) {
            const oldCb = custom.callback;

            custom.callback = (v) => {
                if (oldCb) oldCb(v);
                scheduleLiveSync(node);
            };
        }

        // Восстановление/применение состояния после загрузки воркфлоу.
        // Restore/apply state after loading a workflow.
        const origConfigure = node.onConfigure;

        node.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            setTimeout(() => applyState(this), 50);
        };

        // Быстрый поллинг-страховка: если событие executed потерялось (кэш,
        // реконнект и т.п.), состояние всё равно применится почти мгновенно.
        // Благодаря stamp применение происходит ровно один раз на выполнение.
        // Fast polling safety net: if the executed event got lost (cache,
        // reconnect etc.), the state still gets applied almost instantly.
        // Thanks to stamp it is applied exactly once per execution.
        if (!node.__agsoft_poll) {
            node.__agsoft_poll = setInterval(() => applyState(node), POLL_MS);
        }

        // Чистим поллинг при удалении ноды.
        // Clean up polling when the node is removed.
        const origOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            if (this.__agsoft_poll) {
                clearInterval(this.__agsoft_poll);
                this.__agsoft_poll = null;
            }
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
        };

        setTimeout(() => applyState(node), 50);
    }
});