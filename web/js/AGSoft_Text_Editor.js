// ============================================================================
// AGSoft_Text_Editor.js
// ============================================================================
// Расширение: 📝 AGSoft Text Editor
// Описание / Description:
// Клиентская часть редактора: три буфера с подсветкой и нумерацией строк,
// живой статусбар, поиск и замена, буфер обмена, веса слов, транслит и перевод,
// эмодзи, темы, Zen на весь экран, автосохранение и работа с файлами.
// ---
// Editor frontend: three buffers with highlighting and line numbers, a live
// status bar, search and replace, clipboard tools, word weights, translit and
// translation, emoji, themes, fullscreen Zen, autosave and file handling.
//
// Возможности / Features:
// ⚡ Три буфера: Text, Negative и Result, у каждого своя панель и история отмен
//   Three buffers: Text, Negative and Result, each with its own pane and undo
// ⚡ Split сравнивает Text и Negative либо оригинал и результат перевода
//   Split compares Text with Negative, or the original with a result
// ⚡ Новый документ очищает буферы и путь, текст возвращается по Ctrl+Z
//   A new document clears the buffers and the path, Ctrl+Z brings text back
// ⚡ Подсветка (word:1.2), <lora:..>, __wildcard__, {a|b}, BREAK и заголовков
//   Highlighting of (word:1.2), <lora:..>, __wildcard__, {a|b}, BREAK, headings
// ⚡ Контроль баланса скобок и весов выше 2.0 с переходом к проблеме
//   Bracket balance and weights above 2.0 are checked, click jumps to the issue
// ⚡ Поиск и замена: regex, регистр, слово целиком, замена всего одним шагом
//   Search and replace: regex, case, whole word, replace all as one undo step
// ⚡ Умное копирование, вырезание строки, вставка как строка, с trim, как теги
//   Smart copy, line cut, paste as one line, trimmed, or as a tag list
// ⚡ Веса слов Ctrl+Up и Ctrl+Down с шагом 0.1 в диапазоне от 0 до 2
//   Word weights on Ctrl+Up and Ctrl+Down, step 0.1, clamped between 0 and 2
// ⚡ Транслит RU и EN работает без сети, перевод EN, RU и ZH идёт из браузера
//   RU and EN translit works offline, EN, RU and ZH translation runs in browser
// ⚡ Неудачный перевод не подменяет текст, а сообщает причину в консоль и в UI
//   A failed translation never replaces text and reports the reason to console
// ⚡ Результат можно положить рядом с оригиналом и применить кнопкой ↤
//   A result can sit beside the original and be applied with the ↤ button
// ⚡ Эмодзи из отдельного модуля с поиском по русским и английским словам
//   Emoji come from a separate module, searchable in Russian and English
// ⚡ Десять тем управляют и панелями редактора, и цветом самой ноды
//   Ten themes drive both the editor panels and the node colour
// ⚡ Статусбар: строка и столбец, выделение, предупреждения, токены, счётчики
//   Status bar: line and column, selection, warnings, tokens, counters
// ⚡ Превью Markdown и сворачиваемое дерево JSON в отдельной вкладке
//   Markdown preview and a collapsible JSON tree on a separate tab
// ⚡ Автосохранение в браузере с диалогом восстановления и удаления копии
//   Browser autosave with a dialog to restore or delete the stored copy
// ⚡ Последние десять файлов с полными путями и загрузкой в любой буфер
//   Last ten files with full paths, loadable into any buffer
// ⚡ Приём документов, импорт промпта из PNG, выбор текстов из workflow JSON
//   Document intake, PNG prompt import, text picking from a workflow JSON
// ⚡ Запись и экспорт в txt, md и json, в папку ComfyUI или через диалог браузера
//   Write and export to txt, md and json, into a ComfyUI or browser-chosen folder
// ⚡ Drag and drop на панель и на канву с подсветкой и защитой от двойного drop
//   Drag and drop onto the panel and the canvas with a hint and a double guard
// ⚡ Узкая нода прячет лишние кнопки в панель ⋯ вместо переноса строки
//   A narrow node moves extra buttons into the ⋯ panel instead of wrapping
//
// Горячие клавиши / Shortcuts:
// Ctrl+Alt+N  новый документ / new document
// Ctrl+S      сохранить или экспортировать / save or export
// Ctrl+O      открыть по абсолютному пути / open by absolute path
// Ctrl+F      поиск / search
// Ctrl+H      замена / replace
// Ctrl+Z      отмена / undo
// Ctrl+Y      повтор / redo
// Ctrl+Up     вес слова +0.1 / word weight +0.1
// Ctrl+Down   вес слова -0.1 / word weight -0.1
// Tab         табуляция в позиции курсора / tab at the caret
// Esc         закрыть поиск, панель или Zen / close search, panel or Zen
//
// Хранение / Storage:
// Буферы пакуются в виджет text маркерами [AGSoft Negative], [AGSoft Result]
// и [AGSoft Filepath]; настройки интерфейса лежат в node.properties,
// автосохранение — в localStorage браузера по идентификатору ноды.
// ---
// Buffers are packed into the text widget with the [AGSoft Negative],
// [AGSoft Result] and [AGSoft Filepath] markers; interface settings live in
// node.properties, autosave lives in browser localStorage keyed by node id.
//
// Лимиты / Limits:
// Отмены 200 на буфер, совпадения поиска 4000, подсветка до 20 000 символов
// Undo 200 per buffer, 4000 search matches, highlighting up to 20,000 chars
// Перевод до 4500 символов и 12 с, автосохранение через 900 мс
// Translation up to 4500 chars and 12 s, autosave after 900 ms
// Шрифт от 9 до 28, недавние файлы 10, защита от двойного drop 600 мс
// Font from 9 to 28, 10 recent files, 600 ms double drop guard
//
// Зависимости / Dependencies:
// scripts/app.js, scripts/api.js, ./AGSoft_Text_Editor_emoji.js
//
// Автор / Author: AGSoft
// Дата / Date: 14.11.2026
// ============================================================================

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { mountEmojiPicker } from "./AGSoft_Text_Editor_emoji.js";

console.log("[AGSoft Text Editor] JS extension loaded v11.14 (new document, result split)");

const STYLE_ID = "ags-te-style";
const MIN_EDITOR_H = 240;
const FONT_MIN = 9;
const FONT_MAX = 28;
const LINE_H = 1.5;
const PAD_TOP = 8;
const HL_LIMIT = 20000;
const UNDO_LIMIT = 200;
const MATCH_LIMIT = 4000;
const DROP_GUARD_MS = 600;
const WEIGHT_STEP = 0.1;
const WEIGHT_MIN = 0.0;
const WEIGHT_MAX = 2.0;
const TRANSLATE_LIMIT = 4500;
const TRANSLATE_TIMEOUT = 12000;
const AUTOSAVE_DELAY = 900;
const AUTOSAVE_PREFIX = "agsoft_te_autosave_";
const RECENT_KEY = "agsoft_te_recent";
const LAST_FOLDER_KEY = "agsoft_te_last_folder";
const RECENT_MAX = 10;
const RECENT_TEXT_MAX = 60000;
const NEW_DOCUMENT_LABEL = "New document";
const NEG_MARK = "\n[AGSoft Negative]\n";
const RESULT_MARK = "\n[AGSoft Result]\n";
const PATH_MARK = "\n[AGSoft Filepath]\n";
const INSTANCES = new Set();

const ROOT_GROUP_LABELS = {
    comfy: "ComfyUI",
    places: "This computer / Этот компьютер",
    custom: "Extra save folders / Дополнительные папки записи",
};

const LANGS = {
    en: { label: "English", code: "en" },
    ru: { label: "Русский", code: "ru" },
    zh: { label: "中文", code: "zh-CN" },
};

const THEMES = {
    ink: {
        label: "Ink", bg: "#353a5e", surface: "#3f4570", ink: "#2b2f4d",
        text: "#e7eaf6", muted: "#9aa3c0", accent: "#6cb8ff", amber: "#ffc857",
        ok: "#57d9a3", danger: "#ff5d73", violet: "#c792ea",
        nodeTitle: "#3f4570", nodeBody: "#353a5e",
    },
    light: {
        label: "Light", bg: "#f4f6fb", surface: "#e7ebf4", ink: "#dde2ee",
        text: "#1d2333", muted: "#5b6480", accent: "#1f6feb", amber: "#b26a00",
        ok: "#1a7f4b", danger: "#c62828", violet: "#7c3aed",
        nodeTitle: "#3d4766", nodeBody: "#5d6786",
    },
    sepia: {
        label: "Sepia", bg: "#2b2419", surface: "#3a3122", ink: "#221c13",
        text: "#f0e4cd", muted: "#b39f7d", accent: "#e0a458", amber: "#ffd28a",
        ok: "#9fcf7a", danger: "#e2705a", violet: "#d3a0c9",
        nodeTitle: "#3a3122", nodeBody: "#2b2419",
    },
    matrix: {
        label: "Matrix", bg: "#030903", surface: "#071507", ink: "#010401",
        text: "#46ff7a", muted: "#1f9a4a", accent: "#00ff66", amber: "#baff39",
        ok: "#00ff66", danger: "#ff3b3b", violet: "#7dffb0",
        nodeTitle: "#061206", nodeBody: "#020602",
    },
    amber: {
        label: "Amber", bg: "#0a0703", surface: "#171006", ink: "#050301",
        text: "#ffb347", muted: "#a06a1f", accent: "#ff9f1c", amber: "#ffd166",
        ok: "#9ef01a", danger: "#ff4d4d", violet: "#ff7ab6",
        nodeTitle: "#171006", nodeBody: "#0a0703",
    },
    dracula: {
        label: "Dracula", bg: "#282a36", surface: "#343746", ink: "#1e1f29",
        text: "#f8f8f2", muted: "#6272a4", accent: "#bd93f9", amber: "#f1fa8c",
        ok: "#50fa7b", danger: "#ff5555", violet: "#ff79c6",
        nodeTitle: "#343746", nodeBody: "#282a36",
    },
    nord: {
        label: "Nord", bg: "#2e3440", surface: "#3b4252", ink: "#242933",
        text: "#eceff4", muted: "#90a0b8", accent: "#88c0d0", amber: "#ebcb8b",
        ok: "#a3be8c", danger: "#bf616a", violet: "#b48ead",
        nodeTitle: "#3b4252", nodeBody: "#2e3440",
    },
    monokai: {
        label: "Monokai", bg: "#272822", surface: "#32332c", ink: "#1e1f1a",
        text: "#f8f8f2", muted: "#90917e", accent: "#a6e22e", amber: "#e6db74",
        ok: "#a6e22e", danger: "#f92672", violet: "#ae81ff",
        nodeTitle: "#32332c", nodeBody: "#272822",
    },
    cyber: {
        label: "Cyber", bg: "#0d0221", surface: "#1a0b3b", ink: "#070113",
        text: "#eaf6ff", muted: "#8f7bd8", accent: "#00e5ff", amber: "#ffe600",
        ok: "#39ff88", danger: "#ff2a6d", violet: "#ff29d2",
        nodeTitle: "#1a0b3b", nodeBody: "#0d0221",
    },
    ocean: {
        label: "Ocean", bg: "#0b2530", surface: "#123543", ink: "#071a22",
        text: "#d8f3f7", muted: "#7fa8b5", accent: "#2ec4b6", amber: "#ffbf69",
        ok: "#57d9a3", danger: "#ff6b6b", violet: "#9b8cff",
        nodeTitle: "#123543", nodeBody: "#0b2530",
    },
};

const RU_EN = {
    а: "a", б: "b", в: "v", г: "g", д: "d", е: "e", ё: "yo", ж: "zh", з: "z",
    и: "i", й: "y", к: "k", л: "l", м: "m", н: "n", о: "o", п: "p", р: "r",
    с: "s", т: "t", у: "u", ф: "f", х: "kh", ц: "ts", ч: "ch", ш: "sh",
    щ: "shch", ъ: "", ы: "y", ь: "", э: "e", ю: "yu", я: "ya",
};
const EN_RU_MULTI = [
    ["shch", "щ"], ["zh", "ж"], ["kh", "х"], ["ts", "ц"], ["ch", "ч"],
    ["sh", "ш"], ["yu", "ю"], ["ya", "я"], ["yo", "ё"], ["iy", "ий"], ["ey", "ей"],
];
const EN_RU_SINGLE = {
    a: "а", b: "б", c: "к", d: "д", e: "е", f: "ф", g: "г", h: "х", i: "и",
    j: "дж", k: "к", l: "л", m: "м", n: "н", o: "о", p: "п", q: "к", r: "р",
    s: "с", t: "т", u: "у", v: "в", w: "в", x: "кс", y: "й", z: "з",
};

const SYN_RE = /(<lora:[^>\n]*>)|(__[\w./\\-]+__)|(\{[^{}\n]*\|[^{}\n]*\})|(\([^()\n]*:[+-]?\d*\.?\d+\))|(\bBREAK\b)|(^[ \t]*#{1,6}[ \t].*$)|(^[ \t]*#[^\n]*$)|("(?:[^"\\\n]|\\.)*")|(\b\d+(?:\.\d+)?\b)/gm;
const SYN_CLASSES = ["", "ags-s-l", "ags-s-v", "ags-s-d", "ags-s-w", "ags-s-k",
    "ags-s-h", "ags-s-c", "ags-s-s", "ags-s-n"];

const CSS = `
.ags-te {
    --ags-bg: #353a5e;
    --ags-surface: #3f4570;
    --ags-ink: #2b2f4d;
    --ags-text: #e7eaf6;
    --ags-muted: #9aa3c0;
    --ags-accent: #6cb8ff;
    --ags-amber: #ffc857;
    --ags-ok: #57d9a3;
    --ags-danger: #ff5d73;
    --ags-violet: #c792ea;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    position: relative;
    border: 1px solid #4a5182;
    border: 1px solid color-mix(in srgb, var(--ags-accent) 22%, var(--ags-surface));
    border-radius: 6px;
    background: var(--ags-bg);
    color: var(--ags-text);
    font-family: "Segoe UI", system-ui, sans-serif;
}
.ags-te * { box-sizing: border-box; }
.ags-te.ags-zen {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-width: none !important;
    max-height: none !important;
    transform: none !important;
    z-index: 99999 !important;
    display: flex !important;
    border-radius: 0 !important;
    border: 0 !important;
}
body.ags-zen-open { overflow: hidden !important; }
.ags-te-bar {
    flex: 0 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 3px;
    padding: 3px 5px;
    background: var(--ags-surface);
    border-bottom: 1px solid color-mix(in srgb, var(--ags-accent) 14%, var(--ags-surface));
}
.ags-te-toolbar { flex-wrap: nowrap; overflow: hidden; justify-content: flex-start; }
.ags-te-group { display: flex; align-items: center; gap: 3px; flex: 0 0 auto; }
.ags-te-group.ags-in-overflow { flex-wrap: wrap; max-width: 260px; padding: 2px; }
.ags-te-sep {
    width: 1px;
    height: 15px;
    background: color-mix(in srgb, var(--ags-muted) 40%, transparent);
}
.ags-te-btn {
    appearance: none;
    border: 1px solid transparent;
    border-radius: 5px;
    background: transparent;
    color: var(--ags-muted);
    font: 600 11px "Segoe UI", system-ui, sans-serif;
    line-height: 1.4;
    padding: 1px 6px;
    cursor: pointer;
    transition: background 120ms, color 120ms, border-color 120ms;
}
.ags-te-btn:hover {
    color: var(--ags-text);
    background: color-mix(in srgb, var(--ags-accent) 14%, transparent);
}
.ags-te-btn:disabled { opacity: 0.35; cursor: default; }
.ags-te-btn.ags-on {
    color: var(--ags-accent);
    border-color: color-mix(in srgb, var(--ags-accent) 45%, transparent);
    background: color-mix(in srgb, var(--ags-accent) 10%, transparent);
}
.ags-te-btn.ags-ok { color: var(--ags-ok); }
.ags-te-btn.ags-danger:hover {
    color: var(--ags-danger);
    background: color-mix(in srgb, var(--ags-danger) 12%, transparent);
}
.ags-te-select {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--ags-muted) 35%, transparent);
    border-radius: 5px;
    background: var(--ags-ink);
    color: var(--ags-text);
    font: 600 10px "Segoe UI", system-ui, sans-serif;
    padding: 2px 4px;
    cursor: pointer;
    outline: none;
}
.ags-te-select optgroup {
    background: var(--ags-ink);
    color: var(--ags-muted);
    font-style: normal;
}
.ags-te-select option { background: var(--ags-ink); color: var(--ags-text); }
.ags-te-badge {
    margin-left: 8px;
    font-size: 10px;
    color: var(--ags-amber);
    white-space: nowrap;
}
.ags-te-filetag {
    margin-left: auto;
    max-width: 40%;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px;
    color: var(--ags-text);
    background: color-mix(in srgb, var(--ags-accent) 12%, transparent);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: pointer;
}
.ags-te-filetag.ags-new {
    color: var(--ags-muted);
    font-style: italic;
    background: transparent;
}
.ags-te-filetag:hover { background: color-mix(in srgb, var(--ags-accent) 24%, transparent); }
.ags-te-zenbadge { font-size: 10px; color: var(--ags-accent); white-space: nowrap; display: none; }
.ags-te.ags-zen .ags-te-zenbadge { display: inline; }
.ags-te-tabs { gap: 3px; padding: 2px 5px; flex-wrap: nowrap; }
.ags-te-tab { border-radius: 5px 5px 0 0; padding: 2px 9px; font-size: 10px; }
.ags-te-tab.ags-on {
    color: var(--ags-text);
    background: color-mix(in srgb, var(--ags-accent) 18%, transparent);
    border-color: color-mix(in srgb, var(--ags-accent) 40%, transparent);
}
.ags-te-splitctl { display: none; align-items: center; gap: 3px; flex: 0 0 auto; }
.ags-te.ags-split .ags-te-splitctl { display: flex; }
.ags-te-splitctl .ags-te-select { max-width: 168px; }
.ags-te-search { display: none; }
.ags-te-search.ags-open { display: flex; }
.ags-te-input {
    flex: 1 1 120px;
    min-width: 70px;
    border: 1px solid color-mix(in srgb, var(--ags-muted) 35%, transparent);
    border-radius: 5px;
    background: var(--ags-ink);
    color: var(--ags-text);
    font: 12px Consolas, "Cascadia Mono", Menlo, monospace;
    padding: 2px 6px;
    outline: none;
}
.ags-te-input:focus { border-color: var(--ags-accent); }
.ags-te-count { font-size: 10px; color: var(--ags-muted); min-width: 40px; text-align: center; }
.ags-te-count.ags-bad { color: var(--ags-danger); }
.ags-te-editor {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: row;
    position: relative;
    background: var(--ags-bg);
}
.ags-te-pane {
    flex: 1 1 50%;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.ags-te-pane + .ags-te-pane {
    border-left: 1px solid color-mix(in srgb, var(--ags-accent) 22%, var(--ags-surface));
}
.ags-te-pane-head {
    display: none;
    flex: 0 0 auto;
    padding: 2px 8px;
    font-size: 9px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: var(--ags-muted);
    background: color-mix(in srgb, var(--ags-surface) 60%, var(--ags-bg));
    cursor: pointer;
    user-select: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ags-te.ags-split .ags-te-pane-head { display: block; }
.ags-te-pane.ags-active .ags-te-pane-head {
    color: var(--ags-accent);
    background: color-mix(in srgb, var(--ags-accent) 12%, var(--ags-bg));
}
.ags-te-pane-body { flex: 1 1 auto; min-height: 0; display: flex; position: relative; }
.ags-te-gutter {
    flex: 0 0 auto;
    position: relative;
    overflow: hidden;
    text-align: right;
    color: var(--ags-muted);
    background: color-mix(in srgb, var(--ags-ink) 55%, var(--ags-bg));
    border-right: 1px solid color-mix(in srgb, var(--ags-accent) 10%, var(--ags-surface));
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
    user-select: none;
    pointer-events: none;
}
.ags-te-gtext { position: relative; padding: 8px 6px 8px 8px; white-space: pre; }
.ags-te-textwrap {
    flex: 1 1 auto;
    min-width: 0;
    position: relative;
    overflow: hidden;
    background: var(--ags-bg);
}
.ags-te-linehl {
    position: absolute;
    left: 0;
    right: 0;
    display: none;
    pointer-events: none;
    background: color-mix(in srgb, var(--ags-accent) 8%, transparent);
}
.ags-te-hl {
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    margin: 0;
    border: 0;
    padding: 8px 10px;
    overflow: hidden;
    pointer-events: none;
    white-space: pre;
    color: var(--ags-text);
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
    font-size: 13px;
    line-height: 20px;
}
.ags-te-text {
    position: relative;
    z-index: 2;
    width: 100%;
    height: 100%;
    padding: 8px 10px;
    border: 0;
    outline: none;
    resize: none;
    background: transparent;
    color: var(--ags-text);
    caret-color: var(--ags-accent);
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
    font-size: 13px;
    line-height: 20px;
    white-space: pre;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink)) var(--ags-ink);
}
.ags-te-text.ags-ghost { color: transparent; }
.ags-te-text::selection { background: color-mix(in srgb, var(--ags-accent) 34%, transparent); }
.ags-te-text.ags-ghost::selection { color: transparent; }
.ags-te-text::placeholder { color: var(--ags-muted); opacity: 0.7; }
.ags-te ::-webkit-scrollbar { width: 10px; height: 10px; }
.ags-te ::-webkit-scrollbar-track { background: var(--ags-ink); }
.ags-te ::-webkit-scrollbar-thumb {
    background: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink));
    border-radius: 5px;
    border: 2px solid var(--ags-ink);
}
.ags-te ::-webkit-scrollbar-thumb:hover { background: var(--ags-accent); }
.ags-te ::-webkit-scrollbar-corner { background: var(--ags-ink); }
.ags-s-l { color: var(--ags-ok); }
.ags-s-v { color: var(--ags-violet); }
.ags-s-d { color: var(--ags-accent); }
.ags-s-w { color: var(--ags-amber); }
.ags-s-k { color: var(--ags-danger); font-weight: 700; }
.ags-s-h { color: var(--ags-accent); font-weight: 700; }
.ags-s-c { color: var(--ags-muted); font-style: italic; }
.ags-s-s { color: var(--ags-ok); }
.ags-s-n { color: var(--ags-amber); }
.ags-m { background: color-mix(in srgb, var(--ags-amber) 38%, transparent); border-radius: 2px; }
.ags-mc { background: color-mix(in srgb, var(--ags-accent) 62%, transparent); border-radius: 2px; }
.ags-te-preview {
    display: none;
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    padding: 10px 12px;
    background: var(--ags-bg);
    color: var(--ags-text);
    font-size: 12px;
    line-height: 1.55;
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink)) var(--ags-ink);
}
.ags-te-preview h1, .ags-te-preview h2, .ags-te-preview h3 { color: var(--ags-accent); margin: 8px 0 4px; }
.ags-te-preview h4, .ags-te-preview h5, .ags-te-preview h6 { color: var(--ags-muted); margin: 6px 0 3px; }
.ags-te-preview p { margin: 5px 0; }
.ags-te-preview ul, .ags-te-preview ol { margin: 5px 0; padding-left: 20px; }
.ags-te-preview hr { border: 0; border-top: 1px solid var(--ags-muted); opacity: 0.4; }
.ags-te-preview blockquote {
    margin: 5px 0;
    padding: 2px 8px;
    border-left: 3px solid var(--ags-accent);
    color: var(--ags-muted);
}
.ags-te-preview code {
    background: var(--ags-ink);
    border-radius: 3px;
    padding: 0 4px;
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
}
.ags-md-code {
    background: var(--ags-ink);
    border-radius: 5px;
    padding: 8px;
    overflow: auto;
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
    font-size: 11px;
}
.ags-md-link { color: var(--ags-accent); text-decoration: underline dotted; }
.ags-jt-row {
    padding-left: 10px;
    font-family: Consolas, "Cascadia Mono", Menlo, monospace;
    font-size: 11px;
}
.ags-jt-row summary { cursor: pointer; color: var(--ags-accent); }
.ags-jt-key { color: var(--ags-muted); }
.ags-jt-string { color: var(--ags-ok); }
.ags-jt-number { color: var(--ags-amber); }
.ags-jt-boolean { color: var(--ags-violet); }
.ags-jt-null { color: var(--ags-danger); }
.ags-jt-more { color: var(--ags-muted); padding-left: 12px; }
.ags-te-status {
    flex: 0 0 20px;
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 8px;
    overflow: hidden;
    background: var(--ags-surface);
    border-top: 1px solid color-mix(in srgb, var(--ags-accent) 16%, var(--ags-surface));
    font-size: 10px;
    color: var(--ags-muted);
    white-space: nowrap;
    user-select: none;
}
.ags-te-status .ags-seg { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.ags-te-status .ags-click { cursor: pointer; }
.ags-te-status .ags-click:hover { color: var(--ags-text); text-decoration: underline; }
.ags-te-status .ags-cur { flex: 0 0 auto; color: var(--ags-accent); font-weight: 600; }
.ags-te-status .ags-sel { flex: 0 0 auto; color: var(--ags-amber); }
.ags-te-status .ags-warn { flex: 0 0 auto; color: var(--ags-danger); font-weight: 600; }
.ags-te-status .ags-auto { flex: 0 0 auto; color: var(--ags-ok); }
.ags-te-status .ags-right { margin-left: auto; display: flex; gap: 7px; min-width: 0; overflow: hidden; }
.ags-te-menu, .ags-te-overflow {
    display: none;
    position: absolute;
    right: 6px;
    top: 30px;
    z-index: 20;
    min-width: 210px;
    max-height: 320px;
    overflow: auto;
    padding: 4px;
    border: 1px solid color-mix(in srgb, var(--ags-accent) 35%, var(--ags-surface));
    border-radius: 6px;
    background: var(--ags-surface);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink)) var(--ags-ink);
}
.ags-te-menu.ags-open, .ags-te-overflow.ags-open { display: block; }
.ags-te-paste-menu { left: 6px; right: auto; min-width: 240px; }
.ags-te-recent-menu { min-width: 320px; }
.ags-te-menu.ags-te-emoji-panel {
    display: none;
    padding: 0;
    min-width: 0;
    overflow: hidden;
    max-height: none;
}
.ags-te-menu.ags-te-emoji-panel.ags-open { display: flex; }
.ags-te-lang-panel { min-width: 268px; max-height: none; }
.ags-te-lang-row { display: flex; align-items: center; gap: 4px; padding: 3px 4px; }
.ags-te-lang-row .ags-te-select { flex: 1 1 auto; min-width: 0; }
.ags-te-lang-run { width: 100%; padding: 4px 6px; }
.ags-te-lang-toggle { width: 100%; padding: 3px 6px; text-align: center; }
.ags-te-lang-scope { padding: 2px 8px; font-size: 9px; color: var(--ags-muted); }
.ags-te-lang-scope.ags-bad { color: var(--ags-danger); }
.ags-te-lang-status {
    padding: 4px 8px;
    font-size: 10px;
    line-height: 1.35;
    color: var(--ags-muted);
    word-break: break-word;
    white-space: normal;
}
.ags-te-lang-status.ags-ok { color: var(--ags-ok); }
.ags-te-lang-status.ags-bad { color: var(--ags-danger); }
.ags-te-lang-status.ags-busy { color: var(--ags-amber); }
.ags-te-menu-title {
    padding: 4px 8px 2px;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--ags-muted);
}
.ags-te-menu-item {
    display: block;
    width: 100%;
    text-align: left;
    border: 0;
    border-radius: 4px;
    background: transparent;
    color: var(--ags-text);
    font: 11px "Segoe UI", system-ui, sans-serif;
    padding: 4px 8px;
    cursor: pointer;
}
.ags-te-menu-item:hover { background: color-mix(in srgb, var(--ags-accent) 18%, transparent); }
.ags-te-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 6px;
    border-radius: 5px;
    border-bottom: 1px solid color-mix(in srgb, var(--ags-muted) 16%, transparent);
}
.ags-te-row:hover { background: color-mix(in srgb, var(--ags-accent) 10%, transparent); }
.ags-te-row-main { flex: 1 1 auto; min-width: 0; cursor: pointer; }
.ags-te-row-name {
    font-size: 11px;
    color: var(--ags-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ags-te-row-sub {
    font-size: 9px;
    color: var(--ags-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ags-te-row-btns { flex: 0 0 auto; display: flex; gap: 3px; }
.ags-te-empty { padding: 8px; font-size: 10px; color: var(--ags-muted); }
.ags-te-dialog {
    display: none;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 30;
    width: 280px;
    max-height: 86%;
    overflow: auto;
    padding: 10px;
    border: 1px solid color-mix(in srgb, var(--ags-accent) 40%, var(--ags-surface));
    border-radius: 8px;
    background: var(--ags-surface);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink)) var(--ags-ink);
}
.ags-te-dialog.ags-wide { width: 430px; }
.ags-te-dialog.ags-top { z-index: 40; }
.ags-te-dialog-title { font-size: 11px; color: var(--ags-text); margin-bottom: 6px; }
.ags-te-dialog-note { font-size: 10px; color: var(--ags-muted); margin-bottom: 6px; line-height: 1.4; }
.ags-te-dialog-row { display: flex; gap: 6px; margin-top: 8px; justify-content: flex-end; }
.ags-te-field { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
.ags-te-label { flex: 0 0 96px; font-size: 10px; color: var(--ags-muted); }
.ags-te-field .ags-te-select { flex: 1 1 auto; min-width: 0; }
.ags-te-field .ags-te-input { flex: 1 1 auto; min-width: 0; }
.ags-te-list {
    max-height: 240px;
    overflow: auto;
    border: 1px solid color-mix(in srgb, var(--ags-muted) 22%, transparent);
    border-radius: 5px;
    background: var(--ags-ink);
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent) 35%, var(--ags-ink)) var(--ags-ink);
}
.ags-te-paste-dialog { width: 340px; }
.ags-te-paste-box { width: 100%; height: 96px; resize: vertical; flex: 0 0 auto; }
.ags-te-toast {
    position: absolute;
    right: 8px;
    bottom: 26px;
    z-index: 45;
    display: none;
    max-width: 80%;
    padding: 4px 9px;
    border-radius: 5px;
    background: color-mix(in srgb, var(--ags-ink) 92%, transparent);
    border: 1px solid color-mix(in srgb, var(--ags-accent) 35%, transparent);
    color: var(--ags-text);
    font-size: 10px;
    pointer-events: none;
}
.ags-te-toast.ags-bad { border-color: var(--ags-danger); color: var(--ags-danger); }
.ags-te.ags-linked .ags-te-text { opacity: 0.65; }
.ags-te[data-theme="matrix"] .ags-te-hl,
.ags-te[data-theme="matrix"] .ags-te-text,
.ags-te[data-theme="amber"] .ags-te-hl,
.ags-te[data-theme="amber"] .ags-te-text,
.ags-te[data-theme="cyber"] .ags-te-hl,
.ags-te[data-theme="cyber"] .ags-te-text {
    text-shadow: 0 0 5px color-mix(in srgb, var(--ags-accent) 45%, transparent);
}
`;

function themeCss() {
    let out = "";
    for (const key of Object.keys(THEMES)) {
        const p = THEMES[key];
        out += ".ags-te[data-theme=\"" + key + "\"] {\n";
        out += "    --ags-bg: " + p.bg + ";\n";
        out += "    --ags-surface: " + p.surface + ";\n";
        out += "    --ags-ink: " + p.ink + ";\n";
        out += "    --ags-text: " + p.text + ";\n";
        out += "    --ags-muted: " + p.muted + ";\n";
        out += "    --ags-accent: " + p.accent + ";\n";
        out += "    --ags-amber: " + p.amber + ";\n";
        out += "    --ags-ok: " + p.ok + ";\n";
        out += "    --ags-danger: " + p.danger + ";\n";
        out += "    --ags-violet: " + p.violet + ";\n";
        out += "}\n";
    }
    return out;
}

function injectStyle() {
    const el = document.getElementById(STYLE_ID) || document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS + themeCss();
    if (!el.parentNode) document.head.appendChild(el);
}

function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function timeAgo(ts) {
    const sec = Math.max(0, Math.round((Date.now() - Number(ts || 0)) / 1000));
    if (sec < 60) return sec + "s";
    if (sec < 3600) return Math.round(sec / 60) + "m";
    if (sec < 86400) return Math.round(sec / 3600) + "h";
    return Math.round(sec / 86400) + "d";
}

function baseName(path) {
    const raw = String(path || "");
    const at = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
    return at >= 0 ? raw.slice(at + 1) : raw;
}

function parentDir(path) {
    const raw = String(path || "").replace(/[\\/]+$/, "");
    const at = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
    return at > 0 ? raw.slice(0, at) : "";
}

// Only an absolute location is a usable path / Только абсолютный путь пригоден для графа
function looksAbsolute(path) {
    const raw = String(path || "");
    return /^[A-Za-z]:[\\/]/.test(raw) || raw.startsWith("/") || raw.startsWith("\\\\");
}

function normalizePathValue(value) {
    const raw = String(value || "").trim();
    return looksAbsolute(raw) ? raw : "";
}

function readLastFolder() {
    try {
        return String(localStorage.getItem(LAST_FOLDER_KEY) || "");
    } catch (err) {
        return "";
    }
}

function writeLastFolder(path) {
    try {
        localStorage.setItem(LAST_FOLDER_KEY, String(path || ""));
    } catch (err) {
        console.warn("[AGSoft Text Editor] Last folder not stored: " +
            String((err && err.message) || err));
    }
}

// The comparison buffer travels with the workflow / Буфер сравнения хранится в workflow
function unpackValue(value) {
    let text = typeof value === "string" ? value : "";
    let negative = "";
    let result = "";
    let filepath = "";
    const pathIdx = text.indexOf(PATH_MARK);
    if (pathIdx >= 0) {
        filepath = text.slice(pathIdx + PATH_MARK.length).trim();
        text = text.slice(0, pathIdx);
    }
    const resIdx = text.indexOf(RESULT_MARK);
    if (resIdx >= 0) {
        result = text.slice(resIdx + RESULT_MARK.length);
        text = text.slice(0, resIdx);
    }
    const negIdx = text.indexOf(NEG_MARK);
    if (negIdx >= 0) {
        negative = text.slice(negIdx + NEG_MARK.length);
        text = text.slice(0, negIdx);
    }
    return { text, negative, result, filepath };
}

function capitalizeLike(value, origin) {
    const first = origin.charAt(0);
    if (first && first === first.toUpperCase() && first !== first.toLowerCase()) {
        return value.charAt(0).toUpperCase() + value.slice(1);
    }
    return value;
}

function translitRuEn(text) {
    let out = "";
    for (const ch of text) {
        const low = ch.toLowerCase();
        if (RU_EN[low] !== undefined) {
            const mapped = RU_EN[low];
            out += ch !== low ? capitalizeLike(mapped, ch) : mapped;
        } else {
            out += ch;
        }
    }
    return out;
}

function translitEnRu(text) {
    let out = "";
    let i = 0;
    while (i < text.length) {
        const rest = text.slice(i).toLowerCase();
        let matched = false;
        for (const pair of EN_RU_MULTI) {
            if (rest.startsWith(pair[0])) {
                out += capitalizeLike(pair[1], text.slice(i, i + pair[0].length));
                i += pair[0].length;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        const ch = text.charAt(i);
        const low = ch.toLowerCase();
        out += EN_RU_SINGLE[low] !== undefined ? capitalizeLike(EN_RU_SINGLE[low], ch) : ch;
        i++;
    }
    return out;
}

// A failed translation is reported, never substituted / Неудачный перевод сообщается, не подменяется
async function translateText(text, from, to) {
    const query = text.trim();
    if (!query) return { ok: false, error: "Empty text / Пустой текст" };
    if (from === to) return { ok: false, error: "Same language / Одинаковый язык" };
    if (query.length > TRANSLATE_LIMIT) {
        return {
            ok: false,
            error: "Too long: " + query.length + " > " + TRANSLATE_LIMIT +
                " / Слишком длинный текст",
        };
    }
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TRANSLATE_TIMEOUT);
    const endpoint = "//translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=" +
        LANGS[from].code + "&tl=" + LANGS[to].code + "&q=" + encodeURIComponent(query);
    try {
        const res = await fetch(endpoint, { signal: controller.signal });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        if (!Array.isArray(data) || !Array.isArray(data[0])) {
            throw new Error("Unexpected response format / Неожиданный формат ответа");
        }
        const out = data[0].map((part) => (part && part[0] ? part[0] : "")).join("");
        if (!out.trim()) throw new Error("Empty result / Пустой результат");
        return { ok: true, text: out, provider: "Google" };
    } catch (err) {
        const reason = err && err.name === "AbortError"
            ? "Timeout " + TRANSLATE_TIMEOUT + " ms / Таймаут"
            : String((err && err.message) || err);
        console.warn("[AGSoft Text Editor] Translation failed " + from + " -> " + to + ": " + reason);
        return { ok: false, error: reason };
    } finally {
        clearTimeout(timer);
    }
}

// Both workflow layouts carry prompt nodes / Оба формата workflow содержат ноды промптов
function extractWorkflowTexts(data) {
    const out = [];
    if (data && Array.isArray(data.nodes)) {
        for (const n of data.nodes) {
            if (!n || typeof n !== "object") continue;
            const type = String(n.type || "");
            if (!/CLIPTextEncode/i.test(type)) continue;
            const vals = n.widgets_values;
            let text = "";
            if (Array.isArray(vals)) {
                for (const v of vals) {
                    if (typeof v === "string" && v.length > text.length) text = v;
                }
            } else if (vals && typeof vals === "object" && typeof vals.text === "string") {
                text = vals.text;
            }
            out.push({ id: String(n.id === undefined ? "?" : n.id), label: n.title || type, text });
        }
        return out;
    }
    if (data && typeof data === "object") {
        for (const key of Object.keys(data)) {
            const n = data[key];
            if (!n || typeof n !== "object") continue;
            const type = String(n.class_type || "");
            if (!/CLIPTextEncode/i.test(type)) continue;
            const inputs = n.inputs || {};
            const meta = n._meta || {};
            out.push({
                id: key,
                label: meta.title || type,
                text: typeof inputs.text === "string" ? inputs.text : "",
            });
        }
    }
    return out;
}

function inlineMd(s) {
    let t = escapeHtml(s);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, "<span class=\"ags-md-link\" title=\"$2\">$1</span>");
    return t;
}

function markdownToHtml(src) {
    const lines = String(src).replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let inCode = false;
    let listType = "";
    let para = [];
    const flushPara = () => {
        if (!para.length) return;
        out.push("<p>" + inlineMd(para.join(" ")) + "</p>");
        para = [];
    };
    const closeList = () => {
        if (!listType) return;
        out.push(listType === "ul" ? "</ul>" : "</ol>");
        listType = "";
    };
    for (const raw of lines) {
        const line = raw.replace(/\s+$/, "");
        if (/^\s*```/.test(line)) {
            flushPara();
            closeList();
            out.push(inCode ? "</pre>" : "<pre class=\"ags-md-code\">");
            inCode = !inCode;
            continue;
        }
        if (inCode) {
            out.push(escapeHtml(raw) + "\n");
            continue;
        }
        if (!line.trim()) {
            flushPara();
            closeList();
            continue;
        }
        const head = /^(#{1,6})\s+(.*)$/.exec(line);
        if (head) {
            flushPara();
            closeList();
            const lv = head[1].length;
            out.push("<h" + lv + ">" + inlineMd(head[2]) + "</h" + lv + ">");
            continue;
        }
        if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
            flushPara();
            closeList();
            out.push("<hr>");
            continue;
        }
        const quote = /^\s*>\s?(.*)$/.exec(line);
        if (quote) {
            flushPara();
            closeList();
            out.push("<blockquote>" + inlineMd(quote[1]) + "</blockquote>");
            continue;
        }
        const ul = /^\s*[-*+]\s+(.*)$/.exec(line);
        const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
        if (ul || ol) {
            flushPara();
            const want = ul ? "ul" : "ol";
            if (listType !== want) {
                closeList();
                out.push("<" + want + ">");
                listType = want;
            }
            out.push("<li>" + inlineMd((ul || ol)[1]) + "</li>");
            continue;
        }
        closeList();
        para.push(line.trim());
    }
    if (inCode) out.push("</pre>");
    flushPara();
    closeList();
    return out.join("");
}

function jsonTree(value, key, depth) {
    const row = document.createElement("div");
    row.className = "ags-jt-row";
    const isObj = value !== null && typeof value === "object";
    if (!isObj) {
        const k = document.createElement("span");
        k.className = "ags-jt-key";
        k.textContent = key + ": ";
        const v = document.createElement("span");
        v.className = "ags-jt-" + (value === null ? "null" : typeof value);
        v.textContent = JSON.stringify(value);
        row.append(k, v);
        return row;
    }
    const arr = Array.isArray(value);
    const entries = arr ? value.map((v, i) => [String(i), v]) : Object.entries(value);
    const det = document.createElement("details");
    det.open = depth < 2;
    const sum = document.createElement("summary");
    sum.textContent = key + (arr ? " [" + entries.length + "]" : " {" + entries.length + "}");
    det.appendChild(sum);
    if (depth < 12) {
        for (const pair of entries.slice(0, 500)) det.appendChild(jsonTree(pair[1], pair[0], depth + 1));
        if (entries.length > 500) {
            const more = document.createElement("div");
            more.className = "ags-jt-more";
            more.textContent = "… +" + (entries.length - 500);
            det.appendChild(more);
        }
    }
    row.appendChild(det);
    return row;
}

function findIssues(text) {
    const issues = [];
    const stack = [];
    const pairs = { ")": "(", "]": "[", "}": "{" };
    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (ch === "\\") {
            i++;
            continue;
        }
        if (ch === "(" || ch === "[" || ch === "{") stack.push({ ch: ch, i: i });
        else if (ch === ")" || ch === "]" || ch === "}") {
            const top = stack.pop();
            if (!top || top.ch !== pairs[ch]) issues.push({ pos: i, kind: "bracket" });
        }
        if (issues.length > 40) break;
    }
    for (const item of stack) issues.push({ pos: item.i, kind: "bracket" });
    const re = /:\s*([+-]?\d*\.?\d+)\s*\)/g;
    let m;
    while ((m = re.exec(text)) !== null) {
        const val = parseFloat(m[1]);
        if (isFinite(val) && Math.abs(val) > 2) issues.push({ pos: m.index, kind: "weight" });
        if (issues.length > 60) break;
    }
    issues.sort((a, b) => a.pos - b.pos);
    return issues;
}

function nodeAtEvent(e) {
    const canvas = app.canvas;
    if (!canvas || !canvas.graph) return null;
    const pos = canvas.convertEventToCanvasOffset(e);
    const list = canvas.graph._nodes || [];
    if (canvas.graph.getNodeOnPos) {
        const found = canvas.graph.getNodeOnPos(pos[0], pos[1], list);
        if (found) return found;
    }
    for (let i = list.length - 1; i >= 0; i--) {
        const n = list[i];
        if (pos[0] >= n.pos[0] && pos[0] <= n.pos[0] + n.size[0] &&
            pos[1] >= n.pos[1] && pos[1] <= n.pos[1] + n.size[1]) return n;
    }
    return null;
}

function instanceAtEvent(e) {
    const found = nodeAtEvent(e);
    if (!found) return null;
    for (const inst of INSTANCES) if (inst.node === found) return inst;
    return null;
}

function clearDropHints() {
    for (const inst of INSTANCES) inst.setDropHint(false);
}

function readRecent() {
    try {
        const raw = localStorage.getItem(RECENT_KEY);
        const list = raw ? JSON.parse(raw) : [];
        return Array.isArray(list) ? list : [];
    } catch (err) {
        return [];
    }
}

function writeRecent(list) {
    try {
        localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, RECENT_MAX)));
        return true;
    } catch (err) {
        console.warn("[AGSoft Text Editor] Recent list not stored: " +
            String((err && err.message) || err));
        return false;
    }
}

function pushRecent(name, text, kind, path) {
    const body = String(text || "");
    const fits = body.length <= RECENT_TEXT_MAX;
    const list = readRecent().filter((it) => !(it.name === name && it.kind === kind));
    list.unshift({
        name: String(name || "untitled"),
        kind: String(kind || "text"),
        time: Date.now(),
        size: body.length,
        text: fits ? body : "",
        full: fits,
        path: normalizePathValue(path),
    });
    writeRecent(list);
}

function downloadBody(body, name, mime) {
    const blob = new Blob([body], { type: mime + ";charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
}

app.registerExtension({
    name: "AGSoft.TextEditor",
    setup() {
        const canvasEl = app.canvasEl;
        if (!canvasEl) return;
        canvasEl.addEventListener("dragover", (e) => {
            const inst = instanceAtEvent(e);
            if (!inst) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = "copy";
            clearDropHints();
            inst.setDropHint(true);
        });
        canvasEl.addEventListener("dragleave", () => clearDropHints());
        canvasEl.addEventListener("drop", (e) => {
            const inst = instanceAtEvent(e);
            if (!inst) return;
            e.preventDefault();
            e.stopPropagation();
            clearDropHints();
            inst.drop(e.dataTransfer.files);
        });
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "AGSoftTextEditor") return;
        injectStyle();
        const textWidget = (node.widgets || []).find((w) => w.name === "text");
        if (!textWidget) return;

        let fontSize = 13;
        let wrap = false;
        let hlOn = true;
        let theme = "ink";
        let appliedTheme = "";
        let viewMode = "single";
        let activeBuf = "text";
        let splitTarget = "negative";
        let splitToResult = true;
        let resultSource = "text";
        let zenMode = false;
        let zenParent = null;
        let zenNext = null;
        let dropHint = false;
        let lastDropAt = 0;
        let savedName = "prompt.txt";
        let pathValue = "";
        let fileLabel = "";
        let issues = [];
        let rafId = 0;
        let tokTimer = 0;
        let tokSeq = 0;
        let toastTimer = 0;
        let autosaveTimer = 0;
        let autosaveOk = true;
        let dlgCallback = null;
        let confirmCallback = null;
        let pasteMode = "normal";
        let overflowMode = false;
        let translating = false;
        let rootsLoaded = false;
        let pendingRoot = "";
        let emojiApi = null;
        let pathResolveFn = null;
        let lastPush = 0;
        let undoStack = [];
        let redoStack = [];
        let matches = [];
        let matchIndex = -1;
        let searchOpen = false;
        let searchBad = false;
        let optCase = false;
        let optRegex = false;
        let optWhole = false;
        let pane = null;
        let ta = null;
        let hl = null;
        let gutter = null;
        let gText = null;
        let lineHl = null;
        let gHl = null;

        const bufs = {
            text: { undo: [], redo: [] },
            negative: { undo: [], redo: [] },
            result: { undo: [], redo: [] },
        };

        const container = document.createElement("div");
        container.className = "ags-te";
        container.dataset.theme = theme;

        function mkBtn(label, title, extra) {
            const b = document.createElement("button");
            b.type = "button";
            b.className = "ags-te-btn" + (extra ? " " + extra : "");
            b.textContent = label;
            b.title = title;
            return b;
        }

        function mkSep() {
            const s = document.createElement("span");
            s.className = "ags-te-sep";
            return s;
        }

        function mkSelect(title, items) {
            const s = document.createElement("select");
            s.className = "ags-te-select";
            s.title = title;
            for (const item of items) {
                const o = document.createElement("option");
                o.value = item[0];
                o.textContent = item[1];
                s.appendChild(o);
            }
            return s;
        }

        function mkField(label, control) {
            const row = document.createElement("div");
            row.className = "ags-te-field";
            const cap = document.createElement("span");
            cap.className = "ags-te-label";
            cap.textContent = label;
            row.append(cap, control);
            return row;
        }

        const bar = document.createElement("div");
        bar.className = "ags-te-bar ags-te-toolbar";
        const primaryGroup = document.createElement("div");
        primaryGroup.className = "ags-te-group";
        const secondaryGroup = document.createElement("div");
        secondaryGroup.className = "ags-te-group";
        const newBtn = mkBtn("🆕", "New document, clears text and path (Ctrl+Alt+N)\n---\n" +
            "Новый документ, очищает текст и путь (Ctrl+Alt+N)");
        const openBtn = mkBtn("📂", "Open a document from any folder\n---\n" +
            "Открыть документ из любой папки");
        const pathBtn = mkBtn("🔗", "Open by absolute path (Ctrl+O)\n---\n" +
            "Открыть по абсолютному пути (Ctrl+O)");
        const saveBtn = mkBtn("💾", "Save or export (Ctrl+S)\n---\nСохранить или экспортировать (Ctrl+S)");
        const pngBtn = mkBtn("🖼", "Import prompt from PNG\n---\nИмпорт промпта из PNG");
        const wfBtn = mkBtn("🧩", "Pick texts from a workflow JSON\n---\nВыбрать тексты из workflow JSON");
        const copyBtn = mkBtn("⧉", "Copy selection, or all text when nothing is selected\n---\n" +
            "Копировать выделение, а без выделения — весь текст");
        const pasteBtn = mkBtn("📥", "Paste at cursor as one undo step\n---\nВставить в курсор одним шагом undo");
        const pasteMoreBtn = mkBtn("▾", "Paste special\n---\nСпециальная вставка");
        const cutBtn = mkBtn("✂", "Cut selection, or the current line\n---\n" +
            "Вырезать выделение или текущую строку");
        const undoBtn = mkBtn("↶", "Undo (Ctrl+Z)\n---\nОтменить (Ctrl+Z)");
        const redoBtn = mkBtn("↷", "Redo (Ctrl+Y)\n---\nПовторить (Ctrl+Y)");
        const findBtn = mkBtn("🔍", "Search / replace (Ctrl+F, Ctrl+H)\n---\nПоиск / замена (Ctrl+F, Ctrl+H)");
        const emojiBtn = mkBtn("😀", "Insert emoji, searchable in Russian and English\n---\n" +
            "Вставить эмодзи, поиск по-русски и по-английски");
        const clearBtn = mkBtn("✕", "Clear the focused buffer\n---\nОчистить активный буфер", "ags-danger");
        const splitBtn = mkBtn("⫴", "Split view, two panes side by side\n---\n" +
            "Split: две панели рядом");
        const zenBtn = mkBtn("⛶", "Zen fullscreen, Esc to exit\n---\nZen на весь экран, Esc — выход");
        const recentBtn = mkBtn("🕘", "Recent files, last 10\n---\nПоследние файлы, 10 штук");
        const wrapBtn = mkBtn("Wrap", "Toggle word wrap\n---\nПеренос строк");
        const minusBtn = mkBtn("A−", "Decrease font size\n---\nУменьшить шрифт");
        const plusBtn = mkBtn("A+", "Increase font size\n---\nУвеличить шрифт");
        const hlBtn = mkBtn("HL", "Prompt syntax highlighting\n---\nПодсветка синтаксиса промпта");
        const langBtn = mkBtn("🌐", "Translate: English, Russian, Chinese\n---\n" +
            "Перевод: английский, русский, китайский");
        const themeSel = mkSelect("Editor theme\n---\nТема редактора",
            Object.keys(THEMES).map((k) => [k, THEMES[k].label]));
        const toolsBtn = mkBtn("Tools ▾", "Text utilities\n---\nУтилиты текста");
        const moreBtn = mkBtn("⋯", "More controls\n---\nЕщё элементы");
        moreBtn.style.display = "none";
        primaryGroup.append(newBtn, openBtn, pathBtn, saveBtn, pngBtn, wfBtn, mkSep(),
            copyBtn, pasteBtn, pasteMoreBtn, cutBtn, mkSep(), undoBtn, redoBtn, mkSep(),
            findBtn, emojiBtn);
        secondaryGroup.append(splitBtn, zenBtn, recentBtn, clearBtn, mkSep(), wrapBtn,
            minusBtn, plusBtn, hlBtn, langBtn, mkSep(), themeSel, toolsBtn);
        bar.append(primaryGroup, secondaryGroup, moreBtn);

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = ".txt,.md,.json,.csv,.log,.ini,.yaml,.yml,text/*";
        fileInput.style.display = "none";
        const pngInput = document.createElement("input");
        pngInput.type = "file";
        pngInput.accept = "image/png";
        pngInput.style.display = "none";
        const wfInput = document.createElement("input");
        wfInput.type = "file";
        wfInput.accept = ".json,application/json";
        wfInput.style.display = "none";

        const tabBar = document.createElement("div");
        tabBar.className = "ags-te-bar ags-te-tabs";
        const tabText = mkBtn("Text", "Main text buffer\n---\nОсновной буфер", "ags-te-tab");
        const tabNeg = mkBtn("Negative", "Negative buffer\n---\nБуфер negative", "ags-te-tab");
        const tabRes = mkBtn("Result", "Translation and translit result\n---\n" +
            "Результат перевода и транслита", "ags-te-tab");
        const tabPrev = mkBtn("Preview", "Markdown / JSON preview\n---\nПревью Markdown / JSON", "ags-te-tab");
        const splitCtl = document.createElement("div");
        splitCtl.className = "ags-te-splitctl";
        const splitSel = mkSelect("What the right pane shows\n---\nЧто показывает правая панель", [
            ["negative", "Text | Negative"],
            ["result", "Original | Result"],
        ]);
        const applyResBtn = mkBtn("↤", "Apply the result to the original\n---\n" +
            "Применить результат к оригиналу", "ags-on");
        applyResBtn.style.display = "none";
        splitCtl.append(splitSel, applyResBtn);
        const fileTag = document.createElement("span");
        fileTag.className = "ags-te-filetag ags-new";
        fileTag.textContent = NEW_DOCUMENT_LABEL;
        const zenBadge = document.createElement("span");
        zenBadge.className = "ags-te-zenbadge";
        zenBadge.textContent = "Zen · Esc";
        const badge = document.createElement("span");
        badge.className = "ags-te-badge";
        badge.style.display = "none";
        badge.textContent = "⇄ input linked / вход подключён";
        tabBar.append(tabText, tabNeg, tabRes, tabPrev, splitCtl, fileTag, zenBadge, badge);

        const searchBar = document.createElement("div");
        searchBar.className = "ags-te-bar ags-te-search";
        const findInput = document.createElement("input");
        findInput.className = "ags-te-input";
        findInput.placeholder = "Find / Найти";
        const caseBtn = mkBtn("Aa", "Match case\n---\nУчитывать регистр");
        const regexBtn = mkBtn(".*", "Regular expression\n---\nРегулярное выражение");
        const wholeBtn = mkBtn("W", "Whole word\n---\nСлово целиком");
        const countEl = document.createElement("span");
        countEl.className = "ags-te-count";
        countEl.textContent = "0/0";
        const prevBtn = mkBtn("◀", "Previous match (Shift+Enter)\n---\nПредыдущее (Shift+Enter)");
        const nextBtn = mkBtn("▶", "Next match (Enter)\n---\nСледующее (Enter)");
        const replInput = document.createElement("input");
        replInput.className = "ags-te-input";
        replInput.placeholder = "Replace / Заменить";
        const replBtn = mkBtn("Repl", "Replace current\n---\nЗаменить текущее");
        const replAllBtn = mkBtn("All", "Replace all as one undo step\n---\nЗаменить всё одним шагом");
        const closeSearch = mkBtn("✕", "Close search (Esc)\n---\nЗакрыть поиск (Esc)");
        searchBar.append(findInput, caseBtn, regexBtn, wholeBtn, countEl, prevBtn, nextBtn,
            replInput, replBtn, replAllBtn, closeSearch);

        // Each buffer owns a pane, so any pair can be compared.
        // У каждого буфера своя панель, поэтому сравнить можно любую пару.
        function makePane(bufName) {
            const root = document.createElement("div");
            root.className = "ags-te-pane";
            const head = document.createElement("div");
            head.className = "ags-te-pane-head";
            const body = document.createElement("div");
            body.className = "ags-te-pane-body";
            const gutterEl = document.createElement("div");
            gutterEl.className = "ags-te-gutter";
            const gHlEl = document.createElement("div");
            gHlEl.className = "ags-te-linehl";
            const gTextEl = document.createElement("div");
            gTextEl.className = "ags-te-gtext";
            gutterEl.append(gHlEl, gTextEl);
            const textWrap = document.createElement("div");
            textWrap.className = "ags-te-textwrap";
            const lineHlEl = document.createElement("div");
            lineHlEl.className = "ags-te-linehl";
            const hlEl = document.createElement("pre");
            hlEl.className = "ags-te-hl";
            const taEl = document.createElement("textarea");
            taEl.className = "ags-te-text";
            taEl.spellcheck = false;
            if (bufName === "text") taEl.placeholder = "Text… / Текст…";
            else if (bufName === "negative") taEl.placeholder = "Negative… / Негатив…";
            else taEl.placeholder = "Result… / Результат…";
            taEl.setAttribute("aria-label", "AGSoft Text Editor " + bufName);
            textWrap.append(lineHlEl, hlEl, taEl);
            body.append(gutterEl, textWrap);
            root.append(head, body);
            return {
                buf: bufName, root, head, body, gutter: gutterEl, gHl: gHlEl,
                gText: gTextEl, textWrap, lineHl: lineHlEl, hl: hlEl, ta: taEl,
                gutterLines: -1,
            };
        }

        const paneText = makePane("text");
        const paneNeg = makePane("negative");
        const paneRes = makePane("result");
        const ALL_PANES = [paneText, paneNeg, paneRes];
        const editor = document.createElement("div");
        editor.className = "ags-te-editor";
        editor.append(paneText.root, paneNeg.root, paneRes.root);
        const previewEl = document.createElement("div");
        previewEl.className = "ags-te-preview";

        const status = document.createElement("div");
        status.className = "ags-te-status";

        function mkSeg(cls, title) {
            const s = document.createElement("span");
            s.className = "ags-seg" + (cls ? " " + cls : "");
            if (title) s.title = title;
            return s;
        }

        const segCur = mkSeg("ags-cur ags-click", "Go to line\n---\nПерейти к строке");
        const segSel = mkSeg("ags-sel", "Selected characters\n---\nВыделено символов");
        const segWarn = mkSeg("ags-warn ags-click", "Jump to first issue\n---\nПерейти к первой проблеме");
        const segAuto = mkSeg("ags-auto ags-click",
            "Autosave in this browser, click to restore\n---\nАвтосохранение в браузере, клик — восстановить");
        segAuto.textContent = "AS —";
        const right = document.createElement("span");
        right.className = "ags-right";
        const segBuf = mkSeg("", "Focused buffer\n---\nАктивный буфер");
        const segTok = mkSeg("ags-click",
            "CLIP tokens of the focused buffer\n---\nCLIP-токены активного буфера");
        const segLines = mkSeg("", "Total lines\n---\nВсего строк");
        const segChars = mkSeg("", "Total characters\n---\nВсего символов");
        const segWords = mkSeg("", "Word count\n---\nКоличество слов");
        const segEol = mkSeg("", "Line endings\n---\nПереводы строк");
        const segEnc = mkSeg("", "Output encoding\n---\nКодировка вывода");
        segEnc.textContent = "UTF-8";
        segTok.textContent = "Tok 0/0";
        right.append(segBuf, segTok, segLines, segChars, segWords, segEol, segEnc);
        status.append(segCur, segSel, segWarn, segAuto, right);

        const menu = document.createElement("div");
        menu.className = "ags-te-menu";
        const overflowPanel = document.createElement("div");
        overflowPanel.className = "ags-te-overflow";
        const pasteMenu = document.createElement("div");
        pasteMenu.className = "ags-te-menu ags-te-paste-menu";
        const recentMenu = document.createElement("div");
        recentMenu.className = "ags-te-menu ags-te-recent-menu";
        const emojiPanel = document.createElement("div");
        emojiPanel.className = "ags-te-menu ags-te-emoji-panel";

        const langPanel = document.createElement("div");
        langPanel.className = "ags-te-menu ags-te-lang-panel";
        const langTitle = document.createElement("div");
        langTitle.className = "ags-te-menu-title";
        langTitle.textContent = "Translate / Перевод";
        const langRow = document.createElement("div");
        langRow.className = "ags-te-lang-row";
        const langItems = Object.keys(LANGS).map((k) => [k, LANGS[k].label]);
        const langFrom = mkSelect("Source language\n---\nИсходный язык", langItems);
        const langSwap = mkBtn("⇄", "Swap languages\n---\nПоменять языки местами");
        const langTo = mkSelect("Target language\n---\nЦелевой язык", langItems);
        langFrom.value = "ru";
        langTo.value = "en";
        langRow.append(langFrom, langSwap, langTo);
        const langToggleRow = document.createElement("div");
        langToggleRow.className = "ags-te-lang-row";
        const langSplitBtn = mkBtn("→ Split / → В сплит",
            "Keep the original and put the result in the right pane\n---\n" +
            "Оставить оригинал, результат положить в правую панель",
            "ags-te-lang-toggle");
        langToggleRow.append(langSplitBtn);
        const langRunRow = document.createElement("div");
        langRunRow.className = "ags-te-lang-row";
        const langRun = mkBtn("Translate / Перевести",
            "Translate selection, or the whole buffer\n---\nПеревести выделение или весь буфер",
            "ags-on ags-te-lang-run");
        langRunRow.append(langRun);
        const langScope = document.createElement("div");
        langScope.className = "ags-te-lang-scope";
        const langStatus = document.createElement("div");
        langStatus.className = "ags-te-lang-status";
        langStatus.textContent = "Ready / Готов";
        langPanel.append(langTitle, langRow, langToggleRow, langRunRow, langScope, langStatus);

        const dialog = document.createElement("div");
        dialog.className = "ags-te-dialog ags-top";
        const dlgTitle = document.createElement("div");
        dlgTitle.className = "ags-te-dialog-title";
        const dlgInput = document.createElement("input");
        dlgInput.className = "ags-te-input";
        const dlgRow = document.createElement("div");
        dlgRow.className = "ags-te-dialog-row";
        const dlgCancel = mkBtn("Cancel", "Cancel\n---\nОтмена");
        const dlgOk = mkBtn("OK", "Confirm\n---\nПодтвердить", "ags-on");
        dlgRow.append(dlgCancel, dlgOk);
        dialog.append(dlgTitle, dlgInput, dlgRow);

        const confirmDialog = document.createElement("div");
        confirmDialog.className = "ags-te-dialog ags-wide ags-top";
        const confTitle = document.createElement("div");
        confTitle.className = "ags-te-dialog-title";
        const confNote = document.createElement("div");
        confNote.className = "ags-te-dialog-note";
        const confRow = document.createElement("div");
        confRow.className = "ags-te-dialog-row";
        const confCancel = mkBtn("Cancel", "Cancel\n---\nОтмена");
        const confOk = mkBtn("OK", "Confirm\n---\nПодтвердить", "ags-on");
        confRow.append(confCancel, confOk);
        confirmDialog.append(confTitle, confNote, confRow);

        const pasteDialog = document.createElement("div");
        pasteDialog.className = "ags-te-dialog ags-te-paste-dialog";
        const pasteTitle = document.createElement("div");
        pasteTitle.className = "ags-te-dialog-title";
        pasteTitle.textContent = "Press Ctrl+V here, then OK / Нажмите Ctrl+V здесь, затем OK";
        const pasteBox = document.createElement("textarea");
        pasteBox.className = "ags-te-input ags-te-paste-box";
        const pasteRow = document.createElement("div");
        pasteRow.className = "ags-te-dialog-row";
        const pasteCancel = mkBtn("Cancel", "Cancel\n---\nОтмена");
        const pasteOk = mkBtn("OK", "Insert\n---\nВставить", "ags-on");
        pasteRow.append(pasteCancel, pasteOk);
        pasteDialog.append(pasteTitle, pasteBox, pasteRow);

        const pathDialog = document.createElement("div");
        pathDialog.className = "ags-te-dialog ags-wide ags-top";
        const pathTitle = document.createElement("div");
        pathTitle.className = "ags-te-dialog-title";
        const pathNote = document.createElement("div");
        pathNote.className = "ags-te-dialog-note";
        const pathInput = document.createElement("input");
        pathInput.className = "ags-te-input";
        pathInput.placeholder = "C:\\temp\\prompt.txt";
        const pathList = document.createElement("div");
        pathList.className = "ags-te-list";
        pathList.style.display = "none";
        const pathStatus = document.createElement("div");
        pathStatus.className = "ags-te-lang-status";
        const pathRow = document.createElement("div");
        pathRow.className = "ags-te-dialog-row";
        const pathCancel = mkBtn("Cancel", "Cancel\n---\nОтмена");
        const pathOk = mkBtn("Open / Открыть", "Read the file\n---\nПрочитать файл", "ags-on");
        pathRow.append(pathCancel, pathOk);
        pathDialog.append(pathTitle, pathNote, mkField("Path", pathInput),
            pathList, pathStatus, pathRow);

        const saveDialog = document.createElement("div");
        saveDialog.className = "ags-te-dialog ags-wide";
        const saveTitle = document.createElement("div");
        saveTitle.className = "ags-te-dialog-title";
        saveTitle.textContent = "Save / Export — Сохранить / Экспорт";
        const saveContent = mkSelect("What to write\n---\nЧто записать", [
            ["text", "Text only (.txt)"],
            ["md", "Text + Negative (.md)"],
            ["json", "Text + Negative (.json)"],
        ]);
        const saveDest = mkSelect("Destination\n---\nНазначение", [
            ["server", "Folder on disk, full path"],
            ["local", "Browser dialog"],
        ]);
        saveDest.value = "server";
        const saveRoot = mkSelect("Destination folder\n---\nПапка назначения", [
            ["output", "ComfyUI output"],
            ["input", "ComfyUI input"],
            ["temp", "ComfyUI temp"],
        ]);
        const rootAddBtn = mkBtn("＋", "Add another save folder by absolute path\n---\n" +
            "Добавить папку записи по абсолютному пути");
        const rootDelBtn = mkBtn("－", "Remove the added save folder\n---\nУбрать добавленную папку",
            "ags-danger");
        const saveSub = document.createElement("input");
        saveSub.className = "ags-te-input";
        saveSub.value = "text";
        saveSub.placeholder = "subfolder / подпапка";
        const saveName = document.createElement("input");
        saveName.className = "ags-te-input";
        saveName.value = "prompt.txt";
        const saveNote = document.createElement("div");
        saveNote.className = "ags-te-dialog-note";
        const saveStatus = document.createElement("div");
        saveStatus.className = "ags-te-lang-status";
        const saveRow = document.createElement("div");
        saveRow.className = "ags-te-dialog-row";
        const saveCancel = mkBtn("Cancel", "Cancel\n---\nОтмена");
        const saveRun = mkBtn("Save / Сохранить", "Write the file\n---\nЗаписать файл", "ags-on");
        saveRow.append(saveCancel, saveRun);
        const subField = mkField("Subfolder", saveSub);
        const rootField = mkField("Folder", saveRoot);
        rootField.append(rootAddBtn, rootDelBtn);
        saveDialog.append(saveTitle, mkField("Content", saveContent), mkField("Destination", saveDest),
            rootField, subField, mkField("File name", saveName), saveNote, saveStatus, saveRow);

        const wfDialog = document.createElement("div");
        wfDialog.className = "ags-te-dialog ags-wide";
        const wfTitle = document.createElement("div");
        wfTitle.className = "ags-te-dialog-title";
        wfTitle.textContent = "Workflow texts / Тексты workflow";
        const wfNote = document.createElement("div");
        wfNote.className = "ags-te-dialog-note";
        const wfList = document.createElement("div");
        wfList.className = "ags-te-list";
        const wfRow = document.createElement("div");
        wfRow.className = "ags-te-dialog-row";
        const wfPair = mkBtn("First two as pair", "Load 1 into Text and 2 into Negative\n---\n" +
            "Первый в Text, второй в Negative");
        const wfClose = mkBtn("Close", "Close\n---\nЗакрыть");
        wfRow.append(wfPair, wfClose);
        wfDialog.append(wfTitle, wfNote, wfList, wfRow);

        const restoreDialog = document.createElement("div");
        restoreDialog.className = "ags-te-dialog ags-wide";
        const restTitle = document.createElement("div");
        restTitle.className = "ags-te-dialog-title";
        restTitle.textContent = "Autosave found / Найдено автосохранение";
        const restNote = document.createElement("div");
        restNote.className = "ags-te-dialog-note";
        const restRow = document.createElement("div");
        restRow.className = "ags-te-dialog-row";
        const restDelete = mkBtn("Delete", "Remove the stored copy\n---\nУдалить копию", "ags-danger");
        const restKeep = mkBtn("Keep current", "Keep the workflow text\n---\nОставить текст workflow");
        const restLoad = mkBtn("Restore", "Restore the autosaved text\n---\nВосстановить текст", "ags-on");
        restRow.append(restDelete, restKeep, restLoad);
        restoreDialog.append(restTitle, restNote, restRow);

        const toast = document.createElement("div");
        toast.className = "ags-te-toast";

        container.append(bar, tabBar, searchBar, editor, previewEl, status, menu, overflowPanel,
            pasteMenu, recentMenu, emojiPanel, langPanel, pasteDialog, saveDialog, wfDialog,
            restoreDialog, pathDialog, confirmDialog, dialog, toast, fileInput, pngInput, wfInput);

        node.size = [820, 560];
        const domWidget = node.addDOMWidget("agsoft_editor", "ags_te", container, {
            serialize: false,
            getMinHeight: () => MIN_EDITOR_H,
        });

        const domIdx = node.widgets.indexOf(domWidget);
        if (domIdx > 0) {
            node.widgets.splice(domIdx, 1);
            node.widgets.unshift(domWidget);
        }
        const origType = textWidget.type;
        textWidget.hidden = true;
        textWidget.y = -9999;
        textWidget.type = "hidden";
        const origCompute = textWidget.computeSize;
        textWidget.computeSize = function(width) {
            if (this.hidden) return [0, -4];
            return origCompute ? origCompute.apply(this, arguments) : [width, 20];
        };

        // Zen owns the element, so widget positioning pauses / В Zen позиция виджета не применяется
        const origDomDraw = domWidget.draw;
        domWidget.draw = function() {
            if (zenMode) return;
            if (origDomDraw) origDomDraw.apply(this, arguments);
        };

        function flash(msg, bad) {
            toast.textContent = msg;
            toast.classList.toggle("ags-bad", !!bad);
            toast.style.display = "block";
            if (toastTimer) clearTimeout(toastTimer);
            toastTimer = setTimeout(() => { toast.style.display = "none"; }, 3400);
        }

        function autosaveKey() {
            return AUTOSAVE_PREFIX + String(node.id === undefined ? "x" : node.id);
        }

        function paneFor(name) {
            if (name === "negative") return paneNeg;
            if (name === "result") return paneRes;
            return paneText;
        }

        function updateFileTag() {
            const label = pathValue ? baseName(pathValue) : NEW_DOCUMENT_LABEL;
            if (label === fileLabel) return;
            fileLabel = label;
            fileTag.textContent = label;
            fileTag.classList.toggle("ags-new", !pathValue);
            fileTag.title = pathValue
                ? pathValue + "\n---\nClick to copy the path / Клик — скопировать путь"
                : "Click to save and get a path / Клик — сохранить и получить путь";
        }

        function bindPane(p) {
            pane = p;
            ta = p.ta;
            hl = p.hl;
            gutter = p.gutter;
            gText = p.gText;
            lineHl = p.lineHl;
            gHl = p.gHl;
            activeBuf = p.buf;
            undoStack = bufs[activeBuf].undo;
            redoStack = bufs[activeBuf].redo;
            for (const item of ALL_PANES) {
                item.root.classList.toggle("ags-active", item === p);
            }
            tabText.classList.toggle("ags-on", activeBuf === "text");
            tabNeg.classList.toggle("ags-on", activeBuf === "negative");
            tabRes.classList.toggle("ags-on", activeBuf === "result");
            if (activeBuf === "text") segBuf.textContent = "Text";
            else if (activeBuf === "negative") segBuf.textContent = "Neg";
            else segBuf.textContent = "Res";
        }

        function applyTheme() {
            const pal = THEMES[theme] || THEMES.ink;
            if (theme === appliedTheme) return;
            appliedTheme = theme;
            container.dataset.theme = theme;
            container.style.removeProperty("--ags-bg");
            node.color = pal.nodeTitle;
            node.bgcolor = pal.nodeBody;
            node.setDirtyCanvas(true, true);
            layoutToolbar();
        }

        function lineHeight() {
            return Math.round(fontSize * LINE_H);
        }

        function applyFont() {
            const lh = lineHeight();
            for (const p of ALL_PANES) {
                p.ta.style.fontSize = fontSize + "px";
                p.ta.style.lineHeight = lh + "px";
                p.hl.style.fontSize = fontSize + "px";
                p.hl.style.lineHeight = lh + "px";
                p.gutter.style.fontSize = fontSize + "px";
                p.gutter.style.lineHeight = lh + "px";
                positionHighlightsFor(p);
            }
        }

        function applyWrap() {
            for (const p of ALL_PANES) {
                p.ta.wrap = wrap ? "soft" : "off";
                p.ta.style.whiteSpace = wrap ? "pre-wrap" : "pre";
                p.hl.style.whiteSpace = wrap ? "pre-wrap" : "pre";
                p.gutter.style.display = wrap ? "none" : "";
                renderGutterFor(p);
                positionHighlightsFor(p);
            }
            wrapBtn.classList.toggle("ags-on", wrap);
        }

        function updateSplitControls() {
            splitSel.value = splitTarget;
            applyResBtn.style.display = splitTarget === "result" ? "" : "none";
            langSplitBtn.classList.toggle("ags-on", splitToResult);
        }

        function applyView() {
            const isPreview = viewMode === "preview";
            const split = viewMode === "split";
            container.classList.toggle("ags-split", split);
            editor.style.display = isPreview ? "none" : "flex";
            previewEl.style.display = isPreview ? "block" : "none";
            let shown;
            if (isPreview) {
                shown = [];
            } else if (split) {
                const left = splitTarget === "result" ? paneFor(resultSource) : paneText;
                const rightPane = splitTarget === "result" ? paneRes : paneNeg;
                shown = left === rightPane ? [left] : [left, rightPane];
            } else {
                shown = [paneFor(activeBuf)];
            }
            for (const p of ALL_PANES) {
                p.root.style.display = shown.includes(p) ? "flex" : "none";
            }
            splitBtn.classList.toggle("ags-on", split);
            tabPrev.classList.toggle("ags-on", isPreview);
            updateSplitControls();
            if (isPreview) renderPreview();
        }

        function saveProps() {
            node.properties = node.properties || {};
            node.properties.agsoft_text_editor = {
                fontSize: fontSize, wrap: wrap, hl: hlOn, theme: theme, filename: savedName,
                langFrom: langFrom.value, langTo: langTo.value,
                view: viewMode, buf: activeBuf,
                splitTarget: splitTarget, splitToResult: splitToResult, resultSource: resultSource,
                saveRoot: saveRoot.value || pendingRoot, saveSub: saveSub.value,
                saveContent: saveContent.value, saveDest: saveDest.value,
            };
        }

        function renderGutterFor(p) {
            if (wrap) return;
            const count = p.ta.value.length ? p.ta.value.split("\n").length : 1;
            if (count !== p.gutterLines) {
                const parts = new Array(count);
                for (let i = 0; i < count; i++) parts[i] = i + 1;
                p.gText.textContent = parts.join("\n");
                p.gutterLines = count;
            }
            p.gutter.style.width = "calc(" + String(count).length + "ch + 18px)";
        }

        function positionHighlightsFor(p) {
            if (wrap) {
                p.lineHl.style.display = "none";
                p.gHl.style.display = "none";
                return;
            }
            const caret = p.ta.selectionDirection === "backward"
                ? p.ta.selectionStart : p.ta.selectionEnd;
            const line = p.ta.value.slice(0, caret || 0).split("\n").length;
            const lh = lineHeight();
            const top = PAD_TOP + (line - 1) * lh - p.ta.scrollTop;
            p.lineHl.style.display = "block";
            p.lineHl.style.top = top + "px";
            p.lineHl.style.height = lh + "px";
            p.gHl.style.display = "block";
            p.gHl.style.top = top + "px";
            p.gHl.style.height = lh + "px";
        }

        function syncMetricsFor(p) {
            p.hl.style.width = p.ta.clientWidth + "px";
            p.hl.style.height = p.ta.clientHeight + "px";
            p.hl.scrollTop = p.ta.scrollTop;
            p.hl.scrollLeft = p.ta.scrollLeft;
        }

        function renderOverlayFor(p) {
            const text = p.ta.value;
            if (!hlOn || text.length > HL_LIMIT) {
                p.hl.style.display = "none";
                p.ta.classList.remove("ags-ghost");
                return;
            }
            p.hl.style.display = "";
            p.ta.classList.add("ags-ghost");
            const n = text.length;
            const cls = new Uint8Array(n);
            SYN_RE.lastIndex = 0;
            let m;
            while ((m = SYN_RE.exec(text)) !== null) {
                if (!m[0].length) {
                    SYN_RE.lastIndex++;
                    continue;
                }
                let group = 0;
                for (let i = 1; i < m.length; i++) {
                    if (m[i] !== undefined) {
                        group = i;
                        break;
                    }
                }
                if (!group) continue;
                for (let i = m.index; i < m.index + m[0].length; i++) cls[i] = group;
            }
            const mark = new Uint8Array(n);
            if (p === pane) {
                for (let i = 0; i < matches.length; i++) {
                    const value = i === matchIndex ? 2 : 1;
                    for (let k = matches[i].start; k < matches[i].end; k++) mark[k] = value;
                }
            }
            let html = "";
            let i = 0;
            while (i < n) {
                const c = cls[i];
                const k = mark[i];
                let j = i + 1;
                while (j < n && cls[j] === c && mark[j] === k) j++;
                const chunk = escapeHtml(text.slice(i, j));
                if (c || k) {
                    let name = c ? SYN_CLASSES[c] : "";
                    if (k === 1) name += " ags-m";
                    if (k === 2) name += " ags-mc";
                    html += "<span class=\"" + name.trim() + "\">" + chunk + "</span>";
                } else {
                    html += chunk;
                }
                i = j;
            }
            p.hl.innerHTML = html + "\n";
        }

        function issueLabel(issue, text) {
            const line = text.slice(0, issue.pos).split("\n").length;
            if (issue.kind === "weight") {
                return "Weight above 2.0 at line " + line + "\n---\nВес выше 2.0 на строке " + line;
            }
            return "Unbalanced bracket at line " + line + "\n---\nНесбалансированная скобка на строке " + line;
        }

        function updateLangScope() {
            const selN = ta.selectionEnd - ta.selectionStart;
            const total = selN > 0 ? selN : ta.value.length;
            const what = selN > 0 ? "Selection / Выделение" : "Whole buffer / Весь буфер";
            langScope.textContent = what + ": " + total + " / " + TRANSLATE_LIMIT;
            langScope.classList.toggle("ags-bad", total > TRANSLATE_LIMIT);
        }

        function setLangStatus(kind, message) {
            langStatus.className = "ags-te-lang-status" + (kind ? " ags-" + kind : "");
            langStatus.textContent = message;
        }

        function updatePaneHeads() {
            const compare = viewMode === "split" && splitTarget === "result";
            const source = paneFor(resultSource);
            for (const p of ALL_PANES) {
                let label = "Text / Текст";
                if (p === paneNeg) label = "Negative / Негатив";
                if (p === paneRes) label = "Result / Результат";
                if (compare && p === source) label = "Original / Оригинал";
                p.head.textContent = label + " · " + p.ta.value.length;
            }
        }

        function updateStatus() {
            const value = ta.value;
            const start = ta.selectionStart || 0;
            const end = ta.selectionEnd || 0;
            const caret = ta.selectionDirection === "backward" ? start : end;
            const upto = value.slice(0, caret);
            const curLine = upto.split("\n").length;
            const col = caret - upto.lastIndexOf("\n");
            const totalLines = value.length ? value.split("\n").length : 1;
            const trimmed = value.trim();
            const words = trimmed ? trimmed.split(/\s+/).length : 0;
            const selN = end - start;
            issues = findIssues(value);
            segCur.textContent = "Ln/Стр " + curLine + ", Col/Стлб " + col;
            segSel.textContent = selN > 0 ? "Sel/Выд " + selN : "";
            segSel.style.display = selN > 0 ? "" : "none";
            segWarn.textContent = issues.length ? "⚠ " + issues.length : "";
            segWarn.style.display = issues.length ? "" : "none";
            segWarn.title = issues.length ? issueLabel(issues[0], value) : "";
            segLines.textContent = "Lines/Строк " + totalLines;
            segChars.textContent = "Chars/Симв " + value.length;
            segWords.textContent = "Words/Слов " + words;
            segEol.textContent = value.includes("\r\n") ? "CRLF" : (value.includes("\r") ? "CR" : "LF");
            updatePaneHeads();
            updateFileTag();
            if (langPanel.classList.contains("ags-open")) updateLangScope();
        }

        function renderPreview() {
            const src = ta.value;
            const trimmed = src.trim();
            if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
                try {
                    const obj = JSON.parse(trimmed);
                    previewEl.innerHTML = "";
                    previewEl.appendChild(jsonTree(obj, activeBuf, 0));
                    return;
                } catch (err) {
                    previewEl.innerHTML = "<p class=\"ags-jt-null\">JSON: " +
                        escapeHtml(err.message) + "</p>" + markdownToHtml(src);
                    return;
                }
            }
            previewEl.innerHTML = markdownToHtml(src);
        }

        function renderAll() {
            updateStatus();
            for (const p of ALL_PANES) {
                if (p.root.style.display === "none") continue;
                renderGutterFor(p);
                renderOverlayFor(p);
                syncMetricsFor(p);
                p.gutter.scrollTop = p.ta.scrollTop;
                positionHighlightsFor(p);
            }
            if (viewMode === "preview") renderPreview();
        }

        function scheduleRender() {
            if (rafId) return;
            rafId = requestAnimationFrame(() => {
                rafId = 0;
                renderAll();
            });
        }

        function packValue() {
            const body = paneText.ta.value;
            const negative = paneNeg.ta.value;
            const result = paneRes.ta.value;
            let value = body;
            if (negative || result || pathValue) value += NEG_MARK + negative;
            if (result || pathValue) value += RESULT_MARK + result;
            if (pathValue) value += PATH_MARK + pathValue;
            return value;
        }

        function syncWidget() {
            const packed = packValue();
            if (textWidget.value !== packed) textWidget.value = packed;
        }

        function updateUndoButtons() {
            undoBtn.disabled = !undoStack.length;
            redoBtn.disabled = !redoStack.length;
        }

        function pushUndoFor(p) {
            const stack = bufs[p.buf].undo;
            stack.push({ t: p.ta.value, s: p.ta.selectionStart, e: p.ta.selectionEnd });
            if (stack.length > UNDO_LIMIT) stack.shift();
            bufs[p.buf].redo.length = 0;
        }

        function pushUndo(force) {
            const now = Date.now();
            if (!force && undoStack.length && now - lastPush < 450) {
                lastPush = now;
                return;
            }
            pushUndoFor(pane);
            lastPush = now;
            updateUndoButtons();
        }

        function doUndo() {
            if (!undoStack.length) return;
            redoStack.push({ t: ta.value, s: ta.selectionStart, e: ta.selectionEnd });
            const st = undoStack.pop();
            ta.value = st.t;
            ta.setSelectionRange(st.s, st.e);
            pane.gutterLines = -1;
            afterEdit();
        }

        function doRedo() {
            if (!redoStack.length) return;
            undoStack.push({ t: ta.value, s: ta.selectionStart, e: ta.selectionEnd });
            const st = redoStack.pop();
            ta.value = st.t;
            ta.setSelectionRange(st.s, st.e);
            pane.gutterLines = -1;
            afterEdit();
        }

        function afterEdit() {
            syncWidget();
            recomputeMatches();
            scheduleRender();
            requestTokens();
            updateUndoButtons();
            scheduleAutosave();
        }

        function scheduleAutosave() {
            if (!autosaveOk) return;
            segAuto.textContent = "AS …";
            if (autosaveTimer) clearTimeout(autosaveTimer);
            autosaveTimer = setTimeout(writeAutosave, AUTOSAVE_DELAY);
        }

        function writeAutosave() {
            if (!autosaveOk) return;
            try {
                const payload = {
                    v: 2,
                    time: Date.now(),
                    text: paneText.ta.value,
                    negative: paneNeg.ta.value,
                    result: paneRes.ta.value,
                    filepath: pathValue,
                    filename: savedName,
                    theme: theme,
                    fontSize: fontSize,
                    wrap: wrap,
                    hl: hlOn,
                    view: viewMode,
                    buf: activeBuf,
                    splitTarget: splitTarget,
                    splitToResult: splitToResult,
                    resultSource: resultSource,
                    langFrom: langFrom.value,
                    langTo: langTo.value,
                };
                localStorage.setItem(autosaveKey(), JSON.stringify(payload));
                segAuto.textContent = "AS ✓";
                segAuto.title = "Autosaved " + new Date().toLocaleTimeString() +
                    "\n---\nАвтосохранение " + new Date().toLocaleTimeString();
            } catch (err) {
                autosaveOk = false;
                segAuto.textContent = "AS ✕";
                console.warn("[AGSoft Text Editor] Autosave disabled: " +
                    String((err && err.message) || err));
                flash("Autosave unavailable / Автосохранение недоступно", true);
            }
        }

        function readAutosave() {
            try {
                const raw = localStorage.getItem(autosaveKey());
                if (!raw) return null;
                const data = JSON.parse(raw);
                return data && typeof data === "object" ? data : null;
            } catch (err) {
                return null;
            }
        }

        let restorePayload = null;

        function checkAutosave(force) {
            const data = readAutosave();
            if (!data) {
                if (force) flash("No autosave yet / Автосохранения пока нет", true);
                return;
            }
            const same = String(data.text || "") === paneText.ta.value &&
                String(data.negative || "") === paneNeg.ta.value &&
                String(data.result || "") === paneRes.ta.value &&
                normalizePathValue(data.filepath) === pathValue;
            if (same && !force) return;
            restorePayload = data;
            restNote.textContent = new Date(Number(data.time || 0)).toLocaleString() +
                " · Text " + String(data.text || "").length +
                " · Negative " + String(data.negative || "").length +
                " · Result " + String(data.result || "").length +
                "\n---\nТекущий текст из workflow останется, если выбрать «Keep current».";
            restoreDialog.style.display = "block";
        }

        function applyAutosave(data) {
            for (const p of ALL_PANES) pushUndoFor(p);
            paneText.ta.value = String(data.text || "");
            paneNeg.ta.value = String(data.negative || "");
            paneRes.ta.value = String(data.result || "");
            for (const p of ALL_PANES) p.gutterLines = -1;
            pathValue = normalizePathValue(data.filepath);
            savedName = String(data.filename || savedName);
            fontSize = Math.min(FONT_MAX, Math.max(FONT_MIN, Number(data.fontSize) || 13));
            wrap = !!data.wrap;
            hlOn = data.hl !== false;
            theme = THEMES[data.theme] ? data.theme : "ink";
            viewMode = ["single", "split", "preview"].includes(data.view) ? data.view : "single";
            splitTarget = data.splitTarget === "result" ? "result" : "negative";
            splitToResult = data.splitToResult !== false;
            resultSource = data.resultSource === "negative" ? "negative" : "text";
            langFrom.value = LANGS[data.langFrom] ? data.langFrom : "ru";
            langTo.value = LANGS[data.langTo] ? data.langTo : "en";
            themeSel.value = theme;
            hlBtn.classList.toggle("ags-on", hlOn);
            const wanted = ["text", "negative", "result"].includes(data.buf) ? data.buf : "text";
            bindPane(paneFor(wanted));
            appliedTheme = "";
            applyTheme();
            applyFont();
            applyWrap();
            applyView();
            updateFileTag();
            saveProps();
            afterEdit();
            flash("Autosave restored / Автосохранение восстановлено");
        }

        function requestTokens() {
            if (tokTimer) clearTimeout(tokTimer);
            tokTimer = setTimeout(async () => {
                const seq = ++tokSeq;
                const text = ta.value;
                if (!text.trim()) {
                    segTok.textContent = "Tok 0/0";
                    segTok.classList.remove("ags-warn");
                    return;
                }
                try {
                    const res = await api.fetchApi("/agsoft/text_editor/tokenize", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: text }),
                    });
                    const data = await res.json();
                    if (seq !== tokSeq) return;
                    const sign = data.exact ? "" : "≈";
                    segTok.textContent = "Tok " + sign + data.tokens + "/" + data.chunks;
                    segTok.classList.toggle("ags-warn", data.chunks > 1);
                } catch (err) {
                    segTok.textContent = "Tok —";
                }
            }, 500);
        }

        function scrollToPos(pos) {
            const lh = lineHeight();
            const line = ta.value.slice(0, pos).split("\n").length - 1;
            ta.scrollTop = Math.max(0, line * lh - ta.clientHeight / 2 + lh);
            gutter.scrollTop = ta.scrollTop;
            hl.scrollTop = ta.scrollTop;
        }

        function gotoLine(num) {
            const rows = ta.value.split("\n");
            const target = Math.min(Math.max(1, num || 1), rows.length);
            let pos = 0;
            for (let i = 0; i < target - 1; i++) pos += rows[i].length + 1;
            ta.focus();
            ta.setSelectionRange(pos, pos);
            scrollToPos(pos);
            renderAll();
        }

        function buildSearchRe() {
            let src = findInput.value;
            if (!optRegex) src = src.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
            if (optWhole) src = "\\b(?:" + src + ")\\b";
            return new RegExp(src, optCase ? "g" : "gi");
        }

        function updateSearchCount() {
            countEl.textContent = (matchIndex >= 0 ? matchIndex + 1 : 0) + "/" + matches.length;
            countEl.classList.toggle("ags-bad", searchBad);
        }

        function recomputeMatches() {
            matches = [];
            searchBad = false;
            if (!searchOpen || !findInput.value) {
                matchIndex = -1;
                updateSearchCount();
                return;
            }
            let re;
            try {
                re = buildSearchRe();
            } catch (err) {
                searchBad = true;
                matchIndex = -1;
                updateSearchCount();
                return;
            }
            const text = ta.value;
            let m;
            while ((m = re.exec(text)) !== null) {
                if (!m[0].length) {
                    re.lastIndex++;
                    continue;
                }
                matches.push({ start: m.index, end: m.index + m[0].length });
                if (matches.length >= MATCH_LIMIT) break;
            }
            if (matchIndex >= matches.length) matchIndex = matches.length - 1;
            if (matchIndex < 0 && matches.length) matchIndex = 0;
            updateSearchCount();
        }

        function gotoMatch(index) {
            if (!matches.length) return;
            matchIndex = (index + matches.length) % matches.length;
            const m = matches[matchIndex];
            ta.focus();
            ta.setSelectionRange(m.start, m.end);
            scrollToPos(m.start);
            updateSearchCount();
            scheduleRender();
        }

        function openSearch(withReplace) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            searchOpen = true;
            searchBar.classList.add("ags-open");
            findBtn.classList.add("ags-on");
            const sel = ta.value.slice(ta.selectionStart, ta.selectionEnd);
            if (sel && !sel.includes("\n")) findInput.value = sel;
            recomputeMatches();
            scheduleRender();
            if (withReplace) replInput.focus();
            else {
                findInput.focus();
                findInput.select();
            }
        }

        function hideSearch() {
            searchOpen = false;
            searchBar.classList.remove("ags-open");
            findBtn.classList.remove("ags-on");
            matches = [];
            matchIndex = -1;
            updateSearchCount();
            scheduleRender();
            ta.focus();
        }

        function replaceCurrent() {
            if (matchIndex < 0 || !matches.length) return;
            const m = matches[matchIndex];
            pushUndo(true);
            let value = replInput.value;
            if (optRegex) {
                try {
                    const single = new RegExp(buildSearchRe().source, optCase ? "" : "i");
                    value = ta.value.slice(m.start, m.end).replace(single, replInput.value);
                } catch (err) {
                    value = replInput.value;
                }
            }
            ta.setRangeText(value, m.start, m.end, "end");
            afterEdit();
            if (matches.length) gotoMatch(Math.min(matchIndex, matches.length - 1));
        }

        function replaceAll() {
            if (!matches.length) return;
            const total = matches.length;
            const text = ta.value;
            pushUndo(true);
            let out = "";
            if (optRegex) {
                try {
                    out = text.replace(buildSearchRe(), replInput.value);
                } catch (err) {
                    out = text;
                }
            } else {
                let last = 0;
                for (const m of matches) {
                    out += text.slice(last, m.start) + replInput.value;
                    last = m.end;
                }
                out += text.slice(last);
            }
            ta.value = out;
            pane.gutterLines = -1;
            matchIndex = -1;
            afterEdit();
            flash("Replaced / Заменено: " + total);
        }

        function focusBuf(name) {
            bindPane(paneFor(name));
            if (viewMode === "preview") viewMode = "single";
            if (viewMode === "split" && splitTarget === "result") {
                const left = paneFor(resultSource);
                if (pane !== left && pane !== paneRes) splitTarget = "negative";
            }
            applyView();
            recomputeMatches();
            renderAll();
            updateUndoButtons();
            requestTokens();
            saveProps();
            ta.focus();
        }

        function setSplit(on) {
            viewMode = on ? "split" : "single";
            if (on && splitTarget === "result" && activeBuf === "negative") {
                resultSource = "negative";
            }
            applyView();
            renderAll();
            saveProps();
            scheduleAutosave();
            layoutToolbar();
        }

        function setSplitTarget(value) {
            splitTarget = value === "result" ? "result" : "negative";
            if (splitTarget === "result") {
                resultSource = activeBuf === "negative" ? "negative" : "text";
                bindPane(paneRes);
            } else if (activeBuf === "result") {
                bindPane(paneText);
            }
            viewMode = "split";
            applyView();
            recomputeMatches();
            renderAll();
            updateUndoButtons();
            saveProps();
            scheduleAutosave();
        }

        function setZen(on) {
            if (on === zenMode) return;
            zenMode = on;
            if (on) {
                zenParent = container.parentNode;
                zenNext = container.nextSibling;
                document.body.appendChild(container);
                container.classList.add("ags-zen");
                document.body.classList.add("ags-zen-open");
            } else {
                container.classList.remove("ags-zen");
                document.body.classList.remove("ags-zen-open");
                if (zenParent) zenParent.insertBefore(container, zenNext || null);
                zenParent = null;
                zenNext = null;
            }
            zenBtn.classList.toggle("ags-on", on);
            zenBtn.title = on ? "Exit Zen (Esc)\n---\nВыйти из Zen (Esc)"
                : "Zen fullscreen, Esc to exit\n---\nZen на весь экран, Esc — выход";
            closePopovers();
            layoutToolbar();
            scheduleRender();
            node.setDirtyCanvas(true, true);
        }

        function applyTransform(fn) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            pushUndo(true);
            const s = ta.selectionStart;
            const e = ta.selectionEnd;
            const hasSel = e > s;
            const src = hasSel ? ta.value.slice(s, e) : ta.value;
            const out = fn(src);
            if (hasSel) ta.setRangeText(out, s, e, "select");
            else ta.value = out;
            pane.gutterLines = -1;
            afterEdit();
            ta.focus();
        }

        // A result can land beside the original instead of replacing it.
        // Результат может встать рядом с оригиналом вместо его замены.
        function deliverResult(out, label) {
            if (!splitToResult) return false;
            resultSource = activeBuf === "negative" ? "negative" : "text";
            pushUndoFor(paneRes);
            paneRes.ta.value = out;
            paneRes.ta.setSelectionRange(0, 0);
            paneRes.gutterLines = -1;
            splitTarget = "result";
            viewMode = "split";
            bindPane(paneRes);
            applyView();
            afterEdit();
            saveProps();
            paneRes.ta.focus();
            flash(label + " → Result / Результат");
            return true;
        }

        function applyResultToSource() {
            const body = paneRes.ta.value;
            if (!body) {
                flash("Result is empty / Результат пуст", true);
                return;
            }
            const target = paneFor(resultSource);
            pushUndoFor(target);
            target.ta.value = body;
            target.ta.setSelectionRange(0, 0);
            target.gutterLines = -1;
            bindPane(target);
            applyView();
            afterEdit();
            saveProps();
            target.ta.focus();
            flash("Result applied / Результат применён");
        }

        function askConfirm(title, note, okLabel, cb) {
            confTitle.textContent = title;
            confNote.textContent = note;
            confOk.textContent = okLabel;
            confirmCallback = cb;
            confirmDialog.style.display = "block";
            setTimeout(() => confOk.focus(), 0);
        }

        function closeConfirm(run) {
            confirmDialog.style.display = "none";
            const cb = confirmCallback;
            confirmCallback = null;
            if (run && cb) cb();
            else ta.focus();
        }

        // A fresh document must not keep the previous file attached.
        // Новый документ не должен оставаться привязанным к прежнему файлу.
        function startNewDocument() {
            for (const p of ALL_PANES) {
                pushUndoFor(p);
                p.ta.value = "";
                p.ta.setSelectionRange(0, 0);
                p.gutterLines = -1;
                p.ta.scrollTop = 0;
            }
            pathValue = "";
            savedName = "prompt.txt";
            resultSource = "text";
            splitTarget = "negative";
            viewMode = "single";
            fileLabel = "";
            bindPane(paneText);
            applyView();
            afterEdit();
            saveProps();
            updateFileTag();
            paneText.ta.focus();
            flash("New document / Новый документ");
        }

        function requestNewDocument() {
            const busy = paneText.ta.value.length || paneNeg.ta.value.length ||
                paneRes.ta.value.length;
            if (!busy && !pathValue) {
                startNewDocument();
                return;
            }
            askConfirm("New document / Новый документ",
                "Text, Negative, Result and the file path will be cleared. " +
                "Ctrl+Z brings the text back." +
                "\n---\nТекст, Негатив, Результат и путь к файлу будут очищены. " +
                "Ctrl+Z вернёт текст.",
                "Create / Создать", startNewDocument);
        }

        // Emoji goes to the stored caret, the picker keeps focus.
        // Эмодзи идёт в сохранённую позицию каретки, пикер остаётся в фокусе.
        function insertEmoji(char) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            pushUndo(true);
            ta.setRangeText(char, ta.selectionStart, ta.selectionEnd, "end");
            pane.gutterLines = -1;
            afterEdit();
        }

        function openEmojiPanel() {
            const wasOpen = emojiPanel.classList.contains("ags-open");
            closePopovers();
            if (wasOpen) return;
            emojiPanel.classList.add("ags-open");
            emojiBtn.classList.add("ags-on");
            if (!emojiApi) {
                emojiApi = mountEmojiPicker(emojiPanel, {
                    onPick: insertEmoji,
                    onClose: closePopovers,
                });
            } else {
                emojiApi.refresh();
            }
            emojiApi.focus();
        }

        async function writeClipboard(text) {
            try {
                await navigator.clipboard.writeText(text);
                return true;
            } catch (err) {
                const tmp = document.createElement("textarea");
                tmp.value = text;
                tmp.style.position = "fixed";
                tmp.style.opacity = "0";
                document.body.appendChild(tmp);
                tmp.select();
                let ok = false;
                try {
                    ok = document.execCommand("copy");
                } catch (err2) {
                    ok = false;
                }
                tmp.remove();
                return ok;
            }
        }

        function formatClipboard(text, mode) {
            const raw = String(text || "");
            if (mode === "line") return raw.replace(/\r?\n+/g, " ").replace(/[ \t]{2,}/g, " ").trim();
            if (mode === "trim") {
                return raw.replace(/\r\n/g, "\n").split("\n").map((l) => l.trim()).join("\n").trim();
            }
            if (mode === "tags") {
                const seen = new Set();
                const tags = raw.split(/[\n,]+/).map((t) => t.trim()).filter(Boolean);
                return tags.filter((t) => {
                    const key = t.toLowerCase();
                    if (seen.has(key)) return false;
                    seen.add(key);
                    return true;
                }).join(", ");
            }
            return raw;
        }

        function insertClipboard(text, mode) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            const value = formatClipboard(text, mode);
            pushUndo(true);
            ta.setRangeText(value, ta.selectionStart, ta.selectionEnd, "end");
            afterEdit();
            ta.focus();
            flash("Pasted / Вставлено: " + value.length);
        }

        function openPasteDialog(mode) {
            pasteMode = mode;
            pasteBox.value = "";
            pasteDialog.style.display = "block";
            setTimeout(() => pasteBox.focus(), 0);
        }

        function closePasteDialog(insert) {
            pasteDialog.style.display = "none";
            if (insert) insertClipboard(pasteBox.value, pasteMode);
            else ta.focus();
        }

        function requestPaste(mode) {
            if (navigator.clipboard && navigator.clipboard.readText) {
                navigator.clipboard.readText().then((text) => insertClipboard(text, mode))
                    .catch(() => openPasteDialog(mode));
            } else {
                openPasteDialog(mode);
            }
        }

        async function doCopy() {
            const sel = ta.value.slice(ta.selectionStart, ta.selectionEnd);
            const value = sel || ta.value;
            const ok = await writeClipboard(value);
            if (!ok) {
                flash("Copy failed / Не удалось скопировать", true);
                return;
            }
            copyBtn.classList.add("ags-ok");
            copyBtn.textContent = "✓";
            setTimeout(() => {
                copyBtn.classList.remove("ags-ok");
                copyBtn.textContent = "⧉";
            }, 900);
            flash(sel ? "Copied selection / Скопировано выделение" : "Copied all / Скопировано всё");
        }

        async function doCut() {
            if (ta.readOnly) return;
            let s = ta.selectionStart;
            let e = ta.selectionEnd;
            if (s === e) {
                const text = ta.value;
                s = text.lastIndexOf("\n", s - 1) + 1;
                e = text.indexOf("\n", s);
                e = e < 0 ? text.length : e + 1;
            }
            const value = ta.value.slice(s, e);
            if (!value) return;
            const ok = await writeClipboard(value);
            if (!ok) {
                flash("Cut failed / Не удалось вырезать", true);
                return;
            }
            pushUndo(true);
            ta.setRangeText("", s, e, "end");
            afterEdit();
            ta.focus();
            flash("Cut / Вырезано: " + value.length);
        }

        function formatWeight(value) {
            return String(Math.round(value * 100) / 100);
        }

        function adjustWeight(delta) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            const text = ta.value;
            let s = ta.selectionStart;
            let e = ta.selectionEnd;
            const weightRe = /^\((.*):([+-]?\d*\.?\d+)\)$/;
            if (s === e) {
                const open = text.lastIndexOf("(", s);
                const close = text.indexOf(")", s);
                if (open >= 0 && close > open && weightRe.test(text.slice(open, close + 1))) {
                    s = open;
                    e = close + 1;
                } else {
                    const before = text.slice(0, s).match(/[\p{L}\p{N}_]+$/u);
                    const after = text.slice(s).match(/^[\p{L}\p{N}_]+/u);
                    s -= before ? before[0].length : 0;
                    e += after ? after[0].length : 0;
                    if (s === e) {
                        flash("Place the cursor on a word / Поставьте курсор на слово", true);
                        return;
                    }
                    if (text.charAt(s - 1) === "(") {
                        const tail = text.indexOf(")", e);
                        if (tail > e && weightRe.test(text.slice(s - 1, tail + 1))) {
                            s -= 1;
                            e = tail + 1;
                        }
                    }
                }
            }
            const segment = text.slice(s, e);
            const match = weightRe.exec(segment);
            const inner = match ? match[1] : segment.trim();
            if (!inner) return;
            const base = match ? parseFloat(match[2]) : 1.0;
            const next = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, base + delta));
            pushUndo(true);
            ta.setRangeText("(" + inner + ":" + formatWeight(next) + ")", s, e, "select");
            afterEdit();
            ta.focus();
            flash("Weight / Вес: " + formatWeight(next));
        }

        function applyTranslit(dir) {
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            const s = ta.selectionStart;
            const e = ta.selectionEnd;
            const hasSel = e > s;
            const source = hasSel ? ta.value.slice(s, e) : ta.value;
            if (!source) return;
            const out = dir === "ru-en" ? translitRuEn(source) : translitEnRu(source);
            if (deliverResult(out, "Translit / Транслит")) return;
            pushUndo(true);
            if (hasSel) ta.setRangeText(out, s, e, "select");
            else ta.value = out;
            pane.gutterLines = -1;
            afterEdit();
            ta.focus();
            flash("Translit ready / Транслит готов");
        }

        async function doTranslate() {
            if (translating) return;
            if (viewMode === "preview") {
                viewMode = "single";
                applyView();
            }
            if (ta.readOnly) return;
            const from = langFrom.value;
            const to = langTo.value;
            const s = ta.selectionStart;
            const e = ta.selectionEnd;
            const hasSel = e > s;
            const source = hasSel ? ta.value.slice(s, e) : ta.value;
            updateLangScope();
            if (!source.trim()) {
                setLangStatus("bad", "✕ Nothing to translate / Нечего переводить");
                flash("Nothing to translate / Нечего переводить", true);
                return;
            }
            translating = true;
            langRun.disabled = true;
            setLangStatus("busy", "Translating… / Перевод…");
            const result = await translateText(source, from, to);
            translating = false;
            langRun.disabled = false;
            if (!result.ok) {
                setLangStatus("bad", "✕ " + result.error);
                flash("Translation failed / Перевод не удался: " + result.error, true);
                return;
            }
            const pair = from.toUpperCase() + " → " + to.toUpperCase();
            setLangStatus("ok", "✓ " + result.provider + " · " + pair);
            if (deliverResult(result.text, "Translation / Перевод")) return;
            pushUndo(true);
            if (hasSel) ta.setRangeText(result.text, s, e, "select");
            else ta.value = result.text;
            pane.gutterLines = -1;
            afterEdit();
            ta.focus();
            flash("Translated via " + result.provider + " / Переведено через " + result.provider);
        }

        const rowsOf = (s) => s.replace(/\r\n/g, "\n").split("\n");
        const TOOLS = [
            ["Case / Регистр", [
                ["UPPERCASE", (s) => s.toUpperCase()],
                ["lowercase", (s) => s.toLowerCase()],
                ["Title Case", (s) => s.replace(/\S+/g, (w) =>
                    w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())],
                ["camelCase", (s) => {
                    const parts = s.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
                    return parts.map((w, i) => i === 0 ? w.toLowerCase()
                        : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join("");
                }],
            ]],
            ["Lines / Строки", [
                ["Trim lines / Обрезать пробелы", (s) => rowsOf(s).map((l) => l.trim()).join("\n")],
                ["Remove empty / Убрать пустые", (s) =>
                    rowsOf(s).filter((l) => l.trim() !== "").join("\n")],
                ["Remove duplicates / Убрать дубли", (s) => {
                    const seen = new Set();
                    return rowsOf(s).filter((l) => {
                        const key = l.trim();
                        if (seen.has(key)) return false;
                        seen.add(key);
                        return true;
                    }).join("\n");
                }],
                ["Sort A→Z / Сортировать", (s) =>
                    rowsOf(s).sort((a, b) => a.localeCompare(b)).join("\n")],
                ["Sort Z→A / Обратная сортировка", (s) =>
                    rowsOf(s).sort((a, b) => b.localeCompare(a)).join("\n")],
                ["Reverse / Перевернуть", (s) => rowsOf(s).reverse().join("\n")],
            ]],
        ];

        function addMenuGroup(parent, title, items) {
            const head = document.createElement("div");
            head.className = "ags-te-menu-title";
            head.textContent = title;
            parent.appendChild(head);
            for (const item of items) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "ags-te-menu-item";
                btn.textContent = item[0];
                btn.onclick = () => {
                    closePopovers();
                    item[1]();
                };
                parent.appendChild(btn);
            }
        }

        function buildMenu() {
            menu.innerHTML = "";
            for (const group of TOOLS) {
                addMenuGroup(menu, group[0], group[1].map((item) => [
                    item[0], () => applyTransform(item[1]),
                ]));
            }
            addMenuGroup(menu, "Weight / Вес", [
                ["Weight +0.1 (Ctrl+↑)", () => adjustWeight(WEIGHT_STEP)],
                ["Weight −0.1 (Ctrl+↓)", () => adjustWeight(-WEIGHT_STEP)],
            ]);
            addMenuGroup(menu, "Translit / Транслит (offline)", [
                ["Translit RU→EN", () => applyTranslit("ru-en")],
                ["Translit EN→RU", () => applyTranslit("en-ru")],
                ["Results to split / Результаты в сплит", () => {
                    splitToResult = !splitToResult;
                    updateSplitControls();
                    saveProps();
                    flash(splitToResult
                        ? "Results go to split / Результаты в сплит"
                        : "Results replace text / Результаты заменяют текст");
                }],
            ]);
            addMenuGroup(menu, "View / Вид", [
                ["New document / Новый документ", () => requestNewDocument()],
                ["Zen fullscreen / На весь экран", () => setZen(!zenMode)],
                ["Split Text | Negative", () => {
                    splitTarget = "negative";
                    setSplit(true);
                }],
                ["Split Original | Result", () => setSplitTarget("result")],
                ["Apply result / Применить результат", () => applyResultToSource()],
                ["Restore autosave / Восстановить автосохранение", () => checkAutosave(true)],
            ]);
            addMenuGroup(menu, "Files / Файлы", [
                ["Open by path / Открыть по пути", () => openPathDialog()],
                ["Import workflow texts / Тексты workflow", () => wfInput.click()],
                ["Export Markdown / Экспорт Markdown", () => openSaveDialog("md")],
                ["Export JSON / Экспорт JSON", () => openSaveDialog("json")],
            ]);
            addMenuGroup(menu, "Format / Формат", [
                ["Wrap at N columns / Перенос по N колонок", () => {
                    askValue("Wrap columns / Колонок в строке", "80", (val) => {
                        const width = Math.max(8, Math.min(400, parseInt(val, 10) || 80));
                        applyTransform((s) => rowsOf(s).map((line) => {
                            if (line.length <= width) return line;
                            const words = line.split(" ");
                            const out = [];
                            let cur = "";
                            for (const word of words) {
                                if (!cur.length) cur = word;
                                else if (cur.length + 1 + word.length <= width) cur += " " + word;
                                else {
                                    out.push(cur);
                                    cur = word;
                                }
                            }
                            if (cur.length) out.push(cur);
                            return out.join("\n");
                        }).join("\n"));
                    });
                }],
                ["JSON pretty / Форматировать JSON", () => {
                    applyTransform((s) => {
                        try {
                            return JSON.stringify(JSON.parse(s), null, 4);
                        } catch (err) {
                            flash("JSON: " + err.message, true);
                            return s;
                        }
                    });
                }],
            ]);
        }

        function buildPasteMenu() {
            pasteMenu.innerHTML = "";
            addMenuGroup(pasteMenu, "Paste special / Спецвставка", [
                ["Paste / Вставить", () => requestPaste("normal")],
                ["Paste as one line / Одной строкой", () => requestPaste("line")],
                ["Paste trimmed / С обрезкой пробелов", () => requestPaste("trim")],
                ["Paste as tag list / Списком тегов", () => requestPaste("tags")],
            ]);
        }

        function buildRecentMenu() {
            recentMenu.innerHTML = "";
            const head = document.createElement("div");
            head.className = "ags-te-menu-title";
            head.textContent = "Recent files / Последние файлы";
            recentMenu.appendChild(head);
            const list = readRecent();
            if (!list.length) {
                const empty = document.createElement("div");
                empty.className = "ags-te-empty";
                empty.textContent = "Nothing opened yet / Пока ничего не открыто";
                recentMenu.appendChild(empty);
                return;
            }
            const loadItem = (item, bufName) => {
                closePopovers();
                if (!item.full) {
                    flash("Pick the file again / Выберите файл снова", true);
                    fileInput.click();
                    return;
                }
                pathValue = normalizePathValue(item.path);
                savedName = item.name || savedName;
                loadTextInto(bufName, item.text, item.name);
                updateFileTag();
                saveProps();
            };
            for (const item of list) {
                const row = document.createElement("div");
                row.className = "ags-te-row";
                const main = document.createElement("div");
                main.className = "ags-te-row-main";
                const name = document.createElement("div");
                name.className = "ags-te-row-name";
                name.textContent = item.name;
                const sub = document.createElement("div");
                sub.className = "ags-te-row-sub";
                sub.textContent = item.kind + " · " + item.size + " ch · " + timeAgo(item.time) +
                    (item.path ? " · " + item.path : " · no path");
                sub.title = item.path || "No path / Путь неизвестен";
                main.append(name, sub);
                main.onclick = () => loadItem(item, activeBuf);
                const toNeg = mkBtn("→N", "Load into Negative\n---\nЗагрузить в Negative");
                toNeg.onclick = () => loadItem(item, "negative");
                const del = mkBtn("✕", "Remove from the list\n---\nУбрать из списка", "ags-danger");
                del.onclick = () => {
                    writeRecent(readRecent().filter((it) =>
                        !(it.name === item.name && it.time === item.time)));
                    buildRecentMenu();
                };
                const btns = document.createElement("div");
                btns.className = "ags-te-row-btns";
                btns.append(toNeg, del);
                row.append(main, btns);
                recentMenu.appendChild(row);
            }
            const clear = document.createElement("button");
            clear.type = "button";
            clear.className = "ags-te-menu-item";
            clear.textContent = "Clear list / Очистить список";
            clear.onclick = () => {
                writeRecent([]);
                buildRecentMenu();
            };
            recentMenu.appendChild(clear);
        }

        function closePopovers() {
            menu.classList.remove("ags-open");
            overflowPanel.classList.remove("ags-open");
            pasteMenu.classList.remove("ags-open");
            recentMenu.classList.remove("ags-open");
            langPanel.classList.remove("ags-open");
            emojiPanel.classList.remove("ags-open");
            emojiBtn.classList.remove("ags-on");
        }

        function layoutToolbar() {
            const available = bar.clientWidth;
            if (!available) return;
            if (!overflowMode) {
                const need = primaryGroup.scrollWidth + secondaryGroup.scrollWidth +
                    moreBtn.offsetWidth + 18;
                if (need > available) {
                    overflowMode = true;
                    overflowPanel.appendChild(secondaryGroup);
                    secondaryGroup.classList.add("ags-in-overflow");
                    moreBtn.style.display = "";
                }
                return;
            }
            secondaryGroup.classList.remove("ags-in-overflow");
            secondaryGroup.style.position = "absolute";
            secondaryGroup.style.visibility = "hidden";
            bar.insertBefore(secondaryGroup, moreBtn);
            const need = primaryGroup.scrollWidth + secondaryGroup.scrollWidth +
                moreBtn.offsetWidth + 18;
            if (need <= available) {
                overflowMode = false;
                secondaryGroup.style.position = "";
                secondaryGroup.style.visibility = "";
                moreBtn.style.display = "none";
                overflowPanel.classList.remove("ags-open");
            } else {
                overflowPanel.appendChild(secondaryGroup);
                secondaryGroup.classList.add("ags-in-overflow");
                secondaryGroup.style.position = "";
                secondaryGroup.style.visibility = "";
            }
        }

        function askValue(title, value, cb) {
            dlgTitle.textContent = title;
            dlgInput.value = value;
            dlgCallback = cb;
            dialog.style.display = "block";
            setTimeout(() => {
                dlgInput.focus();
                dlgInput.select();
            }, 0);
        }

        function closeDialog(run) {
            dialog.style.display = "none";
            const cb = dlgCallback;
            dlgCallback = null;
            if (run && cb) cb(dlgInput.value);
            else ta.focus();
        }

        function hasLink(name) {
            for (const inp of node.inputs || []) {
                if (inp.name === name && inp.link != null) return true;
            }
            return false;
        }

        function refreshLinkState() {
            const linked = hasLink("text");
            textWidget.hidden = !linked;
            textWidget.y = linked ? 0 : -9999;
            textWidget.type = linked ? origType : "hidden";
            for (const p of ALL_PANES) p.ta.readOnly = linked;
            badge.style.display = linked ? "" : "none";
            container.classList.toggle("ags-linked", linked);
            node.setDirtyCanvas(true, true);
        }

        function loadTextInto(bufName, text, label) {
            const p = paneFor(bufName);
            pushUndoFor(p);
            p.ta.value = String(text || "");
            p.ta.setSelectionRange(0, 0);
            p.gutterLines = -1;
            bindPane(p);
            if (viewMode === "preview") viewMode = "single";
            if (viewMode === "split" && splitTarget === "result") splitTarget = "negative";
            applyView();
            afterEdit();
            flash("Loaded / Загружено → " + bufName + ": " + label);
        }

        // An opened document is stored in temp, so its path is real but short-lived.
        // Открытый документ кладётся в temp, поэтому путь реальный, но временный.
        async function uploadDocument(file) {
            const form = new FormData();
            form.append("file", file, file.name || "document.txt");
            try {
                const res = await api.fetchApi("/agsoft/text_editor/upload", {
                    method: "POST",
                    body: form,
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                return normalizePathValue(data.filepath);
            } catch (err) {
                console.warn("[AGSoft Text Editor] Document upload failed: " +
                    String((err && err.message) || err));
                return "";
            }
        }

        async function resolveByName(name, size) {
            try {
                const res = await api.fetchApi("/agsoft/text_editor/resolve", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: name, size: size }),
                });
                const data = await res.json();
                if (data.path) return data.path;
                if (Array.isArray(data.matches) && data.matches.length > 1) {
                    return await choosePath(data.matches);
                }
            } catch (err) {
                console.warn("[AGSoft Text Editor] Path resolve failed: " +
                    String((err && err.message) || err));
            }
            return "";
        }

        function choosePath(items) {
            return new Promise((resolve) => {
                pathResolveFn = resolve;
                pathTitle.textContent = "Several files found / Найдено несколько файлов";
                pathNote.textContent = "Pick the file to use / Выберите нужный файл";
                pathInput.value = "";
                pathStatus.textContent = "";
                pathList.innerHTML = "";
                for (const item of items) {
                    const row = document.createElement("div");
                    row.className = "ags-te-row";
                    const main = document.createElement("div");
                    main.className = "ags-te-row-main";
                    const name = document.createElement("div");
                    name.className = "ags-te-row-name";
                    name.textContent = baseName(item);
                    const sub = document.createElement("div");
                    sub.className = "ags-te-row-sub";
                    sub.textContent = item;
                    main.append(name, sub);
                    main.onclick = () => closePathDialog(item);
                    row.appendChild(main);
                    pathList.appendChild(row);
                }
                pathList.style.display = "";
                pathDialog.style.display = "block";
            });
        }

        function openPathDialog() {
            pathResolveFn = null;
            pathTitle.textContent = "Open by absolute path / Открыть по абсолютному пути";
            pathNote.textContent = "Paste the full path to a text file." +
                "\n---\nВставьте полный путь к текстовому файлу.";
            pathInput.value = pathValue || readLastFolder();
            pathStatus.className = "ags-te-lang-status";
            pathStatus.textContent = "";
            pathList.innerHTML = "";
            pathList.style.display = "none";
            pathDialog.style.display = "block";
            setTimeout(() => {
                pathInput.focus();
                pathInput.select();
            }, 0);
        }

        function closePathDialog(value) {
            pathDialog.style.display = "none";
            const fn = pathResolveFn;
            pathResolveFn = null;
            if (fn) {
                fn(value || "");
                return;
            }
            if (value) openByPath(value);
            else ta.focus();
        }

        async function openByPath(raw) {
            const value = String(raw || "").trim().replace(/^"|"$/g, "");
            if (!value) return;
            flash("Reading… / Чтение…");
            try {
                const parent = parentDir(value);
                if (parent) await addSaveFolder(parent, true);
                const res = await api.fetchApi("/agsoft/text_editor/read", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: value }),
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                const body = String(data.text || "");
                const name = baseName(data.filepath);
                if (name.toLowerCase().endsWith(".json")) {
                    let parsed = null;
                    try {
                        parsed = JSON.parse(body);
                    } catch (err) {
                        parsed = null;
                    }
                    const items = parsed ? extractWorkflowTexts(parsed) : [];
                    if (items.length) {
                        pushRecent(name, items[0].text, "workflow", data.filepath);
                        buildRecentMenu();
                        openWorkflowDialog(items, name);
                        return;
                    }
                }
                pathValue = normalizePathValue(data.filepath);
                savedName = name;
                loadTextInto(activeBuf, body, name);
                updateFileTag();
                saveProps();
                pushRecent(name, body, "text", pathValue);
                buildRecentMenu();
                flash("Opened / Открыто: " + pathValue);
            } catch (err) {
                const reason = String((err && err.message) || err);
                console.warn("[AGSoft Text Editor] Open by path failed: " + reason);
                pathStatus.className = "ags-te-lang-status ags-bad";
                pathStatus.textContent = "✕ " + reason;
                flash("Open failed / Не удалось открыть: " + reason, true);
            }
        }

        async function loadPlainFile(file, raw) {
            flash("Uploading… / Загрузка…");
            const stored = await uploadDocument(file);
            pathValue = stored;
            savedName = stored ? baseName(stored) : (file.name || savedName);
            loadTextInto(activeBuf, raw, savedName);
            updateFileTag();
            saveProps();
            pushRecent(savedName, raw, "text", pathValue);
            buildRecentMenu();
            if (pathValue) {
                flash("Opened / Открыто: " + pathValue);
            } else {
                flash("Opened without a stored copy / Открыто без копии на сервере", true);
            }
        }

        async function importPng(file) {
            const form = new FormData();
            form.append("image", file, file.name || "image.png");
            flash("Reading PNG… / Чтение PNG…");
            try {
                const res = await api.fetchApi("/agsoft/text_editor/png_meta", {
                    method: "POST",
                    body: form,
                });
                const data = await res.json();
                if (data.error) {
                    flash(data.error, true);
                    return;
                }
                if (!data.positive && !data.negative) {
                    flash("No prompt inside / В PNG нет промпта", true);
                    return;
                }
                pushUndoFor(paneText);
                pushUndoFor(paneNeg);
                paneText.ta.value = data.positive || "";
                paneNeg.ta.value = data.negative || "";
                paneText.gutterLines = -1;
                paneNeg.gutterLines = -1;
                bindPane(paneText);
                if (viewMode === "preview") viewMode = "single";
                splitTarget = "negative";
                applyView();
                afterEdit();
                pushRecent(file.name || "image.png", data.positive || "", "png", "");
                buildRecentMenu();
                flash("PNG → " + (data.source || "meta"));
            } catch (err) {
                flash("PNG import failed / Ошибка импорта PNG", true);
            }
        }

        let wfItems = [];

        function openWorkflowDialog(items, label) {
            wfItems = items;
            wfNote.textContent = label + " · " + items.length + " CLIPTextEncode";
            wfList.innerHTML = "";
            if (!items.length) {
                const empty = document.createElement("div");
                empty.className = "ags-te-empty";
                empty.textContent = "No CLIPTextEncode found / CLIPTextEncode не найдены";
                wfList.appendChild(empty);
            }
            items.forEach((item, index) => {
                const row = document.createElement("div");
                row.className = "ags-te-row";
                const main = document.createElement("div");
                main.className = "ags-te-row-main";
                const name = document.createElement("div");
                name.className = "ags-te-row-name";
                name.textContent = "#" + item.id + " · " + item.label;
                const sub = document.createElement("div");
                sub.className = "ags-te-row-sub";
                const preview = item.text.replace(/\s+/g, " ").slice(0, 90);
                sub.textContent = item.text.length + " ch · " + (preview || "(empty)");
                main.append(name, sub);
                main.onclick = () => {
                    wfDialog.style.display = "none";
                    loadTextInto(activeBuf, item.text, "#" + item.id);
                };
                const toText = mkBtn("→T", "Load into Text\n---\nЗагрузить в Text");
                toText.onclick = () => {
                    wfDialog.style.display = "none";
                    loadTextInto("text", item.text, "#" + item.id);
                };
                const toNeg = mkBtn("→N", "Load into Negative\n---\nЗагрузить в Negative");
                toNeg.onclick = () => {
                    wfDialog.style.display = "none";
                    loadTextInto("negative", item.text, "#" + item.id);
                };
                const btns = document.createElement("div");
                btns.className = "ags-te-row-btns";
                btns.append(toText, toNeg);
                row.append(main, btns);
                row.dataset.index = String(index);
                wfList.appendChild(row);
            });
            wfPair.disabled = items.length < 2;
            wfDialog.style.display = "block";
        }

        function readWorkflowFile(file) {
            const reader = new FileReader();
            reader.onload = () => {
                const raw = String(reader.result || "");
                let data = null;
                try {
                    data = JSON.parse(raw);
                } catch (err) {
                    loadPlainFile(file, raw);
                    return;
                }
                const items = extractWorkflowTexts(data);
                if (!items.length) {
                    loadPlainFile(file, raw);
                    return;
                }
                pushRecent(file.name, items[0].text, "workflow", "");
                buildRecentMenu();
                openWorkflowDialog(items, file.name);
            };
            reader.onerror = () => flash("Read error / Ошибка чтения", true);
            reader.readAsText(file, "utf-8");
        }

        function handleFiles(files) {
            if (!files || !files.length) return;
            const now = Date.now();
            if (now - lastDropAt < DROP_GUARD_MS) return;
            lastDropAt = now;
            const file = files[0];
            const name = String(file.name || "").toLowerCase();
            if (name.endsWith(".png") || file.type === "image/png") {
                importPng(file);
                return;
            }
            if (name.endsWith(".json")) {
                readWorkflowFile(file);
                return;
            }
            const okExt = /\.(txt|md|csv|log|ini|yaml|yml)$/.test(name);
            if (!okExt && !String(file.type || "").startsWith("text/")) {
                flash("Unsupported file / Неподдерживаемый файл", true);
                return;
            }
            if (file.size > 16 * 1024 * 1024) {
                flash("File too large / Файл слишком большой", true);
                return;
            }
            const reader = new FileReader();
            reader.onload = () => loadPlainFile(file, String(reader.result || ""));
            reader.onerror = () => flash("Read error / Ошибка чтения", true);
            reader.readAsText(file, "utf-8");
        }

        function buildSavePayload() {
            const kind = saveContent.value;
            const text = paneText.ta.value;
            const negative = paneNeg.ta.value;
            if (kind === "md") {
                const body = "# AGSoft Text Editor\n\n## Text\n\n" + text +
                    "\n\n## Negative\n\n" + negative +
                    "\n\n---\nSaved: " + new Date().toISOString() + "\n";
                return { body: body, ext: ".md", mime: "text/markdown" };
            }
            if (kind === "json") {
                const obj = {
                    text: text,
                    negative: negative,
                    result: paneRes.ta.value,
                    filepath: pathValue,
                    saved: new Date().toISOString(),
                    lines: text.split("\n").map((l) => l.trim()).filter(Boolean),
                };
                return { body: JSON.stringify(obj, null, 4), ext: ".json", mime: "application/json" };
            }
            return { body: text, ext: ".txt", mime: "text/plain" };
        }

        function currentSaveName() {
            const payload = buildSavePayload();
            const base = String(saveName.value || "prompt").replace(/\.(txt|md|json)$/i, "");
            return (base || "prompt") + payload.ext;
        }

        function updateRootButtons() {
            const opt = saveRoot.options[saveRoot.selectedIndex];
            const custom = !!(opt && opt.dataset && opt.dataset.custom === "1");
            rootDelBtn.disabled = !custom;
        }

        function updateSaveNote() {
            saveName.value = currentSaveName();
            const local = saveDest.value === "local";
            rootField.style.display = local ? "none" : "flex";
            subField.style.display = local ? "none" : "flex";
            saveNote.textContent = local
                ? "The browser writes the file. Its path is recovered when the folder is known." +
                    "\n---\nФайл пишет браузер. Путь определится, если папка известна."
                : "The server writes the file and returns its absolute path for filepath." +
                    "\n---\nСервер пишет файл и возвращает абсолютный путь для filepath.";
            updateRootButtons();
        }

        // Grouped list keeps ComfyUI and personal folders apart.
        // Группы отделяют папки ComfyUI от личных папок.
        async function loadRoots(force) {
            if (rootsLoaded && !force) return;
            rootsLoaded = true;
            try {
                const res = await api.fetchApi("/agsoft/text_editor/roots");
                const data = await res.json();
                if (!Array.isArray(data.roots) || !data.roots.length) return;
                const keep = pendingRoot || saveRoot.value;
                saveRoot.innerHTML = "";
                const boxes = new Map();
                for (const item of data.roots) {
                    const key = item.group || (item.custom ? "custom" : "comfy");
                    if (!boxes.has(key)) {
                        const box = document.createElement("optgroup");
                        box.label = ROOT_GROUP_LABELS[key] || key;
                        boxes.set(key, box);
                        saveRoot.appendChild(box);
                    }
                    const o = document.createElement("option");
                    o.value = item.key;
                    o.textContent = item.label;
                    o.title = item.path;
                    o.dataset.custom = item.custom ? "1" : "";
                    boxes.get(key).appendChild(o);
                }
                pendingRoot = "";
                saveRoot.value = keep;
                if (!saveRoot.value) saveRoot.value = data.roots[0].key;
                updateRootButtons();
            } catch (err) {
                console.warn("[AGSoft Text Editor] Roots list unavailable: " +
                    String((err && err.message) || err));
            }
        }

        async function addSaveFolder(rawPath, quiet) {
            const value = String(rawPath || "").trim().replace(/^"|"$/g, "");
            if (!value) return null;
            try {
                const res = await api.fetchApi("/agsoft/text_editor/add_root", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: value }),
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                writeLastFolder(data.path || value);
                rootsLoaded = false;
                return data;
            } catch (err) {
                const reason = String((err && err.message) || err);
                console.warn("[AGSoft Text Editor] Save folder not added: " + reason);
                if (!quiet) flash("Folder not added / Папка не добавлена: " + reason, true);
                return null;
            }
        }

        async function pickSaveFolder(rawPath) {
            const data = await addSaveFolder(rawPath, false);
            if (!data) return;
            await loadRoots(true);
            saveRoot.value = data.key;
            saveDest.value = "server";
            updateSaveNote();
            saveProps();
            flash("Save folder added / Папка записи добавлена: " + data.path);
        }

        async function removeSaveFolder() {
            const key = saveRoot.value;
            const opt = saveRoot.options[saveRoot.selectedIndex];
            if (!opt || opt.dataset.custom !== "1") return;
            try {
                const res = await api.fetchApi("/agsoft/text_editor/remove_root", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key: key }),
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);
                await loadRoots(true);
                updateSaveNote();
                saveProps();
                flash("Save folder removed / Папка записи убрана");
            } catch (err) {
                const reason = String((err && err.message) || err);
                flash("Folder not removed / Папка не убрана: " + reason, true);
            }
        }

        async function openSaveDialog(kind) {
            if (kind) saveContent.value = kind;
            saveName.value = savedName || "prompt.txt";
            saveStatus.className = "ags-te-lang-status";
            saveStatus.textContent = pathValue
                ? "Current path / Текущий путь: " + pathValue
                : NEW_DOCUMENT_LABEL + " — not saved yet / ещё не сохранён";
            saveDialog.style.display = "block";
            await loadRoots(false);
            updateSaveNote();
            setTimeout(() => {
                saveName.focus();
                saveName.select();
            }, 0);
        }

        // The dialog leaves the screen once the file is written / Диалог уходит после записи файла
        function closeSaveDialog() {
            saveDialog.style.display = "none";
            saveStatus.className = "ags-te-lang-status";
            saveStatus.textContent = "";
            ta.focus();
        }

        async function runSave() {
            const payload = buildSavePayload();
            const name = currentSaveName();
            saveName.value = name;
            saveRun.disabled = true;
            saveStatus.className = "ags-te-lang-status ags-busy";
            saveStatus.textContent = "Writing… / Запись…";
            let written = false;
            try {
                if (saveDest.value === "local") {
                    if (typeof window.showSaveFilePicker === "function") {
                        const handle = await window.showSaveFilePicker({
                            suggestedName: name,
                            types: [{ description: "Text", accept: { [payload.mime]: [payload.ext] } }],
                        });
                        const stream = await handle.createWritable();
                        await stream.write(payload.body);
                        await stream.close();
                        savedName = handle.name || name;
                    } else {
                        downloadBody(payload.body, name, payload.mime);
                        savedName = name;
                    }
                    const bytes = new TextEncoder().encode(payload.body).length;
                    pathValue = normalizePathValue(await resolveByName(savedName, bytes));
                    syncWidget();
                    updateFileTag();
                    pushRecent(savedName, payload.body, saveContent.value, pathValue);
                    buildRecentMenu();
                    flash(pathValue
                        ? "Saved / Сохранено: " + pathValue
                        : "Saved / Сохранено: " + savedName + " · path unknown / путь неизвестен");
                    written = true;
                } else {
                    const res = await api.fetchApi("/agsoft/text_editor/save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            filename: name,
                            text: payload.body,
                            root: saveRoot.value,
                            subfolder: saveSub.value,
                        }),
                    });
                    const data = await res.json();
                    if (data.error) throw new Error(data.error);
                    savedName = name;
                    pathValue = normalizePathValue(data.filepath);
                    syncWidget();
                    updateFileTag();
                    pushRecent(name, payload.body, saveContent.value, pathValue);
                    buildRecentMenu();
                    flash("Saved / Сохранено: " + pathValue);
                    written = true;
                }
                saveProps();
                scheduleAutosave();
            } catch (err) {
                if (err && err.name === "AbortError") {
                    saveStatus.className = "ags-te-lang-status";
                    saveStatus.textContent = "Cancelled / Отменено";
                } else {
                    const reason = String((err && err.message) || err);
                    console.warn("[AGSoft Text Editor] Save failed: " + reason);
                    saveStatus.className = "ags-te-lang-status ags-bad";
                    saveStatus.textContent = "✕ " + reason;
                    flash("Save failed / Ошибка сохранения: " + reason, true);
                }
            } finally {
                saveRun.disabled = false;
                if (written) closeSaveDialog();
            }
        }

        function onEditorKeyDown(e) {
            const ctrl = e.ctrlKey || e.metaKey;
            const key = e.key.toLowerCase();
            if (ctrl && e.altKey && key === "n") {
                e.preventDefault();
                e.stopPropagation();
                requestNewDocument();
                return;
            }
            if (e.key === "Escape") {
                if (searchOpen) {
                    e.preventDefault();
                    hideSearch();
                    return;
                }
                if (zenMode) {
                    e.preventDefault();
                    setZen(false);
                }
                return;
            }
            if (e.key === "Tab" && !ctrl) {
                e.preventDefault();
                pushUndo(true);
                ta.setRangeText("\t", ta.selectionStart, ta.selectionEnd, "end");
                afterEdit();
                return;
            }
            if (!ctrl) return;
            if (key === "arrowup") {
                e.preventDefault();
                e.stopPropagation();
                adjustWeight(WEIGHT_STEP);
            } else if (key === "arrowdown") {
                e.preventDefault();
                e.stopPropagation();
                adjustWeight(-WEIGHT_STEP);
            } else if (key === "f" || key === "h") {
                e.preventDefault();
                e.stopPropagation();
                openSearch(key === "h");
            } else if (key === "o") {
                e.preventDefault();
                e.stopPropagation();
                openPathDialog();
            } else if (key === "z" && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                doUndo();
            } else if (key === "y" || (key === "z" && e.shiftKey)) {
                e.preventDefault();
                e.stopPropagation();
                doRedo();
            } else if (key === "s") {
                e.preventDefault();
                e.stopPropagation();
                openSaveDialog("");
            }
        }

        function wirePane(p) {
            p.ta.addEventListener("focus", () => {
                if (pane !== p) {
                    bindPane(p);
                    recomputeMatches();
                    updateUndoButtons();
                    requestTokens();
                }
                scheduleRender();
            });
            p.ta.addEventListener("beforeinput", () => {
                if (pane !== p) bindPane(p);
                pushUndo(false);
            });
            p.ta.addEventListener("input", () => {
                if (pane !== p) bindPane(p);
                afterEdit();
            });
            p.ta.addEventListener("keyup", scheduleRender);
            p.ta.addEventListener("click", scheduleRender);
            p.ta.addEventListener("select", scheduleRender);
            p.ta.addEventListener("scroll", () => {
                p.gutter.scrollTop = p.ta.scrollTop;
                p.hl.scrollTop = p.ta.scrollTop;
                p.hl.scrollLeft = p.ta.scrollLeft;
                positionHighlightsFor(p);
            });
            p.ta.addEventListener("keydown", (e) => {
                if (pane !== p) bindPane(p);
                onEditorKeyDown(e);
            });
            p.head.addEventListener("pointerdown", () => focusBuf(p.buf));
        }

        for (const p of ALL_PANES) wirePane(p);

        findInput.addEventListener("input", () => {
            matchIndex = -1;
            recomputeMatches();
            scheduleRender();
        });
        findInput.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Enter") {
                e.preventDefault();
                gotoMatch(matchIndex + (e.shiftKey ? -1 : 1));
            } else if (e.key === "Escape") {
                e.preventDefault();
                hideSearch();
            }
        });
        replInput.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Enter") {
                e.preventDefault();
                replaceCurrent();
            } else if (e.key === "Escape") {
                e.preventDefault();
                hideSearch();
            }
        });
        caseBtn.onclick = () => {
            optCase = !optCase;
            caseBtn.classList.toggle("ags-on", optCase);
            recomputeMatches();
            scheduleRender();
        };
        regexBtn.onclick = () => {
            optRegex = !optRegex;
            regexBtn.classList.toggle("ags-on", optRegex);
            recomputeMatches();
            scheduleRender();
        };
        wholeBtn.onclick = () => {
            optWhole = !optWhole;
            wholeBtn.classList.toggle("ags-on", optWhole);
            recomputeMatches();
            scheduleRender();
        };
        prevBtn.onclick = () => gotoMatch(matchIndex - 1);
        nextBtn.onclick = () => gotoMatch(matchIndex + 1);
        replBtn.onclick = replaceCurrent;
        replAllBtn.onclick = replaceAll;
        closeSearch.onclick = hideSearch;
        findBtn.onclick = () => (searchOpen ? hideSearch() : openSearch(false));

        tabText.onclick = () => focusBuf("text");
        tabNeg.onclick = () => focusBuf("negative");
        tabRes.onclick = () => focusBuf("result");
        tabPrev.onclick = () => {
            viewMode = "preview";
            applyView();
            renderAll();
            saveProps();
        };
        splitSel.onchange = () => setSplitTarget(splitSel.value);
        applyResBtn.onclick = applyResultToSource;
        fileTag.onclick = async () => {
            if (!pathValue) {
                openSaveDialog("");
                return;
            }
            const ok = await writeClipboard(pathValue);
            flash(ok ? "Path copied / Путь скопирован" : "Copy failed / Не удалось скопировать", !ok);
        };

        newBtn.onclick = requestNewDocument;
        openBtn.onclick = () => fileInput.click();
        pathBtn.onclick = openPathDialog;
        pngBtn.onclick = () => pngInput.click();
        wfBtn.onclick = () => wfInput.click();
        emojiBtn.onclick = openEmojiPanel;
        fileInput.onchange = () => {
            handleFiles(fileInput.files);
            fileInput.value = "";
        };
        pngInput.onchange = () => {
            if (pngInput.files && pngInput.files.length) importPng(pngInput.files[0]);
            pngInput.value = "";
        };
        wfInput.onchange = () => {
            if (wfInput.files && wfInput.files.length) readWorkflowFile(wfInput.files[0]);
            wfInput.value = "";
        };
        saveBtn.onclick = () => openSaveDialog("");
        undoBtn.onclick = doUndo;
        redoBtn.onclick = doRedo;
        copyBtn.onclick = doCopy;
        pasteBtn.onclick = () => requestPaste("normal");
        cutBtn.onclick = doCut;
        pasteMoreBtn.onclick = () => {
            const wasOpen = pasteMenu.classList.contains("ags-open");
            closePopovers();
            if (!wasOpen) pasteMenu.classList.add("ags-open");
        };
        pasteOk.onclick = () => closePasteDialog(true);
        pasteCancel.onclick = () => closePasteDialog(false);
        pasteBox.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                closePasteDialog(false);
            }
        });
        splitBtn.onclick = () => setSplit(viewMode !== "split");
        zenBtn.onclick = () => setZen(!zenMode);
        recentBtn.onclick = () => {
            const wasOpen = recentMenu.classList.contains("ags-open");
            closePopovers();
            if (!wasOpen) {
                buildRecentMenu();
                recentMenu.classList.add("ags-open");
            }
        };
        langBtn.onclick = () => {
            const wasOpen = langPanel.classList.contains("ags-open");
            closePopovers();
            if (!wasOpen) {
                langPanel.classList.add("ags-open");
                updateSplitControls();
                updateLangScope();
            }
        };
        langSwap.onclick = () => {
            const from = langFrom.value;
            langFrom.value = langTo.value;
            langTo.value = from;
            saveProps();
            updateLangScope();
        };
        langFrom.onchange = () => {
            saveProps();
            updateLangScope();
        };
        langTo.onchange = () => {
            saveProps();
            updateLangScope();
        };
        langSplitBtn.onclick = () => {
            splitToResult = !splitToResult;
            updateSplitControls();
            saveProps();
        };
        langRun.onclick = doTranslate;
        clearBtn.onclick = () => {
            if (ta.readOnly || !ta.value.length) return;
            pushUndo(true);
            ta.value = "";
            pane.gutterLines = -1;
            afterEdit();
            ta.focus();
        };
        wrapBtn.onclick = () => {
            wrap = !wrap;
            applyWrap();
            saveProps();
            scheduleRender();
            scheduleAutosave();
            ta.focus();
        };
        minusBtn.onclick = () => {
            fontSize = Math.max(FONT_MIN, fontSize - 1);
            applyFont();
            for (const p of ALL_PANES) renderGutterFor(p);
            saveProps();
            scheduleRender();
            scheduleAutosave();
        };
        plusBtn.onclick = () => {
            fontSize = Math.min(FONT_MAX, fontSize + 1);
            applyFont();
            for (const p of ALL_PANES) renderGutterFor(p);
            saveProps();
            scheduleRender();
            scheduleAutosave();
        };
        hlBtn.onclick = () => {
            hlOn = !hlOn;
            hlBtn.classList.toggle("ags-on", hlOn);
            saveProps();
            scheduleRender();
            scheduleAutosave();
        };
        themeSel.onchange = () => {
            theme = THEMES[themeSel.value] ? themeSel.value : "ink";
            applyTheme();
            saveProps();
            scheduleRender();
            scheduleAutosave();
        };
        toolsBtn.onclick = () => {
            const wasOpen = menu.classList.contains("ags-open");
            closePopovers();
            if (!wasOpen) menu.classList.add("ags-open");
        };
        moreBtn.onclick = () => {
            const wasOpen = overflowPanel.classList.contains("ags-open");
            closePopovers();
            if (!wasOpen && overflowMode) overflowPanel.classList.add("ags-open");
        };
        dlgOk.onclick = () => closeDialog(true);
        dlgCancel.onclick = () => closeDialog(false);
        dlgInput.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Enter") {
                e.preventDefault();
                closeDialog(true);
            } else if (e.key === "Escape") {
                e.preventDefault();
                closeDialog(false);
            }
        });
        confOk.onclick = () => closeConfirm(true);
        confCancel.onclick = () => closeConfirm(false);
        confirmDialog.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                closeConfirm(false);
            } else if (e.key === "Enter") {
                e.preventDefault();
                closeConfirm(true);
            }
        });
        pathOk.onclick = () => closePathDialog(pathInput.value.trim());
        pathCancel.onclick = () => closePathDialog("");
        pathDialog.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                closePathDialog("");
            } else if (e.key === "Enter" && e.target === pathInput) {
                e.preventDefault();
                closePathDialog(pathInput.value.trim());
            }
        });
        saveContent.onchange = updateSaveNote;
        saveDest.onchange = () => {
            updateSaveNote();
            saveProps();
        };
        saveRoot.onchange = () => {
            saveProps();
            updateRootButtons();
        };
        saveSub.onchange = saveProps;
        rootAddBtn.onclick = () => {
            askValue("Absolute folder path / Абсолютный путь к папке",
                readLastFolder(), pickSaveFolder);
        };
        rootDelBtn.onclick = removeSaveFolder;
        saveRun.onclick = runSave;
        saveCancel.onclick = closeSaveDialog;
        saveDialog.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                closeSaveDialog();
            } else if (e.key === "Enter" && e.target === saveName) {
                e.preventDefault();
                runSave();
            }
        });
        wfClose.onclick = () => {
            wfDialog.style.display = "none";
            ta.focus();
        };
        wfPair.onclick = () => {
            if (wfItems.length < 2) return;
            wfDialog.style.display = "none";
            pushUndoFor(paneText);
            pushUndoFor(paneNeg);
            paneText.ta.value = wfItems[0].text;
            paneNeg.ta.value = wfItems[1].text;
            paneText.gutterLines = -1;
            paneNeg.gutterLines = -1;
            bindPane(paneText);
            splitTarget = "negative";
            viewMode = "split";
            applyView();
            afterEdit();
            flash("Pair loaded / Пара загружена");
        };
        wfDialog.addEventListener("keydown", (e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
                e.preventDefault();
                wfDialog.style.display = "none";
            }
        });
        restLoad.onclick = () => {
            restoreDialog.style.display = "none";
            if (restorePayload) applyAutosave(restorePayload);
            restorePayload = null;
        };
        restKeep.onclick = () => {
            restoreDialog.style.display = "none";
            restorePayload = null;
            ta.focus();
        };
        restDelete.onclick = () => {
            restoreDialog.style.display = "none";
            restorePayload = null;
            try {
                localStorage.removeItem(autosaveKey());
                segAuto.textContent = "AS —";
            } catch (err) {
                console.warn("[AGSoft Text Editor] Autosave removal failed: " +
                    String((err && err.message) || err));
            }
            flash("Autosave removed / Автосохранение удалено");
        };
        segCur.onclick = () => askValue("Go to line / Перейти к строке", "1", (val) => {
            gotoLine(parseInt(val, 10));
        });
        segWarn.onclick = () => {
            if (!issues.length) return;
            const pos = issues[0].pos;
            ta.focus();
            ta.setSelectionRange(pos, pos + 1);
            scrollToPos(pos);
            renderAll();
        };
        segAuto.onclick = () => checkAutosave(true);
        segTok.onclick = requestTokens;

        const onDocDown = (e) => {
            const inside = menu.contains(e.target) || overflowPanel.contains(e.target) ||
                pasteMenu.contains(e.target) || recentMenu.contains(e.target) ||
                langPanel.contains(e.target) || emojiPanel.contains(e.target) ||
                e.target === toolsBtn || e.target === moreBtn || e.target === pasteMoreBtn ||
                e.target === langBtn || e.target === recentBtn || e.target === emojiBtn;
            if (!inside) closePopovers();
        };
        const onDocKey = (e) => {
            if (e.key !== "Escape") return;
            if (zenMode && !container.contains(document.activeElement)) {
                setZen(false);
                return;
            }
            closePopovers();
        };
        document.addEventListener("pointerdown", onDocDown);
        document.addEventListener("keydown", onDocKey);

        container.addEventListener("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.dataTransfer.dropEffect = "copy";
            dropHint = true;
            node.setDirtyCanvas(true, false);
        });
        container.addEventListener("dragleave", (e) => {
            e.stopPropagation();
            dropHint = false;
            node.setDirtyCanvas(true, false);
        });
        container.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropHint = false;
            node.setDirtyCanvas(true, false);
            handleFiles(e.dataTransfer.files);
        });

        const resizeObserver = new ResizeObserver(() => {
            layoutToolbar();
            scheduleRender();
        });
        resizeObserver.observe(container);

        function restoreState(st) {
            fontSize = Math.min(FONT_MAX, Math.max(FONT_MIN, st.fontSize || 13));
            wrap = !!st.wrap;
            hlOn = st.hl !== false;
            theme = THEMES[st.theme] ? st.theme : "ink";
            savedName = st.filename || "prompt.txt";
            viewMode = ["single", "split", "preview"].includes(st.view) ? st.view : "single";
            splitTarget = st.splitTarget === "result" ? "result" : "negative";
            splitToResult = st.splitToResult !== false;
            resultSource = st.resultSource === "negative" ? "negative" : "text";
            langFrom.value = LANGS[st.langFrom] ? st.langFrom : "ru";
            langTo.value = LANGS[st.langTo] ? st.langTo : "en";
            saveContent.value = ["text", "md", "json"].includes(st.saveContent)
                ? st.saveContent : "text";
            saveDest.value = st.saveDest === "local" ? "local" : "server";
            saveSub.value = typeof st.saveSub === "string" ? st.saveSub : "text";
            pendingRoot = typeof st.saveRoot === "string" ? st.saveRoot : "";
            const packed = unpackValue(textWidget.value);
            pathValue = normalizePathValue(packed.filepath);
            paneText.ta.value = packed.text;
            paneNeg.ta.value = packed.negative;
            paneRes.ta.value = packed.result;
            for (const p of ALL_PANES) p.gutterLines = -1;
            for (const key of Object.keys(bufs)) {
                bufs[key].undo.length = 0;
                bufs[key].redo.length = 0;
            }
            const wanted = ["text", "negative", "result"].includes(st.buf) ? st.buf : "text";
            bindPane(paneFor(wanted));
            themeSel.value = theme;
            hlBtn.classList.toggle("ags-on", hlOn);
            appliedTheme = "";
            applyTheme();
            applyFont();
            applyWrap();
            applyView();
            updateRootButtons();
            updateFileTag();
            syncWidget();
            refreshLinkState();
            renderAll();
            updateUndoButtons();
            updateLangScope();
            requestTokens();
            layoutToolbar();
        }

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function(info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const st = (node.properties && node.properties.agsoft_text_editor) || {};
            restoreState(st);
            setTimeout(() => checkAutosave(false), 250);
        };

        const origOnConn = node.onConnectionsChange;
        node.onConnectionsChange = function() {
            if (origOnConn) origOnConn.apply(this, arguments);
            refreshLinkState();
        };

        const origOnResize = node.onResize;
        node.onResize = function() {
            if (origOnResize) origOnResize.apply(this, arguments);
            for (const p of ALL_PANES) {
                syncMetricsFor(p);
                positionHighlightsFor(p);
            }
            layoutToolbar();
            node.setDirtyCanvas(true, false);
        };

        const origDrawFg = node.onDrawForeground;
        node.onDrawForeground = function(ctx) {
            if (origDrawFg) origDrawFg.apply(this, arguments);
            applyTheme();
            if (zenMode) {
                ctx.save();
                ctx.fillStyle = "#7fd4ff";
                ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Zen mode · Esc", node.size[0] / 2, node.size[1] / 2);
                ctx.restore();
            }
            if (!dropHint) return;
            ctx.save();
            ctx.strokeStyle = "#7fd4ff";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(1, 1, node.size[0] - 2, node.size[1] - 2);
            ctx.restore();
        };

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
            if (zenMode) setZen(false);
            if (emojiApi) emojiApi.destroy();
            document.removeEventListener("pointerdown", onDocDown);
            document.removeEventListener("keydown", onDocKey);
            resizeObserver.disconnect();
            if (autosaveTimer) clearTimeout(autosaveTimer);
            if (tokTimer) clearTimeout(tokTimer);
            INSTANCES.delete(instance);
        };

        const instance = {
            node: node,
            drop: handleFiles,
            setDropHint: (value) => {
                if (dropHint === value) return;
                dropHint = value;
                node.setDirtyCanvas(true, false);
            },
        };
        INSTANCES.add(instance);

        buildMenu();
        buildPasteMenu();
        buildRecentMenu();
        const packed = unpackValue(textWidget.value);
        pathValue = normalizePathValue(packed.filepath);
        paneText.ta.value = packed.text;
        paneNeg.ta.value = packed.negative;
        paneRes.ta.value = packed.result;
        bindPane(paneText);
        themeSel.value = theme;
        hlBtn.classList.toggle("ags-on", hlOn);
        applyTheme();
        applyFont();
        applyWrap();
        applyView();
        updateRootButtons();
        updateFileTag();
        syncWidget();
        refreshLinkState();
        renderAll();
        updateUndoButtons();
        updateLangScope();
        requestTokens();
        layoutToolbar();
        setTimeout(() => checkAutosave(false), 600);
        node.setDirtyCanvas(true, true);
    },
});