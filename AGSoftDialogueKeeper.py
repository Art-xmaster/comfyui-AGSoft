# =================================================================
# AGSoftDialogueKeeper.py
# Нода: 📝💬AGSoft Dialogue Keeper
# Описание / Description:
# Восстанавливает русские диалоги в переведённом тексте по шаблонам и кастомным маркерам.
# ---
# Restores Russian dialogues in translated text using templates and custom markers.
# Возможности / Features:
# ⚡ HTML-безопасные названия шаблонов / HTML-safe template labels
# ⚡ Шаблон <d>[Russian]....</d> / <d>[Russian]....</d> template
# ⚡ Кастомные теги до/после / Custom before/after tags
# ⚡ Поле markers с любыми парами символов / markers field with any symbol pairs
# Автор / Author: AGSoft
# Дата / Date: 04.09.2026
# =================================================================
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AGSoftDialogueKeeper")
#print("[AGSoft Dialogue Keeper] v09.04 loaded (HTML-safe template labels)")

DEFAULT_AGSOFT_MARKERS = '" ", « », “ ”, ‘ ’, „ “, ‚ ‘, ‹ ›'

# Menu shows ‹d› instead of <d>, because the dropdown eats <...> as HTML tags.
# В меню ‹d› вместо <d>, так как выпадающий список съедает <...> как HTML-теги.
TEMPLATE_TAG_MARKERS = '‹d›....‹/d› + markers'
TEMPLATE_TAG_ONLY = '‹d›....‹/d›'
TEMPLATE_TAG_RU_MARKERS = '‹d›[Russian]....‹/d› + markers'
TEMPLATE_TAG_RU_ONLY = '‹d›[Russian]....‹/d›'
TEMPLATE_BRACKET_MARKERS = '[d]....[/d] + markers'
TEMPLATE_BRACKET_ONLY = '[d]....[/d]'
TEMPLATE_MARKERS_ONLY = 'markers only'
TEMPLATE_CUSTOM = 'Custom tags'

TEMPLATE_LIST = [
    TEMPLATE_TAG_MARKERS,
    TEMPLATE_TAG_ONLY,
    TEMPLATE_TAG_RU_MARKERS,
    TEMPLATE_TAG_RU_ONLY,
    TEMPLATE_BRACKET_MARKERS,
    TEMPLATE_BRACKET_ONLY,
    TEMPLATE_MARKERS_ONLY,
    TEMPLATE_CUSTOM,
]

class AGSoftDialogueKeeper:
    DEFAULT_MARKERS = DEFAULT_AGSOFT_MARKERS

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Field original_text: original text before translation. The node searches this text for dialogues inside selected tags/markers and copies them back into the translated result. --- Поле original_text: исходный текст до перевода. Нода ищет в нём диалоги внутри выбранных тегов/маркеров и копирует их обратно в переведённый результат."
                }),
                "translated_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Field translated_text: translated text. The main English translation is kept, but fragments matching dialogues are replaced by the original Russian fragments from original_text. --- Поле translated_text: переведённый текст. Основной английский перевод сохраняется, но фрагменты, признанные диалогами, заменяются русскими фрагментами из original_text."
                }),
            },
            "optional": {
                "template": (TEMPLATE_LIST, {
                    "default": TEMPLATE_TAG_MARKERS,
                    "tooltip": "Field template: dialogue marker template. The menu shows ‹d› instead of <d> (HTML-safe notation). Options: ‹d›....‹/d› = <d>....</d>, ‹d›[Russian]....‹/d› = <d>[Russian]....</d>, [d]....[/d], markers only (all symbol pairs from the markers field), combinations with + markers, Custom tags (uses tag_start, tag_end and markers). --- Поле template: шаблон маркеров диалогов. В меню ‹d› вместо <d> (HTML-безопасная запись). Варианты: ‹d›....‹/d› = <d>....</d>, ‹d›[Russian]....‹/d› = <d>[Russian]....</d>, [d]....[/d], markers only (все пары символов из поля markers), комбинации с + markers, Custom tags (использует tag_start, tag_end и markers)."
                }),
                "tag_start": ("STRING", {
                    "default": "<d>",
                    "tooltip": "Field tag_start: opening tag for Custom tags mode. Examples: <d>, [dialog], {{say}}. Works only together with tag_end. In preset templates this field is ignored. --- Поле tag_start: открывающий тег для режима Custom tags. Примеры: <d>, [dialog], {{say}}. Работает только вместе с tag_end. В пресетных шаблонах это поле игнорируется."
                }),
                "tag_end": ("STRING", {
                    "default": "</d>",
                    "tooltip": "Field tag_end: closing tag for Custom tags mode. Examples: </d>, [/dialog], {{/say}}. Works only together with tag_start. In preset templates this field is ignored. --- Поле tag_end: закрывающий тег для режима Custom tags. Примеры: </d>, [/dialog], {{/say}}. Работает только вместе с tag_start. В пресетных шаблонах это поле игнорируется."
                }),
                "markers": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_AGSOFT_MARKERS,
                    "tooltip": 'Field markers: list of symbol pairs that wrap dialogues. By default it contains quotes " ", « », “ ”, ‘ ’, but you can put any symbols or multi-character markers. Used by templates with + markers and by Custom tags. Write pairs separated by commas, spaces or new lines; for multi-character markers use space or vertical bar: << >> or <<|>>. In Custom tags an empty field means no markers; in preset templates an empty field uses the default list. --- Поле markers: список пар символов, оборачивающих диалоги. По умолчанию содержит кавычки " ", « », “ ”, ‘ ’, но можно вставлять любые символы и многозначные маркеры. Используется шаблонами с + markers и режимом Custom tags. Пишите пары через запятую, пробел или с новой строки; для многозначных маркеров используйте пробел или вертикальную черту: << >> или <<|>>. В Custom tags пустое поле означает без маркеров; в пресетных шаблонах пустое поле использует стандартный список.'
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = """📝💬AGSoft Dialogue Keeper.
Restores original dialogues from source text into translated text.
Fields: original_text (source), translated_text (translation), template (marker template, menu shows ‹d› instead of <d>), tag_start/tag_end (custom tags), markers (symbol pairs list, quotes by default).
---
Восстанавливает оригинальные диалоги из исходного текста в переведённый.
Поля: original_text (исходник), translated_text (перевод), template (шаблон маркеров, в меню ‹d› вместо <d>), tag_start/tag_end (свои теги), markers (список пар символов, по умолчанию кавычки)."""

    # Parse marker pairs / Разбор пар маркеров
    def _parse_marker_pairs(self, raw):
        if not raw or not raw.strip():
            return []
        pairs = []
        chunks = re.split(r"[,;\r\n]+", raw)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            # Explicit open|close syntax / Явный формат открытие|закрытие
            if "|" in chunk:
                left, right = chunk.split("|", 1)
                left = left.strip()
                right = right.strip()
                if left:
                    pairs.append((left, right or left))
                continue
            tokens = [t for t in chunk.split() if t]
            # Space-separated markers / Маркеры через пробел
            if len(tokens) >= 2:
                i = 0
                while i < len(tokens):
                    if i + 1 < len(tokens):
                        pairs.append((tokens[i], tokens[i + 1]))
                        i += 2
                    else:
                        pairs.append((tokens[i], tokens[i]))
                        i += 1
                continue
            # Compact form like «» / Компактная форма вида «»
            token = tokens[0] if tokens else chunk
            compact = re.sub(r"\s+", "", token)
            if not compact:
                continue
            if len(compact) == 1:
                pairs.append((compact, compact))
            elif len(compact) == 2:
                pairs.append((compact[0], compact[1]))
            else:
                for i in range(0, len(compact) - 1, 2):
                    pairs.append((compact[i], compact[i + 1]))
                if len(compact) % 2 == 1:
                    pairs.append((compact[-1], compact[-1]))
        # Unique order / Уникальные без потери порядка
        seen = set()
        out = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
        return out

    # Select markers by template / Выбор маркеров по шаблону
    def _get_markers(self, template, tag_start, tag_end, markers_raw):
        t = (template or TEMPLATE_TAG_MARKERS).strip()
        use_tags = False
        tags = ("", "")
        markers_on = False

        custom_set = {TEMPLATE_CUSTOM, "Custom", "Custom / Свои теги и кавычки"}
        tag_d_markers_set = {TEMPLATE_TAG_MARKERS, '<d>....</d> + markers', '<d>....</d> + quotes', '<d>....</d> + "...."', ' <d >.... </d > +  ".... "', "Tag <d> + quotes", "<d>...</d> + quotes / <d>...</d> + кавычки"}
        tag_d_only_set = {TEMPLATE_TAG_ONLY, '<d>....</d>', ' <d >.... </d >', "Tag <d> only", "<d>...</d> only / только <d>...</d>"}
        tag_ru_markers_set = {TEMPLATE_TAG_RU_MARKERS, '<d>[Russian]....</d> + markers', '<d>[Russian]....</d> + quotes', '<d>[Russian]....</d> + "...."', ' <d >[Russian].... </d > +  ".... "'}
        tag_ru_only_set = {TEMPLATE_TAG_RU_ONLY, '<d>[Russian]....</d>', ' <d >[Russian].... </d >'}
        tag_br_markers_set = {TEMPLATE_BRACKET_MARKERS, '[d]....[/d] + quotes', '[d]....[/d] +  ".... "', "Tag [d] + quotes", "[d]...[/d] + quotes / [d]...[/d] + кавычки"}
        tag_br_only_set = {TEMPLATE_BRACKET_ONLY, "[d]....[/d]", "Tag [d] only", "[d]...[/d] only / только [d]...[/d]"}
        markers_only_set = {TEMPLATE_MARKERS_ONLY, 'quotes only', '"...."', ' ".... "', "Quotes only", "Quotes only / Только кавычки"}

        if t in custom_set:
            if tag_start and tag_end:
                use_tags = True
                tags = (tag_start, tag_end)
            return use_tags, tags, self._parse_marker_pairs(markers_raw)
        if t in tag_d_markers_set:
            use_tags, tags, markers_on = True, ("<d>", "</d>"), True
        elif t in tag_d_only_set:
            use_tags, tags = True, ("<d>", "</d>")
        elif t in tag_ru_markers_set:
            use_tags, tags, markers_on = True, ("<d>[Russian]", "</d>"), True
        elif t in tag_ru_only_set:
            use_tags, tags = True, ("<d>[Russian]", "</d>")
        elif t in tag_br_markers_set:
            use_tags, tags, markers_on = True, ("[d]", "[/d]"), True
        elif t in tag_br_only_set:
            use_tags, tags = True, ("[d]", "[/d]")
        elif t in markers_only_set:
            markers_on = True
        else:
            # Fallback to Custom tags / Откат на Custom tags
            logger.warning(f"[AGSoft Dialogue Keeper] Unknown template '{t}', using Custom tags fallback.")
            if tag_start and tag_end:
                use_tags = True
                tags = (tag_start, tag_end)
            return use_tags, tags, self._parse_marker_pairs(markers_raw)

        marker_list = []
        if markers_on:
            marker_list = self._parse_marker_pairs(markers_raw)
            if not marker_list:
                marker_list = self._parse_marker_pairs(DEFAULT_AGSOFT_MARKERS)
        return use_tags, tags, marker_list

    # Build common regex / Сборка общего regex
    def _build_pattern(self, use_tags, tags, marker_list):
        parts = []
        if use_tags:
            start, end = tags
            if start and end:
                parts.append(re.escape(start) + r".*?" + re.escape(end))
        marker_list = sorted(marker_list, key=lambda x: len(x[0]) + len(x[1]), reverse=True)
        for open_q, close_q in marker_list:
            if open_q and close_q:
                parts.append(re.escape(open_q) + r".*?" + re.escape(close_q))
        if not parts:
            return None
        return "|".join(f"(?:{p})" for p in parts)

    # Marker key for alignment / Ключ маркера для выравнивания
    def _marker_key(self, text, tags, marker_list):
        start, end = tags
        if start and end and text.startswith(start) and text.endswith(end) and len(text) >= len(start) + len(end):
            return f"tag:{start}:{end}"
        for open_q, close_q in marker_list:
            if open_q and close_q and text.startswith(open_q) and text.endswith(close_q) and len(text) >= len(open_q) + len(close_q):
                return f"quote:{open_q}:{close_q}"
        return f"raw:{text[:1]}:{text[-1:]}"

    def process(self, original_text, translated_text, template=TEMPLATE_TAG_MARKERS, tag_start="<d>", tag_end="</d>", markers=None):
        if markers is None:
            markers = DEFAULT_AGSOFT_MARKERS
        original_text = original_text or ""
        translated_text = translated_text or ""
        if not original_text or not translated_text:
            return (translated_text,)
        try:
            use_tags, tags, marker_list = self._get_markers(template, tag_start, tag_end, markers)
            pattern = self._build_pattern(use_tags, tags, marker_list)
            if not pattern:
                logger.warning("[AGSoft Dialogue Keeper] No active markers. Check template, tag_start/tag_end or markers.")
                return (translated_text,)
            matches_orig = list(re.finditer(pattern, original_text, flags=re.DOTALL))
            matches_trans = list(re.finditer(pattern, translated_text, flags=re.DOTALL))
            if not matches_orig or not matches_trans:
                logger.warning("[AGSoft Dialogue Keeper] No dialogues matched in original or translated text.")
                return (translated_text,)
            if len(matches_orig) != len(matches_trans):
                logger.warning(f"[AGSoft Dialogue Keeper] Count mismatch: original={len(matches_orig)}, translated={len(matches_trans)}.")
            # If counts match, use simple order; otherwise align by marker type.
            # Если количество совпадает, используем порядок; иначе выравниваем по типу маркера.
            if len(matches_orig) == len(matches_trans):
                replacements = [(matches_trans[i], matches_orig[i].group(0)) for i in range(len(matches_orig))]
            else:
                used = [False] * len(matches_trans)
                replacements = []
                for om in matches_orig:
                    o_text = om.group(0)
                    o_key = self._marker_key(o_text, tags, marker_list)
                    o_broad = o_key.split(":", 1)[0]
                    idx = None
                    for j, tm in enumerate(matches_trans):
                        if not used[j] and self._marker_key(tm.group(0), tags, marker_list) == o_key:
                            idx = j
                            break
                    if idx is None:
                        for j, tm in enumerate(matches_trans):
                            if not used[j] and self._marker_key(tm.group(0), tags, marker_list).split(":", 1)[0] == o_broad:
                                idx = j
                                break
                    if idx is not None:
                        used[idx] = True
                        replacements.append((matches_trans[idx], o_text))
            if not replacements:
                return (translated_text,)
            result = translated_text
            for tm, repl in sorted(replacements, key=lambda x: x[0].start(), reverse=True):
                result = result[:tm.start()] + repl + result[tm.end():]
            return (result,)
        except Exception as e:
            logger.error(f"[AGSoft Dialogue Keeper] Error: {str(e)}")
            return (translated_text,)

NODE_CLASS_MAPPINGS = {
    "AGSoftDialogueKeeper": AGSoftDialogueKeeper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftDialogueKeeper": "📝💬AGSoft Dialogue Keeper"
}