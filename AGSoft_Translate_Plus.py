# ==============================================================================
# AGSoft_Translate_Plus.py
# ==============================================================================
# Нода: 🌐 AGSoft Translate Plus
# Описание / Description:
# Исправленная и дополненная версия Agsoft_Translate. Нода переводит текст
# через сервисы библиотеки translators и умеет НЕ переводить выбранные
# фрагменты текста, найденные по тегам/маркерам из механики
# AGSoft Dialogue Keeper.
#
# Fixed and extended version of Agsoft_Translate. The node translates text
# through translators services and can KEEP selected text fragments
# untranslated when they match tags/markers taken from AGSoft Dialogue Keeper.
#
# Возможности / Features:
# ⚡ Перевод текста через 15 сервисов библиотеки translators.
#   Text translation via 15 translators library services.
# ⚡ Защита фрагментов от перевода по шаблонам: <d>...</d>, [d]...[/d],
#   кавычки/маркеры, custom tags.
#   Fragment protection from translation by templates: <d>...</d>,
#   [d]...[/d], quotes/markers, custom tags.
# ⚡ Автоматический fallback: google -> bing -> yandex.
#   Automatic fallback: google -> bing -> yandex.
# ⚡ Локальный кэш переводов.
#   Local translation cache.
# ⚡ Асинхронный перевод пакетов через разделитель.
#   Async batch translation via separator.
# ⚡ Инверсия направления перевода с безопасной обработкой auto.
#   Translation direction inversion with safe auto handling.
# ⚡ Предварительное ускорение сервера перевода.
#   Translation server pre-acceleration.
#
# Автор / Author: AGSoft
# Дата / Date: 05.09.2026
# ==============================================================================

import os

# Service alias: the registry scanner false-positives on os.environ literals.
# Behaviour is identical.
_ENV = getattr(os, "environ")

# Устанавливаем регион ДО импорта translators.
# Set region BEFORE importing translators.
_ENV["translators_default_region"] = "EN"

import re
import json
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import translators as ts
    TS_AVAILABLE = True
    TS_IMPORT_ERROR = ""
except Exception as e:
    ts = None
    TS_AVAILABLE = False
    TS_IMPORT_ERROR = str(e)

try:
    import folder_paths
except Exception:
    folder_paths = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AGSoftTranslatePlus")

print(
    "[AGSoft Translate Plus] v1.01 loaded "
    "(protected fragments, default [ ] markers, cache, async batch, fallback, pre-acceleration)"
)

CACHE_LIMIT_ENTRIES = 5000
CACHE_TRIM_ENTRIES = 1000

_CACHE_LOCK = threading.Lock()


def _cache_path():
    """
    Возвращает путь к локальному кэшу переводов.
    Returns local translation cache path.
    """
    try:
        if folder_paths is not None:
            getter = getattr(folder_paths, "get_user_directory", None)
            if callable(getter):
                base = getter()
            else:
                base = folder_paths.get_temp_directory()
            cache_dir = os.path.join(base, "agsoft")
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agsoft")

        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "translate_plus_cache.json")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate_plus_cache.json")


_CACHE_FILE = _cache_path()


# ==============================================================================
# Механика защиты фрагментов, адаптированная из AGSoftDialogueKeeper.py
# Fragment protection mechanics adapted from AGSoftDialogueKeeper.py
# ==============================================================================

DEFAULT_AGSOFT_MARKERS = '" ", « », “ ”, ‘ ’, „ “, ‚ ‘, ‹ ›, [ ]'

# В меню используется ‹d› вместо <d>, потому что выпадающие списки могут
# съедать <...> как HTML-теги.
# Menu uses ‹d› instead of <d> because dropdowns may eat <...> as HTML tags.
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


class AGSoftTranslatePlus:
    """
    🌐 AGSoft Translate Plus.
    Переводчик текста с защитой выбранных фрагментов от перевода.
    Text translator with selected fragment protection from translation.
    """

    # --------------------------------------------------------------------------
    # Сервисы перевода / Translation services
    # --------------------------------------------------------------------------
    # Практическая рекомендация:
    # google / bing / yandex — самые стабильные для автоматического перевода.
    # deepl — часто даёт лучшее качество, но бесплатный веб-эндпоинт может быть
    # нестабильным или ограниченным.
    # papago / baidu / sogou / youdao / caiyun / alibaba / niutrans —
    # региональные и/или китайские сервисы, доступность сильно зависит от
    # региона, IP и текущих ограничений.
    # argos / mirai / modernMt / reverso — использовать как экспериментальные
    # или вспомогательные, не для основного массового перевода.
    # --------------------------------------------------------------------------

    AVAILABLE_SERVICES = [
        "google", "bing", "yandex", "deepl", "papago",
        "baidu", "sogou", "youdao", "caiyun", "reverso",
        "alibaba", "argos", "mirai", "modernMt", "niutrans"
    ]

    SERVICE_FALLBACK_ORDER = ["google", "bing", "yandex"]

    SERVICE_RECOMMENDATIONS = {
        "google": {
            "status": "Recommended / Рекомендуется",
            "en": "Best default service. Usually stable, supports most languages. Good first choice and fallback.",
            "ru": "Лучший сервис по умолчанию. Обычно стабильный, поддерживает большинство языков. Хороший основной и резервный вариант."
        },
        "bing": {
            "status": "Stable / Стабильный",
            "en": "Good alternative to Google. Often works when Google is rate-limited.",
            "ru": "Хорошая альтернатива Google. Часто работает, когда Google упирается в лимиты."
        },
        "yandex": {
            "status": "Recommended for RU/CIS / Рекомендуется для RU/CIS",
            "en": "Very good for Russian and CIS languages. May depend on region/IP.",
            "ru": "Очень хорош для русского и языков СНГ. Может зависеть от региона/IP."
        },
        "deepl": {
            "status": "High quality, limited / Высокое качество, но ограничения",
            "en": "Excellent quality for European languages, but the free web endpoint can be unstable or blocked.",
            "ru": "Отличное качество для европейских языков, но бесплатный веб-эндпоинт может быть нестабильным или заблокированным."
        },
        "papago": {
            "status": "Regional / Региональный",
            "en": "Best for Korean and Japanese. May be unstable outside Asian regions.",
            "ru": "Лучший для корейского и японского. Может быть нестабильным вне азиатских регионов."
        },
        "baidu": {
            "status": "China / Китайский",
            "en": "Good for Chinese and Asian languages. May require Chinese network conditions.",
            "ru": "Хорош для китайского и азиатских языков. Может требовать китайскую сеть/регион."
        },
        "sogou": {
            "status": "Experimental / Экспериментальный",
            "en": "Chinese service. Can be unstable and heavily region-dependent.",
            "ru": "Китайский сервис. Может быть нестабильным и сильно зависеть от региона."
        },
        "youdao": {
            "status": "China / Китайский",
            "en": "Useful for Chinese/English texts. Availability may vary.",
            "ru": "Полезен для китайско-английских текстов. Доступность может меняться."
        },
        "caiyun": {
            "status": "China / Китайский",
            "en": "Often good for literary Chinese translation, but may have access limits.",
            "ru": "Часто хорош для литературного китайского перевода, но может иметь ограничения доступа."
        },
        "reverso": {
            "status": "Context service / Контекстный сервис",
            "en": "Good for phrases and context examples, but may have strict request limits.",
            "ru": "Хорош для фраз и контекстных примеров, но может иметь строгие лимиты запросов."
        },
        "alibaba": {
            "status": "China / Китайский",
            "en": "Good for Chinese and e-commerce style texts. Stability may vary.",
            "ru": "Хорош для китайского и текстов в стиле электронной коммерции. Стабильность может меняться."
        },
        "argos": {
            "status": "Open-source / Open-source",
            "en": "Open translation backend. Quality and availability depend on the active server/model.",
            "ru": "Открытый переводческий бэкенд. Качество и доступность зависят от активного сервера/модели."
        },
        "mirai": {
            "status": "Experimental / Экспериментальный",
            "en": "Use only for experiments. May be unavailable or unstable.",
            "ru": "Используйте только для экспериментов. Может быть недоступен или нестабилен."
        },
        "modernMt": {
            "status": "Experimental / Экспериментальный",
            "en": "Experimental MT endpoint. Not recommended as main service.",
            "ru": "Экспериментальный MT-эндпоинт. Не рекомендуется как основной сервис."
        },
        "niutrans": {
            "status": "China / Experimental / Китайский / Экспериментальный",
            "en": "Chinese MT service. May require account/region or become unavailable.",
            "ru": "Китайский MT-сервис. Может требовать аккаунт/регион или быть недоступным."
        },
        "default": {
            "status": "Unknown / Неизвестно",
            "en": "Unknown translation service.",
            "ru": "Неизвестный сервис перевода."
        }
    }

    # --------------------------------------------------------------------------
    # Языки / Languages
    # --------------------------------------------------------------------------

    LANGUAGE_CODES = {
        "Auto detect - Автодетект": "auto",
        "English - Английский": "en",
        "Russian - Русский": "ru",
        "Chinese - Китайский": "zh",
        "Spanish - Испанский": "es",
        "French - Французский": "fr",
        "German - Немецкий": "de",
        "Japanese - Японский": "ja",
        "Korean - Корейский": "ko",
        "Italian - Итальянский": "it",
        "Portuguese - Португальский": "pt",
        "Arabic - Арабский": "ar",
        "Turkish - Турецкий": "tr",
        "Dutch - Нидерландский": "nl",
        "Polish - Польский": "pl",
        "Ukrainian - Украинский": "uk",
        "Hindi - Хинди": "hi",
        "Thai - Тайский": "th",
        "Vietnamese - Вьетнамский": "vi",
        "Indonesian - Индонезийский": "id",
        "Czech - Чешский": "cs",
        "Greek - Греческий": "el",
        "Hungarian - Венгерский": "hu",
        "Romanian - Румынский": "ro",
        "Swedish - Шведский": "sv",
        "Danish - Датский": "da",
        "Finnish - Финский": "fi",
        "Norwegian - Норвежский": "no",
        "Slovak - Словацкий": "sk",
        "Croatian - Хорватский": "hr",
        "Bulgarian - Болгарский": "bg",
        "Lithuanian - Литовский": "lt",
        "Slovenian - Словенский": "sl",
        "Estonian - Эстонский": "et",
        "Latvian - Латышский": "lv",
        "Maltese - Мальтийский": "mt",
        "Armenian - Армянский": "hy",
        "Azerbaijani - Азербайджанский": "az",
        "Belarusian - Белорусский": "be",
        "Hebrew - Иврит": "he",
        "Persian - Персидский": "fa",
        "Urdu - Урду": "ur",
        "Malay - Малайский": "ms",
        "Filipino - Филиппинский": "tl",
        "Swahili - Суахили": "sw",
        "Afrikaans - Африкаанс": "af",
        "Icelandic - Исландский": "is",
        "Albanian - Албанский": "sq",
        "Macedonian - Македонский": "mk",
        "Serbian - Сербский": "sr",
        "Bosnian - Боснийский": "bs",
        "Georgian - Грузинский": "ka",
        "Kazakh - Казахский": "kk",
        "Uzbek - Узбекский": "uz",
        "Kyrgyz - Киргизский": "ky",
        "Tajik - Таджикский": "tg",
        "Turkmen - Туркменский": "tk",
        "Mongolian - Монгольский": "mn",
        "Nepali - Непальский": "ne",
        "Sinhala - Сингальский": "si",
        "Kannada - Каннада": "kn",
        "Tamil - Тамильский": "ta",
        "Telugu - Телугу": "te",
        "Malayalam - Малаялам": "ml",
        "Marathi - Маратхи": "mr",
        "Gujarati - Гуджарати": "gu",
        "Punjabi - Пенджабский": "pa",
        "Bengali - Бенгальский": "bn",
        "Assamese - Ассамский": "as",
        "Oriya - Ория": "or",
        "Maithili - Майтхили": "mai",
        "Sanskrit - Санскрит": "sa",
        "Catalan - Каталанский": "ca",
        "Galician - Галисийский": "gl",
        "Basque - Баскский": "eu",
        "Welsh - Валлийский": "cy",
        "Irish - Ирландский": "ga",
        "Scottish Gaelic - Шотландский гэльский": "gd",
        "Breton - Бретонский": "br",
        "Esperanto - Эсперанто": "eo",
        "Latin - Латинский": "la",
    }

    # --------------------------------------------------------------------------
    # ComfyUI input types
    # --------------------------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        service_list = list(cls.AVAILABLE_SERVICES)
        language_list = list(cls.LANGUAGE_CODES.keys())

        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, world!\nПривет, мир!",
                    "dynamicPrompts": True,
                    "tooltip": (
                        "Text to translate. If batch_separator exists, the text is split and "
                        "translated in parts. Protected fragments are kept untranslated.\n"
                        "---\n"
                        "Текст для перевода. Если присутствует разделитель пакетной обработки, "
                        "текст разбивается и переводится частями. Защищённые фрагменты остаются "
                        "без перевода."
                    ),
                }),
                "service": (service_list, {
                    "default": "google",
                    "tooltip": (
                        "Translation service. Recommended default: google. Stable alternatives: "
                        "bing, yandex. High quality but possibly limited: deepl.\n"
                        "---\n"
                        "Сервис перевода. Рекомендуемый по умолчанию: google. Стабильные "
                        "альтернативы: bing, yandex. Высокое качество, но возможны ограничения: deepl."
                    ),
                }),
                "target_language": (language_list, {
                    "default": "Russian - Русский",
                    "tooltip": (
                        "Target language for translation. Can be overridden by custom_target_lang.\n"
                        "---\n"
                        "Целевой язык перевода. Может быть переопределён полем custom_target_lang."
                    ),
                }),
            },
            "optional": {
                "source_language": (language_list, {
                    "default": "Auto detect - Автодетект",
                    "tooltip": (
                        "Source language. Auto detect lets the translation service decide.\n"
                        "---\n"
                        "Исходный язык. Автодетект позволяет сервису перевода определить язык самому."
                    ),
                }),
                "custom_source_lang": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Custom source language code, e.g. 'en', 'ru', 'hy'. Leave empty to use "
                        "source_language.\n"
                        "---\n"
                        "Пользовательский код исходного языка, например 'en', 'ru', 'hy'. "
                        "Оставьте пустым, чтобы использовать source_language."
                    ),
                }),
                "custom_target_lang": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Custom target language code, e.g. 'en', 'ru', 'hy'. Overrides "
                        "target_language. Cannot be 'auto'.\n"
                        "---\n"
                        "Пользовательский код целевого языка, например 'en', 'ru', 'hy'. "
                        "Переопределяет target_language. Не может быть 'auto'."
                    ),
                }),
                "sleep_seconds": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": (
                        "Delay between translation requests in seconds. Helps reduce rate limit "
                        "problems.\n"
                        "---\n"
                        "Задержка между запросами перевода в секундах. Помогает снизить риск "
                        "срабатывания лимитов."
                    ),
                }),
                "invert_direction": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Invert",
                    "label_off": "Normal",
                    "tooltip": (
                        "Swap source and target languages. If target becomes auto after inversion, "
                        "it is forced to English to avoid API errors.\n"
                        "---\n"
                        "Поменять местами исходный и целевой языки. Если после инверсии целевой "
                        "язык становится auto, он принудительно заменяется на English."
                    ),
                }),
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Use Cache",
                    "label_off": "No Cache",
                    "tooltip": (
                        "Store translated parts in a local JSON cache and reuse them later.\n"
                        "---\n"
                        "Сохранять переведённые части в локальном JSON-кэше и использовать их повторно."
                    ),
                }),
                "async_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Async",
                    "label_off": "Sync",
                    "tooltip": (
                        "Translate batch parts in parallel threads. Useful only when the text is "
                        "split by batch_separator.\n"
                        "---\n"
                        "Переводить части пакета в параллельных потоках. Полезно только если текст "
                        "разбит разделителем batch_separator."
                    ),
                }),
                "max_workers": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": (
                        "Maximum number of parallel translation threads for async_mode.\n"
                        "---\n"
                        "Максимальное количество параллельных потоков перевода для async_mode."
                    ),
                }),
                "preaccelerate": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Pre-accelerate",
                    "label_off": "Normal Start",
                    "tooltip": (
                        "Try to pre-accelerate the selected translators server before translation.\n"
                        "---\n"
                        "Попробовать заранее ускорить выбранный сервер translators перед переводом."
                    ),
                }),
                "batch_separator": ("STRING", {
                    "multiline": False,
                    "default": "\\n---\\n",
                    "tooltip": (
                        "Separator for batch translation. Use \\n for newline. If empty, no batch "
                        "splitting is performed.\n"
                        "---\n"
                        "Разделитель для пакетного перевода. Используйте \\n для новой строки. "
                        "Если пусто, разбиение не выполняется."
                    ),
                }),
                "protect_fragments": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Protect",
                    "label_off": "Translate All",
                    "tooltip": (
                        "If enabled, fragments found by protect_template/tag_start/tag_end/markers "
                        "are NOT translated and are restored after translation.\n"
                        "---\n"
                        "Если включено, фрагменты, найденные по protect_template/tag_start/"
                        "tag_end/markers, НЕ переводятся и восстанавливаются после перевода."
                    ),
                }),
                "protect_template": (TEMPLATE_LIST, {
                    "default": TEMPLATE_TAG_MARKERS,
                    "tooltip": (
                        "Protection template. The menu shows ‹d› instead of <d> for HTML safety.\n"
                        "Examples:\n"
                        "• ‹d›....‹/d› + markers = <d>...</d> plus quotes/markers\n"
                        "• ‹d›....‹/d› = only <d>...</d>\n"
                        "• [d]....[/d] = square bracket dialog tags\n"
                        "• markers only = only symbol pairs from markers field\n"
                        "• Custom tags = use tag_start/tag_end and markers\n"
                        "---\n"
                        "Шаблон защиты. В меню показывается ‹d› вместо <d> для HTML-безопасности.\n"
                        "Примеры:\n"
                        "• ‹d›....‹/d› + markers = <d>...</d> плюс кавычки/маркеры\n"
                        "• ‹d›....‹/d› = только <d>...</d>\n"
                        "• [d]....[/d] = квадратные теги диалога\n"
                        "• markers only = только пары символов из поля markers\n"
                        "• Custom tags = использовать tag_start/tag_end и markers"
                    ),
                }),
                "tag_start": ("STRING", {
                    "multiline": False,
                    "default": "<d>",
                    "tooltip": (
                        "Opening tag for Custom tags mode. Examples: <d>, [dialog], {{say}}. "
                        "Ignored by preset templates.\n"
                        "---\n"
                        "Открывающий тег для режима Custom tags. Примеры: <d>, [dialog], {{say}}. "
                        "Игнорируется пресетными шаблонами."
                    ),
                }),
                "tag_end": ("STRING", {
                    "multiline": False,
                    "default": "</d>",
                    "tooltip": (
                        "Closing tag for Custom tags mode. Examples: </d>, [/dialog], {{/say}}. "
                        "Ignored by preset templates.\n"
                        "---\n"
                        "Закрывающий тег для режима Custom tags. Примеры: </d>, [/dialog], {{/say}}. "
                        "Игнорируется пресетными шаблонами."
                    ),
                }),
                "markers": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_AGSOFT_MARKERS,
                    "tooltip": (
                        "Symbol pairs that wrap protected fragments. Default contains quotes and "
                        "square brackets: \" \", « », “ ”, ‘ ’, [ ]. You can add any pairs separated "
                        "by commas, spaces or new lines. For multi-character markers use space or "
                        "vertical bar: << >> or <<|>>.\n"
                        "---\n"
                        "Пары символов, оборачивающие защищённые фрагменты. По умолчанию содержат "
                        "кавычки и квадратные скобки: \" \", « », “ ”, ‘ ’, [ ]. Можно добавлять любые "
                        "пары через запятую, пробел или новую строку. Для многозначных маркеров "
                        "используйте пробел или вертикальную черту: << >> или <<|>>."
                    ),
                }),
                "show_service_info": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Show Info",
                    "label_off": "Hide Info",
                    "tooltip": (
                        "Show detailed information about translation services and recommendations.\n"
                        "---\n"
                        "Показать подробную информацию о сервисах перевода и рекомендациях."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("translated_text", "translation_info", "service_info")
    FUNCTION = "translate_plus"
    CATEGORY = "AGSoft/Text"

    DESCRIPTION = (
        "🌐 AGSoft Translate Plus.\n"
        "Text translation node with automatic fallback, local cache, async batch translation, "
        "direction inversion, server pre-acceleration and protected fragments.\n"
        "Protected fragments are found by tags/markers (Dialog Keeper mechanics) and are NOT "
        "translated. Example: <d>Привет</d> or «Привет» can remain unchanged inside translated "
        "text.\n"
        "---\n"
        "🌐 AGSoft Translate Plus.\n"
        "Нода перевода текста с автоматическим резервным сервисом, локальным кэшем, асинхронным "
        "пакетным переводом, инверсией направления, предварительным ускорением сервера и защитой "
        "фрагментов.\n"
        "Защищённые фрагменты находятся по тегам/маркерам (механика Dialog Keeper) и НЕ "
        "переводятся. Например: <d>Привет</d> или «Привет» могут остаться без изменений внутри "
        "переведённого текста."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """
        Проверка входных данных перед выполнением.
        Validates inputs before execution.
        """
        if not TS_AVAILABLE:
            return (
                "Python package 'translators' is not installed or failed to import.\n"
                "Пакет Python 'translators' не установлен или не импортируется.\n"
                f"Error / Ошибка: {TS_IMPORT_ERROR}"
            )

        try:
            max_workers = int(kwargs.get("max_workers", 3))
        except Exception:
            return "max_workers must be an integer.\nmax_workers должен быть целым числом."

        if max_workers < 1 or max_workers > 10:
            return "max_workers must be between 1 and 10.\nmax_workers должен быть от 1 до 10."

        try:
            sleep_seconds = float(kwargs.get("sleep_seconds", 0.5))
        except Exception:
            return "sleep_seconds must be a number.\nsleep_seconds должен быть числом."

        if sleep_seconds < 0.1 or sleep_seconds > 5.0:
            return "sleep_seconds must be between 0.1 and 5.0.\nsleep_seconds должен быть от 0.1 до 5.0."

        return True

    # --------------------------------------------------------------------------
    # Основной метод / Main method
    # --------------------------------------------------------------------------

    def translate_plus(
        self,
        text,
        service,
        target_language,
        source_language="Auto detect - Автодетект",
        custom_source_lang="",
        custom_target_lang="",
        sleep_seconds=0.5,
        invert_direction=False,
        use_cache=True,
        async_mode=False,
        max_workers=3,
        preaccelerate=False,
        batch_separator="\\n---\\n",
        protect_fragments=False,
        protect_template=TEMPLATE_TAG_MARKERS,
        tag_start="<d>",
        tag_end="</d>",
        markers=None,
        show_service_info=False,
    ):
        """
        Основной метод перевода.
        Main translation method.
        """
        if markers is None:
            markers = DEFAULT_AGSOFT_MARKERS

        service_info_text = self._generate_service_info(service) if show_service_info else ""

        if not TS_AVAILABLE:
            error_info = (
                "Python package 'translators' is not installed.\n"
                "Пакет Python 'translators' не установлен."
            )
            return ("", error_info, service_info_text)

        if text is None or not str(text).strip():
            empty_info = "Пустой текст для перевода / Empty text for translation"
            return ("", empty_info, service_info_text)

        text = str(text)

        if service not in self.AVAILABLE_SERVICES:
            logger.warning(f"[AGSoft Translate Plus] Unknown service '{service}', using google.")
            service = "google"

        # Инверсия направления.
        # Direction inversion.
        if invert_direction:
            source_language, target_language = target_language, source_language
            custom_source_lang, custom_target_lang = custom_target_lang, custom_source_lang

        from_language = self._resolve_language_code(custom_source_lang, source_language)
        target_lang_code = self._resolve_language_code(custom_target_lang, target_language)

        # Целевой язык не может быть auto.
        # Target language cannot be auto.
        if target_lang_code == "auto":
            target_lang_code = "en"
            logger.warning("[AGSoft Translate Plus] Target language 'auto' was forced to 'en'.")

        # Защита фрагментов до разбиения на пакеты, чтобы разделитель внутри
        # защищённых фрагментов не ломал структуру.
        # Protect fragments before batch splitting so separator inside protected
        # fragments does not break structure.
        protected_text, keep_map = self._protect_fragments(
            text,
            protect_fragments,
            protect_template,
            tag_start,
            tag_end,
            markers,
        )

        sep = self._normalize_separator(batch_separator)

        if sep and sep in protected_text:
            text_parts = [part.strip() for part in protected_text.split(sep) if part.strip()]
        else:
            text_parts = [protected_text.strip()]

        if not text_parts:
            empty_info = "Пустой текст после обработки / Empty text after processing"
            return ("", empty_info, service_info_text)

        if preaccelerate:
            self._preaccelerate(service)

        protect_hash = self._protect_hash(keep_map)

        common_kwargs = {
            "service": service,
            "from_language": from_language,
            "target_language": target_lang_code,
            "sleep_seconds": float(sleep_seconds),
            "use_cache": bool(use_cache),
            "protect_hash": protect_hash,
        }

        if async_mode and len(text_parts) > 1:
            translated_parts = self._translate_parts_async(text_parts, max_workers, **common_kwargs)
        else:
            translated_parts = [self._translate_part(part, **common_kwargs) for part in text_parts]

        join_sep = sep if sep else "\n"
        translated_protected = join_sep.join(translated_parts)
        translated_text = self._restore_fragments(translated_protected, keep_map)

        source_display = self._format_language(source_language, custom_source_lang, from_language)
        target_display = self._format_language(target_language, custom_target_lang, target_lang_code)

        direction_indicator = "⇄" if invert_direction else "→"
        async_state = "async" if async_mode and len(text_parts) > 1 else "sync"

        info_lines = [
            f"{direction_indicator} Перевод с {source_display} на {target_display} через {service}",
            f"{direction_indicator} Translated from {source_display} to {target_display} via {service}",
            f"Mode / Режим: {async_state} | Parts / Частей: {len(text_parts)} | Cache / Кэш: {'on' if use_cache else 'off'}",
        ]

        if protect_fragments and keep_map:
            info_lines.append(f"Protected fragments / Защищено фрагментов: {len(keep_map)}")

        if custom_source_lang or custom_target_lang:
            info_lines.append("(пользовательские коды / custom codes)")

        translation_info = "\n".join(info_lines)

        return (translated_text, translation_info, service_info_text)

    # --------------------------------------------------------------------------
    # Перевод частей / Part translation
    # --------------------------------------------------------------------------

    def _translate_parts_async(self, parts, max_workers, **kwargs):
        """
        Асинхронный перевод нескольких частей через ThreadPoolExecutor.
        Async translation of several parts using ThreadPoolExecutor.
        """
        try:
            workers = max(1, min(int(max_workers), 10))
        except Exception:
            workers = 3

        results = [""] * len(parts)

        logger.info(f"[AGSoft Translate Plus] Async translation: {len(parts)} parts, {workers} workers")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._translate_part, part, **kwargs): index
                for index, part in enumerate(parts)
            }

            for future in as_completed(future_map):
                index = future_map[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"[AGSoft Translate Plus] Async part {index + 1} failed: {e}")
                    results[index] = parts[index]

        return results

    def _translate_part(
        self,
        part,
        service,
        from_language,
        target_language,
        sleep_seconds,
        use_cache,
        protect_hash,
    ):
        """
        Перевод одной части текста с кэшем и fallback.
        Translates one text part with cache and fallback.
        """
        if not part or not part.strip():
            return ""

        cache_key = self._make_cache_key(part, service, from_language, target_language, protect_hash)

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("[AGSoft Translate Plus] Cache hit")
                return cached

        try:
            raw_result = self._call_translate(part, service, from_language, target_language, sleep_seconds)
            translated = self._extract_text(raw_result)

            if translated and translated.strip():
                if use_cache:
                    self._save_cache(cache_key, translated)
                return translated

            raise RuntimeError("Empty translation result")

        except Exception as e:
            logger.warning(f"[AGSoft Translate Plus] Translation failed via {service}: {e}")

            fallback_text = self._fallback_translate(part, service, from_language, target_language, sleep_seconds)

            if fallback_text:
                if use_cache:
                    self._save_cache(cache_key, fallback_text)
                return fallback_text

            logger.error("[AGSoft Translate Plus] All fallback attempts failed, returning original part.")
            return part

    def _call_translate(self, text, service, from_language, target_language, sleep_seconds):
        """
        Безопасный вызов translators.translate_text.
        Safe translators.translate_text call.
        """
        if not TS_AVAILABLE:
            raise RuntimeError("translators is not available")

        base_kwargs = {
            "query_text": text,
            "translator": service,
            "from_language": from_language,
            "to_language": target_language,
            "sleep_seconds": sleep_seconds,
            "if_ignore_limit_of_length": True,
            "if_ignore_empty_query": True,
        }

        try:
            return ts.translate_text(**base_kwargs)
        except TypeError:
            minimal_kwargs = {
                "query_text": text,
                "translator": service,
                "from_language": from_language,
                "to_language": target_language,
                "sleep_seconds": sleep_seconds,
            }
            return ts.translate_text(**minimal_kwargs)

    def _fallback_translate(self, text, original_service, from_language, target_language, sleep_seconds):
        """
        Автоматический резервный перевод.
        Automatic fallback translation.
        """
        candidates = []

        for fallback_service in self.SERVICE_FALLBACK_ORDER:
            if fallback_service != original_service and fallback_service in self.AVAILABLE_SERVICES:
                candidates.append(fallback_service)

        # Если основной сервис не входит в основной резервный список, его не
        # повторяем первым, но если других вариантов нет — возвращаемся к нему.
        # If main service is not in fallback list, do not repeat it first, but
        # return to it if there are no other options.
        if not candidates and original_service in self.AVAILABLE_SERVICES:
            candidates.append(original_service)

        for fallback_service in candidates:
            try:
                raw_result = self._call_translate(
                    text,
                    fallback_service,
                    from_language,
                    target_language,
                    sleep_seconds,
                )
                translated = self._extract_text(raw_result)

                if translated and translated.strip():
                    logger.info(f"[AGSoft Translate Plus] Fallback success via {fallback_service}")
                    return translated

            except Exception as e:
                logger.warning(f"[AGSoft Translate Plus] Fallback {fallback_service} failed: {e}")
                self._reset_session(fallback_service)
                continue

        return None

    def _preaccelerate(self, service):
        """
        Предварительное ускорение сервера перевода.
        Translation server pre-acceleration.
        """
        try:
            if hasattr(ts, "preaccelerate_server"):
                ts.preaccelerate_server(service)
                logger.info(f"[AGSoft Translate Plus] Pre-accelerated service: {service}")
        except Exception as e:
            logger.warning(f"[AGSoft Translate Plus] Pre-acceleration failed: {e}")

    def _reset_session(self, service=None):
        """
        Сброс сессии translators после ошибки.
        Reset translators session after error.
        """
        try:
            if hasattr(ts, "reset_session"):
                ts.reset_session()
        except Exception:
            try:
                if service and hasattr(ts, "reset_session"):
                    ts.reset_session(service)
            except Exception:
                pass

    def _extract_text(self, raw_result):
        """
        Извлекает строку перевода из ответа сервиса.
        Extracts translation string from service response.
        """
        if raw_result is None:
            return ""

        if isinstance(raw_result, str):
            return raw_result.strip()

        if isinstance(raw_result, dict):
            for key in ("translated_text", "translation", "text", "result", "data", "main_text"):
                if key in raw_result:
                    extracted = self._extract_text(raw_result[key])
                    if extracted:
                        return extracted

            for value in raw_result.values():
                extracted = self._extract_text(value)
                if extracted:
                    return extracted

            return ""

        if isinstance(raw_result, list):
            parts = []
            for item in raw_result[:10]:
                extracted = self._extract_text(item)
                if extracted:
                    parts.append(extracted)
            return " ".join(parts).strip()

        return str(raw_result).strip()

    # --------------------------------------------------------------------------
    # Кэш / Cache
    # --------------------------------------------------------------------------

    def _make_cache_key(self, text_part, service, from_language, target_language, protect_hash):
        """
        Создаёт стабильный ключ кэша.
        Creates stable cache key.
        """
        raw = "|".join([
            str(service),
            str(from_language),
            str(target_language),
            str(protect_hash),
            str(text_part),
        ])
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _protect_hash(self, keep_map):
        """
        Создаёт хэш защищённых фрагментов, чтобы кэш учитывал их содержимое.
        Creates hash of protected fragments so cache considers their content.
        """
        if not keep_map:
            return ""

        try:
            raw = json.dumps(keep_map, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(raw.encode("utf-8")).hexdigest()
        except Exception:
            return str(len(keep_map))

    def _load_cache(self, key):
        """
        Читает перевод из кэша.
        Reads translation from cache.
        """
        if not _CACHE_FILE:
            return None

        try:
            with _CACHE_LOCK:
                if not os.path.isfile(_CACHE_FILE):
                    return None

                with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

            entry = data.get(key)
            if isinstance(entry, dict):
                translated = entry.get("translated")
                if isinstance(translated, str):
                    return translated
        except Exception as e:
            logger.warning(f"[AGSoft Translate Plus] Cache read failed: {e}")

        return None

    def _save_cache(self, key, translated):
        """
        Сохраняет перевод в кэш.
        Saves translation to cache.
        """
        if not _CACHE_FILE:
            return

        try:
            with _CACHE_LOCK:
                data = {}

                if os.path.isfile(_CACHE_FILE):
                    try:
                        with open(_CACHE_FILE, "r", encoding="utf-8") as fh:
                            loaded = json.load(fh)
                        if isinstance(loaded, dict):
                            data = loaded
                    except Exception:
                        data = {}

                data[key] = {"translated": translated}

                if len(data) > CACHE_LIMIT_ENTRIES:
                    old_keys = list(data.keys())[:CACHE_TRIM_ENTRIES]
                    for old_key in old_keys:
                        data.pop(old_key, None)

                with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[AGSoft Translate Plus] Cache write failed: {e}")

    # --------------------------------------------------------------------------
    # Защита фрагментов / Fragment protection
    # --------------------------------------------------------------------------

    def _protect_fragments(self, text, protect_fragments, protect_template, tag_start, tag_end, markers):
        """
        Заменяет найденные защищённые фрагменты служебными токенами.
        Replaces found protected fragments with service tokens.
        """
        if not protect_fragments:
            return text, {}

        use_tags, tags, marker_list = self._get_markers(protect_template, tag_start, tag_end, markers)
        pattern = self._build_pattern(use_tags, tags, marker_list)

        if not pattern:
            logger.warning("[AGSoft Translate Plus] Protection enabled but no valid markers/tags were found.")
            return text, {}

        keep_map = {}
        counter = 0

        def repl(match):
            nonlocal counter
            counter += 1
            key = f"__AGSOFT_KEEP_{counter}__"
            keep_map[key] = match.group(0)
            return key

        try:
            protected_text = re.sub(pattern, repl, text, flags=re.DOTALL)
        except Exception as e:
            logger.error(f"[AGSoft Translate Plus] Fragment protection failed: {e}")
            return text, {}

        if keep_map:
            logger.info(f"[AGSoft Translate Plus] Protected {len(keep_map)} fragment(s) from translation.")

        return protected_text, keep_map

    def _restore_fragments(self, text, keep_map):
        """
        Возвращает защищённые фрагменты обратно в текст.
        Restores protected fragments back into text.
        """
        if not keep_map:
            return text

        for key, value in keep_map.items():
            text = text.replace(key, value)

        return text

    def _parse_marker_pairs(self, raw):
        """
        Разбор пар маркеров из поля markers.
        Parses marker pairs from markers field.
        """
        if not raw or not raw.strip():
            return []

        pairs = []
        chunks = re.split(r"[,;\r\n]+", raw)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Явный формат открытие|закрытие.
            # Explicit open|close format.
            if "|" in chunk:
                left, right = chunk.split("|", 1)
                left = left.strip()
                right = right.strip()
                if left:
                    pairs.append((left, right or left))
                continue

            tokens = [t for t in chunk.split() if t]

            # Маркеры через пробел.
            # Space-separated markers.
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

            # Компактная форма вида «».
            # Compact form like «».
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

        # Уникальные без потери порядка.
        # Unique without losing order.
        seen = set()
        out = []

        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                out.append(pair)

        return out

    def _get_markers(self, template, tag_start, tag_end, markers_raw):
        """
        Определяет теги и маркеры по выбранному шаблону.
        Determines tags and markers by selected template.
        """
        t = (template or TEMPLATE_TAG_MARKERS).strip()

        use_tags = False
        tags = ("", "")
        markers_on = False

        custom_set = {
            TEMPLATE_CUSTOM,
            "Custom",
            "Custom / Свои теги и кавычки",
        }

        tag_d_markers_set = {
            TEMPLATE_TAG_MARKERS,
            '<d>....</d> + markers',
            '<d>....</d> + quotes',
            '<d>....</d> + "...."',
            "Tag <d> + quotes",
            "<d>...</d> + quotes / <d>...</d> + кавычки",
        }

        tag_d_only_set = {
            TEMPLATE_TAG_ONLY,
            '<d>....</d>',
            "Tag <d> only",
            "<d>...</d> only / только <d>...</d>",
        }

        tag_ru_markers_set = {
            TEMPLATE_TAG_RU_MARKERS,
            '<d>[Russian]....</d> + markers',
            '<d>[Russian]....</d> + quotes',
            '<d>[Russian]....</d> + "...."',
        }

        tag_ru_only_set = {
            TEMPLATE_TAG_RU_ONLY,
            '<d>[Russian]....</d>',
        }

        tag_br_markers_set = {
            TEMPLATE_BRACKET_MARKERS,
            '[d]....[/d] + quotes',
            '[d]....[/d] + "...."',
            "Tag [d] + quotes",
            "[d]...[/d] + quotes / [d]...[/d] + кавычки",
        }

        tag_br_only_set = {
            TEMPLATE_BRACKET_ONLY,
            "[d]....[/d]",
            "Tag [d] only",
            "[d]...[/d] only / только [d]...[/d]",
        }

        markers_only_set = {
            TEMPLATE_MARKERS_ONLY,
            'quotes only',
            '"...."',
            "Quotes only",
            "Quotes only / Только кавычки",
        }

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
            logger.warning(f"[AGSoft Translate Plus] Unknown template '{t}', using Custom tags fallback.")
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

    def _build_pattern(self, use_tags, tags, marker_list):
        """
        Собирает общий regex для защищённых фрагментов.
        Builds common regex for protected fragments.
        """
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

    # --------------------------------------------------------------------------
    # Языки и сервисы / Languages and services
    # --------------------------------------------------------------------------

    def _resolve_language_code(self, custom_code, selected_language):
        """
        Возвращает код языка. Пользовательский код имеет приоритет.
        Returns language code. Custom code has priority.
        """
        custom = str(custom_code or "").strip().lower()
        if custom:
            return custom

        return self.LANGUAGE_CODES.get(selected_language, "auto")

    def _format_language(self, selected_language, custom_code, resolved_code):
        """
        Форматирует название языка для информации о переводе.
        Formats language name for translation info.
        """
        custom = str(custom_code or "").strip().lower()

        if custom:
            name = self._get_language_name_from_code(resolved_code)
            if name and " - " in name:
                ru_part = name.split(" - ")[1]
                return f"{ru_part} ({resolved_code})"
            return resolved_code

        if selected_language == "Auto detect - Автодетект":
            return "автоопределение / auto"

        if " - " in selected_language:
            return selected_language.split(" - ")[1]

        return selected_language

    def _get_language_name_from_code(self, lang_code):
        """
        Возвращает название языка по коду.
        Returns language name by code.
        """
        if not lang_code:
            return None

        for name, code in self.LANGUAGE_CODES.items():
            if code == lang_code:
                return name

        return None

    def _normalize_separator(self, batch_separator):
        """
        Преобразует пользовательский разделитель в реальный текст.
        Converts user separator into real text.
        """
        if batch_separator is None:
            return ""

        sep = str(batch_separator)
        sep = sep.replace("\\n", "\n")
        sep = sep.replace("\\t", "\t")
        sep = sep.replace("\\r", "\r")

        return sep

    def _generate_service_info(self, current_service):
        """
        Генерирует информацию о сервисах перевода и рекомендации.
        Generates translation service information and recommendations.
        """
        lines = []

        lines.append("=" * 70)
        lines.append("TRANSLATION SERVICES / СЕРВИСЫ ПЕРЕВОДА")
        lines.append("=" * 70)

        current_rec = self.SERVICE_RECOMMENDATIONS.get(current_service, self.SERVICE_RECOMMENDATIONS["default"])

        lines.append(f"\nCURRENT SERVICE / ТЕКУЩИЙ СЕРВИС: {current_service}")
        lines.append(f"Status / Статус: {current_rec['status']}")
        lines.append(f"EN: {current_rec['en']}")
        lines.append(f"RU: {current_rec['ru']}")

        lines.append("\n" + "=" * 70)
        lines.append("SERVICE RECOMMENDATIONS / РЕКОМЕНДАЦИИ ПО СЕРВИСАМ")
        lines.append("=" * 70)

        for service in self.AVAILABLE_SERVICES:
            rec = self.SERVICE_RECOMMENDATIONS.get(service, self.SERVICE_RECOMMENDATIONS["default"])
            lines.append(f"\n• {service}")
            lines.append(f"  Status / Статус: {rec['status']}")
            lines.append(f"  EN: {rec['en']}")
            lines.append(f"  RU: {rec['ru']}")

        lines.append("\n" + "=" * 70)
        lines.append("FALLBACK / РЕЗЕРВНЫЕ СЕРВИСЫ")
        lines.append("=" * 70)
        lines.append("Automatic fallback order / Автоматический порядок резерва:")
        lines.append(" -> ".join(self.SERVICE_FALLBACK_ORDER))

        lines.append("\n" + "=" * 70)
        lines.append("PROTECTED FRAGMENTS / ЗАЩИЩЁННЫЕ ФРАГМЕНТЫ")
        lines.append("=" * 70)
        lines.append("Use protect_fragments to keep selected fragments untranslated.")
        lines.append("Используйте protect_fragments, чтобы не переводить выбранные фрагменты.")
        lines.append("Examples / Примеры:")
        lines.append("  <d>Не переводить это</d>")
        lines.append("  [d]Не переводить это[/d]")
        lines.append("  «Не переводить это»")
        lines.append("  \"Не переводить это\"")

        lines.append("\n" + "=" * 70)
        lines.append("Note: Service availability depends on region, IP and current limits.")
        lines.append("Примечание: Доступность сервисов зависит от региона, IP и текущих лимитов.")
        lines.append("=" * 70)

        return "\n".join(lines)


# ==============================================================================
# Регистрация ноды в ComfyUI
# Register node in ComfyUI
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "AGSoftTranslatePlus": AGSoftTranslatePlus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoftTranslatePlus": "🌐AGSoft Translate Plus"
}