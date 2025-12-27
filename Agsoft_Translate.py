"""
# AGSoft Translate
# Автор: AGSoft
# Дата: 28.12.2025 г.
"""

# Agsoft_Translate.py
import torch
import comfy.utils
import translators as ts
import re

class AgsoftTranslate:
    """
    Advanced text translation node with service selection, language options,
    and additional features like translation direction inversion.
    
    Усовершенствованная нода для перевода текста с выбором сервиса,
    языковых опций и дополнительными функциями, включая инверсию направления перевода.
    """
    
    # Available translation services
    AVAILABLE_SERVICES = [
        "google", "bing", "yandex", "deepl", "papago",
        "baidu", "sogou", "youdao", "caiyun", "reverso",
        "alibaba", "argos", "mirai", "modernMt", "niutrans"
    ]
    
    # Language codes with bilingual names (English - Russian)
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
    
    # Service descriptions in English and Russian
    SERVICE_DESCRIPTIONS = {
        "google": {
            "en": "Most popular, supports all languages, free",
            "ru": "Самый популярный, поддерживает все языки, бесплатный"
        },
        "bing": {
            "en": "Stable, from Microsoft, good quality",
            "ru": "Стабильный, от Microsoft, хорошее качество"
        },
        "yandex": {
            "en": "Excellent for Russian and related languages",
            "ru": "Отличный для русского и родственных языков"
        },
        "deepl": {
            "en": "Highest quality, best for European languages",
            "ru": "Высшее качество, лучший для европейских языков"
        },
        "papago": {
            "en": "Best for Korean and Japanese",
            "ru": "Лучший для корейского и японского"
        },
        "baidu": {
            "en": "Chinese service, good for Asian languages",
            "ru": "Китайский сервис, хорош для азиатских языков"
        },
        "sogou": {
            "en": "Chinese search engine translation",
            "ru": "Перевод от китайской поисковой системы"
        },
        "youdao": {
            "en": "Another good Chinese translation service",
            "ru": "Еще один хороший китайский сервис перевода"
        },
        "default": {
            "en": "Standard translation service",
            "ru": "Стандартный сервис перевода"
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input parameters for the node with detailed tooltips
        """
        # Create lists for ComfyUI
        service_list = list(cls.AVAILABLE_SERVICES)
        language_list = list(cls.LANGUAGE_CODES.keys())
        
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, world!\n" 
                                "Привет, мир!",
                    "dynamicPrompts": True,
                    "tooltip": "Text to translate. Can be multiple lines.\nТекст для перевода. Может быть многострочным."
                }),
                "service": (service_list, {
                    "default": "google",
                    "tooltip": "Translation service to use.\nСервис перевода для использования."
                }),
                "target_language": (language_list, {
                    "default": "Russian - Русский",
                    "tooltip": "Target language for translation. Format: English - Russian.\nЦелевой язык для перевода. Формат: Английский - Русский."
                }),
            },
            "optional": {
                "source_language": (language_list, {
                    "default": "Auto detect - Автодетект",
                    "tooltip": "Source language. 'Auto detect' for automatic detection.\nИсходный язык. 'Auto detect' для автоопределения."
                }),
                "custom_target_lang": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom language code (e.g., 'en', 'ru', 'hy'). Overrides selected target language.\nПользовательский код языка (например, 'en', 'ru', 'hy'). Переопределяет выбранный целевой язык."
                }),
                "custom_source_lang": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom source language code. Leave empty for auto-detection.\nПользовательский код исходного языка. Оставьте пустым для автоопределения."
                }),
                "sleep_seconds": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Delay between translation requests (seconds). Prevents rate limiting.\nЗадержка между запросами перевода (секунды). Предотвращает ограничение частоты."
                }),
                "invert_direction": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Invert",
                    "label_off": "Normal",
                    "tooltip": "Invert translation direction: swap source and target languages.\nИнвертировать направление перевода: поменять местами исходный и целевой языки."
                }),
                "show_service_info": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Show Info",
                    "label_off": "Hide Info",
                    "tooltip": "Show detailed information about translation services and language codes.\nПоказать подробную информацию о сервисах перевода и кодах языков."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("translated_text", "translation_info", "service_info")
    FUNCTION = "translate"
    
    CATEGORY = "AGSoft/Text"
    
    DESCRIPTION = """Advanced Text Translation Node
    Features:
    - Multiple translation services (Google, Bing, Yandex, DeepL, etc.)
    - 80+ languages with bilingual names
    - Custom language code input
    - Translation direction inversion
    - Service information display
    - Automatic fallback on errors
    
    Усовершенствованная нода перевода текста
    Особенности:
    - Множество сервисов перевода (Google, Bing, Yandex, DeepL и др.)
    - 80+ языков с двуязычными названиями
    - Пользовательский ввод кодов языков
    - Инверсия направления перевода
    - Отображение информации о сервисах
    - Автоматический резервный сервис при ошибках"""
    
    def translate(self, text, service, target_language, 
                  source_language="Auto detect - Автодетект", custom_target_lang="", custom_source_lang="",
                  sleep_seconds=0.5, invert_direction=False, show_service_info=False):
        """
        Main translation function with all features
        
        Основная функция перевода со всеми функциями
        """
        # Initialize service info output
        service_info_text = ""
        
        # Generate service information if requested
        if show_service_info:
            service_info_text = self._generate_service_info(service)
        
        # Check for empty text
        if not text or text.strip() == "":
            empty_info = "Пустой текст для перевода / Empty text for translation"
            return ("", empty_info, service_info_text)
        
        try:
            # Handle direction inversion
            if invert_direction:
                # Swap source and target
                temp_source = source_language
                temp_target = target_language
                temp_custom_source = custom_source_lang
                temp_custom_target = custom_target_lang
                
                source_language = temp_target
                target_language = temp_source
                custom_source_lang = temp_custom_target
                custom_target_lang = temp_custom_source
                
                # If source was auto, set it to auto for the new direction
                if source_language == "Auto detect - Автодетект":
                    source_language = "Auto detect - Автодетект"
                    custom_source_lang = ""
                
                print(f"[AgsoftTranslate] Direction inverted")
            
            # Determine target language code
            if custom_target_lang and custom_target_lang.strip():
                # Use custom language code
                target_lang_code = custom_target_lang.strip().lower()
                target_lang_display = f"'{target_lang_code}'"
                target_lang_name = self._get_language_name_from_code(target_lang_code)
                if target_lang_name:
                    target_lang_display = f"{target_lang_name} ({target_lang_code})"
            else:
                # Use selected language from list
                target_lang_code = self.LANGUAGE_CODES[target_language]
                target_lang_display = target_language.split(" - ")[1]  # Russian part
            
            # Determine source language code
            if custom_source_lang and custom_source_lang.strip():
                # Use custom source language
                from_language = custom_source_lang.strip().lower()
                source_lang_display = f"'{from_language}'"
                source_lang_name = self._get_language_name_from_code(from_language)
                if source_lang_name:
                    source_lang_display = f"{source_lang_name} ({from_language})"
            elif source_language != "Auto detect - Автодетект":
                # Use selected source language from list
                from_language = self.LANGUAGE_CODES[source_language]
                source_lang_display = source_language.split(" - ")[1]  # Russian part
            else:
                # Auto-detect - use "auto" string (not None)
                from_language = "auto"
                source_lang_display = "автоопределение / auto-detect"
            
            # Clean the text
            text = text.strip()
            
            # Get service description
            service_desc = self.SERVICE_DESCRIPTIONS.get(service, self.SERVICE_DESCRIPTIONS["default"])
            
            print(f"[AgsoftTranslate] Translating {len(text)} chars via {service}")
            print(f"[AgsoftTranslate] From: {source_lang_display} -> To: {target_lang_display}")
            print(f"[AgsoftTranslate] From language code: {from_language}")
            
            # Perform translation
            translated_text = ts.translate_text(
                query_text=text,
                translator=service,
                from_language=from_language,  # "auto" для автоопределения
                to_language=target_lang_code,
                sleep_seconds=sleep_seconds,
                if_ignore_limit_of_length=True,
                if_ignore_empty_query=True,
                update_session_after_freq=1,
                update_session_after_seconds=1000,
                proxies=None,
                timeout=None,
                if_show_time_stat=False,
                if_print_warning=True,
            )
            
            # Build translation info
            direction_indicator = "⇄ " if invert_direction else "→ "
            translation_info = (
                f"{direction_indicator}Перевод с {source_lang_display} на {target_lang_display} через {service}\n"
                f"{direction_indicator}Translated from {source_lang_display} to {target_lang_display} via {service}"
            )
            
            if custom_target_lang or custom_source_lang:
                translation_info += "\n(пользовательские коды / custom codes)"
            
            # Add service description to info
            service_info_line = f"\n{service_desc['ru']} / {service_desc['en']}"
            translation_info += service_info_line
            
            print(f"[AgsoftTranslate] Success: {len(translated_text)} chars translated")
            
            return (translated_text, translation_info, service_info_text)
            
        except Exception as e:
            error_msg = f"Ошибка перевода через {service} / Translation error via {service}: {str(e)}"
            print(f"[AgsoftTranslate] {error_msg}")
            
            # Try fallback services
            fallback_result = self._try_fallback_services(
                text, target_language, sleep_seconds, 
                custom_target_lang, custom_source_lang, invert_direction
            )
            if fallback_result:
                translated_text, translation_info = fallback_result
                return (translated_text, translation_info, service_info_text)
            
            error_info = f"Ошибка / Error: {str(e)[:100]}"
            if "Connection" in str(e) or "timeout" in str(e).lower():
                error_info += "\nПроверьте подключение к интернету / Check internet connection"
            
            return (text, error_info, service_info_text)
    
    def _try_fallback_services(self, text, target_language, sleep_seconds, 
                               custom_target_lang="", custom_source_lang="", invert_direction=False):
        """
        Try fallback services in case of error
        
        Попробовать резервные сервисы в случае ошибки
        """
        # Determine target language code
        if custom_target_lang and custom_target_lang.strip():
            target_lang_code = custom_target_lang.strip().lower()
        else:
            target_lang_code = self.LANGUAGE_CODES[target_language]
        
        # Determine source language code
        if custom_source_lang and custom_source_lang.strip():
            from_language = custom_source_lang.strip().lower()
        else:
            from_language = "auto"  # Используем "auto" вместо None
        
        # Try popular services in order
        fallback_services = ["google", "bing", "yandex"]
        
        for fallback_service in fallback_services:
            if fallback_service in self.AVAILABLE_SERVICES:
                try:
                    translated_text = ts.translate_text(
                        query_text=text,
                        translator=fallback_service,
                        from_language=from_language,
                        to_language=target_lang_code,
                        sleep_seconds=sleep_seconds,
                        if_ignore_limit_of_length=True,
                    )
                    
                    # Get language display names
                    if custom_target_lang:
                        target_display = f"'{custom_target_lang}'"
                        target_name = self._get_language_name_from_code(custom_target_lang)
                        if target_name:
                            target_display = f"{target_name} ({custom_target_lang})"
                    else:
                        target_display = target_language.split(" - ")[1]
                    
                    direction_indicator = "⇄ " if invert_direction else "→ "
                    info = (
                        f"{direction_indicator}Переведено на {target_display} через {fallback_service} (резервный)\n"
                        f"{direction_indicator}Translated to {target_display} via {fallback_service} (fallback)"
                    )
                    
                    print(f"[AgsoftTranslate] Success via fallback service {fallback_service}")
                    
                    return (translated_text, info)
                    
                except Exception as e:
                    print(f"[AgsoftTranslate] Fallback {fallback_service} failed: {e}")
                    continue
        
        return None
    
    def _generate_service_info(self, current_service):
        """
        Generate detailed information about translation services and language codes
        
        Сгенерировать подробную информацию о сервисах перевода и кодах языков
        """
        info_lines = []
        
        # Header
        info_lines.append("=" * 70)
        info_lines.append("TRANSLATION SERVICES & LANGUAGE CODES INFO")
        info_lines.append("ИНФОРМАЦИЯ О СЕРВИСАХ ПЕРЕВОДА И КОДАХ ЯЗЫКОВ")
        info_lines.append("=" * 70)
        
        # Current service
        info_lines.append(f"\nCURRENT SERVICE / ТЕКУЩИЙ СЕРВИС: {current_service}")
        service_desc = self.SERVICE_DESCRIPTIONS.get(current_service, self.SERVICE_DESCRIPTIONS["default"])
        info_lines.append(f"Description / Описание: {service_desc['en']}")
        info_lines.append(f"Описание: {service_desc['ru']}")
        
        # All available services
        info_lines.append(f"\nAVAILABLE SERVICES / ДОСТУПНЫЕ СЕРВИСЫ ({len(self.AVAILABLE_SERVICES)}):")
        info_lines.append("-" * 40)
        
        for service in self.AVAILABLE_SERVICES:
            desc = self.SERVICE_DESCRIPTIONS.get(service, self.SERVICE_DESCRIPTIONS["default"])
            info_lines.append(f"• {service}: {desc['en']}")
            info_lines.append(f"  {service}: {desc['ru']}")
        
        # Language codes section - для custom_source_lang и custom_target_lang
        info_lines.append("\n" + "=" * 70)
        info_lines.append("LANGUAGE CODES FOR CUSTOM FIELDS")
        info_lines.append("КОДЫ ЯЗЫКОВ ДЛЯ ПОЛЬЗОВАТЕЛЬСКИХ ПОЛЕЙ")
        info_lines.append("=" * 70)
        
        info_lines.append("\nUse these codes in 'custom_source_lang' and 'custom_target_lang' fields:")
        info_lines.append("Используйте эти коды в полях 'custom_source_lang' и 'custom_target_lang':")
        info_lines.append("")
        
        # Group languages for better readability
        language_groups = {
            "Most Common / Самые распространенные": [
                "Auto detect - Автодетект",
                "English - Английский", 
                "Russian - Русский",
                "Chinese - Китайский",
                "Spanish - Испанский",
                "French - Французский",
                "German - Немецкий",
                "Japanese - Японский",
            ],
            "Eastern European / Восточноевропейские": [
                "Ukrainian - Украинский",
                "Polish - Польский",
                "Czech - Чешский",
                "Slovak - Словацкий",
                "Hungarian - Венгерский",
                "Romanian - Румынский",
                "Bulgarian - Болгарский",
                "Croatian - Хорватский",
                "Serbian - Сербский",
                "Slovenian - Словенский",
                "Bosnian - Боснийский",
            ],
            "Baltic & Nordic / Балтийские и Северные": [
                "Lithuanian - Литовский",
                "Latvian - Латышский",
                "Estonian - Эстонский",
                "Finnish - Финский",
                "Swedish - Шведский",
                "Norwegian - Норвежский",
                "Danish - Датский",
                "Icelandic - Исландский",
            ],
            "Caucasus & Central Asia / Кавказ и Центральная Азия": [
                "Armenian - Армянский",
                "Azerbaijani - Азербайджанский",
                "Georgian - Грузинский",
                "Kazakh - Казахский",
                "Uzbek - Узбекский",
                "Kyrgyz - Киргизский",
                "Tajik - Таджикский",
                "Turkmen - Туркменский",
            ],
            "Middle Eastern / Ближневосточные": [
                "Arabic - Арабский",
                "Turkish - Турецкий",
                "Hebrew - Иврит",
                "Persian - Персидский",
                "Urdu - Урду",
            ],
            "South Asian / Южноазиатские": [
                "Hindi - Хинди",
                "Bengali - Бенгальский",
                "Punjabi - Пенджабский",
                "Tamil - Тамильский",
                "Telugu - Телугу",
                "Marathi - Маратхи",
                "Gujarati - Гуджарати",
                "Malayalam - Малаялам",
                "Kannada - Каннада",
                "Sinhala - Сингальский",
                "Nepali - Непальский",
            ],
            "Southeast Asian / Юго-Восточной Азии": [
                "Thai - Тайский",
                "Vietnamese - Вьетнамский",
                "Indonesian - Индонезийский",
                "Malay - Малайский",
                "Filipino - Филиппинский",
                "Burmese - Бирманский",
                "Khmer - Кхмерский",
                "Lao - Лаосский",
            ],
            "Other European / Другие европейские": [
                "Dutch - Нидерландский",
                "Italian - Итальянский",
                "Portuguese - Португальский",
                "Greek - Греческий",
                "Catalan - Каталанский",
                "Galician - Галисийский",
                "Basque - Баскский",
                "Welsh - Валлийский",
                "Irish - Ирландский",
                "Albanian - Албанский",
                "Macedonian - Македонский",
                "Belarusian - Белорусский",
            ],
            "African / Африканские": [
                "Swahili - Суахили",
                "Afrikaans - Африкаанс",
                "Amharic - Амхарский",
                "Yoruba - Йоруба",
                "Hausa - Хауса",
                "Igbo - Игбо",
                "Somali - Сомалийский",
                "Zulu - Зулу",
            ],
            "Other / Другие": [
                "Mongolian - Монгольский",
                "Maltese - Мальтийский",
                "Esperanto - Эсперанто",
                "Latin - Латинский",
                "Sanskrit - Санскрит",
                "Maithili - Майтхили",
                "Oriya - Ория",
                "Assamese - Ассамский",
            ]
        }
        
        # Display languages in groups
        for group_name, languages in language_groups.items():
            info_lines.append(f"\n{group_name}:")
            info_lines.append("-" * 40)
            
            # Format: English - Russian (code)
            for lang_display in languages:
                if lang_display in self.LANGUAGE_CODES:
                    lang_code = self.LANGUAGE_CODES[lang_display]
                    # Split the display name
                    if " - " in lang_display:
                        en_part, ru_part = lang_display.split(" - ")
                        info_lines.append(f"  {en_part} - {ru_part} ({lang_code})")
                    else:
                        info_lines.append(f"  {lang_display} ({lang_code})")
        
        # Example usage
        info_lines.append("\n" + "=" * 70)
        info_lines.append("EXAMPLES / ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
        info_lines.append("=" * 70)
        
        examples = [
            ("English to Russian", "Английский на русский", "en", "ru"),
            ("Russian to Armenian", "Русский на армянский", "ru", "hy"),
            ("Auto to Belarusian", "Авто на белорусский", "auto", "be"),
            ("Azerbaijani to English", "Азербайджанский на английский", "az", "en"),
            ("French to Chinese", "Французский на китайский", "fr", "zh"),
        ]
        
        info_lines.append("\nFor custom_source_lang / custom_target_lang fields:")
        info_lines.append("Для полей custom_source_lang / custom_target_lang:")
        info_lines.append("")
        
        for en_desc, ru_desc, from_code, to_code in examples:
            info_lines.append(f"• {en_desc} / {ru_desc}:")
            info_lines.append(f"  custom_source_lang = '{from_code}'")
            info_lines.append(f"  custom_target_lang = '{to_code}'")
            info_lines.append("")
        
        # Quick reference for added languages
        info_lines.append("\nQUICK REFERENCE / БЫСТРАЯ СПРАВКА:")
        info_lines.append("-" * 40)
        quick_ref = [
            ("Armenian", "Армянский", "hy"),
            ("Azerbaijani", "Азербайджанский", "az"),
            ("Belarusian", "Белорусский", "be"),
        ]
        
        for en_name, ru_name, code in quick_ref:
            info_lines.append(f"  {en_name} / {ru_name}: code = '{code}'")
        
        # Footer
        info_lines.append("\n" + "=" * 70)
        info_lines.append("Note: Codes are ISO 639-1 (2-letter) or ISO 639-2 (3-letter)")
        info_lines.append("Примечание: Коды соответствуют ISO 639-1 (2-буквенные) или ISO 639-2 (3-буквенные)")
        info_lines.append("=" * 70)
        
        return "\n".join(info_lines)
    
    def _get_language_name_from_code(self, lang_code):
        """
        Get language name from code
        
        Получить название языка по коду
        """
        if not lang_code:
            return None
        
        for name, code in self.LANGUAGE_CODES.items():
            if code == lang_code:
                # Return both English and Russian parts
                return name
        
        return None


# Register the single node in ComfyUI
NODE_CLASS_MAPPINGS = {
    "AgsoftTranslate": AgsoftTranslate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgsoftTranslate": "AGSoft Translate",
}

# For backward compatibility
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']