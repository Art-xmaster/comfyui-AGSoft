// ============================================================================
// AGSoft_Text_Editor_emoji.js
// ============================================================================
// Модуль: 😀 эмодзи для AGSoft Text Editor
// Описание / Description:
// Данные эмодзи и панель выбора с поиском по английскому названию, русскому
// названию и дополнительным словам. Модуль только экспортирует данные и
// функцию сборки панели, поэтому загрузка расширений его не активирует.
// ---
// Emoji data and a picker panel searchable by English name, Russian name and
// extra keywords. The module only exports data and a mount function, so the
// extension loader importing it causes no side effects.
//
// Возможности / Features:
// ⚡ Восемь тематических групп: смайлы, люди, животные, еда, спорт, места,
//   предметы, символы и флаги
//   Eight themed groups: smileys, people, animals, food, sport, places,
//   objects, symbols and flags
// ⚡ Каждая запись хранит символ, английское имя, русское имя и ключевые слова
//   Every entry keeps the char, English name, Russian name and keywords
// ⚡ Поиск требует все введённые слова и сортирует выдачу по позиции совпадения
//   Search requires every typed word and ranks hits by match position
// ⚡ Недавние эмодзи запоминаются в браузере и показываются первыми
//   Recent emoji are remembered in the browser and shown first
// ⚡ Указатели групп прыгают к разделу и подсвечиваются при прокрутке
//   Group tabs jump to a section and highlight while scrolling
// ⚡ Заголовки групп липнут к верху, список прокручивается колесом и клавишами
//   Group titles stick to the top, the list scrolls by wheel and by keys
// ⚡ Внизу показывается крупный символ и двуязычное название под курсором
//   The bottom shows a large char and the bilingual name under the cursor
// ⚡ Enter вставляет первый результат, стрелки ходят по сетке, Esc закрывает
//   Enter inserts the first hit, arrows walk the grid, Esc closes the panel
// ⚡ Размеры панели задаются инлайном, поэтому прокрутка работает в любом хосте
//   Panel size is set inline, so scrolling works inside any host element
// ⚡ Цвета наследуются от темы редактора через переменные --ags-*
//   Colours inherit the editor theme through the --ags-* variables
// ⚡ Плоский индекс поиска строится один раз на страницу
//   The flat search index is built once per page
//
// Экспорт / Exports:
// EMOJI_GROUPS                     данные групп / group data
// searchEmoji(query)               поиск по словам / keyword search
// mountEmojiPicker(host, options)  сборка панели / panel builder
// options.onPick(char)             вставка символа / char insertion
// options.onClose()                закрытие панели / panel closing
// возвращает focus, refresh, destroy / returns focus, refresh, destroy
//
// Формат записи / Entry format:
// [символ, английское имя, русское имя, дополнительные слова]
// [char, english name, russian name, extra keywords]
//
// Лимиты / Limits:
// Недавние 30 символов, выдача поиска до 240 записей
// Recent 30 chars, search output up to 240 entries
//
// Зависимости / Dependencies:
// Нет импортов, модуль автономен / No imports, the module is self-contained
//
// Автор / Author: AGSoft
// Дата / Date: 14.11.2026
// ============================================================================
console.log("[AGSoft Text Editor] emoji module loaded v11.14 (scroll and sections fixed)");

const STYLE_ID = "ags-te-emoji-style";
const RECENT_KEY = "agsoft_te_emoji_recent";
const RECENT_MAX = 30;
const RESULT_MAX = 240;
const CELL = 30;
const PANEL_W = 344;
const PANEL_H = 408;

// [char, english name, russian name, extra keywords]
// [символ, английское имя, русское имя, дополнительные слова]
export const EMOJI_GROUPS = [
    {
        id: "smileys",
        icon: "😀",
        label: "Smileys / Смайлы",
        items: [
            ["😀", "grinning", "улыбка", "smile happy grin радость"],
            ["😃", "smiley", "радость", "smile happy joy"],
            ["😄", "smile", "смех", "happy joy laugh"],
            ["😁", "grin", "ухмылка", "beaming teeth зубы"],
            ["😆", "laughing", "хохот", "laugh xd смех"],
            ["😅", "sweat smile", "смех с потом", "nervous нервно"],
            ["🤣", "rolling laughing", "ржач", "rofl lol катаюсь"],
            ["😂", "joy", "слёзы смеха", "tears lol плачу от смеха"],
            ["🙂", "slight smile", "лёгкая улыбка", "calm спокойно"],
            ["🙃", "upside down", "перевёрнутый", "silly irony ирония"],
            ["😉", "wink", "подмигивание", "flirt намёк"],
            ["😊", "blush", "румянец", "shy довольный милый"],
            ["😇", "innocent", "ангел", "halo нимб святой"],
            ["🥰", "hearts face", "влюблённый", "love сердечки"],
            ["😍", "heart eyes", "глаза сердца", "love влюблён"],
            ["🤩", "star struck", "звёзды в глазах", "wow восторг"],
            ["😘", "kiss", "поцелуй", "love чмок"],
            ["😗", "kissing", "целую", "kiss поцелуй"],
            ["😚", "kissing closed eyes", "поцелуй глаза", "kiss"],
            ["😙", "kissing smile", "поцелуй улыбка", "kiss"],
            ["🥲", "smile tear", "улыбка со слезой", "sad грусть"],
            ["😋", "yum", "вкусно", "tongue язык еда"],
            ["😛", "tongue", "язык", "teasing дразню"],
            ["😜", "wink tongue", "язык подмиг", "joke шучу"],
            ["🤪", "zany", "дурачусь", "crazy безумный"],
            ["😝", "squint tongue", "щурюсь язык", "joke шутка"],
            ["🤑", "money mouth", "деньги", "rich богатство"],
            ["🤗", "hugging", "обнимаю", "hug объятие"],
            ["🤭", "hand over mouth", "хихикаю", "giggle смешок"],
            ["🤫", "shushing", "тс молчок", "quiet тихо секрет"],
            ["🤔", "thinking", "думаю", "think размышление хм"],
            ["🤐", "zipper mouth", "рот на замке", "silent молчу"],
            ["🤨", "raised eyebrow", "бровь", "skeptic недоверие"],
            ["😐", "neutral", "нейтральный", "meh безразлично"],
            ["😑", "expressionless", "без выражения", "blank пусто"],
            ["😶", "no mouth", "без рта", "silent молчание"],
            ["😏", "smirk", "усмешка", "sly хитро"],
            ["😒", "unamused", "недоволен", "meh скучно"],
            ["🙄", "eye roll", "закатываю глаза", "annoyed бесит"],
            ["😬", "grimace", "гримаса", "awkward неловко"],
            ["🤥", "lying", "вру", "pinocchio нос ложь"],
            ["😌", "relieved", "облегчение", "calm спокойно"],
            ["😔", "pensive", "задумчивый", "sad грусть"],
            ["😪", "sleepy", "сонный", "tired устал"],
            ["🤤", "drooling", "слюни", "yum вкусно хочу"],
            ["😴", "sleeping", "сплю", "sleep сон zzz"],
            ["😷", "mask", "маска", "sick болезнь врач"],
            ["🤒", "thermometer", "температура", "sick больной"],
            ["🤕", "bandage", "бинт", "hurt травма"],
            ["🤢", "nauseated", "тошнит", "sick плохо"],
            ["🤮", "vomiting", "рвота", "sick тошнота"],
            ["🤧", "sneezing", "чихаю", "cold простуда"],
            ["🥵", "hot face", "жарко", "heat жара"],
            ["🥶", "cold face", "холодно", "freeze мороз"],
            ["🥴", "woozy", "пьяный", "drunk dizzy"],
            ["😵", "dizzy", "головокружение", "spiral обморок"],
            ["🤯", "exploding head", "взрыв мозга", "shock шок"],
            ["🤠", "cowboy", "ковбой", "west дикий запад"],
            ["🥳", "partying", "праздник", "party вечеринка др"],
            ["😎", "sunglasses", "крутой", "cool очки"],
            ["🤓", "nerd", "ботан", "geek очки умный"],
            ["🧐", "monocle", "монокль", "inspect изучаю"],
            ["😕", "confused", "смущён", "puzzled непонятно"],
            ["😟", "worried", "обеспокоен", "worry тревога"],
            ["🙁", "slight frown", "грусть", "sad печаль"],
            ["😮", "open mouth", "удивление", "wow ого"],
            ["😯", "hushed", "ошеломлён", "surprise удивлён"],
            ["😲", "astonished", "изумление", "shock шок"],
            ["😳", "flushed", "покраснел", "embarrassed смущение"],
            ["🥺", "pleading", "умоляю", "puppy щенячьи глаза"],
            ["😨", "fearful", "страх", "scared боюсь"],
            ["😰", "anxious", "тревога", "nervous пот"],
            ["😢", "cry", "плач", "tear слеза грусть"],
            ["😭", "sob", "рыдаю", "cry плач слёзы"],
            ["😱", "scream", "крик", "horror ужас"],
            ["😖", "confounded", "растерян", "frustrated"],
            ["😞", "disappointed", "разочарован", "sad грусть"],
            ["😩", "weary", "устал", "tired усталость"],
            ["😫", "tired", "измотан", "exhausted устал"],
            ["🥱", "yawning", "зеваю", "sleep сон скука"],
            ["😤", "triumph", "злость пар", "angry生气"],
            ["😡", "rage", "гнев", "angry злюсь"],
            ["😠", "angry", "сердитый", "mad злой"],
            ["🤬", "cursing", "ругань", "swear мат"],
            ["😈", "smiling imp", "чёртик", "devil дьявол"],
            ["👿", "imp", "злой чёрт", "devil демон"],
            ["💀", "skull", "череп", "death смерть"],
            ["💩", "poop", "какашка", "shit плохо"],
            ["🤡", "clown", "клоун", "circus цирк"],
            ["👻", "ghost", "привидение", "ghost призрак бу"],
            ["👽", "alien", "пришелец", "ufo инопланетянин"],
            ["👾", "monster", "монстр", "game игра пиксель"],
            ["🤖", "robot", "робот", "bot ai бот"],
        ],
    },
    {
        id: "people",
        icon: "👍",
        label: "People / Люди и жесты",
        items: [
            ["👋", "wave", "привет", "hello hi пока рука"],
            ["🤚", "raised back hand", "рука", "stop стоп"],
            ["🖐️", "hand", "ладонь", "fingers пальцы"],
            ["✋", "raised hand", "стоп", "stop high five"],
            ["🖖", "vulcan", "вулкан", "spock стар трек"],
            ["👌", "ok hand", "окей", "ok хорошо отлично"],
            ["🤌", "pinched fingers", "щепотка", "italy итальянский"],
            ["🤏", "pinching", "чуть-чуть", "small мало"],
            ["✌️", "victory", "виктория", "peace мир два"],
            ["🤞", "crossed fingers", "пальцы скрещены", "luck удача"],
            ["🤟", "love you", "люблю", "rock жест"],
            ["🤘", "horns", "коза", "rock рок металл"],
            ["🤙", "call me", "позвони", "phone звонок"],
            ["👈", "point left", "влево", "left указатель"],
            ["👉", "point right", "вправо", "right указатель"],
            ["👆", "point up", "вверх", "up указатель"],
            ["👇", "point down", "вниз", "down указатель"],
            ["☝️", "index up", "палец вверх", "attention внимание"],
            ["👍", "thumbs up", "палец вверх", "like лайк класс ок"],
            ["👎", "thumbs down", "палец вниз", "dislike дизлайк плохо"],
            ["✊", "fist", "кулак", "power сила"],
            ["👊", "punch", "удар", "fist кулак бро"],
            ["🤛", "left fist", "кулак влево", "bump"],
            ["🤜", "right fist", "кулак вправо", "bump"],
            ["👏", "clap", "аплодисменты", "applause хлопаю браво"],
            ["🙌", "raised hands", "ура", "hooray руки вверх"],
            ["👐", "open hands", "раскрытые ладони", "open"],
            ["🤲", "palms up", "ладони", "pray просьба"],
            ["🤝", "handshake", "рукопожатие", "deal сделка"],
            ["🙏", "pray", "молитва", "thanks спасибо пожалуйста"],
            ["✍️", "writing hand", "пишу", "write заметка"],
            ["💅", "nail polish", "маникюр", "nails ногти"],
            ["🤳", "selfie", "селфи", "photo фото"],
            ["💪", "muscle", "бицепс", "strong сила мощь"],
            ["🦵", "leg", "нога", "kick"],
            ["🦶", "foot", "ступня", "feet нога"],
            ["👂", "ear", "ухо", "hear слушаю"],
            ["👃", "nose", "нос", "smell нюх"],
            ["🧠", "brain", "мозг", "mind ум идея"],
            ["👀", "eyes", "глаза", "look смотрю"],
            ["👁️", "eye", "глаз", "watch слежу"],
            ["👅", "tongue", "язык", "taste вкус"],
            ["👄", "mouth", "губы", "lips рот"],
            ["👶", "baby", "малыш", "child ребёнок"],
            ["🧒", "child", "ребёнок", "kid"],
            ["👦", "boy", "мальчик", "kid парень"],
            ["👧", "girl", "девочка", "kid"],
            ["🧑", "person", "человек", "human"],
            ["👨", "man", "мужчина", "male парень"],
            ["👩", "woman", "женщина", "female девушка"],
            ["🧓", "older person", "пожилой", "old возраст"],
            ["👴", "old man", "дедушка", "grandpa старик"],
            ["👵", "old woman", "бабушка", "grandma старушка"],
            ["🙅", "no good", "нельзя", "no нет запрет"],
            ["🙆", "ok person", "можно", "ok согласен"],
            ["💁", "tipping hand", "информация", "info помощь"],
            ["🙋", "raising hand", "поднимаю руку", "question вопрос"],
            ["🙇", "bow", "поклон", "sorry извинение"],
            ["🤦", "facepalm", "фейспалм", "рука лицо ужас"],
            ["🤷", "shrug", "пожимаю плечами", "не знаю"],
            ["👮", "police", "полицейский", "cop мент"],
            ["🕵️", "detective", "детектив", "spy шпион"],
            ["👷", "worker", "строитель", "build рабочий"],
            ["🤴", "prince", "принц", "king король"],
            ["👸", "princess", "принцесса", "queen королева"],
            ["🦸", "superhero", "супергерой", "hero герой"],
            ["🦹", "villain", "злодей", "bad зло"],
            ["🧙", "mage", "маг", "wizard волшебник"],
            ["🧚", "fairy", "фея", "magic волшебство"],
            ["🧛", "vampire", "вампир", "dracula дракула"],
            ["🧜", "mermaid", "русалка", "sea море"],
            ["🧝", "elf", "эльф", "fantasy фэнтези"],
            ["🧞", "genie", "джинн", "wish желание"],
            ["🧟", "zombie", "зомби", "undead мертвец"],
            ["🚶", "walking", "иду", "walk пешком"],
            ["🏃", "running", "бегу", "run бег"],
            ["💃", "dancer", "танцую", "dance танец"],
            ["🕺", "man dancing", "танцует", "dance танец"],
            ["👫", "couple", "пара", "love любовь"],
            ["👪", "family", "семья", "family родители дети"],
        ],
    },
    {
        id: "animals",
        icon: "🐯",
        label: "Animals / Животные и природа",
        items: [
            ["🐶", "dog face", "собака", "puppy щенок пёс"],
            ["🐕", "dog", "пёс", "dog собака"],
            ["🐩", "poodle", "пудель", "dog собака"],
            ["🐺", "wolf", "волк", "wild дикий"],
            ["🦊", "fox", "лиса", "fox лисичка рыжая хитрая"],
            ["🦝", "raccoon", "енот", "raccoon"],
            ["🐱", "cat face", "кот", "cat кошка kitty"],
            ["🐈", "cat", "кошка", "cat кот"],
            ["🦁", "lion", "лев", "king царь зверей"],
            ["🐯", "tiger face", "тигр", "tiger тигр полосатый хищник"],
            ["🐅", "tiger", "тигр", "tiger тигрица большой кот"],
            ["🐆", "leopard", "леопард", "spots пятна"],
            ["🐴", "horse face", "лошадь", "horse конь"],
            ["🐎", "horse", "конь", "race скачки бег"],
            ["🦄", "unicorn", "единорог", "magic магия радуга"],
            ["🦓", "zebra", "зебра", "stripes полоски"],
            ["🦌", "deer", "олень", "deer рога"],
            ["🐮", "cow face", "корова", "cow му"],
            ["🐂", "ox", "бык", "bull"],
            ["🐄", "cow", "корова", "milk молоко"],
            ["🐷", "pig face", "свинья", "pig поросёнок"],
            ["🐖", "pig", "свинья", "pork"],
            ["🐗", "boar", "кабан", "wild дикий"],
            ["🐽", "pig nose", "пятачок", "nose нос"],
            ["🐏", "ram", "баран", "sheep"],
            ["🐑", "sheep", "овца", "wool шерсть"],
            ["🐐", "goat", "коза", "goat"],
            ["🐪", "camel", "верблюд", "desert пустыня"],
            ["🐫", "two hump camel", "двугорбый верблюд", "desert"],
            ["🦙", "llama", "лама", "alpaca альпака"],
            ["🦒", "giraffe", "жираф", "tall высокий"],
            ["🐘", "elephant", "слон", "big большой"],
            ["🦏", "rhino", "носорог", "horn рог"],
            ["🦛", "hippo", "бегемот", "hippo гиппопотам"],
            ["🐭", "mouse face", "мышь", "mouse мышка"],
            ["🐁", "mouse", "мышка", "small маленькая"],
            ["🐀", "rat", "крыса", "rat"],
            ["🐹", "hamster", "хомяк", "pet питомец"],
            ["🐰", "rabbit face", "кролик", "bunny заяц"],
            ["🐇", "rabbit", "кролик", "bunny заяц"],
            ["🐿️", "chipmunk", "бурундук", "squirrel белка"],
            ["🦔", "hedgehog", "ёж", "spikes колючки"],
            ["🦇", "bat", "летучая мышь", "night ночь"],
            ["🐻", "bear", "медведь", "bear мишка"],
            ["🐨", "koala", "коала", "australia"],
            ["🐼", "panda", "панда", "china китай бамбук"],
            ["🦥", "sloth", "ленивец", "slow медленный"],
            ["🦦", "otter", "выдра", "water вода"],
            ["🦨", "skunk", "скунс", "smell запах"],
            ["🦘", "kangaroo", "кенгуру", "jump прыжок"],
            ["🦡", "badger", "барсук", "badger"],
            ["🐾", "paw prints", "лапы", "paws следы"],
            ["🦃", "turkey", "индейка", "bird птица"],
            ["🐔", "chicken", "курица", "bird птица"],
            ["🐓", "rooster", "петух", "morning утро"],
            ["🐣", "hatching chick", "цыплёнок", "egg яйцо"],
            ["🐤", "chick", "цыплёнок", "baby малыш"],
            ["🐦", "bird", "птица", "bird"],
            ["🐧", "penguin", "пингвин", "cold холод"],
            ["🕊️", "dove", "голубь", "peace мир"],
            ["🦅", "eagle", "орёл", "bird хищник"],
            ["🦆", "duck", "утка", "bird птица"],
            ["🦢", "swan", "лебедь", "white белый"],
            ["🦉", "owl", "сова", "night ночь мудрость"],
            ["🦩", "flamingo", "фламинго", "pink розовый"],
            ["🦚", "peacock", "павлин", "feathers перья"],
            ["🦜", "parrot", "попугай", "bird птица"],
            ["🐸", "frog", "лягушка", "frog ква"],
            ["🐊", "crocodile", "крокодил", "reptile"],
            ["🐢", "turtle", "черепаха", "slow медленная"],
            ["🦎", "lizard", "ящерица", "reptile"],
            ["🐍", "snake", "змея", "snake"],
            ["🐲", "dragon face", "дракон", "dragon"],
            ["🐉", "dragon", "дракон", "fantasy фэнтези"],
            ["🦕", "sauropod", "динозавр", "dino"],
            ["🦖", "t rex", "тираннозавр", "dino динозавр"],
            ["🐳", "whale", "кит", "ocean океан"],
            ["🐬", "dolphin", "дельфин", "sea море"],
            ["🐟", "fish", "рыба", "fish"],
            ["🐠", "tropical fish", "рыбка", "fish аквариум"],
            ["🐡", "blowfish", "фугу", "fish рыба"],
            ["🦈", "shark", "акула", "sea море"],
            ["🐙", "octopus", "осьминог", "sea море"],
            ["🐚", "shell", "ракушка", "sea море"],
            ["🐌", "snail", "улитка", "slow медленно"],
            ["🦋", "butterfly", "бабочка", "butterfly"],
            ["🐛", "bug", "гусеница", "worm червяк"],
            ["🐜", "ant", "муравей", "insect"],
            ["🐝", "bee", "пчела", "honey мёд"],
            ["🐞", "ladybug", "божья коровка", "bug жук"],
            ["🕷️", "spider", "паук", "spider"],
            ["🕸️", "web", "паутина", "spider паук"],
            ["🦂", "scorpion", "скорпион", "scorpion"],
            ["🦟", "mosquito", "комар", "bite укус"],
            ["🦠", "microbe", "микроб", "virus вирус"],
            ["💐", "bouquet", "букет", "flowers цветы"],
            ["🌸", "cherry blossom", "сакура", "spring весна цветок"],
            ["🌹", "rose", "роза", "flower love цветок"],
            ["🥀", "wilted flower", "увядшая роза", "sad грусть"],
            ["🌺", "hibiscus", "гибискус", "flower цветок"],
            ["🌻", "sunflower", "подсолнух", "sun солнце"],
            ["🌼", "blossom", "цветок", "flower"],
            ["🌷", "tulip", "тюльпан", "flower цветок"],
            ["🌱", "seedling", "росток", "grow рост"],
            ["🌲", "evergreen", "ель", "tree ёлка дерево"],
            ["🌳", "tree", "дерево", "tree"],
            ["🌴", "palm", "пальма", "beach пляж"],
            ["🌵", "cactus", "кактус", "desert пустыня"],
            ["🌾", "rice", "колос", "wheat пшеница"],
            ["🌿", "herb", "трава", "green зелень"],
            ["🍀", "four leaf clover", "клевер", "luck удача"],
            ["🍁", "maple leaf", "лист клёна", "autumn осень"],
            ["🍂", "fallen leaves", "листья", "autumn осень"],
            ["🍃", "leaf", "лист", "wind ветер"],
        ],
    },
    {
        id: "food",
        icon: "🍕",
        label: "Food / Еда и напитки",
        items: [
            ["🍇", "grapes", "виноград", "fruit фрукт"],
            ["🍈", "melon", "дыня", "fruit фрукт"],
            ["🍉", "watermelon", "арбуз", "fruit фрукт лето"],
            ["🍊", "tangerine", "мандарин", "orange апельсин"],
            ["🍋", "lemon", "лимон", "sour кислый"],
            ["🍌", "banana", "банан", "fruit фрукт"],
            ["🍍", "pineapple", "ананас", "fruit фрукт"],
            ["🥭", "mango", "манго", "fruit фрукт"],
            ["🍎", "apple", "яблоко", "fruit фрукт"],
            ["🍏", "green apple", "зелёное яблоко", "fruit"],
            ["🍐", "pear", "груша", "fruit фрукт"],
            ["🍑", "peach", "персик", "fruit фрукт"],
            ["🍒", "cherries", "вишня", "berry ягода"],
            ["🍓", "strawberry", "клубника", "berry ягода"],
            ["🥝", "kiwi", "киви", "fruit фрукт"],
            ["🍅", "tomato", "помидор", "vegetable овощ"],
            ["🥥", "coconut", "кокос", "tropical"],
            ["🥑", "avocado", "авокадо", "green зелёный"],
            ["🍆", "eggplant", "баклажан", "vegetable овощ"],
            ["🥔", "potato", "картошка", "vegetable овощ"],
            ["🥕", "carrot", "морковь", "vegetable овощ"],
            ["🌽", "corn", "кукуруза", "vegetable овощ"],
            ["🌶️", "hot pepper", "острый перец", "spicy острое"],
            ["🥒", "cucumber", "огурец", "vegetable овощ"],
            ["🥬", "leafy green", "салат", "cabbage капуста"],
            ["🥦", "broccoli", "брокколи", "vegetable овощ"],
            ["🧄", "garlic", "чеснок", "spice"],
            ["🧅", "onion", "лук", "vegetable"],
            ["🍄", "mushroom", "гриб", "mushroom"],
            ["🥜", "peanuts", "арахис", "nuts орехи"],
            ["🍞", "bread", "хлеб", "bread"],
            ["🥐", "croissant", "круассан", "bakery выпечка"],
            ["🥖", "baguette", "багет", "bread хлеб"],
            ["🥨", "pretzel", "крендель", "bakery"],
            ["🥯", "bagel", "бублик", "bread"],
            ["🥞", "pancakes", "блины", "breakfast завтрак"],
            ["🧇", "waffle", "вафля", "breakfast"],
            ["🧀", "cheese", "сыр", "cheese"],
            ["🍖", "meat on bone", "мясо на кости", "meat"],
            ["🍗", "poultry leg", "куриная ножка", "chicken курица"],
            ["🥩", "steak", "стейк", "meat мясо"],
            ["🥓", "bacon", "бекон", "meat мясо"],
            ["🍔", "burger", "бургер", "fastfood"],
            ["🍟", "fries", "картошка фри", "fastfood"],
            ["🍕", "pizza", "пицца", "fastfood еда"],
            ["🌭", "hot dog", "хот-дог", "fastfood"],
            ["🥪", "sandwich", "сэндвич", "bread хлеб"],
            ["🌮", "taco", "тако", "mexican"],
            ["🌯", "burrito", "буррито", "mexican"],
            ["🥙", "falafel", "шаурма", "kebab кебаб"],
            ["🥚", "egg", "яйцо", "egg"],
            ["🍳", "cooking", "яичница", "fry жарю"],
            ["🍲", "stew", "суп", "soup рагу"],
            ["🥣", "bowl", "миска", "cereal каша"],
            ["🥗", "salad", "салат", "healthy здоровое"],
            ["🍿", "popcorn", "попкорн", "movie кино"],
            ["🧂", "salt", "соль", "spice"],
            ["🍱", "bento", "бенто", "japan япония"],
            ["🍙", "rice ball", "онигири", "rice рис"],
            ["🍚", "rice", "рис", "rice"],
            ["🍛", "curry", "карри", "spicy острое"],
            ["🍜", "ramen", "рамен", "noodles лапша"],
            ["🍝", "spaghetti", "спагетти", "pasta паста"],
            ["🍣", "sushi", "суши", "japan япония рыба"],
            ["🍤", "fried shrimp", "креветка", "tempura темпура"],
            ["🥟", "dumpling", "пельмень", "varenik вареник"],
            ["🦀", "crab", "краб", "sea море"],
            ["🦞", "lobster", "лобстер", "sea море"],
            ["🦐", "shrimp", "креветка", "sea море"],
            ["🦑", "squid", "кальмар", "sea море"],
            ["🍦", "ice cream", "мороженое", "sweet сладкое"],
            ["🍩", "doughnut", "пончик", "sweet сладкое"],
            ["🍪", "cookie", "печенье", "sweet сладкое"],
            ["🎂", "birthday cake", "торт", "birthday день рождения"],
            ["🍰", "cake", "пирожное", "sweet торт"],
            ["🧁", "cupcake", "кекс", "sweet сладкое"],
            ["🥧", "pie", "пирог", "bakery выпечка"],
            ["🍫", "chocolate", "шоколад", "sweet сладкое"],
            ["🍬", "candy", "конфета", "sweet сладкое"],
            ["🍭", "lollipop", "леденец", "sweet сладкое"],
            ["🍯", "honey", "мёд", "sweet сладкое"],
            ["🥛", "milk", "молоко", "drink напиток"],
            ["☕", "coffee", "кофе", "drink напиток утро"],
            ["🍵", "tea", "чай", "drink напиток"],
            ["🍾", "champagne", "шампанское", "party праздник"],
            ["🍷", "wine", "вино", "drink алкоголь"],
            ["🍸", "cocktail", "коктейль", "drink алкоголь"],
            ["🍹", "tropical drink", "тропический коктейль", "drink"],
            ["🍺", "beer", "пиво", "drink алкоголь"],
            ["🍻", "beers", "пиво чоканье", "cheers"],
            ["🥂", "cheers", "бокалы", "toast тост"],
            ["🥃", "whisky", "виски", "drink алкоголь"],
            ["🥤", "cup", "стакан", "drink напиток"],
            ["🧃", "juice", "сок", "drink напиток"],
            ["🧊", "ice", "лёд", "cold холод"],
        ],
    },
    {
        id: "activity",
        icon: "⚽",
        label: "Activity / Спорт и занятия",
        items: [
            ["⚽", "soccer", "футбол", "ball мяч sport"],
            ["🏀", "basketball", "баскетбол", "ball мяч"],
            ["🏈", "football", "американский футбол", "ball мяч"],
            ["⚾", "baseball", "бейсбол", "ball мяч"],
            ["🎾", "tennis", "теннис", "ball мяч"],
            ["🏐", "volleyball", "волейбол", "ball мяч"],
            ["🏉", "rugby", "регби", "ball мяч"],
            ["🎱", "billiards", "бильярд", "pool пул"],
            ["🏓", "ping pong", "пинг-понг", "tennis настольный"],
            ["🏸", "badminton", "бадминтон", "shuttle волан"],
            ["🏒", "hockey", "хоккей", "ice лёд"],
            ["⛳", "golf", "гольф", "flag флаг"],
            ["🏹", "archery", "стрельба из лука", "bow лук"],
            ["🎣", "fishing", "рыбалка", "fish рыба"],
            ["🥊", "boxing", "бокс", "glove перчатка"],
            ["🥋", "martial arts", "единоборства", "karate карате"],
            ["🛹", "skateboard", "скейтборд", "skate скейт"],
            ["🛼", "roller skate", "ролики", "skate"],
            ["⛸️", "ice skate", "коньки", "ice лёд"],
            ["🎿", "skis", "лыжи", "winter зима"],
            ["🏂", "snowboard", "сноуборд", "winter зима"],
            ["🏋️", "weight lifting", "штанга", "gym зал сила"],
            ["🤸", "cartwheel", "колесо", "gymnastics гимнастика"],
            ["🤺", "fencing", "фехтование", "sword шпага"],
            ["🧘", "yoga", "йога", "meditation медитация"],
            ["🏄", "surfing", "сёрфинг", "wave волна"],
            ["🏊", "swimming", "плавание", "water вода"],
            ["🚴", "cycling", "велосипед", "bike вело"],
            ["🏆", "trophy", "кубок", "win победа чемпион"],
            ["🥇", "gold medal", "золото", "first первое место"],
            ["🥈", "silver medal", "серебро", "second второе"],
            ["🥉", "bronze medal", "бронза", "third третье"],
            ["🏅", "medal", "медаль", "award награда"],
            ["🎫", "ticket", "билет", "ticket"],
            ["🎪", "circus", "цирк", "tent шапито"],
            ["🎭", "masks", "театр", "theatre маски"],
            ["🎨", "palette", "палитра", "art рисование цвет"],
            ["🎬", "clapper", "хлопушка", "movie кино съёмка"],
            ["🎤", "microphone", "микрофон", "sing пение"],
            ["🎧", "headphones", "наушники", "music музыка"],
            ["🎼", "score", "ноты", "music музыка"],
            ["🎹", "piano", "пианино", "music музыка"],
            ["🥁", "drum", "барабан", "music музыка"],
            ["🎷", "saxophone", "саксофон", "music джаз"],
            ["🎺", "trumpet", "труба", "music музыка"],
            ["🎸", "guitar", "гитара", "music рок"],
            ["🎻", "violin", "скрипка", "music классика"],
            ["🎲", "dice", "кубик", "random случай кости"],
            ["♟️", "chess", "шахматы", "chess pawn пешка"],
            ["🎯", "dart", "дартс", "target мишень цель"],
            ["🎳", "bowling", "боулинг", "ball шар"],
            ["🎮", "video game", "игра", "game геймпад"],
            ["🎰", "slot machine", "автомат", "casino казино"],
            ["🧩", "puzzle", "пазл", "puzzle головоломка"],
        ],
    },
    {
        id: "travel",
        icon: "🚗",
        label: "Travel / Транспорт и места",
        items: [
            ["🚗", "car", "машина", "auto авто"],
            ["🚕", "taxi", "такси", "car машина"],
            ["🚙", "suv", "внедорожник", "car машина"],
            ["🚌", "bus", "автобус", "transport"],
            ["🏎️", "racing car", "гоночная машина", "speed скорость"],
            ["🚓", "police car", "полиция", "cop мент"],
            ["🚑", "ambulance", "скорая", "medical медицина"],
            ["🚒", "fire engine", "пожарная", "fire огонь"],
            ["🚚", "truck", "грузовик", "delivery доставка"],
            ["🚜", "tractor", "трактор", "farm ферма"],
            ["🏍️", "motorcycle", "мотоцикл", "bike мото"],
            ["🛵", "scooter", "скутер", "moped мопед"],
            ["🚲", "bike", "велосипед", "bicycle вело"],
            ["🛴", "kick scooter", "самокат", "scooter"],
            ["🚨", "siren", "мигалка", "alert тревога"],
            ["🚋", "tram", "трамвай", "transport"],
            ["🚂", "locomotive", "паровоз", "train поезд"],
            ["🚆", "train", "поезд", "rail железная дорога"],
            ["🚇", "metro", "метро", "subway подземка"],
            ["🚉", "station", "вокзал", "station станция"],
            ["✈️", "airplane", "самолёт", "flight полёт"],
            ["🛫", "departure", "вылет", "flight вылет"],
            ["🛬", "arrival", "прилёт", "flight посадка"],
            ["🚀", "rocket", "ракета", "space космос старт"],
            ["🛸", "ufo", "нло", "alien тарелка"],
            ["🚁", "helicopter", "вертолёт", "fly полёт"],
            ["⛵", "sailboat", "парусник", "sea море"],
            ["🚤", "speedboat", "катер", "boat лодка"],
            ["🛳️", "cruise ship", "лайнер", "cruise круиз"],
            ["🚢", "ship", "корабль", "sea море"],
            ["⚓", "anchor", "якорь", "sea море"],
            ["⛽", "fuel", "заправка", "gas бензин"],
            ["🚧", "construction", "стройка", "work работы"],
            ["🚦", "traffic light", "светофор", "traffic"],
            ["🗺️", "map", "карта", "map мир"],
            ["🗿", "moai", "моаи", "statue статуя"],
            ["🗽", "liberty", "статуя свободы", "usa сша"],
            ["🗼", "tower", "башня", "tokyo токио"],
            ["🏰", "castle", "замок", "castle крепость"],
            ["🏟️", "stadium", "стадион", "sport спорт"],
            ["🎡", "ferris wheel", "колесо обозрения", "park парк"],
            ["🎢", "roller coaster", "американские горки", "park"],
            ["⛲", "fountain", "фонтан", "water вода"],
            ["🏖️", "beach", "пляж", "sea море отдых"],
            ["🏝️", "island", "остров", "sea море"],
            ["🏜️", "desert", "пустыня", "sand песок"],
            ["🌋", "volcano", "вулкан", "lava лава"],
            ["⛰️", "mountain", "гора", "mountain"],
            ["🏔️", "snow mountain", "снежная гора", "snow снег"],
            ["⛺", "tent", "палатка", "camping кемпинг"],
            ["🏠", "house", "дом", "home домой"],
            ["🏡", "house garden", "дом с садом", "home дом"],
            ["🏢", "office", "офис", "building здание"],
            ["🏥", "hospital", "больница", "medical"],
            ["🏦", "bank", "банк", "money деньги"],
            ["🏨", "hotel", "отель", "hotel гостиница"],
            ["🏫", "school", "школа", "school учёба"],
            ["⛪", "church", "церковь", "church храм"],
            ["🕌", "mosque", "мечеть", "mosque"],
            ["🌃", "night city", "ночной город", "night ночь"],
            ["🏙️", "cityscape", "город", "city мегаполис"],
            ["🌅", "sunrise", "рассвет", "morning утро"],
            ["🌇", "sunset", "закат", "evening вечер"],
            ["🌉", "bridge night", "мост ночью", "bridge мост"],
            ["🌌", "milky way", "млечный путь", "space космос"],
            ["🎆", "fireworks", "фейерверк", "party праздник"],
            ["🌈", "rainbow", "радуга", "rainbow цвета"],
            ["☀️", "sun", "солнце", "sun лето"],
            ["⛅", "partly cloudy", "облачно", "cloud облако"],
            ["☁️", "cloud", "облако", "cloud"],
            ["🌧️", "rain", "дождь", "rain"],
            ["⛈️", "storm", "гроза", "thunder гром"],
            ["🌨️", "snow", "снег", "winter зима"],
            ["❄️", "snowflake", "снежинка", "cold холод"],
            ["⛄", "snowman", "снеговик", "winter зима"],
            ["🌪️", "tornado", "торнадо", "wind ветер"],
            ["🌫️", "fog", "туман", "fog"],
            ["🌊", "wave", "волна", "sea море вода"],
            ["💧", "droplet", "капля", "water вода"],
            ["🔥", "fire", "огонь", "fire hot пламя жар"],
            ["⭐", "star", "звезда", "star"],
            ["🌟", "glowing star", "сияющая звезда", "star блеск"],
            ["✨", "sparkles", "блёстки", "sparkle искры магия"],
            ["⚡", "zap", "молния", "lightning энергия"],
            ["💥", "boom", "взрыв", "explosion бах"],
            ["🌙", "moon", "луна", "night ночь"],
            ["🌚", "new moon", "новолуние", "moon луна"],
            ["🌝", "full moon", "полнолуние", "moon луна"],
            ["🪐", "planet", "планета", "saturn сатурн космос"],
            ["🌍", "earth", "земля", "world мир глобус"],
        ],
    },
    {
        id: "objects",
        icon: "💡",
        label: "Objects / Предметы",
        items: [
            ["⌚", "watch", "часы", "time время"],
            ["📱", "phone", "телефон", "mobile смартфон"],
            ["💻", "laptop", "ноутбук", "computer компьютер"],
            ["⌨️", "keyboard", "клавиатура", "typing печать"],
            ["🖥️", "desktop", "монитор", "computer пк"],
            ["🖨️", "printer", "принтер", "print печать"],
            ["🖱️", "mouse", "мышь", "computer компьютер"],
            ["💾", "floppy", "дискета", "save сохранить"],
            ["💿", "cd", "диск", "disk cd"],
            ["📷", "camera", "фотоаппарат", "photo фото"],
            ["📹", "video camera", "видеокамера", "video видео"],
            ["🔍", "magnifier", "лупа", "search поиск"],
            ["🔎", "magnifier right", "лупа", "search поиск"],
            ["🕯️", "candle", "свеча", "light свет"],
            ["💡", "bulb", "лампочка", "idea идея свет"],
            ["🔦", "flashlight", "фонарик", "light свет"],
            ["📔", "notebook", "блокнот", "notes записи"],
            ["📕", "book", "книга", "book читать"],
            ["📖", "open book", "открытая книга", "read читать"],
            ["📚", "books", "книги", "library библиотека"],
            ["📓", "notebook", "тетрадь", "notes"],
            ["📜", "scroll", "свиток", "scroll древний"],
            ["📄", "page", "документ", "document файл"],
            ["📰", "newspaper", "газета", "news новости"],
            ["🔖", "bookmark", "закладка", "bookmark"],
            ["🏷️", "label", "ярлык", "tag метка"],
            ["💰", "money bag", "мешок денег", "money деньги"],
            ["💵", "dollar", "доллар", "money деньги"],
            ["💶", "euro", "евро", "money деньги"],
            ["💸", "money wings", "деньги улетают", "money трата"],
            ["💳", "card", "карта", "credit кредитка"],
            ["🧾", "receipt", "чек", "receipt"],
            ["✉️", "envelope", "конверт", "mail письмо"],
            ["📧", "email", "почта", "mail письмо"],
            ["📦", "package", "посылка", "box коробка"],
            ["📥", "inbox", "входящие", "inbox"],
            ["📤", "outbox", "исходящие", "outbox"],
            ["✏️", "pencil", "карандаш", "write писать"],
            ["🖊️", "pen", "ручка", "write писать"],
            ["🖌️", "brush", "кисть", "paint рисовать"],
            ["📝", "memo", "заметка", "note писать текст"],
            ["💼", "briefcase", "портфель", "work работа"],
            ["📁", "folder", "папка", "folder файлы"],
            ["📂", "open folder", "открытая папка", "folder"],
            ["📅", "calendar", "календарь", "date дата"],
            ["📈", "chart up", "график рост", "growth рост"],
            ["📉", "chart down", "график падение", "drop спад"],
            ["📊", "bar chart", "диаграмма", "stats статистика"],
            ["📋", "clipboard", "планшет", "list список"],
            ["📌", "pin", "булавка", "pin закрепить"],
            ["📍", "round pin", "метка", "location геолокация"],
            ["📎", "paperclip", "скрепка", "attach вложение"],
            ["📏", "ruler", "линейка", "measure измерение"],
            ["✂️", "scissors", "ножницы", "cut резать"],
            ["🗑️", "wastebasket", "корзина", "trash мусор удалить"],
            ["🔒", "lock", "замок закрыт", "lock защита"],
            ["🔓", "unlock", "замок открыт", "unlock открыто"],
            ["🔑", "key", "ключ", "key пароль"],
            ["🔨", "hammer", "молоток", "tool инструмент"],
            ["🪓", "axe", "топор", "tool инструмент"],
            ["🛠️", "tools", "инструменты", "tools ремонт"],
            ["⚔️", "swords", "мечи", "battle битва"],
            ["🛡️", "shield", "щит", "defense защита"],
            ["🔧", "wrench", "гаечный ключ", "fix чинить"],
            ["⚙️", "gear", "шестерёнка", "settings настройки"],
            ["⚖️", "scales", "весы", "balance баланс право"],
            ["🔗", "link", "ссылка", "link цепь"],
            ["🧰", "toolbox", "ящик инструментов", "tools"],
            ["🧲", "magnet", "магнит", "magnet"],
            ["🧪", "test tube", "пробирка", "science химия"],
            ["🧬", "dna", "днк", "dna генетика"],
            ["🔬", "microscope", "микроскоп", "science наука"],
            ["🔭", "telescope", "телескоп", "space космос"],
            ["📡", "satellite", "спутник", "antenna антенна"],
            ["💉", "syringe", "шприц", "medical укол"],
            ["💊", "pill", "таблетка", "medical лекарство"],
            ["🩹", "bandage", "пластырь", "medical"],
            ["🚪", "door", "дверь", "door"],
            ["🛏️", "bed", "кровать", "sleep сон"],
            ["🛋️", "couch", "диван", "sofa мебель"],
            ["🪑", "chair", "стул", "chair мебель"],
            ["🚿", "shower", "душ", "water вода"],
            ["🛁", "bathtub", "ванна", "bath"],
            ["🧼", "soap", "мыло", "clean чистота"],
            ["🧹", "broom", "метла", "clean уборка"],
            ["🧯", "extinguisher", "огнетушитель", "fire пожар"],
            ["🛒", "cart", "тележка", "shop магазин"],
            ["🔮", "crystal ball", "шар предсказаний", "magic магия"],
            ["🧿", "nazar", "назар", "eye глаз оберег"],
            ["🎁", "gift", "подарок", "gift подарок др"],
            ["🎈", "balloon", "шарик", "party праздник"],
            ["🎉", "party popper", "хлопушка", "party праздник ура"],
            ["🎊", "confetti", "конфетти", "party праздник"],
        ],
    },
    {
        id: "symbols",
        icon: "❤️",
        label: "Symbols / Символы и флаги",
        items: [
            ["❤️", "red heart", "красное сердце", "love любовь"],
            ["🧡", "orange heart", "оранжевое сердце", "love"],
            ["💛", "yellow heart", "жёлтое сердце", "love"],
            ["💚", "green heart", "зелёное сердце", "love"],
            ["💙", "blue heart", "синее сердце", "love"],
            ["💜", "purple heart", "фиолетовое сердце", "love"],
            ["🖤", "black heart", "чёрное сердце", "love dark"],
            ["🤍", "white heart", "белое сердце", "love"],
            ["💔", "broken heart", "разбитое сердце", "sad боль"],
            ["💕", "two hearts", "два сердца", "love любовь"],
            ["💞", "revolving hearts", "сердца кружатся", "love"],
            ["💓", "beating heart", "сердце бьётся", "love pulse"],
            ["💖", "sparkling heart", "сердце блестит", "love"],
            ["💘", "heart arrow", "сердце со стрелой", "cupid купидон"],
            ["💝", "heart ribbon", "сердце подарок", "love gift"],
            ["☮️", "peace", "мир", "peace пацифик"],
            ["✝️", "cross", "крест", "cross religion"],
            ["☯️", "yin yang", "инь ян", "balance баланс"],
            ["♈", "aries", "овен", "zodiac зодиак"],
            ["♉", "taurus", "телец", "zodiac зодиак"],
            ["♊", "gemini", "близнецы", "zodiac зодиак"],
            ["♋", "cancer", "рак", "zodiac зодиак"],
            ["♌", "leo", "лев", "zodiac зодиак"],
            ["♍", "virgo", "дева", "zodiac зодиак"],
            ["♎", "libra", "весы", "zodiac зодиак"],
            ["♏", "scorpio", "скорпион", "zodiac зодиак"],
            ["♐", "sagittarius", "стрелец", "zodiac зодиак"],
            ["♑", "capricorn", "козерог", "zodiac зодиак"],
            ["♒", "aquarius", "водолей", "zodiac зодиак"],
            ["♓", "pisces", "рыбы", "zodiac зодиак"],
            ["⚛️", "atom", "атом", "science наука"],
            ["✅", "check", "галочка", "done готово ok"],
            ["✔️", "check mark", "галочка", "done готово"],
            ["❌", "cross mark", "крест", "no нет ошибка"],
            ["❎", "cross button", "крест", "no нет"],
            ["➕", "plus", "плюс", "add добавить"],
            ["➖", "minus", "минус", "remove убрать"],
            ["♾️", "infinity", "бесконечность", "infinity"],
            ["❓", "question", "вопрос", "question"],
            ["❗", "exclamation", "восклицание", "important важно"],
            ["‼️", "double exclamation", "два восклицания", "alert"],
            ["⁉️", "exclamation question", "вопрос восклицание", "what"],
            ["💯", "hundred", "сто", "100 perfect идеально"],
            ["🔴", "red circle", "красный круг", "red"],
            ["🟠", "orange circle", "оранжевый круг", "orange"],
            ["🟡", "yellow circle", "жёлтый круг", "yellow"],
            ["🟢", "green circle", "зелёный круг", "green"],
            ["🔵", "blue circle", "синий круг", "blue"],
            ["🟣", "purple circle", "фиолетовый круг", "purple"],
            ["⚫", "black circle", "чёрный круг", "black"],
            ["⚪", "white circle", "белый круг", "white"],
            ["🟥", "red square", "красный квадрат", "red"],
            ["🟩", "green square", "зелёный квадрат", "green"],
            ["🟦", "blue square", "синий квадрат", "blue"],
            ["⬛", "black square", "чёрный квадрат", "black"],
            ["⬜", "white square", "белый квадрат", "white"],
            ["🔺", "triangle", "треугольник", "triangle"],
            ["🔻", "triangle down", "треугольник вниз", "triangle"],
            ["⬆️", "up", "вверх", "arrow up"],
            ["⬇️", "down", "вниз", "arrow down"],
            ["⬅️", "left", "влево", "arrow left"],
            ["➡️", "right", "вправо", "arrow right"],
            ["🔄", "sync", "обновить", "refresh sync"],
            ["🔝", "top", "наверх", "top"],
            ["🎵", "note", "нота", "music музыка"],
            ["🎶", "notes", "ноты", "music музыка"],
            ["♻️", "recycle", "переработка", "eco экология"],
            ["🔱", "trident", "трезубец", "trident"],
            ["📛", "name badge", "бейдж", "badge имя"],
            ["🚫", "prohibited", "запрещено", "no ban нельзя"],
            ["⛔", "no entry", "въезд запрещён", "stop стоп"],
            ["🔞", "underage", "18+", "adult только взрослым"],
            ["☢️", "radioactive", "радиация", "radiation"],
            ["☣️", "biohazard", "биологическая опасность", "biohazard"],
            ["🏁", "checkered flag", "финиш", "finish флаг"],
            ["🚩", "triangular flag", "флаг", "flag красный"],
            ["🏳️", "white flag", "белый флаг", "surrender сдача"],
            ["🏴", "black flag", "чёрный флаг", "flag"],
            ["🇷🇺", "russia", "россия", "flag флаг ru"],
            ["🇺🇸", "usa", "сша", "flag флаг usa америка"],
            ["🇬🇧", "uk", "британия", "flag флаг england"],
            ["🇨🇳", "china", "китай", "flag флаг china"],
            ["🇯🇵", "japan", "япония", "flag флаг japan"],
            ["🇩🇪", "germany", "германия", "flag флаг"],
            ["🇫🇷", "france", "франция", "flag флаг"],
            ["🇮🇹", "italy", "италия", "flag флаг"],
            ["🇪🇸", "spain", "испания", "flag флаг"],
            ["🇺🇦", "ukraine", "украина", "flag флаг"],
            ["🇰🇿", "kazakhstan", "казахстан", "flag флаг"],
            ["🇧🇾", "belarus", "беларусь", "flag флаг"],
        ],
    },
];

const CSS = `
.ags-emoji-search-row {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 6px 4px;
}
.ags-emoji-search {
    flex: 1 1 auto;
    min-width: 0;
    padding: 5px 9px;
    border: 1px solid color-mix(in srgb, var(--ags-muted, #9aa3c0) 35%, transparent);
    border-radius: 7px;
    background: var(--ags-ink, #2b2f4d);
    color: var(--ags-text, #e7eaf6);
    font: 12px "Segoe UI", system-ui, sans-serif;
    outline: none;
}
.ags-emoji-search:focus { border-color: var(--ags-accent, #6cb8ff); }
.ags-emoji-clear {
    flex: 0 0 auto;
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: var(--ags-muted, #9aa3c0);
    font: 600 12px "Segoe UI", system-ui, sans-serif;
    padding: 4px 7px;
    cursor: pointer;
}
.ags-emoji-clear:hover {
    color: var(--ags-danger, #ff5d73);
    background: color-mix(in srgb, var(--ags-danger, #ff5d73) 14%, transparent);
}
.ags-emoji-tabs {
    flex: 0 0 auto;
    display: flex;
    gap: 2px;
    padding: 0 6px 4px;
    border-bottom: 1px solid color-mix(in srgb, var(--ags-muted, #9aa3c0) 22%, transparent);
}
.ags-emoji-tab {
    flex: 1 1 auto;
    border: 0;
    border-bottom: 2px solid transparent;
    border-radius: 6px 6px 0 0;
    background: transparent;
    font-size: 16px;
    line-height: 1.25;
    padding: 3px 0;
    cursor: pointer;
    opacity: 0.62;
    transition: opacity 110ms, background 110ms, border-color 110ms;
}
.ags-emoji-tab:hover {
    opacity: 1;
    background: color-mix(in srgb, var(--ags-accent, #6cb8ff) 16%, transparent);
}
.ags-emoji-tab.ags-on {
    opacity: 1;
    border-bottom-color: var(--ags-accent, #6cb8ff);
    background: color-mix(in srgb, var(--ags-accent, #6cb8ff) 20%, transparent);
}
.ags-emoji-list {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 0 6px 6px;
    scrollbar-width: thin;
    scrollbar-color: color-mix(in srgb, var(--ags-accent, #6cb8ff) 45%, var(--ags-ink, #2b2f4d))
        var(--ags-ink, #2b2f4d);
}
.ags-emoji-list::-webkit-scrollbar { width: 10px; }
.ags-emoji-list::-webkit-scrollbar-track {
    background: var(--ags-ink, #2b2f4d);
    border-radius: 5px;
}
.ags-emoji-list::-webkit-scrollbar-thumb {
    background: color-mix(in srgb, var(--ags-accent, #6cb8ff) 45%, var(--ags-ink, #2b2f4d));
    border-radius: 5px;
    border: 2px solid var(--ags-ink, #2b2f4d);
}
.ags-emoji-list::-webkit-scrollbar-thumb:hover { background: var(--ags-accent, #6cb8ff); }
.ags-emoji-title {
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 6px 4px 4px;
    font-size: 9px;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: var(--ags-muted, #9aa3c0);
    background: var(--ags-surface, #3f4570);
    border-bottom: 1px solid color-mix(in srgb, var(--ags-muted, #9aa3c0) 16%, transparent);
}
.ags-emoji-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(30px, 1fr));
    gap: 2px;
    padding-top: 4px;
}
.ags-emoji-cell {
    border: 0;
    border-radius: 7px;
    background: transparent;
    font-size: 20px;
    line-height: 1;
    padding: 4px 0;
    cursor: pointer;
    transition: background 110ms, transform 110ms;
}
.ags-emoji-cell:hover {
    background: color-mix(in srgb, var(--ags-accent, #6cb8ff) 24%, transparent);
    transform: scale(1.16);
}
.ags-emoji-cell:focus {
    outline: none;
    background: color-mix(in srgb, var(--ags-accent, #6cb8ff) 38%, transparent);
}
.ags-emoji-empty {
    padding: 18px 8px;
    font-size: 10px;
    text-align: center;
    color: var(--ags-muted, #9aa3c0);
}
.ags-emoji-preview {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 9px;
    border-top: 1px solid color-mix(in srgb, var(--ags-muted, #9aa3c0) 22%, transparent);
    background: color-mix(in srgb, var(--ags-ink, #2b2f4d) 55%, var(--ags-surface, #3f4570));
    font-size: 10px;
    color: var(--ags-muted, #9aa3c0);
}
.ags-emoji-preview-char { font-size: 22px; line-height: 1; }
.ags-emoji-preview-text {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ags-emoji-preview-hint { flex: 0 0 auto; font-size: 9px; opacity: 0.75; }
`;

let INDEX = null;

function injectStyle() {
    const el = document.getElementById(STYLE_ID) || document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = CSS;
    if (!el.parentNode) document.head.appendChild(el);
}

// Flat search index is built once per page / Плоский индекс строится один раз
function buildIndex() {
    if (INDEX) return INDEX;
    const list = [];
    for (const group of EMOJI_GROUPS) {
        for (const item of group.items) {
            list.push({
                c: item[0],
                en: item[1],
                ru: item[2],
                hay: (item[1] + " " + item[2] + " " + (item[3] || "")).toLowerCase(),
            });
        }
    }
    INDEX = list;
    return list;
}

export function searchEmoji(query) {
    const text = String(query || "").trim().toLowerCase();
    if (!text) return [];
    const parts = text.split(/\s+/);
    const hits = [];
    for (const item of buildIndex()) {
        let score = 0;
        let matched = true;
        for (const part of parts) {
            const at = item.hay.indexOf(part);
            if (at < 0) {
                matched = false;
                break;
            }
            score += at === 0 ? 0 : (at < 14 ? 1 : 2);
        }
        if (matched) hits.push({ item: item, score: score });
    }
    hits.sort((a, b) => a.score - b.score);
    return hits.slice(0, RESULT_MAX).map((row) => row.item);
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

function pushRecent(char) {
    try {
        const list = readRecent().filter((c) => c !== char);
        list.unshift(char);
        localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, RECENT_MAX)));
    } catch (err) {
        console.warn("[AGSoft Text Editor] Emoji history not stored: " +
            String((err && err.message) || err));
    }
}

export function mountEmojiPicker(host, options) {
    injectStyle();
    const opts = options || {};
    const index = buildIndex();
    const byChar = new Map();
    for (const item of index) byChar.set(item.c, item);

    host.innerHTML = "";
    host.classList.add("ags-emoji");
    // Inline box sizing beats host styles and keeps the list scrollable.
    // Инлайновые размеры перебивают стили хоста и оставляют список прокручиваемым.
    host.style.boxSizing = "border-box";
    host.style.flexDirection = "column";
    host.style.width = PANEL_W + "px";
    host.style.height = "min(" + PANEL_H + "px, calc(100% - 44px))";
    host.style.minHeight = "230px";
    host.style.maxHeight = "none";
    host.style.padding = "0";
    host.style.overflow = "hidden";
    host.style.color = "var(--ags-text, #e7eaf6)";
    host.style.fontFamily = "\"Segoe UI\", system-ui, sans-serif";

    const searchRow = document.createElement("div");
    searchRow.className = "ags-emoji-search-row";
    const search = document.createElement("input");
    search.className = "ags-emoji-search";
    search.placeholder = "Search emoji / Поиск эмодзи";
    search.spellcheck = false;
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "ags-emoji-clear";
    clearBtn.textContent = "✕";
    clearBtn.title = "Clear search / Очистить поиск";
    searchRow.append(search, clearBtn);

    const tabs = document.createElement("div");
    tabs.className = "ags-emoji-tabs";
    const list = document.createElement("div");
    list.className = "ags-emoji-list";
    const preview = document.createElement("div");
    preview.className = "ags-emoji-preview";
    const previewChar = document.createElement("span");
    previewChar.className = "ags-emoji-preview-char";
    previewChar.textContent = "🙂";
    const previewText = document.createElement("span");
    previewText.className = "ags-emoji-preview-text";
    previewText.textContent = "Pick an emoji / Выберите эмодзи";
    const previewHint = document.createElement("span");
    previewHint.className = "ags-emoji-preview-hint";
    previewHint.textContent = "Enter · ↑↓";
    preview.append(previewChar, previewText, previewHint);
    host.append(searchRow, tabs, list, preview);

    let cells = [];
    let sections = new Map();
    const tabButtons = new Map();

    function showPreview(item) {
        if (!item) return;
        previewChar.textContent = item.c;
        previewText.textContent = item.en + " / " + item.ru;
    }

    function pick(char) {
        pushRecent(char);
        if (opts.onPick) opts.onPick(char);
    }

    function makeCell(item) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ags-emoji-cell";
        btn.textContent = item.c;
        btn.title = item.en + " / " + item.ru;
        btn.onmouseenter = () => showPreview(item);
        btn.onfocus = () => showPreview(item);
        btn.onclick = () => pick(item.c);
        cells.push(btn);
        return btn;
    }

    function makeGrid(items) {
        const grid = document.createElement("div");
        grid.className = "ags-emoji-grid";
        for (const item of items) {
            if (item) grid.appendChild(makeCell(item));
        }
        return grid;
    }

    function setActiveTab(id) {
        for (const pair of tabButtons) {
            pair[1].classList.toggle("ags-on", pair[0] === id);
        }
    }

    function renderBrowse() {
        cells = [];
        sections = new Map();
        list.innerHTML = "";
        const chars = readRecent().filter((c) => byChar.has(c));
        if (chars.length) {
            const box = document.createElement("div");
            const title = document.createElement("div");
            title.className = "ags-emoji-title";
            title.textContent = "Recent / Недавние";
            box.appendChild(title);
            box.appendChild(makeGrid(chars.map((c) => byChar.get(c))));
            list.appendChild(box);
        }
        for (const group of EMOJI_GROUPS) {
            const section = document.createElement("div");
            section.dataset.group = group.id;
            const title = document.createElement("div");
            title.className = "ags-emoji-title";
            title.textContent = group.label;
            section.appendChild(title);
            section.appendChild(makeGrid(group.items.map((row) => byChar.get(row[0]))));
            list.appendChild(section);
            sections.set(group.id, section);
        }
        list.scrollTop = 0;
        setActiveTab(EMOJI_GROUPS[0].id);
    }

    function renderSearch(query) {
        cells = [];
        sections = new Map();
        list.innerHTML = "";
        const hits = searchEmoji(query);
        setActiveTab("");
        if (!hits.length) {
            const empty = document.createElement("div");
            empty.className = "ags-emoji-empty";
            empty.textContent = "Nothing found / Ничего не найдено";
            list.appendChild(empty);
            previewText.textContent = "Try another word / Попробуйте другое слово";
            return;
        }
        const title = document.createElement("div");
        title.className = "ags-emoji-title";
        title.textContent = "Found / Найдено: " + hits.length;
        list.appendChild(title);
        list.appendChild(makeGrid(hits));
        list.scrollTop = 0;
        showPreview(hits[0]);
    }

    function refresh() {
        const query = search.value.trim();
        if (query) renderSearch(query);
        else renderBrowse();
    }

    for (const group of EMOJI_GROUPS) {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = "ags-emoji-tab";
        tab.textContent = group.icon;
        tab.title = group.label;
        tab.onclick = () => {
            if (search.value) {
                search.value = "";
                renderBrowse();
            }
            const section = sections.get(group.id);
            if (section) list.scrollTop = section.offsetTop;
            setActiveTab(group.id);
        };
        tabButtons.set(group.id, tab);
        tabs.appendChild(tab);
    }

    // Scroll spy keeps the active section in sync / Подсветка следует за прокруткой
    list.addEventListener("scroll", () => {
        if (search.value.trim() || !sections.size) return;
        const top = list.scrollTop + 6;
        let current = EMOJI_GROUPS[0].id;
        for (const group of EMOJI_GROUPS) {
            const section = sections.get(group.id);
            if (section && section.offsetTop <= top) current = group.id;
        }
        setActiveTab(current);
    });

    clearBtn.onclick = () => {
        search.value = "";
        refresh();
        search.focus();
    };

    search.addEventListener("input", refresh);
    search.addEventListener("keydown", (e) => {
        e.stopPropagation();
        if (e.key === "Escape") {
            e.preventDefault();
            if (search.value) {
                search.value = "";
                refresh();
                return;
            }
            if (opts.onClose) opts.onClose();
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (cells.length) cells[0].click();
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            if (cells.length) cells[0].focus();
        }
    });

    list.addEventListener("keydown", (e) => {
        const at = cells.indexOf(document.activeElement);
        if (at < 0) return;
        const cols = Math.max(1, Math.floor(list.clientWidth / CELL));
        let next = at;
        if (e.key === "ArrowRight") next = at + 1;
        else if (e.key === "ArrowLeft") next = at - 1;
        else if (e.key === "ArrowDown") next = at + cols;
        else if (e.key === "ArrowUp") next = at - cols;
        else if (e.key === "Escape") {
            e.preventDefault();
            e.stopPropagation();
            if (opts.onClose) opts.onClose();
            return;
        } else return;
        e.preventDefault();
        e.stopPropagation();
        if (next < 0) {
            search.focus();
            return;
        }
        if (next < cells.length) cells[next].focus();
    });

    refresh();

    return {
        focus: () => search.focus(),
        refresh: refresh,
        destroy: () => {
            host.innerHTML = "";
            host.classList.remove("ags-emoji");
            host.style.cssText = "";
            cells = [];
            sections = new Map();
            tabButtons.clear();
        },
    };
}