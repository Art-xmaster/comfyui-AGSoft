"""
==============================================================================
AGSoft_Text_Editor.py
==============================================================================
Нода: 📝 AGSoft Text Editor
Описание / Description:
Серверная часть редактора промптов: хранит упакованное значение виджета,
считает CLIP-токены, принимает и отдаёт файлы, разбирает метаданные PNG
и возвращает текст, негатив, список строк и абсолютный путь документа.
---
Backend of the prompt editor: keeps the packed widget value, counts CLIP
tokens, receives and serves files, parses PNG metadata and returns the text,
the negative buffer, a list of lines and the absolute document path.

Возможности / Features:
⚡ Один виджет хранит текст, негатив, буфер сравнения и путь документа
  A single widget stores the text, negative, comparison buffer and path
⚡ Буфер сравнения отрезается при разборе и никогда не попадает в выходы
  The comparison buffer is stripped while unpacking and never reaches outputs
⚡ Открытый файл потоком пишется в temp/AGSoft_Text и исчезает при перезапуске
  An opened file is streamed into temp/AGSoft_Text and vanishes on restart
⚡ Одинаковое содержимое переиспользует копию, разное получает суффикс _1, _2
  Identical content reuses a copy, different content gets an _1, _2 suffix
⚡ Счётчик CLIP-токенов и чанков по 75 через встроенный токенайзер ComfyUI
  CLIP token and 75-token chunk counter based on the bundled ComfyUI tokenizer
⚡ Без токенайзера отдаётся оценка, помеченная как приблизительная
  Without a tokenizer an estimate is returned and marked as approximate
⚡ Папки назначения сгруппированы: ComfyUI, личные папки, добавленные вручную
  Destination folders are grouped: ComfyUI, personal folders, manually added
⚡ Рабочий стол, Документы, Загрузки и Изображения находятся с учётом OneDrive
  Desktop, Documents, Downloads and Pictures are found even under OneDrive
⚡ Поиск файла по имени и размеру ограничен глубиной и числом записей
  File lookup by name and size is bounded by depth and entry count
⚡ Метаданные PNG читаются из tEXt, zTXt и iTXt, включая граф ComfyUI
  PNG metadata is read from tEXt, zTXt and iTXt, including a ComfyUI graph
⚡ Кириллица и другие алфавиты сохраняются в именах файлов
  Cyrillic and other alphabets survive in file names
⚡ Пока документ не сохранён, путь возвращается как New document
  While the document is unsaved the path is reported as New document

Выходы / Outputs:
text      основной текст, UTF-8 / main text, UTF-8
negative  буфер негатива / negative buffer
lines     непустые строки списком для итерации / non-empty lines for iteration
filepath  абсолютный путь или New document / absolute path or New document

Роуты / Routes:
GET  /agsoft/text_editor/roots        список папок / folder list
POST /agsoft/text_editor/add_root     добавить папку записи / add save folder
POST /agsoft/text_editor/remove_root  убрать папку записи / remove save folder
POST /agsoft/text_editor/upload       приём документа / document intake
POST /agsoft/text_editor/resolve      поиск пути по имени / path lookup by name
POST /agsoft/text_editor/read         чтение по пути / read by path
POST /agsoft/text_editor/tokenize     счёт токенов / token count
POST /agsoft/text_editor/save         запись и экспорт / write and export
POST /agsoft/text_editor/png_meta     метаданные PNG / PNG metadata

Хранение / Storage:
Маркеры упаковки: [AGSoft Negative], [AGSoft Result], [AGSoft Filepath]
Packing markers: [AGSoft Negative], [AGSoft Result], [AGSoft Filepath]
Список добавленных папок лежит в папке user ComfyUI
The list of added folders lives in the ComfyUI user folder
Кэш токенайзера живёт в памяти модуля / The tokenizer cache lives in module memory

Лимиты / Limits:
Запись 4 000 000 символов, чтение 8 МБ, приём 16 МБ, PNG 32 МБ
Write 4,000,000 chars, read 8 MB, intake 16 MB, PNG 32 MB
Обход папок: глубина 6, до 20 000 записей, до 5 совпадений
Folder walk: depth 6, up to 20,000 entries, up to 5 matches
Добавленных папок до 12, суффиксов имени до 999
Up to 12 added folders, up to 999 name suffixes

Зависимости / Dependencies:
folder_paths, aiohttp.web, server.PromptServer
comfy.sd1_tokenizer используется при наличии / used when available

Автор / Author: AGSoft
Дата / Date: 14.11.2026
==============================================================================
"""

import hashlib
import json
import logging
import os
import re
import zlib
from typing import Any, Dict, Iterator, List, Tuple

import folder_paths
from aiohttp import web
from server import PromptServer

# print("[AGSoft Text Editor] v11.14 loaded (temp uploads, result buffer, new document)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SAVE_CHARS = 4000000
MAX_READ_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_CUSTOM_ROOTS = 12
MAX_SEARCH_FILES = 20000
MAX_SEARCH_DEPTH = 6
MAX_RESOLVE_HITS = 5
MAX_NAME_TRIES = 999
CHUNK_BYTES = 1024 * 1024
NEW_DOCUMENT_LABEL = "New document"
TEXT_UPLOAD_SUBDIR = "AGSoft_Text"
ALLOWED_EXT = (".txt", ".md", ".json", ".csv", ".log")
SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv",
    "AppData", "Application Data", "Local Settings",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
NEG_MARK = "\n[AGSoft Negative]\n"
RESULT_MARK = "\n[AGSoft Result]\n"
PATH_MARK = "\n[AGSoft Filepath]\n"
ROOTS_CONFIG_NAME = "agsoft_text_editor_roots.json"
COMFY_ROOT_KEYS = ("output", "input", "temp", "user")
PLACE_ROOT_KEYS = ("desktop", "documents", "downloads", "pictures", "home")
ROOT_LABELS = {
    "output": "ComfyUI output",
    "input": "ComfyUI input",
    "temp": "ComfyUI temp",
    "user": "ComfyUI user",
    "desktop": "Desktop / Рабочий стол",
    "documents": "Documents / Документы",
    "downloads": "Downloads / Загрузки",
    "pictures": "Pictures / Изображения",
    "home": "Home / Домашняя папка",
}
TOKENIZER_CACHE: Dict[str, Any] = {}


def _is_loopback(request) -> bool:
    """Write access is limited to the local machine / Запись только с локальной машины."""
    remote = str(request.remote or "")
    return remote in ("127.0.0.1", "::1", "localhost")


def _is_contained(path: str, base: str) -> bool:
    """Different drives must not raise / Разные диски не должны ронять проверку."""
    try:
        return os.path.commonpath([base, path]) == base
    except ValueError:
        return False


def _user_config_path() -> str:
    """Settings live in the ComfyUI user folder / Настройки живут в папке user ComfyUI."""
    if not hasattr(folder_paths, "get_user_directory"):
        return ""
    base = folder_paths.get_user_directory()
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, ROOTS_CONFIG_NAME)


def _root_key(path: str) -> str:
    """Stable identifier for a saved folder / Стабильный ключ сохранённой папки."""
    return "x" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:10]


def _read_custom_roots() -> List[str]:
    """Folders added as save destinations / Папки, добавленные как назначения записи."""
    config = _user_config_path()
    if not config or not os.path.isfile(config):
        return []
    try:
        with open(config, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.warning("Custom roots unreadable: %s", exc)
        return []
    items = data.get("roots") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        real = os.path.realpath(item)
        if os.path.isdir(real) and real not in out:
            out.append(real)
    return out[:MAX_CUSTOM_ROOTS]


def _write_custom_roots(roots: List[str]) -> bool:
    """Persist the destination list / Сохранение списка назначений."""
    config = _user_config_path()
    if not config:
        return False
    try:
        with open(config, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"roots": roots[:MAX_CUSTOM_ROOTS]}, handle,
                      ensure_ascii=False, indent=4)
    except Exception as exc:
        logger.warning("Custom roots not stored: %s", exc)
        return False
    return True


def _user_places() -> Dict[str, str]:
    """Windows may move these folders into OneDrive.
    Windows может перенести эти папки в OneDrive."""
    home = os.path.realpath(os.path.expanduser("~"))
    if not os.path.isdir(home):
        return {}
    try:
        entries = sorted(os.listdir(home))
    except OSError as exc:
        logger.debug("Home listing unavailable: %s", exc)
        entries = []
    out: Dict[str, str] = {}
    wanted = (
        ("desktop", "Desktop"),
        ("documents", "Documents"),
        ("downloads", "Downloads"),
        ("pictures", "Pictures"),
    )
    for key, folder in wanted:
        direct = os.path.realpath(os.path.join(home, folder))
        if os.path.isdir(direct):
            out[key] = direct
            continue
        for name in entries:
            if not name.startswith("OneDrive"):
                continue
            alt = os.path.realpath(os.path.join(home, name, folder))
            if os.path.isdir(alt):
                out[key] = alt
                break
    out["home"] = home
    return out


def _allowed_roots() -> Dict[str, str]:
    """Every destination the editor may touch / Все папки, доступные редактору."""
    roots = {
        "output": folder_paths.get_output_directory(),
        "input": folder_paths.get_input_directory(),
        "temp": folder_paths.get_temp_directory(),
    }
    if hasattr(folder_paths, "get_user_directory"):
        try:
            roots["user"] = folder_paths.get_user_directory()
        except Exception as exc:
            logger.debug("User directory unavailable: %s", exc)
    resolved = {key: os.path.realpath(value) for key, value in roots.items()}
    for key, path in _user_places().items():
        resolved[key] = path
    for path in _read_custom_roots():
        resolved[_root_key(path)] = path
    return resolved


def _unique_roots() -> List[str]:
    """Nested roots must not be scanned twice / Вложенные корни не сканируем дважды."""
    out: List[str] = []
    for path in _allowed_roots().values():
        if path not in out:
            out.append(path)
    return out


def _sanitize_chunk(value: str) -> str:
    """Letters of any alphabet stay readable / Буквы любого алфавита остаются читаемыми."""
    return re.sub(r"[^\w.\- ]+", "_", str(value), flags=re.UNICODE).strip(" .")


def _upload_dir() -> str:
    """Opened copies belong to temp, so a restart clears them.
    Копии открытых файлов живут в temp, поэтому перезапуск их убирает."""
    base = os.path.realpath(folder_paths.get_temp_directory())
    folder = os.path.realpath(os.path.join(base, TEXT_UPLOAD_SUBDIR))
    if not _is_contained(folder, base):
        raise ValueError("Upload folder outside temp / Папка загрузки вне temp")
    os.makedirs(folder, exist_ok=True)
    return folder


def _unique_target(folder: str, name: str) -> str:
    """A different document must not overwrite an older one.
    Другой документ не должен затирать предыдущий."""
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(folder, name)
    index = 1
    while os.path.exists(candidate) and index <= MAX_NAME_TRIES:
        candidate = os.path.join(folder, stem + "_" + str(index) + ext)
        index += 1
    return candidate


def _file_digest(path: str) -> str:
    """Streaming hash keeps memory flat / Потоковый хэш не раздувает память."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_target_path(root_key: str, subfolder: str, name: str) -> Tuple[str, str]:
    """Resolve a contained destination / Разрешение пути внутри выбранного корня."""
    roots = _allowed_roots()
    base = roots.get(root_key, roots["output"])
    parts = []
    for chunk in str(subfolder or "").replace("\\", "/").split("/"):
        clean = _sanitize_chunk(chunk)
        if clean:
            parts.append(clean)
    folder = os.path.realpath(os.path.join(base, *parts))
    if not _is_contained(folder, base):
        raise ValueError("Subfolder outside root / Подпапка вне корня")
    clean_name = _sanitize_chunk(os.path.basename(str(name).strip()))
    if not clean_name:
        clean_name = "prompt.txt"
    if not clean_name.lower().endswith(ALLOWED_EXT):
        clean_name += ".txt"
    path = os.path.realpath(os.path.join(folder, clean_name))
    if not _is_contained(path, base):
        raise ValueError("Path outside root / Путь вне корня")
    os.makedirs(folder, exist_ok=True)
    return path, base


def _iter_root_files(base: str) -> Iterator[Tuple[str, str]]:
    """A bounded walk keeps big folders responsive / Ограниченный обход бережёт скорость."""
    root_depth = base.rstrip(os.sep).count(os.sep)
    seen = 0
    for folder, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs
                   if not d.startswith(".") and d not in SKIP_DIRS]
        if folder.count(os.sep) - root_depth >= MAX_SEARCH_DEPTH:
            dirs[:] = []
        for name in files:
            seen += 1
            if seen > MAX_SEARCH_FILES:
                return
            yield folder, name


def _resolve_named(name: str, size: int = -1) -> List[str]:
    """Locate a file inside known folders / Поиск файла в известных папках."""
    target = os.path.basename(str(name or "")).lower()
    if not target:
        return []
    found: List[str] = []
    for base in _unique_roots():
        if not os.path.isdir(base):
            continue
        try:
            for folder, file_name in _iter_root_files(base):
                if file_name.lower() != target:
                    continue
                path = os.path.realpath(os.path.join(folder, file_name))
                if not path.lower().endswith(ALLOWED_EXT):
                    continue
                if size >= 0:
                    try:
                        if os.path.getsize(path) != size:
                            continue
                    except OSError:
                        continue
                if path not in found:
                    found.append(path)
                if len(found) >= MAX_RESOLVE_HITS:
                    return found
        except OSError as exc:
            logger.debug("Root scan skipped: %s", exc)
    return found


def _unpack_text(value: str) -> Tuple[str, str, str]:
    """The comparison buffer stays out of the outputs.
    Буфер сравнения не попадает в выходы ноды."""
    text = value if isinstance(value, str) else ""
    negative = ""
    filepath = ""
    path_idx = text.find(PATH_MARK)
    if path_idx >= 0:
        filepath = text[path_idx + len(PATH_MARK):].strip()
        text = text[:path_idx]
    result_idx = text.find(RESULT_MARK)
    if result_idx >= 0:
        text = text[:result_idx]
    neg_idx = text.find(NEG_MARK)
    if neg_idx >= 0:
        negative = text[neg_idx + len(NEG_MARK):]
        text = text[:neg_idx]
    return text, negative, filepath


def _line_list(text: str) -> List[str]:
    """Non-empty lines for downstream iteration / Непустые строки для итерации."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _clip_tokenizer():
    """Reuse the bundled CLIP tokenizer / Переиспользуем встроенный CLIP-токенайзер."""
    if "clip" in TOKENIZER_CACHE:
        return TOKENIZER_CACHE["clip"]
    tokenizer = None
    try:
        from comfy.sd1_tokenizer import SD1Tokenizer
        tokenizer = SD1Tokenizer()
    except Exception as exc:
        logger.warning("CLIP tokenizer unavailable, estimate mode: %s", exc)
        tokenizer = None
    TOKENIZER_CACHE["clip"] = tokenizer
    return tokenizer


def _count_tokens(text: str) -> Dict[str, Any]:
    """Exact CLIP counts when possible / Точные CLIP-счётчики, когда это возможно."""
    if not text.strip():
        return {"tokens": 0, "chunks": 0, "exact": True}
    tokenizer = _clip_tokenizer()
    if tokenizer is not None:
        try:
            parsed = tokenizer.tokenize_with_weights(text)
            groups = parsed.get("l") if isinstance(parsed, dict) else None
            if groups:
                total = sum(len(chunk) for chunk in groups)
                return {"tokens": int(total), "chunks": len(groups), "exact": True}
        except Exception as exc:
            logger.warning("Token counting failed, estimate mode: %s", exc)
    approx = max(1, int(len(text) / 4))
    return {"tokens": approx, "chunks": (approx + 74) // 75, "exact": False}


def _png_text_chunks(data: bytes) -> Dict[str, str]:
    """Read tEXt/zTXt/iTXt metadata / Чтение метаданных tEXt/zTXt/iTXt."""
    out: Dict[str, str] = {}
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        return out
    pos = 8
    total = len(data)
    while pos + 8 <= total:
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > total:
            break
        body = data[start:end]
        if kind == b"IEND":
            break
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            try:
                key, _, rest = body.partition(b"\x00")
                name = key.decode("latin-1")
                if kind == b"tEXt":
                    out[name] = rest.decode("utf-8", "replace")
                elif kind == b"zTXt":
                    out[name] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                else:
                    flag = rest[0] if rest else 0
                    tail = rest[2:]
                    _, _, tail = tail.partition(b"\x00")
                    _, _, tail = tail.partition(b"\x00")
                    if flag:
                        tail = zlib.decompress(tail)
                    out[name] = tail.decode("utf-8", "replace")
            except Exception as exc:
                logger.debug("PNG chunk skipped: %s", exc)
        pos = end + 4
    return out


def _split_parameters(params: str) -> Tuple[str, str]:
    """Split A1111 parameters block / Разбор блока parameters из A1111."""
    text = params.replace("\r\n", "\n").strip()
    marker = "\nNegative prompt:"
    idx = text.find(marker)
    positive = text if idx < 0 else text[:idx]
    rest = "" if idx < 0 else text[idx + len(marker):]
    tail = re.search(r"\n[A-Z][A-Za-z ]{0,30}:\s", rest)
    negative = rest[:tail.start()] if tail else rest
    if idx < 0:
        steps = re.search(r"\nSteps:\s", positive)
        if steps:
            positive = positive[:steps.start()]
    return positive.strip(), negative.strip()


def _node_text(node: Any) -> str:
    """Text field of an API node / Текстовое поле ноды в API-формате."""
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if isinstance(inputs, dict) and isinstance(inputs.get("text"), str):
        return inputs["text"]
    return ""


def _comfy_prompt_pair(payload: Any) -> Tuple[str, str]:
    """Recover positive/negative from a graph / Восстановление positive/negative из графа."""
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        texts = []
        for item in payload["nodes"]:
            if not isinstance(item, dict):
                continue
            if "CLIPTextEncode" not in str(item.get("type", "")):
                continue
            for value in item.get("widgets_values") or []:
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                    break
        if texts:
            return texts[0], texts[1] if len(texts) > 1 else ""
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    nodes = payload

    def resolve(ref: Any, depth: int = 0) -> str:
        if depth > 8 or not isinstance(ref, list) or not ref:
            return ""
        node = nodes.get(str(ref[0]))
        if not isinstance(node, dict):
            return ""
        if "CLIPTextEncode" in str(node.get("class_type", "")):
            return _node_text(node)
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            for key in ("text", "string", "conditioning", "value", "input"):
                if key in inputs:
                    found = resolve(inputs[key], depth + 1)
                    if found:
                        return found
        return ""

    for node in nodes.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict) and "positive" in inputs and "negative" in inputs:
            pos = resolve(inputs["positive"])
            neg = resolve(inputs["negative"])
            if pos or neg:
                return pos, neg
    texts = []
    for node in nodes.values():
        if isinstance(node, dict) and "CLIPTextEncode" in str(node.get("class_type", "")):
            value = _node_text(node)
            if value:
                texts.append(value)
    if texts:
        return texts[0], texts[1] if len(texts) > 1 else ""
    return "", ""


@PromptServer.instance.routes.post("/agsoft/text_editor/upload")
async def agsoft_text_editor_upload(request):
    if not _is_loopback(request):
        return web.json_response(
            {"error": "Local access only / Только локальный доступ"}, status=403
        )
    temp_path = ""
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or not field.filename:
            return web.json_response({"error": "No file / Файл не передан"}, status=400)
        name = _sanitize_chunk(os.path.basename(str(field.filename)))
        if not name:
            name = "document.txt"
        if not name.lower().endswith(ALLOWED_EXT):
            name += ".txt"
        folder = _upload_dir()
        temp_path = os.path.join(folder, ".part_" + os.urandom(6).hex())
        size = 0
        with open(temp_path, "wb") as handle:
            while True:
                chunk = await field.read_chunk(CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("File too large / Файл слишком большой")
                handle.write(chunk)
        target = os.path.join(folder, name)
        if os.path.isfile(target) and _file_digest(target) != _file_digest(temp_path):
            target = _unique_target(folder, name)
        os.replace(temp_path, target)
        temp_path = ""
    except Exception as exc:
        logger.warning("Document upload failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)
    finally:
        if temp_path and os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError as exc:
                logger.debug("Temporary file left behind: %s", exc)
    return web.json_response({
        "filepath": target,
        "filename": os.path.basename(target),
        "bytes": size,
    })


@PromptServer.instance.routes.get("/agsoft/text_editor/roots")
async def agsoft_text_editor_roots(request):
    items = []
    for key, path in _allowed_roots().items():
        if key in COMFY_ROOT_KEYS:
            group = "comfy"
        elif key in PLACE_ROOT_KEYS:
            group = "places"
        else:
            group = "custom"
        items.append({
            "key": key,
            "label": ROOT_LABELS.get(key, path),
            "path": path,
            "group": group,
            "custom": group == "custom",
        })
    return web.json_response({"roots": items})


@PromptServer.instance.routes.post("/agsoft/text_editor/add_root")
async def agsoft_text_editor_add_root(request):
    if not _is_loopback(request):
        return web.json_response(
            {"error": "Local access only / Только локальный доступ"}, status=403
        )
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    raw = str(data.get("path", "")).strip().strip('"')
    if not raw or not os.path.isabs(raw):
        return web.json_response(
            {"error": "Absolute path required / Нужен абсолютный путь"}, status=400
        )
    real = os.path.realpath(raw)
    if not os.path.isdir(real):
        return web.json_response(
            {"error": "Folder not found / Папка не найдена"}, status=400
        )
    roots = _read_custom_roots()
    if real not in roots:
        if len(roots) >= MAX_CUSTOM_ROOTS:
            return web.json_response(
                {"error": "Too many folders / Слишком много папок"}, status=400
            )
        roots.append(real)
    if not _write_custom_roots(roots):
        return web.json_response(
            {"error": "Config not writable / Конфиг недоступен для записи"}, status=400
        )
    return web.json_response({"key": _root_key(real), "path": real})


@PromptServer.instance.routes.post("/agsoft/text_editor/remove_root")
async def agsoft_text_editor_remove_root(request):
    if not _is_loopback(request):
        return web.json_response(
            {"error": "Local access only / Только локальный доступ"}, status=403
        )
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    key = str(data.get("key", ""))
    roots = [path for path in _read_custom_roots() if _root_key(path) != key]
    if not _write_custom_roots(roots):
        return web.json_response(
            {"error": "Config not writable / Конфиг недоступен для записи"}, status=400
        )
    return web.json_response({"removed": key})


@PromptServer.instance.routes.post("/agsoft/text_editor/resolve")
async def agsoft_text_editor_resolve(request):
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    name = str(data.get("name", ""))
    try:
        size = int(data.get("size", -1))
    except (TypeError, ValueError):
        size = -1
    try:
        matches = _resolve_named(name, size)
        if not matches and size >= 0:
            matches = _resolve_named(name, -1)
    except Exception as exc:
        logger.warning("Path resolving failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({
        "path": matches[0] if len(matches) == 1 else "",
        "matches": matches,
    })


@PromptServer.instance.routes.post("/agsoft/text_editor/read")
async def agsoft_text_editor_read(request):
    if not _is_loopback(request):
        return web.json_response(
            {"error": "Local access only / Только локальный доступ"}, status=403
        )
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    raw = str(data.get("path", "")).strip().strip('"')
    if not raw or not os.path.isabs(raw):
        return web.json_response(
            {"error": "Absolute path required / Нужен абсолютный путь"}, status=400
        )
    path = os.path.realpath(raw)
    if not os.path.isfile(path):
        return web.json_response(
            {"error": "File not found / Файл не найден"}, status=400
        )
    if not path.lower().endswith(ALLOWED_EXT):
        return web.json_response(
            {"error": "Unsupported extension / Неподдерживаемое расширение"}, status=400
        )
    if not any(_is_contained(path, base) for base in _unique_roots()):
        return web.json_response(
            {"error": "Folder is not a destination / Папка не входит в назначения"},
            status=400,
        )
    try:
        size = os.path.getsize(path)
        if size > MAX_READ_BYTES:
            return web.json_response(
                {"error": "File too large / Файл слишком большой"}, status=413
            )
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except Exception as exc:
        logger.warning("Text read failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"text": text, "filepath": path, "bytes": size})


@PromptServer.instance.routes.post("/agsoft/text_editor/tokenize")
async def agsoft_text_editor_tokenize(request):
    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)
    text = data.get("text", "")
    if not isinstance(text, str):
        text = ""
    if len(text) > MAX_SAVE_CHARS:
        text = text[:MAX_SAVE_CHARS]
    return web.json_response(_count_tokens(text))


@PromptServer.instance.routes.post("/agsoft/text_editor/save")
async def agsoft_text_editor_save(request):
    if not _is_loopback(request):
        return web.json_response(
            {"error": "Local access only / Только локальный доступ"}, status=403
        )
    try:
        data = await request.json()
        text = data.get("text", "")
        if not isinstance(text, str):
            return web.json_response({"error": "Bad text / Некорректный текст"}, status=400)
        if len(text) > MAX_SAVE_CHARS:
            return web.json_response(
                {"error": "Text too large / Текст слишком большой"}, status=413
            )
        path, base = _safe_target_path(
            str(data.get("root", "output")),
            str(data.get("subfolder", "")),
            data.get("filename", ""),
        )
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except Exception as exc:
        logger.warning("Text save failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({
        "filepath": path,
        "relative": os.path.relpath(path, base).replace("\\", "/"),
        "root": os.path.basename(base),
        "bytes": os.path.getsize(path),
    })


@PromptServer.instance.routes.post("/agsoft/text_editor/png_meta")
async def agsoft_text_editor_png_meta(request):
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"error": "No file / Файл не передан"}, status=400)
        buf = bytearray()
        while True:
            chunk = await field.read_chunk(CHUNK_BYTES)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_PNG_BYTES:
                return web.json_response(
                    {"error": "File too large / Файл слишком большой"}, status=413
                )
        meta = _png_text_chunks(bytes(buf))
    except Exception as exc:
        logger.warning("PNG metadata read failed: %s", exc)
        return web.json_response({"error": str(exc)}, status=400)
    positive, negative, source = "", "", ""
    params = meta.get("parameters", "")
    if params:
        positive, negative = _split_parameters(params)
        source = "parameters"
    if not positive:
        for key in ("prompt", "workflow"):
            raw = meta.get(key, "")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            pos, neg = _comfy_prompt_pair(payload)
            if pos or neg:
                positive, negative, source = pos, neg, key
                break
    if not positive:
        for key in ("Prompt", "positive", "Description"):
            if meta.get(key):
                positive, source = meta[key], key
                break
    return web.json_response({
        "positive": positive,
        "negative": negative,
        "source": source,
        "keys": sorted(meta.keys()),
    })


class AGSoftTextEditor:
    """
    Prompt text editor with a single packed storage widget.
    ---
    Редактор промптов с одним упакованным виджетом хранения.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Packed editor value: text, negative, comparison and path."
                        "\n---\n"
                        "Упакованное значение: текст, негатив, сравнение и путь."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "negative", "lines", "filepath")
    OUTPUT_TOOLTIPS = (
        "Main text (UTF-8).\n---\nОсновной текст (UTF-8).",
        "Negative buffer (UTF-8).\n---\nБуфер negative (UTF-8).",
        "Non-empty lines as an iterable list.\n---\nНепустые строки итерируемым списком.",
        (
            "Absolute file path, or New document while the text is unsaved."
            "\n---\n"
            "Абсолютный путь файла или New document, пока текст не сохранён."
        ),
    )
    OUTPUT_IS_LIST = (False, False, True, False)
    FUNCTION = "process"
    CATEGORY = "AGSoft/Text"
    DESCRIPTION = (
        "Prompt editor: opened copies live in the ComfyUI temp folder and vanish on "
        "the next start, while a saved file keeps its absolute path. New document "
        "clears the text and the path. Split view compares Text with Negative or the "
        "original with a translation result. CLIP tokens, regex search and replace, "
        "clipboard tools, word weights, EN/RU/ZH translate, emoji picker, Zen "
        "fullscreen, autosave, recent files, workflow text import, Markdown and JSON "
        "export, saving to Desktop, Documents, Downloads and ComfyUI folders."
        "\n---\n"
        "Редактор промптов: копии открытых файлов лежат в temp ComfyUI и исчезают при "
        "следующем запуске, а сохранённый файл держит абсолютный путь. Новый документ "
        "очищает текст и путь. Split сравнивает Text и Negative либо оригинал и "
        "результат перевода. CLIP-токены, поиск и замена с regex, буфер обмена, веса "
        "слов, перевод EN/RU/ZH, пикер эмодзи, Zen на весь экран, автосохранение, "
        "последние файлы, импорт текстов из workflow, экспорт Markdown и JSON, запись "
        "на Рабочий стол, в Документы, Загрузки и папки ComfyUI."
    )

    def process(self, text: str) -> Tuple[str, str, List[str], str]:
        # The client packs buffers; the node only unpacks and forwards them.
        # Клиент упаковывает буферы; нода только распаковывает и передаёт их.
        positive, negative, filepath = _unpack_text(text)
        return (positive, negative, _line_list(positive), filepath or NEW_DOCUMENT_LABEL)

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> str:
        # Content hash keeps the downstream cache in sync with edits.
        # Хэш содержимого синхронизирует кэш нижестоящих нод с правками.
        value = kwargs.get("text", "")
        if not isinstance(value, str):
            value = ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


NODE_CLASS_MAPPINGS = {"AGSoftTextEditor": AGSoftTextEditor}
NODE_DISPLAY_NAME_MAPPINGS = {"AGSoftTextEditor": "📝AGSoft Text Editor"}
WEB_DIRECTORY = "./web"