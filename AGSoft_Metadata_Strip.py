# ==============================================================================
# AGSoft_Metadata_Strip.py
# ==============================================================================
# Ноды: 🖼️🧹AGSoft Image Strip Metadata  +  🎬🧹AGSoft Video Strip Metadata
# Описание:
# Вырезает метаданные (prompt/workflow/EXIF/XMP/теги) из изображений и видео
# БЕЗ ПЕРЕКОДИРОВАНИЯ: чистая байтовая хирургия контейнеров — пиксели и кадры
# копируются 1:1. ОРИГИНАЛ НЕ ИЗМЕНЯЕТСЯ НИКОГДА: чистая копия пишется в
# output (опционально — в подпапку, создаётся автоматически).
# Strips metadata (prompt/workflow/EXIF/XMP/tags) from images & videos WITHOUT
# re-encoding: pure byte surgery, pixels/frames copied 1:1.
# The ORIGINAL is NEVER modified: a clean copy goes to the OUTPUT folder
# (optionally into an auto-created subfolder).
#
# Автор / Author: AGSoft
# Дата / Date: 29.08.2026
# ==============================================================================
import os
import folder_paths

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
# Markers scanned in raw bytes for the report (prompt/workflow = ComfyUI chunks).
MARKERS = [b"prompt", b"workflow", b"tEXt", b"zTXt", b"iTXt", b"exif", b"EXIF",
           b"XMP", b"Comment", b"Title", b"encoder", b"parameters", b"udta", b"meta"]


# ==============================================================================
# SCAN (head+tail for big files)
# ==============================================================================
def _scan_markers(data, limit=32 * 1024 * 1024):
    try:
        n = len(data)
        blob = data if n <= limit else data[:limit // 2] + data[n - limit // 2:]
        return [m.decode() for m in MARKERS if m in blob]
    except Exception:
        return []


# ==============================================================================
# IMAGE STRIPPERS (pure Python, lossless)
# ==============================================================================
def _strip_png(data):
    """PNG: drop tEXt/zTXt/iTXt/eXIf/tIME/dSIG chunks, keep the rest 1:1."""
    SIG = b"\x89PNG\r\n\x1a\n"
    if data[:8] != SIG:
        raise ValueError("not png")
    DROP = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME", b"dSIG"}
    out = [SIG]
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8]
        end = pos + 8 + length + 4
        if end > n:
            out.append(data[pos:])
            break
        if ctype not in DROP:
            out.append(data[pos:end])
        pos = end
    return b"".join(out)


def _strip_jpeg(data):
    """JPEG: drop APP1(EXIF/XMP)/APP12/APP13(IPTC)/COM, keep the rest 1:1.
    APP0(JFIF)/APP2(ICC)/APP14(Adobe) are kept so colors stay correct."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not jpeg")
    DROP = {0xE1, 0xEC, 0xED, 0xFE}
    out = [data[:2]]
    pos = 2
    n = len(data)
    while pos + 1 < n:
        if data[pos] != 0xFF:
            out.append(data[pos:])
            break
        m = data[pos + 1]
        if m == 0xFF:
            pos += 1
            continue
        if m == 0xD9 or (0xD0 <= m <= 0xD7) or m == 0x01:
            out.append(data[pos:pos + 2])
            pos += 2
            continue
        if pos + 4 > n:
            out.append(data[pos:])
            break
        seglen = int.from_bytes(data[pos + 2:pos + 4], "big")
        end = pos + 2 + seglen
        if end > n:
            out.append(data[pos:])
            break
        if m not in DROP:
            out.append(data[pos:end])
        if m == 0xDA:  # SOS: scan data follows, copy everything as is
            out.append(data[end:])
            break
        pos = end
    return b"".join(out)


def _strip_webp(data):
    """WebP (RIFF): drop EXIF/XMP chunks, recompute the RIFF size."""
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not webp")
    body = [b"WEBP"]
    pos = 12
    n = len(data)
    while pos + 8 <= n:
        cc = data[pos:pos + 4]
        sz = int.from_bytes(data[pos + 4:pos + 8], "little")
        end = pos + 8 + sz
        if end > n:
            body.append(data[pos:])
            break
        if cc not in (b"EXIF", b"XMP "):
            body.append(data[pos:end])
            if (end & 1) and end + 1 <= n:
                body.append(data[end:end + 1])  # RIFF padding
        pos = end + (end & 1)
    payload = b"".join(body)
    return b"RIFF" + len(payload).to_bytes(4, "little") + payload


def _strip_gif(data):
    """GIF: drop comment (0xFE) and XMP app-extensions, keep loop & frames."""
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("not gif")
    out = [data[:13]]
    pos = 13
    n = len(data)
    packed = data[10]
    if packed & 0x80:  # global color table
        gct = 3 * (2 << (packed & 7))
        out.append(data[13:13 + gct])
        pos = 13 + gct
    while pos < n:
        b0 = data[pos]
        if b0 == 0x3B:  # trailer
            out.append(data[pos:pos + 1])
            break
        if b0 == 0x21:  # extension
            label = data[pos + 1]
            end = pos + 2
            first = None
            while end < n:
                sz = data[end]
                if first is None:
                    first = data[end + 1:end + 12]
                end += sz + 1
                if sz == 0:
                    break
            keep = True
            if label == 0xFE:
                keep = False  # comment
            if label == 0xFF and first == b"XMP DataXMP":
                keep = False  # XMP
            if keep:
                out.append(data[pos:end])
            pos = end
        elif b0 == 0x2C:  # image descriptor
            out.append(data[pos:pos + 10])
            pos += 10
            lp = data[pos - 1]
            if lp & 0x80:  # local color table
                lct = 3 * (2 << (lp & 7))
                out.append(data[pos:pos + lct])
                pos += lct
            out.append(data[pos:pos + 1])  # LZW min code size
            pos += 1
            while pos < n:
                sz = data[pos]
                out.append(data[pos:pos + sz + 1])
                pos += sz + 1
                if sz == 0:
                    break
        else:
            pos += 1
    return b"".join(out)


def _strip_image_bytes(data):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return _strip_png(data)
    if data[:2] == b"\xff\xd8":
        return _strip_jpeg(data)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _strip_webp(data)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _strip_gif(data)
    raise ValueError("unsupported image format (supported: PNG/JPG/WebP/GIF)")


# ==============================================================================
# VIDEO STRIPPERS (pure Python, stream copy)
# ==============================================================================
_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"moof", b"mvex"}
_MP4_DROP = {b"meta", b"udta", b"tags"}


def _mp4_walk(buf):
    """MP4: recursively remove meta/udta/tags boxes, recompute sizes."""
    out = []
    pos = 0
    n = len(buf)
    while pos + 8 <= n:
        size = int.from_bytes(buf[pos:pos + 4], "big")
        btype = buf[pos + 4:pos + 8]
        hdr = 8
        if size == 1:
            size = int.from_bytes(buf[pos + 8:pos + 16], "big")
            hdr = 16
        elif size == 0:
            size = n - pos
        if size < hdr or pos + size > n:
            out.append(buf[pos:])
            break
        chunk = buf[pos:pos + size]
        if btype in _MP4_DROP:
            pass  # metadata removed
        elif btype in _MP4_CONTAINERS:
            inner = _mp4_walk(chunk[hdr:])
            if hdr == 16:
                nb = b"\x00\x00\x00\x01" + btype + (16 + len(inner)).to_bytes(8, "big") + inner
            else:
                nb = (8 + len(inner)).to_bytes(4, "big") + btype + inner
            out.append(nb)
        else:
            out.append(chunk)
        pos += size
    return b"".join(out)


def _strip_mp4(data):
    if data[4:8] not in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        raise ValueError("not mp4")
    if data[8:12] in (b"heic", b"heix", b"mif1", b"msf1"):
        raise ValueError("HEIC: 'meta' box is structural, cannot strip")
    return _mp4_walk(data)


_MKV_SEGMENT = 0x18538067
_MKV_INFO = 0x1549A966
_MKV_DROP_TOP = {0x1254C367, 0x1941A469}      # Tags, Attachments
_MKV_DROP_INFO = {0x7BA9, 0x4D80, 0x5741}     # Title, MuxingApp, WritingApp


def _vint_len(b0):
    for i in range(8):
        if b0 & (0x80 >> i):
            return i + 1
    return 9


def _ebml_read(buf, pos):
    il = _vint_len(buf[pos])
    eid = int.from_bytes(buf[pos:pos + il], "big")
    sl = _vint_len(buf[pos + il])
    raw = int.from_bytes(buf[pos + il:pos + il + sl], "big")
    maxv = (1 << (7 * sl)) - 1
    val = raw & maxv
    return eid, il, val, sl, (val == maxv)


def _encode_vint_same(value, sl):
    b = bytearray()
    b.append((0x100 >> sl) | (value >> (7 * (sl - 1))))
    for i in range(sl - 2, -1, -1):
        b.append((value >> (7 * i)) & 0x7F)
    return bytes(b)


def _mkv_filter_children(buf, drop):
    out = []
    pos = 0
    n = len(buf)
    while pos < n:
        eid, il, size, sl, unk = _ebml_read(buf, pos)
        hdr = il + sl
        end = n if unk else min(n, pos + hdr + size)
        if eid not in drop:
            out.append(buf[pos:end])
        pos = end
    return b"".join(out)


def _mkv_segment_body(buf):
    out = []
    pos = 0
    n = len(buf)
    while pos < n:
        eid, il, size, sl, unk = _ebml_read(buf, pos)
        hdr = il + sl
        end = n if unk else min(n, pos + hdr + size)
        if eid in _MKV_DROP_TOP:
            pos = end
            continue
        if eid == _MKV_INFO and not unk:
            inner = _mkv_filter_children(buf[pos + hdr:end], _MKV_DROP_INFO)
            out.append(buf[pos:pos + il] + _encode_vint_same(len(inner), sl) + inner)
        else:
            out.append(buf[pos:end])
        pos = end
    return b"".join(out)


def _strip_mkv(data):
    """Matroska (WebM/MKV): remove Tags/Attachments + Info title/app tags."""
    if data[:4] != b"\x1a\x45\xdf\xa3":
        raise ValueError("not matroska")
    out = []
    pos = 0
    n = len(data)
    while pos < n:
        eid, il, size, sl, unk = _ebml_read(data, pos)
        hdr = il + sl
        end = n if unk else min(n, pos + hdr + size)
        if eid == _MKV_SEGMENT and not unk:
            inner = _mkv_segment_body(data[pos + hdr:end])
            out.append(data[pos:pos + il] + _encode_vint_same(len(inner), sl) + inner)
        else:
            out.append(data[pos:end])
        pos = end
    return b"".join(out)


def _strip_video_bytes(data):
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return _strip_mkv(data)
    if data[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        return _strip_mp4(data)
    raise ValueError("unsupported video format (supported: MP4/MOV/WebM/MKV)")


# ==============================================================================
# OUTPUT HELPERS (subfolder + filename_prefix)
# ==============================================================================
def _resolve_out_dir(subfolder):
    """output[/subfolder] — subfolder is auto-created; '..' and absolute
    paths are rejected so we never escape the output root."""
    out_root = os.path.abspath(folder_paths.get_output_directory())
    sub = os.path.normpath((subfolder or "").strip())
    if sub in ("", "."):
        return out_root
    if sub.startswith("..") or os.path.isabs(sub):
        raise ValueError(f"Invalid subfolder '{subfolder}': must be relative, no '..'")
    out_dir = os.path.join(out_root, sub)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_name(base, ext, prefix):
    p = (prefix or "").strip()
    if p:
        return f"{p}_{base}{ext}"
    return f"{base}_clean{ext}"


# ==============================================================================
# COMMON PROCESSING (original is NEVER touched; clean copy -> output[/sub])
# ==============================================================================
def _process_files(src, write_mode, subfolder, filename_prefix,
                   tag, emoji, strip_fn, exts):
    is_dir = os.path.isdir(src)
    if is_dir:
        files = [os.path.join(src, f) for f in sorted(os.listdir(src))
                 if os.path.isfile(os.path.join(src, f))
                 and os.path.splitext(f)[1].lower() in exts]
        if not files:
            raise ValueError(f"No supported files in folder: {src}")
    elif os.path.isfile(src):
        files = [src]
    else:
        raise ValueError(f"File or folder not found: '{src}'")

    out_dir = _resolve_out_dir(subfolder)
    lines = []
    last_dst = src
    for f in files:
        base, ext = os.path.splitext(os.path.basename(f))
        with open(f, "rb") as fh:
            raw = fh.read()
        before = _scan_markers(raw)
        if write_mode == "scan_only":
            lines.append(f"{emoji} {base}{ext}: " +
                         (f"found [{', '.join(before)}]" if before else "clean"))
            last_dst = f
            continue
        try:
            cleaned = strip_fn(raw)
        except ValueError as e:
            raise RuntimeError(f"{base}{ext}: {e}")
        name = _build_name(base, ext, filename_prefix)
        final = os.path.join(out_dir, name)
        if os.path.abspath(final) == os.path.abspath(f):
            final = os.path.join(out_dir, f"{base}_clean2{ext}")
        tb, te = os.path.splitext(final)
        tmp = tb + ".agtmp" + te
        with open(tmp, "wb") as fh:
            fh.write(cleaned)
        os.replace(tmp, final)
        after = _scan_markers(cleaned)
        note = ("metadata stripped" if cleaned != raw
                else "no metadata found (1:1 copy)")
        lines.append(
            f"{emoji} {base}{ext}: was [{', '.join(before) or '—'}] -> "
            f"[{', '.join(after) or 'clean'}] · {note} -> {final}")
        print(f"[AGSoft {tag} Strip] {note}: {final}")
        last_dst = final
    return (last_dst, "\n".join(lines))


# ==============================================================================
# NODE: 🖼️🧹AGSoft Image Strip Metadata
# ==============================================================================
class AGSoft_Image_Strip_Metadata:
    DESCRIPTION = (
        "🖼️🧹Strips metadata (prompt/workflow/EXIF/XMP) from PNG/JPG/WebP/GIF.\n"
        "NO re-encoding: pixels copied 1:1. The ORIGINAL is NEVER modified:\n"
        "a clean copy is written to the OUTPUT folder (optional subfolder).\n"
        "---\n"
        "🖼️🧹Вырезает метаданные (prompt/workflow/EXIF/XMP) из PNG/JPG/WebP/GIF.\n"
        "БЕЗ перекодирования: пиксели 1:1. ОРИГИНАЛ НЕ ТРОГАЕТСЯ:\n"
        "чистая копия пишется в OUTPUT (опционально в подпапку)."
    )
    CATEGORY = "AGSoft/Utils"
    FUNCTION = "strip"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filepath", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Absolute path to an image file or a folder (all supported images inside).\n---\nАбсолютный путь к файлу изображения или папке (все поддерживаемые картинки внутри)."}),
                "write_mode": (["clean_copy", "scan_only"], {
                    "default": "clean_copy",
                    "tooltip": "clean_copy = write a clean copy into OUTPUT (original untouched); scan_only = report only, no writes.\n---\nclean_copy = чистая копия в OUTPUT (оригинал не трогается); scan_only = только отчёт, без записи."}),
                "subfolder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional subfolder inside OUTPUT (created automatically; nested 'a/b' allowed).\n---\nОпциональная подпапка внутри OUTPUT (создаётся автоматически; можно 'a/b')."}),
                "filename_prefix": ("STRING", {
                    "default": "clean",
                    "tooltip": "Prefix for the clean copy: '<prefix>_<name><ext>'. Empty -> '<name>_clean<ext>'.\n---\nПрефикс чистой копии: '<префикс>_<имя><ext>'. Пусто -> '<имя>_clean<ext>'."}),
            }
        }

    def strip(self, file_path, write_mode, subfolder, filename_prefix):
        src = (file_path or "").strip().strip('"').strip("'")
        if not src:
            raise ValueError("file_path is empty: specify a file or a folder.")
        return _process_files(src, write_mode, subfolder, filename_prefix,
                              "Image", "🖼️", _strip_image_bytes, IMAGE_EXTS)


# ==============================================================================
# NODE: 🎬🧹AGSoft Video Strip Metadata
# ==============================================================================
class AGSoft_Video_Strip_Metadata:
    DESCRIPTION = (
        "🎬🧹 Strips metadata (prompt/workflow/encoder/udta/meta tags) from\n"
        "MP4/MOV/WebM/MKV. NO re-encoding: streams are copied.\n"
        "The ORIGINAL is NEVER modified: a clean copy is written to the OUTPUT\n"
        "folder (optional subfolder).\n"
        "---\n"
        "🎬🧹 Вырезает метаданные (prompt/workflow/encoder/udta/meta) из\n"
        "MP4/MOV/WebM/MKV. БЕЗ перекодирования: потоки копируются.\n"
        "ОРИГИНАЛ НЕ ТРОГАЕТСЯ: чистая копия пишется в OUTPUT (можно в подпапку)."
    )
    CATEGORY = "AGSoft/Utils"
    FUNCTION = "strip"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filepath", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Absolute path to a video file or a folder (all supported videos inside).\n---\nАбсолютный путь к видеофайлу или папке (все поддерживаемые видео внутри)."}),
                "write_mode": (["clean_copy", "scan_only"], {
                    "default": "clean_copy",
                    "tooltip": "clean_copy = write a clean copy into OUTPUT (original untouched); scan_only = report only, no writes.\n---\nclean_copy = чистая копия в OUTPUT (оригинал не трогается); scan_only = только отчёт, без записи."}),
                "subfolder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional subfolder inside OUTPUT (created automatically; nested 'a/b' allowed).\n---\nОпциональная подпапка внутри OUTPUT (создаётся автоматически; можно 'a/b')."}),
                "filename_prefix": ("STRING", {
                    "default": "clean",
                    "tooltip": "Prefix for the clean copy: '<prefix>_<name><ext>'. Empty -> '<name>_clean<ext>'.\n---\nПрефикс чистой копии: '<префикс>_<имя><ext>'. Пусто -> '<имя>_clean<ext>'."}),
            }
        }

    def strip(self, file_path, write_mode, subfolder, filename_prefix):
        src = (file_path or "").strip().strip('"').strip("'")
        if not src:
            raise ValueError("file_path is empty: specify a file or a folder.")
        return _process_files(src, write_mode, subfolder, filename_prefix,
                              "Video", "🎬", _strip_video_bytes, VIDEO_EXTS)


NODE_CLASS_MAPPINGS = {
    "AGSoft_Image_Strip_Metadata": AGSoft_Image_Strip_Metadata,
    "AGSoft_Video_Strip_Metadata": AGSoft_Video_Strip_Metadata,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AGSoft_Image_Strip_Metadata": "🖼️🧹AGSoft Image Strip Metadata",
    "AGSoft_Video_Strip_Metadata": "🎬🧹AGSoft Video Strip Metadata",
}