"""
Vercel Python serverless function: translate a digital PDF IN PLACE.

Reads the PDF's real text layer, translates each text block with the free
Google endpoint, whites out the original text (using the sampled local
background colour) and redraws the translation in the same place with the
original text colour. Images and vector graphics are left untouched, so the
output PDF looks like the source with only the text changed.

POST  /api/translate_pdf?sl=<source|auto>&tl=<target>
body: the raw PDF bytes (application/pdf / octet-stream)
resp: the translated PDF bytes
"""

import os
import re
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_REG = os.path.join(HERE, "fonts", "DejaVuSans.ttf")
FONT_BLD = os.path.join(HERE, "fonts", "DejaVuSans-Bold.ttf")

HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)  # a real letter somewhere
MAX_CHARS = 4500
_CACHE = {}


# --------------------------------------------------------------------------
# Translation via the free Google endpoint
# --------------------------------------------------------------------------
def _google_once(text, sl, tl):
    url = ("https://translate.googleapis.com/translate_a/single?client=gtx&sl="
           + urllib.parse.quote(sl) + "&tl=" + urllib.parse.quote(tl)
           + "&dt=t&q=" + urllib.parse.quote(text))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8"))
    return "".join(seg[0] for seg in (data[0] or []) if seg and seg[0])


def _split(text, limit):
    parts, rem = [], text
    while len(rem) > limit:
        win = rem[:limit]
        cut = max(win.rfind("\n"), win.rfind(". "), win.rfind("! "),
                  win.rfind("? "), win.rfind(" "))
        if cut <= 0:
            cut = limit
        parts.append(rem[:cut]); rem = rem[cut:]
    if rem:
        parts.append(rem)
    return parts


def translate(text, sl, tl):
    if not text or not text.strip():
        return text
    key = sl + "|" + tl + "|" + text
    if key in _CACHE:
        return _CACHE[key]
    stripped = text.strip()
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]

    def one(t):
        last = None
        for a in range(3):
            try:
                return _google_once(t, sl, tl)
            except Exception as e:
                last = e; time.sleep(0.6 * (a + 1))
        print("translate failed:", last)
        return t  # fall back to original on repeated failure

    if len(stripped) <= MAX_CHARS:
        core = one(stripped)
    else:
        core = "".join(one(p) for p in _split(stripped, MAX_CHARS))
    result = lead + core + trail
    _CACHE[key] = result
    return result


# --------------------------------------------------------------------------
# Background colour sampling from a rendered page pixmap
# --------------------------------------------------------------------------
def _px(pixmap, x, y):
    x = max(0, min(pixmap.width - 1, int(x)))
    y = max(0, min(pixmap.height - 1, int(y)))
    return pixmap.pixel(x, y)  # (r,g,b)


def sample_bg(pixmap, rect, zoom):
    ring = 3
    x0, y0, x1, y1 = (rect.x0 * zoom, rect.y0 * zoom, rect.x1 * zoom, rect.y1 * zoom)
    xs = [x0, (x0 + x1) / 2, x1 - 1]
    ys = [y0, (y0 + y1) / 2, y1 - 1]
    rs, gs, bs = [], [], []
    for px in xs:
        for py in (y0 - ring, y1 + ring):
            r, g, b = _px(pixmap, px, py); rs.append(r); gs.append(g); bs.append(b)
    for py in ys:
        for px in (x0 - ring, x1 + ring):
            r, g, b = _px(pixmap, px, py); rs.append(r); gs.append(g); bs.append(b)
    med = lambda a: sorted(a)[len(a) // 2] if a else 255
    return (med(rs) / 255.0, med(gs) / 255.0, med(bs) / 255.0)


def int_color_to_rgb(c):
    return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)


# --------------------------------------------------------------------------
# Core: translate one PDF (bytes -> bytes)
# --------------------------------------------------------------------------
def translate_pdf_bytes(pdf_bytes, sl, tl):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    zoom = 2
    mat = pymupdf.Matrix(zoom, zoom)

    # -------- Pass A: collect every block (with bg sampled from the ORIGINAL) --------
    page_jobs = []
    unique = set()
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        dd = page.get_text("dict")
        jobs = []
        for bl in dd["blocks"]:
            if bl.get("type") != 0:
                continue
            lines = []
            for ln in bl["lines"]:
                s = "".join(sp["text"] for sp in ln["spans"])
                if s.strip():
                    lines.append((ln, s))
            if not lines:
                continue
            full = " ".join(s for _, s in lines).strip()
            if not HAS_LETTER.search(full):
                continue
            x0 = min(l["bbox"][0] for l, _ in lines); y0 = min(l["bbox"][1] for l, _ in lines)
            x1 = max(l["bbox"][2] for l, _ in lines); y1 = max(l["bbox"][3] for l, _ in lines)
            sp0 = lines[0][0]["spans"][0]
            size = sp0["size"]; bold = bool(sp0["flags"] & 16)
            color = int_color_to_rgb(sp0.get("color", 0))
            rect = pymupdf.Rect(x0, y0, x1, y1)
            bg = sample_bg(pix, rect, zoom)
            jobs.append((rect, full, size, bold, color, bg))
            unique.add(full)
        page_jobs.append(jobs)

    # -------- Translate all unique block texts concurrently (fills _CACHE) --------
    uniq = list(unique)
    if uniq:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda t: translate(t, sl, tl), uniq))

    # -------- Pass B: white out originals and redraw the (cached) translations --------
    for page, jobs in zip(doc, page_jobs):
        # For each block, find how far down it may grow before hitting the next
        # block that sits below it and overlaps its horizontal span. This caps
        # the redraw area so a longer translation can never overlap neighbours.
        ph = page.rect.height
        def avail_bottom(rect):
            limit = ph - 20
            for other, *_ in jobs:
                if other is rect:
                    continue
                horiz = not (other.x1 <= rect.x0 + 2 or other.x0 >= rect.x1 - 2)
                # a block that STARTS clearly below this one caps how far it can grow
                if horiz and other.y0 > rect.y0 + 3:
                    limit = min(limit, other.y0 - 1)
            return max(limit, rect.y0 + 8)  # keep a sane minimum height

        # white/paint out originals with their local background colour
        for rect, _, _, _, _, bg in jobs:
            page.draw_rect(pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 2),
                           color=None, fill=bg)
        # translate + redraw, bounded to the available space
        for rect, full, size, bold, color, bg in jobs:
            en = translate(full, sl, tl)
            ff = FONT_BLD if bold else FONT_REG
            fn = "djvb" if bold else "djv"
            wext = 25 if len(en) <= 4 else 0
            bottom = avail_bottom(rect)
            box = pymupdf.Rect(rect.x0, rect.y0 - 1, rect.x1 + wext, bottom)
            s = size
            while s > 3.5:
                rc = page.insert_textbox(box, en, fontsize=s, fontname=fn, fontfile=ff,
                                         color=color, align=0, lineheight=1.12)
                if rc >= 0:
                    break
                s -= 0.5
    out = doc.tobytes(deflate=True, garbage=3)
    doc.close()
    return out


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        try:
            qs = urllib.parse.urlparse(self.path).query
            q = urllib.parse.parse_qs(qs)
            sl = (q.get("sl", ["auto"])[0]) or "auto"
            tl = (q.get("tl", ["en"])[0]) or "en"
            length = int(self.headers.get("Content-Length", "0"))
            pdf_bytes = self.rfile.read(length)
            out = translate_pdf_bytes(pdf_bytes, sl, tl)
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self._cors()
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(msg)
