"""
Vercel Python serverless function: in-place PDF translation, in two modes.
Translation itself is done in the BROWSER (Google Translate works there but is
blocked from datacenter IPs), so this function makes NO external calls.

POST /api/translate_pdf?mode=extract
  body: raw PDF bytes
  resp: {"texts": [unique block texts ...]}

POST /api/translate_pdf?mode=render
  body: JSON {"pdf": "<base64 pdf>", "map": {"<src text>": "<translated>", ...}}
  resp: the translated PDF bytes (original text whited out, translations drawn
        in place; images and layout untouched)
"""

import os
import re
import json
import base64
import urllib.parse
from http.server import BaseHTTPRequestHandler

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_REG = os.path.join(HERE, "fonts", "DejaVuSans.ttf")
FONT_BLD = os.path.join(HERE, "fonts", "DejaVuSans-Bold.ttf")
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def norm(t):
    return " ".join(t.split())


def _blocks(page):
    """Yield (rect, full_text, size, bold, color_int) for each text block."""
    for bl in page.get_text("dict")["blocks"]:
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
        yield (pymupdf.Rect(x0, y0, x1, y1), full, sp0["size"],
               bool(sp0["flags"] & 16), sp0.get("color", 0))


def extract_texts(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    seen, order = set(), []
    for page in doc:
        for _, full, _, _, _ in _blocks(page):
            k = norm(full)
            if k not in seen:
                seen.add(k); order.append(k)
    doc.close()
    return order


def _px(pix, x, y):
    x = max(0, min(pix.width - 1, int(x))); y = max(0, min(pix.height - 1, int(y)))
    return pix.pixel(x, y)


def sample_bg(pix, rect, zoom):
    ring = 3
    x0, y0, x1, y1 = rect.x0 * zoom, rect.y0 * zoom, rect.x1 * zoom, rect.y1 * zoom
    rs, gs, bs = [], [], []
    for px in (x0, (x0 + x1) / 2, x1 - 1):
        for py in (y0 - ring, y1 + ring):
            r, g, b = _px(pix, px, py); rs.append(r); gs.append(g); bs.append(b)
    for py in (y0, (y0 + y1) / 2, y1 - 1):
        for px in (x0 - ring, x1 + ring):
            r, g, b = _px(pix, px, py); rs.append(r); gs.append(g); bs.append(b)
    med = lambda a: sorted(a)[len(a) // 2] if a else 255
    return (med(rs) / 255.0, med(gs) / 255.0, med(bs) / 255.0)


def color_rgb(c):
    return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)


def render_translated(pdf_bytes, tmap):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    zoom = 2
    mat = pymupdf.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        jobs = []
        for rect, full, size, bold, cint in _blocks(page):
            en = tmap.get(norm(full))
            if en is None or en == "":
                continue
            jobs.append((rect, en, size, bold, color_rgb(cint), sample_bg(pix, rect, zoom)))

        ph = page.rect.height
        def avail_bottom(rect):
            limit = ph - 20
            for other, *_ in jobs:
                if other is rect:
                    continue
                horiz = not (other.x1 <= rect.x0 + 2 or other.x0 >= rect.x1 - 2)
                if horiz and other.y0 > rect.y0 + 3:
                    limit = min(limit, other.y0 - 1)
            return max(limit, rect.y0 + 8)

        # Truly remove the original text (redaction), filling with the local
        # background colour. Images are never touched (PDF_REDACT_IMAGE_NONE).
        for rect, _, _, _, _, bg in jobs:
            page.add_redact_annot(pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 2), fill=bg)
        if jobs:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
        for rect, en, size, bold, color, bg in jobs:
            ff = FONT_BLD if bold else FONT_REG
            fn = "djvb" if bold else "djv"
            wext = 25 if len(en) <= 4 else 0
            box = pymupdf.Rect(rect.x0, rect.y0 - 1, rect.x1 + wext, avail_bottom(rect))
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


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def _json(self, code, obj):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self._cors(); self.send_header("Content-Length", str(len(b))); self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        try:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            mode = q.get("mode", ["extract"])[0]
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if mode == "extract":
                self._json(200, {"texts": extract_texts(body)})
            elif mode == "render":
                payload = json.loads(body.decode("utf-8"))
                pdf_bytes = base64.b64decode(payload["pdf"])
                out = render_translated(pdf_bytes, payload.get("map", {}))
                self.send_response(200); self.send_header("Content-Type", "application/pdf")
                self._cors(); self.send_header("Content-Length", str(len(out))); self.end_headers()
                self.wfile.write(out)
            else:
                self._json(400, {"error": "unknown mode"})
        except Exception as e:
            self._json(500, {"error": str(e)})
