"""
Manga Translator — FastAPI backend

Pipeline
--------
  POST /detect   → RT-DETR-v2 detects bubbles, manga-ocr reads text
                   Returns: original_image, image_w/h, bubbles[]
                   Each bubble: {id, bx,by,bw,bh, tx,ty,tw,th, has_text_box, japanese}
                     b* = full balloon box  (inpaint mask)
                     t* = tight text box   (OCR crop + render target)

  POST /ocr-crop → OCR a single manually-drawn region

  POST /inpaint  → flood-fill erases text pixels inside each text_box using
                   the white balloon interior (border strokes are preserved)
                   Returns: inpainted_image (base64 PNG)

  POST /render   → Draws translated text inside each text_box on the
                   inpainted image, returns final result
"""

import io
import os
import json
import base64
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ocr       import run_ocr,          get_backend_name as ocr_backend
from translator import translate_text,  get_backend_name as translator_backend
from detector  import detect,            get_backend_name as detector_backend
from inpainter import inpaint_bubbles,  get_backend_name as inpainter_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── image helpers ─────────────────────────────────────────────────────────────

def crop_box(pil_img: Image.Image, b: dict, pad: int = 2) -> Image.Image:
    """Crop a {x, y, w, h} region from a PIL image with optional padding."""
    iw, ih = pil_img.size
    return pil_img.crop((
        max(0, b["x"] - pad),
        max(0, b["y"] - pad),
        min(iw, b["x"] + b["w"] + pad),
        min(ih, b["y"] + b["h"] + pad),
    ))


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


# ── text rendering ────────────────────────────────────────────────────────────

# Set MANGA_FONT=filename.ttf (or .otf) to pick a specific font from fonts/.
# If unset, the first file alphabetically is used.
_FONTS_DIR    = Path("fonts")
_SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


def _resolve_font_path():
    """
    Return the Path of the first usable font file, priority:
      1. MANGA_FONT env var — filename (or full path) of a .ttf/.otf font
      2. Any .ttf / .otf in the fonts/ folder (alphabetical order)
      3. Known system font paths
    Returns None if nothing is found (falls back to PIL default).
    """
    env = os.environ.get("MANGA_FONT", "").strip()
    if env:
        # Accept bare filename (looked up inside fonts/) or an absolute path
        candidate = Path(env) if Path(env).is_absolute() else _FONTS_DIR / env
        if candidate.suffix.lower() not in {".ttf", ".otf"}:
            log.warning("[Font] MANGA_FONT '%s' is not a .ttf/.otf file — ignoring.", env)
        else:
            try:
                ImageFont.truetype(str(candidate), 12)
                log.info("[Font] Using MANGA_FONT: %s", candidate)
                return candidate
            except Exception:
                log.warning("[Font] MANGA_FONT '%s' could not be loaded — falling back.", candidate)

    if _FONTS_DIR.is_dir():
        candidates = sorted(
            p for p in _FONTS_DIR.iterdir()
            if p.suffix.lower() in {".ttf", ".otf"}
        )
        for p in candidates:
            try:
                ImageFont.truetype(str(p), 12)
                log.info("[Font] Using custom font: %s", p)
                return p
            except Exception:
                pass

    for path_str in _SYSTEM_FONTS:
        p = Path(path_str)
        if p.exists():
            log.info("[Font] Using system font: %s", p)
            return p

    log.warning("[Font] No font found — PIL default will be used.")
    return None


_FONT_PATH = _resolve_font_path()

def list_fonts() -> list[str]:
    """Return sorted list of .ttf/.otf filenames available in fonts/ recursively."""
    if not _FONTS_DIR.is_dir():
        return []

    fonts = list(_FONTS_DIR.rglob("*.ttf")) + list(_FONTS_DIR.rglob("*.otf"))
    return sorted(p.name for p in fonts)


# Single reusable draw context for text measurement (never rendered).
_DUMMY_DRAW = ImageDraw.Draw(Image.new("RGB", (1, 1)))


def get_font(size: int, font_path: Path | None = None) -> ImageFont.FreeTypeFont:
    path = font_path or _FONT_PATH
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            log.warning("[Font] truetype load failed for size %d — using PIL default.", size)
    # load_default(size=) respects the requested size and supports extended Latin
    return ImageFont.load_default(size=size)


def _text_width(text: str, font) -> int:
    """Accurate pixel width: right edge minus left bearing."""
    bb = _DUMMY_DRAW.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def wrap_text(text: str, font, max_width: int) -> list[str]:
    words   = text.split()
    lines   = []
    current = ""
    for word in words:
        trial = (current + " " + word).strip()
        if _text_width(trial, font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _block_height(lines: list[str], font, line_gap: int) -> int:
    """Total pixel height of a wrapped text block using real font metrics."""
    ascent, descent = font.getmetrics()
    return (ascent + descent) * len(lines) + line_gap * (len(lines) - 1)


def draw_text_in_box(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, text: str,
                     font_path: Path | None = None):
    """
    Render *text* centred inside the given box, picking the largest font size
    that allows the wrapped text to fit without clipping.

    Size range is derived from the box:
      - ceiling: min(72, max_h) — lets single short lines fill the box
      - floor: 7 px — smallest legible size; text is always rendered so
        content is never silently dropped
    Line spacing is 15 % of line height, matching comic-book convention.
    """
    padding = max(4, int(min(w, h) * 0.06))
    max_w   = w - padding * 2
    max_h   = h - padding * 2
    if max_w <= 0 or max_h <= 0:
        return

    min_size = 7
    max_size = max(min_size, min(72, max_h))   # let the fit-check do the work

    chosen_font  = get_font(min_size, font_path)
    chosen_lines = wrap_text(text, chosen_font, max_w)

    for size in range(max_size, min_size - 1, -1):
        font            = get_font(size, font_path)
        lines           = wrap_text(text, font, max_w)
        ascent, descent = font.getmetrics()
        line_gap        = max(1, int((ascent + descent) * 0.15))
        if _block_height(lines, font, line_gap) <= max_h:
            chosen_font, chosen_lines = font, lines
            break
    # Loop exhausted without break → nothing fit; chosen_font stays at min_size
    # so text is still rendered (possibly clipped) rather than silently lost.

    ascent, descent = chosen_font.getmetrics()
    line_h          = ascent + descent
    line_gap        = max(1, int(line_h * 0.15))
    total_h         = _block_height(chosen_lines, chosen_font, line_gap)

    ty = y + padding + max(0, (max_h - total_h) // 2)
    for ln in chosen_lines:
        lw = _text_width(ln, chosen_font)
        draw.text(
            (x + padding + max(0, (max_w - lw) // 2), ty),
            ln, fill="black", font=chosen_font,
        )
        ty += line_h + line_gap


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Manga Translator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "detector":   detector_backend(),
        "ocr":        ocr_backend(),
        "inpainter":  inpainter_backend(),
        "translator": translator_backend(),
    }


# ── Step 1: detect + OCR ──────────────────────────────────────────────────────

@app.post("/detect")
async def detect_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Envie um arquivo de imagem.")
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Não foi possível abrir a imagem.")

    raw_bubbles = detect(pil_img)
    bubbles     = []

    for i, db in enumerate(raw_bubbles):
        try:
            jp = run_ocr(crop_box(pil_img, db.text_box)).strip()
        except Exception as e:
            log.warning("OCR error bubble %d: %s", i, e)
            jp = ""
        bubbles.append(db.to_dict(i, jp))

    log.info("Detected %d bubbles.", len(bubbles))
    return JSONResponse({
        "original_image": pil_to_b64(pil_img),
        "image_width":    pil_img.width,
        "image_height":   pil_img.height,
        "bubbles":        bubbles,
    })


# ── OCR on a single manually-drawn crop ──────────────────────────────────────

@app.post("/ocr-crop")
async def ocr_crop_endpoint(
    file: UploadFile = File(...),
    x:    int        = Form(...),
    y:    int        = Form(...),
    w:    int        = Form(...),
    h:    int        = Form(...),
):
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagem inválida.")

    try:
        text = run_ocr(crop_box(pil_img, {"x": x, "y": y, "w": w, "h": h})).strip()
    except Exception as e:
        raise HTTPException(500, f"OCR error: {e}")

    return JSONResponse({"japanese": text})


# ── Step 2 → 3: inpaint balloon regions ──────────────────────────────────────

@app.post("/inpaint")
async def inpaint_endpoint(
    file:    UploadFile = File(...),   # original manga page
    bubbles: str        = Form(...),   # JSON [{id, bx,by,bw,bh, tx,ty,tw,th, ...}]
):
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Imagem inválida.")

    try:
        bubble_list = json.loads(bubbles)
    except Exception:
        raise HTTPException(400, "Lista de balões inválida.")

    # Pass the tight text boxes — the flood-fill eraser works on these,
    # detecting the white balloon interior and leaving border strokes intact.
    bbox_list = [{"x": b["tx"], "y": b["ty"], "w": b["tw"], "h": b["th"]}
                 for b in bubble_list]

    try:
        inpainted = inpaint_bubbles(pil_img, bbox_list)
    except Exception as e:
        log.error("Inpainting failed: %s", e)
        raise HTTPException(500, f"Inpainting error: {e}")

    return JSONResponse({
        "inpainted_image": pil_to_b64(inpainted),
    })


# ── List available fonts ─────────────────────────────────────────────────────

@app.get("/fonts")
def fonts_endpoint():
    return JSONResponse({"fonts": list_fonts()})


# ── Step 3 → 4: translate + render on inpainted image ────────────────────────

class RenderRequest(BaseModel):
    inpainted_b64: str
    bubbles: list[dict]
    font: str | None = None   # filename from fonts/, e.g. "MyComic.ttf"


@app.post("/render")
async def render_endpoint(body: RenderRequest):
    try:
        pil_img = b64_to_pil(body.inpainted_b64)
    except Exception:
        raise HTTPException(400, "Imagem inpainted inválida.")

    bubble_list = body.bubbles

    # Resolve requested font (must be in fonts/ for safety)
    chosen_font_path: Path | None = None
    if body.font:
        candidate = _FONTS_DIR / body.font
        if candidate.suffix.lower() in {".ttf", ".otf"} and candidate.exists():
            chosen_font_path = candidate
        else:
            log.warning("[Font] Requested font '%s' not found — using default.", body.font)

    translated_img = pil_img.copy()
    draw           = ImageDraw.Draw(translated_img)
    results        = []

    for b in bubble_list:
        jp = b.get("japanese", "").strip()
        if not jp:
            continue
        try:
            pt = translate_text(jp)
            # Render inside the tight text box, not the full balloon
            draw_text_in_box(
                draw,
                b["tx"], b["ty"], b["tw"], b["th"],
                pt,
                font_path=chosen_font_path,
            )
            results.append({"id": b["id"], "japanese": jp, "portuguese": pt})
            log.info("Bubble %s: %s → %s", b["id"], jp, pt)
        except Exception as e:
            log.warning("Translation/render failed bubble %s: %s", b["id"], e)

    return JSONResponse({
        "translated_image": pil_to_b64(translated_img),
        "bubbles":          results,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
