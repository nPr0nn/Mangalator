"""
Inpainting module — flood-fill interior eraser.

No ML model required.  For each text box the algorithm:

  1. Crops the image to the text bounding box.
  2. Finds a white seed pixel (starts from centre, searches outward).
  3. BFS flood-fills through white pixels (>= WHITE_THRESH) to map the
     balloon's interior.  Black border strokes act as walls and are never
     crossed, so the fill stays inside the balloon.
  4. For every row in the crop, fills the *full horizontal span* between
     the leftmost and rightmost interior pixel white.  This covers text
     strokes that would otherwise interrupt the fill — they sit inside the
     span and get painted over.
  5. Writes the result back into a copy of the original image.

Public API
----------
    inpaint_bubbles(pil_image, text_boxes) -> PIL.Image
        text_boxes: list of {x, y, w, h} dicts (tight text-region boxes).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── tuneable ──────────────────────────────────────────────────────────────────
WHITE_THRESH    = 200   # pixels >= this in all channels count as "white"
MAX_SEED_SEARCH = 30    # grid steps when centre pixel is not white

# ── helpers ───────────────────────────────────────────────────────────────────

def _find_seed(gray: np.ndarray) -> tuple[int, int] | None:
    """Return (row, col) of a white seed pixel, or None."""
    h, w  = gray.shape
    cr, cc = h // 2, w // 2

    if gray[cr, cc] >= WHITE_THRESH:
        return cr, cc

    step = max(1, min(h, w) // (MAX_SEED_SEARCH * 2))
    for dr in range(-MAX_SEED_SEARCH, MAX_SEED_SEARCH + 1):
        for dc in range(-MAX_SEED_SEARCH, MAX_SEED_SEARCH + 1):
            r, c = cr + dr * step, cc + dc * step
            if 0 <= r < h and 0 <= c < w and gray[r, c] >= WHITE_THRESH:
                return r, c

    # last resort — any bright pixel
    ys, xs = np.where(gray >= WHITE_THRESH)
    if len(ys):
        return int(ys[0]), int(xs[0])
    return None


def _erase_text_in_box(img_rgb: np.ndarray, box: dict) -> None:
    """Erase text inside *box* in-place on *img_rgb* (H×W×3 uint8)."""
    h_img, w_img = img_rgb.shape[:2]
    x1 = max(0, box["x"])
    y1 = max(0, box["y"])
    x2 = min(w_img, box["x"] + box["w"])
    y2 = min(h_img, box["y"] + box["h"])

    if x2 <= x1 or y2 <= y1:
        return

    crop = img_rgb[y1:y2, x1:x2]                    # view into original
    gray = np.mean(crop, axis=2).astype(np.uint8)    # single-channel

    seed = _find_seed(gray)
    if seed is None:
        # No white interior found — paint the whole box white
        logger.debug("[Inpainter] No white seed in box %s — solid fill.", box)
        crop[:] = 255
        return

    # BFS through white pixels to map balloon interior
    h, w    = gray.shape
    visited  = np.zeros((h, w), dtype=bool)
    interior = np.zeros((h, w), dtype=bool)
    queue    = deque([seed])
    visited[seed] = True

    while queue:
        r, c = queue.popleft()
        if gray[r, c] < WHITE_THRESH:
            continue                      # border stroke — wall, skip
        interior[r, c] = True
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Row-span fill: paint everything between leftmost and rightmost
    # interior pixel white — covers text strokes inside the balloon
    result = crop.copy()
    for row in range(h):
        cols = np.where(interior[row])[0]
        if len(cols) == 0:
            continue
        result[row, cols[0]:cols[-1] + 1] = 255

    crop[:] = result       # write back through the view


# ── public API ────────────────────────────────────────────────────────────────

def inpaint_bubbles(pil_image: Image.Image, text_boxes: list[dict]) -> Image.Image:
    """
    Erase text from all text-box regions and return a new clean PIL image.

    Args:
        pil_image:  RGB PIL image of the manga page.
        text_boxes: List of {x, y, w, h} dicts (tight text-region boxes).
    """
    if not text_boxes:
        return pil_image.copy()

    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)

    for i, box in enumerate(text_boxes):
        logger.debug("[Inpainter] box %d: %s", i, box)
        _erase_text_in_box(img_rgb, box)

    logger.info("[Inpainter] Erased %d box(es).", len(text_boxes))
    return Image.fromarray(img_rgb)


def get_backend_name() -> str:
    return "flood-fill-inpainter"
