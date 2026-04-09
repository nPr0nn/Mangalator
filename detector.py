"""
Bubble detection module using ogkalu/comic-text-and-bubble-detector.

Model classes
-------------
  0 → bubble       (the full balloon shape — used as inpaint mask)
  1 → text_bubble  (text region inside a balloon — used as OCR crop & render target)
  2 → text_free    (free-floating text, no balloon)

Two bounding boxes per entry
-----------------------------
  bubble_box  – the full balloon outline.  Used to build the inpaint mask so
                the entire white area is erased before the translation is drawn.

  text_box    – the tight region around the actual glyphs.  Used for:
                  • OCR crop  (better accuracy on a tight crop)
                  • Translation render target  (text is drawn inside this box,
                    not the larger balloon box)

  When the detector finds no text-region box inside a balloon, text_box
  falls back to bubble_box so the pipeline always has a valid render target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── model constants ────────────────────────────────────────────────────────────

REPO_ID           = "ogkalu/comic-text-and-bubble-detector"
INPUT_SIZE        = {"width": 640, "height": 640}
LABEL_BUBBLE      = 0   # full balloon
LABEL_TEXT_BUBBLE = 1   # text region inside balloon
LABEL_TEXT_FREE   = 2   # free-floating text (no balloon)


# ── data class ────────────────────────────────────────────────────────────────

@dataclass
class DetectedBubble:
    """
    One detected speech balloon.

    Coordinates are in original image pixels, format (x, y, w, h).

    bubble_box   — full balloon outline → inpaint mask
    text_box     — tight text region    → OCR crop + render target
    has_text_box — True when the detector found a dedicated text-region box
    """
    # Balloon outline
    bx: int
    by: int
    bw: int
    bh: int
    # Text region (defaults to balloon outline when no text box found)
    tx: int = field(default=None)
    ty: int = field(default=None)
    tw: int = field(default=None)
    th: int = field(default=None)
    has_text_box: bool = field(default=False)

    def __post_init__(self):
        if self.tx is None: self.tx = self.bx
        if self.ty is None: self.ty = self.by
        if self.tw is None: self.tw = self.bw
        if self.th is None: self.th = self.bh

    @property
    def bubble_box(self) -> dict:
        """Full balloon box as {x, y, w, h}."""
        return {"x": self.bx, "y": self.by, "w": self.bw, "h": self.bh}

    @property
    def text_box(self) -> dict:
        """Tight text-region box as {x, y, w, h}."""
        return {"x": self.tx, "y": self.ty, "w": self.tw, "h": self.th}

    def to_dict(self, idx: int, japanese: str = "") -> dict:
        """Serialise to the JSON shape used by the API."""
        return {
            "id":       idx,
            # bubble box (inpaint mask)
            "bx": self.bx, "by": self.by, "bw": self.bw, "bh": self.bh,
            # text box (OCR crop + render target)
            "tx": self.tx, "ty": self.ty, "tw": self.tw, "th": self.th,
            "has_text_box": self.has_text_box,
            "japanese": japanese,
        }


# ── lazy model state ───────────────────────────────────────────────────────────

_model:                object = None
_processor:            object = None
_device:               str    = "cpu"
_confidence_threshold: float  = 0.3
_ready:                bool   = False


def initialize(device: str = "cpu", confidence_threshold: float = 0.3) -> None:
    """Load the RT-DETR-v2 model from Hugging Face. Safe to call multiple times."""
    global _model, _processor, _device, _confidence_threshold, _ready

    if _ready:
        return

    import torch
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    _device               = device
    _confidence_threshold = confidence_threshold

    logger.info("[Detector] Loading %s …", REPO_ID)
    _processor = RTDetrImageProcessor.from_pretrained(REPO_ID, size=INPUT_SIZE)
    _model     = RTDetrV2ForObjectDetection.from_pretrained(REPO_ID)
    _model     = _model.to(_device)
    _model.eval()

    if _device.startswith("cuda"):
        try:
            _model = torch.compile(_model)
        except Exception:
            pass

    _ready = True
    logger.info(
        "[Detector] Ready on device=%s  threshold=%.2f", _device, _confidence_threshold
    )


def _ensure_ready() -> None:
    if not _ready:
        initialize()


# ── geometry helpers ───────────────────────────────────────────────────────────

def _intersection_area(a: list[int], b: list[int]) -> float:
    """Intersection area of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))


def _best_text_box(
    bubble_xyxy: list[int],
    text_boxes:  list[list[int]],
) -> tuple[int, list[int]] | tuple[None, None]:
    """
    Return (index, box) of the text box most contained within the bubble.
    Score = intersection / text-box-area  (1.0 = text box fully inside bubble).
    """
    best_score, best_idx, best_box = 0.0, None, None
    for i, tb in enumerate(text_boxes):
        tb_area = max(1, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        score   = _intersection_area(bubble_xyxy, tb) / tb_area
        if score > best_score:
            best_score, best_idx, best_box = score, i, tb
    return best_idx, best_box


# ── main detection function ────────────────────────────────────────────────────

def detect(pil_image: Image.Image) -> list[DetectedBubble]:
    """
    Run the RT-DETR-v2 detector and return DetectedBubble objects sorted in
    manga reading order (top-to-bottom, right-to-left within each ~60px row).

    Each bubble carries:
      • bubble_box  — full balloon (for inpainting)
      • text_box    — tight text area (for OCR and translation rendering)
    """
    import torch

    _ensure_ready()

    inputs = _processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]], device=_device)
    results      = _processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=_confidence_threshold,
    )[0]

    bubble_xyxy: list[list[int]] = []
    text_xyxy:   list[list[int]] = []

    for box, label in zip(results["boxes"], results["labels"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        if label.item() == LABEL_BUBBLE:
            bubble_xyxy.append([x1, y1, x2, y2])
        elif label.item() in (LABEL_TEXT_BUBBLE, LABEL_TEXT_FREE):
            text_xyxy.append([x1, y1, x2, y2])

    logger.info(
        "[Detector] Raw boxes — bubbles: %d  text: %d",
        len(bubble_xyxy), len(text_xyxy),
    )

    detected:  list[DetectedBubble] = []
    used_text: set[int]             = set()

    for bx in bubble_xyxy:
        x1, y1, x2, y2 = bx
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue

        db = DetectedBubble(bx=x1, by=y1, bw=bw, bh=bh)

        t_idx, tb = _best_text_box(bx, text_xyxy)
        if tb is not None:
            tx1, ty1, tx2, ty2 = tb
            db.tx, db.ty       = tx1, ty1
            db.tw, db.th       = tx2 - tx1, ty2 - ty1
            db.has_text_box    = True
            used_text.add(t_idx)

        detected.append(db)

    # Free-floating text boxes not matched to any balloon
    for i, tb in enumerate(text_xyxy):
        if i in used_text:
            continue
        x1, y1, x2, y2 = tb
        tw, th = x2 - x1, y2 - y1
        if tw < 10 or th < 10:
            continue
        detected.append(DetectedBubble(bx=x1, by=y1, bw=tw, bh=th, has_text_box=False))

    detected.sort(key=lambda d: (d.by // 60, -(d.bx + d.bw)))
    logger.info("[Detector] Returning %d final bubbles.", len(detected))
    return detected


def get_backend_name() -> str:
    return REPO_ID if _ready else f"{REPO_ID} (not loaded)"
