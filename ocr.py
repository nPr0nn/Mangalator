"""
OCR module for manga text extraction.

Primary:  manga-ocr  — fine-tuned transformer for manga Japanese, handles
          stylised fonts, vertical text, and low-contrast panels very well.

Fallback: EasyOCR with Japanese — activates automatically if manga-ocr is
          not installed or fails to load.
"""

from PIL import Image

_mocr = None
_easyocr_reader = None
_backend = None   # "manga-ocr" | "easyocr" | None


def _load_backend():
    global _mocr, _easyocr_reader, _backend

    if _backend is not None:
        return

    # 1. Try manga-ocr
    try:
        from manga_ocr import MangaOcr
        print("[OCR] Loading manga-ocr model (first run downloads ~400 MB)…")
        _mocr = MangaOcr()
        _backend = "manga-ocr"
        print("[OCR] manga-ocr ready.")
        return
    except Exception as e:
        print(f"[OCR] manga-ocr unavailable: {e}")

    # 2. Try EasyOCR
    try:
        import easyocr
        print("[OCR] Loading EasyOCR (Japanese)…")
        _easyocr_reader = easyocr.Reader(["ja"], gpu=False, verbose=False)
        _backend = "easyocr"
        print("[OCR] EasyOCR ready.")
        return
    except Exception as e:
        print(f"[OCR] EasyOCR unavailable: {e}")

    _backend = "none"
    print("[OCR] WARNING: no OCR backend found. Install manga-ocr or easyocr.")


def run_ocr(image: Image.Image) -> str:
    """
    Extract Japanese text from a PIL Image crop of a speech bubble.
    Returns a string (may be empty if nothing detected).
    """
    _load_backend()

    if _backend == "manga-ocr":
        try:
            text = _mocr(image)
            return text.strip()
        except Exception as e:
            print(f"[OCR] manga-ocr error: {e}")
            return ""

    if _backend == "easyocr":
        try:
            import numpy as np
            arr = np.array(image.convert("RGB"))
            results = _easyocr_reader.readtext(arr, detail=0)
            return " ".join(results).strip()
        except Exception as e:
            print(f"[OCR] EasyOCR error: {e}")
            return ""

    # No backend — return placeholder so the pipeline still runs
    return "[OCR não disponível]"


def get_backend_name() -> str:
    _load_backend()
    return _backend or "none"
