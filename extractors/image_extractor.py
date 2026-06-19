"""
extractors/image_extractor.py
Uses Tesseract OCR (via pytesseract) to extract text from image files.
Supports PNG, JPG, JPEG, TIFF, BMP.

Tesseract configuration notes:
  --psm 6  : Assume a single uniform block of text (best for single-page
              documents such as resumes, reports, and slides).
  --oem 3  : Use the best available engine (LSTM or Legacy, whichever is loaded).
"""

import logging
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

# Tesseract config: PSM 6 treats the whole image as one text block.
# This drastically reduces spurious single-character "sentences" that appear
# when PSM 3 (auto-segmentation) splits columns incorrectly.
_TESS_CONFIG = "--psm 6 --oem 3"

# If the image is narrower than this, upscale it so Tesseract has enough
# resolution to resolve fine character details (especially punctuation).
_MIN_WIDTH_PX = 1800


def extract_image(filepath: str) -> str:
    """Return OCR text from an image file."""
    image = None
    try:
        image = Image.open(filepath)

        # Convert to greyscale to improve OCR accuracy on coloured backgrounds
        image = image.convert("L")

        # Upscale small images — Tesseract performs poorly on low-res inputs
        w, h = image.size
        if w < _MIN_WIDTH_PX:
            scale = _MIN_WIDTH_PX / w
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.LANCZOS)
            logger.info(
                "Image upscaled from %dx%d to %dx%d for better OCR quality.",
                w, h, new_size[0], new_size[1],
            )

        text = pytesseract.image_to_string(image, lang="eng", config=_TESS_CONFIG)
        logger.info("Image OCR complete: %d characters.", len(text))
        return text
    finally:
        # CWE-400/664 FIX: Ensure image resource is closed
        if image is not None:
            try:
                image.close()
            except Exception:
                pass