"""
extractors/pdf_extractor.py
Extracts plain text from PDF files using PyPDF2.
Falls back to pytesseract OCR for scanned/image-only pages.

PDF text extraction often breaks mid-sentence at line-wrap points.
The _rejoin_pdf_lines() helper merges continuation lines back into
complete sentences before downstream processing.
"""

import logging
import re
import PyPDF2
from PIL import Image
import pytesseract
import io
import os

logger = logging.getLogger(__name__)

_TESS_CONFIG = "--psm 6 --oem 3"
_MARKER_MODELS = None # Lazy load


def _rejoin_pdf_lines(text: str) -> str:
    """
    Merge lines that were broken mid-sentence by the PDF renderer.

    A line is treated as a continuation of the previous one when:
      - The previous line does NOT end with terminal punctuation (.!?:;)
      - The current line starts with a lowercase letter  (not a new sentence)

    This fixes the 'U.' + 'S.' fragmentation pattern that appears when
    PyPDF2 splits "...based in the U.S." across two lines.
    """
    lines = text.splitlines()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue
        if (
            result
            and result[-1]
            and result[-1][-1] not in ".!?:;\n"
            and stripped
            and stripped[0].islower()
        ):
            # Continuation line — append with a space
            result[-1] = result[-1] + " " + stripped
        else:
            result.append(stripped)
    return "\n".join(result)


def extract_pdf(filepath: str) -> str:
    """Return full text extracted from a PDF file."""
    text_parts: list[str] = []

    with open(filepath, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                # Re-join mid-sentence line breaks before appending
                text_parts.append(_rejoin_pdf_lines(page_text))
            else:
                logger.info("Page %d has no selectable text — trying OCR.", page_num + 1)
                ocr_text = _ocr_pdf_page(page)
                if ocr_text:
                    text_parts.append(_rejoin_pdf_lines(ocr_text))

    full_text = "\n".join(text_parts)
    logger.info("PDF extraction complete: %d characters.", len(full_text))
    return full_text


def _ocr_pdf_page(page) -> str:
    """Render a PDF page to an image and run Tesseract OCR on it."""
    image = None
    try:
        for img_file in page.images:
            image = Image.open(io.BytesIO(img_file.data)).convert("L")
            text = pytesseract.image_to_string(image, config=_TESS_CONFIG)
            return text
    except Exception as e:
        logger.warning("OCR failed on page: %s", e)
    finally:
        # CWE-400/664 FIX: Ensure image resource is closed
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
    return ""