"""
Bill extraction adapter.

This wraps the EXISTING free Tesseract OCR pipeline
(`app.services.ocr_service.extract_bill_data`) and maps its output into the
richer, stable extraction schema the WhatsApp flow expects.

Why an adapter instead of calling ocr_service directly?
  - It gives the rest of the app a single, stable shape to depend on.
  - It is the one seam to swap in Claude Vision later (just reimplement
    `extract_bill()` to call Claude and return the same dict) without touching
    the orchestrator or storage layers.

Tesseract reliably reads only: state, total_kwh, total_amount_rm, confidence.
Fields like customer_name / address / bill_date are returned empty — we never
hallucinate them. The schema keeps the keys so a future Claude swap is drop-in.
"""

import logging

from PIL import Image

from app.services import ocr_service

logger = logging.getLogger("suriasnap.extraction")

# Extraction quality below this is treated as "low confidence" by the
# orchestrator, which then asks the user to confirm / type their kWh.
LOW_CONFIDENCE_THRESHOLD = 0.7


def _load_image(path: str) -> Image.Image:
    """Open an image, or render the first page of a PDF eBill at 300 DPI."""
    if path.lower().endswith(".pdf"):
        from pdf2image import convert_from_path

        pages = convert_from_path(path, dpi=300, first_page=1, last_page=1)
        if not pages:
            raise ValueError("PDF has no pages")
        return pages[0]
    return Image.open(path)


def _empty_schema() -> dict:
    return {
        "customer_name": "",
        "address": "",
        "state": "",
        "bill_date": "",
        "billing_period_start": "",
        "billing_period_end": "",
        "total_kwh": 0,
        "total_amount_rm": 0,
        "meter_phase": "unknown",
        "confidence": 0.0,
        "notes": [],
    }


def extract_bill(path: str) -> dict:
    """
    Run OCR on a saved bill file and return the extraction schema dict.

    Never raises for a bad scan — on any failure it returns the empty schema
    with confidence 0 and an explanatory note, so the orchestrator can fall
    back to asking the user for their kWh manually.
    """
    schema = _empty_schema()
    try:
        image = _load_image(path)
    except Exception as exc:
        logger.warning("Could not open bill file %s: %s", path, exc)
        schema["notes"].append("Could not open the uploaded file.")
        return schema

    try:
        ocr = ocr_service.extract_bill_data(image)
    except Exception as exc:
        logger.exception("OCR failed for %s", path)
        schema["notes"].append(f"OCR error: {exc}")
        return schema

    schema["state"] = ocr.get("state") or ""
    schema["total_kwh"] = ocr.get("consumption_kwh") or 0
    schema["total_amount_rm"] = ocr.get("bill_amount_rm") or 0
    schema["confidence"] = ocr.get("confidence_score") or 0.0
    schema["notes"].append("Extracted via on-device Tesseract OCR (free).")
    if not ocr.get("success"):
        schema["notes"].append(
            ocr.get("message") or "Monthly usage could not be read confidently."
        )
    return schema


def is_low_confidence(extraction: dict) -> bool:
    return (extraction.get("confidence") or 0.0) < LOW_CONFIDENCE_THRESHOLD


def plausible_kwh(value) -> bool:
    """Monthly residential usage sanity range."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return 50 <= v <= 5000
