import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image: Image.Image) -> Image.Image:
    """
    Upscale if small, convert to grayscale, denoise, then apply Otsu threshold.
    Larger images give Tesseract more detail to work with.
    """
    img = np.array(image.convert("RGB"))

    # Upscale if the shorter side is below 1000 px
    h, w = img.shape[:2]
    if min(h, w) < 1000:
        scale = 1000 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


# ---------------------------------------------------------------------------
# Field parsers
# ---------------------------------------------------------------------------

def _parse_account_number(text: str) -> str | None:
    """
    TNB account numbers are 12 digits formatted as xxxx-xxxx-xxxx.
    Also accept 12 consecutive digits that OCR may have merged.
    """
    # Formatted: 1234-5678-9012
    m = re.search(r"\b(\d{4}[- ]\d{4}[- ]\d{4})\b", text)
    if m:
        return re.sub(r"[ ]", "-", m.group(1))

    # Unformatted 12-digit run (not part of a longer number)
    m = re.search(r"(?<!\d)(\d{12})(?!\d)", text)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:8]}-{d[8:]}"

    return None


def _parse_consumption(text: str) -> float | None:
    """
    Find monthly consumption in kWh.
    Priority: labelled patterns first, then any number preceding 'kWh'.
    Sanity range: 50–5000 kWh (typical Malaysian residential).
    """
    text_lower = text.lower()
    patterns = [
        # "Jumlah Unit / Total Units  450 kWh"
        r"(?:jumlah\s+unit|total\s+units?)[^\d]{0,30}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Penggunaan / Consumption  450 kWh"
        r"(?:penggunaan|consumption)[^\d]{0,30}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Unit Guna  450 kWh"
        r"unit\s+guna[^\d]{0,20}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # Fallback: any number immediately before "kwh"
        r"(\d[\d,]*(?:\.\d+)?)\s*kwh",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            value = float(m.group(1).replace(",", ""))
            if 50 <= value <= 5000:
                return value
    return None


def _parse_bill_amount(text: str) -> float | None:
    """
    Find the total amount payable in RM.
    Priority: labelled totals, then highest RM value found (bills show itemised
    amounts that are always smaller than the total).
    """
    text_lower = text.lower()

    labelled_patterns = [
        # "Jumlah / Amount  RM 123.45"  or  "RM123.45"
        r"(?:jumlah\s+(?:yang\s+)?(?:perlu\s+)?dibayar|amount\s+(?:due|payable))[^\d]{0,30}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        r"(?:jumlah|amount\s+due)[^\d]{0,20}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        # "TOTAL  RM 123.45"
        r"total[^\d]{0,20}(?:rm\s*)?(\d[\d,]*\.\d{2})",
    ]
    for pattern in labelled_patterns:
        m = re.search(pattern, text_lower)
        if m:
            value = float(m.group(1).replace(",", ""))
            if 1 <= value <= 100_000:
                return value

    # Fallback: collect all RM-prefixed values, return the largest
    rm_values = [
        float(v.replace(",", ""))
        for v in re.findall(r"rm\s*(\d[\d,]*\.\d{2})", text_lower)
    ]
    rm_values = [v for v in rm_values if 1 <= v <= 100_000]
    return max(rm_values) if rm_values else None


def _parse_tariff_category(text: str) -> str | None:
    """
    Extract tariff category from lines containing 'Tarif' / 'Tariff'.
    TNB residential tariff is typically 'Domestic' / 'E1' / 'Domestik'.
    """
    text_lower = text.lower()

    # Look for an explicit tariff label followed by the category on the same line
    m = re.search(r"tariff?[:\s]+([A-Za-z0-9/ ]{2,30})", text_lower)
    if m:
        return m.group(1).strip().title()

    # Known TNB tariff codes that may appear standalone
    known = re.search(
        r"\b(domestik|domestic|e1|e[1-9]|c[1-9]|d[1-9]|f[1-9]|tarif\s+[a-z]\d?)\b",
        text_lower,
    )
    if known:
        return known.group(1).strip().title()

    return None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _confidence(account: str | None, consumption: float | None,
                 amount: float | None, tariff: str | None) -> float:
    """
    Simple field-presence score: each field found = 25 points.
    Consumption is weighted higher as it's the most important field.
    """
    weights = {
        "account":     0.20,
        "consumption": 0.40,
        "amount":      0.25,
        "tariff":      0.15,
    }
    found = {
        "account":     account is not None,
        "consumption": consumption is not None,
        "amount":      amount is not None,
        "tariff":      tariff is not None,
    }
    score = sum(w for k, w in weights.items() if found[k])
    return round(score, 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bill_data(image: Image.Image) -> dict:
    """
    Run OCR on a TNB bill image and return all parsed fields.

    Returns
    -------
    dict with keys:
        account_number        str | None
        consumption_kwh       float | None
        bill_amount_rm        float | None
        tariff_category       str | None
        confidence_score      float   (0.0 – 1.0)
        raw_text              str
        success               bool    (True when at least consumption found)
        message               str
    """
    processed = _preprocess(image)
    raw_text = pytesseract.image_to_string(processed, lang="eng")

    account     = _parse_account_number(raw_text)
    consumption = _parse_consumption(raw_text)
    amount      = _parse_bill_amount(raw_text)
    tariff      = _parse_tariff_category(raw_text)
    confidence  = _confidence(account, consumption, amount, tariff)

    success = consumption is not None

    if success:
        found_fields = [
            f for f, v in [
                ("account number", account),
                ("consumption", consumption),
                ("bill amount", amount),
                ("tariff", tariff),
            ] if v is not None
        ]
        message = f"Extracted: {', '.join(found_fields)}."
    else:
        message = (
            "Could not read monthly consumption from the bill. "
            "Please ensure the image is clear and well-lit, or enter the value manually."
        )

    return {
        "account_number":   account,
        "consumption_kwh":  consumption,
        "bill_amount_rm":   amount,
        "tariff_category":  tariff,
        "confidence_score": confidence,
        "raw_text":         raw_text,
        "success":          success,
        "message":          message,
    }
