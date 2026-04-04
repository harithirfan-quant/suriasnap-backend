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

def _parse_state(text: str) -> str | None:
    """
    Detect the Malaysian state from the bill text.
    TNB bills show the customer address which includes the state name.
    """
    text_lower = text.lower()

    # Ordered by specificity — longer/more specific patterns first
    state_patterns = [
        # W.P. / Wilayah Persekutuan variants
        (r'w\.?\s*p\.?\s*putrajaya|putrajaya',              'Putrajaya'),
        (r'w\.?\s*p\.?\s*labuan|labuan',                    'Labuan'),
        (r'w\.?\s*p\.?\s*kuala\s+lumpur|wilayah\s+persekutuan\s+kuala\s+lumpur', 'Kuala Lumpur'),
        # States
        (r'pulau\s+pinang|p\.?\s*pinang|penang',            'Penang'),
        (r'negeri\s+sembilan|n\.?\s*sembilan',               'Negeri Sembilan'),
        (r'\bperlis\b',                                      'Perlis'),
        (r'\bkedah\b',                                       'Kedah'),
        (r'\bperak\b',                                       'Perak'),
        (r'\bselangor\b',                                    'Selangor'),
        (r'kuala\s+lumpur|\bkl\b',                          'Kuala Lumpur'),
        (r'\bmelaka\b|\bmalacca\b',                          'Melaka'),
        (r'\bjohor\b',                                       'Johor'),
        (r'\bpahang\b',                                      'Pahang'),
        (r'\bterengganu\b',                                  'Terengganu'),
        (r'\bkelantan\b',                                    'Kelantan'),
        (r'\bsabah\b',                                       'Sabah'),
        (r'\bsarawak\b',                                     'Sarawak'),
    ]

    for pattern, state_name in state_patterns:
        if re.search(pattern, text_lower):
            return state_name

    return None


def _parse_consumption(text: str) -> float | None:
    """
    Find monthly consumption in kWh.
    Priority: labelled patterns first, then any number near 'kWh'.
    Sanity range: 50–5000 kWh (typical Malaysian residential).

    Key challenge: OCR of table-layout bills often puts the numeric value
    and the "kWh" unit on separate lines. We handle this by also searching
    a whitespace-flattened version of the text where newlines become spaces.
    """
    text_lower = text.lower()
    # Collapse all whitespace (including newlines) into a single space.
    # This lets patterns match across OCR line breaks, e.g. "450\nkWh" → "450 kwh".
    text_flat = re.sub(r'\s+', ' ', text_lower)

    def _extract(pattern, source):
        for m in re.finditer(pattern, source):
            try:
                val = float(m.group(1).replace(",", ""))
                if 50 <= val <= 5000:
                    return val
            except ValueError:
                pass
        return None

    # ── Labelled patterns (most reliable) ────────────────────────────────────
    labeled = [
        # "Jumlah Unit / Total Units  450 kWh"
        r"(?:jumlah\s+unit|total\s+units?)[^\d]{0,80}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Penggunaan / Penggunaan Semasa / Consumption"
        r"(?:penggunaan|consumption)[^\d]{0,80}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Unit Guna / Unit Semasa"
        r"unit\s+(?:guna|semasa)[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Current Month / Current Consumption"
        r"current\s+(?:month|consumption|use)[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Semasa" (standalone label common in newer TNB bills)
        r"\bsemasa\b[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Bil Semasa"
        r"bil\s+semasa[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
    ]

    # ── Fallback patterns ─────────────────────────────────────────────────────
    fallback = [
        # Number immediately before kWh (same line or after flattening)
        r"(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # kWh appears as a column header; the value follows (e.g. "kWh\n450")
        r"kwh\W{0,20}(\d[\d,]*(?:\.\d+)?)\b",
    ]

    # Search labelled patterns on flattened text first, then original
    for pat in labeled:
        for src in (text_flat, text_lower):
            result = _extract(pat, src)
            if result is not None:
                return result

    # Then fallback patterns
    for pat in fallback:
        for src in (text_flat, text_lower):
            result = _extract(pat, src)
            if result is not None:
                return result

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

def _confidence(consumption: float | None, amount: float | None,
                tariff: str | None, state: str | None) -> float:
    """
    Field-presence score weighted by importance to the solar assessment.
    Consumption is the most critical field.
    """
    weights = {
        "consumption": 0.50,
        "amount":      0.30,
        "tariff":      0.10,
        "state":       0.10,
    }
    found = {
        "consumption": consumption is not None,
        "amount":      amount is not None,
        "tariff":      tariff is not None,
        "state":       state is not None,
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
        state                 str | None
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

    consumption = _parse_consumption(raw_text)
    amount      = _parse_bill_amount(raw_text)
    tariff      = _parse_tariff_category(raw_text)
    state       = _parse_state(raw_text)
    confidence  = _confidence(consumption, amount, tariff, state)

    success = consumption is not None

    if success:
        found_fields = [
            f for f, v in [
                ("consumption", consumption),
                ("bill amount", amount),
                ("tariff", tariff),
                ("state", state),
            ] if v is not None
        ]
        message = f"Extracted: {', '.join(found_fields)}."
    else:
        message = (
            "Could not read monthly consumption from the bill. "
            "Please ensure the image is clear and well-lit, or enter the value manually."
        )

    return {
        "state":            state,
        "consumption_kwh":  consumption,
        "bill_amount_rm":   amount,
        "tariff_category":  tariff,
        "confidence_score": confidence,
        "raw_text":         raw_text,
        "success":          success,
        "message":          message,
    }
