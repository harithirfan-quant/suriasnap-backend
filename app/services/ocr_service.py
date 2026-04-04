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
    Priority (highest → lowest):
      1. Meter reading difference (Semasa − Dahulu) — most reliable, immune to
         tariff-tier references like "600 kWh" appearing elsewhere in the bill.
      2. Explicit TNB labels ("Jumlah Penggunaan Anda", "Total Units", etc.)
      3. Fallback: any number immediately before or after "kWh".
    Sanity range: 50–5000 kWh (typical Malaysian residential).
    """
    text_lower = text.lower()
    # Collapse all whitespace/newlines into single spaces so that table cells
    # split across lines (e.g. "467\nkWh") are matched as "467 kwh".
    text_flat = re.sub(r'\s+', ' ', text_lower)

    def _safe_float(s):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None

    def _extract_first(pattern, source):
        for m in re.finditer(pattern, source):
            val = _safe_float(m.group(1))
            if val is not None and 50 <= val <= 5000:
                return val
        return None

    # ── 1. Meter reading difference: Semasa − Dahulu ─────────────────────────
    # TNB bills always show a meter table: Dahulu (previous) and Semasa (current).
    # Computing the difference avoids any confusion with tariff-tier references.
    meter_patterns = [
        # "Dahulu  6,695  Semasa  7,162"  (flattened table row)
        r"dahulu[^\d]{0,20}([\d,]+)[^\d]{0,50}semasa[^\d]{0,20}([\d,]+)",
        # "Previous  6695  Current  7162"
        r"previous[^\d]{0,20}([\d,]+)[^\d]{0,50}current[^\d]{0,20}([\d,]+)",
        # "Bacaan Lama  6695  Bacaan Baru  7162"
        r"bacaan\s+lama[^\d]{0,20}([\d,]+)[^\d]{0,50}bacaan\s+baru[^\d]{0,20}([\d,]+)",
    ]
    for pat in meter_patterns:
        for src in (text_flat, text_lower):
            m = re.search(pat, src)
            if m:
                prev = _safe_float(m.group(1))
                curr = _safe_float(m.group(2))
                if prev is not None and curr is not None:
                    diff = curr - prev
                    if 50 <= diff <= 5000:
                        return diff

    # ── 2. Explicit TNB bill labels ───────────────────────────────────────────
    labeled = [
        # "Jumlah Penggunaan Anda  kWh  467.00"  (table: label | unit | value)
        r"jumlah\s+penggunaan\s+anda\s+kwh\s+(\d[\d,]*(?:\.\d+)?)",
        # "Jumlah Penggunaan Anda  467 kWh"
        r"jumlah\s+penggunaan\s+anda[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Jumlah Unit / Total Units  450 kWh"
        r"(?:jumlah\s+unit|total\s+units?)[^\d]{0,80}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Penggunaan Semasa / Consumption"
        r"(?:penggunaan\s+semasa|consumption)[^\d]{0,80}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Unit Guna / Unit Semasa"
        r"unit\s+(?:guna|semasa)[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Current Month / Current Consumption"
        r"current\s+(?:month|consumption|use)[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
        # "Bil Semasa  467 kWh"
        r"bil\s+semasa[^\d]{0,50}(\d[\d,]*(?:\.\d+)?)\s*kwh",
    ]
    for pat in labeled:
        for src in (text_flat, text_lower):
            result = _extract_first(pat, src)
            if result is not None:
                return result

    # ── 3. Fallback: any number near "kWh" ───────────────────────────────────
    # Collect ALL candidate values and return the most plausible one.
    # We prefer values that do NOT equal common tariff tier thresholds (300, 600)
    # unless nothing else is available.
    candidates = []
    fallback_patterns = [
        r"(\d[\d,]*(?:\.\d+)?)\s*kwh",   # number before kWh
        r"kwh\W{0,20}(\d[\d,]*(?:\.\d+)?)\b",  # kWh header then number
    ]
    for pat in fallback_patterns:
        for src in (text_flat, text_lower):
            for m in re.finditer(pat, src):
                val = _safe_float(m.group(1))
                if val is not None and 50 <= val <= 5000:
                    candidates.append(val)

    if not candidates:
        return None

    # Prefer values that aren't exact tariff tier boundaries
    tier_boundaries = {300, 600, 900}
    non_tier = [v for v in candidates if v not in tier_boundaries]
    return non_tier[0] if non_tier else candidates[0]


def _parse_bill_amount(text: str) -> float | None:
    """
    Find the total amount payable in RM.
    Priority: labelled totals, then highest RM value found (bills show itemised
    amounts that are always smaller than the total).

    Uses a whitespace-flattened copy of the text so amounts split across
    OCR lines (e.g. "RM\n185.50") are still matched.
    """
    text_lower = text.lower()
    # Collapse all whitespace/newlines — handles table-layout OCR output
    text_flat = re.sub(r'\s+', ' ', text_lower)

    def _extract_first(pattern, source):
        m = re.search(pattern, source)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if 1 <= val <= 100_000:
                    return val
            except ValueError:
                pass
        return None

    labelled_patterns = [
        # "Jumlah Yang Perlu Dibayar / Amount Due  RM 185.50"
        r"(?:jumlah\s+(?:yang\s+)?(?:perlu\s+)?dibayar|amount\s+(?:due|payable))[^\d]{0,50}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        r"(?:jumlah|amount\s+due)[^\d]{0,40}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        # "TOTAL AMOUNT  RM 185.50" / "TOTAL  185.50"
        r"total\s+(?:amount\s+)?(?:due\s+)?[^\d]{0,30}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        # "Amaun Dibayar / Amount Paid"
        r"amaun[^\d]{0,40}(?:rm\s*)?(\d[\d,]*\.\d{2})",
        # "Bayaran / Payment"
        r"bayaran[^\d]{0,40}(?:rm\s*)?(\d[\d,]*\.\d{2})",
    ]

    # Try labelled patterns on flattened text first, then original
    for pat in labelled_patterns:
        for src in (text_flat, text_lower):
            result = _extract_first(pat, src)
            if result is not None:
                return result

    # Fallback: collect all RM-prefixed values from flattened text, return largest
    # (itemised charges are always less than the total)
    all_rm = []
    for src in (text_flat, text_lower):
        for v in re.findall(r"rm\s*(\d[\d,]*\.\d{2})", src):
            try:
                val = float(v.replace(",", ""))
                if 1 <= val <= 100_000:
                    all_rm.append(val)
            except ValueError:
                pass

    return max(all_rm) if all_rm else None


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
