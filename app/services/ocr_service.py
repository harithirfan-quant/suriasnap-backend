import base64
import io
import json
import logging
import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger("suriasnap.ocr")

tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Bill text is small, especially on myTNB screenshots. Upscale the short side
# generously — Tesseract is far more reliable on big glyphs, and this is what
# made real bills (and WhatsApp's JPEG-compressed versions) read consistently
# as 467 instead of flipping to 316/600. Cap the long side to bound OCR time.
TARGET_MIN_SIDE = 2000
MAX_LONG_SIDE   = 3000

# Tilt correction is only applied within this band: below the minimum the
# rotation isn't worth the interpolation blur, above the maximum the detected
# angle is almost certainly a false positive (e.g. a diagonal fold or shadow).
DESKEW_MIN_DEG = 0.3
DESKEW_MAX_DEG = 15.0


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Straighten a tilted bill using the median angle of near-horizontal lines
    found by a probabilistic Hough transform. Bills are full of horizontal
    table rules and text baselines, so this is a strong signal; if fewer than
    5 such lines agree, the image is left untouched.
    """
    H, W = gray.shape
    # Detect the skew angle on a downscaled copy — Hough is the costliest step
    # and angle estimation doesn't need full resolution.
    ds = 900 / max(H, W)
    small = cv2.resize(gray, (int(W * ds), int(H * ds))) if ds < 1 else gray

    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=small.shape[1] // 4, maxLineGap=20,
    )
    if lines is None:
        return gray

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
        if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) <= DESKEW_MAX_DEG
    ]
    if len(angles) < 5:
        return gray

    skew = float(np.median(angles))
    if abs(skew) < DESKEW_MIN_DEG:
        return gray

    matrix = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
    return cv2.warpAffine(
        gray, matrix, (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _preprocess(image: Image.Image) -> Image.Image:
    """
    Pipeline: bound size → grayscale → deskew → despeckle → binarise.

    Kept deliberately cheap so OCR finishes within the request timeout on
    Render's throttled free CPU. Binarisation is chosen per image: photos with
    shadows or glare (uneven illumination) get adaptive Gaussian thresholding,
    while evenly-lit scans/eBills keep global Otsu.
    """
    img = np.array(image.convert("RGB"))

    # 1. Downscale large phone photos first — the single biggest lever for OCR
    #    latency; a 12 MP photo would otherwise take minutes on the free CPU.
    h, w = img.shape[:2]
    if max(h, w) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 2. Upscale small scans / eBills so Tesseract has enough detail.
    if min(h, w) < TARGET_MIN_SIDE:
        scale = TARGET_MIN_SIDE / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = _deskew(gray)

    # Global Otsu threshold, no denoise. Both adaptive thresholding and median/
    # bilateral blur fragmented or merged the small digits on coloured-row bill
    # screenshots — especially after WhatsApp's JPEG compression — turning 467
    # into 316/600/None. Plain Otsu on the raw grayscale reads them reliably
    # (and is the fastest option, which also helps the free-tier timeout).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


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


def _safe_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_meter_readings(text: str) -> tuple[float, float] | None:
    """
    Extract the (previous, current) meter readings from the bill's meter table.
    TNB bills always show Dahulu (previous) and Semasa (current); the
    difference is the month's consumption and is immune to tariff-tier
    references like "600 kWh" appearing elsewhere on the page.
    """
    text_lower = text.lower()
    text_flat = re.sub(r'\s+', ' ', text_lower)

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
                if prev is not None and curr is not None and 50 <= curr - prev <= 5000:
                    return prev, curr
    return None


def _parse_consumption(text: str) -> float | None:
    """
    Find monthly consumption in kWh.
    Priority (highest → lowest):
      1. Meter reading difference (Semasa − Dahulu) — most reliable, immune to
         tariff-tier references like "600 kWh" appearing elsewhere in the bill.
      2. Explicit TNB labels ("Jumlah Penggunaan Anda", "Unit Used", etc.)
      3. Fallback: any number immediately before or after "kWh".
    Sanity range: 50–5000 kWh (typical Malaysian residential).
    """
    text_lower = text.lower()
    # Collapse all whitespace/newlines into single spaces so that table cells
    # split across lines (e.g. "467\nkWh") are matched as "467 kwh".
    text_flat = re.sub(r'\s+', ' ', text_lower)

    def _extract_first(pattern, source):
        for m in re.finditer(pattern, source):
            val = _safe_float(m.group(1))
            if val is not None and 50 <= val <= 5000:
                return val
        return None

    # ── 1. Meter reading difference: Semasa − Dahulu ─────────────────────────
    readings = _parse_meter_readings(text)
    if readings is not None:
        prev, curr = readings
        return curr - prev

    # ── 2. Explicit TNB bill labels ───────────────────────────────────────────
    labeled = [
        # "Penggunaan Anda … 467.00" — most reliable anchor: on real bills OCR
        # frequently mangles "Jumlah" (→ "Real"/"dumiah") and "kWh" (→ "ww"),
        # but "Penggunaan Anda" reads cleanly and is immediately followed by the
        # monthly total. Capture the first number after it.
        r"penggunaan\s+anda[^\d]{0,30}(\d[\d,]*(?:\.\d+)?)",
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
        # "Unit Used : 467" / "Penggunaan Unit : 467" (eBill layout)
        r"(?:unit\s+used|penggunaan\s+unit)[^\d]{0,40}(\d[\d,]*(?:\.\d+)?)\s*(?:kwh)?",
        # number with kWh unit shortly after a standalone "Semasa" header
        r"semasa[^\d]{0,30}(\d[\d,]*(?:\.\d+)?)\s*kwh",
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
        r"(\d[\d,]*(?:\.\d+)?)\s*kwh(?!\s*,)",   # number before kWh (not a rate line like \"316kWh,\")
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
        # NOTE: no bare "jumlah …" pattern — it matched "Jumlah Penggunaan Anda
        # … 467.00" (the usage row) and returned the kWh value as the amount.
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

# --oem 3 LSTM, --psm 6 (assume a uniform block of text). With the upscaled
# image plus the 'Penggunaan Anda' anchor, psm 6 reads the usage reliably AND
# is faster than psm 3's full page-segmentation — which was flakier here
# (e.g. grabbing 316 from the AFA line). preserve_interword_spaces keeps table
# columns adjacent for the label→value parsers.
TESS_CONFIG_PRIMARY  = "--oem 3 --psm 6 -c preserve_interword_spaces=1"


# ── Claude Vision (primary; falls back to Tesseract if unset or on error) ────
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"   # fast + cheap vision, ~1-2 sen/bill
CLAUDE_MAX_EDGE = 1568                 # Claude reads best ≤1568px; caps token cost

_CLAUDE_PROMPT = (
    "You are a precise data extractor for Malaysian TNB (Tenaga Nasional) "
    "electricity bills. From the bill image, return ONLY a JSON object (no prose) "
    "with these keys: "
    '{"state": <Malaysian state name e.g. "Kedah"/"Selangor"/"Penang", or null>, '
    '"monthly_kwh": <the month\'s total electricity usage in kWh as a number — '
    'the value next to "Jumlah Penggunaan Anda" or the meter "Penggunaan" column, '
    'or null>, '
    '"bill_amount_rm": <total amount payable in RM as a number, or null>, '
    '"tariff_category": <e.g. "Domestik", or null>, '
    '"confidence": <0.0 to 1.0>}. '
    "Use null if a value is not clearly visible; do NOT guess or hallucinate. "
    "monthly_kwh is the month's usage (typically 100-2000), never a tariff "
    "threshold like 300/600/900 from fine print. Return only the JSON."
)


def _extract_with_claude(image: Image.Image) -> dict:
    """Extract bill fields with Claude Vision. Raises on any API/parse error so
    the caller can fall back to Tesseract."""
    import anthropic

    rgb = image.convert("RGB")
    if max(rgb.size) > CLAUDE_MAX_EDGE:
        s = CLAUDE_MAX_EDGE / max(rgb.size)
        rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)))
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system=_CLAUDE_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": b64}},
            {"type": "text", "text": "Extract the fields as JSON."},
        ]}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])

    kwh = data.get("monthly_kwh")
    consumption = float(kwh) if isinstance(kwh, (int, float)) and 50 <= kwh <= 5000 else None
    amt = data.get("bill_amount_rm")
    amount = float(amt) if isinstance(amt, (int, float)) and 1 <= amt <= 100_000 else None
    state = (data.get("state") or None)
    tariff = (data.get("tariff_category") or None)
    conf = data.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) else (0.95 if consumption else 0.0)

    if consumption is not None:
        found = [f for f, v in [("consumption", consumption), ("bill amount", amount),
                                ("tariff", tariff), ("state", state)] if v is not None]
        message = f"Extracted via Claude Vision: {', '.join(found)}."
    else:
        message = "Claude could not read monthly consumption from the bill."

    return {
        "state": state, "consumption_kwh": consumption, "bill_amount_rm": amount,
        "tariff_category": tariff, "meter_previous_kwh": None, "meter_current_kwh": None,
        "confidence_score": round(confidence, 2), "raw_text": text,
        "success": consumption is not None, "message": message,
    }


def extract_bill_data(image: Image.Image) -> dict:
    """
    Extract TNB bill fields. Uses Claude Vision when ANTHROPIC_API_KEY is set
    (fast + accurate), falling back to local Tesseract OCR otherwise or on any
    Claude error. Returns the dict documented on _extract_with_tesseract.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            result = _extract_with_claude(image)
            if result["consumption_kwh"] is not None:
                return result
            logger.info("Claude returned no usage; falling back to Tesseract")
        except Exception:
            logger.exception("Claude Vision failed; falling back to Tesseract")
    return _extract_with_tesseract(image)


def _extract_with_tesseract(image: Image.Image) -> dict:
    """
    Local Tesseract OCR fallback. Returns a dict with keys:
        state, consumption_kwh, bill_amount_rm, tariff_category,
        meter_previous_kwh, meter_current_kwh, confidence_score,
        raw_text, success, message.
    """
    processed = _preprocess(image)
    # Single Tesseract pass with a hard time budget: a pathological image then
    # fails gracefully ("enter manually") instead of exceeding the request
    # timeout. psm 3 handles the bill's multi-column layout.
    try:
        raw_text = pytesseract.image_to_string(
            processed, lang="eng", config=TESS_CONFIG_PRIMARY, timeout=105,
        )
    except Exception:
        raw_text = ""  # timeout or OCR error → treat as unreadable, fall back

    consumption = _parse_consumption(raw_text)
    amount      = _parse_bill_amount(raw_text)
    tariff      = _parse_tariff_category(raw_text)
    state       = _parse_state(raw_text)
    readings    = _parse_meter_readings(raw_text)
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
        "state":              state,
        "consumption_kwh":    consumption,
        "bill_amount_rm":     amount,
        "tariff_category":    tariff,
        "meter_previous_kwh": readings[0] if readings else None,
        "meter_current_kwh":  readings[1] if readings else None,
        "confidence_score":   confidence,
        "raw_text":           raw_text,
        "success":            success,
        "message":            message,
    }
