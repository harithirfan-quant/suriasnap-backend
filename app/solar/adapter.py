"""
Solar calculation adapter.

The WhatsApp flow funnels through here so it NEVER duplicates the business
logic in `app.services.solar_calc`. This module only:
  - normalizes free-text / OCR state names to the canonical keys solar_calc
    expects (e.g. "KL" / "pulau pinang" → "Kuala Lumpur" / "Penang"),
  - applies a default roof orientation (bills don't contain it),
  - delegates the actual maths to `solar_calc.assess()`.
"""

import logging

from app.services import solar_calc

logger = logging.getLogger("suriasnap.solar")

# Bills don't reveal roof orientation; South is the optimal/most-common default
# in Malaysia and only swings generation by ~5–10% vs other directions.
DEFAULT_ORIENTATION = "South"

CANONICAL_STATES = list(solar_calc.STATES.keys())

# Common spellings / abbreviations users (or OCR) produce → canonical key
_STATE_ALIASES = {
    "kl": "Kuala Lumpur",
    "wp kuala lumpur": "Kuala Lumpur",
    "wilayah persekutuan kuala lumpur": "Kuala Lumpur",
    "pulau pinang": "Penang",
    "p. pinang": "Penang",
    "pg": "Penang",
    "malacca": "Melaka",
    "n9": "Negeri Sembilan",
    "n. sembilan": "Negeri Sembilan",
    "wp labuan": "Labuan",
    "wp putrajaya": "Putrajaya",
}


def normalize_state(raw: str | None) -> str | None:
    """Map a free-text state to a canonical solar_calc key, or None if unknown."""
    if not raw:
        return None
    key = raw.strip().lower()

    # 1. exact canonical match (case-insensitive)
    for s in CANONICAL_STATES:
        if s.lower() == key:
            return s
    # 2. known alias
    if key in _STATE_ALIASES:
        return _STATE_ALIASES[key]
    # 3. substring either direction (handles "i live in selangor")
    for s in CANONICAL_STATES:
        if s.lower() in key or key in s.lower():
            return s
    for alias, canonical in _STATE_ALIASES.items():
        if alias in key:
            return canonical
    return None


def run_assessment(
    state: str,
    monthly_kwh: float,
    roof_area_sqm: float,
    orientation: str = DEFAULT_ORIENTATION,
) -> dict:
    """
    Validate inputs and delegate to the existing solar engine.
    Returns the same dict shape as the website's /api/assess endpoint.
    """
    canonical = normalize_state(state)
    if canonical is None:
        raise ValueError(f"Unknown state: {state!r}")
    if orientation not in solar_calc.ORIENTATION_FACTORS:
        orientation = DEFAULT_ORIENTATION

    return solar_calc.assess(
        state=canonical,
        monthly_consumption_kwh=float(monthly_kwh),
        roof_area_sqm=float(roof_area_sqm),
        roof_orientation=orientation,
    )
