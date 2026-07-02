"""Shared utility helpers."""

_UTILITY = {
    "Sabah":   "SESB",
    "Sarawak": "SESCO",
}

_UTILITY_FULL = {
    "Sabah":   "SESB (Sabah Electricity Sdn Bhd)",
    "Sarawak": "SESCO (Sarawak Energy Berhad)",
}


def utility_name(state: str | None) -> str:
    """Return the electricity utility abbreviation for a Malaysian state."""
    return _UTILITY.get(state or "", "TNB")


def utility_full(state: str | None) -> str:
    """Return the full utility name for a Malaysian state."""
    return _UTILITY_FULL.get(state or "", "TNB (Tenaga Nasional Berhad)")
