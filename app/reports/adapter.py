"""
Report adapter.

Reuses the EXISTING PDF generator (`app.services.report_generator.generate_report`)
so the WhatsApp PDF is byte-for-byte the same report the website produces.
This module just assembles the `assessment_data` dict the generator expects.
"""

from app.services.report_generator import generate_report


def build_assessment_data(
    state: str,
    monthly_kwh: float,
    roof_area_sqm: float,
    orientation: str,
    calc_result: dict,
) -> dict:
    """Merge calc outputs with the original inputs into the report's shape."""
    return {
        **calc_result,
        "state": state,
        "monthly_consumption_kwh": float(monthly_kwh),
        "roof_area_sqm": float(roof_area_sqm),
        "roof_orientation": orientation,
    }


def generate_pdf_bytes(
    state: str,
    monthly_kwh: float,
    roof_area_sqm: float,
    orientation: str,
    calc_result: dict,
) -> bytes:
    data = build_assessment_data(state, monthly_kwh, roof_area_sqm, orientation, calc_result)
    return generate_report(data)
