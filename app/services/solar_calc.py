import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "states.json") as f:
    STATES = json.load(f)

with open(DATA_DIR / "tariffs.json") as f:
    TARIFFS = json.load(f)

# Constants
PANEL_EFFICIENCY = 0.21
AREA_UTILIZATION = 0.70
PERFORMANCE_RATIO = 0.80
COST_PER_KWP_RM = 4000.0
GRID_EMISSION_FACTOR = 0.585  # kgCO2/kWh
DEGRADATION_RATE = 0.005      # 0.5% per year
PANEL_WATTAGE = 400           # W
DAYS_PER_MONTH = 30
PROJECTION_YEARS = 25

ORIENTATION_FACTORS = {
    "South": 1.00,
    "North": 0.95,
    "East":  0.90,
    "West":  0.90,
}


def _tnb_bill(consumption_kwh: float) -> float:
    """Calculate TNB monthly bill in RM for a given consumption."""
    tnb = TARIFFS["tnb"]

    if consumption_kwh <= 1500:
        energy = consumption_kwh * tnb["energy_charge"]["first_1500_kwh_rm"]
    else:
        energy = (1500 * tnb["energy_charge"]["first_1500_kwh_rm"] +
                  (consumption_kwh - 1500) * tnb["energy_charge"]["above_1500_kwh_rm"])

    capacity = consumption_kwh * tnb["capacity_charge_rm"]
    network  = consumption_kwh * tnb["network_charge_rm"]
    retail   = 0.0 if consumption_kwh < tnb["retail_waived_below_kwh"] else tnb["retail_charge_rm"]

    return energy + capacity + network + retail


def assess(
    state: str,
    monthly_consumption_kwh: float,
    roof_area_sqm: float,
    roof_orientation: str,
) -> dict:
    ghi               = STATES[state]["ghi"]
    orientation_factor = ORIENTATION_FACTORS[roof_orientation]
    export_rate       = TARIFFS["solar_atap"]["export_rate_rm"]

    # --- System sizing ---
    panel_area_sqm      = PANEL_WATTAGE / (1000 * PANEL_EFFICIENCY)   # ~1.905 m²
    usable_area         = roof_area_sqm * AREA_UTILIZATION
    num_panels          = int(usable_area / panel_area_sqm)
    system_kwp          = num_panels * PANEL_WATTAGE / 1000

    # --- Monthly generation ---
    daily_gen_kwh       = system_kwp * ghi * orientation_factor * PERFORMANCE_RATIO
    monthly_gen_kwh     = daily_gen_kwh * DAYS_PER_MONTH

    # --- Monthly savings ---
    old_bill            = _tnb_bill(monthly_consumption_kwh)
    net_consumption     = max(0.0, monthly_consumption_kwh - monthly_gen_kwh)
    new_bill            = _tnb_bill(net_consumption)
    export_kwh          = max(0.0, monthly_gen_kwh - monthly_consumption_kwh)
    export_revenue      = export_kwh * export_rate
    monthly_savings_rm  = (old_bill - new_bill) + export_revenue

    # --- CO2 ---
    annual_co2_offset_kg = monthly_gen_kwh * 12 * GRID_EMISSION_FACTOR

    # --- Financials ---
    system_cost_rm      = system_kwp * COST_PER_KWP_RM
    annual_savings      = monthly_savings_rm * 12
    payback_years       = system_cost_rm / annual_savings if annual_savings > 0 else float("inf")

    # 25-year ROI with 0.5%/year degradation
    total_savings_25yr  = sum(
        annual_savings * ((1 - DEGRADATION_RATE) ** year)
        for year in range(PROJECTION_YEARS)
    )
    roi_25_year_rm = total_savings_25yr - system_cost_rm

    return {
        "recommended_system_kwp":  round(system_kwp, 2),
        "num_panels_400w":         num_panels,
        "monthly_generation_kwh":  round(monthly_gen_kwh, 2),
        "monthly_savings_rm":      round(monthly_savings_rm, 2),
        "annual_co2_offset_kg":    round(annual_co2_offset_kg, 2),
        "system_cost_rm":          round(system_cost_rm, 2),
        "payback_years":           round(payback_years, 1),
        "roi_25_year_rm":          round(roi_25_year_rm, 2),
    }
