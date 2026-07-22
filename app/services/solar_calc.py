import json
import math
from pathlib import Path

from app.services.utils import utility_name

DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "states.json") as f:
    STATES = json.load(f)

with open(DATA_DIR / "tariffs.json") as f:
    TARIFFS = json.load(f)

# Constants
PANEL_EFFICIENCY = 0.21
AREA_UTILIZATION = 0.70
PERFORMANCE_RATIO = 0.80
COST_PER_KWP_RM = 7000.0
GRID_EMISSION_FACTOR = 0.758  # kgCO2/kWh (Suruhanjaya Tenaga, 2024 GEF)
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


def _sesb_bill(consumption_kwh: float) -> float:
    """Approximate SESB (Sabah) monthly bill using their published average
    domestic rate — see tariffs.json["sesb"]["_source"] for caveats."""
    return consumption_kwh * TARIFFS["sesb"]["flat_rate_rm"]


def _sesco_rate_for(consumption_kwh: float) -> float:
    """SESCO's domestic tariff is a stepped-average design: whichever
    bracket total monthly consumption falls into sets the rate for ALL
    units that month (not a marginal block system like TNB's)."""
    sesco = TARIFFS["sesco"]
    for bracket in sesco["brackets_rm"]:
        if consumption_kwh <= bracket["max_kwh"]:
            return bracket["rate_rm"]
    return sesco["above_800_rate_rm"]


def _sesco_bill(consumption_kwh: float) -> float:
    """Approximate SESCO (Sarawak) monthly bill — see
    tariffs.json["sesco"]["_source"] for caveats (temporary 25% discount
    running Apr-Dec 2026 is NOT applied here)."""
    return consumption_kwh * _sesco_rate_for(consumption_kwh)


def _bill_and_scheme(state: str) -> tuple:
    """
    Return (bill_fn, export_rate_fn, scheme_name) for the utility serving
    `state`. TNB territory (Peninsular + Labuan) is on Solar ATAP
    (GP/ST/No.60/2025) — a net-billing scheme, NOT a feed-in tariff: surplus
    credits the SAME billing period's bill at the domestic Energy Charge
    rate, capped at zeroing that bill (no cash beyond that — guideline
    §14.1(e)) and at the lower of the MAQ or that period's grid consumption
    (§14.1(b)-(d)); unused surplus is forfeited, not banked. Sabah (SESB)
    and Sarawak (SESCO) are NOT on Solar ATAP — they run their own separate
    net-metering schemes, which we approximate as 1:1 crediting at the same
    tariff used for consumption (no separate published export rate found
    for either).
    """
    utility = utility_name(state)

    if utility == "SESB":
        rate = TARIFFS["sesb"]["flat_rate_rm"]
        return _sesb_bill, (lambda consumption_kwh: rate), TARIFFS["sesb"]["scheme_name"]

    if utility == "SESCO":
        # Net-metered exports offset at the rate for the consumer's own
        # (pre-solar) usage level — the closest available proxy for "their
        # normal rate" absent a published export-specific rate.
        return _sesco_bill, (lambda consumption_kwh: _sesco_rate_for(consumption_kwh)), TARIFFS["sesco"]["scheme_name"]

    atap = TARIFFS["solar_atap"]

    def _atap_export_rate(consumption_kwh: float) -> float:
        return atap["export_rate_low_rm"] if consumption_kwh <= atap["threshold_kwh"] else atap["export_rate_high_rm"]

    return _tnb_bill, _atap_export_rate, atap["scheme_name"]


def assess(
    state: str,
    monthly_consumption_kwh: float,
    roof_area_sqm: float,
    roof_orientation: str,
) -> dict:
    ghi                 = STATES[state]["ghi"]
    orientation_factor  = ORIENTATION_FACTORS[roof_orientation]
    bill_fn, export_rate_fn, scheme_name = _bill_and_scheme(state)
    export_rate         = export_rate_fn(monthly_consumption_kwh)

    # --- System sizing ---
    panel_area_sqm      = PANEL_WATTAGE / (1000 * PANEL_EFFICIENCY)   # ~1.905 m²
    usable_area         = roof_area_sqm * AREA_UTILIZATION
    num_panels          = int(usable_area / panel_area_sqm)
    system_kwp          = num_panels * PANEL_WATTAGE / 1000

    # --- Monthly generation ---
    daily_gen_kwh       = system_kwp * ghi * orientation_factor * PERFORMANCE_RATIO
    monthly_gen_kwh     = daily_gen_kwh * DAYS_PER_MONTH

    # --- Monthly savings ---
    old_bill            = bill_fn(monthly_consumption_kwh)
    net_consumption     = max(0.0, monthly_consumption_kwh - monthly_gen_kwh)
    new_bill            = bill_fn(net_consumption)
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
        "export_rate_rm":          export_rate,
        "scheme_name":             scheme_name,
    }
