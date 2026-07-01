import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.reports import design_preview

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
TEAL       = colors.HexColor("#0D7377")
TEAL_LIGHT = colors.HexColor("#14BDBD")
TEAL_BG    = colors.HexColor("#E8F7F7")
DARK       = colors.HexColor("#1A1A2E")
GREY       = colors.HexColor("#6B7280")
WHITE      = colors.white

CO2_PER_TREE_KG = 22  # kg CO2 absorbed per tree per year

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=TEAL_LIGHT,
            alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=TEAL,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK,
            leading=15,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Helvetica",
            fontSize=8,
            textColor=GREY,
            leading=12,
        ),
        "footer": ParagraphStyle(
            "Footer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=GREY,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            fontName="Helvetica",
            fontSize=9,
            textColor=GREY,
            alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=TEAL,
            alignment=TA_CENTER,
        ),
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(styles: dict, state: str) -> list:
    """Teal banner with title + date."""
    title = Paragraph("SuriaSnap Solar Assessment Report", styles["title"])
    sub   = Paragraph(
        f"Prepared for a home in <b>{state}</b> &nbsp;|&nbsp; {date.today().strftime('%d %B %Y')}",
        styles["subtitle"],
    )
    banner = Table(
        [[title], [sub]],
        colWidths=[17 * cm],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), TEAL),
            ("TOPPADDING",    (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("ROUNDEDCORNERS", [6]),
        ]),
    )
    return [banner, Spacer(1, 0.4 * cm)]


def _user_details(styles: dict, inputs: dict) -> list:
    """State, roof area, orientation, monthly consumption."""
    rows = [
        ["State / Negeri",            inputs["state"]],
        ["Roof Area",                 f"{inputs['roof_area_sqm']} m²"],
        ["Roof Orientation",          inputs["roof_orientation"]],
        ["Monthly Consumption",       f"{inputs['monthly_consumption_kwh']:.0f} kWh"],
    ]
    tbl = Table(rows, colWidths=[7 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (0, 0), (0, -1), DARK),
        ("TEXTCOLOR",     (1, 0), (1, -1), GREY),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, TEAL_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [
        Paragraph("Your Property Details", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        tbl,
    ]


def _savings_callout(styles: dict, data: dict) -> list:
    """Eye-catching 'install now' savings banner placed right after the
    property details — leads with the yearly saving and the cost of waiting."""
    monthly = data["monthly_savings_rm"]
    annual  = monthly * 12
    roi25   = data["roi_25_year_rm"]
    payback = data["payback_years"]

    lead = ParagraphStyle("clLead", fontName="Helvetica-Bold", fontSize=11,
                          textColor=TEAL_LIGHT, alignment=TA_CENTER, spaceAfter=6)
    big  = ParagraphStyle("clBig", fontName="Helvetica-Bold", fontSize=24,
                          textColor=WHITE, alignment=TA_CENTER, leading=28, spaceAfter=6)
    sub  = ParagraphStyle("clSub", fontName="Helvetica", fontSize=10,
                          textColor=WHITE, alignment=TA_CENTER, leading=14)

    # One cell containing stacked paragraphs — avoids row-height overlap.
    box = Table(
        [[[
            Paragraph("IF YOU GO SOLAR NOW, YOU COULD SAVE", lead),
            Paragraph(f"RM {annual:,.0f} <font size=13>per year</font>", big),
            Paragraph(
                f"≈ <b>RM {roi25:,.0f}</b> net profit over 25 years — the system pays for "
                f"itself in about <b>{payback} years</b>, then it is essentially free electricity.",
                sub),
        ]]],
        colWidths=[17 * cm],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
            ("TOPPADDING",    (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ("LEFTPADDING",   (0, 0), (-1, -1), 18),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
            ("ROUNDEDCORNERS", [8]),
        ]),
    )
    why = Paragraph(
        f"<b>Why?</b> Right now about <b>RM {monthly:,.0f}/month</b> leaves your pocket for TNB. "
        "Solar offsets your usage first and exports the surplus for Solar ATAP credits, so most of that "
        "money stays with you instead. Panels run for 25+ years — every month you wait is a saving "
        "you don't get back.",
        styles["small"],
    )
    return [box, Spacer(1, 0.15 * cm), why, Spacer(1, 0.35 * cm)]


def _system_recommendation(styles: dict, data: dict) -> list:
    """KPI tiles: system size and panel count."""
    kpi_data = [
        [
            Paragraph("Recommended System", styles["kpi_label"]),
            Paragraph("Number of Panels", styles["kpi_label"]),
            Paragraph("Monthly Generation", styles["kpi_label"]),
        ],
        [
            Paragraph(f"{data['recommended_system_kwp']} kWp", styles["kpi_value"]),
            Paragraph(f"{data['num_panels_400w']} × 400W", styles["kpi_value"]),
            Paragraph(f"{data['monthly_generation_kwh']:.0f} kWh", styles["kpi_value"]),
        ],
    ]
    tbl = Table(kpi_data, colWidths=[5.67 * cm, 5.67 * cm, 5.67 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), TEAL_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LINEBELOW",     (0, 0), (-1, 0),  0.5, TEAL_LIGHT),
        ("GRID",          (0, 0), (-1, -1), 0.5, WHITE),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [
        Paragraph("System Recommendation", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        tbl,
    ]


def _design_preview(styles: dict, data: dict) -> list:
    """Representative Arka-360-style rooftop layout image + honest note (best-effort)."""
    try:
        kwp = data["recommended_system_kwp"]
        specific_yield = round(data["monthly_generation_kwh"] * 12 / kwp)
        png = design_preview.render_design_png(
            data["num_panels_400w"], data["roof_orientation"], kwp, specific_yield,
        )
        img = RLImage(io.BytesIO(png), width=16 * cm, height=10 * cm)
    except Exception:
        return []  # never let a rendering hiccup break the whole report

    note = Paragraph(
        "Representative layout — panel count and orientation reflect your inputs. "
        "A SEDA-registered installer performs a site survey and produces the certified "
        "engineering design (e.g. with Arka 360).",
        styles["small"],
    )
    return [
        Paragraph("Professional Design Preview", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        img,
        Spacer(1, 0.2 * cm),
        note,
    ]


def _financial_summary(styles: dict, data: dict) -> list:
    """Two hero numbers (annual savings + 25-year net profit) over a clean
    detail table — leads with impact, then backs it up with the breakdown."""
    monthly = data["monthly_savings_rm"]
    annual  = monthly * 12
    cost    = data["system_cost_rm"]
    payback = data["payback_years"]
    roi25   = data["roi_25_year_rm"]
    total25 = roi25 + cost   # gross lifetime savings (net profit + the cost it repaid)

    # --- hero tiles ---
    tile_label = ParagraphStyle("fH", fontName="Helvetica", fontSize=9,
                                textColor=TEAL_LIGHT, alignment=TA_CENTER, spaceAfter=3)
    tile_val   = ParagraphStyle("fV", fontName="Helvetica-Bold", fontSize=19,
                                textColor=WHITE, alignment=TA_CENTER)
    hero = Table(
        [
            [Paragraph("ANNUAL SAVINGS", tile_label), Paragraph("25-YEAR NET PROFIT", tile_label)],
            [Paragraph(f"RM {annual:,.0f}", tile_val), Paragraph(f"RM {roi25:,.0f}", tile_val)],
        ],
        colWidths=[8.5 * cm, 8.5 * cm],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
            ("GRID",          (0, 0), (-1, -1), 1.2, WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 11),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
            ("ROUNDEDCORNERS", [6]),
        ]),
    )

    # --- detail breakdown ---
    cell = ParagraphStyle("dC", fontName="Helvetica", fontSize=10, textColor=DARK)
    val  = ParagraphStyle("dV", fontName="Helvetica-Bold", fontSize=10,
                          textColor=TEAL, alignment=TA_RIGHT)
    detail_rows = [
        [Paragraph("Monthly savings", cell),            Paragraph(f"RM {monthly:,.2f}", val)],
        [Paragraph("Estimated system cost", cell),      Paragraph(f"RM {cost:,.0f}", val)],
        [Paragraph("Payback period", cell),             Paragraph(f"{payback} years", val)],
        [Paragraph("Total saved over 25 years", cell),  Paragraph(f"RM {total25:,.0f}", val)],
    ]
    detail = Table(detail_rows, colWidths=[10 * cm, 7 * cm], style=TableStyle([
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [WHITE, TEAL_BG]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.3, TEAL_LIGHT),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [
        Paragraph("Financial Summary", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        hero,
        Spacer(1, 0.25 * cm),
        detail,
    ]


def _environmental_impact(styles: dict, data: dict) -> list:
    """CO2 offset and tree-equivalent."""
    annual_co2 = data["annual_co2_offset_kg"]
    trees       = int(annual_co2 / CO2_PER_TREE_KG)

    # Cells are Paragraphs so the CO<sub>2</sub> subscript renders properly —
    # ReportLab's Helvetica has no ₂ glyph and would draw a tofu box otherwise.
    lbl = ParagraphStyle("envL", fontName="Helvetica-Bold", fontSize=10, textColor=DARK)
    v   = ParagraphStyle("envV", fontName="Helvetica", fontSize=10, textColor=TEAL)
    rows = [
        [Paragraph("Annual CO<sub>2</sub> Offset", lbl),
         Paragraph(f"{annual_co2:,.1f} kg CO<sub>2</sub>", v)],
        [Paragraph("Tree Equivalent", lbl),
         Paragraph(f"≈ {trees:,} trees planted per year", v)],
    ]
    tbl = Table(rows, colWidths=[7 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [TEAL_BG, WHITE]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [
        Paragraph("Environmental Impact", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        tbl,
        Spacer(1, 0.2 * cm),
        Paragraph(
            "CO<sub>2</sub> offset calculated using Malaysia's grid emission factor of "
            "0.758 kgCO<sub>2</sub>/kWh (Suruhanjaya Tenaga, 2024 GEF). Tree equivalence "
            "based on 22 kg CO<sub>2</sub> absorbed per tree per year.",
            styles["small"],
        ),
    ]


def _solar_atap_explainer(styles: dict) -> list:
    """Brief Solar ATAP scheme description and export rates."""
    body = (
        "The <b>Solar ATAP</b> (Skim Suria Atap) programme by SEDA Malaysia enables residential "
        "solar owners to sell surplus electricity back to TNB. Export rates: "
        "<b>RM 0.27 per kWh</b> (≤1,500 kWh/month consumption) or <b>RM 0.37 per kWh</b> "
        "(>1,500 kWh/month consumption). Generation offsets your consumption first; "
        "any excess is exported to the grid for credit on your next TNB bill. "
        "10-year contract, no quota limits. System capacity is capped at your contracted demand or 12 kWp, whichever is lower."
    )
    return [
        Paragraph("About Solar ATAP", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        Paragraph(body, styles["body"]),
    ]


def _next_steps(styles: dict) -> list:
    steps = [
        "1. Contact a <b>SEDA-registered installer</b> for a site survey and detailed quotation.",
        "2. Submit your Solar ATAP application via the <b>SEDA Malaysia portal</b>.",
        "3. After approval, your installer connects the system and TNB installs a bidirectional meter.",
        "4. Start generating clean energy and earning Solar ATAP credits on your monthly bill.",
    ]
    items = [Paragraph(s, styles["body"]) for s in steps]
    note  = Paragraph(
        "Verify installer registration at <b>seda.gov.my</b> before signing any contract.",
        styles["small"],
    )
    return (
        [Paragraph("Next Steps", styles["section"]),
         HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6)]
        + items
        + [Spacer(1, 0.2 * cm), note]
    )


def _footer(styles: dict) -> list:
    return [
        Spacer(1, 0.6 * cm),
        HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Generated by SuriaSnap &nbsp;|&nbsp; Data: Global Solar Atlas &amp; SEDA Malaysia",
            styles["footer"],
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(assessment_data: dict) -> bytes:
    """
    Build a PDF report from assessment_data and return raw PDF bytes.

    assessment_data must contain the keys returned by solar_calc.assess()
    plus the original inputs: state, monthly_consumption_kwh,
    roof_area_sqm, roof_orientation.
    """
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2 * cm,
    )
    styles = _build_styles()

    inputs = {
        "state":                   assessment_data["state"],
        "roof_area_sqm":           assessment_data["roof_area_sqm"],
        "roof_orientation":        assessment_data["roof_orientation"],
        "monthly_consumption_kwh": assessment_data["monthly_consumption_kwh"],
    }

    story = []
    story += _header(styles, inputs["state"])
    story += _user_details(styles, inputs)
    story += _savings_callout(styles, assessment_data)
    story += _system_recommendation(styles, assessment_data)
    story += _design_preview(styles, assessment_data)
    story += _financial_summary(styles, assessment_data)
    story += _environmental_impact(styles, assessment_data)
    story += _solar_atap_explainer(styles)
    story += _next_steps(styles)
    story += _footer(styles)

    doc.build(story)
    return buf.getvalue()
