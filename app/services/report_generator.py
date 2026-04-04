import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

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


def _financial_summary(styles: dict, data: dict) -> list:
    """Table with monthly/annual savings, payback, 25-year ROI."""
    header_style = ParagraphStyle(
        "TH", fontName="Helvetica-Bold", fontSize=10,
        textColor=WHITE, alignment=TA_CENTER,
    )
    cell_style = ParagraphStyle(
        "TD", fontName="Helvetica", fontSize=10,
        textColor=DARK, alignment=TA_LEFT,
    )
    val_style = ParagraphStyle(
        "TDV", fontName="Helvetica-Bold", fontSize=10,
        textColor=TEAL, alignment=TA_RIGHT,
    )

    rows = [
        [Paragraph("Financial Metric", header_style), Paragraph("Value", header_style)],
        ["Monthly Savings",       f"RM {data['monthly_savings_rm']:,.2f}"],
        ["Annual Savings",        f"RM {data['monthly_savings_rm'] * 12:,.2f}"],
        ["System Cost",           f"RM {data['system_cost_rm']:,.2f}"],
        ["Payback Period",        f"{data['payback_years']} years"],
        ["25-Year ROI",           f"RM {data['roi_25_year_rm']:,.2f}"],
    ]

    styled_rows = [rows[0]]
    for label, value in rows[1:]:
        styled_rows.append([
            Paragraph(label, cell_style),
            Paragraph(value, val_style),
        ])

    tbl = Table(styled_rows, colWidths=[10 * cm, 7 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  TEAL),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, TEAL_BG]),
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
        tbl,
    ]


def _environmental_impact(styles: dict, data: dict) -> list:
    """CO2 offset and tree-equivalent."""
    annual_co2 = data["annual_co2_offset_kg"]
    trees       = int(annual_co2 / CO2_PER_TREE_KG)

    rows = [
        ["Annual CO₂ Offset",     f"{annual_co2:,.1f} kg CO₂"],
        ["Tree Equivalent",       f"≈ {trees:,} trees planted per year"],
    ]
    tbl = Table(rows, colWidths=[7 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (0, 0), (0, -1), DARK),
        ("TEXTCOLOR",     (1, 0), (1, -1), TEAL),
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
            "CO₂ offset calculated using Malaysia's grid emission factor of 0.585 kgCO₂/kWh "
            "(Suruhanjaya Tenaga). Tree equivalence based on 22 kg CO₂ absorbed per tree per year.",
            styles["small"],
        ),
    ]


def _solar_atap_explainer(styles: dict) -> list:
    """Brief Solar ATAP scheme description and export rates."""
    body = (
        "The <b>Solar ATAP</b> (Skim Suria Atap) programme by SEDA Malaysia enables residential "
        "solar owners to sell surplus electricity back to TNB at the Net Energy Metering (NEM) "
        "export rate of <b>RM 0.2703 per kWh</b>. Generation offsets your consumption first; "
        "any excess is exported to the grid for credit on your next TNB bill. "
        "System capacity is capped at your contracted demand or 12 kWp, whichever is lower."
    )
    return [
        Paragraph("About Solar ATAP (NEM Scheme)", styles["section"]),
        HRFlowable(width="100%", thickness=1, color=TEAL_LIGHT, spaceAfter=6),
        Paragraph(body, styles["body"]),
    ]


def _next_steps(styles: dict) -> list:
    steps = [
        "1. Contact a <b>SEDA-registered installer</b> for a site survey and detailed quotation.",
        "2. Submit your Solar ATAP application via the <b>SEDA Malaysia portal</b>.",
        "3. After approval, your installer connects the system and TNB installs a bidirectional meter.",
        "4. Start generating clean energy and earning NEM credits on your monthly bill.",
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
    story += _system_recommendation(styles, assessment_data)
    story += _financial_summary(styles, assessment_data)
    story += _environmental_impact(styles, assessment_data)
    story += _solar_atap_explainer(styles)
    story += _next_steps(styles)
    story += _footer(styles)

    doc.build(story)
    return buf.getvalue()
