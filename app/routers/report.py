from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from app.services import solar_calc
from app.services.report_generator import generate_report

router = APIRouter()

VALID_STATES       = list(solar_calc.STATES.keys())
VALID_ORIENTATIONS = list(solar_calc.ORIENTATION_FACTORS.keys())


class ReportRequest(BaseModel):
    state:                    str   = Field(..., examples=["Selangor"])
    monthly_consumption_kwh:  float = Field(..., gt=0, examples=[350])
    roof_area_sqm:            float = Field(..., gt=0, examples=[40])
    roof_orientation:         str   = Field(..., examples=["South"])


@router.post(
    "/report",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}, "description": "Solar assessment PDF report"}},
)
def create_report(payload: ReportRequest):
    if payload.state not in VALID_STATES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid state. Valid options: {VALID_STATES}",
        )
    if payload.roof_orientation not in VALID_ORIENTATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid orientation. Valid options: {VALID_ORIENTATIONS}",
        )

    calc_result = solar_calc.assess(
        state=payload.state,
        monthly_consumption_kwh=payload.monthly_consumption_kwh,
        roof_area_sqm=payload.roof_area_sqm,
        roof_orientation=payload.roof_orientation,
    )

    assessment_data = {
        **calc_result,
        "state":                   payload.state,
        "monthly_consumption_kwh": payload.monthly_consumption_kwh,
        "roof_area_sqm":           payload.roof_area_sqm,
        "roof_orientation":        payload.roof_orientation,
    }

    pdf_bytes = generate_report(assessment_data)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=suriasnap-report.pdf"},
    )
