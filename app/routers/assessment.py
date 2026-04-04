from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services import solar_calc

router = APIRouter()

VALID_STATES = list(solar_calc.STATES.keys())
VALID_ORIENTATIONS = list(solar_calc.ORIENTATION_FACTORS.keys())


class AssessRequest(BaseModel):
    state: str = Field(..., examples=["Selangor"])
    monthly_consumption_kwh: float = Field(..., gt=0, examples=[350])
    roof_area_sqm: float = Field(..., gt=0, examples=[40])
    roof_orientation: str = Field(..., examples=["South"])


class AssessResponse(BaseModel):
    recommended_system_kwp: float
    num_panels_400w: int
    monthly_generation_kwh: float
    monthly_savings_rm: float
    annual_co2_offset_kg: float
    system_cost_rm: float
    payback_years: float
    roi_25_year_rm: float


@router.post("/assess", response_model=AssessResponse)
def assess(payload: AssessRequest):
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

    result = solar_calc.assess(
        state=payload.state,
        monthly_consumption_kwh=payload.monthly_consumption_kwh,
        roof_area_sqm=payload.roof_area_sqm,
        roof_orientation=payload.roof_orientation,
    )
    return result
