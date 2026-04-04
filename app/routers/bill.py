import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

from app.services import ocr_service

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE  = 10 * 1024 * 1024  # 10 MB


class BillScanResponse(BaseModel):
    success:          bool
    state:            str | None
    consumption_kwh:  float | None
    bill_amount_rm:   float | None
    tariff_category:  str | None
    confidence_score: float
    message:          str


@router.post("/scan-bill", response_model=BillScanResponse)
async def scan_bill(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG images are accepted.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit. Please upload a smaller image.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image. The file may be corrupted.")

    result = ocr_service.extract_bill_data(image)

    return BillScanResponse(
        success=result["success"],
        state=result["state"],
        consumption_kwh=result["consumption_kwh"],
        bill_amount_rm=result["bill_amount_rm"],
        tariff_category=result["tariff_category"],
        confidence_score=result["confidence_score"],
        message=result["message"],
    )
