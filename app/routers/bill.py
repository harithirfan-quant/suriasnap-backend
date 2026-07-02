import io

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from PIL import Image

from app.rate_limit import limiter
from app.services import ocr_service

router = APIRouter()


ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE  = 10 * 1024 * 1024  # 10 MB


class BillScanResponse(BaseModel):
    success:            bool
    state:              str | None
    consumption_kwh:    float | None
    bill_amount_rm:     float | None
    tariff_category:    str | None
    meter_previous_kwh: float | None
    meter_current_kwh:  float | None
    confidence_score:   float
    message:            str


def _pdf_to_image(contents: bytes) -> Image.Image:
    """Render the first page of a TNB eBill PDF at 300 DPI for OCR."""
    from pdf2image import convert_from_bytes

    pages = convert_from_bytes(contents, dpi=300, first_page=1, last_page=1)
    if not pages:
        raise ValueError("PDF has no pages")
    return pages[0]


@router.post("/scan-bill", response_model=BillScanResponse)
@limiter.limit("10/hour")
async def scan_bill(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG, or PDF bills are accepted.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit. Please upload a smaller file.",
        )

    try:
        if file.content_type == "application/pdf":
            image = _pdf_to_image(contents)
        else:
            image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open the file. It may be corrupted or password-protected.",
        )

    result = ocr_service.extract_bill_data(image)

    return BillScanResponse(
        success=result["success"],
        state=result["state"],
        consumption_kwh=result["consumption_kwh"],
        bill_amount_rm=result["bill_amount_rm"],
        tariff_category=result["tariff_category"],
        meter_previous_kwh=result["meter_previous_kwh"],
        meter_current_kwh=result["meter_current_kwh"],
        confidence_score=result["confidence_score"],
        message=result["message"],
    )
