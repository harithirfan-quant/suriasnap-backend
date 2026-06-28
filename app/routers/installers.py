from fastapi import APIRouter, Query

from app.services import installers as svc

router = APIRouter()


@router.get("/installers")
def get_installers(state: str = Query(..., description="Malaysian state, e.g. 'Kedah'")):
    """SEDA-registered installers for a state, with nearest-state fallback."""
    return svc.find_installers(state)
