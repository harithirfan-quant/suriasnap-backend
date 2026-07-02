import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import assessment, bill, installers, report, whatsapp
from app.conversations import store

load_dotenv()

app = FastAPI(
    title="SuriaSnap API",
    description="AI solar assessment backend for Malaysian homes",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    # Create the SQLite tables used by the WhatsApp conversation flow.
    store.init_db()

    # Enforce our data-retention promise (see LegalPage / Privacy Notice) —
    # old message logs / bill extractions get purged, and any bill media
    # file that somehow survived past OCR gets swept. Render's free tier
    # sleeps and wakes often, so startup fires regularly enough to matter.
    store.purge_old_data()
    store.purge_orphaned_media(os.getenv("MEDIA_DIR", "media"))

_default_origins = "http://localhost:5173,http://localhost:3000"
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router, prefix="/api", tags=["Assessment"])
app.include_router(bill.router,        prefix="/api", tags=["Bill Scan"])
app.include_router(report.router,      prefix="/api", tags=["Report"])
app.include_router(installers.router,  prefix="/api", tags=["Installers"])
# WhatsApp webhook is intentionally unprefixed (Meta points at /webhooks/whatsapp)
app.include_router(whatsapp.router,    tags=["WhatsApp"])


@app.get("/", tags=["Health"])
def root():
    # `ocr` marker lets us confirm which build is live after a deploy.
    return {"status": "ok", "service": "SuriaSnap API", "ocr": "solar-atap-v15"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
