import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.rate_limit import limiter
from app.routers import assessment, bill, installers, report, whatsapp
from app.conversations import store

load_dotenv()

# Render sets RENDER=true in every service's environment — use it to tell
# production apart from local dev without any extra configuration.
IS_PRODUCTION = bool(os.getenv("RENDER"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Create the tables used by the WhatsApp conversation flow (SQLite or
    # Postgres, depending on DATABASE_URL).
    store.init_db()

    # Enforce our data-retention promise (see LegalPage / Privacy Notice) —
    # old message logs / bill extractions get purged, and any bill media
    # file that somehow survived past OCR gets swept. Render's free tier
    # sleeps and wakes often, so startup fires regularly enough to matter.
    store.purge_old_data()
    store.purge_orphaned_media(os.getenv("MEDIA_DIR", "media"))

    yield


app = FastAPI(
    title="SuriaSnap API",
    description="AI solar assessment backend for Malaysian homes",
    version="1.0.0",
    lifespan=_lifespan,
    # Swagger/ReDoc/openapi.json hand any visitor an interactive map of the
    # full API surface — useful locally, needless exposure in production.
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)


@app.middleware("http")
async def _security_headers(request, call_next):
    """Baseline hardening headers on every response. The Vercel-hosted
    frontend already gets these from Vercel's edge; the API did not."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if IS_PRODUCTION:
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    return response

# Rate limiting — CORS only stops browsers, not curl/scripts, and the bill
# scan endpoint calls paid Claude Vision per request. Keyed by client IP;
# fine at MVP scale, no Redis needed (in-memory, per-process).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

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
