import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import assessment, bill, report

load_dotenv()

app = FastAPI(
    title="SuriaSnap API",
    description="AI solar assessment backend for Malaysian homes",
    version="1.0.0",
)

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


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "SuriaSnap API"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
