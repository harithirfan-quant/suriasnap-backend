# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

# System packages:
#   tesseract-ocr       – OCR engine used by pytesseract
#   tesseract-ocr-eng   – English language pack
#   libgl1              – OpenCV requires libGL
#   libglib2.0-0        – OpenCV runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=/install /install /usr/local

# Copy application source
COPY app/ ./app/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Render sets PORT at runtime; default 8000 for local docker run
ENV PORT=8000 \
    TESSERACT_CMD=/usr/bin/tesseract \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

# Use shell form so $PORT is expanded at container start
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
