"""
WhatsApp Cloud API webhook router.

    GET  /webhooks/whatsapp   → Meta verification handshake
    POST /webhooks/whatsapp   → inbound messages (acked fast, processed in bg)

The POST handler does the bare minimum synchronously (parse + schedule) and
returns 200 immediately, because Meta expects a quick ack and will retry /
eventually disable a webhook that's slow or errors. The heavy work (media
download + OCR + reply) runs in a FastAPI BackgroundTask.
"""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.conversations import orchestrator
from app.whatsapp import parser

logger = logging.getLogger("suriasnap.webhook")

router = APIRouter()


@router.get("/webhooks/whatsapp")
async def verify(request: Request):
    """Meta sends hub.mode / hub.verify_token / hub.challenge as query params."""
    params = request.query_params
    challenge = parser.verify_webhook(
        params.get("hub.mode"),
        params.get("hub.verify_token"),
        params.get("hub.challenge"),
    )
    if challenge is not None:
        # Must be returned as plain text (Meta compares the raw body).
        return PlainTextResponse(content=challenge)
    return PlainTextResponse(content="Verification failed", status_code=403)


@router.post("/webhooks/whatsapp")
async def receive(request: Request, background: BackgroundTasks):
    """Ack immediately; process each message in the background."""
    raw_body = await request.body()

    if not parser.verify_signature(raw_body, request.headers.get("X-Hub-Signature-256")):
        logger.warning("Rejected webhook POST with invalid signature")
        # Still 200 — a 4xx here doesn't help (Meta won't retry a forged
        # request either way) and avoids leaking verification behaviour.
        return JSONResponse({"status": "ignored"}, status_code=200)

    try:
        payload = json.loads(raw_body)
    except Exception:
        logger.warning("Webhook POST with non-JSON body")
        return JSONResponse({"status": "ignored"}, status_code=200)

    try:
        messages = parser.parse_inbound(payload)
        for msg in messages:
            background.add_task(orchestrator.handle_inbound, msg)
        if messages:
            logger.info("Scheduled %d inbound message(s)", len(messages))
    except Exception:
        # Always 200 so Meta keeps the subscription healthy; we log for debugging.
        logger.exception("Failed to parse/schedule webhook payload")

    return JSONResponse({"status": "received"}, status_code=200)
