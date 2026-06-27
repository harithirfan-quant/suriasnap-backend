"""
Thin client for the Meta WhatsApp Cloud API (Graph API), using plain `httpx`.

Covers exactly what the MVP needs:
    - send_text()       outbound text reply
    - send_document()   outbound PDF (by media id)
    - upload_media()    upload a file, get a media id back (free)
    - download_media()  fetch an inbound image/PDF the user sent us

Set WHATSAPP_DRY_RUN=true to skip real API calls and just log outbound
messages — handy for local testing without a Meta number or access token.
"""

import logging
import os

import httpx

logger = logging.getLogger("suriasnap.whatsapp")

API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")
GRAPH_BASE = f"https://graph.facebook.com/{API_VERSION}"

# These are read lazily inside each call so the module imports cleanly even when
# env vars aren't set (e.g. the GET webhook-verify path needs no access token).
def _access_token() -> str:
    token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not set")
    return token


def _phone_number_id() -> str:
    pid = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    if not pid:
        raise RuntimeError("WHATSAPP_PHONE_NUMBER_ID is not set")
    return pid


def _dry_run() -> bool:
    return os.getenv("WHATSAPP_DRY_RUN", "false").lower() in ("1", "true", "yes")


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_access_token()}"}


# ── outbound ──────────────────────────────────────────────────────────────────

def send_text(to: str, body: str) -> None:
    """Send a plain text WhatsApp message."""
    if _dry_run():
        logger.info("[DRY_RUN] → %s:\n%s", to, body)
        return

    url = f"{GRAPH_BASE}/{_phone_number_id()}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        logger.error("send_text failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()


def send_image(to: str, media_id: str, caption: str | None = None) -> None:
    """Send a previously-uploaded image by its media id."""
    if _dry_run():
        logger.info("[DRY_RUN] → %s: [image %s] %s", to, media_id, caption or "")
        return

    url = f"{GRAPH_BASE}/{_phone_number_id()}/messages"
    image = {"id": media_id}
    if caption:
        image["caption"] = caption
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": image,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        logger.error("send_image failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()


def send_list(to: str, body: str, button: str, rows: list[dict],
              section_title: str = "Questions", header: str | None = None) -> None:
    """Send an interactive list message (a tappable menu). Each row is
    {id, title, description}. Tapping a row sends its id back to the webhook."""
    if _dry_run():
        logger.info("[DRY_RUN] → %s: [list %s] %d rows", to, button, len(rows))
        return

    interactive = {
        "type": "list",
        "body": {"text": body},
        "action": {"button": button,
                   "sections": [{"title": section_title, "rows": rows}]},
    }
    if header:
        interactive["header"] = {"type": "text", "text": header}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    url = f"{GRAPH_BASE}/{_phone_number_id()}/messages"
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        logger.error("send_list failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()


def send_document(to: str, media_id: str, filename: str, caption: str | None = None) -> None:
    """Send a previously-uploaded document (PDF) by its media id."""
    if _dry_run():
        logger.info("[DRY_RUN] → %s: [document %s] %s", to, media_id, caption or "")
        return

    url = f"{GRAPH_BASE}/{_phone_number_id()}/messages"
    document = {"id": media_id, "filename": filename}
    if caption:
        document["caption"] = caption
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": document,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, headers=_auth_headers(), json=payload)
    if resp.status_code >= 400:
        logger.error("send_document failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()


def upload_media(file_bytes: bytes, filename: str, mime_type: str) -> str:
    """Upload a file to WhatsApp and return its media id (free)."""
    if _dry_run():
        logger.info("[DRY_RUN] upload_media %s (%s, %d bytes)", filename, mime_type, len(file_bytes))
        return "DRYRUN_MEDIA_ID"

    url = f"{GRAPH_BASE}/{_phone_number_id()}/media"
    files = {"file": (filename, file_bytes, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=_auth_headers(), data=data, files=files)
    if resp.status_code >= 400:
        logger.error("upload_media failed (%s): %s", resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()["id"]


# ── inbound media download ────────────────────────────────────────────────────

def download_media(media_id: str) -> tuple[bytes, str]:
    """
    Download an inbound media file in two steps (per Cloud API):
      1. GET /{media_id}      → temporary download URL + mime type
      2. GET <that url>       → the binary (still needs the Bearer token)
    Returns (raw_bytes, mime_type).
    """
    with httpx.Client(timeout=60) as client:
        meta = client.get(f"{GRAPH_BASE}/{media_id}", headers=_auth_headers())
        meta.raise_for_status()
        info = meta.json()
        media_url = info["url"]
        mime_type = info.get("mime_type", "application/octet-stream")

        binary = client.get(media_url, headers=_auth_headers())
        binary.raise_for_status()
        return binary.content, mime_type
