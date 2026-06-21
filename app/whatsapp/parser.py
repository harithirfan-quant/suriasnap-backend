"""
Parse inbound WhatsApp Cloud API webhook payloads into a simple, flat shape
the orchestrator can consume. Also holds the webhook-verification helper.

We deliberately ignore status callbacks (delivered/read receipts) — those
arrive under `value.statuses` with no `messages` array.
"""

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("suriasnap.whatsapp.parser")


@dataclass
class InboundMessage:
    from_number: str
    wa_message_id: str
    msg_type: str                       # 'text' | 'image' | 'document' | 'other'
    text: str | None = None
    media_id: str | None = None
    media_mime: str | None = None
    media_filename: str | None = None
    profile_name: str | None = None
    extras: dict = field(default_factory=dict)


def verify_webhook(mode: str | None, token: str | None, challenge: str | None) -> str | None:
    """
    Meta calls GET /webhooks/whatsapp once when you subscribe. We must echo back
    `hub.challenge` iff the mode is 'subscribe' and the verify token matches
    WHATSAPP_VERIFY_TOKEN. Returns the challenge string on success, else None.
    """
    expected = os.getenv("WHATSAPP_VERIFY_TOKEN")
    if mode == "subscribe" and token and token == expected:
        return challenge
    logger.warning("Webhook verification failed (mode=%s)", mode)
    return None


def parse_inbound(payload: dict) -> list[InboundMessage]:
    """Flatten a webhook payload into zero or more InboundMessage objects."""
    messages: list[InboundMessage] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            # Map wa_id → profile name from the contacts block, if present
            names = {
                c.get("wa_id"): c.get("profile", {}).get("name")
                for c in value.get("contacts", [])
            }

            for m in value.get("messages", []):
                msg = _parse_single(m, names)
                if msg:
                    messages.append(msg)

    return messages


def _parse_single(m: dict, names: dict) -> InboundMessage | None:
    from_number = m.get("from")
    wamid = m.get("id")
    if not from_number or not wamid:
        return None

    mtype = m.get("type", "other")
    profile_name = names.get(from_number)

    base = dict(
        from_number=from_number,
        wa_message_id=wamid,
        profile_name=profile_name,
    )

    if mtype == "text":
        return InboundMessage(msg_type="text", text=m["text"]["body"], **base)

    if mtype == "image":
        img = m.get("image", {})
        return InboundMessage(
            msg_type="image",
            media_id=img.get("id"),
            media_mime=img.get("mime_type", "image/jpeg"),
            text=img.get("caption"),
            **base,
        )

    if mtype == "document":
        doc = m.get("document", {})
        return InboundMessage(
            msg_type="document",
            media_id=doc.get("id"),
            media_mime=doc.get("mime_type", "application/pdf"),
            media_filename=doc.get("filename", "bill.pdf"),
            text=doc.get("caption"),
            **base,
        )

    # Stickers, audio, location, button replies, etc. — handled generically
    return InboundMessage(msg_type="other", **base)
