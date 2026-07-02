"""
Minimal SQLite persistence for the WhatsApp conversation flow.

Uses only the Python standard library (`sqlite3`) — no ORM, no DB server, no
extra dependency. A new connection is opened per operation, which is perfectly
fine at WhatsApp-MVP volume and avoids cross-thread connection issues when
FastAPI runs background tasks in its threadpool.

Tables (kept intentionally small):
    contacts          — one row per phone number + current conversation state
    messages          — append-only inbound/outbound log (also used for dedupe)
    bill_extractions  — raw file path + extracted JSON for each scanned bill
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.conversations import states

logger = logging.getLogger("suriasnap.store")

# Default lives at the repo root; override with SQLITE_DB_PATH. On Render's free
# tier this file is ephemeral (resets on deploy/sleep) — acceptable for an MVP.
DB_PATH = os.getenv("SQLITE_DB_PATH", "suriasnap.db")

# How long to keep message logs and bill-extraction records, per our Privacy
# Notice ("deleted on a short, rolling basis"). Override with RETENTION_DAYS.
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number  TEXT UNIQUE NOT NULL,
                name          TEXT,
                current_state TEXT NOT NULL DEFAULT 'NEW',
                pending_json  TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number    TEXT NOT NULL,
                direction       TEXT NOT NULL,          -- 'in' | 'out'
                message_type    TEXT,                   -- text | image | document | ...
                text            TEXT,
                media_path      TEXT,
                wa_message_id   TEXT,                   -- Meta's wamid, for dedupe
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bill_extractions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number   TEXT NOT NULL,
                raw_file_path  TEXT,
                extraction_json TEXT NOT NULL,
                confidence     REAL,
                created_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_wamid
                ON messages (wa_message_id);
            """
        )
    logger.info("SQLite ready at %s", DB_PATH)


# ── contacts ──────────────────────────────────────────────────────────────────

def get_contact(phone: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM contacts WHERE phone_number = ?", (phone,)
        ).fetchone()
        return dict(row) if row else None


def get_or_create_contact(phone: str, name: str | None = None) -> dict:
    existing = get_contact(phone)
    if existing:
        # Backfill the profile name if we learn it later
        if name and not existing.get("name"):
            with _conn() as conn:
                conn.execute(
                    "UPDATE contacts SET name = ?, updated_at = ? WHERE phone_number = ?",
                    (name, _now(), phone),
                )
            existing["name"] = name
        return existing

    ts = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO contacts (phone_number, name, current_state, pending_json, "
            "created_at, updated_at) VALUES (?, ?, ?, '{}', ?, ?)",
            (phone, name, states.NEW, ts, ts),
        )
    logger.info("New contact %s", phone)
    return get_contact(phone)


def set_state(phone: str, state: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE contacts SET current_state = ?, updated_at = ? WHERE phone_number = ?",
            (state, _now(), phone),
        )


def get_pending(phone: str) -> dict:
    contact = get_contact(phone)
    if not contact:
        return {}
    try:
        return json.loads(contact["pending_json"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def set_pending(phone: str, pending: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE contacts SET pending_json = ?, updated_at = ? WHERE phone_number = ?",
            (json.dumps(pending), _now(), phone),
        )


def merge_pending(phone: str, **fields) -> dict:
    """Update a few keys of the pending-inputs blob, keeping the rest."""
    pending = get_pending(phone)
    pending.update({k: v for k, v in fields.items() if v is not None})
    set_pending(phone, pending)
    return pending


# ── messages ──────────────────────────────────────────────────────────────────

def log_message(
    phone: str,
    direction: str,
    message_type: str | None = None,
    text: str | None = None,
    media_path: str | None = None,
    wa_message_id: str | None = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (phone_number, direction, message_type, text, "
            "media_path, wa_message_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (phone, direction, message_type, text, media_path, wa_message_id, _now()),
        )


def already_processed(wa_message_id: str | None) -> bool:
    """True if we've already logged this inbound wamid (Meta retries webhooks)."""
    if not wa_message_id:
        return False
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM messages WHERE wa_message_id = ? AND direction = 'in' LIMIT 1",
            (wa_message_id,),
        ).fetchone()
        return row is not None


# ── bill extractions ──────────────────────────────────────────────────────────

def save_extraction(
    phone: str, raw_file_path: str | None, extraction: dict, confidence: float
) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO bill_extractions (phone_number, raw_file_path, "
            "extraction_json, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (phone, raw_file_path, json.dumps(extraction), confidence, _now()),
        )


# ── retention ─────────────────────────────────────────────────────────────────

def purge_old_data(days: int = RETENTION_DAYS) -> None:
    """
    Delete message logs and bill-extraction records older than `days`. Bill
    media files are already removed right after OCR (see orchestrator), so
    this covers the remaining personal data: message text and extracted bill
    fields (usage, amount, state). Contacts/conversation state are kept —
    they hold no bill content, just the current step, so a returning user
    doesn't have to restart.

    Safe to call anytime; cheap at MVP volume (indexed on created_at-adjacent
    columns via row scan, fine for a few thousand rows).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as conn:
        msgs = conn.execute(
            "DELETE FROM messages WHERE created_at < ?", (cutoff,)
        ).rowcount
        extractions = conn.execute(
            "DELETE FROM bill_extractions WHERE created_at < ?", (cutoff,)
        ).rowcount
    if msgs or extractions:
        logger.info(
            "Purged %d message(s) and %d extraction(s) older than %d days",
            msgs, extractions, days,
        )


def purge_orphaned_media(media_dir: str, days: int = RETENTION_DAYS) -> None:
    """
    Safety net: the bill image/PDF is deleted right after OCR in the normal
    path, but a crash between save and delete could leave a file behind.
    Sweep anything older than `days` on startup so nothing lingers past our
    stated retention window.
    """
    root = Path(media_dir)
    if not root.exists():
        return
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    removed = 0
    for f in root.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            logger.warning("Could not remove orphaned media file %s", f)
    if removed:
        logger.info("Swept %d orphaned media file(s) older than %d days", removed, days)
