"""
Persistence for the WhatsApp conversation flow — SQLite locally/in tests,
Postgres in production when DATABASE_URL is set.

On Render's free tier, local disk (including a SQLite file) is wiped on every
deploy and on wake from sleep, which silently resets any in-progress WhatsApp
conversation. Setting DATABASE_URL (a free Neon/Supabase Postgres instance)
switches this module to Postgres with no other code changes required —
everything downstream of `_conn()` uses the same `? `placeholders and
dict-like rows either way. If DATABASE_URL is unset, behaviour is unchanged
from before: a local SQLite file, fine for local dev and the test suite.

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

# Default lives at the repo root; override with SQLITE_DB_PATH. Only used
# when DATABASE_URL is unset (local dev / tests).
DB_PATH = os.getenv("SQLITE_DB_PATH", "suriasnap.db")

# When set (a Postgres connection string, e.g. from Neon or Supabase), all
# persistence goes through Postgres instead of the ephemeral local SQLite
# file. This is what makes conversation state survive Render deploys/sleeps.
DATABASE_URL = os.getenv("DATABASE_URL")

# How long to keep message logs and bill-extraction records, per our Privacy
# Notice ("deleted on a short, rolling basis"). Override with RETENTION_DAYS.
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "30"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Conn:
    """
    Thin wrapper giving SQLite and Postgres the same call surface used
    throughout this module: `conn.execute(sql, params).fetchone()/.fetchall()
    /.rowcount`, plus a context manager that commits on success, rolls back
    on exception, and always closes the connection (SQLite's own `with conn:`
    only commits — it never closes, which is a real connection leak against
    Postgres's much lower connection limits on free tiers).

    Query strings are written once, using SQLite's `?` placeholder; for
    Postgres they're translated to `%s` before executing. None of our SQL
    contains a literal `?` outside of a placeholder, so this is a safe 1:1
    swap.
    """

    def __init__(self, raw, backend: str):
        self._raw = raw
        self.backend = backend

    def execute(self, sql: str, params: tuple = ()):
        if self.backend == "postgres":
            sql = sql.replace("?", "%s")
        cur = self._raw.cursor()
        cur.execute(sql, params)
        return cur

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._raw.commit()
        else:
            self._raw.rollback()
        self._raw.close()
        return False


def _conn() -> _Conn:
    if DATABASE_URL:
        import psycopg2
        import psycopg2.extras

        raw = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        return _Conn(raw, "postgres")

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(DB_PATH, timeout=10)
    raw.row_factory = sqlite3.Row
    return _Conn(raw, "sqlite")


_SQLITE_SCHEMA = """
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
    direction       TEXT NOT NULL,
    message_type    TEXT,
    text            TEXT,
    media_path      TEXT,
    wa_message_id   TEXT,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bill_extractions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_number    TEXT NOT NULL,
    raw_file_path   TEXT,
    extraction_json TEXT NOT NULL,
    confidence      REAL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_wamid ON messages (wa_message_id);
"""

# Same shape, Postgres syntax: SERIAL instead of INTEGER PRIMARY KEY AUTOINCREMENT.
_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id            SERIAL PRIMARY KEY,
    phone_number  TEXT UNIQUE NOT NULL,
    name          TEXT,
    current_state TEXT NOT NULL DEFAULT 'NEW',
    pending_json  TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    phone_number    TEXT NOT NULL,
    direction       TEXT NOT NULL,
    message_type    TEXT,
    text            TEXT,
    media_path      TEXT,
    wa_message_id   TEXT,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bill_extractions (
    id              SERIAL PRIMARY KEY,
    phone_number    TEXT NOT NULL,
    raw_file_path   TEXT,
    extraction_json TEXT NOT NULL,
    confidence      REAL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_wamid ON messages (wa_message_id);
"""


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as conn:
        if conn.backend == "postgres":
            cur = conn._raw.cursor()
            for stmt in _POSTGRES_SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        else:
            conn._raw.executescript(_SQLITE_SCHEMA)
    logger.info(
        "%s ready at %s",
        "Postgres" if DATABASE_URL else "SQLite",
        "(DATABASE_URL)" if DATABASE_URL else DB_PATH,
    )


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
