"""
Offline end-to-end test of the WhatsApp conversation state machine.

Stubs the heavy native deps (cv2/numpy/pytesseract/PIL/httpx) so it runs with
just the stdlib, monkeypatches media download + OCR extraction with canned
data, and drives the orchestrator through two realistic conversations:

  Run A — OCR reads usage + state  → only roof is asked
  Run B — OCR misses usage         → asks kWh, then state, then roof

Outbound replies are captured and asserted on. No real WhatsApp calls happen.
"""
import os
import sys
import tempfile
import types

# ── stub native deps BEFORE importing app code ───────────────────────────────
for name in ("cv2", "pytesseract", "numpy"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["numpy"].ndarray = object
sys.modules["pytesseract"].pytesseract = types.SimpleNamespace(tesseract_cmd="")
pil = types.ModuleType("PIL")
pil_image = types.ModuleType("PIL.Image")
pil_image.Image = object        # used as the type annotation Image.Image
pil_image.open = lambda *a, **k: object()
pil_image.BICUBIC = 3
pil.Image = pil_image
pil.ImageDraw = types.ModuleType("PIL.ImageDraw")
pil.ImageFont = types.ModuleType("PIL.ImageFont")
sys.modules["PIL"] = pil
sys.modules["PIL.Image"] = pil_image
sys.modules["PIL.ImageDraw"] = pil.ImageDraw
sys.modules["PIL.ImageFont"] = pil.ImageFont
sys.modules["httpx"] = types.ModuleType("httpx")  # never called in DRY_RUN

# ── isolated env ─────────────────────────────────────────────────────────────
_tmp = tempfile.mkdtemp()
os.environ["WHATSAPP_DRY_RUN"] = "true"
os.environ["SQLITE_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["MEDIA_DIR"] = os.path.join(_tmp, "media")

from app.conversations import orchestrator, states, store
from app.extraction import bill_extractor
from app.reports import adapter as reports
from app.reports import design_preview
from app.services import assistant
from app.whatsapp import client as wa
from app.whatsapp.parser import InboundMessage

store.init_db()

# ── capture outbound + fake the external calls ───────────────────────────────
SENT: list[str] = []
wa.send_text = lambda to, body: SENT.append(body)
wa.send_document = lambda to, mid, fn, caption=None: SENT.append(f"[DOC {fn}] {caption}")
wa.send_image = lambda to, mid, caption=None: SENT.append(f"[IMG] {caption}")
wa.send_list = lambda to, body, button, rows, section_title="", header=None: SENT.append(f"[LIST {len(rows)} rows]")
wa.upload_media = lambda b, fn, mime: "FAKE_MEDIA_ID"
wa.download_media = lambda mid: (b"fake-bytes", "image/jpeg")
reports.generate_pdf_bytes = lambda *a, **k: b"%PDF-fake"
design_preview.render_design_png = lambda *a, **k: b"PNG"
assistant.answer_question = lambda q: f"[ANSWER:{q}]"   # no real Claude call in tests

_n = 0
def send(phone, *, text=None, media=False, interactive_id=None):
    """Simulate one inbound WhatsApp message hitting the webhook."""
    global _n
    _n += 1
    msg = InboundMessage(
        from_number=phone,
        wa_message_id=f"wamid-{_n}",
        msg_type="interactive" if interactive_id else ("image" if media else "text"),
        text=interactive_id or text,
        media_id="MID" if media else None,
        media_mime="image/jpeg" if media else None,
        profile_name="Test User",
    )
    orchestrator.handle_inbound(msg)

FAILURES = []
def check(label, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILURES.append(label)

def state_of(phone):
    return store.get_contact(phone)["current_state"]

# ── Run A: OCR gives kWh + state → only roof asked ───────────────────────────
print("\n=== Run A: OCR reads usage + state ===")
bill_extractor.extract_bill = lambda path: {
    "state": "Selangor", "total_kwh": 467, "total_amount_rm": 207.49,
    "confidence": 0.9, "notes": [],
}
A = "60123000001"

send(A, text="hi")
check("A: greeting → WAITING_FOR_BILL", state_of(A) == states.WAITING_FOR_BILL)
check("A: intro mentions manual option", "manual" in SENT[-1].lower())

send(A, media=True)
check("A: bill read, state+kwh known → asks ROOF", state_of(A) == states.WAITING_FOR_ROOF)

send(A, text="40")
check("A: roof given → DONE", state_of(A) == states.DONE)
summary = next((m for m in SENT if "Solar Estimate" in m), "")
check("A: summary shows state", "Selangor" in summary)
check("A: summary shows usage", "467 kWh" in summary)
check("A: summary shows system size", "kWp" in summary)
check("A: summary shows payback", "Payback" in summary or "payback" in summary)
check("A: PDF document sent", any(m.startswith("[DOC") for m in SENT))
check("A: design image sent", any(m.startswith("[IMG]") for m in SENT))
check("A: summary mentions design preview", "design preview" in summary.lower())
check("A: summary lists a real Selangor installer", "Solarvest" in summary)
check("A: Selangor is a direct match (no fallback wording)",
      "No installers are based in Selangor" not in summary)

# ── Run B: OCR misses usage → asks kWh, state, roof ──────────────────────────
print("\n=== Run B: OCR misses usage ===")
SENT.clear()
bill_extractor.extract_bill = lambda path: {
    "state": "", "total_kwh": 0, "total_amount_rm": 0,
    "confidence": 0.3, "notes": ["low"],
}
B = "60123000002"

send(B, text="hello")
check("B: greeting → WAITING_FOR_BILL", state_of(B) == states.WAITING_FOR_BILL)

send(B, media=True)
check("B: OCR missed kwh → WAITING_FOR_KWH", state_of(B) == states.WAITING_FOR_KWH)

send(B, text="garbage")
check("B: invalid kwh stays in WAITING_FOR_KWH", state_of(B) == states.WAITING_FOR_KWH)

send(B, text="450")
check("B: valid kwh, no state → WAITING_FOR_STATE", state_of(B) == states.WAITING_FOR_STATE)

send(B, text="Mordor")
check("B: invalid state stays in WAITING_FOR_STATE", state_of(B) == states.WAITING_FOR_STATE)

send(B, text="Putrajaya")
check("B: valid state → WAITING_FOR_ROOF", state_of(B) == states.WAITING_FOR_ROOF)

send(B, text="50")
check("B: roof given → DONE", state_of(B) == states.DONE)
summaryB = next((m for m in SENT if "Solar Estimate" in m), "")
check("B: summary shows Putrajaya", "Putrajaya" in summaryB)
check("B: summary shows 450 kWh", "450 kWh" in summaryB)
check("B: Putrajaya has no local installers → frank fallback wording",
      "No installers are based in Putrajaya" in summaryB)
check("B: fallback names the nearest state with installers", "Selangor" in summaryB)

# ── Run C: FAQ menu + tap + free-text assistant ──────────────────────────────
print("\n=== Run C: FAQ + assistant ===")
SENT.clear()
C = "60123000003"
send(C, text="hi")
check("C: intro mentions menu", any("menu" in m.lower() for m in SENT))

SENT.clear()
send(C, text="menu")
check("C: 'menu' sends the FAQ list", any(m.startswith("[LIST") for m in SENT))

SENT.clear()
send(C, interactive_id="faq_nem")
check("C: tapping an FAQ row returns the canned answer",
      any("Solar ATAP" in m for m in SENT))

SENT.clear()
send(C, text="how do solar panels work?")
check("C: free-text question routes to the assistant",
      any(m.startswith("[ANSWER:") for m in SENT))

SENT.clear()
send(C, interactive_id="unknown_row")
check("C: unknown FAQ id re-sends the menu", any(m.startswith("[LIST") for m in SENT))

# ── Run D: installer lookup service ──────────────────────────────────────────
print("\n=== Run D: installer lookup ===")
from app.services import installers as inst

sel = inst.find_installers("Selangor")
check("D: Selangor resolves directly (no fallback)",
      sel["resolved"] and not sel["fallback"] and sel["count"] > 0)

lab = inst.find_installers("Labuan")
check("D: Labuan falls back to the nearest state (Sabah)",
      lab["resolved"] and lab["fallback"] and lab["nearest_state"] == "Sabah")

pin = inst.find_installers("Pulau Pinang")
check("D: 'Pulau Pinang' normalises to Penang", pin["requested_state"] == "Penang")

bad = inst.find_installers("Atlantis")
check("D: unknown state is reported as unresolved",
      not bad["resolved"] and bad["count"] == 0)

# ── Run E: manual assessment (no bill) ───────────────────────────────────────
print("\n=== Run E: manual assessment ===")
SENT.clear()
E = "60123000005"
send(E, text="hi")
check("E: greeting → WAITING_FOR_BILL", state_of(E) == states.WAITING_FOR_BILL)

SENT.clear()
send(E, text="manual")
check("E: 'manual' → WAITING_FOR_KWH", state_of(E) == states.WAITING_FOR_KWH)
check("E: manual intro asks for kWh", any("kWh" in m or "kwh" in m.lower() for m in SENT))

send(E, text="380")
check("E: kWh given → WAITING_FOR_STATE", state_of(E) == states.WAITING_FOR_STATE)

send(E, text="Johor")
check("E: state given → WAITING_FOR_ROOF", state_of(E) == states.WAITING_FOR_ROOF)

send(E, text="45")
check("E: roof given → DONE", state_of(E) == states.DONE)
summaryE = next((m for m in SENT if "Solar Estimate" in m), "")
check("E: summary shows Johor", "Johor" in summaryE)
check("E: summary shows 380 kWh", "380 kWh" in summaryE)

# ── Run F: bill media is deleted right after OCR ─────────────────────────────
print("\n=== Run F: bill media cleanup ===")
SENT.clear()
bill_extractor.extract_bill = lambda path: {
    "state": "Selangor", "total_kwh": 467, "total_amount_rm": 207.49,
    "confidence": 0.9, "notes": [],
}
F = "60123000006"
send(F, text="hi")
media_dir = os.environ["MEDIA_DIR"]
before = set(os.listdir(media_dir)) if os.path.isdir(media_dir) else set()
send(F, media=True)
after = set(os.listdir(media_dir)) if os.path.isdir(media_dir) else set()
check("F: no new bill file left on disk after OCR", after == before)

# ── Run G: webhook signature verification ────────────────────────────────────
print("\n=== Run G: webhook signature verification ===")
import hashlib
import hmac as hmac_mod

from app.whatsapp import parser as wa_parser

body = b'{"entry": []}'

check("G: no secret configured → verification skipped (dev mode)",
      wa_parser.verify_signature(body, None) is True)

os.environ["WHATSAPP_APP_SECRET"] = "test-secret"
good_sig = "sha256=" + hmac_mod.new(b"test-secret", body, hashlib.sha256).hexdigest()
check("G: correct signature accepted", wa_parser.verify_signature(body, good_sig) is True)

bad_sig = "sha256=" + hmac_mod.new(b"wrong-secret", body, hashlib.sha256).hexdigest()
check("G: forged signature rejected", wa_parser.verify_signature(body, bad_sig) is False)

check("G: missing header rejected once secret is set",
      wa_parser.verify_signature(body, None) is False)

check("G: malformed header rejected",
      wa_parser.verify_signature(body, "not-sha256-prefixed") is False)

tampered_body = b'{"entry": [1]}'
check("G: signature for different body is rejected",
      wa_parser.verify_signature(tampered_body, good_sig) is False)

del os.environ["WHATSAPP_APP_SECRET"]

# ── Run H: data retention purge ───────────────────────────────────────────────
print("\n=== Run H: data retention purge ===")
import sqlite3
from datetime import datetime, timedelta, timezone

H = "60123000008"
send(H, text="hi")  # creates a fresh, recent message row

old_ts = (datetime.now(timezone.utc) - timedelta(days=99)).isoformat()
with sqlite3.connect(os.environ["SQLITE_DB_PATH"]) as conn:
    conn.execute(
        "INSERT INTO messages (phone_number, direction, message_type, text, created_at) "
        "VALUES (?, 'in', 'text', 'ancient message', ?)", (H, old_ts),
    )
    conn.execute(
        "INSERT INTO bill_extractions (phone_number, raw_file_path, extraction_json, "
        "confidence, created_at) VALUES (?, NULL, '{}', 0.9, ?)", (H, old_ts),
    )

store.purge_old_data(days=30)

with sqlite3.connect(os.environ["SQLITE_DB_PATH"]) as conn:
    remaining = conn.execute(
        "SELECT text FROM messages WHERE phone_number = ?", (H,)
    ).fetchall()
    remaining_extractions = conn.execute(
        "SELECT id FROM bill_extractions WHERE phone_number = ?", (H,)
    ).fetchall()

check("H: old message purged, recent one kept",
      "ancient message" not in [r[0] for r in remaining] and len(remaining) >= 1)
check("H: old bill extraction purged", len(remaining_extractions) == 0)

# ── dedupe ───────────────────────────────────────────────────────────────────
print("\n=== Dedupe ===")
SENT.clear()
dup = InboundMessage(from_number=B, wa_message_id="dup-1", msg_type="text", text="hi")
orchestrator.handle_inbound(dup)
first = len(SENT)
orchestrator.handle_inbound(dup)  # same wamid again
check("duplicate wamid ignored", len(SENT) == first)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("All WhatsApp flow tests passed.")
