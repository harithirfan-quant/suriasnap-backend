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
pil.Image = pil_image
sys.modules["PIL"] = pil
sys.modules["PIL.Image"] = pil_image
sys.modules["httpx"] = types.ModuleType("httpx")  # never called in DRY_RUN

# ── isolated env ─────────────────────────────────────────────────────────────
_tmp = tempfile.mkdtemp()
os.environ["WHATSAPP_DRY_RUN"] = "true"
os.environ["SQLITE_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["MEDIA_DIR"] = os.path.join(_tmp, "media")

from app.conversations import orchestrator, states, store
from app.extraction import bill_extractor
from app.reports import adapter as reports
from app.whatsapp import client as wa
from app.whatsapp.parser import InboundMessage

store.init_db()

# ── capture outbound + fake the external calls ───────────────────────────────
SENT: list[str] = []
wa.send_text = lambda to, body: SENT.append(body)
wa.send_document = lambda to, mid, fn, caption=None: SENT.append(f"[DOC {fn}] {caption}")
wa.upload_media = lambda b, fn, mime: "FAKE_MEDIA_ID"
wa.download_media = lambda mid: (b"fake-bytes", "image/jpeg")
reports.generate_pdf_bytes = lambda *a, **k: b"%PDF-fake"

_n = 0
def send(phone, *, text=None, media=False):
    """Simulate one inbound WhatsApp message hitting the webhook."""
    global _n
    _n += 1
    msg = InboundMessage(
        from_number=phone,
        wa_message_id=f"wamid-{_n}",
        msg_type="image" if media else "text",
        text=text,
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
check("A: intro mentions TNB bill", "TNB bill" in SENT[-1])

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

send(B, text="Penang")
check("B: valid state → WAITING_FOR_ROOF", state_of(B) == states.WAITING_FOR_ROOF)

send(B, text="50")
check("B: roof given → DONE", state_of(B) == states.DONE)
summaryB = next((m for m in SENT if "Solar Estimate" in m), "")
check("B: summary shows Penang", "Penang" in summaryB)
check("B: summary shows 450 kWh", "450 kWh" in summaryB)

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
