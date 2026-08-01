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
from app.conversations.i18n import t
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
wa.send_list = lambda to, body, button, rows, section_title="", header=None: SENT.append(f"[LIST {len(rows)} rows] {body}")
wa.send_buttons = lambda to, body, buttons, header=None: SENT.append(body)
wa.upload_media = lambda b, fn, mime: "FAKE_MEDIA_ID"

# Template sends are recorded separately (name/language are what we assert on)
# and also mirrored into SENT: the approved template body carries the same
# intro copy the interactive/text tiers send, so intro assertions elsewhere in
# this file keep testing what the user actually receives.
TEMPLATES: list[dict] = []
TEMPLATE_FAILS = {"on": False}
def _fake_send_template(to, name, language, components=None):
    TEMPLATES.append({"to": to, "name": name, "language": language, "components": components})
    if TEMPLATE_FAILS["on"]:
        raise RuntimeError("(#132001) Template name does not exist in the translation")
    SENT.append(f"[TEMPLATE {name}/{language}] " + t("intro", "en"))
wa.send_template = _fake_send_template

wa.download_media = lambda mid: (b"fake-bytes", "image/jpeg")
reports.generate_pdf_bytes = lambda *a, **k: b"%PDF-fake"
design_preview.render_design_png = lambda *a, **k: b"PNG"
assistant.answer_question = lambda q, lang="en": f"[ANSWER:{lang}:{q}]"   # no real Claude call in tests

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

# ── Run C2: assistant daily cap ───────────────────────────────────────────────
print("\n=== Run C2: assistant daily cap ===")
orchestrator.ASSISTANT_DAILY_CAP = 3
C2 = "60123000004"
for i in range(orchestrator.ASSISTANT_DAILY_CAP):
    SENT.clear()
    send(C2, text=f"question number {i}")
    check(f"C2: question {i} answered normally", any(m.startswith("[ANSWER:") for m in SENT))

SENT.clear()
send(C2, text="one question too many")
check("C2: over the cap → polite limit message, no Claude call",
      not any(m.startswith("[ANSWER:") for m in SENT) and any("limit" in m.lower() for m in SENT))
orchestrator.ASSISTANT_DAILY_CAP = 20  # restore default for any later runs

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

# ── Run I: Sabah/Sarawak use their own scheme, not Solar ATAP ────────────────
print("\n=== Run I: SESB/SESCO scheme correctness ===")
from app.services import solar_calc

sabah_result = solar_calc.assess("Sabah", 400, 40, "South")
check("I: Sabah scheme is SESB NEM, not Solar ATAP",
      sabah_result["scheme_name"] == "SESB Net Energy Metering")

sarawak_result = solar_calc.assess("Sarawak", 400, 40, "South")
check("I: Sarawak scheme is Sarawak Energy NEM, not Solar ATAP",
      sarawak_result["scheme_name"] == "Sarawak Energy Net Energy Metering (NEM)")

selangor_result = solar_calc.assess("Selangor", 400, 40, "South")
check("I: TNB territory still uses Solar ATAP",
      selangor_result["scheme_name"] == "Solar ATAP")

labuan_result = solar_calc.assess("Labuan", 400, 40, "South")
check("I: Labuan (TNB territory) uses Solar ATAP",
      labuan_result["scheme_name"] == "Solar ATAP")

SENT.clear()
I = "60123000009"
send(I, text="hi")
send(I, text="manual")
send(I, text="400")
send(I, text="Sabah")
send(I, text="40")
summaryI = next((m for m in SENT if "Solar Estimate" in m), "")
check("I: Sabah WhatsApp summary mentions SESB scheme, not Solar ATAP",
      "SESB Net Energy Metering" in summaryI and "Solar ATAP" not in summaryI)

# ── Run J: BM language switching ──────────────────────────────────────────────
print("\n=== Run J: BM language switch ===")
J = "60123000010"
send(J, text="hi")
check("J: contact defaults to English", store.get_contact(J)["lang"] == "en")

SENT.clear()
send(J, text="bm")
check("J: 'bm' switches language", store.get_contact(J)["lang"] == "bm")
check("J: BM switch confirmation shown", "Bahasa Malaysia" in SENT[-1])

SENT.clear()
send(J, text="manual")
check("J: manual intro is in BM", "Langkah 1" in SENT[-1])

send(J, text="380")
check("J: kwh accepted, now waiting for state", state_of(J) == states.WAITING_FOR_STATE)

SENT.clear()
send(J, text="en")
check("J: 'en' switches back to English", store.get_contact(J)["lang"] == "en")
check("J: re-prompts the current step (state) in English after switch",
      any(m.startswith("[LIST") for m in SENT))

# ── Run K: region → state picker (interactive) ────────────────────────────────
print("\n=== Run K: region/state picker ===")
K = "60123000011"
send(K, text="hi")
send(K, text="manual")
send(K, text="420")
check("K: now waiting for state", state_of(K) == states.WAITING_FOR_STATE)

SENT.clear()
send(K, interactive_id="region_east_m")
check("K: tapping a region sends the state list for that region",
      any(m.startswith("[LIST") for m in SENT))

send(K, interactive_id="state_Sabah")
check("K: tapping a state resolves it and advances", state_of(K) == states.WAITING_FOR_ROOF)
check("K: Sabah was recorded", store.get_pending(K).get("state") == "Sabah")

send(K, text="35")
check("K: roof given → DONE", state_of(K) == states.DONE)

# ── Run L: intro buttons ───────────────────────────────────────────────────────
print("\n=== Run L: intro buttons ===")
L = "60123000012"
send(L, text="hi")
check("L: greeting delivers the intro copy",
      any(m.startswith("👋") or "SuriaSnap" in m for m in SENT))

SENT.clear()
send(L, interactive_id="intro_manual")
check("L: tapping 'Enter manually' button starts manual flow",
      state_of(L) == states.WAITING_FOR_KWH)

# ── Run M: oversized bill is rejected before OCR ─────────────────────────────
print("\n=== Run M: oversized bill rejection ===")
SENT.clear()
M = "60123000013"
send(M, text="hi")

# Temporarily make download_media return an 11MB blob (> the 10MB cap).
_orig_download = wa.download_media
wa.download_media = lambda mid: (b"x" * (11 * 1024 * 1024), "image/jpeg")
extraction_called = {"hit": False}
_orig_extract = bill_extractor.extract_bill
def _tracking_extract(path):
    extraction_called["hit"] = True
    return _orig_extract(path)
bill_extractor.extract_bill = _tracking_extract

SENT.clear()
send(M, media=True)
check("M: oversized bill → too-large message, OCR never runs",
      not extraction_called["hit"] and any("too large" in m.lower() for m in SENT))
check("M: state falls back to WAITING_FOR_BILL after rejection",
      state_of(M) == states.WAITING_FOR_BILL)

# Restore stubs so later runs are unaffected.
wa.download_media = _orig_download
bill_extractor.extract_bill = _orig_extract

# ── Run N: Solar ATAP savings never exceed the pre-solar bill ────────────────
# Regression test for the net-billing fix (SEDA GP/ST/No.60/2025 §14.1(e)):
# generation beyond consumption forfeits, it never becomes bonus cash on top
# of a fully-offset bill.
print("\n=== Run N: savings capped at pre-solar bill (no bonus export cash) ===")
from app.services import solar_calc

small_kwh = 150
oversized_roof = solar_calc.assess("Selangor", small_kwh, 300, "South")
check("N: oversized system generates far more than it consumes",
      oversized_roof["monthly_generation_kwh"] > small_kwh * 3)
pre_solar_bill = round(solar_calc._tnb_bill(small_kwh), 2)
check("N: monthly savings capped exactly at the pre-solar bill, no bonus",
      oversized_roof["monthly_savings_rm"] == pre_solar_bill)

undersized_roof = solar_calc.assess("Selangor", 1000, 20, "South")
check("N: undersized system still generates less than it consumes",
      undersized_roof["monthly_generation_kwh"] < 1000)
check("N: normal partial-offset case still saves less than the full bill",
      undersized_roof["monthly_savings_rm"] < round(solar_calc._tnb_bill(1000), 2))

check("N: Sabah oversized savings capped correctly",
      solar_calc.assess("Sabah", 150, 300, "South")["monthly_savings_rm"]
      == round(solar_calc._sesb_bill(150), 2))
check("N: Sarawak oversized savings capped correctly",
      solar_calc.assess("Sarawak", 150, 300, "South")["monthly_savings_rm"]
      == round(solar_calc._sesco_bill(150), 2))

# ── Run O: intro send cascade (template → buttons → text) ────────────────────
# The intro must lead with an approved template: interactive messages are
# rejected outside the 24-hour customer-service window, templates are not.
print("\n=== Run O: intro cascade ===")
O = "60123000014"

SENT.clear()
TEMPLATES.clear()
send(O, text="hi")
check("O: tier 1 — greeting sends a template first",
      len(TEMPLATES) == 1)
check("O: tier 1 — template name matches the approved template",
      TEMPLATES[-1]["name"] == orchestrator.INTRO_TEMPLATE_NAME == "suriasnap_intro")
check("O: tier 1 — template language matches the constant",
      TEMPLATES[-1]["language"] == orchestrator.INTRO_TEMPLATE_LANG)
check("O: tier 1 — template goes to the right number", TEMPLATES[-1]["to"] == O)
check("O: tier 1 — no interactive/text fallback when the template succeeds",
      len(SENT) == 1 and SENT[0].startswith("[TEMPLATE "))

# Tier 2: template rejected (e.g. not approved yet) → interactive buttons.
SENT.clear()
TEMPLATES.clear()
TEMPLATE_FAILS["on"] = True
orchestrator._send_intro(O, "en")
check("O: tier 2 — template was still attempted first", len(TEMPLATES) == 1)
check("O: tier 2 — falls back to the interactive buttons (intro copy, no template marker)",
      len(SENT) == 1 and "SuriaSnap" in SENT[0] and not SENT[0].startswith("[TEMPLATE "))

# Tier 3: template AND buttons rejected (e.g. outside the 24h window) → text.
SENT.clear()
TEMPLATES.clear()
_orig_buttons = wa.send_buttons
def _failing_buttons(to, body, buttons, header=None):
    raise RuntimeError("(#131047) Re-engagement message outside the 24 hour window")
wa.send_buttons = _failing_buttons
orchestrator._send_intro(O, "en")
check("O: tier 3 — both interactive tiers attempted before plain text",
      len(TEMPLATES) == 1)
check("O: tier 3 — plain-text intro still reaches the user",
      len(SENT) == 1 and "SuriaSnap" in SENT[0])

wa.send_buttons = _orig_buttons
TEMPLATE_FAILS["on"] = False

# BM contacts still get the intro through the cascade (template is EN-only for
# now; tiers 2/3 stay bilingual).
SENT.clear()
TEMPLATE_FAILS["on"] = True
orchestrator._send_intro(O, "bm")
check("O: BM fallback intro is in Bahasa Malaysia", "Hai! Saya" in SENT[-1])
TEMPLATE_FAILS["on"] = False

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
