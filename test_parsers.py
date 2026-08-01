"""Quick offline sanity test for the OCR field parsers (no cv2/tesseract needed)."""
import sys
import types

# Stub native deps so ocr_service imports without cv2/tesseract installed
for name in ("cv2", "pytesseract", "numpy"):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
sys.modules["numpy"].ndarray = object
sys.modules["numpy"].pi = 3.141592653589793
sys.modules["pytesseract"].pytesseract = types.SimpleNamespace(tesseract_cmd="")

from app.services.ocr_service import (
    _parse_consumption, _parse_meter_readings, _parse_bill_amount, _parse_state,
)

FAILURES = []

def check(label, actual, expected):
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        FAILURES.append(label)

# Real-bill trap: tariff-tier note "600kWh" appears BEFORE the actual usage
real_bill = """
Tenaga Nasional Berhad
NO. 12 JALAN MAWAR, TAMAN SERI
68000 AMPANG, SELANGOR
Tarif: Domestik
Kegunaan melebihi 600kWh sebulan akan dikenakan kadar berbeza
Bacaan Meter  Dahulu 6,695  Semasa 7,162
Jumlah Penggunaan Anda kWh 467.00
Jumlah Yang Perlu Dibayar RM 207.49
"""
check("meter diff beats tier note", _parse_consumption(real_bill), 467.0)
check("meter readings extracted", _parse_meter_readings(real_bill), (6695.0, 7162.0))
check("bill amount", _parse_bill_amount(real_bill), 207.49)
check("state Selangor", _parse_state(real_bill), "Selangor")

# eBill layout with "Unit Used"
ebill = "Account Statement\nPENANG\nUnit Used : 312\nTotal Amount Due RM 145.20"
check("Unit Used label", _parse_consumption(ebill), 312.0)
check("eBill amount", _parse_bill_amount(ebill), 145.2)
check("state Penang", _parse_state(ebill), "Penang")

# BM eBill with "Penggunaan Unit"
bm_ebill = "Penyata\nJOHOR\nPenggunaan Unit: 528 kWh\nJumlah RM 251.30"
check("Penggunaan Unit label", _parse_consumption(bm_ebill), 528.0)

# Semasa + kWh proximity (no meter table)
semasa = "Penggunaan\nSemasa 389 kWh\nAmaun RM 180.55"
check("Semasa proximity", _parse_consumption(semasa), 389.0)

# Tier-boundary fallback avoidance
tier_trap = "kegunaan 600 kWh tier\nusage was 455 kWh this month"
check("fallback avoids 600 boundary", _parse_consumption(tier_trap), 455.0)

# Nothing extractable
check("no kWh returns None", _parse_consumption("hello world, no numbers"), None)

# ── inbound webhook shapes ───────────────────────────────────────────────────
# Quick-reply taps on a *template* arrive as type "button" with a payload,
# unlike free-form interactive buttons (interactive.button_reply.id). Both must
# reach the orchestrator as msg_type="interactive" so the intro_* branches match.
from app.whatsapp.parser import parse_inbound


def _envelope(message: dict) -> dict:
    """Wrap one message in the entry/changes/value envelope Meta posts."""
    return {"entry": [{"changes": [{"value": {
        "contacts": [{"wa_id": "60123000001", "profile": {"name": "Test User"}}],
        "messages": [{"from": "60123000001", "id": "wamid-btn", **message}],
    }}]}]}


def _button(payload: str, text: str = "📄 Scan bill") -> dict:
    return {"type": "button", "button": {"payload": payload, "text": text}}


tap = parse_inbound(_envelope(_button("intro_scan")))[0]
check("template button tap → interactive", tap.msg_type, "interactive")
check("template button payload becomes text", tap.text, "intro_scan")
check("template button title kept in extras", tap.extras["title"], "📄 Scan bill")

unknown = parse_inbound(_envelope(_button("not_a_real_id", "Mystery")))[0]
check("unknown button payload still interactive, not other", unknown.msg_type, "interactive")
check("unknown button payload preserved", unknown.text, "not_a_real_id")

manual = parse_inbound(_envelope(_button("intro_manual", "✏️ Enter manually")))[0]
check("intro_manual payload round-trips", manual.text, "intro_manual")

# A button message with no payload must not crash the parser.
empty = parse_inbound(_envelope({"type": "button", "button": {}}))[0]
check("button with no payload parses with text=None", (empty.msg_type, empty.text),
      ("interactive", None))

# Free-form interactive replies keep working unchanged.
reply = parse_inbound(_envelope({
    "type": "interactive",
    "interactive": {"type": "button_reply",
                    "button_reply": {"id": "intro_faq", "title": "❓ FAQ"}},
}))[0]
check("interactive button_reply still parses", (reply.msg_type, reply.text),
      ("interactive", "intro_faq"))

sticker = parse_inbound(_envelope({"type": "sticker", "sticker": {"id": "S1"}}))[0]
check("unhandled type still falls through to other", sticker.msg_type, "other")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("All parser tests passed.")
