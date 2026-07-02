"""
Conversation orchestrator — the simple state machine that drives the WhatsApp
flow. No agent framework, no LLM-in-the-loop for routing: just explicit states
and plain Python branching, which is the cheapest and most predictable design.

Entry point: handle_inbound(msg) — called as a FastAPI BackgroundTask so the
webhook can return 200 immediately while OCR runs.

Collected inputs accumulate in the contact's `pending` blob:
    { "total_kwh": float, "state": str, "roof_area_sqm": float }
Once all three are present we run the assessment and send the result + PDF.
"""

import logging
import os
import re
import uuid
from pathlib import Path

from app.conversations import faq, states, store
from app.extraction import bill_extractor
from app.reports import adapter as reports
from app.reports import design_preview
from app.services import assistant
from app.services import installers
from app.services.utils import utility_name
from app.solar import adapter as solar
from app.whatsapp import client as wa
from app.whatsapp.parser import InboundMessage

logger = logging.getLogger("suriasnap.orchestrator")

MEDIA_DIR = os.getenv("MEDIA_DIR", "media")

CO2_PER_TREE_KG = 22
DEFAULT_ROOF_HINT = 40  # m², a typical Malaysian terrace — suggested if unsure

# SEDA's official registered PV service provider directory (verified 200 OK).
SEDA_RPVSP_URL = "https://www.seda.gov.my/directory/registered-pv-service-provider-directory/"

_GREETINGS = {
    "hi", "hello", "hey", "start", "hai", "helo",
    "salam", "assalamualaikum", "suria", "suriasnap",
}

# Words that open the tappable FAQ menu (free — no Claude call).
_FAQ_COMMANDS = {
    "menu", "faq", "help", "question", "questions", "tanya", "soalan", "info",
}

# Words that start a manual (no-bill) assessment.
_MANUAL_COMMANDS = {
    "manual", "manually", "no bill", "nobill", "tanpa bil", "masukkan manual",
    "enter", "type", "input", "taip",
}

INTRO = (
    "👋 Hi! I'm *SuriaSnap*.\n\n"
    "📄 *Option 1 — Snap your bill:* Send a photo or PDF of your latest electricity bill "
    "and I'll read your usage automatically.\n\n"
    "✏️ *Option 2 — Enter manually:* Type *manual* and I'll walk you through a few quick "
    "questions instead.\n\n"
    "Either way you'll get your rooftop solar estimate — system size, monthly savings, "
    "payback period, and CO₂ reduction — free, in under a minute.\n\n"
    "Type *menu* for common questions about solar."
)

MANUAL_INTRO = (
    "✏️ No problem — let's do it manually!\n\n"
    "I'll ask you 3 quick things and give you your solar estimate right away.\n\n"
    "*Step 1 of 3:* What is your average *monthly electricity usage* in kWh?\n"
    "(Check a recent bill or your utility's app — e.g. *450*)"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _send(phone: str, body: str) -> None:
    """Send text and log it; never let a send failure crash the flow."""
    try:
        wa.send_text(phone, body)
    except Exception:
        logger.exception("Failed to send text to %s", phone)
    store.log_message(phone, "out", "text", body)


def _send_faq_menu(phone: str) -> None:
    """Send the tappable FAQ list (free). Falls back to plain text if the
    interactive message can't be sent."""
    try:
        wa.send_list(
            phone,
            body="Tap a question below for an instant answer — or just type your "
                 "own question about solar. 👇",
            button="Common questions",
            rows=faq.faq_rows(),
            section_title="SuriaSnap FAQ",
        )
    except Exception:
        logger.exception("FAQ list failed for %s; sending plain text", phone)
        lines = "\n".join(f"• {f['title']}" for f in faq.FAQ)
        _send(phone, "Common questions — just ask me any of these:\n" + lines)
    store.log_message(phone, "out", "interactive", "faq menu")


def _is_greeting(text: str) -> bool:
    return text.strip().lower() in _GREETINGS


def _is_manual_command(text: str) -> bool:
    t = text.strip().lower()
    return t in _MANUAL_COMMANDS or t.startswith("manual")


def _start_manual(phone: str) -> None:
    store.set_pending(phone, {})
    store.set_state(phone, states.WAITING_FOR_KWH)
    _send(phone, MANUAL_INTRO)


def _parse_number(text: str) -> float | None:
    """Pull the first number out of a free-text reply ('about 450 kwh' → 450)."""
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else None


def _next_missing(pending: dict) -> str | None:
    if not pending.get("total_kwh"):
        return "kwh"
    if not pending.get("state"):
        return "state"
    if not pending.get("roof_area_sqm"):
        return "roof"
    return None


def _save_media(phone: str, data: bytes, mime: str, filename: str | None) -> str:
    Path(MEDIA_DIR).mkdir(parents=True, exist_ok=True)
    ext = ".pdf" if "pdf" in mime else (".png" if "png" in mime else ".jpg")
    safe = f"{phone}_{uuid.uuid4().hex[:8]}{ext}"
    path = str(Path(MEDIA_DIR) / safe)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ── public entry point ────────────────────────────────────────────────────────

def handle_inbound(msg: InboundMessage) -> None:
    """Top-level handler with dedupe + a safety net around the whole turn."""
    phone = msg.from_number

    # Meta retries webhooks; skip anything we've already logged inbound.
    if store.already_processed(msg.wa_message_id):
        logger.info("Skipping duplicate wamid %s", msg.wa_message_id)
        return

    store.get_or_create_contact(phone, msg.profile_name)
    store.log_message(
        phone, "in", msg.msg_type, msg.text, msg.media_id, msg.wa_message_id
    )

    try:
        _route(phone, msg)
    except Exception:
        logger.exception("Unhandled error for %s", phone)
        _send(phone, "⚠️ Sorry, something went wrong on our end. Type *hi* to start again.")
        store.set_state(phone, states.ERROR)


# ── routing ───────────────────────────────────────────────────────────────────

def _route(phone: str, msg: InboundMessage) -> None:
    contact = store.get_contact(phone)
    state = contact["current_state"]
    text = (msg.text or "").strip()

    # A greeting always restarts the conversation cleanly.
    if msg.msg_type == "text" and _is_greeting(text):
        store.set_pending(phone, {})
        store.set_state(phone, states.WAITING_FOR_BILL)
        _send(phone, INTRO)
        return

    # FAQ menu command — free, no Claude call.
    if msg.msg_type == "text" and text.lower() in _FAQ_COMMANDS:
        _send_faq_menu(phone)
        return

    # Manual assessment — user wants to enter kWh / state / roof without a bill.
    if msg.msg_type == "text" and _is_manual_command(text):
        _start_manual(phone)
        return

    # User tapped an FAQ row → canned answer (free).
    if msg.msg_type == "interactive":
        answer = faq.faq_answer(msg.text)
        if answer:
            _send(phone, answer)
            _send(phone, "Type *menu* for more, or send your *electricity bill* for a free estimate.")
        else:
            _send_faq_menu(phone)
        return

    # A bill image/PDF can arrive at any point and kicks off processing.
    if msg.msg_type in ("image", "document"):
        _handle_bill(phone, msg)
        return

    # Plain text: interpret based on what we're waiting for.
    if msg.msg_type == "text":
        _handle_text(phone, state, text)
        return

    # Stickers, audio, etc.
    _send(phone, "Please send your *electricity bill* as a photo or PDF 📄, or type *hi* to start.")


def _handle_text(phone: str, state: str, text: str) -> None:
    if state == states.WAITING_FOR_KWH:
        num = _parse_number(text)
        if num and bill_extractor.plausible_kwh(num):
            store.merge_pending(phone, total_kwh=num)
            _advance(phone)
        else:
            _send(phone, "Please send just your *monthly usage* in kWh, e.g. *450*")
        return

    if state == states.WAITING_FOR_STATE:
        canonical = solar.normalize_state(text)
        if canonical:
            store.merge_pending(phone, state=canonical)
            _advance(phone)
        else:
            _send(
                phone,
                "I didn't recognise that state 🤔. Please type one of:\n"
                + ", ".join(solar.CANONICAL_STATES),
            )
        return

    if state == states.WAITING_FOR_ROOF:
        num = _parse_number(text)
        if num and 5 <= num <= 1000:
            store.merge_pending(phone, roof_area_sqm=num)
            _advance(phone)
        else:
            _send(
                phone,
                "Please send your approximate *usable roof area* in square "
                f"metres, e.g. *{DEFAULT_ROOF_HINT}*",
            )
        return

    # Not waiting on a specific input → treat it as a question and let the
    # assistant answer (solar / SuriaSnap topics only). Cheap Claude Haiku call;
    # off-topic messages are politely declined by the system prompt.
    store.set_state(phone, states.WAITING_FOR_BILL)
    _send(phone, assistant.answer_question(text))


def _handle_bill(phone: str, msg: InboundMessage) -> None:
    store.set_state(phone, states.PROCESSING_BILL)
    _send(phone, "Got it! 📄 Reading your bill now — this takes a few seconds… ⏳")

    if not msg.media_id:
        _send(phone, "I couldn't find the file. Please try sending the bill again.")
        store.set_state(phone, states.WAITING_FOR_BILL)
        return

    # 1. download from WhatsApp
    try:
        data, mime = wa.download_media(msg.media_id)
    except Exception:
        logger.exception("Media download failed for %s", phone)
        _send(phone, "I couldn't download that file 😕. Please try sending it again.")
        store.set_state(phone, states.WAITING_FOR_BILL)
        return

    path = _save_media(phone, data, mime, msg.media_filename)

    # 2. extract (Tesseract OCR)
    extraction = bill_extractor.extract_bill(path)
    store.save_extraction(phone, path, extraction, extraction.get("confidence") or 0.0)

    # 3. seed what we learned into pending
    if extraction.get("total_kwh"):
        store.merge_pending(phone, total_kwh=extraction["total_kwh"])
    canonical = solar.normalize_state(extraction.get("state"))
    if canonical:
        store.merge_pending(phone, state=canonical)

    # 4. if usage is missing or low-confidence, confirm with the user
    if not extraction.get("total_kwh") or bill_extractor.is_low_confidence(extraction):
        store.set_state(phone, states.WAITING_FOR_KWH)
        confidence_note = (
            "I couldn't read your usage clearly"
            if not extraction.get("total_kwh")
            else "Just to be safe"
        )
        _send(
            phone,
            f"{confidence_note}. What's your *average monthly usage* in kWh? "
            "(it's on your bill — e.g. *450*)",
        )
        return

    _advance(phone)


def _advance(phone: str) -> None:
    """Ask for the next missing input, or finish if we have everything."""
    pending = store.get_pending(phone)
    missing = _next_missing(pending)

    if missing == "kwh":
        store.set_state(phone, states.WAITING_FOR_KWH)
        _send(phone, "What's your *average monthly usage* in kWh? (e.g. *450*)")
    elif missing == "state":
        store.set_state(phone, states.WAITING_FOR_STATE)
        _send(phone, "Which *Malaysian state* is the home in? (e.g. *Selangor*)")
    elif missing == "roof":
        store.set_state(phone, states.WAITING_FOR_ROOF)
        _send(
            phone,
            "Almost there! Roughly how big is your *usable roof area* in square "
            f"metres? A typical terrace is ~{DEFAULT_ROOF_HINT} m². If unsure, "
            f"just reply *{DEFAULT_ROOF_HINT}*. 🏠",
        )
    else:
        _finish(phone, pending)


def _finish(phone: str, pending: dict) -> None:
    state = pending["state"]
    kwh = pending["total_kwh"]
    roof = pending["roof_area_sqm"]
    orientation = solar.DEFAULT_ORIENTATION

    result = solar.run_assessment(state, kwh, roof, orientation)

    _send(phone, _format_summary(state, kwh, roof, orientation, result))

    # Professional design preview image (representative Arka-360-style layout).
    try:
        specific_yield = round(
            result["monthly_generation_kwh"] * 12 / result["recommended_system_kwp"]
        )
        png = design_preview.render_design_png(
            result["num_panels_400w"], orientation,
            result["recommended_system_kwp"], specific_yield,
        )
        img_id = wa.upload_media(png, "SuriaSnap-Design.png", "image/png")
        wa.send_image(
            phone, img_id,
            caption="🛠️ Your professional design preview — a representative layout. "
                    "A SEDA-registered installer finalises the certified design.",
        )
        store.log_message(phone, "out", "image", "design preview")
    except Exception:
        logger.exception("Design image delivery failed for %s", phone)

    # Optional PDF — reuse the existing report generator + free media upload.
    try:
        pdf = reports.generate_pdf_bytes(state, kwh, roof, orientation, result)
        media_id = wa.upload_media(pdf, "SuriaSnap-Report.pdf", "application/pdf")
        wa.send_document(
            phone, media_id, "SuriaSnap-Report.pdf",
            caption="📄 Your full SuriaSnap solar report",
        )
        store.log_message(phone, "out", "document", "SuriaSnap-Report.pdf")
    except Exception:
        logger.exception("PDF report delivery failed for %s (summary already sent)", phone)

    store.set_state(phone, states.DONE)
    store.set_pending(phone, {})


def _format_summary(state, kwh, roof, orientation, r: dict) -> str:
    co2 = r["annual_co2_offset_kg"]
    trees = int(co2 / CO2_PER_TREE_KG)
    monthly = r["monthly_savings_rm"]
    annual = monthly * 12
    roi25 = r["roi_25_year_rm"]
    sy = round(r["monthly_generation_kwh"] * 12 / r["recommended_system_kwp"])
    return (
        "☀️ *Your SuriaSnap Solar Estimate*\n\n"
        f"📍 {state}  ·  ⚡ {kwh:.0f} kWh/month\n"
        f"🏠 Roof ~{roof:.0f} m² ({orientation}-facing, assumed)\n\n"
        "Here's what your roof could be earning you 👇\n\n"
        f"🔋 *{r['recommended_system_kwp']} kWp* system — "
        f"{r['num_panels_400w']} × 400W panels\n"
        f"💰 *RM {monthly:,.0f}/month* saved — that's *RM {annual:,.0f}* a year "
        "back in your pocket\n"
        f"📈 *RM {roi25:,.0f}* net profit over 25 years\n"
        f"⏳ *Payback:* ~{r['payback_years']} years — after that, it's "
        "basically free electricity\n"
        f"🌳 *{co2:,.0f} kg* less CO₂ a year — like planting *{trees:,} trees* 🌲\n\n"
        "⏰ *The hidden cost of waiting:* every month you stay on the grid, "
        f"~RM {monthly:,.0f} leaves your pocket for {utility_name(state)} — about *RM {annual:,.0f} "
        "a year* you'll never get back. Solar panels run for 25+ years, so the "
        "sooner you switch, the more you keep. Your roof is already sitting in the "
        "sun — it might as well be paying you. ☀️\n\n"
        "🛠️ *Professional design preview*\n"
        f"{r['num_panels_400w']} × 400W · 15° tilt · {orientation}-facing · "
        f"~{sy:,} kWh/kWp/yr — see the design image & full report below.\n\n"
        + _installer_block(state)
        + f"_Based on {utility_name(state)} 2025/26 tariffs & SEDA Solar ATAP rates. Get a free site survey "
        "from an installer for exact figures._"
    )


def _installer_block(state: str) -> str:
    """Real SEDA-registered installers for the user's state (with nearest-state
    fallback), formatted for WhatsApp. Falls back to the directory link if the
    state isn't recognised."""
    rec = installers.find_installers(state)
    url = rec.get("official_directory") or SEDA_RPVSP_URL

    if not rec["resolved"] or not rec["installers"]:
        return ("👉 *Next step:* browse trusted, SEDA-registered installers near "
                f"you:\n{url}\n\n")

    lines = []
    for i in rec["installers"][:3]:
        web = i["website"].split("//")[-1].rstrip("/")
        loc = i["city"] if i["city"] == i["hq_state"] else f"{i['city']}, {i['hq_state']}"
        lines.append(f"• *{i['name']}* — {loc}\n  {web}")
    body = "\n".join(lines)

    if rec["fallback"]:
        header = (f"🛠️ *No installers are based in {rec['requested_state']}* — here are "
                  f"the nearest SEDA-registered ones (in {rec['nearest_state']}; most "
                  f"also serve {rec['requested_state']}):")
    else:
        header = f"🛠️ *SEDA-registered installers in {rec['requested_state']}:*"

    return f"{header}\n{body}\n\nFull SEDA directory:\n{url}\n\n"
