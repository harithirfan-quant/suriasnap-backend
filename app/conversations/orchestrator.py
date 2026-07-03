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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.conversations import faq, states, store
from app.conversations.i18n import t
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

# Free-text questions call Claude Haiku (small cost per call). Capped per
# phone number per rolling 24h so one number can't run up the API bill —
# the FAQ menu, greeting, and bill scan all stay free regardless.
ASSISTANT_DAILY_CAP = int(os.getenv("ASSISTANT_DAILY_CAP", "20"))

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

# Words that switch the contact's language.
_LANG_COMMANDS = {
    "bm": {"bm", "bahasa", "bahasa malaysia", "malay"},
    "en": {"en", "english", "inggeris", "bahasa inggeris"},
}

# Regions group the 16 states/territories into WhatsApp-list-sized chunks —
# a single interactive list message can hold at most 10 rows, so all 16
# states can't fit in one message. Two-step drill-down (region → state)
# keeps every list well under that limit.
_REGIONS = {
    "north":   {"en": "Northern",      "bm": "Utara",          "states": ["Perlis", "Kedah", "Penang", "Perak"]},
    "central": {"en": "Central",       "bm": "Tengah",         "states": ["Selangor", "Kuala Lumpur", "Putrajaya", "Negeri Sembilan"]},
    "south":   {"en": "Southern",      "bm": "Selatan",        "states": ["Melaka", "Johor"]},
    "east_c":  {"en": "East Coast",    "bm": "Pantai Timur",   "states": ["Pahang", "Terengganu", "Kelantan"]},
    "east_m":  {"en": "East Malaysia", "bm": "Malaysia Timur", "states": ["Sabah", "Sarawak", "Labuan"]},
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _send(phone: str, body: str, message_type: str = "text") -> None:
    """Send text and log it; never let a send failure crash the flow."""
    try:
        wa.send_text(phone, body)
    except Exception:
        logger.exception("Failed to send text to %s", phone)
    store.log_message(phone, "out", message_type, body)


def _send_intro(phone: str, lang: str) -> None:
    """Greeting with tappable Scan/Manual/FAQ buttons. Falls back to plain
    text (with typed instructions) if the interactive message can't be sent."""
    try:
        wa.send_buttons(
            phone,
            body=t("intro", lang),
            buttons=[
                {"id": "intro_scan",   "title": t("intro_btn_scan", lang)},
                {"id": "intro_manual", "title": t("intro_btn_manual", lang)},
                {"id": "intro_faq",    "title": t("intro_btn_faq", lang)},
            ],
        )
    except Exception:
        logger.exception("Intro buttons failed for %s; sending plain text", phone)
        _send(phone, t("intro", lang))
    store.log_message(phone, "out", "interactive", "intro buttons")


def _send_faq_menu(phone: str, lang: str) -> None:
    """Send the tappable FAQ list (free). Falls back to plain text if the
    interactive message can't be sent."""
    try:
        wa.send_list(
            phone,
            body=t("faq_menu_body", lang),
            button=t("faq_menu_button", lang),
            rows=faq.faq_rows(lang),
            section_title=t("faq_menu_section_title", lang),
        )
    except Exception:
        logger.exception("FAQ list failed for %s; sending plain text", phone)
        lines = "\n".join(f"• {title}" for title in faq.faq_titles(lang))
        _send(phone, t("faq_menu_fallback_intro", lang) + "\n" + lines)
    store.log_message(phone, "out", "interactive", "faq menu")


def _send_region_list(phone: str, lang: str) -> None:
    rows = [{"id": f"region_{key}", "title": r[lang]} for key, r in _REGIONS.items()]
    try:
        wa.send_list(
            phone,
            body=t("state_region_prompt", lang),
            button=t("state_region_button", lang),
            rows=rows,
            section_title=t("state_region_section", lang),
        )
    except Exception:
        logger.exception("Region list failed for %s; sending plain text", phone)
        _send(phone, t("state_region_prompt", lang) + "\n" +
              ", ".join(r[lang] for r in _REGIONS.values()))
    store.log_message(phone, "out", "interactive", "region list")


def _send_state_list(phone: str, lang: str, region_key: str) -> None:
    region = _REGIONS.get(region_key)
    if not region:
        _send_region_list(phone, lang)
        return
    rows = [{"id": f"state_{s}", "title": s} for s in region["states"]]
    try:
        wa.send_list(
            phone,
            body=t("state_state_prompt", lang),
            button=t("state_state_button", lang),
            rows=rows,
            section_title=t("state_state_section", lang),
        )
    except Exception:
        logger.exception("State list failed for %s; sending plain text", phone)
        _send(phone, t("state_state_prompt", lang) + "\n" + ", ".join(region["states"]))
    store.log_message(phone, "out", "interactive", "state list")


def _is_greeting(text: str) -> bool:
    return text.strip().lower() in _GREETINGS


def _is_manual_command(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in _MANUAL_COMMANDS or lowered.startswith("manual")


def _lang_switch(text: str) -> str | None:
    """Return 'bm'/'en' if the text is a language-switch command, else None."""
    lowered = text.strip().lower()
    for lang, words in _LANG_COMMANDS.items():
        if lowered in words:
            return lang
    return None


def _start_manual(phone: str, lang: str) -> None:
    store.set_pending(phone, {})
    store.set_state(phone, states.WAITING_FOR_KWH)
    _send(phone, t("manual_intro", lang))


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

    contact = store.get_contact(phone)
    lang = (contact or {}).get("lang") or "en"

    try:
        _route(phone, msg, lang)
    except Exception:
        logger.exception("Unhandled error for %s", phone)
        _send(phone, t("error_message", lang))
        store.set_state(phone, states.ERROR)


# ── routing ───────────────────────────────────────────────────────────────────

def _route(phone: str, msg: InboundMessage, lang: str) -> None:
    contact = store.get_contact(phone)
    state = contact["current_state"]
    text = (msg.text or "").strip()

    # A greeting always restarts the conversation cleanly.
    if msg.msg_type == "text" and _is_greeting(text):
        store.set_pending(phone, {})
        store.set_state(phone, states.WAITING_FOR_BILL)
        _send_intro(phone, lang)
        return

    # Language switch — works at any point, doesn't reset progress.
    if msg.msg_type == "text":
        new_lang = _lang_switch(text)
        if new_lang:
            store.set_lang(phone, new_lang)
            _send(phone, t("lang_switched", new_lang))
            if state in (states.WAITING_FOR_KWH, states.WAITING_FOR_STATE, states.WAITING_FOR_ROOF):
                _advance(phone, new_lang)
            return

    # FAQ menu command — free, no Claude call.
    if msg.msg_type == "text" and text.lower() in _FAQ_COMMANDS:
        _send_faq_menu(phone, lang)
        return

    # Manual assessment — user wants to enter kWh / state / roof without a bill.
    if msg.msg_type == "text" and _is_manual_command(text):
        _start_manual(phone, lang)
        return

    # User tapped an interactive button/row.
    if msg.msg_type == "interactive":
        row_id = msg.text or ""

        if row_id == "intro_scan":
            _send(phone, t("intro_scan_reply", lang))
            return
        if row_id == "intro_manual":
            _start_manual(phone, lang)
            return
        if row_id == "intro_faq":
            _send_faq_menu(phone, lang)
            return
        if row_id.startswith("region_"):
            _send_state_list(phone, lang, row_id.removeprefix("region_"))
            return
        if row_id.startswith("state_"):
            canonical = solar.normalize_state(row_id.removeprefix("state_"))
            if canonical:
                store.merge_pending(phone, state=canonical)
                _advance(phone, lang)
            else:
                _send_region_list(phone, lang)
            return

        answer = faq.faq_answer(row_id, lang)
        if answer:
            _send(phone, answer)
            _send(phone, t("after_faq_answer", lang))
        else:
            _send_faq_menu(phone, lang)
        return

    # A bill image/PDF can arrive at any point and kicks off processing.
    if msg.msg_type in ("image", "document"):
        _handle_bill(phone, msg, lang)
        return

    # Plain text: interpret based on what we're waiting for.
    if msg.msg_type == "text":
        _handle_text(phone, state, text, lang)
        return

    # Stickers, audio, etc.
    _send(phone, t("unknown_message", lang))


def _handle_text(phone: str, state: str, text: str, lang: str) -> None:
    if state == states.WAITING_FOR_KWH:
        num = _parse_number(text)
        if num and bill_extractor.plausible_kwh(num):
            store.merge_pending(phone, total_kwh=num)
            _advance(phone, lang)
        else:
            _send(phone, t("kwh_invalid", lang))
        return

    if state == states.WAITING_FOR_STATE:
        canonical = solar.normalize_state(text)
        if canonical:
            store.merge_pending(phone, state=canonical)
            _advance(phone, lang)
        else:
            _send(phone, t("state_invalid", lang, states=", ".join(solar.CANONICAL_STATES)))
        return

    if state == states.WAITING_FOR_ROOF:
        num = _parse_number(text)
        if num and 5 <= num <= 1000:
            store.merge_pending(phone, roof_area_sqm=num)
            _advance(phone, lang)
        else:
            _send(phone, t("roof_invalid", lang, hint=DEFAULT_ROOF_HINT))
        return

    # Not waiting on a specific input → treat it as a question and let the
    # assistant answer (solar / SuriaSnap topics only). Cheap Claude Haiku call;
    # off-topic messages are politely declined by the system prompt.
    store.set_state(phone, states.WAITING_FOR_BILL)

    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    asked_today = store.count_messages_since(phone, "assistant", since)
    if asked_today >= ASSISTANT_DAILY_CAP:
        _send(phone, t("assistant_cap_hit", lang))
        return

    _send(phone, assistant.answer_question(text, lang), message_type="assistant")


def _handle_bill(phone: str, msg: InboundMessage, lang: str) -> None:
    store.set_state(phone, states.PROCESSING_BILL)
    _send(phone, t("bill_reading", lang))

    if not msg.media_id:
        _send(phone, t("bill_not_found", lang))
        store.set_state(phone, states.WAITING_FOR_BILL)
        return

    # 1. download from WhatsApp
    try:
        data, mime = wa.download_media(msg.media_id)
    except Exception:
        logger.exception("Media download failed for %s", phone)
        _send(phone, t("bill_download_failed", lang))
        store.set_state(phone, states.WAITING_FOR_BILL)
        return

    path = _save_media(phone, data, mime, msg.media_filename)

    # 2. extract (Tesseract OCR)
    extraction = bill_extractor.extract_bill(path)
    store.save_extraction(phone, path, extraction, extraction.get("confidence") or 0.0)

    # The raw bill image/PDF is never read again after OCR — delete it
    # immediately rather than leaving a copy of the user's electricity bill
    # sitting on disk (matches the retention promise in our Privacy Notice).
    try:
        os.remove(path)
    except OSError:
        logger.warning("Could not delete bill media at %s", path)

    # 3. seed what we learned into pending
    if extraction.get("total_kwh"):
        store.merge_pending(phone, total_kwh=extraction["total_kwh"])
    canonical = solar.normalize_state(extraction.get("state"))
    if canonical:
        store.merge_pending(phone, state=canonical)

    # 4. if usage is missing or low-confidence, confirm with the user
    if not extraction.get("total_kwh") or bill_extractor.is_low_confidence(extraction):
        store.set_state(phone, states.WAITING_FOR_KWH)
        note_key = "bill_confirm_no_kwh" if not extraction.get("total_kwh") else "bill_confirm_low_conf"
        _send(phone, t("bill_confirm_ask_kwh", lang, note=t(note_key, lang)))
        return

    _advance(phone, lang)


def _advance(phone: str, lang: str) -> None:
    """Ask for the next missing input, or finish if we have everything."""
    pending = store.get_pending(phone)
    missing = _next_missing(pending)

    if missing == "kwh":
        store.set_state(phone, states.WAITING_FOR_KWH)
        _send(phone, t("kwh_prompt", lang))
    elif missing == "state":
        store.set_state(phone, states.WAITING_FOR_STATE)
        _send_region_list(phone, lang)
    elif missing == "roof":
        store.set_state(phone, states.WAITING_FOR_ROOF)
        _send(phone, t("roof_prompt", lang, hint=DEFAULT_ROOF_HINT))
    else:
        _finish(phone, pending, lang)


def _finish(phone: str, pending: dict, lang: str) -> None:
    state = pending["state"]
    kwh = pending["total_kwh"]
    roof = pending["roof_area_sqm"]
    orientation = solar.DEFAULT_ORIENTATION

    result = solar.run_assessment(state, kwh, roof, orientation)

    _send(phone, _format_summary(state, kwh, roof, orientation, result, lang))

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
            caption=t("design_preview_caption", lang),
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
            caption=t("pdf_caption", lang),
        )
        store.log_message(phone, "out", "document", "SuriaSnap-Report.pdf")
    except Exception:
        logger.exception("PDF report delivery failed for %s (summary already sent)", phone)

    store.set_state(phone, states.DONE)
    store.set_pending(phone, {})


def _format_summary(state, kwh, roof, orientation, r: dict, lang: str) -> str:
    co2 = r["annual_co2_offset_kg"]
    trees = int(co2 / CO2_PER_TREE_KG)
    monthly = r["monthly_savings_rm"]
    annual = monthly * 12
    roi25 = r["roi_25_year_rm"]
    sy = round(r["monthly_generation_kwh"] * 12 / r["recommended_system_kwp"])
    utility = utility_name(state)

    if lang == "bm":
        return (
            "☀️ *Anggaran Solar SuriaSnap Anda*\n\n"
            f"📍 {state}  ·  ⚡ {kwh:.0f} kWh/bulan\n"
            f"🏠 Bumbung ~{roof:.0f} m² (menghadap {orientation}, andaian)\n\n"
            "Inilah yang bumbung anda boleh perolehi 👇\n\n"
            f"🔋 Sistem *{r['recommended_system_kwp']} kWp* — "
            f"{r['num_panels_400w']} × panel 400W\n"
            f"💰 *RM {monthly:,.0f}/bulan* dijimatkan — bersamaan *RM {annual:,.0f}* "
            "setahun kembali ke poket anda\n"
            f"📈 *RM {roi25:,.0f}* keuntungan bersih dalam 25 tahun\n"
            f"⏳ *Pulang modal:* ~{r['payback_years']} tahun — selepas itu, "
            "elektrik hampir percuma\n"
            f"🌳 *{co2:,.0f} kg* kurang CO₂ setahun — seperti menanam *{trees:,} pokok* 🌲\n\n"
            "⏰ *Kos tersembunyi jika tunggu:* setiap bulan anda kekal bergantung pada grid, "
            f"~RM {monthly:,.0f} meninggalkan poket anda untuk {utility} — anggaran *RM {annual:,.0f} "
            "setahun* yang tidak akan kembali. Panel solar bertahan 25+ tahun, jadi lebih "
            "awal anda beralih, lebih banyak anda simpan. Bumbung anda sudah pun terdedah "
            "kepada matahari — biarlah ia membayar anda pula. ☀️\n\n"
            "🛠️ *Pratonton reka bentuk profesional*\n"
            f"{r['num_panels_400w']} × 400W · condong 15° · menghadap {orientation} · "
            f"~{sy:,} kWh/kWp/thn — lihat imej reka bentuk & laporan penuh di bawah.\n\n"
            + _installer_block(state, lang)
            + f"_Berdasarkan tarif {utility} 2025/26 & kadar {r['scheme_name']}. Dapatkan tinjauan tapak "
            "percuma daripada pemasang untuk angka sebenar._"
        )

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
        f"~RM {monthly:,.0f} leaves your pocket for {utility} — about *RM {annual:,.0f} "
        "a year* you'll never get back. Solar panels run for 25+ years, so the "
        "sooner you switch, the more you keep. Your roof is already sitting in the "
        "sun — it might as well be paying you. ☀️\n\n"
        "🛠️ *Professional design preview*\n"
        f"{r['num_panels_400w']} × 400W · 15° tilt · {orientation}-facing · "
        f"~{sy:,} kWh/kWp/yr — see the design image & full report below.\n\n"
        + _installer_block(state, lang)
        + f"_Based on {utility} 2025/26 tariffs & {r['scheme_name']} rates. Get a free site survey "
        "from an installer for exact figures._"
    )


def _installer_block(state: str, lang: str) -> str:
    """Real SEDA-registered installers for the user's state (with nearest-state
    fallback), formatted for WhatsApp. Falls back to the directory link if the
    state isn't recognised."""
    rec = installers.find_installers(state)
    url = rec.get("official_directory") or SEDA_RPVSP_URL

    if not rec["resolved"] or not rec["installers"]:
        if lang == "bm":
            return (f"👉 *Langkah seterusnya:* semak imbas pemasang berdaftar SEDA "
                     f"berhampiran anda:\n{url}\n\n")
        return ("👉 *Next step:* browse trusted, SEDA-registered installers near "
                f"you:\n{url}\n\n")

    lines = []
    for i in rec["installers"][:3]:
        web = i["website"].split("//")[-1].rstrip("/")
        loc = i["city"] if i["city"] == i["hq_state"] else f"{i['city']}, {i['hq_state']}"
        lines.append(f"• *{i['name']}* — {loc}\n  {web}")
    body = "\n".join(lines)

    if lang == "bm":
        if rec["fallback"]:
            header = (f"🛠️ *Tiada pemasang berpangkalan di {rec['requested_state']}* — berikut "
                      f"yang paling hampir berdaftar SEDA (di {rec['nearest_state']}; kebanyakan "
                      f"turut berkhidmat di {rec['requested_state']}):")
        else:
            header = f"🛠️ *Pemasang berdaftar SEDA di {rec['requested_state']}:*"
        directory_label = "Direktori penuh SEDA:"
    else:
        if rec["fallback"]:
            header = (f"🛠️ *No installers are based in {rec['requested_state']}* — here are "
                      f"the nearest SEDA-registered ones (in {rec['nearest_state']}; most "
                      f"also serve {rec['requested_state']}):")
        else:
            header = f"🛠️ *SEDA-registered installers in {rec['requested_state']}:*"
        directory_label = "Full SEDA directory:"

    return f"{header}\n{body}\n\n{directory_label}\n{url}\n\n"
