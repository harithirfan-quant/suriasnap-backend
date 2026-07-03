"""
Topic-constrained Q&A for the WhatsApp bot, powered by Claude Haiku (cheapest
model, same ANTHROPIC_API_KEY as the bill OCR).

Only used for free-text questions that aren't covered by the tappable FAQ, so
cost stays low (~0.5-1 sen per question, max_tokens capped). The system prompt
keeps Claude strictly on solar / SuriaSnap topics and declines everything else.
If no key is set or the call fails, returns a helpful static fallback (no cost).
"""

import logging
import os

logger = logging.getLogger("suriasnap.assistant")

ASSISTANT_MODEL = "claude-haiku-4-5-20251001"   # cheapest current model

_SYSTEM = (
    "You are the SuriaSnap WhatsApp assistant. SuriaSnap is a Malaysian service "
    "that gives instant rooftop-solar estimates from a photo of an electricity bill "
    "(TNB for Peninsular Malaysia, SESB for Sabah, SESCO for Sarawak).\n\n"
    "ONLY answer questions about these topics: SuriaSnap and how it works; "
    "rooftop/home solar energy; SEDA (Sustainable Energy Development Authority); "
    "electricity bills, tariffs and usage (TNB / SESB / SESCO); Solar ATAP (Skim Suria Atap) "
    "and the separate SESB/Sarawak Energy net-metering schemes; "
    "solar panels, inverters, batteries, installation, cost, savings and "
    "payback; and Arka 360 (solar design software). If a question is outside "
    "these topics, politely decline in ONE sentence and steer back to solar.\n\n"
    "Important: Solar ATAP is a SEDA programme that ONLY covers TNB's jurisdiction "
    "(Peninsular Malaysia + Labuan). Sabah (SESB) and Sarawak (SESCO/Sarawak Energy) are "
    "NOT on Solar ATAP — they each run their own separate net-metering scheme. Never call "
    "a Sabah or Sarawak user's scheme \"Solar ATAP\"; call it \"SESB's Net Energy Metering\" "
    "or \"Sarawak Energy's Net Energy Metering\" respectively. Sarawak Energy also offers a "
    "one-off NEM Subsidy Scheme (NEMSS) of RM8,000-12,000 toward install cost.\n\n"
    "Useful facts (Malaysia): Solar ATAP export rate RM 0.27/kWh (≤1,500 kWh/month) or "
    "RM 0.37/kWh (>1,500 kWh/month), TNB territory only; installed cost "
    "~RM 7,000 per kWp; grid emission factor 0.758 kgCO2/kWh; system usually "
    "pays back in ~6-13 years; always recommend a SEDA-registered installer for a "
    "site survey and quote. Peninsular Malaysia uses TNB, Sabah uses SESB, Sarawak uses SESCO. "
    "If asked about the user's own savings or bill, tell "
    "them to send their electricity bill photo here for a personalised estimate.\n\n"
    "Style: friendly, concise WhatsApp tone — 1 to 4 short sentences, plain text, "
    "Malaysian context (RM, SEDA). Use *single asterisks* for bold sparingly. "
    "Do not invent precise numbers you're unsure of; suggest an installer quote."
)

_LANG_INSTRUCTION = {
    "en": "Respond in English.",
    "bm": "Respond in Bahasa Malaysia (Malay), natural and conversational — not a stiff translation.",
}

_FALLBACK = {
    "en": (
        "I can help with questions about solar, SEDA, Solar ATAP, electricity bills and SuriaSnap. "
        "Type *menu* for common questions, or send a photo of your *electricity bill* for a "
        "free solar estimate."
    ),
    "bm": (
        "Saya boleh bantu dengan soalan tentang solar, SEDA, Solar ATAP, bil elektrik dan SuriaSnap. "
        "Taip *menu* untuk soalan lazim, atau hantar foto *bil elektrik* anda untuk anggaran "
        "solar percuma."
    ),
}


def answer_question(question: str, lang: str = "en") -> str:
    """Answer a solar/SuriaSnap question. Best-effort: never raises."""
    fallback = _FALLBACK.get(lang, _FALLBACK["en"])
    if not os.getenv("ANTHROPIC_API_KEY"):
        return fallback
    try:
        import anthropic

        client = anthropic.Anthropic()
        system = _SYSTEM + "\n\n" + _LANG_INSTRUCTION.get(lang, _LANG_INSTRUCTION["en"])
        resp = client.messages.create(
            model=ASSISTANT_MODEL,
            max_tokens=350,
            system=system,
            messages=[{"role": "user", "content": question.strip()[:600]}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return text or fallback
    except Exception:
        logger.exception("Assistant Q&A failed")
        return fallback
