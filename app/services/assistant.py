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
    "that gives instant rooftop-solar estimates from a photo of a TNB electricity "
    "bill.\n\n"
    "ONLY answer questions about these topics: SuriaSnap and how it works; "
    "rooftop/home solar energy; SEDA (Sustainable Energy Development Authority); "
    "TNB electricity bills, tariffs and usage; Solar ATAP (Skim Suria Atap); "
    "solar panels, inverters, batteries, installation, cost, savings and "
    "payback; and Arka 360 (solar design software). If a question is outside "
    "these topics, politely decline in ONE sentence and steer back to solar.\n\n"
    "Useful facts (Malaysia): Solar ATAP export rate RM 0.27/kWh (≤1,500 kWh/month) or RM 0.37/kWh (>1,500 kWh/month); installed cost "
    "~RM 7,000 per kWp; grid emission factor 0.758 kgCO2/kWh; system usually "
    "pays back in ~6-13 years; always recommend a SEDA-registered installer for a "
    "site survey and quote. If asked about the user's own savings or bill, tell "
    "them to send their TNB bill photo here for a personalised estimate.\n\n"
    "Style: friendly, concise WhatsApp tone — 1 to 4 short sentences, plain text, "
    "Malaysian context (RM, TNB, SEDA). Use *single asterisks* for bold sparingly. "
    "Do not invent precise numbers you're unsure of; suggest an installer quote."
)

_FALLBACK = (
    "I can help with questions about solar, SEDA, Solar ATAP, TNB bills and SuriaSnap. "
    "Type *menu* for common questions, or send a photo of your *TNB bill* for a "
    "free solar estimate."
)


def answer_question(question: str) -> str:
    """Answer a solar/SuriaSnap question. Best-effort: never raises."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _FALLBACK
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=ASSISTANT_MODEL,
            max_tokens=350,
            system=_SYSTEM,
            messages=[{"role": "user", "content": question.strip()[:600]}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return text or _FALLBACK
    except Exception:
        logger.exception("Assistant Q&A failed")
        return _FALLBACK
