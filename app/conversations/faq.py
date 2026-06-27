"""
Tappable FAQ for the WhatsApp bot — canned answers with ZERO API cost.

The bot sends these as a WhatsApp interactive *list*; when the user taps a row,
WhatsApp returns the row's `id`, and we reply with the matching answer below.
Only free-text questions outside this list fall through to Claude.

WhatsApp list constraints: row title ≤ 24 chars, description ≤ 72 chars,
max 10 rows total.
"""

SEDA_URL = "https://www.seda.gov.my/directory/registered-pv-service-provider-directory/"

FAQ = [
    {
        "id": "faq_what",
        "title": "What is SuriaSnap?",
        "desc": "How this service works",
        "answer": (
            "*SuriaSnap* gives you an instant rooftop-solar estimate for your "
            "Malaysian home. Just send a photo of your *TNB bill* and you'll get "
            "your recommended system size, monthly savings, payback period and "
            "CO2 reduction in seconds — free."
        ),
    },
    {
        "id": "faq_nem",
        "title": "What is NEM?",
        "desc": "Net Energy Metering explained",
        "answer": (
            "*NEM* (Net Energy Metering) under SEDA's *Solar ATAP* scheme lets you "
            "sell surplus solar power back to TNB at *RM 0.2703/kWh*. Your panels "
            "offset your own usage first; any extra is exported to the grid and "
            "credited on your next TNB bill."
        ),
    },
    {
        "id": "faq_save",
        "title": "How much can I save?",
        "desc": "Typical savings on your bill",
        "answer": (
            "It depends on your usage and roof, but typical Malaysian homes cut "
            "*50–90%* of their TNB bill. Send me your latest *TNB bill* and I'll "
            "calculate your exact savings and payback period."
        ),
    },
    {
        "id": "faq_roof",
        "title": "Is my roof suitable?",
        "desc": "What makes a good solar roof",
        "answer": (
            "Most roofs work! South-facing, unshaded roofs perform best. A "
            "SEDA-registered installer confirms with a site survey. Send your bill "
            "and I'll estimate a system for your roof size and direction."
        ),
    },
    {
        "id": "faq_cost",
        "title": "How much does it cost?",
        "desc": "Rough installed price",
        "answer": (
            "Around *RM 7,000 per kWp* installed (2025). A typical home system is "
            "4–8 kWp, so roughly RM 28,000–56,000 before savings. It usually pays "
            "for itself in about *6–13 years*, then it's largely free electricity."
        ),
    },
    {
        "id": "faq_apply",
        "title": "How do I apply?",
        "desc": "Steps to go solar",
        "answer": (
            "1) Get quotes from *SEDA-registered installers*.\n"
            "2) They submit your *Solar ATAP / NEM* application to SEDA.\n"
            "3) After approval, they install and TNB fits a bi-directional meter.\n"
            f"Find installers: {SEDA_URL}"
        ),
    },
    {
        "id": "faq_seda",
        "title": "What is SEDA?",
        "desc": "The authority behind NEM",
        "answer": (
            "*SEDA* (Sustainable Energy Development Authority) is Malaysia's agency "
            "that runs the *Solar ATAP / NEM* programme and registers approved "
            "solar installers (RPVSP). Always use a SEDA-registered installer."
        ),
    },
    {
        "id": "faq_install",
        "title": "How long to install?",
        "desc": "Installation timeline",
        "answer": (
            "Once approved, a home system usually takes *1–3 days* to install, "
            "plus a few weeks for SEDA/TNB approval and the bi-directional meter."
        ),
    },
]

_BY_ID = {f["id"]: f for f in FAQ}


def faq_rows() -> list[dict]:
    """Rows for a WhatsApp interactive list message."""
    return [{"id": f["id"], "title": f["title"], "description": f["desc"]} for f in FAQ]


def faq_answer(row_id: str | None) -> str | None:
    f = _BY_ID.get(row_id)
    return f["answer"] if f else None
