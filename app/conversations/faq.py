"""
Tappable FAQ for the WhatsApp bot — canned answers with ZERO API cost.

The bot sends these as a WhatsApp interactive *list*; when the user taps a row,
WhatsApp returns the row's `id`, and we reply with the matching answer below.
Only free-text questions outside this list fall through to Claude.

WhatsApp list constraints: row title ≤ 24 chars, description ≤ 72 chars,
max 10 rows total.

Each entry has both "en" and "bm" copies keyed by the same id, so a contact's
current language (see app/conversations/i18n.py) determines which is shown.
"""

SEDA_URL = "https://www.seda.gov.my/directory/registered-pv-service-provider-directory/"

FAQ = [
    {
        "id": "faq_what",
        "en": {
            "title": "What is SuriaSnap?",
            "desc": "How this service works",
            "answer": (
                "*SuriaSnap* gives you an instant rooftop-solar estimate for your "
                "Malaysian home. Just send a photo of your *electricity bill* and you'll get "
                "your recommended system size, monthly savings, payback period and "
                "CO2 reduction in seconds — free."
            ),
        },
        "bm": {
            "title": "Apa itu SuriaSnap?",
            "desc": "Cara perkhidmatan ini berfungsi",
            "answer": (
                "*SuriaSnap* memberi anda anggaran solar bumbung segera untuk rumah "
                "Malaysia anda. Hantar sahaja foto *bil elektrik* anda dan anda akan dapat "
                "saiz sistem yang disyorkan, penjimatan bulanan, tempoh pulang modal dan "
                "pengurangan CO2 dalam beberapa saat — percuma."
            ),
        },
    },
    {
        "id": "faq_nem",
        "en": {
            "title": "What is Solar ATAP?",
            "desc": "Solar ATAP programme explained",
            "answer": (
                "*Solar ATAP* (Skim Suria Atap) is SEDA's programme that lets *TNB* customers "
                "(Peninsular Malaysia + Labuan) sell surplus solar power back to the grid. Export rates: "
                "*RM 0.27/kWh* (≤1,500 kWh/month) or *RM 0.37/kWh* (>1,500 kWh/month). Your panels offset "
                "your own usage first; any excess is exported and credited on your next bill. 10-year "
                "contract, no quota limits.\n\n"
                "*Sabah (SESB)* and *Sarawak (SESCO)* aren't on Solar ATAP — they each run their own "
                "separate net-metering scheme. Sarawak also offers a one-off subsidy (RM8,000–12,000) "
                "toward install cost. Send your bill and I'll tell you exactly what applies to you."
            ),
        },
        "bm": {
            "title": "Apa itu Solar ATAP?",
            "desc": "Penjelasan skim Solar ATAP",
            "answer": (
                "*Solar ATAP* (Skim Suria Atap) ialah program SEDA yang membolehkan pelanggan *TNB* "
                "(Semenanjung Malaysia + Labuan) menjual lebihan kuasa solar semula ke grid. Kadar eksport: "
                "*RM 0.27/kWh* (≤1,500 kWh/bulan) atau *RM 0.37/kWh* (>1,500 kWh/bulan). Panel anda "
                "mengimbangi penggunaan sendiri dahulu; lebihan dieksport dan dikreditkan pada bil "
                "seterusnya. Kontrak 10 tahun, tiada had kuota.\n\n"
                "*Sabah (SESB)* dan *Sarawak (SESCO)* tidak termasuk dalam Solar ATAP — masing-masing "
                "menjalankan skim net-metering berasingan sendiri. Sarawak juga menawarkan subsidi "
                "sekali sahaja (RM8,000–12,000) untuk kos pemasangan. Hantar bil anda dan saya akan "
                "beritahu apa yang terpakai untuk anda."
            ),
        },
    },
    {
        "id": "faq_save",
        "en": {
            "title": "How much can I save?",
            "desc": "Typical savings on your bill",
            "answer": (
                "It depends on your usage and roof, but typical Malaysian homes cut "
                "*50–90%* of their electricity bill. Send me your latest *electricity bill* and I'll "
                "calculate your exact savings and payback period."
            ),
        },
        "bm": {
            "title": "Berapa saya boleh jimat?",
            "desc": "Anggaran penjimatan bil anda",
            "answer": (
                "Bergantung pada penggunaan dan bumbung anda, tetapi rumah Malaysia biasanya "
                "mengurangkan *50–90%* bil elektrik mereka. Hantar *bil elektrik* terkini anda dan saya "
                "akan kira penjimatan dan tempoh pulang modal sebenar anda."
            ),
        },
    },
    {
        "id": "faq_roof",
        "en": {
            "title": "Is my roof suitable?",
            "desc": "What makes a good solar roof",
            "answer": (
                "Most roofs work! South-facing, unshaded roofs perform best. A "
                "SEDA-registered installer confirms with a site survey. Send your bill "
                "and I'll estimate a system for your roof size and direction."
            ),
        },
        "bm": {
            "title": "Bumbung saya sesuai?",
            "desc": "Ciri bumbung solar yang baik",
            "answer": (
                "Kebanyakan bumbung sesuai! Bumbung menghadap selatan tanpa bayangan berprestasi "
                "terbaik. Pemasang berdaftar SEDA akan sahkan dengan tinjauan tapak. Hantar bil anda "
                "dan saya akan anggarkan sistem mengikut saiz dan arah bumbung anda."
            ),
        },
    },
    {
        "id": "faq_cost",
        "en": {
            "title": "How much does it cost?",
            "desc": "Rough installed price",
            "answer": (
                "Around *RM 7,000 per kWp* installed (2025). A typical home system is "
                "4–8 kWp, so roughly RM 28,000–56,000 before savings. It usually pays "
                "for itself in about *6–13 years*, then it's largely free electricity."
            ),
        },
        "bm": {
            "title": "Berapa kosnya?",
            "desc": "Anggaran harga pemasangan",
            "answer": (
                "Sekitar *RM 7,000 setiap kWp* dipasang (2025). Sistem rumah biasa ialah "
                "4–8 kWp, jadi anggaran RM 28,000–56,000 sebelum penjimatan. Biasanya pulang modal "
                "dalam *6–13 tahun*, selepas itu elektrik hampir percuma."
            ),
        },
    },
    {
        "id": "faq_apply",
        "en": {
            "title": "How do I apply?",
            "desc": "Steps to go solar",
            "answer": (
                "1) Get quotes from *SEDA-registered installers*.\n"
                "2) They submit your *Solar ATAP* application to SEDA.\n"
                "3) After approval, they install and your utility (TNB / SESB / SESCO) fits a bi-directional meter.\n"
                f"Find installers: {SEDA_URL}"
            ),
        },
        "bm": {
            "title": "Macam mana nak mohon?",
            "desc": "Langkah untuk beralih ke solar",
            "answer": (
                "1) Dapatkan sebut harga daripada *pemasang berdaftar SEDA*.\n"
                "2) Mereka hantar permohonan *Solar ATAP* anda ke SEDA.\n"
                "3) Selepas lulus, mereka pasang sistem dan utiliti anda (TNB / SESB / SESCO) memasang meter dwi-arah.\n"
                f"Cari pemasang: {SEDA_URL}"
            ),
        },
    },
    {
        "id": "faq_seda",
        "en": {
            "title": "What is SEDA?",
            "desc": "The authority behind Solar ATAP",
            "answer": (
                "*SEDA* (Sustainable Energy Development Authority) is Malaysia's agency "
                "that runs the *Solar ATAP* programme and registers approved "
                "solar installers (RPVSP). Always use a SEDA-registered installer."
            ),
        },
        "bm": {
            "title": "Apa itu SEDA?",
            "desc": "Pihak berkuasa di sebalik Solar ATAP",
            "answer": (
                "*SEDA* (Sustainable Energy Development Authority) ialah agensi Malaysia "
                "yang menjalankan program *Solar ATAP* dan mendaftar pemasang solar "
                "yang diluluskan (RPVSP). Sentiasa gunakan pemasang berdaftar SEDA."
            ),
        },
    },
    {
        "id": "faq_install",
        "en": {
            "title": "How long to install?",
            "desc": "Installation timeline",
            "answer": (
                "Once approved, a home system usually takes *1–3 days* to install, "
                "plus a few weeks for SEDA and your utility to approve and fit the bi-directional meter."
            ),
        },
        "bm": {
            "title": "Berapa lama pemasangan?",
            "desc": "Garis masa pemasangan",
            "answer": (
                "Selepas lulus, sistem rumah biasanya mengambil *1–3 hari* untuk dipasang, "
                "ditambah beberapa minggu untuk SEDA dan utiliti anda meluluskan dan memasang meter dwi-arah."
            ),
        },
    },
]

_BY_ID = {f["id"]: f for f in FAQ}


def faq_rows(lang: str = "en") -> list[dict]:
    """Rows for a WhatsApp interactive list message, in the given language."""
    return [{"id": f["id"], "title": f[lang]["title"], "description": f[lang]["desc"]} for f in FAQ]


def faq_answer(row_id: str | None, lang: str = "en") -> str | None:
    f = _BY_ID.get(row_id)
    return f[lang]["answer"] if f else None


def faq_titles(lang: str = "en") -> list[str]:
    """Plain titles, for the non-interactive text fallback."""
    return [f[lang]["title"] for f in FAQ]
