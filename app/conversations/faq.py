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
                "(Peninsular Malaysia + Labuan) offset their bill with surplus solar power. Credit rate: "
                "*RM 0.27/kWh* (≤1,500 kWh/month) or *RM 0.37/kWh* (>1,500 kWh/month) — the same rate as "
                "your normal energy charge. Your panels offset your own usage first; any surplus credits "
                "that same month's bill, down to *RM 0* — there's no cash payout beyond that, and unused "
                "surplus isn't carried to the next month. 10-year contract.\n\n"
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
                "(Semenanjung Malaysia + Labuan) mengimbangi bil dengan lebihan kuasa solar. Kadar kredit: "
                "*RM 0.27/kWh* (≤1,500 kWh/bulan) atau *RM 0.37/kWh* (>1,500 kWh/bulan) — sama seperti "
                "caj tenaga biasa anda. Panel anda mengimbangi penggunaan sendiri dahulu; lebihan "
                "mengkredit bil bulan yang sama, sehingga *RM 0* — tiada bayaran tunai selepas itu, dan "
                "lebihan tidak dibawa ke bulan seterusnya. Kontrak 10 tahun.\n\n"
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
        # Titles are capped at 24 chars by WhatsApp, so the row reads "How is my
        # estimate made?" while the answer carries the full engine explanation.
        # Every number here mirrors app/services/solar_calc.py.
        "id": "faq_how",
        "en": {
            "title": "How is my estimate made?",
            "desc": "How SuriaSnap calculates your numbers",
            "answer": (
                "We size the system from your *bill* and your *roof*. Panels are 400 W each and we "
                "only count *70%* of your roof as usable, so a 23.4 m2 roof fits about 8 panels = "
                "*3.2 kWp*.\n\n"
                "Generation uses your state's sunlight (GHI — e.g. Selangor 4.7 kWh/m2/day), your "
                "roof direction (south is best) and a 0.80 performance ratio. We then run your real "
                "tariff — *Solar ATAP* for TNB, or SESB / Sarawak Energy net metering — to get "
                "monthly savings, payback and 25-year ROI (with 0.5%/year panel degradation) at "
                "*RM 7,000 per kWp* installed.\n\n"
                "Example: 400 kWh/month in Selangor → 3.2 kWp, 8 panels, about *RM 1,924/year* "
                "saved and an *11.6-year* payback. It's an estimate, not a quote — a SEDA-registered "
                "installer confirms with a site survey."
            ),
        },
        "bm": {
            "title": "Bagaimana ia dikira?",
            "desc": "Cara SuriaSnap mengira anggaran anda",
            "answer": (
                "Kami saiz sistem daripada *bil* dan *bumbung* anda. Setiap panel 400 W dan hanya "
                "*70%* bumbung dikira boleh guna, jadi bumbung 23.4 m2 muat kira-kira 8 panel = "
                "*3.2 kWp*.\n\n"
                "Penjanaan mengikut cahaya matahari negeri anda (GHI — cth. Selangor 4.7 "
                "kWh/m2/hari), arah bumbung (selatan terbaik) dan nisbah prestasi 0.80. Kemudian "
                "kami guna tarif sebenar anda — *Solar ATAP* untuk TNB, atau net metering SESB / "
                "Sarawak Energy — untuk dapat penjimatan bulanan, pulang modal dan ROI 25 tahun "
                "(susut panel 0.5% setahun) pada *RM 7,000 sekWp* dipasang.\n\n"
                "Contoh: 400 kWh/bulan di Selangor → 3.2 kWp, 8 panel, jimat kira-kira "
                "*RM 1,924/tahun* dan pulang modal *11.6 tahun*. Ini anggaran, bukan sebut harga — "
                "pemasang berdaftar SEDA akan sahkan dengan tinjauan tapak."
            ),
        },
    },
    {
        "id": "faq_business",
        "en": {
            "title": "Are you an installer?",
            "desc": "List your company or update your contact info",
            "answer": (
                "SuriaSnap isn't an installer — we're a free estimate tool. We list about *435 "
                "SEDA-registered* installers at suriasnap.my, searchable by company name, and "
                "customers can send an installer their estimate in one tap.\n\n"
                "If you *are* a SEDA-registered installer and want to be listed, or want to update "
                "your contact details, email *harithirfanworkspace@gmail.com*."
            ),
        },
        "bm": {
            "title": "Anda pemasang solar?",
            "desc": "Senaraikan syarikat anda atau kemas kini maklumat",
            "answer": (
                "SuriaSnap bukan pemasang — kami alat anggaran percuma. Kami menyenaraikan "
                "kira-kira *435* pemasang *berdaftar SEDA* di suriasnap.my, boleh dicari mengikut "
                "nama syarikat, dan pelanggan boleh hantar anggaran kepada pemasang dengan satu "
                "ketikan.\n\n"
                "Jika anda *memang* pemasang berdaftar SEDA dan mahu disenaraikan, atau mahu kemas "
                "kini maklumat perhubungan, e-mel *harithirfanworkspace@gmail.com*."
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
