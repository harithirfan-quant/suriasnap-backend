"""
Bilingual (EN/BM) strings for the WhatsApp bot.

Each entry is {"en": ..., "bm": ...}. Templated strings use {placeholder}
and are filled with t(key, lang, **kwargs) — plain str.format(), no i18n
framework needed at this scale.

Default language is English; a contact switches with the words "bm" / "en"
(handled in orchestrator._is_lang_switch). The website mirrors this same
EN/BM toggle pattern.
"""

_STRINGS = {
    "intro": {
        "en": (
            "👋 Hi! I'm *SuriaSnap*.\n\n"
            "📄 *Option 1 — Snap your bill:* Send a photo or PDF of your latest electricity bill "
            "and I'll read your usage automatically.\n\n"
            "✏️ *Option 2 — Enter manually:* Tap below and I'll walk you through a few quick "
            "questions instead.\n\n"
            "Either way you'll get your rooftop solar estimate — system size, monthly savings, "
            "payback period, and CO₂ reduction — free, in under a minute.\n\n"
            "Type *menu* for common questions, or *bm* for Bahasa Malaysia."
        ),
        "bm": (
            "👋 Hai! Saya *SuriaSnap*.\n\n"
            "📄 *Pilihan 1 — Hantar bil anda:* Hantar foto atau PDF bil elektrik terkini anda "
            "dan saya akan baca penggunaan anda secara automatik.\n\n"
            "✏️ *Pilihan 2 — Isi secara manual:* Ketik di bawah dan saya akan tanya beberapa "
            "soalan ringkas.\n\n"
            "Kedua-dua cara akan beri anda anggaran solar bumbung anda — saiz sistem, "
            "penjimatan bulanan, tempoh pulang modal, dan pengurangan CO₂ — percuma, dalam "
            "kurang seminit.\n\n"
            "Taip *menu* untuk soalan lazim, atau *en* untuk Bahasa Inggeris."
        ),
    },
    "intro_btn_scan": {"en": "📄 Scan bill", "bm": "📄 Hantar bil"},
    "intro_btn_manual": {"en": "✏️ Enter manually", "bm": "✏️ Isi manual"},
    "intro_btn_faq": {"en": "❓ FAQ", "bm": "❓ Soalan lazim"},
    "intro_scan_reply": {
        "en": "Send your electricity bill as a photo or PDF 📄",
        "bm": "Hantar bil elektrik anda sebagai foto atau PDF 📄",
    },

    "manual_intro": {
        "en": (
            "✏️ No problem — let's do it manually!\n\n"
            "I'll ask you 3 quick things and give you your solar estimate right away.\n\n"
            "*Step 1 of 3:* What is your average *monthly electricity usage* in kWh?\n"
            "(Check a recent bill or your utility's app — e.g. *450*)"
        ),
        "bm": (
            "✏️ Tiada masalah — mari kita buat secara manual!\n\n"
            "Saya akan tanya 3 perkara ringkas dan terus beri anggaran solar anda.\n\n"
            "*Langkah 1 daripada 3:* Berapakah *penggunaan elektrik bulanan* purata anda dalam kWh?\n"
            "(Semak bil terkini atau app utiliti anda — cth. *450*)"
        ),
    },

    "faq_menu_body": {
        "en": "Tap a question below for an instant answer — or just type your own question about solar. 👇",
        "bm": "Ketik satu soalan di bawah untuk jawapan segera — atau taip soalan anda sendiri tentang solar. 👇",
    },
    "faq_menu_button": {"en": "Common questions", "bm": "Soalan lazim"},
    "faq_menu_section_title": {"en": "SuriaSnap FAQ", "bm": "Soalan Lazim SuriaSnap"},
    "faq_menu_fallback_intro": {
        "en": "Common questions — just ask me any of these:",
        "bm": "Soalan lazim — tanya saya mana-mana soalan ini:",
    },
    "after_faq_answer": {
        "en": "Type *menu* for more, or send your *electricity bill* for a free estimate.",
        "bm": "Taip *menu* untuk lagi, atau hantar *bil elektrik* anda untuk anggaran percuma.",
    },
    "unknown_message": {
        "en": "Please send your *electricity bill* as a photo or PDF 📄, or type *hi* to start.",
        "bm": "Sila hantar *bil elektrik* anda sebagai foto atau PDF 📄, atau taip *hai* untuk mula.",
    },

    "kwh_invalid": {
        "en": "Please send just your *monthly usage* in kWh, e.g. *450*",
        "bm": "Sila hantar *penggunaan bulanan* anda sahaja dalam kWh, cth. *450*",
    },
    "kwh_prompt": {
        "en": "What's your *average monthly usage* in kWh? (e.g. *450*)",
        "bm": "Berapakah *penggunaan bulanan* purata anda dalam kWh? (cth. *450*)",
    },

    "state_region_prompt": {
        "en": "Which *region* is the home in?",
        "bm": "Rumah anda di *kawasan* mana?",
    },
    "state_region_button": {"en": "Choose region", "bm": "Pilih kawasan"},
    "state_region_section": {"en": "Regions", "bm": "Kawasan"},
    "state_state_prompt": {
        "en": "Which *state*?",
        "bm": "Negeri yang mana?",
    },
    "state_state_button": {"en": "Choose state", "bm": "Pilih negeri"},
    "state_state_section": {"en": "States", "bm": "Negeri"},
    "state_invalid": {
        "en": "I didn't recognise that state 🤔. Please type one of:\n{states}",
        "bm": "Saya tidak kenal negeri itu 🤔. Sila taip salah satu:\n{states}",
    },

    "roof_prompt": {
        "en": (
            "Almost there! Roughly how big is your *usable roof area* in square "
            "metres? A typical terrace is ~{hint} m². If unsure, just reply *{hint}*. 🏠"
        ),
        "bm": (
            "Hampir siap! Anggaran berapa besar *keluasan bumbung boleh guna* anda dalam "
            "meter persegi? Rumah teres biasa ~{hint} m². Jika tidak pasti, balas *{hint}*. 🏠"
        ),
    },
    "roof_invalid": {
        "en": "Please send your approximate *usable roof area* in square metres, e.g. *{hint}*",
        "bm": "Sila hantar anggaran *keluasan bumbung boleh guna* anda dalam meter persegi, cth. *{hint}*",
    },

    "assistant_cap_hit": {
        "en": (
            "You've hit today's question limit 🙏. Type *menu* for common questions, "
            "or send your *electricity bill* for a free estimate — that's always available."
        ),
        "bm": (
            "Anda telah mencapai had soalan hari ini 🙏. Taip *menu* untuk soalan lazim, "
            "atau hantar *bil elektrik* anda untuk anggaran percuma — itu sentiasa percuma."
        ),
    },

    "bill_reading": {
        "en": "Got it! 📄 Reading your bill now — this takes a few seconds… ⏳",
        "bm": "Baik! 📄 Sedang membaca bil anda — ambil beberapa saat sahaja… ⏳",
    },
    "bill_not_found": {
        "en": "I couldn't find the file. Please try sending the bill again.",
        "bm": "Saya tidak jumpa fail itu. Sila cuba hantar bil semula.",
    },
    "bill_download_failed": {
        "en": "I couldn't download that file 😕. Please try sending it again.",
        "bm": "Saya tidak dapat muat turun fail itu 😕. Sila cuba hantar semula.",
    },
    "bill_confirm_no_kwh": {
        "en": "I couldn't read your usage clearly",
        "bm": "Saya tidak dapat baca penggunaan anda dengan jelas",
    },
    "bill_confirm_low_conf": {
        "en": "Just to be safe",
        "bm": "Untuk memastikan ketepatan",
    },
    "bill_confirm_ask_kwh": {
        "en": "{note}. What's your *average monthly usage* in kWh? (it's on your bill — e.g. *450*)",
        "bm": "{note}. Berapakah *penggunaan bulanan* purata anda dalam kWh? (ada pada bil anda — cth. *450*)",
    },

    "design_preview_caption": {
        "en": (
            "🛠️ Your professional design preview — a representative layout. "
            "A SEDA-registered installer finalises the certified design."
        ),
        "bm": (
            "🛠️ Pratonton reka bentuk profesional anda — susun atur contoh. "
            "Pemasang berdaftar SEDA akan menghasilkan reka bentuk akhir diperakui."
        ),
    },
    "pdf_caption": {
        "en": "📄 Your full SuriaSnap solar report",
        "bm": "📄 Laporan solar penuh SuriaSnap anda",
    },
    "error_message": {
        "en": "⚠️ Sorry, something went wrong on our end. Type *hi* to start again.",
        "bm": "⚠️ Maaf, ada masalah di pihak kami. Taip *hai* untuk mula semula.",
    },
    "lang_switched": {
        "en": "✅ Switched to *English*.",
        "bm": "✅ Ditukar kepada *Bahasa Malaysia*.",
    },
}


def t(key: str, lang: str, **kwargs) -> str:
    """Look up a translated string and fill in any {placeholder} kwargs."""
    entry = _STRINGS.get(key)
    if not entry:
        return key
    text = entry.get(lang) or entry.get("en") or key
    return text.format(**kwargs) if kwargs else text
