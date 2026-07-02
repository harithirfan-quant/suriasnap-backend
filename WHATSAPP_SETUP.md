# SuriaSnap on WhatsApp — Setup & Run Guide

The WhatsApp bot is an **additional interface** on the same FastAPI backend. The
website (`/api/*`) is untouched. Bill reading uses the existing **free Tesseract
OCR** (no Claude / no paid API in v1), and the solar maths + PDF report reuse the
existing engine. **Monthly cost: ~RM 0.**

```
WhatsApp user → POST /webhooks/whatsapp → state machine (SQLite)
                    ↓
   Tesseract OCR → solar_calc.assess() → report_generator → WhatsApp reply + PDF
        (reused)        (reused)            (reused)
```

---

## STEP 4 — Environment variables

Add these to `.env` locally (copy from `.env.example`) and to the **Render
dashboard** in production. Never commit real values.

| Variable | Required | What it is / where to get it |
|---|---|---|
| `WHATSAPP_VERIFY_TOKEN` | ✅ | A random string **you invent**. Paste the *same* value into Meta's webhook config — the GET handshake must match. |
| `WHATSAPP_APP_SECRET` | ✅ | Meta App → **Settings → Basic → App secret**. Used to verify the `X-Hub-Signature-256` header on every inbound webhook POST, so forged requests are rejected. If unset, verification is skipped (logged as a warning) — fine for local dev, **must be set in production**. |
| `WHATSAPP_ACCESS_TOKEN` | ✅ | Meta App → **WhatsApp → API Setup**. Temp token (24h) for testing, or a permanent System User token for production. |
| `WHATSAPP_PHONE_NUMBER_ID` | ✅ | On the same API Setup page — the **Phone number ID** (a long number, *not* the phone number itself). |
| `WHATSAPP_API_VERSION` | ⬜ | Graph API version. Defaults to `v21.0`. |
| `WHATSAPP_DRY_RUN` | ⬜ | `true` = log outbound replies instead of calling Meta (local testing without a token). Leave `false`/unset in prod. |
| `DATABASE_URL` | ⬜* | Postgres connection string. **Set this in production** — see below. Unset = falls back to local SQLite (fine for dev, ephemeral on Render). |
| `SQLITE_DB_PATH` | ⬜ | SQLite path, only used when `DATABASE_URL` is unset. Default `suriasnap.db`. |
| `MEDIA_DIR` | ⬜ | Where downloaded bills are saved (deleted right after OCR). Default `media/`. |
| `RETENTION_DAYS` | ⬜ | Days to keep message logs / bill extractions before auto-purge. Default `30`. |

> **No `ANTHROPIC_API_KEY` needed** — Claude Vision is not used in v1.

---

## STEP 4b — Persistent storage (production)

Render's free tier wipes local disk on every deploy and every sleep→wake
cycle. Without `DATABASE_URL`, that means an in-progress WhatsApp
conversation (or the dedupe table, or the FAQ history) can reset mid-flow.
Fix it once with a free Postgres database:

1. Create a free project at **[neon.tech](https://neon.tech)** or
   **[supabase.com](https://supabase.com)** (no credit card required on
   either free tier).
2. Copy the connection string (starts with `postgresql://…`).
3. In the **Render dashboard** → your service → **Environment**, add
   `DATABASE_URL` with that value.
4. Redeploy. On the next startup the backend creates its tables in Postgres
   automatically (`store.init_db()` runs at boot either way) — no manual
   migration step.

No other code or config changes are needed; the backend detects
`DATABASE_URL` and switches over automatically.

---

## STEP 5 — Run & test

### A. Local run

```bash
cd suriasnap-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs system tesseract + poppler installed
cp .env.example .env                      # then fill in the WhatsApp values
uvicorn app.main:app --reload --port 8000
```

System tools (already in the Docker image; for local macOS):
```bash
brew install tesseract poppler
```

### B. Offline tests (no WhatsApp account needed)

```bash
python test_parsers.py          # OCR field parsers
python test_whatsapp_flow.py    # full conversation state machine (dry-run)
```

### C. Test the webhook with curl

Verification handshake (use your real `WHATSAPP_VERIFY_TOKEN`):
```bash
curl "http://localhost:8000/webhooks/whatsapp?hub.mode=subscribe&hub.verify_token=YOUR_TOKEN&hub.challenge=12345"
# → 12345
```

Simulate an inbound "hi" (set `WHATSAPP_DRY_RUN=true` first — the reply prints in
the server log instead of going to Meta):
```bash
curl -X POST http://localhost:8000/webhooks/whatsapp \
  -H "Content-Type: application/json" \
  -d '{"entry":[{"changes":[{"value":{"contacts":[{"wa_id":"60123456789","profile":{"name":"Test"}}],"messages":[{"from":"60123456789","id":"wamid.1","type":"text","text":{"body":"hi"}}]}}]}]}'
```

### D. Connect a real WhatsApp number (Meta Cloud API)

1. Create a Meta app at **developers.facebook.com** → add the **WhatsApp** product (free).
2. Expose your local server publicly: `ngrok http 8000` → copy the `https://…ngrok…` URL.
3. In **WhatsApp → Configuration → Webhook**:
   - Callback URL: `https://<your-domain>/webhooks/whatsapp`
   - Verify token: your `WHATSAPP_VERIFY_TOKEN`
   - Subscribe to the **messages** field.
4. On **API Setup**, add your own phone as a **recipient** (test mode only messages
   pre-registered numbers until the business is verified).
5. Message your test number on WhatsApp → say "hi" → send a TNB bill photo.

For production, point the webhook at your Render URL instead of ngrok:
`https://suriasnap-backend.onrender.com/webhooks/whatsapp`.

---

## Free-tier caveats (fine for a hackathon)

- **Render free tier sleeps** after 15 min → the first webhook after idle may hit a
  ~30–60s cold start. Meta retries webhooks, so messages aren't lost, just delayed.
- **Ephemeral storage** — the SQLite DB and saved bills reset on each deploy/sleep.
  Conversations don't survive restarts. Acceptable for a demo; add a Render Disk
  (paid) or external DB only if you need persistence.
- **WhatsApp test mode** only messages numbers you pre-register until Meta business
  verification.

## Enabling Claude Vision later (optional, paid)

Bill reading is isolated behind one function. To upgrade accuracy:
1. `pip install anthropic`, add `ANTHROPIC_API_KEY`.
2. Reimplement `app/extraction/bill_extractor.py:extract_bill()` to call Claude and
   return the same schema dict. Nothing else changes.
