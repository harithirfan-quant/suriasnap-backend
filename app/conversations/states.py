"""
Conversation states for the WhatsApp flow.

Plain string constants (not an Enum) so they store/read cleanly in SQLite and
are easy to eyeball in the database during a demo.

Flow:
    NEW ──greeting──► WAITING_FOR_BILL ──bill──► PROCESSING_BILL
        │                                            │
        │                              ┌─────────────┼──────────────┐
        │                              ▼             ▼              ▼
        │                      WAITING_FOR_KWH  WAITING_FOR_STATE  WAITING_FOR_ROOF
        │                              └─────────────┴──────────────┘
        │                                            ▼
        └────────────────────────────────────────► DONE
"""

NEW               = "NEW"                # first contact, nothing asked yet
WAITING_FOR_BILL  = "WAITING_FOR_BILL"   # intro sent, expecting a bill image/PDF
PROCESSING_BILL   = "PROCESSING_BILL"    # downloading + OCR in progress
WAITING_FOR_KWH   = "WAITING_FOR_KWH"    # OCR missed usage, asked user to type kWh
WAITING_FOR_STATE = "WAITING_FOR_STATE"  # state unknown, asked user for it
WAITING_FOR_ROOF  = "WAITING_FOR_ROOF"   # asking for usable roof area (not on bills)
DONE              = "DONE"               # result + report delivered
ERROR             = "ERROR"              # unrecoverable error, user can restart

# States in which a plain text number is meaningful to the state machine
NUMERIC_INPUT_STATES = {WAITING_FOR_KWH, WAITING_FOR_ROOF}
