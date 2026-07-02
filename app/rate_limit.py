"""
Single shared slowapi Limiter instance.

Routers import `limiter` from here (not from main.py) to avoid a circular
import, and so every `@limiter.limit(...)` decorator shares the same
in-memory rate-limit storage as the one registered on `app.state.limiter`
in main.py. Creating a separate `Limiter()` per file would give each one
its own storage and the limits would never actually trigger.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
