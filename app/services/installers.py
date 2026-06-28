"""
SEDA-registered PV service provider (RPVSP) lookup.

Given a Malaysian state, returns the SEDA-registered installers based there. If a
state has none, we are upfront about it and surface the *nearest* state that does
have registered installers (walking a hand-built proximity map by driving
distance), so a user in e.g. Kedah still gets a useful, honest answer.

Data lives in app/data/rpvsp.json — a curated subset of SEDA's RPVSP directory
(the official directory is login-gated and not publicly browsable). Swapping in a
fuller export later is just a matter of replacing that JSON file.
"""
import json
import os
from functools import lru_cache

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rpvsp.json")

# Ordered nearest-state lists (closest first), by rough driving distance. Every
# list ends with a Klang Valley hub (Selangor / Kuala Lumpur) which always has
# installers, so the fallback walk is guaranteed to resolve.
PROXIMITY: dict[str, list[str]] = {
    "Perlis":          ["Kedah", "Penang", "Perak", "Kelantan", "Selangor", "Kuala Lumpur"],
    "Kedah":           ["Penang", "Perlis", "Perak", "Kelantan", "Selangor", "Kuala Lumpur"],
    "Penang":          ["Kedah", "Perak", "Perlis", "Selangor", "Kuala Lumpur", "Kelantan"],
    "Perak":           ["Penang", "Selangor", "Kuala Lumpur", "Kedah", "Pahang", "Negeri Sembilan", "Kelantan", "Perlis"],
    "Selangor":        ["Kuala Lumpur", "Putrajaya", "Negeri Sembilan", "Perak", "Pahang", "Melaka", "Penang", "Johor"],
    "Kuala Lumpur":    ["Selangor", "Putrajaya", "Negeri Sembilan", "Pahang", "Perak", "Melaka", "Johor"],
    "Putrajaya":       ["Selangor", "Kuala Lumpur", "Negeri Sembilan", "Melaka", "Perak", "Pahang"],
    "Negeri Sembilan": ["Melaka", "Selangor", "Kuala Lumpur", "Putrajaya", "Johor", "Pahang", "Perak"],
    "Melaka":          ["Negeri Sembilan", "Johor", "Selangor", "Kuala Lumpur", "Putrajaya", "Pahang"],
    "Johor":           ["Melaka", "Negeri Sembilan", "Pahang", "Selangor", "Kuala Lumpur", "Putrajaya"],
    "Pahang":          ["Selangor", "Kuala Lumpur", "Perak", "Negeri Sembilan", "Terengganu", "Kelantan", "Johor"],
    "Terengganu":      ["Kelantan", "Pahang", "Selangor", "Kuala Lumpur", "Johor"],
    "Kelantan":        ["Terengganu", "Pahang", "Perak", "Kedah", "Penang", "Selangor", "Kuala Lumpur"],
    "Sabah":           ["Labuan", "Sarawak", "Kuala Lumpur", "Selangor"],
    "Sarawak":         ["Labuan", "Sabah", "Kuala Lumpur", "Selangor"],
    "Labuan":          ["Sabah", "Sarawak", "Kuala Lumpur", "Selangor"],
}

# Common spellings / variants → canonical state name used in the dataset.
_ALIASES = {
    "pulau pinang": "Penang",
    "pinang": "Penang",
    "malacca": "Melaka",
    "negri sembilan": "Negeri Sembilan",
    "n. sembilan": "Negeri Sembilan",
    "n sembilan": "Negeri Sembilan",
    "kl": "Kuala Lumpur",
    "w.p. kuala lumpur": "Kuala Lumpur",
    "wp kuala lumpur": "Kuala Lumpur",
    "wilayah persekutuan kuala lumpur": "Kuala Lumpur",
    "kuala lumpur (wp)": "Kuala Lumpur",
    "w.p. putrajaya": "Putrajaya",
    "wp putrajaya": "Putrajaya",
    "w.p. labuan": "Labuan",
    "wp labuan": "Labuan",
}


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _canonical_states() -> set[str]:
    return {i["hq_state"] for i in _load()["installers"]} | set(PROXIMITY.keys())


def normalize_state(state: str | None) -> str | None:
    """Map a free-form state string to the canonical name, or None if unknown."""
    if not state:
        return None
    key = " ".join(state.strip().lower().split())
    if key in _ALIASES:
        return _ALIASES[key]
    for canon in _canonical_states():
        if canon.lower() == key:
            return canon
    return None


def installers_in(state: str) -> list[dict]:
    """All installers with a presence in `state` — i.e. HQ'd there, or with a
    listed branch office there (canonical name)."""
    return [
        i for i in _load()["installers"]
        if i["hq_state"] == state or state in i.get("branches", [])
    ]


def meta() -> dict:
    return _load()["_meta"]


def find_installers(state: str | None) -> dict:
    """
    Resolve installers for a user's state.

    Returns:
      {
        "requested_state": <normalized or original>,
        "resolved": bool,            # did we recognise the state at all
        "fallback": bool,            # True when we had to use a nearby state
        "nearest_state": str | None, # set when fallback is True
        "installers": [...],
        "count": int,
        "official_directory": str,
      }
    """
    official = meta().get("official_directory", "")
    canon = normalize_state(state)

    if canon is None:
        return {
            "requested_state": state, "resolved": False, "fallback": False,
            "nearest_state": None, "installers": [], "count": 0,
            "official_directory": official,
        }

    direct = installers_in(canon)
    if direct:
        return {
            "requested_state": canon, "resolved": True, "fallback": False,
            "nearest_state": None, "installers": direct, "count": len(direct),
            "official_directory": official,
        }

    # No installers based in this state — walk outward to the nearest one that has them.
    for near in PROXIMITY.get(canon, []):
        near_list = installers_in(near)
        if near_list:
            return {
                "requested_state": canon, "resolved": True, "fallback": True,
                "nearest_state": near, "installers": near_list, "count": len(near_list),
                "official_directory": official,
            }

    # Should never happen (hubs always populated), but stay safe.
    return {
        "requested_state": canon, "resolved": True, "fallback": False,
        "nearest_state": None, "installers": [], "count": 0,
        "official_directory": official,
    }
