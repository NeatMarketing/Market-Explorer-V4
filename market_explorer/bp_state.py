"""
Business plan state storage per profile.

"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


BP_DIR = Path("bp_saves")


def _profile_path(profile: str) -> Path:
    BP_DIR.mkdir(exist_ok=True)
    return BP_DIR / f"{profile}.json"


def load_bp_state(profile: str) -> Dict[str, Any]:
    """Load the saved BP store state for a profile."""
    path = _profile_path(profile)
    if not path.exists():
        return {"bps":{}, "last_opened_id":None}
    return _normalize_store(json.loads(path.read_text()))

def _normalize_store(data: Dict[str,Any]) -> Dict[str,Any]:

    if not data:

        return {"bps" : {}, "last_opened_id": None}
    if "bps" in data:
        data.setdefault("last_opened_id", None)
        return data
    if "name" in data:
        legacy_id = data.get("id") or "legacy"
        entry = dict(data)
        entry["id"] = legacy_id
        return {"bps": {legacy_id: entry}, "last_opened_id": legacy_id}
    return {"bps": {}, "last_opened_id": None}


def list_bp_states(profile: str) -> list[Dict[str, Any]]:
    """List saved BPs for a profile."""
    store = load_bp_state(profile)
    entries = list(store.get("bps", {}).values())
    return sorted(entries, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_bp_state(profile: str, bp_id: str) -> Dict[str, Any] | None:
    """Fetch a saved BP by id."""
    store = load_bp_state(profile)
    return store.get("bps", {}).get(bp_id)


def save_bp_state(profile: str, data: Dict[str, Any], bp_id: str | None = None) -> Dict[str, Any]:
    """Save a BP entry, returning the updated payload."""
    store = load_bp_state(profile)
    if not bp_id:
        bp_id = uuid.uuid4().hex
    updated = dict(data)
    updated["id"] = bp_id
    updated["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    store.setdefault("bps", {})
    store["bps"][bp_id] = updated
    store["last_opened_id"] = bp_id
    path = _profile_path(profile)
    path.write_text(json.dumps(store, indent = 2))
    return updated

def clear_bp_state(profile: str, bp_id: str | None = None) -> None:
    """Remove the BP state for a profile."""
    path = _profile_path(profile)
    if not path.exists():
        return
    if not bp_id:
        path.unlink()
        return
    store = load_bp_state(profile)
    store.get("bps", {}).pop(bp_id, None)
    if store.get("last_opened_id") == bp_id:
        store["last_opened_id"] = None
    path.write_text(json.dumps(store, indent=2))