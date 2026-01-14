"""
Notes management per profile and company.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


NOTES_DIR = Path("notes")


def _profile_path(profile: str) -> Path:
    NOTES_DIR.mkdir(exist_ok=True)
    return NOTES_DIR / f"{profile}.json"


def load_notes(profile: str) -> Dict[str, Dict[str, Any]]:
    """Load notes for a given user profile."""
    path = _profile_path(profile)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_notes(profile: str, notes: Dict[str, Dict[str, Any]]) -> None:
    """Save notes for a given user profile."""
    path = _profile_path(profile)
    path.write_text(json.dumps(notes, indent=2))


def reset_notes(profile: str) -> None:
    """Delete all notes for a profile."""
    path = _profile_path(profile)
    if path.exists():
        path.unlink()


def company_key(name: str, country: str) -> str:
    """Create a stable key for a company."""
    return f"{name}__{country}".lower()


def upsert_note(
    notes: Dict[str, Dict[str, Any]],
    key: str,
    tag: str,
    note: str,
     *,
    display_name: str,
    country: str,
    linkedin_url: str,
    zone: str,
    market: str,
    vertical: str,
) -> Dict[str, Dict[str, Any]]:
    """Add or update a note entry for a company key."""
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    notes[key] = {
        "display_name": display_name,
        "country": country,
        "linkedin_url": linkedin_url,
        "zone": zone,
        "market": market,
        "vertical": vertical,
        "tag": tag,
        "note": note,
        "updated_at": updated_at,
    }
    return notes
