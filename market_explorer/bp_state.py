"""
Business plan state storage per profile.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


BP_DIR = Path("bp_saves")


def _profile_path(profile: str) -> Path:
    BP_DIR.mkdir(exist_ok=True)
    return BP_DIR / f"{profile}.json"


def load_bp_state(profile: str) -> Dict[str, Any]:
    """Load the saved BP state for a profile."""
    path = _profile_path(profile)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_bp_state(profile: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Save the BP state for a profile, returning the updated payload."""
    updated = dict(data)
    updated["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = _profile_path(profile)
    path.write_text(json.dumps(updated, indent=2))
    return updated


def clear_bp_state(profile: str) -> None:
    """Remove the BP state for a profile."""
    path = _profile_path(profile)
    if path.exists():
        path.unlink()