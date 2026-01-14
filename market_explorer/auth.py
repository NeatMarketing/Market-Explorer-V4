from __future__ import annotations

import json
import os
import time
import hmac
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from market_explorer.passwords import hash_password, verify_password

from datetime import datetime, timedelta, timezone

# -----------------------
# Config
# -----------------------
DEFAULT_USER_DB_PATH = "data/users.json"
USER_DB_ENV = "MARKET_EXPLORER_USER_DB"

CREDENTIALS_ENV = "MARKET_EXPLORER_AUTH_JSON"       # JSON string in env
CREDENTIALS_FILE_ENV = "MARKET_EXPLORER_AUTH_FILE"  # path to JSON file in env

ENABLE_SIGNUP_ENV = "MARKET_EXPLORER_ENABLE_SIGNUP"  # "true"/"false"
DEFAULT_ENABLE_SIGNUP = True

MIN_PASSWORD_LEN = 10

# Simple lockout
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 10 * 60  # 10 minutes

# Cookies (Remember me)
COOKIE_NAME = "me_auth"
COOKIE_DAYS = 30

# Optional dependency for cookies
try:
    import extra_streamlit_components as stx  # type: ignore
except Exception:
    stx = None


# -----------------------
# Helpers
# -----------------------
def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def user_db_path() -> Path:
    return Path(os.getenv(USER_DB_ENV, DEFAULT_USER_DB_PATH))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_user_db() -> Dict[str, dict]:
    """
    Load writable users DB.
    Supports BOTH formats:
      - legacy: {"user": "<hash>"}
      - new: {"user": {"password_hash": "<hash>", "role": "Sales", "created_at": 123}}
    Returns normalized: {username: {"password_hash": ..., "role": ...}}
    """
    path = user_db_path()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    out: Dict[str, dict] = {}
    for u, rec in data.items():
        uu = normalize_username(str(u))

        # legacy string hash
        if isinstance(rec, str) and rec.strip():
            out[uu] = {"password_hash": rec, "role": "Stagiaire"}

        # new dict format
        elif isinstance(rec, dict) and rec.get("password_hash"):
            out[uu] = {
                "password_hash": str(rec.get("password_hash")),
                "role": str(rec.get("role", "Stagiaire")),
                "created_at": rec.get("created_at"),
            }

    return out


def save_user_db(users: Dict[str, dict]) -> None:
    """Atomic-ish write to users DB."""
    path = user_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(users, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)



def load_credentials() -> Dict[str, Any]:
    """
    Load static credentials (read-only).
    This function MUST NEVER crash the app.
    If credentials are malformed, they are ignored.
    """
    data = {}

    # 1) Try Streamlit secrets
    try:
        secrets_auth = st.secrets.get("auth", {})
        if isinstance(secrets_auth, dict):
            data = secrets_auth
    except Exception:
        data = {}

    # 2) Normalize format
    normalized: Dict[str, Any] = {}

    for u, rec in data.items():
        username = normalize_username(str(u))

        # legacy format: username -> hash
        if isinstance(rec, str) and rec.strip():
            normalized[username] = rec

        # new format: username -> {password_hash, role}
        elif isinstance(rec, dict) and "password_hash" in rec:
            normalized[username] = {
                "password_hash": str(rec.get("password_hash")),
                "role": str(rec.get("role", "Stagiaire")),
            }

        # everything else is silently ignored

    return normalized



def merged_credentials() -> Dict[str, dict]:
    """
    Combine writable users DB + static creds.
    Output normalized: username -> {"password_hash": ..., "role": ...}
    """
    db = load_user_db()               # dict values (new) or normalized from legacy
    static_raw = load_credentials()   # may return strings or dicts

    static: Dict[str, dict] = {}
    for u, rec in static_raw.items():
        if isinstance(rec, str) and rec.strip():
            static[u] = {"password_hash": rec, "role": "Stagiaire"}
        elif isinstance(rec, dict) and rec.get("password_hash"):
            static[u] = {"password_hash": str(rec.get("password_hash")), "role": str(rec.get("role", "Stagiaire"))}

    merged = dict(db)
    merged.update(static)  # static wins
    return merged


def signup_enabled() -> bool:
    return _as_bool(os.getenv(ENABLE_SIGNUP_ENV), DEFAULT_ENABLE_SIGNUP)


# -----------------------
# Lockout
# -----------------------
def _is_locked_out(now: float) -> bool:
    locked_until = float(st.session_state.get("auth_locked_until", 0) or 0)
    return now < locked_until


def _lockout_remaining(now: float) -> int:
    locked_until = float(st.session_state.get("auth_locked_until", 0) or 0)
    return int(max(0, locked_until - now))


def _register_failed_attempt(now: float) -> None:
    attempts = int(st.session_state.get("auth_failed_attempts", 0) or 0) + 1
    st.session_state["auth_failed_attempts"] = attempts
    if attempts >= MAX_FAILED_ATTEMPTS:
        st.session_state["auth_locked_until"] = now + LOCKOUT_SECONDS


# -----------------------
# Cookies (Remember me)
# -----------------------
def _cookie_manager():
    if stx is None:
        return None

    # Create exactly once per session + unique key
    if "_cookie_manager_instance" not in st.session_state:
        st.session_state["_cookie_manager_instance"] = stx.CookieManager(key="me_cookie_manager")
    return st.session_state["_cookie_manager_instance"]


def _secret() -> str:
    # En prod: mets une vraie valeur via st.secrets["AUTH_SECRET"] ou env AUTH_SECRET
    return st.secrets.get("AUTH_SECRET", os.getenv("AUTH_SECRET", "dev-secret-change-me"))


def _sign(username: str) -> str:
    return hmac.new(_secret().encode("utf-8"), username.encode("utf-8"), hashlib.sha256).hexdigest()


def set_remember_cookie(username: str) -> None:
    cm = _cookie_manager()
    if cm is None:
        st.session_state["_remember_fallback"] = username
        return

    expires_at = datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS)
    cm.set(COOKIE_NAME, f"{username}:{_sign(username)}", expires_at=expires_at)

def clear_remember_cookie() -> None:
    cm = _cookie_manager()
    if cm is None:
        st.session_state.pop("_remember_fallback", None)
        return
    cm.delete(COOKIE_NAME)


def get_remembered_username() -> str | None:
    cm = _cookie_manager()
    if cm is None:
        return st.session_state.get("_remember_fallback")

    raw = cm.get(COOKIE_NAME)
    if not raw or ":" not in raw:
        return None

    username, sig = raw.split(":", 1)
    username = normalize_username(username)

    if hmac.compare_digest(sig, _sign(username)):
        return username
    return None


# -----------------------
# Auth API (used by pages)
# -----------------------
def login_required() -> bool:
    """
    Returns True if session is authenticated.
    If not, tries remember-me cookie to restore session.
    """
    if st.session_state.get("is_auth"):
        return True

    u = get_remembered_username()
    if not u:
        return False

    creds = merged_credentials()
    rec = creds.get(u)
    if not rec:
        return False

    st.session_state["is_auth"] = True
    st.session_state["profile"] = u
    st.session_state["role"] = rec.get("role", "Stagiaire")
    return True


def logout() -> None:
    clear_remember_cookie()
    # On garde éventuellement quelques clés non-auth si tu veux, mais simple:
    for k in ["is_auth", "profile", "role", "auth_failed_attempts", "auth_locked_until"]:
        st.session_state.pop(k, None)


def require_auth() -> None:
    """
    Call in pages that require login.
    """
    if not login_required():
        st.error("You must be signed in to access this page.")
        st.stop()


def attempt_login(username: str, password: str, credentials: Dict[str, Any] | None = None, remember: bool = False) -> Tuple[bool, str]:
    now = time.time()

    if _is_locked_out(now):
        return False, f"Too many failed attempts. Try again in {_lockout_remaining(now)}s."

    user = normalize_username(username)
    if not user:
        return False, "Please enter a username."

    if not password:
        return False, "Please enter a password."

    creds = credentials if credentials is not None else merged_credentials()
    rec = creds.get(user)

    # compat: if someone passed legacy dict directly
    if isinstance(rec, str):
        rec = {"password_hash": rec, "role": "Stagiaire"}

    if not rec:
        _register_failed_attempt(now)
        return False, "Invalid username or password."

    stored_hash = str(rec.get("password_hash", ""))
    if not stored_hash or not verify_password(password, stored_hash):
        _register_failed_attempt(now)
        return False, "Invalid username or password."

    # success → reset lockout counters
    st.session_state["auth_failed_attempts"] = 0
    st.session_state["auth_locked_until"] = 0

    # set session
    st.session_state["is_auth"] = True
    st.session_state["profile"] = user
    st.session_state["role"] = str(rec.get("role", "Stagiaire"))

    # remember
    if remember:
        set_remember_cookie(user)
    else:
        clear_remember_cookie()

    return True, "Signed in."


def user_exists(username: str, credentials: Dict[str, Any] | None = None) -> bool:
    user = normalize_username(username)
    creds = credentials if credentials is not None else merged_credentials()
    return user in creds


def create_user(username: str, password: str, role: str = "Stagiaire", credentials=None) -> Tuple[bool, str]:
    """Create a new user in the writable DB (MVP)."""
    if not signup_enabled():
        return False, "Account creation is disabled."

    user = normalize_username(username)
    if not user:
        return False, "Please enter a username."
    if not password or len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."

    creds = credentials if credentials is not None else merged_credentials()
    if user in creds:
        return False, "This username is already taken."

    users = load_user_db()
    if user in users:
        return False, "This username is already taken."

    users[user] = {
        "password_hash": hash_password(password),
        "role": role or "Stagiaire",
        "created_at": int(time.time()),
    }

    try:
        save_user_db(users)
    except Exception:
        return False, "Could not save the account. Check file permissions / deployment storage."

    return True, "Account created. You can now sign in."

