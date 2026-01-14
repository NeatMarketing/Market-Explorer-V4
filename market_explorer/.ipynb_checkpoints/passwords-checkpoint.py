from __future__ import annotations

import base64
import hashlib
import hmac
import os
from typing import Tuple


HASH_PREFIX = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 390_000


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{HASH_PREFIX}${iterations}${_b64encode(salt)}${_b64encode(dk)}"


def _parse_hash(stored: str) -> Tuple[int, bytes, bytes]:
    parts = stored.split("$")
    if len(parts) != 4 or parts[0] != HASH_PREFIX:
        raise ValueError("Unsupported password hash format.")
    iterations = int(parts[1])
    salt = _b64decode(parts[2])
    digest = _b64decode(parts[3])
    return iterations, salt, digest


def verify_password(password: str, stored: str) -> bool:
    iterations, salt, digest = _parse_hash(stored)
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(computed, digest)