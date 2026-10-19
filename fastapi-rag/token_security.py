"""Helpers for protecting bearer credentials at rest."""

import hashlib


def hash_token(token_value: str) -> str:
    """Return a one-way SHA-256 digest suitable for token lookup."""
    return hashlib.sha256(token_value.encode("utf-8")).hexdigest()