from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def whitespace_tokens(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return normalized.split(" ")


def lowercase_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def contains_diacritics(text: str) -> bool:
    return remove_diacritics(text) != text


def has_combining_marks(text: str) -> bool:
    return any(unicodedata.category(char) == "Mn" for char in unicodedata.normalize("NFD", text))


def token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in whitespace_tokens(lowercase_text(left)) if token}
    right_tokens = {token for token in whitespace_tokens(lowercase_text(right)) if token}
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def safe_slug(value: str) -> str:
    lowered = remove_diacritics(value).lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "figure"

