from __future__ import annotations

import unicodedata

from ..core.text_utils import lowercase_text, normalize_whitespace, remove_diacritics


def normalize_text(text: str) -> str:
    return normalize_whitespace(unicodedata.normalize("NFC", text))


def lowercase_normalized_text(text: str) -> str:
    return lowercase_text(normalize_text(text))


def ascii_fold_text(text: str) -> str:
    return remove_diacritics(normalize_text(text))
