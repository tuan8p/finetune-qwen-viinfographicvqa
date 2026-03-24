from __future__ import annotations

import re


SEMICOLON_RE = re.compile(r"\s*;\s*")
NUMBER_OPERATOR_RE = re.compile(r"\s*([%/:+\-])\s*")


def normalize_answer_by_type(answer: str, answer_type: str) -> str:
    normalized = SEMICOLON_RE.sub("; ", answer).strip()
    if answer_type != "number":
        return normalized
    return NUMBER_OPERATOR_RE.sub(r"\1", normalized)


def split_answer_segments(answer: str) -> tuple[str, ...]:
    if ";" not in answer:
        return (answer,)
    segments = tuple(segment.strip() for segment in answer.split(";") if segment.strip())
    return segments or (answer,)
