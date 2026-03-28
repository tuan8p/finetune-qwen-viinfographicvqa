from __future__ import annotations

import re
import unicodedata

from .contracts import DatasetSample
from .rules import HEURISTIC_RULES
from .text_utils import (
    contains_diacritics,
    has_combining_marks,
    lowercase_text,
    remove_diacritics,
    token_overlap_ratio,
    whitespace_tokens,
)


NUMBER_RE = re.compile(str(HEURISTIC_RULES["answer_type"]["number_like_pattern"]))
NUMERIC_UNIT_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["numeric_unit_pattern"]), re.IGNORECASE)
SLASH_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["slash_pattern"]))
HYPHEN_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["hyphen_pattern"]))
ABBREVIATION_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["abbreviation_pattern"]))
DATE_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["date_pattern"]), re.IGNORECASE)
TIME_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["time_pattern"]), re.IGNORECASE)
CURRENCY_RE = re.compile(str(HEURISTIC_RULES["tokenization"]["currency_pattern"]), re.IGNORECASE)
KNOWN_ABBREVIATIONS = {str(item).lower() for item in HEURISTIC_RULES["tokenization"]["known_abbreviations"]}


def _normalized_match_text(text: str) -> tuple[str, str]:
    lowered = lowercase_text(text)
    ascii_lowered = remove_diacritics(lowered).lower()
    return lowered, ascii_lowered


def _contains_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    lowered, ascii_lowered = _normalized_match_text(text)
    for cue in cues:
        lowered_cue, ascii_cue = _normalized_match_text(cue)
        if lowered_cue in lowered or ascii_cue in ascii_lowered:
            return True
    return False


def _has_known_abbreviation(text: str) -> bool:
    tokens = ABBREVIATION_RE.findall(text)
    for token in tokens:
        lowered = token.lower()
        if lowered in KNOWN_ABBREVIATIONS:
            return True
        if token.isupper() and len(token) >= 2:
            return True
        if any(char.isdigit() for char in token) and any(char.isalpha() for char in token) and lowered in KNOWN_ABBREVIATIONS:
            return True
    return False


def classify_answer_type(sample: DatasetSample) -> str:
    answer = lowercase_text(sample.answer)
    answer_ascii = remove_diacritics(answer)
    yes_no_tokens = set(HEURISTIC_RULES["answer_type"]["yes_no_tokens"])
    entity_cues = tuple(str(item) for item in HEURISTIC_RULES["answer_type"]["entity_question_cues"])
    span_sources = set(HEURISTIC_RULES["answer_type"]["span_sources"])

    if answer_ascii in yes_no_tokens:
        return "other"
    if NUMBER_RE.fullmatch(answer) or NUMBER_RE.fullmatch(answer.replace(";", "")):
        return "number"
    if _contains_any_cue(sample.question, entity_cues) and len(whitespace_tokens(answer)) <= 6:
        return "entity_like"
    if sample.answer_source in span_sources or len(whitespace_tokens(answer)) > 3:
        return "text_span_like"
    return "other"


def classify_question_type(sample: DatasetSample) -> str:
    ordered_rules = [
        ("compare", HEURISTIC_RULES["question_type"]["compare_cues"]),
        ("how_many", HEURISTIC_RULES["question_type"]["how_many_cues"]),
        ("when", HEURISTIC_RULES["question_type"]["when_cues"]),
        ("which", HEURISTIC_RULES["question_type"]["which_cues"]),
        ("what", HEURISTIC_RULES["question_type"]["what_cues"]),
        ("list", HEURISTIC_RULES["question_type"]["list_cues"]),
        ("count", HEURISTIC_RULES["question_type"]["count_cues"]),
    ]
    for label, cues in ordered_rules:
        if _contains_any_cue(sample.question, tuple(str(cue) for cue in cues)):
            return label
    return "other"


def classify_reasoning_mode(sample: DatasetSample) -> str:
    reasoning_cues = tuple(str(item) for item in HEURISTIC_RULES["question_type"]["reasoning_cues"])
    reasoning_sources = set(HEURISTIC_RULES["question_type"]["reasoning_sources"])
    if sample.answer_source in reasoning_sources:
        return "reasoning"
    if _contains_any_cue(sample.question, reasoning_cues):
        return "reasoning"
    return "lookup"


def classify_cross_image_dependency(sample: DatasetSample) -> str:
    comparison_cues = tuple(str(item) for item in HEURISTIC_RULES["cross_image_dependency"]["comparison_cues"])
    aggregation_cues = tuple(str(item) for item in HEURISTIC_RULES["cross_image_dependency"]["aggregation_cues"])
    if sample.answer_source == "Cross-Image Synthesis" or _contains_any_cue(sample.question, comparison_cues):
        return "comparison"
    if sample.answer_source == "Multi-Image Spans":
        return "aggregation"
    if sample.answer_source == "Non-Span":
        if _contains_any_cue(sample.question, comparison_cues):
            return "comparison"
        return "aggregation"
    if _contains_any_cue(sample.question, aggregation_cues):
        return "aggregation"
    return "independent_like"


def analyze_diacritics(text: str) -> dict[str, bool | str]:
    nfc_text = unicodedata.normalize("NFC", text)
    nfd_text = unicodedata.normalize("NFD", text)
    has_marks = contains_diacritics(text)
    ascii_folded = remove_diacritics(text)
    has_letters = any(char.isalpha() for char in text)
    ascii_only_letters = has_letters and ascii_folded == text
    normalization_suspect = has_combining_marks(text) and text != nfc_text
    is_nfc = text == nfc_text
    return {
        "has_diacritics": has_marks,
        "has_letters": has_letters,
        "ascii_only_letters": ascii_only_letters,
        "normalization_suspect": normalization_suspect,
        "is_nfc": is_nfc,
        "nfc_changed": not is_nfc,
        "nfd_changed": text != nfd_text,
    }


def classify_diacritic_consistency(sample: DatasetSample) -> str:
    question_info = analyze_diacritics(sample.question)
    answer_info = analyze_diacritics(sample.answer)
    if bool(question_info["normalization_suspect"]) or bool(answer_info["normalization_suspect"]):
        return "normalization_suspect"
    if bool(question_info["has_letters"]) and not bool(answer_info["has_letters"]):
        if bool(question_info["has_diacritics"]):
            return "question_text_with_diacritics"
        if bool(question_info["ascii_only_letters"]):
            return "question_text_without_diacritics"
    if not bool(question_info["has_letters"]) and bool(answer_info["has_letters"]):
        if bool(answer_info["has_diacritics"]):
            return "answer_text_with_diacritics"
        if bool(answer_info["ascii_only_letters"]):
            return "answer_text_without_diacritics"
    if bool(question_info["has_diacritics"]) and bool(answer_info["has_diacritics"]):
        return "both_with_diacritics"
    if bool(question_info["ascii_only_letters"]) and bool(answer_info["ascii_only_letters"]):
        return "both_without_diacritics"
    if bool(question_info["ascii_only_letters"]) and bool(answer_info["has_diacritics"]):
        return "question_missing_diacritics"
    if bool(question_info["has_diacritics"]) and bool(answer_info["ascii_only_letters"]):
        return "answer_missing_diacritics"
    return "mixed_form"


def detect_tokenization_flags(text: str) -> list[str]:
    lowered = lowercase_text(text)
    flags: list[str] = []
    if NUMERIC_UNIT_RE.search(lowered):
        flags.append("numeric_unit_compact")
    if SLASH_RE.search(lowered):
        flags.append("slash_form")
    if HYPHEN_RE.search(lowered):
        flags.append("hyphen_form")
    if ";" in lowered:
        flags.append("multi_span_delimiter")
    if DATE_RE.search(lowered) or TIME_RE.search(lowered) or CURRENCY_RE.search(lowered):
        flags.append("date_time_currency")

    if _has_known_abbreviation(text):
        flags.append("abbreviation")
    return sorted(set(flags))
