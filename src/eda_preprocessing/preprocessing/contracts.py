from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessedSample:
    question_id: str
    split_name: str
    subdataset_split: str
    task_family: str
    image_type: str
    answer_source: str
    image_paths: tuple[str, ...]
    image_count: int
    raw_question: str
    raw_answer: str
    question_normalized: str
    answer_normalized: str
    question_lower: str
    answer_lower: str
    question_ascii_folded: str
    answer_ascii_folded: str
    answer_tokens: int
    answer_type: str
    question_type: str
    reasoning_mode: str
    cross_image_dependency: str
    diacritic_consistency: str
    tokenization_flags: tuple[str, ...]
    answer_segments: tuple[str, ...]
