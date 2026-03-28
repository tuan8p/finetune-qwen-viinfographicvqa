from __future__ import annotations

from ..core.contracts import DatasetSample
from ..core.heuristics import (
    classify_answer_type,
    classify_cross_image_dependency,
    classify_diacritic_consistency,
    classify_question_type,
    classify_reasoning_mode,
    detect_tokenization_flags,
)
from .answer_processor import normalize_answer_by_type, split_answer_segments
from .contracts import PreprocessedSample
from .sample_filter import count_answer_tokens, should_keep_sample
from .text_normalizer import ascii_fold_text, lowercase_normalized_text, normalize_text


def _build_normalized_sample(sample: DatasetSample, question: str, answer: str) -> DatasetSample:
    return type(sample)(
        question_id=sample.question_id,
        split_name=sample.split_name,
        task_family=sample.task_family,
        image_type=sample.image_type,
        answer_source=sample.answer_source,
        question=question,
        answer=answer,
        image_paths=sample.image_paths,
        element=sample.element,
    )


def preprocess_sample(sample: DatasetSample, subdataset_split: str) -> PreprocessedSample | None:
    question_normalized = normalize_text(sample.question)
    answer_base = normalize_text(sample.answer)
    answer_tokens = count_answer_tokens(answer_base)
    if not should_keep_sample(answer_tokens):
        return None

    normalized_sample = _build_normalized_sample(sample, question_normalized, answer_base)
    answer_type = classify_answer_type(normalized_sample)
    answer_normalized = normalize_answer_by_type(answer_base, answer_type)
    normalized_sample = _build_normalized_sample(sample, question_normalized, answer_normalized)

    tokenization_flags = tuple(detect_tokenization_flags(f"{question_normalized} || {answer_normalized}"))

    return PreprocessedSample(
        question_id=sample.question_id,
        split_name=sample.split_name,
        subdataset_split=subdataset_split,
        task_family=sample.task_family,
        image_type=sample.image_type,
        answer_source=sample.answer_source,
        element=sample.element,
        image_paths=sample.image_paths,
        image_count=len(sample.image_paths),
        raw_question=sample.question,
        raw_answer=sample.answer,
        question_normalized=question_normalized,
        answer_normalized=answer_normalized,
        question_lower=lowercase_normalized_text(question_normalized),
        answer_lower=lowercase_normalized_text(answer_normalized),
        question_ascii_folded=ascii_fold_text(question_normalized),
        answer_ascii_folded=ascii_fold_text(answer_normalized),
        answer_tokens=answer_tokens,
        answer_type=answer_type,
        question_type=classify_question_type(normalized_sample),
        reasoning_mode=classify_reasoning_mode(normalized_sample),
        cross_image_dependency=classify_cross_image_dependency(normalized_sample),
        diacritic_consistency=classify_diacritic_consistency(normalized_sample),
        tokenization_flags=tokenization_flags,
        answer_segments=split_answer_segments(answer_normalized),
    )


def preprocess_samples(
    samples: list[DatasetSample] | tuple[DatasetSample, ...],
    subdataset_split: str,
) -> list[PreprocessedSample]:
    processed_samples: list[PreprocessedSample] = []
    for sample in samples:
        processed = preprocess_sample(sample, subdataset_split=subdataset_split)
        if processed is not None:
            processed_samples.append(processed)
    return processed_samples
