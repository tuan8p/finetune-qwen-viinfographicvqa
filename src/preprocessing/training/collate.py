from __future__ import annotations

from collections.abc import Callable

from ..preprocessing.contracts import PreprocessedSample


def build_collate_fn() -> Callable[[list[PreprocessedSample]], dict[str, object]]:
    def collate_fn(batch: list[PreprocessedSample]) -> dict[str, object]:
        return {
            "question_id": [sample.question_id for sample in batch],
            "split_name": [sample.split_name for sample in batch],
            "subdataset_split": [sample.subdataset_split for sample in batch],
            "task_family": [sample.task_family for sample in batch],
            "image_type": [sample.image_type for sample in batch],
            "answer_source": [sample.answer_source for sample in batch],
            "image_paths": [sample.image_paths for sample in batch],
            "image_count": [sample.image_count for sample in batch],
            "raw_question": [sample.raw_question for sample in batch],
            "raw_answer": [sample.raw_answer for sample in batch],
            "question_text": [sample.question_normalized for sample in batch],
            "answer_text": [sample.answer_normalized for sample in batch],
            "question_lower": [sample.question_lower for sample in batch],
            "answer_lower": [sample.answer_lower for sample in batch],
            "question_ascii_folded": [sample.question_ascii_folded for sample in batch],
            "answer_ascii_folded": [sample.answer_ascii_folded for sample in batch],
            "answer_tokens": [sample.answer_tokens for sample in batch],
            "answer_type": [sample.answer_type for sample in batch],
            "question_type": [sample.question_type for sample in batch],
            "reasoning_mode": [sample.reasoning_mode for sample in batch],
            "cross_image_dependency": [sample.cross_image_dependency for sample in batch],
            "diacritic_consistency": [sample.diacritic_consistency for sample in batch],
            "tokenization_flags": [sample.tokenization_flags for sample in batch],
            "answer_segments": [sample.answer_segments for sample in batch],
        }

    return collate_fn
