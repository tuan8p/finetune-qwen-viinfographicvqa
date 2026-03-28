from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSample:
    question_id: str
    split_name: str
    task_family: str
    image_type: str
    answer_source: str
    question: str
    answer: str
    image_paths: tuple[str, ...]
    element: str | None = None


@dataclass(frozen=True, slots=True)
class SingleSample(DatasetSample):
    """Typed single-image sample."""


@dataclass(frozen=True, slots=True)
class MultiSample(DatasetSample):
    """Typed multi-image sample."""
