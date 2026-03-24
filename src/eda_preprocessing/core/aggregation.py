from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from eda_preprocessing.core.contracts import DatasetSample, ImageRecord


@dataclass(frozen=True, slots=True)
class Scope:
    level: str
    name: str


def build_sample_scopes(samples: Iterable[DatasetSample]) -> dict[Scope, list[DatasetSample]]:
    materialized = list(samples)
    scoped: dict[Scope, list[DatasetSample]] = {Scope("overall", "overall"): materialized}
    by_task: dict[str, list[DatasetSample]] = defaultdict(list)
    by_split: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in materialized:
        by_task[sample.task_family].append(sample)
        by_split[sample.split_name].append(sample)
    for key, items in by_task.items():
        scoped[Scope("task_family", key)] = items
    for key, items in by_split.items():
        scoped[Scope("split", key)] = items
    return scoped


def build_image_scopes(records: Iterable[ImageRecord]) -> dict[Scope, list[ImageRecord]]:
    materialized = list(records)
    scoped: dict[Scope, list[ImageRecord]] = {Scope("overall", "overall"): materialized}
    by_task: dict[str, list[ImageRecord]] = defaultdict(list)
    by_split: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in materialized:
        for task_family in record.task_families:
            by_task[task_family].append(record)
        for split_name in record.split_names:
            by_split[split_name].append(record)
    for key, items in by_task.items():
        scoped[Scope("task_family", key)] = items
    for key, items in by_split.items():
        scoped[Scope("split", key)] = items
    return scoped

