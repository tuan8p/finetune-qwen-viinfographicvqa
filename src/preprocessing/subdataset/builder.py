from __future__ import annotations

import random
from collections import defaultdict

from ..core.contracts import DatasetSample
from .contracts import SubdatasetSplits


TRAIN_SPLITS = ("single_train", "multi_train")
TEST_SPLITS = ("single_test", "multi_test")


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0 inclusive, got {value}")


def _fraction_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return total
    return int(total * fraction)


def _sample_without_replacement(
    items: list[DatasetSample],
    fraction: float,
    rng: random.Random,
) -> list[DatasetSample]:
    sample_count = min(len(items), _fraction_count(len(items), fraction))
    if sample_count == len(items):
        return list(items)
    return rng.sample(items, sample_count)


def _split_train_valid(
    items: list[DatasetSample],
    valid_fraction: float,
    rng: random.Random,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    if len(items) <= 1 or valid_fraction <= 0.0:
        return list(items), []

    valid_count = _fraction_count(len(items), valid_fraction)
    valid_count = min(valid_count, len(items) - 1)
    if valid_count <= 0:
        return list(items), []

    valid_indices = set(rng.sample(range(len(items)), valid_count))
    train_items = [sample for index, sample in enumerate(items) if index not in valid_indices]
    valid_items = [sample for index, sample in enumerate(items) if index in valid_indices]
    return train_items, valid_items


def build_subdataset_splits(
    samples: list[DatasetSample] | tuple[DatasetSample, ...],
    enabled: bool = True,
    seed: int = 42,
    train_fraction: float = 0.1,
    valid_fraction_within_sampled: float = 0.2,
    valid_fraction_when_disabled: float = 0.2,
) -> SubdatasetSplits:
    _validate_fraction("train_fraction", train_fraction)
    _validate_fraction("valid_fraction_within_sampled", valid_fraction_within_sampled)
    _validate_fraction("valid_fraction_when_disabled", valid_fraction_when_disabled)

    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.split_name].append(sample)

    rng = random.Random(seed)

    if not enabled:
        train_samples: list[DatasetSample] = []
        valid_samples: list[DatasetSample] = []
        for split_name in TRAIN_SPLITS:
            train_chunk, valid_chunk = _split_train_valid(grouped[split_name], valid_fraction_when_disabled, rng)
            train_samples.extend(train_chunk)
            valid_samples.extend(valid_chunk)

        return SubdatasetSplits(
            train_samples=tuple(train_samples),
            valid_samples=tuple(valid_samples),
            test_samples=tuple(grouped["single_test"] + grouped["multi_test"]),
        )

    train_samples: list[DatasetSample] = []
    valid_samples: list[DatasetSample] = []

    for split_name in TRAIN_SPLITS:
        sampled = _sample_without_replacement(grouped[split_name], train_fraction, rng)
        train_chunk, valid_chunk = _split_train_valid(sampled, valid_fraction_within_sampled, rng)
        train_samples.extend(train_chunk)
        valid_samples.extend(valid_chunk)

    test_samples: list[DatasetSample] = []
    for split_name in TEST_SPLITS:
        test_samples.extend(grouped[split_name])

    return SubdatasetSplits(
        train_samples=tuple(train_samples),
        valid_samples=tuple(valid_samples),
        test_samples=tuple(test_samples),
    )
