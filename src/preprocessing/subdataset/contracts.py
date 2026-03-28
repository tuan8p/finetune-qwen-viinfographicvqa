from __future__ import annotations

from dataclasses import dataclass

from ..core.contracts import DatasetSample


@dataclass(frozen=True, slots=True)
class SubdatasetSplits:
    train_samples: tuple[DatasetSample, ...]
    valid_samples: tuple[DatasetSample, ...]
    test_samples: tuple[DatasetSample, ...]
