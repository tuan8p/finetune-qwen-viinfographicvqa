from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    class Dataset:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls

from eda_preprocessing.core.contracts import DatasetSample
from eda_preprocessing.io.dataset_loader import DatasetLoader, SUPPORTED_SPLITS
from eda_preprocessing.preprocessing import PreprocessedSample, preprocess_samples
from eda_preprocessing.subdataset import build_subdataset_splits


class PreprocessedTrainingDataset(Dataset[PreprocessedSample]):
    def __init__(
        self,
        samples: Sequence[DatasetSample] | Sequence[PreprocessedSample],
        subdataset_split: str,
    ) -> None:
        self.subdataset_split = subdataset_split
        self.samples = tuple(self._coerce_samples(samples, subdataset_split))

    def _coerce_samples(
        self,
        samples: Sequence[DatasetSample] | Sequence[PreprocessedSample],
        subdataset_split: str,
    ) -> list[PreprocessedSample]:
        materialized = list(samples)
        if not materialized:
            return []

        if isinstance(materialized[0], PreprocessedSample):
            return [sample for sample in materialized if isinstance(sample, PreprocessedSample)]
        return preprocess_samples(materialized, subdataset_split=subdataset_split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreprocessedSample:
        return self.samples[index]


def build_finetune_datasets(
    dataset_root: str | Path,
    use_subdataset: bool = True,
    seed: int = 42,
) -> tuple[PreprocessedTrainingDataset, PreprocessedTrainingDataset, PreprocessedTrainingDataset]:
    resolved_root = Path(dataset_root).resolve()
    loader = DatasetLoader(resolved_root)
    raw_samples = loader.load_splits(SUPPORTED_SPLITS)
    logical_splits = build_subdataset_splits(raw_samples, enabled=use_subdataset, seed=seed)

    train_dataset = PreprocessedTrainingDataset(logical_splits.train_samples, subdataset_split="train")
    valid_dataset = PreprocessedTrainingDataset(logical_splits.valid_samples, subdataset_split="valid")
    test_dataset = PreprocessedTrainingDataset(logical_splits.test_samples, subdataset_split="test")
    return train_dataset, valid_dataset, test_dataset
