from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    class Dataset:  # type: ignore[no-redef]
        def __class_getitem__(cls, _item):
            return cls

from ..core.contracts import DatasetSample
from ..io.dataset_loader import DatasetLoader, SUPPORTED_SPLITS
from ..preprocessing import PreprocessedSample, preprocess_samples
from ..subdataset import build_subdataset_splits
from .contracts import DATA_MODES, FinetuneDatasetBundle


class PreprocessedTrainingDataset(Dataset[PreprocessedSample]):
    def __init__(
        self,
        samples: Sequence[DatasetSample] | Sequence[PreprocessedSample],
        subdataset_split: str,
        *,
        filter_answers_over_20_tokens: bool = True,
    ) -> None:
        self.subdataset_split = subdataset_split
        self.samples = tuple(
            self._coerce_samples(samples, subdataset_split, filter_answers_over_20_tokens)
        )

    def _coerce_samples(
        self,
        samples: Sequence[DatasetSample] | Sequence[PreprocessedSample],
        subdataset_split: str,
        filter_answers_over_20_tokens: bool,
    ) -> list[PreprocessedSample]:
        materialized = list(samples)
        if not materialized:
            return []

        if isinstance(materialized[0], PreprocessedSample):
            return [sample for sample in materialized if isinstance(sample, PreprocessedSample)]
        return preprocess_samples(
            materialized,
            subdataset_split=subdataset_split,
            filter_answers_over_20_tokens=filter_answers_over_20_tokens,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreprocessedSample:
        return self.samples[index]


def _filter_samples_by_split(
    samples: Sequence[DatasetSample],
    allowed_splits: tuple[str, ...],
) -> tuple[DatasetSample, ...]:
    allowed = set(allowed_splits)
    return tuple(sample for sample in samples if sample.split_name in allowed)


def _build_dataset_bundle_from_logical_splits(
    logical_splits,
    data_mode: str,
    *,
    filter_answers_over_20_tokens: bool = True,
) -> FinetuneDatasetBundle:
    if data_mode not in DATA_MODES:
        raise ValueError(f"Unsupported data_mode: {data_mode}. Expected one of {DATA_MODES}")

    extra_test_samples: tuple[DatasetSample, ...] | None = None
    extra_test_name: str | None = None

    if data_mode == "single":
        train_samples = _filter_samples_by_split(logical_splits.train_samples, ("single_train",))
        valid_samples = _filter_samples_by_split(logical_splits.valid_samples, ("single_train",))
        test_samples = _filter_samples_by_split(logical_splits.test_samples, ("single_test",))
    elif data_mode == "multi":
        train_samples = _filter_samples_by_split(logical_splits.train_samples, ("multi_train",))
        valid_samples = _filter_samples_by_split(logical_splits.valid_samples, ("multi_train",))
        test_samples = _filter_samples_by_split(logical_splits.test_samples, ("multi_test",))
    else:
        train_samples = tuple(logical_splits.train_samples)
        valid_samples = tuple(logical_splits.valid_samples)
        test_samples = tuple(logical_splits.test_samples)
        extra_test_samples = _filter_samples_by_split(logical_splits.test_samples, ("single_test",))
        extra_test_name = "test_single"

    train_dataset = PreprocessedTrainingDataset(
        train_samples,
        subdataset_split="train",
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
    )
    valid_dataset = PreprocessedTrainingDataset(
        valid_samples,
        subdataset_split="valid",
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
    )
    test_dataset = PreprocessedTrainingDataset(
        test_samples,
        subdataset_split="test",
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
    )
    extra_test_dataset = (
        PreprocessedTrainingDataset(
            extra_test_samples,
            subdataset_split="test",
            filter_answers_over_20_tokens=filter_answers_over_20_tokens,
        )
        if extra_test_samples is not None
        else None
    )

    return FinetuneDatasetBundle(
        data_mode=data_mode,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        extra_test_dataset=extra_test_dataset,
        extra_test_name=extra_test_name,
    )


def build_finetune_dataset_bundle(
    dataset_root: str | Path,
    use_subdataset: bool = True,
    seed: int = 42,
    data_mode: str = "single_and_multi",
    *,
    filter_answers_over_20_tokens: bool = True,
) -> FinetuneDatasetBundle:
    resolved_root = Path(dataset_root).resolve()
    loader = DatasetLoader(resolved_root)
    raw_samples = loader.load_splits(SUPPORTED_SPLITS)
    logical_splits = build_subdataset_splits(raw_samples, enabled=use_subdataset, seed=seed)
    return _build_dataset_bundle_from_logical_splits(
        logical_splits,
        data_mode=data_mode,
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
    )


def build_finetune_datasets(
    dataset_root: str | Path,
    use_subdataset: bool = True,
    seed: int = 42,
    data_mode: str = "single_and_multi",
    *,
    filter_answers_over_20_tokens: bool = True,
) -> tuple[PreprocessedTrainingDataset, PreprocessedTrainingDataset, PreprocessedTrainingDataset]:
    bundle = build_finetune_dataset_bundle(
        dataset_root=dataset_root,
        use_subdataset=use_subdataset,
        seed=seed,
        data_mode=data_mode,
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
    )
    return bundle.train_dataset, bundle.valid_dataset, bundle.test_dataset
