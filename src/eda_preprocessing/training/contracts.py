from __future__ import annotations

from dataclasses import dataclass

from eda_preprocessing.preprocessing.contracts import PreprocessedSample


DATA_MODES = (
    "single",
    "multi",
    "single_and_multi",
)


@dataclass(frozen=True, slots=True)
class FinetuneDatasetBundle:
    data_mode: str
    train_dataset: object
    valid_dataset: object
    test_dataset: object
    extra_test_dataset: object | None = None
    extra_test_name: str | None = None


@dataclass(frozen=True, slots=True)
class FinetuneDataLoaderBundle:
    data_mode: str
    train_loader: object
    valid_loader: object
    test_loader: object
    extra_test_loader: object | None = None
    extra_test_name: str | None = None
