from __future__ import annotations

import json
import random
from pathlib import Path

from eda_preprocessing.core.config import PipelineConfig
from eda_preprocessing.core.contracts import DatasetSample, MultiSample, SingleSample


SUPPORTED_SPLITS = (
    "single_train",
    "single_test",
    "multi_train",
    "multi_test",
)


class DatasetLoader:
    """Load local ViInfographicVQA JSON splits into typed samples."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.data_root = dataset_root / "data"

    def load(self, config: PipelineConfig) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        rng = random.Random(config.seed)

        for split_name in config.splits:
            raw_samples = self._load_split(split_name)
            if config.mode == "sample":
                sample_size = min(len(raw_samples), config.sample_size)
                raw_samples = rng.sample(raw_samples, sample_size)
            samples.extend(raw_samples)
        return samples

    def load_splits(self, split_names: tuple[str, ...]) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for split_name in split_names:
            samples.extend(self._load_split(split_name))
        return samples

    def _load_split(self, split_name: str) -> list[DatasetSample]:
        if split_name not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split_name}")

        split_path = self.data_root / f"{split_name}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        raw_items = json.loads(split_path.read_text(encoding="utf-8"))
        task_family = "single" if split_name.startswith("single") else "multi"
        typed_samples: list[DatasetSample] = []
        for item in raw_items:
            base_kwargs = {
                "question_id": str(item["question_id"]),
                "split_name": split_name,
                "task_family": task_family,
                "image_type": str(item["image_type"]),
                "answer_source": str(item["answer_source"]),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
            if task_family == "single":
                typed_samples.append(
                    SingleSample(
                        **base_kwargs,
                        image_paths=(str(item["image_path"]),),
                        element=str(item.get("element")) if item.get("element") is not None else None,
                    )
                )
            else:
                typed_samples.append(
                    MultiSample(
                        **base_kwargs,
                        image_paths=tuple(str(path) for path in item["image_paths"]),
                    )
                )
        return typed_samples
