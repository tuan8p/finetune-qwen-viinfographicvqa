from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.preprocessing.training import FinetuneDatasetBundle, build_finetune_dataset_bundle

from src.inference_core.config import InferenceConfig


@dataclass(frozen=True, slots=True)
class InferenceRuntimeDataBundle:
    data_mode: str
    test_dataset: object
    extra_test_dataset: object | None = None
    extra_test_name: str | None = None


class InferencePreprocessedDataset:
    def __init__(self, preprocessing_dataset: object, dataset_root: str | Path) -> None:
        self.preprocessing_dataset = preprocessing_dataset
        self.dataset_root = Path(dataset_root).resolve()
        self.samples = [self._adapt_sample(preprocessing_dataset[index]) for index in range(len(preprocessing_dataset))]

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_paths(self, image_paths: tuple[str, ...]) -> list[str]:
        return [str((self.dataset_root / relative_path).resolve()) for relative_path in image_paths]

    def _adapt_sample(self, sample) -> dict[str, Any]:
        image_paths = self._resolve_image_paths(sample.image_paths)
        return {
            "question_id": sample.question_id,
            "split_name": sample.split_name,
            "subdataset_split": sample.subdataset_split,
            "task_family": sample.task_family,
            "image_type": sample.image_type,
            "answer_source": sample.answer_source,
            "element": sample.element or "",
            "image_paths": image_paths,
            "image_path": image_paths[0] if len(image_paths) == 1 else None,
            "image_mode": "single" if len(image_paths) == 1 else "multi",
            "question": sample.raw_question,
            "answer": sample.raw_answer,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def inference_collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    collated: dict[str, list[Any]] = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


def build_inference_data_bundle(config: InferenceConfig) -> InferenceRuntimeDataBundle:
    preprocessing_bundle: FinetuneDatasetBundle = build_finetune_dataset_bundle(
        dataset_root=Path(config.dataset_root).resolve(),
        use_subdataset=config.use_subdataset,
        seed=config.seed,
        data_mode=config.data_mode,
        filter_answers_over_20_tokens=config.filter_answers_over_20_tokens,
    )

    bundle = InferenceRuntimeDataBundle(
        data_mode=config.data_mode,
        test_dataset=InferencePreprocessedDataset(preprocessing_bundle.test_dataset, config.dataset_root),
        extra_test_dataset=(
            InferencePreprocessedDataset(preprocessing_bundle.extra_test_dataset, config.dataset_root)
            if preprocessing_bundle.extra_test_dataset is not None
            else None
        ),
        extra_test_name=preprocessing_bundle.extra_test_name,
    )
    print("\n[Inference] Chuẩn bị dữ liệu test bằng preprocessing pipeline...")
    print(f"- data_mode: {bundle.data_mode}")
    print(f"- use_subdataset: {config.use_subdataset}")
    print(f"- filter_answers_over_20_tokens: {config.filter_answers_over_20_tokens}")
    print(f"- test_samples: {len(bundle.test_dataset)}")
    if bundle.extra_test_dataset is not None and bundle.extra_test_name is not None:
        print(f"- {bundle.extra_test_name}_samples: {len(bundle.extra_test_dataset)}")
    return bundle
