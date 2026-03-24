from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.contracts import MultiSample, SingleSample
from eda_preprocessing.training.collate import build_collate_fn
from eda_preprocessing.training.dataset import PreprocessedTrainingDataset, build_finetune_datasets


class TrainingDatasetTest(unittest.TestCase):
    def test_preprocessed_training_dataset_filters_eagerly(self) -> None:
        valid_sample = SingleSample(
            question_id="1",
            split_name="single_train",
            task_family="single",
            image_type="Loai",
            answer_source="image-span",
            question="câu hỏi",
            answer="1",
            image_paths=("images/1.jpg",),
            element="Text",
        )
        invalid_sample = SingleSample(
            question_id="2",
            split_name="single_train",
            task_family="single",
            image_type="Loai",
            answer_source="image-span",
            question="câu hỏi",
            answer=" ".join(f"w{index}" for index in range(21)),
            image_paths=("images/2.jpg",),
            element="Text",
        )

        dataset = PreprocessedTrainingDataset([valid_sample, invalid_sample], subdataset_split="train")

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].question_id, "1")

    def test_collate_fn_preserves_image_path_order(self) -> None:
        samples = [
            MultiSample(
                question_id="2",
                split_name="multi_train",
                task_family="multi",
                image_type="Loai",
                answer_source="Multi-Image Spans",
                question="Câu hỏi",
                answer="1; 2",
                image_paths=("images/2.jpg", "images/3.jpg"),
            )
        ]
        dataset = PreprocessedTrainingDataset(samples, subdataset_split="train")
        batch = build_collate_fn()([dataset[0]])

        self.assertEqual(batch["image_paths"][0], ("images/2.jpg", "images/3.jpg"))
        self.assertEqual(batch["subdataset_split"][0], "train")

    def test_build_finetune_datasets_creates_train_valid_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            data_root = dataset_root / "data"
            data_root.mkdir(parents=True, exist_ok=True)

            def write_split(name: str, rows: list[dict[str, object]]) -> None:
                (data_root / f"{name}.json").write_text(
                    json.dumps(rows, ensure_ascii=False),
                    encoding="utf-8",
                )

            write_split(
                "single_train",
                [
                    {
                        "question_id": str(index),
                        "image_type": "Loai",
                        "answer_source": "image-span",
                        "element": "Text",
                        "question": "Câu hỏi",
                        "answer": "1",
                        "image_path": f"images/{index}.jpg",
                    }
                    for index in range(10)
                ],
            )
            write_split(
                "multi_train",
                [
                    {
                        "question_id": f"m-{index}",
                        "image_type": "Loai",
                        "answer_source": "Multi-Image Spans",
                        "question": "Câu hỏi",
                        "answer": "1; 2",
                        "image_paths": [f"images/m-{index}.jpg", f"images/m-{index}-2.jpg"],
                    }
                    for index in range(10)
                ],
            )
            write_split(
                "single_test",
                [
                    {
                        "question_id": f"t-{index}",
                        "image_type": "Loai",
                        "answer_source": "image-span",
                        "element": "Text",
                        "question": "Câu hỏi",
                        "answer": "1",
                        "image_path": f"images/t-{index}.jpg",
                    }
                    for index in range(3)
                ],
            )
            write_split(
                "multi_test",
                [
                    {
                        "question_id": f"tm-{index}",
                        "image_type": "Loai",
                        "answer_source": "Multi-Image Spans",
                        "question": "Câu hỏi",
                        "answer": "1; 2",
                        "image_paths": [f"images/tm-{index}.jpg", f"images/tm-{index}-2.jpg"],
                    }
                    for index in range(2)
                ],
            )

            train_dataset, valid_dataset, test_dataset = build_finetune_datasets(dataset_root, use_subdataset=True, seed=42)

            self.assertEqual(len(train_dataset), 4)
            self.assertEqual(len(valid_dataset), 0)
            self.assertEqual(len(test_dataset), 5)
