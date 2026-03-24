from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.contracts import MultiSample, SingleSample
from eda_preprocessing.subdataset.builder import build_subdataset_splits


class SubdatasetBuilderTest(unittest.TestCase):
    def test_builder_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            build_subdataset_splits([], train_fraction=1.5)

    def test_builder_samples_train_and_keeps_test(self) -> None:
        samples = []
        for index in range(10):
            samples.append(
                SingleSample(
                    question_id=f"st-{index}",
                    split_name="single_train",
                    task_family="single",
                    image_type="Loai",
                    answer_source="image-span",
                    question="Câu hỏi",
                    answer="1",
                    image_paths=(f"images/st-{index}.jpg",),
                    element="Text",
                )
            )
            samples.append(
                MultiSample(
                    question_id=f"mt-{index}",
                    split_name="multi_train",
                    task_family="multi",
                    image_type="Loai",
                    answer_source="Multi-Image Spans",
                    question="Câu hỏi",
                    answer="1",
                    image_paths=(f"images/mt-{index}.jpg", f"images/mt-{index}-2.jpg"),
                )
            )
            samples.append(
                SingleSample(
                    question_id=f"se-{index}",
                    split_name="single_test",
                    task_family="single",
                    image_type="Loai",
                    answer_source="image-span",
                    question="Câu hỏi",
                    answer="1",
                    image_paths=(f"images/se-{index}.jpg",),
                    element="Text",
                )
            )
            samples.append(
                MultiSample(
                    question_id=f"me-{index}",
                    split_name="multi_test",
                    task_family="multi",
                    image_type="Loai",
                    answer_source="Multi-Image Spans",
                    question="Câu hỏi",
                    answer="1",
                    image_paths=(f"images/me-{index}.jpg", f"images/me-{index}-2.jpg"),
                )
            )

        splits = build_subdataset_splits(samples, enabled=True, seed=42)

        self.assertEqual(len(splits.train_samples), 4)
        self.assertEqual(len(splits.valid_samples), 0)
        self.assertEqual(len(splits.test_samples), 20)

    def test_builder_disables_sampling_when_flag_is_false(self) -> None:
        samples = [
            SingleSample(
                question_id="1",
                split_name="single_train",
                task_family="single",
                image_type="Loai",
                answer_source="image-span",
                question="Câu hỏi",
                answer="1",
                image_paths=("images/1.jpg",),
                element="Text",
            ),
            MultiSample(
                question_id="2",
                split_name="multi_test",
                task_family="multi",
                image_type="Loai",
                answer_source="Multi-Image Spans",
                question="Câu hỏi",
                answer="1",
                image_paths=("images/2.jpg", "images/3.jpg"),
            ),
        ]

        splits = build_subdataset_splits(samples, enabled=False)

        self.assertEqual(len(splits.train_samples), 1)
        self.assertEqual(len(splits.valid_samples), 0)
        self.assertEqual(len(splits.test_samples), 1)
