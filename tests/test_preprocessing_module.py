from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.contracts import MultiSample, SingleSample
from eda_preprocessing.preprocessing.sample_preprocessor import preprocess_sample


class PreprocessingModuleTest(unittest.TestCase):
    def test_preprocess_keeps_answer_with_20_tokens(self) -> None:
        answer = " ".join(f"w{index}" for index in range(20))
        sample = SingleSample(
            question_id="1",
            split_name="single_train",
            task_family="single",
            image_type="Loai",
            answer_source="image-span",
            question="câu hỏi",
            answer=answer,
            image_paths=("images/1.jpg",),
            element="Text",
        )

        processed = preprocess_sample(sample, subdataset_split="train")

        self.assertIsNotNone(processed)
        self.assertEqual(processed.answer_tokens, 20)

    def test_preprocess_filters_answer_over_20_tokens(self) -> None:
        answer = " ".join(f"w{index}" for index in range(21))
        sample = SingleSample(
            question_id="1",
            split_name="single_train",
            task_family="single",
            image_type="Loai",
            answer_source="image-span",
            question="câu hỏi",
            answer=answer,
            image_paths=("images/1.jpg",),
            element="Text",
        )

        processed = preprocess_sample(sample, subdataset_split="train")

        self.assertIsNone(processed)

    def test_preprocess_splits_multi_span_answer_and_preserves_image_paths(self) -> None:
        sample = MultiSample(
            question_id="2",
            split_name="multi_train",
            task_family="multi",
            image_type="Loai",
            answer_source="Multi-Image Spans",
            question="GDP của các nước là bao nhiêu?",
            answer="10 % ; 20 %",
            image_paths=("images/2.jpg", "images/3.jpg"),
        )

        processed = preprocess_sample(sample, subdataset_split="valid")

        self.assertIsNotNone(processed)
        self.assertEqual(processed.answer_normalized, "10%; 20%")
        self.assertEqual(processed.answer_segments, ("10%", "20%"))
        self.assertEqual(processed.image_paths, ("images/2.jpg", "images/3.jpg"))
        self.assertEqual(processed.subdataset_split, "valid")
        self.assertEqual(processed.question_normalized, "GDP của các nước là bao nhiêu?")
