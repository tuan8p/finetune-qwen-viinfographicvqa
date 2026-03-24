from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.contracts import MultiSample, SingleSample
from eda_preprocessing.core.heuristics import (
    classify_answer_type,
    classify_cross_image_dependency,
    classify_diacritic_consistency,
    classify_question_type,
    classify_reasoning_mode,
    detect_tokenization_flags,
)


class HeuristicsTest(unittest.TestCase):
    def test_answer_type_and_reasoning_classifiers(self) -> None:
        number_sample = SingleSample(
            question_id="1",
            split_name="single_train",
            task_family="single",
            image_type="Kinh tế",
            answer_source="non-extractive",
            question="Tổng số là bao nhiêu?",
            answer="12",
            image_paths=("images/1.jpg",),
            element="Sequence",
        )
        entity_sample = SingleSample(
            question_id="2",
            split_name="single_train",
            task_family="single",
            image_type="Thời tiết",
            answer_source="image-span",
            question="Vùng nào có nhiệt độ cao nhất?",
            answer="Nam Bộ",
            image_paths=("images/2.jpg",),
            element="Map",
        )
        yes_no_sample = SingleSample(
            question_id="3",
            split_name="single_train",
            task_family="single",
            image_type="Y tế",
            answer_source="question-span",
            question="Có dịch bệnh hay không?",
            answer="có",
            image_paths=("images/3.jpg",),
            element="Text",
        )

        self.assertEqual(classify_answer_type(number_sample), "number")
        self.assertEqual(classify_answer_type(entity_sample), "entity_like")
        self.assertEqual(classify_answer_type(yes_no_sample), "other")
        self.assertEqual(classify_question_type(number_sample), "how_many")
        self.assertEqual(classify_reasoning_mode(number_sample), "reasoning")
        self.assertEqual(
            classify_question_type(
                SingleSample(
                    question_id="5",
                    split_name="single_train",
                    task_family="single",
                    image_type="Kinh tế",
                    answer_source="question-span",
                    question="So sanh hai phương án, phương án nào tốt hơn?",
                    answer="A",
                    image_paths=("images/5.jpg",),
                    element="Text",
                )
            ),
            "compare",
        )

    def test_cross_image_dependency(self) -> None:
        multi_sample = MultiSample(
            question_id="3",
            split_name="multi_train",
            task_family="multi",
            image_type="Thể thao",
            answer_source="Cross-Image Synthesis",
            question="So sánh thành tích hai ngày, ngày nào cao hơn và hơn bao nhiêu?",
            answer="18/06/2017; 1°C",
            image_paths=("images/1.jpg", "images/2.jpg"),
        )

        self.assertEqual(classify_cross_image_dependency(multi_sample), "comparison")
        aggregation_sample = MultiSample(
            question_id="4",
            split_name="multi_train",
            task_family="multi",
            image_type="Thể thao",
            answer_source="Non-Span",
            question="Tổng số huy chương của hai ngày là bao nhiêu?",
            answer="4",
            image_paths=("images/1.jpg", "images/2.jpg"),
        )
        self.assertEqual(classify_cross_image_dependency(aggregation_sample), "aggregation")

    def test_diacritics_and_tokenization_flags(self) -> None:
        sample = SingleSample(
            question_id="4",
            split_name="single_test",
            task_family="single",
            image_type="Y tế",
            answer_source="question-span",
            question="So nguoi mac vao ngay 24/11/2020 la bao nhieu?",
            answer="15.874; 188.678",
            image_paths=("images/3.jpg",),
            element="Text",
        )

        flags = detect_tokenization_flags(f"{sample.question} || {sample.answer}")
        self.assertIn("slash_form", flags)
        self.assertIn("date_time_currency", flags)
        self.assertIn("multi_span_delimiter", flags)
        self.assertIn(
            classify_diacritic_consistency(sample),
            {"question_missing_diacritics", "question_text_without_diacritics", "both_without_diacritics", "mixed_form"},
        )
        diacritic_sample = SingleSample(
            question_id="5",
            split_name="single_test",
            task_family="single",
            image_type="Kinh tế",
            answer_source="image-span",
            question="Nền kinh tế tăng trưởng bao nhiêu phần trăm?",
            answer="12%",
            image_paths=("images/4.jpg",),
            element="Text",
        )
        self.assertEqual(classify_diacritic_consistency(diacritic_sample), "question_text_with_diacritics")
        self.assertEqual(detect_tokenization_flags("Tăng trưởng GDP của CPTPP là bao nhiêu?"), ["abbreviation"])
        self.assertEqual(detect_tokenization_flags("Đây là câu tiếng Việt bình thường"), [])


if __name__ == "__main__":
    unittest.main()
