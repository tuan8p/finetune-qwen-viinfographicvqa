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

from eda_preprocessing.core.config import PipelineConfig
from eda_preprocessing.io.dataset_loader import DatasetLoader


class DatasetLoaderTest(unittest.TestCase):
    def test_loader_reads_single_and_multi_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            data_root = dataset_root / "data"
            data_root.mkdir(parents=True, exist_ok=True)

            (data_root / "single_train.json").write_text(
                json.dumps(
                    [
                        {
                            "question_id": "1",
                            "image_type": "Kinh tế",
                            "answer_source": "image-span",
                            "element": "Text",
                            "question": "Câu hỏi gì?",
                            "answer": "Đáp án",
                            "image_path": "images/1.jpg",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_root / "multi_train.json").write_text(
                json.dumps(
                    [
                        {
                            "question_id": "2",
                            "image_type": "Y tế",
                            "answer_source": "Multi-Image Spans",
                            "question": "So sánh gì?",
                            "answer": "A; B",
                            "image_paths": ["images/2.jpg", "images/3.jpg"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = PipelineConfig(
                dataset_root=dataset_root,
                image_root=dataset_root / "images",
                output_root=dataset_root / "outputs",
                mode="full",
                splits=("single_train", "multi_train"),
                sample_size=4,
                seed=42,
                cases=("case_1_1_answer_length",),
            )
            samples = DatasetLoader(dataset_root).load(config)

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0].task_family, "single")
            self.assertEqual(samples[0].image_paths, ("images/1.jpg",))
            self.assertEqual(samples[1].task_family, "multi")
            self.assertEqual(samples[1].image_paths, ("images/2.jpg", "images/3.jpg"))
            self.assertIsNone(samples[1].element)


if __name__ == "__main__":
    unittest.main()

