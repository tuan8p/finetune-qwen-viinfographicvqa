from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload, SingleSample
from eda_preprocessing.core.paths import ProjectPaths
from eda_preprocessing.io.image_reader import ImageInventoryBuilder
from eda_preprocessing.io.writers import ArtifactWriter
from eda_preprocessing.visualization.matplotlib_renderer import MatplotlibFigureRenderer


class ImageAndWriterTest(unittest.TestCase):
    def test_image_inventory_handles_valid_and_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            image_dir = dataset_root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image_path = image_dir / "valid.jpg"
            image = np.full((32, 48, 3), 255, dtype=np.uint8)
            cv2.putText(image, "VN", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(str(image_path), image)

            samples = [
                SingleSample(
                    question_id="1",
                    split_name="single_train",
                    task_family="single",
                    image_type="Kinh tế",
                    answer_source="image-span",
                    question="Câu hỏi",
                    answer="Đáp án",
                    image_paths=("images/valid.jpg",),
                    element="Text",
                ),
                SingleSample(
                    question_id="2",
                    split_name="single_train",
                    task_family="single",
                    image_type="Kinh tế",
                    answer_source="image-span",
                    question="Câu hỏi",
                    answer="Đáp án",
                    image_paths=("images/missing.jpg",),
                    element="Text",
                ),
            ]

            records = ImageInventoryBuilder(dataset_root).build(samples)
            status_map = {record.image_path: record.status for record in records}
            self.assertEqual(status_map["images/valid.jpg"], "ok")
            self.assertEqual(status_map["images/missing.jpg"], "missing")

    def test_artifact_writer_creates_csv_json_and_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            paths = ProjectPaths.from_output_root(output_root)
            paths.ensure()
            writer = ArtifactWriter(paths=paths, renderer=MatplotlibFigureRenderer())

            result = AnalysisResult(
                case_id="demo_case",
                title="Demo",
                summary_rows=[{"level": "overall", "scope_name": "overall", "count": 2}],
                detail_rows=[{"row_id": 1, "value": 10}],
                metadata={"ok": True},
                figure_payloads=[
                    FigurePayload(
                        figure_name="demo_plot",
                        kind="bar",
                        title="Demo Plot",
                        x_label="Label",
                        y_label="Value",
                        data={"labels": ["a", "b"], "values": [1, 2]},
                    )
                ],
            )

            manifest = writer.write_analysis(result)
            self.assertTrue(Path(manifest["summary_csv"]).exists())
            self.assertTrue(Path(manifest["detail_csv"]).exists())
            self.assertTrue(Path(manifest["metadata_json"]).exists())
            self.assertEqual(len(manifest["figures"]), 1)
            self.assertTrue(Path(manifest["figures"][0]).exists())


if __name__ == "__main__":
    unittest.main()

