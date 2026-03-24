from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.config import PipelineConfig
from eda_preprocessing.core.paths import ProjectPaths
from eda_preprocessing.io.writers import ArtifactWriter
from eda_preprocessing.pipeline.registry import build_analyzers
from eda_preprocessing.pipeline.runner import PipelineRunner
from eda_preprocessing.visualization.matplotlib_renderer import MatplotlibFigureRenderer


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_runs_in_sample_mode_on_real_dataset(self) -> None:
        dataset_root = PROJECT_ROOT.parent / "ViInfographicVQA_dataset"
        self.assertTrue(dataset_root.exists(), "Local ViInfographicVQA_dataset is required for smoke test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            paths = ProjectPaths.from_output_root(output_root)
            paths.ensure()

            config = PipelineConfig(
                dataset_root=dataset_root,
                image_root=dataset_root / "images",
                output_root=output_root,
                mode="sample",
                splits=("single_train", "single_test", "multi_train", "multi_test"),
                sample_size=2,
                seed=42,
                cases=tuple(analyzer.case_id for analyzer in build_analyzers()),
            )
            writer = ArtifactWriter(paths=paths, renderer=MatplotlibFigureRenderer())
            manifest = PipelineRunner(config=config, writer=writer).run()

            self.assertEqual(len(manifest["cases"]), 10)
            self.assertTrue((output_root / "json" / "dataset_overview.json").exists())
            self.assertTrue((output_root / "json" / "heuristic_rules.json").exists())
            self.assertTrue((output_root / "json" / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
