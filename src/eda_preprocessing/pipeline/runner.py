from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, UTC

from eda_preprocessing.core.config import PipelineConfig
from eda_preprocessing.core.contracts import AnalysisContext, DatasetSample
from eda_preprocessing.core.heuristics import image_record_to_dict
from eda_preprocessing.core.paths import ProjectPaths
from eda_preprocessing.core.rules import HEURISTIC_RULES
from eda_preprocessing.io.dataset_loader import DatasetLoader
from eda_preprocessing.io.image_reader import ImageInventoryBuilder
from eda_preprocessing.io.writers import ArtifactWriter
from eda_preprocessing.pipeline.registry import build_analyzers


LOGGER = logging.getLogger(__name__)


class PipelineRunner:
    """Run all requested EDA analyzers and persist outputs."""

    def __init__(self, config: PipelineConfig, writer: ArtifactWriter) -> None:
        self.config = config
        self.writer = writer

    def run(self) -> dict[str, object]:
        started_at = datetime.now(UTC).isoformat()
        loader = DatasetLoader(self.config.dataset_root)
        samples = loader.load(self.config)
        LOGGER.info("Loaded %s samples", len(samples))

        inventory_builder = ImageInventoryBuilder(self.config.dataset_root)
        image_records = inventory_builder.build(samples)
        LOGGER.info("Built image inventory with %s unique images", len(image_records))

        context = AnalysisContext(
            dataset_root=self.config.dataset_root,
            image_root=self.config.image_root,
            output_root=self.config.output_root,
            mode=self.config.mode,
            seed=self.config.seed,
            selected_splits=self.config.splits,
            selected_cases=self.config.cases,
            samples=tuple(samples),
            image_records=tuple(image_records),
            started_at=started_at,
        )

        dataset_overview = self._build_dataset_overview(samples, image_records)
        overview_path = self.writer.write_json_artifact("dataset_overview.json", dataset_overview)
        rules_path = self.writer.write_json_artifact("heuristic_rules.json", {"heuristic_rules": HEURISTIC_RULES})

        case_manifests: list[dict[str, object]] = []
        analyzers = [analyzer for analyzer in build_analyzers() if analyzer.case_id in self.config.cases]
        for analyzer in analyzers:
            LOGGER.info("Running analyzer %s", analyzer.case_id)
            started = time.perf_counter()
            result = analyzer.analyze(context)
            artifact_manifest = self.writer.write_analysis(result)
            duration_seconds = round(time.perf_counter() - started, 3)
            case_manifests.append(
                {
                    "case_id": analyzer.case_id,
                    "title": analyzer.title,
                    "duration_seconds": duration_seconds,
                    "artifacts": artifact_manifest,
                }
            )
            LOGGER.info("Completed %s in %ss", analyzer.case_id, duration_seconds)

        manifest = {
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "mode": self.config.mode,
            "splits": list(self.config.splits),
            "sample_size_per_split": self.config.sample_size if self.config.mode == "sample" else None,
            "seed": self.config.seed,
            "dataset_overview_json": overview_path,
            "heuristic_rules_json": rules_path,
            "cases": case_manifests,
        }
        manifest_path = self.writer.write_json_artifact("run_manifest.json", manifest)
        manifest["run_manifest_json"] = manifest_path
        return manifest

    def _build_dataset_overview(self, samples, image_records) -> dict[str, object]:
        split_counts = Counter(sample.split_name for sample in samples)
        task_counts = Counter(sample.task_family for sample in samples)
        answer_source_counts = Counter(sample.answer_source for sample in samples)
        image_type_counts = Counter(sample.image_type for sample in samples)

        unique_images_by_split: dict[str, set[str]] = defaultdict(set)
        unique_images_by_task: dict[str, set[str]] = defaultdict(set)
        for sample in samples:
            for image_path in sample.image_paths:
                unique_images_by_split[sample.split_name].add(image_path)
                unique_images_by_task[sample.task_family].add(image_path)

        return {
            "mode": self.config.mode,
            "sample_count": len(samples),
            "split_counts": dict(split_counts),
            "task_family_counts": dict(task_counts),
            "answer_source_counts": dict(answer_source_counts),
            "top_10_image_types": dict(image_type_counts.most_common(10)),
            "unique_image_counts_by_split": {key: len(value) for key, value in unique_images_by_split.items()},
            "unique_image_counts_by_task_family": {key: len(value) for key, value in unique_images_by_task.items()},
            "image_inventory": {
                "total_unique_images": len(image_records),
                "status_counts": dict(Counter(record.status for record in image_records)),
                "records_preview": [image_record_to_dict(record) for record in image_records[:10]],
            },
        }

