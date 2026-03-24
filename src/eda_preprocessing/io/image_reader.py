from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eda_preprocessing.core.contracts import DatasetSample, ImageRecord


LOGGER = logging.getLogger(__name__)


class ImageInventoryBuilder:
    """Build a deduplicated inventory of images and compute proxy metrics once."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root

    def build(self, samples: list[DatasetSample]) -> list[ImageRecord]:
        image_usage: dict[str, int] = defaultdict(int)
        image_splits: dict[str, set[str]] = defaultdict(set)
        image_tasks: dict[str, set[str]] = defaultdict(set)

        for sample in samples:
            for image_path in sample.image_paths:
                image_usage[image_path] += 1
                image_splits[image_path].add(sample.split_name)
                image_tasks[image_path].add(sample.task_family)

        records: list[ImageRecord] = []
        total_images = len(image_usage)
        for index, image_path in enumerate(sorted(image_usage), start=1):
            if index == 1 or index % 250 == 0 or index == total_images:
                LOGGER.info("Computing image metrics: %s/%s", index, total_images)
            absolute_path = self.dataset_root / image_path
            records.append(
                self._build_record(
                    image_path=image_path,
                    absolute_path=absolute_path,
                    split_names=tuple(sorted(image_splits[image_path])),
                    task_families=tuple(sorted(image_tasks[image_path])),
                    usage_count=image_usage[image_path],
                )
            )
        return records

    def _build_record(
        self,
        image_path: str,
        absolute_path: Path,
        split_names: tuple[str, ...],
        task_families: tuple[str, ...],
        usage_count: int,
    ) -> ImageRecord:
        if not absolute_path.exists():
            return ImageRecord(
                image_path=image_path,
                absolute_path=absolute_path,
                split_names=split_names,
                task_families=task_families,
                usage_count=usage_count,
                width=None,
                height=None,
                aspect_ratio=None,
                shorter_side=None,
                edge_density=None,
                entropy=None,
                connected_components=None,
                status="missing",
                error_message="Image file not found",
            )

        image = cv2.imread(str(absolute_path), cv2.IMREAD_COLOR)
        if image is None:
            return ImageRecord(
                image_path=image_path,
                absolute_path=absolute_path,
                split_names=split_names,
                task_families=task_families,
                usage_count=usage_count,
                width=None,
                height=None,
                aspect_ratio=None,
                shorter_side=None,
                edge_density=None,
                entropy=None,
                connected_components=None,
                status="decode_error",
                error_message="OpenCV failed to decode image",
            )

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        histogram = np.histogram(gray, bins=256, range=(0, 256), density=True)[0] + 1e-12
        entropy = float(-(histogram * np.log2(histogram)).sum())
        binary_mask = (gray < 220).astype("uint8")
        connected_components = int(cv2.connectedComponents(binary_mask)[0] - 1)

        return ImageRecord(
            image_path=image_path,
            absolute_path=absolute_path,
            split_names=split_names,
            task_families=task_families,
            usage_count=usage_count,
            width=int(width),
            height=int(height),
            aspect_ratio=round(width / height, 6) if height else None,
            shorter_side=int(min(width, height)),
            edge_density=round(float(edges.mean() / 255.0), 6),
            entropy=round(entropy, 6),
            connected_components=connected_components,
            status="ok",
        )

