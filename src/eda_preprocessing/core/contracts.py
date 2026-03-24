from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetSample:
    question_id: str
    split_name: str
    task_family: str
    image_type: str
    answer_source: str
    question: str
    answer: str
    image_paths: tuple[str, ...]
    element: str | None = None


@dataclass(frozen=True, slots=True)
class SingleSample(DatasetSample):
    """Typed single-image sample."""


@dataclass(frozen=True, slots=True)
class MultiSample(DatasetSample):
    """Typed multi-image sample."""


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_path: str
    absolute_path: Path
    split_names: tuple[str, ...]
    task_families: tuple[str, ...]
    usage_count: int
    width: int | None
    height: int | None
    aspect_ratio: float | None
    shorter_side: int | None
    edge_density: float | None
    entropy: float | None
    connected_components: int | None
    status: str
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class FigurePayload:
    figure_name: str
    kind: str
    title: str
    data: dict[str, Any]
    x_label: str = ""
    y_label: str = ""


@dataclass(slots=True)
class AnalysisResult:
    case_id: str
    title: str
    summary_rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    detail_rows: list[dict[str, Any]] = field(default_factory=list)
    figure_payloads: list[FigurePayload] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    dataset_root: Path
    image_root: Path
    output_root: Path
    mode: str
    seed: int
    selected_splits: tuple[str, ...]
    selected_cases: tuple[str, ...]
    samples: tuple[DatasetSample, ...]
    image_records: tuple[ImageRecord, ...]
    started_at: str

