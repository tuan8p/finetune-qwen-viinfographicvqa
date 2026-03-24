from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    dataset_root: Path
    image_root: Path
    output_root: Path
    mode: str
    splits: tuple[str, ...]
    sample_size: int
    seed: int
    cases: tuple[str, ...]
    log_level: str = "INFO"

