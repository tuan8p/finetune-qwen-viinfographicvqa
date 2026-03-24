from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.core.config import PipelineConfig
from eda_preprocessing.core.logging_utils import configure_logging
from eda_preprocessing.core.paths import ProjectPaths
from eda_preprocessing.io.dataset_loader import SUPPORTED_SPLITS
from eda_preprocessing.io.writers import ArtifactWriter
from eda_preprocessing.pipeline.registry import build_analyzers
from eda_preprocessing.pipeline.runner import PipelineRunner
from eda_preprocessing.visualization.matplotlib_renderer import MatplotlibFigureRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA preprocessing for ViInfographicVQA")
    parser.add_argument(
        "--dataset-root",
        default=str(CURRENT_DIR.parent / "ViInfographicVQA_dataset"),
        help="Path to the local ViInfographicVQA_dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CURRENT_DIR / "outputs"),
        help="Directory where EDA artifacts will be written",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "sample"),
        default="full",
        help="Run mode: full dataset or seeded sample per split",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated split list, or 'all'",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="Sample size per split when mode=sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample mode",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case IDs, or 'all'",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level",
    )
    return parser.parse_args()


def resolve_splits(raw_value: str) -> tuple[str, ...]:
    if raw_value == "all":
        return SUPPORTED_SPLITS
    splits = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    invalid = [split for split in splits if split not in SUPPORTED_SPLITS]
    if invalid:
        raise ValueError(f"Unsupported splits requested: {invalid}")
    return splits


def resolve_cases(raw_value: str) -> tuple[str, ...]:
    available_cases = tuple(analyzer.case_id for analyzer in build_analyzers())
    if raw_value == "all":
        return available_cases
    cases = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    invalid = [case_id for case_id in cases if case_id not in available_cases]
    if invalid:
        raise ValueError(f"Unsupported cases requested: {invalid}")
    return cases


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_dir).resolve()
    splits = resolve_splits(args.splits)
    cases = resolve_cases(args.cases)

    paths = ProjectPaths.from_output_root(output_root)
    paths.ensure()
    configure_logging(paths.log_root, args.log_level)

    config = PipelineConfig(
        dataset_root=dataset_root,
        image_root=dataset_root / "images",
        output_root=output_root,
        mode=args.mode,
        splits=splits,
        sample_size=args.sample_size,
        seed=args.seed,
        cases=cases,
        log_level=args.log_level,
    )
    writer = ArtifactWriter(paths=paths, renderer=MatplotlibFigureRenderer())
    PipelineRunner(config=config, writer=writer).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
