from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from eda_preprocessing.core.contracts import AnalysisResult
from eda_preprocessing.core.paths import ProjectPaths
from eda_preprocessing.core.text_utils import safe_slug
from eda_preprocessing.visualization.matplotlib_renderer import MatplotlibFigureRenderer


class ArtifactWriter:
    """Persist structured analysis outputs and figures."""

    def __init__(self, paths: ProjectPaths, renderer: MatplotlibFigureRenderer) -> None:
        self.paths = paths
        self.renderer = renderer

    def write_analysis(self, result: AnalysisResult) -> dict[str, list[str] | str]:
        summary_path = self.paths.csv_root / f"{result.case_id}_summary.csv"
        self._write_csv(summary_path, result.summary_rows)

        detail_csv_path = self.paths.csv_root / f"{result.case_id}_details.csv"
        if result.detail_rows:
            self._write_csv(detail_csv_path, result.detail_rows)

        metadata_path = self.paths.json_root / f"{result.case_id}_metadata.json"
        metadata_payload = {
            "case_id": result.case_id,
            "title": result.title,
            "metadata": result.metadata,
            "summary_row_count": len(result.summary_rows),
            "detail_row_count": len(result.detail_rows),
        }
        self._write_json(metadata_path, metadata_payload)

        figure_paths: list[str] = []
        for payload in result.figure_payloads:
            figure_path = self.paths.figure_root / f"{result.case_id}__{safe_slug(payload.figure_name)}.png"
            self.renderer.render(payload, figure_path)
            figure_paths.append(str(figure_path))

        return {
            "summary_csv": str(summary_path),
            "detail_csv": str(detail_csv_path) if result.detail_rows else "",
            "metadata_json": str(metadata_path),
            "figures": figure_paths,
        }

    def write_json_artifact(self, file_name: str, payload: dict[str, Any]) -> str:
        output_path = self.paths.json_root / file_name
        self._write_json(output_path, payload)
        return str(output_path)

    def _write_csv(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            rows = [{"status": "empty"}]
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, output_path: Path, payload: dict[str, Any]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
