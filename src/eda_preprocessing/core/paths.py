from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    output_root: Path
    csv_root: Path
    json_root: Path
    figure_root: Path
    log_root: Path

    @classmethod
    def from_output_root(cls, output_root: Path) -> "ProjectPaths":
        return cls(
            root=output_root.parent,
            output_root=output_root,
            csv_root=output_root / "csv",
            json_root=output_root / "json",
            figure_root=output_root / "figures",
            log_root=output_root / "logs",
        )

    def ensure(self) -> None:
        for directory in (
            self.output_root,
            self.csv_root,
            self.json_root,
            self.figure_root,
            self.log_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

