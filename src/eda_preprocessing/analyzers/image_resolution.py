from __future__ import annotations

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_image_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload, ImageRecord


class ImageResolutionAnalyzer(Analyzer):
    case_id = "case_3_1_image_resolution"
    title = "Image Resolution Distribution"

    def analyze(self, context) -> AnalysisResult:
        valid_records = [record for record in context.image_records if record.status == "ok"]
        detail_rows = [self._to_row(record) for record in valid_records]
        summary_rows: list[dict[str, object]] = []

        for scope, records in build_image_scopes(valid_records).items():
            widths = [record.width for record in records if record.width is not None]
            heights = [record.height for record in records if record.height is not None]
            aspect_ratios = [record.aspect_ratio for record in records if record.aspect_ratio is not None]
            shorter_sides = [record.shorter_side for record in records if record.shorter_side is not None]
            shorter_side_ratio = sum(side < 512 for side in shorter_sides) / len(shorter_sides) if shorter_sides else 0.0
            summary_rows.append(
                {
                    "level": scope.level,
                    "scope_name": scope.name,
                    "image_count": len(records),
                    "mean_width": self._numeric_summary(widths)["mean"],
                    "mean_height": self._numeric_summary(heights)["mean"],
                    "mean_aspect_ratio": self._numeric_summary(aspect_ratios)["mean"],
                    "shorter_side_lt_512_ratio": round(shorter_side_ratio, 6),
                }
            )

        figures = [
            FigurePayload(
                figure_name="image_resolution_scatter",
                kind="scatter",
                title="Image Width vs Height",
                x_label="Width",
                y_label="Height",
                data={
                    "x_values": [record.width for record in valid_records if record.width is not None],
                    "y_values": [record.height for record in valid_records if record.height is not None],
                },
            ),
        ]

        return AnalysisResult(
            case_id=self.case_id,
            title=self.title,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata={
                "valid_image_count": len(valid_records),
                "invalid_image_count": len(context.image_records) - len(valid_records),
            },
            figure_payloads=figures,
        )

    def _to_row(self, record: ImageRecord) -> dict[str, object]:
        return {
            "image_path": record.image_path,
            "split_names": ",".join(record.split_names),
            "task_families": ",".join(record.task_families),
            "width": record.width,
            "height": record.height,
            "aspect_ratio": record.aspect_ratio,
            "shorter_side": record.shorter_side,
            "usage_count": record.usage_count,
        }
