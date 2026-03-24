from __future__ import annotations

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_image_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload, ImageRecord


class TextDensityAnalyzer(Analyzer):
    case_id = "case_3_2_text_density"
    title = "Text Density Proxy Analysis"

    def analyze(self, context) -> AnalysisResult:
        valid_records = [record for record in context.image_records if record.status == "ok"]
        detail_rows = [self._to_row(record) for record in valid_records]
        summary_rows: list[dict[str, object]] = []

        for scope, records in build_image_scopes(valid_records).items():
            edge_values = [record.edge_density for record in records if record.edge_density is not None]
            entropy_values = [record.entropy for record in records if record.entropy is not None]
            component_values = [record.connected_components for record in records if record.connected_components is not None]
            summary_rows.append(
                {
                    "level": scope.level,
                    "scope_name": scope.name,
                    "image_count": len(records),
                    "mean_edge_density": self._numeric_summary(edge_values)["mean"],
                    "mean_entropy": self._numeric_summary(entropy_values)["mean"],
                    "mean_connected_components": self._numeric_summary(component_values)["mean"],
                    "p90_connected_components": self._numeric_summary(component_values)["p90"],
                }
            )

        figures = [
            FigurePayload(
                figure_name="edge_density_histogram",
                kind="histogram",
                title="Edge Density Distribution",
                x_label="Edge density",
                y_label="Frequency",
                data={
                    "values": [record.edge_density for record in valid_records if record.edge_density is not None],
                    "bins": 30,
                    "color": "#f4a261",
                    "summary_stats": self._numeric_summary(
                        [record.edge_density for record in valid_records if record.edge_density is not None]
                    ),
                },
            ),
            FigurePayload(
                figure_name="entropy_vs_components",
                kind="scatter",
                title="Entropy vs Connected Components",
                x_label="Entropy",
                y_label="Connected components",
                data={
                    "x_values": [record.entropy for record in valid_records if record.entropy is not None],
                    "y_values": [record.connected_components for record in valid_records if record.connected_components is not None],
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
                "metrics": ["edge_density", "entropy", "connected_components"],
            },
            figure_payloads=figures,
        )

    def _to_row(self, record: ImageRecord) -> dict[str, object]:
        return {
            "image_path": record.image_path,
            "split_names": ",".join(record.split_names),
            "task_families": ",".join(record.task_families),
            "edge_density": record.edge_density,
            "entropy": record.entropy,
            "connected_components": record.connected_components,
            "usage_count": record.usage_count,
        }
