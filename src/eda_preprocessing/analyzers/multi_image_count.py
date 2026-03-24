from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload


class MultiImageCountAnalyzer(Analyzer):
    case_id = "case_4_1_multi_image_count"
    title = "Number of Images per Multi-image Sample"

    def analyze(self, context) -> AnalysisResult:
        multi_samples = [sample for sample in context.samples if sample.task_family == "multi"]
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in multi_samples:
            image_count = len(sample.image_paths)
            overall_counts[str(image_count)] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "image_count": image_count,
                    "answer_source": sample.answer_source,
                }
            )

        for scope, samples in build_sample_scopes(multi_samples).items():
            scope_counts = Counter(str(len(sample.image_paths)) for sample in samples)
            summary_rows.extend(self._counter_rows(scope_counts, scope.level, scope.name, "image_count"))

        figures = [
            FigurePayload(
                figure_name="multi_image_count_overall",
                kind="pie",
                title="Number of Images per Multi-image Sample",
                data={
                    "labels": list(overall_counts.keys()),
                    "values": list(overall_counts.values()),
                    "colors": ["#277da1", "#90be6d", "#f8961e"],
                },
            )
        ]

        return AnalysisResult(
            case_id=self.case_id,
            title=self.title,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata={"overall_counts": dict(overall_counts)},
            figure_payloads=figures,
        )
