from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.heuristics import classify_cross_image_dependency


class CrossImageDependencyAnalyzer(Analyzer):
    case_id = "case_4_2_cross_image_dependency"
    title = "Cross-image Dependency Classification"

    def analyze(self, context) -> AnalysisResult:
        multi_samples = [sample for sample in context.samples if sample.task_family == "multi"]
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in multi_samples:
            dependency = classify_cross_image_dependency(sample)
            overall_counts[dependency] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "cross_image_dependency": dependency,
                    "answer_source": sample.answer_source,
                }
            )

        for scope, samples in build_sample_scopes(multi_samples).items():
            scope_counts = Counter(classify_cross_image_dependency(sample) for sample in samples)
            summary_rows.extend(self._counter_rows(scope_counts, scope.level, scope.name, "cross_image_dependency"))

        figures = [
            FigurePayload(
                figure_name="cross_image_dependency_overall",
                kind="pie",
                title="Cross-image Dependency Distribution",
                data={
                    "labels": list(overall_counts.keys()),
                    "values": list(overall_counts.values()),
                    "colors": ["#577590", "#43aa8b", "#f94144"],
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
