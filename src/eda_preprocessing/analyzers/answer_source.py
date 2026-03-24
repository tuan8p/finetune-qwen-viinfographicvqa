from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload


class AnswerSourceAnalyzer(Analyzer):
    case_id = "case_2_1_answer_source"
    title = "Answer Source Distribution"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in context.samples:
            overall_counts[sample.answer_source] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "answer_source": sample.answer_source,
                }
            )

        for scope, samples in build_sample_scopes(context.samples).items():
            scope_counts = Counter(sample.answer_source for sample in samples)
            summary_rows.extend(self._counter_rows(scope_counts, scope.level, scope.name, "answer_source"))

        figures = [
            FigurePayload(
                figure_name="answer_source_overall",
                kind="bar",
                title="Answer Source Distribution",
                x_label="Answer source",
                y_label="Count",
                data={
                    "labels": list(overall_counts.keys()),
                    "values": list(overall_counts.values()),
                    "color": "#3a5a40",
                    "show_value_labels": True,
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
