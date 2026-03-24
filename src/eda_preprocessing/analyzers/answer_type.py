from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.heuristics import classify_answer_type


class AnswerTypeAnalyzer(Analyzer):
    case_id = "case_1_2_answer_type"
    title = "Answer Semantic Type Distribution"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in context.samples:
            answer_type = classify_answer_type(sample)
            overall_counts[answer_type] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "answer_type": answer_type,
                    "answer_source": sample.answer_source,
                }
            )

        for scope, samples in build_sample_scopes(context.samples).items():
            scope_counts = Counter(classify_answer_type(sample) for sample in samples)
            summary_rows.extend(self._counter_rows(scope_counts, scope.level, scope.name, "answer_type"))

        figures = [
            FigurePayload(
                figure_name="answer_type_overall",
                kind="pie",
                title="Answer Type Distribution",
                data={
                    "labels": list(overall_counts.keys()),
                    "values": list(overall_counts.values()),
                    "colors": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
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
