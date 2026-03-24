from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.heuristics import classify_question_type, classify_reasoning_mode


class QuestionTypeAnalyzer(Analyzer):
    case_id = "case_1_3_question_type"
    title = "Question Type Distribution"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        question_type_counts: Counter[str] = Counter()
        reasoning_counts: Counter[str] = Counter()

        for sample in context.samples:
            question_type = classify_question_type(sample)
            reasoning_mode = classify_reasoning_mode(sample)
            question_type_counts[question_type] += 1
            reasoning_counts[reasoning_mode] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "question_type": question_type,
                    "reasoning_mode": reasoning_mode,
                }
            )

        for scope, samples in build_sample_scopes(context.samples).items():
            type_counts = Counter(classify_question_type(sample) for sample in samples)
            mode_counts = Counter(classify_reasoning_mode(sample) for sample in samples)
            summary_rows.extend(self._counter_rows(type_counts, scope.level, scope.name, "question_type"))
            summary_rows.extend(self._counter_rows(mode_counts, scope.level, scope.name, "reasoning_mode"))

        figures = [
            FigurePayload(
                figure_name="question_type_overall",
                kind="bar",
                title="Question Type Distribution",
                x_label="Question type",
                y_label="Count",
                data={
                    "labels": list(question_type_counts.keys()),
                    "values": list(question_type_counts.values()),
                    "color": "#2a6f97",
                    "show_value_labels": True,
                },
            ),
            FigurePayload(
                figure_name="reasoning_vs_lookup",
                kind="pie",
                title="Reasoning vs Lookup",
                data={
                    "labels": list(reasoning_counts.keys()),
                    "values": list(reasoning_counts.values()),
                    "colors": ["#6d597a", "#e56b6f"],
                },
            ),
        ]

        return AnalysisResult(
            case_id=self.case_id,
            title=self.title,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata={
                "overall_question_type_counts": dict(question_type_counts),
                "overall_reasoning_mode_counts": dict(reasoning_counts),
            },
            figure_payloads=figures,
        )
