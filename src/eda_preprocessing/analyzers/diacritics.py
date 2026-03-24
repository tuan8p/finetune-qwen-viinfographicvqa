from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.heuristics import analyze_diacritics, classify_diacritic_consistency


class DiacriticsAnalyzer(Analyzer):
    case_id = "case_5_1_diacritics"
    title = "Vietnamese Diacritics Consistency"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in context.samples:
            consistency = classify_diacritic_consistency(sample)
            overall_counts[consistency] += 1
            question_info = analyze_diacritics(sample.question)
            answer_info = analyze_diacritics(sample.answer)
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "diacritic_consistency": consistency,
                    "question_has_diacritics": question_info["has_diacritics"],
                    "answer_has_diacritics": answer_info["has_diacritics"],
                    "question_normalization_suspect": question_info["normalization_suspect"],
                    "answer_normalization_suspect": answer_info["normalization_suspect"],
                }
            )

        for scope, samples in build_sample_scopes(context.samples).items():
            scope_counts = Counter(classify_diacritic_consistency(sample) for sample in samples)
            summary_rows.extend(self._counter_rows(scope_counts, scope.level, scope.name, "diacritic_consistency"))

        figures = [
            FigurePayload(
                figure_name="diacritic_consistency_overall",
                kind="bar",
                title="Diacritics Consistency",
                x_label="Category",
                y_label="Count",
                data={"labels": list(overall_counts.keys()), "values": list(overall_counts.values())},
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

