from __future__ import annotations

from collections import Counter

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.heuristics import detect_tokenization_flags


class TokenizationAmbiguityAnalyzer(Analyzer):
    case_id = "case_5_2_tokenization_ambiguity"
    title = "Vietnamese Tokenization Ambiguity"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        overall_counts: Counter[str] = Counter()

        for sample in context.samples:
            flags = detect_tokenization_flags(f"{sample.question} || {sample.answer}")
            if not flags:
                flags = ["none"]
            for flag in flags:
                overall_counts[flag] += 1
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "tokenization_flags": ";".join(flags),
                    "flag_count": len(flags),
                }
            )

        for scope, samples in build_sample_scopes(context.samples).items():
            scope_counts: Counter[str] = Counter()
            for sample in samples:
                flags = detect_tokenization_flags(f"{sample.question} || {sample.answer}") or ["none"]
                scope_counts.update(flags)
            sample_count = len(samples)
            for flag, count in scope_counts.most_common():
                summary_rows.append(
                    {
                        "level": scope.level,
                        "scope_name": scope.name,
                        "tokenization_flag": flag,
                        "count": count,
                        "sample_ratio": round(count / sample_count, 6) if sample_count else 0.0,
                    }
                )

        figures = [
            FigurePayload(
                figure_name="tokenization_flags_overall",
                kind="bar",
                title="Tokenization Ambiguity Flags",
                x_label="Flag",
                y_label="Count",
                data={
                    "labels": list(overall_counts.keys()),
                    "values": list(overall_counts.values()),
                    "color": "#8d99ae",
                    "show_value_labels": True,
                },
            )
        ]

        return AnalysisResult(
            case_id=self.case_id,
            title=self.title,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata={
                "overall_counts": dict(overall_counts),
                "total_samples": len(context.samples),
            },
            figure_payloads=figures,
        )
