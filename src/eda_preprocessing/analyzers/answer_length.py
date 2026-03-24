from __future__ import annotations

from eda_preprocessing.analyzers.base import Analyzer
from eda_preprocessing.core.aggregation import build_sample_scopes
from eda_preprocessing.core.contracts import AnalysisResult, FigurePayload
from eda_preprocessing.core.text_utils import whitespace_tokens


class AnswerLengthAnalyzer(Analyzer):
    case_id = "case_1_1_answer_length"
    title = "Answer Length Distribution"

    def analyze(self, context) -> AnalysisResult:
        detail_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        token_values: list[int] = []

        for sample in context.samples:
            answer_tokens = len(whitespace_tokens(sample.answer))
            token_values.append(answer_tokens)
            detail_rows.append(
                {
                    "question_id": sample.question_id,
                    "split_name": sample.split_name,
                    "task_family": sample.task_family,
                    "answer_tokens": answer_tokens,
                    "short_answer_le_3_tokens": answer_tokens <= 3,
                }
            )

        scopes = build_sample_scopes(context.samples)
        for scope, samples in scopes.items():
            token_lengths = [len(whitespace_tokens(sample.answer)) for sample in samples]
            short_ratio = sum(length <= 3 for length in token_lengths) / len(token_lengths) if token_lengths else 0.0
            token_summary = self._numeric_summary(token_lengths)
            summary_rows.append(
                {
                    "level": scope.level,
                    "scope_name": scope.name,
                    "sample_count": len(samples),
                    "mean_answer_tokens": token_summary["mean"],
                    "median_answer_tokens": token_summary["median"],
                    "p90_answer_tokens": token_summary["p90"],
                    "short_answer_ratio_le_3_tokens": round(short_ratio, 6),
                }
            )

        figures = [
            FigurePayload(
                figure_name="answer_tokens_histogram",
                kind="histogram",
                title="Answer Token Length Distribution",
                x_label="Token count",
                y_label="Frequency",
                data={
                    "values": token_values,
                    "bins": 30,
                    "color": "#457b9d",
                    "summary_stats": self._numeric_summary(token_values),
                },
            )
        ]

        return AnalysisResult(
            case_id=self.case_id,
            title=self.title,
            summary_rows=summary_rows,
            detail_rows=detail_rows,
            metadata={
                "total_samples": len(context.samples),
                "token_distribution_summary": self._numeric_summary(token_values),
            },
            figure_payloads=figures,
        )
