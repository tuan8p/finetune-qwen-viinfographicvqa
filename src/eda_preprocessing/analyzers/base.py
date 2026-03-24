from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from statistics import mean, median
from typing import Iterable

import numpy as np

from eda_preprocessing.core.contracts import AnalysisContext, AnalysisResult


class Analyzer(ABC):
    case_id: str
    title: str

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        raise NotImplementedError

    def _numeric_summary(self, values: Iterable[float | int]) -> dict[str, float | int]:
        materialized = list(values)
        if not materialized:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p90": 0.0,
            }
        return {
            "count": len(materialized),
            "mean": round(float(mean(materialized)), 4),
            "median": round(float(median(materialized)), 4),
            "min": round(float(min(materialized)), 4),
            "max": round(float(max(materialized)), 4),
            "p90": round(float(np.percentile(materialized, 90)), 4),
        }

    def _counter_rows(
        self,
        counts: Counter[str],
        level: str,
        scope_name: str,
        label_name: str,
    ) -> list[dict[str, object]]:
        total = sum(counts.values())
        rows: list[dict[str, object]] = []
        for label, count in counts.most_common():
            rows.append(
                {
                    "level": level,
                    "scope_name": scope_name,
                    label_name: label,
                    "count": count,
                    "ratio": round(count / total, 6) if total else 0.0,
                }
            )
        return rows
