from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from eda_preprocessing.core.contracts import FigurePayload


class MatplotlibFigureRenderer:
    """Render supported figure payloads into PNG files."""

    def render(self, payload: FigurePayload, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        if payload.kind == "bar":
            self._render_bar(payload)
        elif payload.kind == "histogram":
            self._render_histogram(payload)
        elif payload.kind == "overlay_histogram":
            self._render_overlay_histogram(payload)
        elif payload.kind == "scatter":
            self._render_scatter(payload)
        elif payload.kind == "pie":
            self._render_pie(payload)
        else:
            raise ValueError(f"Unsupported figure kind: {payload.kind}")
        plt.title(payload.title)
        plt.xlabel(payload.x_label)
        plt.ylabel(payload.y_label)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()

    def _render_bar(self, payload: FigurePayload) -> None:
        labels = payload.data["labels"]
        values = payload.data["values"]
        positions = list(range(len(labels)))
        color = payload.data.get("color", "#2a6f97")
        bars = plt.bar(positions, values, color=color)
        plt.xticks(positions, labels, rotation=30, ha="right")
        if payload.data.get("show_value_labels", False):
            max_value = max(values) if values else 0
            offset = max(max_value * 0.01, 0.5)
            for bar, value in zip(bars, values, strict=False):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{value}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    def _render_histogram(self, payload: FigurePayload) -> None:
        values = payload.data["values"]
        bins = payload.data.get("bins", 20)
        color = payload.data.get("color", "#f4a261")
        plt.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.85)
        summary_stats = payload.data.get("summary_stats")
        if summary_stats:
            self._draw_summary_stats(summary_stats)

    def _render_scatter(self, payload: FigurePayload) -> None:
        x_values = payload.data["x_values"]
        y_values = payload.data["y_values"]
        plt.scatter(x_values, y_values, alpha=0.6, color="#e76f51", s=20)

    def _render_pie(self, payload: FigurePayload) -> None:
        labels = payload.data["labels"]
        values = payload.data["values"]
        colors = payload.data.get("colors")
        total = sum(values)
        wedges, _ = plt.pie(
            values,
            colors=colors,
            startangle=90,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        plt.axis("equal")
        for wedge, label, value in zip(wedges, labels, values, strict=False):
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            percent = (value / total * 100.0) if total else 0.0
            plt.annotate(
                f"{label}\n{value} ({percent:.1f}%)",
                xy=(x * 0.7, y * 0.7),
                xytext=(1.25 * np.sign(x), 1.2 * y),
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbbbbb"},
            )

    def _render_overlay_histogram(self, payload: FigurePayload) -> None:
        for series in payload.data["series"]:
            plt.hist(
                series["values"],
                bins=series.get("bins", 20),
                color=series.get("color", "#f4a261"),
                edgecolor="black",
                alpha=0.55,
                label=series.get("label", "Series"),
            )
        plt.legend()

    def _draw_summary_stats(self, summary_stats: dict[str, float | int]) -> None:
        axis = plt.gca()
        text = (
            f"min: {summary_stats['min']}\n"
            f"max: {summary_stats['max']}\n"
            f"mean: {summary_stats['mean']}\n"
            f"median: {summary_stats['median']}"
        )
        axis.text(
            0.98,
            0.98,
            text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#bbbbbb", "alpha": 0.95},
        )

        blended = transforms.blended_transform_factory(axis.transData, axis.transAxes)
        axis.axvline(float(summary_stats["mean"]), color="#1d3557", linestyle="--", linewidth=1.4, alpha=0.9)
        axis.axvline(float(summary_stats["median"]), color="#d62828", linestyle="-.", linewidth=1.4, alpha=0.9)
        axis.text(float(summary_stats["mean"]), 0.88, "mean", transform=blended, color="#1d3557", fontsize=8, ha="left")
        axis.text(float(summary_stats["median"]), 0.8, "median", transform=blended, color="#d62828", fontsize=8, ha="left")
