"""Score calculation for mixed single-and-multi-image VQA predictions."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.common.metrics import calculate_averages as _calc_avg, compute_anls


def analyze_by_categories(preds: list[dict]) -> dict:
    category_scores = {
        "Overall": {"total_anls": 0.0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0.0, "count": 0}),
        "Image type": defaultdict(lambda: {"total_anls": 0.0, "count": 0}),
        "Question mode": defaultdict(lambda: {"total_anls": 0.0, "count": 0}),
    }

    for item in preds:
        if "answer" not in item or "predict" not in item:
            continue

        gt_clean = item["answer"].lower().strip()
        pr_clean = item["predict"].lower().strip().rstrip(".").replace('"', "")
        anls_score = compute_anls(gt_clean, pr_clean)
        item["anls"] = anls_score

        category_scores["Overall"]["total_anls"] += anls_score
        category_scores["Overall"]["count"] += 1

        answer_type = str(item.get("answer_source", "N/A")).lower()
        image_type = str(item.get("image_type", "N/A")).lower()
        question_mode = "multi" if "image_paths" in item else "single"

        category_scores["Answer type"][answer_type]["total_anls"] += anls_score
        category_scores["Answer type"][answer_type]["count"] += 1
        category_scores["Image type"][image_type]["total_anls"] += anls_score
        category_scores["Image type"][image_type]["count"] += 1
        category_scores["Question mode"][question_mode]["total_anls"] += anls_score
        category_scores["Question mode"][question_mode]["count"] += 1

    return category_scores


def calculate_averages(stats: dict[str, float | int]) -> dict[str, float | int]:
    return _calc_avg(stats)


def create_detailed_report(category_scores: dict) -> dict:
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    for key, sub_cats in category_scores.items():
        if key == "Overall":
            continue
        report[key] = {
            sub: calculate_averages(stats) for sub, stats in sorted(sub_cats.items())
        }
    return report


def main(results_dir: str = "results/single_and_multi") -> None:
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    result_root = Path(results_dir)
    files = sorted(
        path for path in result_root.rglob("*.json")
        if "scores" not in path.name and "analysis" not in path.name
    )
    if not files:
        print("No prediction files found")
        return

    detailed_analysis = {}
    results = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as file:
            preds = json.load(file)

        category_scores = analyze_by_categories(preds)
        model_name = path.relative_to(result_root).with_suffix("").as_posix().replace("/", "__")
        detailed_analysis[model_name] = create_detailed_report(category_scores)
        results[model_name] = {"anls": detailed_analysis[model_name]["Overall"]["anls"]}

        with open(path, "w", encoding="utf-8") as file:
            json.dump(preds, file, indent=2, ensure_ascii=False)

    with open(result_root / "final_scores.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with open(result_root / "detailed_analysis.json", "w", encoding="utf-8") as file:
        json.dump(detailed_analysis, file, indent=2, ensure_ascii=False)

    print("Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ANLS scores for mixed single-and-multi-image VQA")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/single_and_multi",
        help="Directory with prediction files",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir)
