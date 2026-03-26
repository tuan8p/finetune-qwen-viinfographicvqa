"""Multi-image VQA inference runner."""

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from src.common.utils import set_seed
from src.dataset import MultiImageVQADataset, multi_image_collate_fn

WANDB_LOG_CHUNK = 50

MODELS = {
    "internvl": "src.inference.multi.models.internvl.InternVLModel",
    "qwenvl":   "src.inference.multi.models.qwenvl.QwenVLModel",
    "ovis":     "src.inference.multi.models.ovis.OvisModel",
}


def import_model_class(model_key: str):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    module_path, class_name = MODELS[model_key].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-image VQA inference")
    # positional
    parser.add_argument(
        "model",
        type=str,
        choices=MODELS.keys(),
        help=f"Model to use: {list(MODELS.keys())}",
    )
    # data
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/multi_image_test.json",
        help="Path to the JSON annotation file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data root directory (images live under here)",
    )
    # misc
    parser.add_argument("--test_loading",  action="store_true", help="Load model in 4-bit for pipeline testing")
    parser.add_argument("--batch_size",    type=int, default=1,               help="Batch size")
    parser.add_argument("--seed",          type=int, default=42,              help="Random seed")
    parser.add_argument("--output_dir",    type=str, default="results/multi", help="Output directory")
    parser.add_argument("--save_interval", type=int, default=100,             help="Save progress every N samples")
    # wandb
    parser.add_argument("--use_wandb",      action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project",  type=str, default="",   help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name: [task]_[model]_[config]")
    parser.add_argument("--wandb_api_key",  type=str, default=None, help="W&B API key (or set WANDB_API_KEY env var)")
    return parser.parse_args()


def init_wandb(args, model_name: str):
    """Login and initialise a W&B run."""
    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model":      args.model,
            "model_name": model_name,
            "batch_size": args.batch_size,
            "json_path":  args.json_path,
            "data_dir":   args.data_dir,
            "dataset":    "ViInfographicVQA-Multi",
        },
    )
    print(f"W&B run: {run.url}")
    return run


def make_wandb_table():
    return wandb.Table(columns=[
        "question_id", "images", "image_type", "answer_source",
        "element", "question", "ground_truth", "prediction",
    ])


def flush_wandb_table(table, chunk_idx: int):
    wandb.log({"eval/predictions": table, "eval/chunk": chunk_idx})
    return make_wandb_table()


def main():
    args = parse_args()

    # Resolve data_dir
    if args.data_dir is None:
        from src.config import get_images_dir
        args.data_dir = get_images_dir()
        if args.data_dir is None:
            print("Error: --data_dir not set and VQA_IMAGES_DIR not configured")
            return 1

    set_seed(args.seed)

    # Load model
    model_class = import_model_class(args.model)
    model = model_class(load_test=args.test_loading)
    model_name = model.model_name

    # Init W&B after model is loaded (so model_name is known)
    wandb_run   = None
    wandb_table = None
    if args.use_wandb:
        wandb_run   = init_wandb(args, model_name)
        wandb_table = make_wandb_table()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    # Dataset & loader
    test_dataset = MultiImageVQADataset(
        json_path=args.json_path,
        data_dir=args.data_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=multi_image_collate_fn,
    )

    records     = []
    error_count = 0
    totals      = 0
    chunk_idx   = 0

    with torch.inference_mode():
        pbar = tqdm(test_loader, desc=f"Inference [{model_name}]")
        for batch in pbar:
            for i in range(len(batch["question"])):
                img_paths = batch["image_paths"][i]   # list[str] for this sample
                question  = batch["question"][i]
                gt        = batch["raw_answer"][i]

                if not img_paths:
                    pred = "ERROR: No images"
                    error_count += 1
                else:
                    try:
                        pred = model.infer(question=question, images=img_paths)
                    except Exception as e:
                        pred = f"ERROR: {e}"
                        error_count += 1

                records.append({
                    "question_id":   batch["question_id"][i],
                    "image_paths":   img_paths,
                    "image_type":    batch.get("image_type",    [""])[i],
                    "answer_source": batch.get("answer_source", [""])[i],
                    "element":       batch.get("element",       [""])[i],
                    "question":      question,
                    "answer":        gt,     # key matches calculate_scores.py
                    "predict":       pred,   # key matches calculate_scores.py
                })

                if args.use_wandb:
                    wandb_table.add_data(
                        batch["question_id"][i],
                        str(img_paths),                         # log paths as string
                        batch.get("image_type",    [""])[i],
                        batch.get("answer_source", [""])[i],
                        batch.get("element",       [""])[i],
                        question,
                        gt,
                        pred,
                    )

                totals += 1

                # Periodic save to disk
                if totals % args.save_interval == 0:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)

            if args.use_wandb:
                wandb.log({"eval/total_samples": totals})
                if totals % WANDB_LOG_CHUNK == 0:
                    wandb_table = flush_wandb_table(wandb_table, chunk_idx)
                    chunk_idx += 1

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {totals} records → {output_path}  (errors: {error_count})")
    print("Completed infer stage!!!! Huraaaaa")

    if args.use_wandb:
        if len(wandb_table.data) > 0:
            flush_wandb_table(wandb_table, chunk_idx)
        wandb.summary["total_samples"] = totals
        wandb.summary["error_count"]   = error_count
        wandb_run.finish()
        print("Pushed all results to Weights & Biases.")

    return 0


if __name__ == "__main__":
    exit(main())