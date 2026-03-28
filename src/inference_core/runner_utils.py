from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.utils import set_seed
from src.inference_core.config import InferenceConfig, build_config, save_config_yaml
from src.inference_core.runtime_data import (
    InferenceRuntimeDataBundle,
    build_inference_data_bundle,
    inference_collate_fn,
)
from src.inference_core.wandb_utils import (
    build_artifact_name,
    get_wandb_module,
    init_wandb_run,
    log_wandb_file_artifact,
    log_wandb_runtime_metadata,
)


def build_arg_parser(description: str, models: dict[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("model", choices=models.keys(), help=f"Model to use: {list(models.keys())}")
    parser.add_argument("--config", type=str, default=None, help="Path to shared inference YAML config")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root override")
    parser.add_argument("--model-path", type=str, default=None, help="Model path or adapter checkpoint override")
    parser.add_argument("--output-dir", type=str, default=None, help="Prediction output root override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--save-interval", type=int, default=None, help="Periodic save interval override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--test-loading", action="store_true", help="Load model in 4-bit for quick pipeline testing")
    parser.add_argument("--use-wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--disable-subdataset", action="store_true", help="Force use_subdataset=False")
    return parser


def build_runtime_config(
    args: argparse.Namespace,
    *,
    data_mode: str,
    default_output_subdir: str,
) -> InferenceConfig:
    output_dir = args.output_dir or str(Path(default_output_subdir).resolve())
    use_subdataset_override = False if args.disable_subdataset else None
    return build_config(
        args.config,
        data_mode=data_mode,
        dataset_root=args.dataset_root,
        model_key=args.model,
        model_path=args.model_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        seed=args.seed,
        test_loading=True if args.test_loading else None,
        use_wandb=True if args.use_wandb else None,
        use_subdataset=use_subdataset_override,
    )


def import_model_class(models: dict[str, str], model_key: str):
    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(models.keys())}")
    module_path, class_name = models[model_key].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def build_record(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    record = {
        "question_id": sample["question_id"],
        "image_type": sample.get("image_type", ""),
        "answer_source": sample.get("answer_source", ""),
        "element": sample.get("element", ""),
        "question": sample["question"],
        "answer": sample["answer"],
        "predict": prediction,
    }
    if sample.get("image_mode") == "single":
        record["image_path"] = sample["image_path"]
    else:
        record["image_paths"] = sample["image_paths"]
    return record


def save_predictions(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_prediction_loop(
    *,
    model: Any,
    dataset: object,
    config: InferenceConfig,
    output_path: Path,
    split_label: str,
    inference_style: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=inference_collate_fn,
    )

    records: list[dict[str, Any]] = []
    error_count = 0
    total_samples = 0

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"Inference [{model.model_name}] {split_label}")
        for batch in progress:
            batch_size = len(batch["question"])
            for index in range(batch_size):
                sample = {key: value[index] for key, value in batch.items()}
                try:
                    if not sample["image_paths"]:
                        raise ValueError("No images found in sample")
                    if inference_style == "single":
                        prediction = model.infer(question=sample["question"], image_path=sample["image_path"])
                    else:
                        prediction = model.infer(question=sample["question"], images=sample["image_paths"])
                except Exception as exc:
                    prediction = f"ERROR: {exc}"
                    error_count += 1

                records.append(build_record(sample, prediction))
                total_samples += 1
                if total_samples % config.save_interval == 0:
                    save_predictions(output_path, records)

            wandb_module = get_wandb_module()
            if config.use_wandb and wandb_module is not None and wandb_module.run is not None:
                wandb_module.log(
                    {
                        f"{split_label}/processed_samples": total_samples,
                        f"{split_label}/error_count": error_count,
                    }
                )

    save_predictions(output_path, records)
    print(f"Saved {total_samples} records -> {output_path} (errors: {error_count})")

    wandb_module = get_wandb_module()
    if config.use_wandb and wandb_module is not None and wandb_module.run is not None:
        wandb_module.run.summary[f"{split_label}_output_path"] = str(output_path)
        wandb_module.run.summary[f"{split_label}_samples"] = total_samples
        wandb_module.run.summary[f"{split_label}_error_count"] = error_count
        if config.wandb_log_artifacts:
            log_wandb_file_artifact(
                artifact_name=build_artifact_name(config, f"{split_label}-predictions"),
                artifact_type="predictions",
                path=output_path,
                metadata={
                    "data_mode": config.data_mode,
                    "split_label": split_label,
                    "model_key": config.model_key,
                    "model_name": model.model_name,
                    "sample_count": total_samples,
                    "error_count": error_count,
                },
            )

    return {
        "output_path": output_path,
        "sample_count": total_samples,
        "error_count": error_count,
    }


def output_path_for_split(config: InferenceConfig, model_name: str, split_label: str) -> Path:
    output_root = Path(config.output_dir).resolve()
    if split_label == "test":
        return output_root / f"{model_name}.json"
    return output_root / split_label / f"{model_name}.json"


def initialise_runtime(
    config: InferenceConfig,
) -> tuple[dict[str, str], InferenceRuntimeDataBundle, Path]:
    set_seed(config.seed)
    effective_config_path = save_config_yaml(config, Path(config.output_dir) / "effective_config.yaml")
    data_bundle = build_inference_data_bundle(config)
    loaded_env: dict[str, str] = {}
    if config.use_wandb:
        loaded_env = init_wandb_run(config, effective_config_path)
        log_wandb_runtime_metadata(config, data_bundle, effective_config_path, loaded_env)
        if config.wandb_log_artifacts:
            log_wandb_file_artifact(
                artifact_name=build_artifact_name(config, "effective-config"),
                artifact_type="config",
                path=effective_config_path,
                metadata={"data_mode": config.data_mode, "stage": "inference"},
            )
    return loaded_env, data_bundle, effective_config_path
