from __future__ import annotations

import os
import re
import socket
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from config import QwenFinetuneConfig, config_to_dict, load_wandb_env_file
from runtime_data import QwenRuntimeDataBundle


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "run"


def get_wandb_module():
    return wandb


def require_wandb():
    if wandb is None:
        raise RuntimeError(
            "WandB chưa được cài trong environment hiện tại. "
            "Hãy cài package `wandb` hoặc tắt `use_wandb` trong config."
        )
    return wandb


def short_model_name(model_id: str) -> str:
    return slugify(model_id.split("/")[-1])


def build_run_slug(config: QwenFinetuneConfig) -> str:
    subdataset_suffix = "sub" if config.use_subdataset else "full"
    return (
        f"{short_model_name(config.model_id)}"
        f"-{config.data_mode}"
        f"-{subdataset_suffix}"
        f"-bs{config.batch_size}"
        f"-ga{config.grad_accum}"
        f"-seed{config.seed}"
    )


def build_wandb_tags(config: QwenFinetuneConfig) -> list[str]:
    tags = list(config.wandb_tags)
    tags.extend(
        [
            f"data_mode:{config.data_mode}",
            f"subdataset:{'on' if config.use_subdataset else 'off'}",
            f"attn:{config.attn_implementation}",
            f"quant:{'4bit' if config.load_in_4bit else 'full'}",
        ]
    )
    return sorted(set(tags))


def build_wandb_run_name(config: QwenFinetuneConfig) -> str:
    return config.wandb_run_name or build_run_slug(config)


def build_artifact_name(config: QwenFinetuneConfig, artifact_role: str) -> str:
    project_name = slugify(config.wandb_project or "wandb")
    return f"{project_name}-{build_run_slug(config)}-{slugify(artifact_role)}"


def init_wandb_run(config: QwenFinetuneConfig, effective_config_path: Path) -> dict[str, str]:
    wandb_module = require_wandb()
    loaded_env = load_wandb_env_file(config.wandb_env_file)

    entity = config.wandb_entity or os.environ.get("WANDB_ENTITY")
    project = config.wandb_project or os.environ.get("WANDB_PROJECT")
    if config.wandb_mode == "online" and "WANDB_API_KEY" not in os.environ:
        print(
            "WANDB_API_KEY không có trong environment. Sẽ thử dùng phiên đăng nhập WandB hiện có "
            "(netrc/credential store) nếu server đã đăng nhập sẵn."
        )
    wandb_module.init(
        entity=entity,
        project=project,
        group=config.wandb_group,
        job_type=config.wandb_job_type,
        name=build_wandb_run_name(config),
        tags=build_wandb_tags(config),
        notes=config.wandb_notes,
        mode=config.wandb_mode,
        save_code=config.wandb_save_code,
        dir=str(Path(config.adapter_out_dir).resolve()),
        config=config_to_dict(config),
    )
    wandb_module.config.update({"effective_config_path": str(effective_config_path)}, allow_val_change=True)
    return loaded_env


def log_wandb_runtime_metadata(
    config: QwenFinetuneConfig,
    data_bundle: QwenRuntimeDataBundle,
    effective_config_path: Path,
    loaded_env: dict[str, str],
) -> None:
    wandb_module = require_wandb()
    effective_batch_size = config.batch_size * config.grad_accum
    metadata = {
        "effective_config_path": str(effective_config_path),
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "adapter_out_dir": str(Path(config.adapter_out_dir).resolve()),
        "hostname": socket.gethostname(),
        "train_samples": len(data_bundle.train_dataset),
        "valid_samples": len(data_bundle.valid_dataset),
        "effective_batch_size": effective_batch_size,
        "wandb_entity_effective": config.wandb_entity or loaded_env.get("WANDB_ENTITY", ""),
        "wandb_project_effective": config.wandb_project or loaded_env.get("WANDB_PROJECT", ""),
    }
    wandb_module.config.update(metadata, allow_val_change=True)
    for key, value in metadata.items():
        wandb_module.run.summary[key] = value


def log_best_checkpoint_metadata(config: QwenFinetuneConfig, trainer, has_validation: bool) -> None:
    if not has_validation:
        return

    best_metric_name = "eval_loss"
    best_metric_value = trainer.state.best_metric
    best_checkpoint_path = trainer.state.best_model_checkpoint

    print(f"Best checkpoint selection metric : {best_metric_name} (lower is better)")
    print(f"Best validation metric value     : {best_metric_value}")
    print(f"Best checkpoint path             : {best_checkpoint_path}")

    if config.use_wandb and wandb is not None and wandb.run is not None:
        wandb_module = require_wandb()
        wandb_module.run.summary["best_metric_name"] = best_metric_name
        wandb_module.run.summary["best_metric_mode"] = "min"
        wandb_module.run.summary["best_metric_value"] = best_metric_value
        wandb_module.run.summary["best_model_checkpoint"] = best_checkpoint_path


def log_wandb_file_artifact(
    artifact_name: str,
    artifact_type: str,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    wandb_module = require_wandb()
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        return

    artifact = wandb_module.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(str(resolved_path), name=resolved_path.name)
    wandb_module.log_artifact(artifact)


def log_wandb_dir_artifact(
    artifact_name: str,
    artifact_type: str,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    wandb_module = require_wandb()
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        return

    artifact = wandb_module.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})
    artifact.add_dir(str(resolved_path), name=resolved_path.name)
    wandb_module.log_artifact(artifact)
