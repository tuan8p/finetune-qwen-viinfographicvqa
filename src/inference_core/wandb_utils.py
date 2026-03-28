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

from src.inference_core.config import InferenceConfig, config_to_dict, load_wandb_env_file
from src.inference_core.runtime_data import InferenceRuntimeDataBundle


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


def build_run_slug(config: InferenceConfig) -> str:
    model_name = slugify(config.model_path or config.model_key)
    subdataset_suffix = "sub" if config.use_subdataset else "full"
    return f"{model_name}-{config.data_mode}-{subdataset_suffix}-bs{config.batch_size}-seed{config.seed}"


def build_wandb_tags(config: InferenceConfig) -> list[str]:
    tags = list(config.wandb_tags)
    tags.extend(
        [
            f"data_mode:{config.data_mode}",
            f"subdataset:{'on' if config.use_subdataset else 'off'}",
            f"model:{config.model_key}",
        ]
    )
    return sorted(set(tags))


def build_wandb_run_name(config: InferenceConfig) -> str:
    return config.wandb_run_name or build_run_slug(config)


def build_artifact_name(config: InferenceConfig, artifact_role: str) -> str:
    project_name = slugify(config.wandb_project or "wandb")
    return f"{project_name}-{build_run_slug(config)}-{slugify(artifact_role)}"


def init_wandb_run(config: InferenceConfig, effective_config_path: Path) -> dict[str, str]:
    wandb_module = require_wandb()
    loaded_env = load_wandb_env_file(config.wandb_env_file)
    if config.wandb_mode == "online" and "WANDB_API_KEY" not in os.environ:
        print(
            "WANDB_API_KEY không có trong environment. Sẽ thử dùng phiên đăng nhập WandB hiện có "
            "(netrc/credential store) nếu server đã đăng nhập sẵn."
        )

    entity = config.wandb_entity or os.environ.get("WANDB_ENTITY")
    project = config.wandb_project or os.environ.get("WANDB_PROJECT")
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
        dir=str(Path(config.output_dir).resolve()),
        config=config_to_dict(config),
    )
    wandb_module.config.update({"effective_config_path": str(effective_config_path)}, allow_val_change=True)
    return loaded_env


def log_wandb_runtime_metadata(
    config: InferenceConfig,
    data_bundle: InferenceRuntimeDataBundle,
    effective_config_path: Path,
    loaded_env: dict[str, str],
) -> None:
    wandb_module = require_wandb()
    metadata = {
        "effective_config_path": str(effective_config_path),
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "hostname": socket.gethostname(),
        "test_samples": len(data_bundle.test_dataset),
        "extra_test_samples": len(data_bundle.extra_test_dataset) if data_bundle.extra_test_dataset is not None else 0,
        "wandb_entity_effective": config.wandb_entity or loaded_env.get("WANDB_ENTITY", ""),
        "wandb_project_effective": config.wandb_project or loaded_env.get("WANDB_PROJECT", ""),
    }
    wandb_module.config.update(metadata, allow_val_change=True)
    for key, value in metadata.items():
        wandb_module.run.summary[key] = value


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
