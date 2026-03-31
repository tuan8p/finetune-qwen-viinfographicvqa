from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import socket
from pathlib import Path
from typing import Any

# wandb.sdk.artifacts._validators.validate_artifact_name
WANDB_ARTIFACT_NAME_MAX_LEN = 128

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from src.inference_core.config import InferenceConfig, config_to_dict, load_wandb_env_file
from src.inference_core.runtime_data import InferenceRuntimeDataBundle

# Load output_naming trực tiếp — tránh `import src.common` (kéo torch qua common/__init__.py).
_output_naming_mod: Any = None


def _build_stage_output_name(stage: str, model_ref: str, data_mode: str) -> str:
    global _output_naming_mod
    if _output_naming_mod is None:
        path = Path(__file__).resolve().parent.parent / "common" / "output_naming.py"
        spec = importlib.util.spec_from_file_location("_viinfographic_output_naming", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load output_naming from {path}")
        _output_naming_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_output_naming_mod)
    return _output_naming_mod.build_stage_output_name(stage, model_ref, data_mode)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "run"


def _truncate_wandb_artifact_name(name: str, max_len: int = WANDB_ARTIFACT_NAME_MAX_LEN) -> str:
    if len(name) <= max_len:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:12]
    suffix = f"-{digest}"
    return name[: max_len - len(suffix)] + suffix


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
    """Cùng quy tắc với thư mục output (vd. inference_qwen_single); không dùng model_path."""
    return _build_stage_output_name("inference", config.model_key, config.data_mode).lower()


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
    name = config.wandb_run_name or build_run_slug(config)
    return _truncate_wandb_artifact_name(name)


def build_artifact_name(config: InferenceConfig, artifact_role: str) -> str:
    # Chỉ run_slug + role; không ghép thêm project prefix (tránh dài). Project đã có trên WandB UI.
    raw = f"{build_run_slug(config)}-{slugify(artifact_role)}"
    return _truncate_wandb_artifact_name(raw)


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
