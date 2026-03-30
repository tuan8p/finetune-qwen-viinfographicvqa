from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "ViInfographicVQA_dataset"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "config.yaml"
DEFAULT_WANDB_ENV_PATH = PROJECT_ROOT / "ft-qwen" / ".wandb.env"
DEFAULT_MODEL_REGISTRY = {
    "qwenvl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3_5-8B",
    "phi": "microsoft/Phi-4-multimodal-instruct",
    "minicpm": "openbmb/MiniCPM-o-2-6",
    "molmo": "allenai/Molmo-7B-D-0924",
    "ovis": "AIDC-AI/Ovis2.5-9B",
    "videollama": "DAMO-NLP-SG/VideoLLaMA3-7B-Image",
    "llava": "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
    "aya_vision": "CohereLabs/aya-vision-8b",
}


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    data_mode: str = "single"
    use_subdataset: bool = False
    filter_answers_over_20_tokens: bool = True
    seed: int = 42
    model_key: str = "qwenvl"
    model_path: str | None = None
    model_registry: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_REGISTRY))
    attn_implementation: str = "flash_attention_2"
    test_loading: bool = False
    batch_size: int = 1
    save_interval: int = 100
    output_dir: str = "results"
    dataloader_num_workers: int = 4
    use_wandb: bool = False
    wandb_env_file: str = str(DEFAULT_WANDB_ENV_PATH)
    wandb_project: str = "vqa-inference"
    wandb_entity: str | None = None
    wandb_group: str | None = "inference"
    wandb_job_type: str | None = "predict"
    wandb_run_name: str | None = None
    wandb_notes: str | None = None
    wandb_mode: str = "online"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_save_code: bool = True
    wandb_log_artifacts: bool = True


def _default_config_values() -> dict[str, Any]:
    defaults = asdict(InferenceConfig())
    if not defaults["wandb_tags"]:
        defaults["wandb_tags"] = ["viinfographicvqa", "inference"]
    return defaults


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing config file: {resolved_path}")

    loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {resolved_path}")
    if "wandb_tags" in loaded:
        if loaded["wandb_tags"] is None:
            loaded["wandb_tags"] = []
        elif isinstance(loaded["wandb_tags"], str):
            loaded["wandb_tags"] = [loaded["wandb_tags"]]
    return loaded


def build_config(config_path: str | Path | None = None, **overrides: Any) -> InferenceConfig:
    values = _default_config_values()
    values.update(load_yaml_config(config_path))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return InferenceConfig(**values)


def config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    return asdict(config)


def resolve_model_path(config: InferenceConfig) -> str:
    if config.model_path:
        return config.model_path
    if config.model_key not in config.model_registry:
        raise ValueError(
            f"Missing model path for key '{config.model_key}' in inference config. "
            f"Available keys: {sorted(config.model_registry.keys())}"
        )
    return config.model_registry[config.model_key]


def save_config_yaml(config: InferenceConfig, output_path: str | Path) -> Path:
    resolved_path = Path(output_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return resolved_path


def load_wandb_env_file(env_path: str | Path | None) -> dict[str, str]:
    if env_path is None:
        return {}

    resolved_path = Path(env_path).resolve()
    if not resolved_path.exists():
        return {}

    loaded_vars: dict[str, str] = {}
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        if key:
            loaded_vars[key] = os.environ.get(key, value)
    return loaded_vars
