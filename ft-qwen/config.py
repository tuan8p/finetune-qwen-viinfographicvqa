from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "ViInfographicVQA_dataset"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "config.yaml"
DEFAULT_WANDB_ENV_PATH = CURRENT_DIR / ".wandb.env"


@dataclass(frozen=True, slots=True)
class QwenFinetuneConfig:
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    data_mode: str = "single"
    use_subdataset: bool = True
    adapter_out_dir: str = "checkpoints/lora_adapters"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    system_prompt: str = (
        "Answer the following question based solely on the image content "
        "concisely with a single term."
    )
    attn_implementation: str = "flash_attention_2"
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    max_seq_length: int = 1024
    use_wandb: bool = False
    wandb_env_file: str = str(DEFAULT_WANDB_ENV_PATH)
    wandb_project: str = "vqa-qlora"
    wandb_entity: str | None = None
    wandb_group: str | None = "qwen-finetune"
    wandb_job_type: str | None = "train"
    wandb_run_name: str | None = None
    wandb_notes: str | None = None
    wandb_mode: str = "online"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_save_code: bool = True
    wandb_log_artifacts: bool = True
    wandb_log_final_checkpoint: bool = True
    bf16: bool = True
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


def _default_config_values() -> dict[str, Any]:
    defaults = asdict(QwenFinetuneConfig())
    if not defaults["wandb_tags"]:
        defaults["wandb_tags"] = ["qwen2.5-vl", "viinfographicvqa", "qlora"]
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


def build_config(config_path: str | Path | None = None, **overrides: Any) -> QwenFinetuneConfig:
    values = _default_config_values()
    values.update(load_yaml_config(config_path))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return QwenFinetuneConfig(**values)


def config_to_dict(config: QwenFinetuneConfig) -> dict[str, Any]:
    return asdict(config)


def save_config_yaml(config: QwenFinetuneConfig, output_path: str | Path) -> Path:
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
