"""Helpers for consistent output directory naming."""

from __future__ import annotations

import re
from pathlib import Path


MODEL_ALIAS_MAP = {
    "qwenvl": "qwen",
    "qwen2.5-vl": "qwen",
    "qwen2_5_vl": "qwen",
    "qwen": "qwen",
    "ovis": "ovis",
    "internvl": "internvl",
    "phi": "phi",
    "minicpm": "minicpm",
    "molmo": "molmo",
    "videollama": "videollama",
    "llava": "llava",
    "aya_vision": "aya_vision",
}


def resolve_model_alias(model_ref: str) -> str:
    normalized = model_ref.strip().lower().replace("\\", "/")
    leaf_name = normalized.split("/")[-1]

    for needle, alias in MODEL_ALIAS_MAP.items():
        if needle in normalized or needle in leaf_name:
            return alias

    fallback = re.sub(r"[^a-z0-9]+", "_", leaf_name).strip("_")
    return fallback or "model"


def build_stage_output_name(stage: str, model_ref: str, data_mode: str) -> str:
    return f"{stage}_{resolve_model_alias(model_ref)}_{data_mode}"


def resolve_stage_output_dir(base_path: str | Path, *, stage: str, model_ref: str, data_mode: str) -> Path:
    candidate = Path(base_path)
    output_name = build_stage_output_name(stage, model_ref, data_mode)

    if not candidate.is_absolute() and len(candidate.parts) == 1:
        return candidate / output_name
    return candidate.parent / output_name
