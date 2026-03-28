from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def build_4bit_quant_config(enabled: bool) -> BitsAndBytesConfig | None:
    if not enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def is_local_peft_adapter(model_path: str) -> bool:
    resolved = Path(model_path).expanduser()
    return resolved.is_dir() and (resolved / "adapter_config.json").exists()


def _has_local_processor_files(model_path: str) -> bool:
    resolved = Path(model_path).expanduser()
    expected_files = (
        "processor_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    return resolved.is_dir() and any((resolved / filename).exists() for filename in expected_files)


def load_qwenvl_model_and_processor(
    model_path: str,
    *,
    load_test: bool = False,
    attn_implementation: str | None = "flash_attention_2",
):
    quant_config = build_4bit_quant_config(load_test)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": quant_config,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if is_local_peft_adapter(model_path):
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path).eval()
        processor_source = model_path if _has_local_processor_files(model_path) else peft_config.base_model_name_or_path
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs).eval()
        processor_source = model_path

    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "left"
    return model, processor
