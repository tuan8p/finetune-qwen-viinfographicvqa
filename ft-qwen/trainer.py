from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from config import QwenFinetuneConfig
from runtime_data import QwenRuntimeDataBundle, VQADataCollator
from wandb_utils import (
    build_artifact_name,
    log_best_checkpoint_metadata,
    log_wandb_dir_artifact,
    log_wandb_file_artifact,
)


class SaveMetricsCallback(TrainerCallback):
    """Ghi metrics train/eval ra <output_dir>/training_metrics.json."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_log: list[dict[str, Any]] = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            entry = {"epoch": state.epoch, "step": state.global_step}
            entry.update(metrics)
            self.metrics_log.append(entry)
            self._save()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            entry = {"epoch": state.epoch, "step": state.global_step}
            entry.update({key: value for key, value in logs.items()})
            self.metrics_log.append(entry)
            self._save()

    def _save(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(self.metrics_log, file, indent=2)


def train_qlora(config: QwenFinetuneConfig, data_bundle: QwenRuntimeDataBundle) -> str:
    print("\n[Bước 2] Bắt đầu training QLoRA...")

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        use_fast=True,
    )
    processor.tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = VQADataCollator(processor)
    has_validation = len(data_bundle.valid_dataset) > 0

    training_kwargs: dict[str, Any] = {
        "output_dir": config.adapter_out_dir,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "gradient_checkpointing": True,
        "learning_rate": config.lr,
        "bf16": config.bf16,
        "optim": "paged_adamw_32bit",
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "logging_steps": config.logging_steps,
        "report_to": "wandb" if config.use_wandb else "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.pin_memory,
    }
    if "dataloader_persistent_workers" in TrainingArguments.__dataclass_fields__:
        training_kwargs["dataloader_persistent_workers"] = bool(
            config.persistent_workers and config.dataloader_num_workers > 0
        )
    if has_validation:
        training_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": config.save_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    else:
        training_kwargs.update({"eval_strategy": "no", "load_best_model_at_end": False})

    training_args = TrainingArguments(**training_kwargs)
    metrics_callback = SaveMetricsCallback(output_dir=config.adapter_out_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=data_bundle.train_dataset,
        eval_dataset=data_bundle.valid_dataset if has_validation else None,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        callbacks=[metrics_callback],
    )

    trainer.train()
    log_best_checkpoint_metadata(config, trainer, has_validation)

    final_adapter_path = os.path.join(config.adapter_out_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    processor.save_pretrained(final_adapter_path)
    print(f"Đã lưu LoRA adapter tại      : {final_adapter_path}")
    print(f"Metrics đã lưu tại           : {os.path.join(config.adapter_out_dir, 'training_metrics.json')}")

    if config.use_wandb and config.wandb_log_artifacts:
        metrics_path = Path(config.adapter_out_dir) / "training_metrics.json"
        effective_config_path = Path(config.adapter_out_dir) / "effective_config.yaml"

        log_wandb_file_artifact(
            artifact_name=build_artifact_name(config, "effective-config"),
            artifact_type="config",
            path=effective_config_path,
            metadata={"data_mode": config.data_mode, "stage": "post-train"},
        )
        log_wandb_file_artifact(
            artifact_name=build_artifact_name(config, "training-metrics"),
            artifact_type="metrics",
            path=metrics_path,
            metadata={"data_mode": config.data_mode, "stage": "post-train"},
        )
        if config.wandb_log_final_checkpoint:
            log_wandb_dir_artifact(
                artifact_name=build_artifact_name(config, "final-adapter"),
                artifact_type="model",
                path=final_adapter_path,
                metadata={
                    "data_mode": config.data_mode,
                    "model_id": config.model_id,
                    "artifact_role": "final_checkpoint",
                    "train_samples": len(data_bundle.train_dataset),
                    "valid_samples": len(data_bundle.valid_dataset),
                },
            )

    del model
    del trainer
    torch.cuda.empty_cache()
    return final_adapter_path
