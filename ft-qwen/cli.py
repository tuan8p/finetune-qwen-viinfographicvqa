from __future__ import annotations

import argparse
from dataclasses import replace

from bootstrap import ensure_src_path

ensure_src_path()

from common.output_naming import resolve_stage_output_dir
from config import DEFAULT_CONFIG_PATH, QwenFinetuneConfig, build_config
from preprocessing.training import DATA_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified QLoRA Pipeline for Qwen-VL with preprocessing integration")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--data_mode",
        "--data-mode",
        dest="data_mode",
        choices=DATA_MODES,
        default=None,
        help="Training data mode: single, multi, or single_and_multi",
    )
    parser.add_argument("--task", dest="data_mode", choices=DATA_MODES, help="Deprecated alias for --data_mode")
    parser.add_argument(
        "--dataset_root",
        "--dataset-root",
        "--root_dir",
        dest="dataset_root",
        type=str,
        default=None,
        help="Path to ViInfographicVQA_dataset root containing data/ and images/",
    )
    parser.add_argument("--disable-subdataset", action="store_true")
    parser.add_argument(
        "--no-filter-answers-over-20-tokens",
        action="store_true",
        help="Keep all samples regardless of answer length (overrides config filter_answers_over_20_tokens)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapter_out_dir", "--adapter-out-dir", dest="adapter_out_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-notes", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--wandb-env-file", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=None)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> QwenFinetuneConfig:
    use_subdataset = None if not args.disable_subdataset else False
    use_wandb = True if args.use_wandb else None
    filter_answers_over_20_tokens = False if args.no_filter_answers_over_20_tokens else None
    config = build_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        data_mode=args.data_mode,
        use_subdataset=use_subdataset,
        filter_answers_over_20_tokens=filter_answers_over_20_tokens,
        adapter_out_dir=args.adapter_out_dir,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataloader_num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_run_name=args.wandb_run_name,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
        wandb_env_file=args.wandb_env_file,
        wandb_tags=args.wandb_tags,
    )
    named_adapter_out_dir = resolve_stage_output_dir(
        config.adapter_out_dir,
        stage="finetune",
        model_ref=config.model_id,
        data_mode=config.data_mode,
    )
    return replace(config, adapter_out_dir=str(named_adapter_out_dir))
