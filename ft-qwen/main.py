from __future__ import annotations

from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from common.archive_utils import zip_directory_output
from common.utils import set_seed
from cli import build_runtime_config, parse_args
from config import save_config_yaml
from runtime_data import build_runtime_data_bundle
from trainer import train_qlora
from wandb_utils import get_wandb_module, init_wandb_run, log_wandb_runtime_metadata


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    set_seed(config.seed)
    wandb_started = False

    try:
        effective_config_path = save_config_yaml(config, Path(config.adapter_out_dir) / "effective_config.yaml")
        print(f"Effective config saved at    : {effective_config_path}")

        data_bundle = build_runtime_data_bundle(config)
        if config.use_wandb:
            loaded_env = init_wandb_run(config, effective_config_path)
            wandb_started = True
            log_wandb_runtime_metadata(config, data_bundle, effective_config_path, loaded_env)

        adapter_path = train_qlora(config, data_bundle)
        print(f"Training completed. Final adapter saved at: {adapter_path}")
        finetune_archive_path = zip_directory_output(config.adapter_out_dir)
        print(f"Finetune output archive saved at: {finetune_archive_path}")
        print("finish")
    finally:
        wandb_module = get_wandb_module()
        if config.use_wandb and wandb_started and wandb_module is not None:
            wandb_module.finish()


if __name__ == "__main__":
    main()
