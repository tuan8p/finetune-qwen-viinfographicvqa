from __future__ import annotations

from src.common.archive_utils import zip_directory_output
from src.inference_core.runner_utils import (
    build_arg_parser,
    build_runtime_config,
    get_wandb_module,
    import_model_class,
    initialise_runtime,
    output_path_for_split,
    run_prediction_loop,
)
from src.inference_core.config import resolve_model_path


MODELS = {
    "internvl": "src.inference.single_and_multi.models.internvl.InternVLModel",
    "qwenvl": "src.inference.single_and_multi.models.qwenvl.QwenVLModel",
    "ovis": "src.inference.single_and_multi.models.ovis.OvisModel",
}


def parse_args():
    parser = build_arg_parser("Run single+multi-image VQA inference", MODELS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_runtime_config(
        args,
        data_mode="single_and_multi",
        default_output_subdir="results/single_and_multi",
    )

    wandb_started = False
    try:
        _loaded_env, data_bundle, effective_config_path = initialise_runtime(config)
        model_class = import_model_class(MODELS, config.model_key)
        model_path = resolve_model_path(config)
        model = model_class(
            model_path=model_path,
            load_test=config.test_loading,
            attn_implementation=config.attn_implementation,
        )
        wandb_module = get_wandb_module()
        if config.use_wandb and wandb_module is not None and wandb_module.run is not None:
            wandb_started = True
            wandb_module.config.update({"effective_config_path": str(effective_config_path)}, allow_val_change=True)

        run_prediction_loop(
            model=model,
            dataset=data_bundle.test_dataset,
            config=config,
            output_path=output_path_for_split(config, model.model_name, "test_all"),
            split_label="test_all",
            inference_style="multi",
        )

        if data_bundle.extra_test_dataset is not None and data_bundle.extra_test_name is not None:
            run_prediction_loop(
                model=model,
                dataset=data_bundle.extra_test_dataset,
                config=config,
                output_path=output_path_for_split(config, model.model_name, data_bundle.extra_test_name),
                split_label=data_bundle.extra_test_name,
                inference_style="multi",
            )

        inference_archive_path = zip_directory_output(config.output_dir)
        print(f"Inference output archive saved at: {inference_archive_path}")
        print("Completed single-and-multi inference.")
        return 0
    finally:
        wandb_module = get_wandb_module()
        if config.use_wandb and wandb_started and wandb_module is not None:
            wandb_module.finish()


if __name__ == "__main__":
    raise SystemExit(main())
