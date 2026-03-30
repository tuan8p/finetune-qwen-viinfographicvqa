from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.training import DATA_MODES, build_finetune_dataloaders, build_finetune_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline for ViInfographicVQA")
    parser.add_argument(
        "--dataset-root",
        default=str(CURRENT_DIR.parent / "ViInfographicVQA_dataset"),
        help="Path to the local ViInfographicVQA_dataset directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subdataset sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for dataloader preview",
    )
    parser.add_argument(
        "--disable-subdataset",
        action="store_true",
        help="Disable subdataset mode; split valid directly from the full raw train instead",
    )
    parser.add_argument(
        "--data-mode",
        choices=DATA_MODES,
        default="single_and_multi",
        help="Dataset mode: single-only, multi-only, or mixed; mixed mode also exposes an extra single-only test loader",
    )
    parser.add_argument(
        "--preview-batch",
        action="store_true",
        help="Build dataloaders and print one preview batch summary",
    )
    parser.add_argument(
        "--no-filter-answers-over-20-tokens",
        action="store_true",
        help="Keep samples whose normalized answer has more than 20 whitespace tokens",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    bundle = build_finetune_dataset_bundle(
        dataset_root=Path(args.dataset_root).resolve(),
        use_subdataset=not args.disable_subdataset,
        seed=args.seed,
        data_mode=args.data_mode,
        filter_answers_over_20_tokens=not args.no_filter_answers_over_20_tokens,
    )

    print("Preprocessing summary")
    print(f"- dataset_root: {Path(args.dataset_root).resolve()}")
    print(f"- use_subdataset: {not args.disable_subdataset}")
    print(f"- seed: {args.seed}")
    print(f"- data_mode: {args.data_mode}")
    print(f"- filter_answers_over_20_tokens: {not args.no_filter_answers_over_20_tokens}")
    print(f"- train_samples: {len(bundle.train_dataset)}")
    print(f"- valid_samples: {len(bundle.valid_dataset)}")
    print(f"- test_all_samples: {len(bundle.test_dataset)}")
    if bundle.extra_test_dataset is not None:
        print(f"- {bundle.extra_test_name}_samples: {len(bundle.extra_test_dataset)}")

    if args.preview_batch:
        dataloader_bundle = build_finetune_dataloaders(
            dataset_root=Path(args.dataset_root).resolve(),
            batch_size=args.batch_size,
            use_subdataset=not args.disable_subdataset,
            seed=args.seed,
            filter_answers_over_20_tokens=not args.no_filter_answers_over_20_tokens,
            data_mode=args.data_mode,
        )

        print(f"- train_batches: {len(dataloader_bundle.train_loader)}")
        print(f"- valid_batches: {len(dataloader_bundle.valid_loader)}")
        print(f"- test_all_batches: {len(dataloader_bundle.test_loader)}")
        if dataloader_bundle.extra_test_loader is not None:
            print(f"- {dataloader_bundle.extra_test_name}_batches: {len(dataloader_bundle.extra_test_loader)}")

        loaders = [
            ("train", dataloader_bundle.train_loader),
            ("valid", dataloader_bundle.valid_loader),
            ("test_all", dataloader_bundle.test_loader),
        ]
        if dataloader_bundle.extra_test_loader is not None:
            loaders.append((str(dataloader_bundle.extra_test_name), dataloader_bundle.extra_test_loader))

        for split_name, loader in loaders:
            if len(loader) == 0:
                print(f"- {split_name}_preview: empty")
                continue
            batch = next(iter(loader))
            print(f"- {split_name}_preview_batch_size: {len(batch['question_id'])}")
            print(f"- {split_name}_preview_keys: {', '.join(batch.keys())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
