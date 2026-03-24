from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_preprocessing.training import build_collate_fn, build_finetune_datasets


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
        help="Disable 20%% train sampling and 10%% valid split; use full raw train instead",
    )
    parser.add_argument(
        "--preview-batch",
        action="store_true",
        help="Build dataloaders and print one preview batch summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_dataset, valid_dataset, test_dataset = build_finetune_datasets(
        dataset_root=Path(args.dataset_root).resolve(),
        use_subdataset=not args.disable_subdataset,
        seed=args.seed,
    )

    print("Preprocessing summary")
    print(f"- dataset_root: {Path(args.dataset_root).resolve()}")
    print(f"- use_subdataset: {not args.disable_subdataset}")
    print(f"- seed: {args.seed}")
    print(f"- train_samples: {len(train_dataset)}")
    print(f"- valid_samples: {len(valid_dataset)}")
    print(f"- test_samples: {len(test_dataset)}")

    if args.preview_batch:
        try:
            from torch.utils.data import DataLoader
        except (ImportError, OSError) as exc:
            raise SystemExit(
                "PyTorch is required for --preview-batch because it builds DataLoader objects. "
                f"Original import error: {exc}"
            ) from exc

        collate_fn = build_collate_fn()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        print(f"- train_batches: {len(train_loader)}")
        print(f"- valid_batches: {len(valid_loader)}")
        print(f"- test_batches: {len(test_loader)}")

        for split_name, loader in (("train", train_loader), ("valid", valid_loader), ("test", test_loader)):
            if len(loader) == 0:
                print(f"- {split_name}_preview: empty")
                continue
            batch = next(iter(loader))
            print(f"- {split_name}_preview_batch_size: {len(batch['question_id'])}")
            print(f"- {split_name}_preview_keys: {', '.join(batch.keys())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
