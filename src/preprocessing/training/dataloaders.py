from __future__ import annotations

import math
import random

from .collate import build_collate_fn
from .contracts import FinetuneDataLoaderBundle
from .dataset import build_finetune_dataset_bundle


try:
    from torch.utils.data import DataLoader
except (ImportError, OSError):
    class DataLoader:  # type: ignore[no-redef]
        def __init__(self, dataset, batch_size: int, shuffle: bool, collate_fn):
            self.dataset = dataset
            self.batch_size = max(1, batch_size)
            self.shuffle = shuffle
            self.collate_fn = collate_fn

        def __len__(self) -> int:
            if len(self.dataset) == 0:
                return 0
            return math.ceil(len(self.dataset) / self.batch_size)

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng = random.Random(42)
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch = [self.dataset[index] for index in batch_indices]
                yield self.collate_fn(batch)


def build_finetune_dataloaders(
    dataset_root,
    batch_size: int = 4,
    use_subdataset: bool = True,
    seed: int = 42,
    data_mode: str = "single_and_multi",
    shuffle_train: bool = True,
) -> FinetuneDataLoaderBundle:
    bundle = build_finetune_dataset_bundle(
        dataset_root=dataset_root,
        use_subdataset=use_subdataset,
        seed=seed,
        data_mode=data_mode,
    )
    collate_fn = build_collate_fn()

    train_loader = DataLoader(bundle.train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
    valid_loader = DataLoader(bundle.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(bundle.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    extra_test_loader = (
        DataLoader(bundle.extra_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        if bundle.extra_test_dataset is not None
        else None
    )

    return FinetuneDataLoaderBundle(
        data_mode=data_mode,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        extra_test_loader=extra_test_loader,
        extra_test_name=bundle.extra_test_name,
    )
