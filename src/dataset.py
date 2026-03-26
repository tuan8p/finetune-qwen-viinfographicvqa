import os
import json
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Single-image dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    """Dataset for single-image VQA.

    JSON format expected per item:
        {
            "question_id": "...",
            "image_path":  "images/xxx.jpg",   # relative to data_dir
            "question":    "...",
            "answer":      "...",
            "image_type":  "...",               # optional
            "answer_source": "...",             # optional
            "element":     "..." | [...]        # optional
        }
    """

    def __init__(self, json_path: str, data_dir: str, transform=None, vocab=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = data_dir
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item["image_path"])
        question = item["question"]
        answer   = item["answer"]

        element = item.get("element", "")
        if isinstance(element, list):
            element = ", ".join(element)

        return {
            "image_path":    img_path,
            "question":      question,
            "raw_answer":    answer,
            "question_id":   item.get("question_id",   str(idx)),
            "image_type":    item.get("image_type",    ""),
            "answer_source": item.get("answer_source", ""),
            "element":       element,
        }


# ---------------------------------------------------------------------------
# Multi-image dataset
# ---------------------------------------------------------------------------

class MultiImageVQADataset(Dataset):
    """Dataset for multi-image VQA.

    JSON format expected per item:
        {
            "question_id":  "...",
            "image_paths":  ["images/a.jpg", "images/b.jpg", ...],  # relative to data_dir
            "question":     "...",
            "answer":       "...",
            "image_type":   "...",          # optional
            "answer_source": "...",         # optional
            "element":      "..." | [...]   # optional
        }
    """

    def __init__(self, json_path: str, data_dir: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Resolve every image path relative to data_dir
        raw_paths = item.get("image_paths", [])
        image_paths = [os.path.join(self.data_dir, p) for p in raw_paths]

        element = item.get("element", "")
        if isinstance(element, list):
            element = ", ".join(element)

        return {
            "image_paths":   image_paths,          # list[str]
            "question":      item["question"],
            "raw_answer":    item["answer"],
            "question_id":   item.get("question_id",   str(idx)),
            "image_type":    item.get("image_type",    ""),
            "answer_source": item.get("answer_source", ""),
            "element":       element,
        }


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------

def custom_collate_fn(batch):
    """Collate for single-image: all values are flat lists."""
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


def multi_image_collate_fn(batch):
    """Collate for multi-image: image_paths is a list-of-lists, rest are flat lists."""
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


# ---------------------------------------------------------------------------
# Utilities (kept for compatibility)
# ---------------------------------------------------------------------------

def get_transforms(img_size: int = 224):
    """Return train and val transform pipelines."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, val_transform


def build_answer_vocab(train_data: list, val_data: list, max_answers: int = 10000) -> dict:
    """Build answer vocabulary from train + val splits."""
    all_answers = []
    for item in train_data:
        all_answers.extend(item.get("answers", [item["answer"]]))
    for item in val_data:
        all_answers.extend(item.get("answers", [item["answer"]]))

    answer_counts = Counter(all_answers)
    unique_answers = list(answer_counts.keys())

    if len(unique_answers) > max_answers:
        unique_answers = [ans for ans, _ in answer_counts.most_common(max_answers)]

    vocab = {ans: idx for idx, ans in enumerate(sorted(unique_answers))}
    return vocab