from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bootstrap import ensure_src_path

ensure_src_path()

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from config import QwenFinetuneConfig
from preprocessing.training import FinetuneDatasetBundle, build_finetune_dataset_bundle


@dataclass(frozen=True, slots=True)
class QwenRuntimeDataBundle:
    data_mode: str
    train_dataset: Dataset
    valid_dataset: Dataset


class QwenVLPreprocessedDataset(Dataset):
    """Adapter from preprocessing dataset output to Qwen finetuning samples."""

    def __init__(self, preprocessing_dataset: Dataset, dataset_root: str | Path, system_prompt: str) -> None:
        self.preprocessing_dataset = preprocessing_dataset
        self.dataset_root = Path(dataset_root).resolve()
        self.system_prompt = system_prompt
        self.samples = [self._adapt_sample(preprocessing_dataset[index]) for index in range(len(preprocessing_dataset))]

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_paths(self, image_paths: tuple[str, ...]) -> list[str]:
        return [str((self.dataset_root / relative_path).resolve()) for relative_path in image_paths]

    def _adapt_sample(self, sample) -> dict[str, Any]:
        image_paths = self._resolve_image_paths(sample.image_paths)
        user_prompt = f"Question: {sample.question_normalized}\nAnswer:"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": image_path} for image_path in image_paths]
                + [{"type": "text", "text": user_prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample.answer_normalized}]},
        ]
        return {
            "question_id": sample.question_id,
            "split_name": sample.split_name,
            "subdataset_split": sample.subdataset_split,
            "task_family": sample.task_family,
            "image_type": sample.image_type,
            "answer_source": sample.answer_source,
            "element": sample.element or "",
            "image_paths": image_paths,
            "image_count": sample.image_count,
            "question_text": sample.question_normalized,
            "answer_text": sample.answer_normalized,
            "raw_question": sample.raw_question,
            "raw_answer": sample.raw_answer,
            "answer_type": sample.answer_type,
            "question_type": sample.question_type,
            "reasoning_mode": sample.reasoning_mode,
            "cross_image_dependency": sample.cross_image_dependency,
            "diacritic_consistency": sample.diacritic_consistency,
            "tokenization_flags": sample.tokenization_flags,
            "answer_segments": sample.answer_segments,
            "user_prompt": user_prompt,
            "messages": messages,
            "generation_messages": messages[:2],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class VQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.assistant_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_role_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        texts: list[str] = []
        image_inputs_list: list[list[Any] | None] = []

        for item in batch:
            messages = item["messages"]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        all_images = [image for images in image_inputs_list for image in (images or [])]
        inputs = self.processor(text=texts, images=all_images if all_images else None, padding=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()

        for row_index, row in enumerate(labels):
            ids = row.tolist()
            mask_until = 0
            for pos in range(len(ids) - len(self.assistant_role_ids) - 1):
                if (
                    ids[pos] == self.assistant_start_id
                    and ids[pos + 1 : pos + 1 + len(self.assistant_role_ids)] == self.assistant_role_ids
                ):
                    mask_until = pos + 1 + len(self.assistant_role_ids)
            labels[row_index, :mask_until] = -100

        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs


def build_runtime_data_bundle(config: QwenFinetuneConfig) -> QwenRuntimeDataBundle:
    preprocessing_bundle: FinetuneDatasetBundle = build_finetune_dataset_bundle(
        dataset_root=Path(config.dataset_root).resolve(),
        use_subdataset=config.use_subdataset,
        seed=config.seed,
        data_mode=config.data_mode,
    )

    bundle = QwenRuntimeDataBundle(
        data_mode=config.data_mode,
        train_dataset=QwenVLPreprocessedDataset(preprocessing_bundle.train_dataset, config.dataset_root, config.system_prompt),
        valid_dataset=QwenVLPreprocessedDataset(preprocessing_bundle.valid_dataset, config.dataset_root, config.system_prompt),
    )

    print("\n[Bước 1] Chuẩn bị dữ liệu bằng preprocessing pipeline...")
    print(f"- data_mode: {bundle.data_mode}")
    print(f"- use_subdataset: {config.use_subdataset}")
    print(f"- train_samples: {len(bundle.train_dataset)}")
    print(f"- valid_samples: {len(bundle.valid_dataset)}")
    return bundle
