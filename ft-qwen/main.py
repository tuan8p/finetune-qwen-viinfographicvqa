import argparse
import json
import os
import random
import copy
from typing import Any
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SYSTEM_PROMPT = (
    "Answer the following question based solely on the image content "
    "concisely with a single term."
)


def convert_to_chat_format(item: dict, task: str) -> dict:
    if task == "single":
        user_content = [
            {"type": "image", "image": item["image_path"]},
            {"type": "text", "text": f"Question: {item['question']}\nAnswer:"},
        ]
    else: # multi
        user_content = [{"type": "image", "image": p} for p in item["image_paths"]]
        user_content.append({"type": "text", "text": f"Question: {item['question']}\nAnswer:"})

    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]},
        ]
    }

class SingleVQADataset(Dataset):
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

def prepare_data(args):
    """Đọc JSON gốc, chia Train/Val và lưu lại (không cần convert nữa)."""
    print("\n[Bước 1] Chuẩn bị dữ liệu...")
    with open(args.raw_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    random.seed(args.seed)
    random.shuffle(raw_data)
    val_size = max(1, int(len(raw_data) * args.val_ratio))
    train_data, val_data = raw_data[val_size:], raw_data[:val_size]

    os.makedirs(args.data_out_dir, exist_ok=True)
    train_path = os.path.join(args.data_out_dir, f"{args.task}_train.json")
    val_path = os.path.join(args.data_out_dir, f"{args.task}_val.json")

    with open(train_path, "w", encoding="utf-8") as f: json.dump(train_data, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f: json.dump(val_data, f, indent=2)
    
    print(f"Đã lưu {len(train_data)} mẫu Train vào {train_path}")
    print(f"Đã lưu {len(val_data)} mẫu Val vào {val_path}")
    return train_path, val_path

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



class VQADataCollator:
    # FIX 3: Bỏ tham số img_dir thừa — constructor chỉ cần processor
    def __init__(self, processor):
        self.processor = processor
        self.system_prompt = SYSTEM_PROMPT

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        texts, image_inputs_list = [], []

        for item in batch:
            user_content = []
            
            if "image_paths" in item:
                for img_path in item["image_paths"]:
                    user_content.append({"type": "image", "image": img_path})
            elif "image_path" in item:
                user_content.append({"type": "image", "image": item["image_path"]})
                
            user_content.append({"type": "text", "text": f"Question: {item['question']}\nAnswer:"})
            
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": item["raw_answer"]}]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        all_images = [img for imgs in image_inputs_list for img in (imgs or [])]
        
        inputs = self.processor(
            text=texts, 
            images=all_images if all_images else None, 
            padding=True, 
            return_tensors="pt"
        )
        # label masking
        labels = inputs["input_ids"].clone()
        start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        role_ids = self.processor.tokenizer.encode("assistant\n", add_special_tokens=False)

        for i, row in enumerate(labels):
            ids = row.tolist()
            mask_until = 0
            for pos in range(len(ids) - len(role_ids) - 1):
                if ids[pos] == start_id and ids[pos + 1 : pos + 1 + len(role_ids)] == role_ids:
                    mask_until = pos + 1 + len(role_ids)
            labels[i, :mask_until] = -100
        
        labels[inputs["attention_mask"] == 0] = -100
        
        inputs["labels"] = labels
        return inputs


def train_qlora(args, train_path, val_path):
    
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=256*28*28, max_pixels=1280*28*28, use_fast=True)
    processor.tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # FIX 1 & 2: Dùng đúng dataset theo task, bỏ VQAFinetuneDataset không tồn tại
    # FIX 4: Dùng args.img_dir thay vì args.data_dir
    if args.task == "single":
        train_dataset = SingleVQADataset(train_path, args.img_dir)
        val_dataset   = SingleVQADataset(val_path,   args.img_dir)
    else:  # multi
        train_dataset = MultiImageVQADataset(train_path, args.img_dir)
        val_dataset   = MultiImageVQADataset(val_path,   args.img_dir)

    # FIX 3: Bỏ tham số args.img_dir thừa khi khởi tạo collator
    collator = VQADataCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.adapter_out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        bf16=True,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        max_seq_length=1024,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()
    
    final_adapter_path = os.path.join(args.adapter_out_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    processor.save_pretrained(final_adapter_path)
    print(f"Đã lưu LoRA adapter tại: {final_adapter_path}")
    
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return final_adapter_path

# ==============================================================================
# 4. GỘP MODEL
# ==============================================================================
def merge_model(args, adapter_path):
    print("\n[Bước 3] Sáp nhập LoRA vào Base Model...")
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    print("Đang áp dụng adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Đang hợp nhất trọng số...")
    merged_model = model.merge_and_unload()
    
    merged_model.save_pretrained(args.final_model_dir, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(args.final_model_dir)
    print(f"Hoàn tất! Model hoàn chỉnh đã lưu tại: {args.final_model_dir}")

# ==============================================================================
# 5. ĐIỀU PHỐI (Main)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser("Unified QLoRA Pipeline for Qwen-VL")
    parser.add_argument("--task", choices=["single", "multi"], default="single")
    parser.add_argument("--raw_json", type=str, required=True, help="File JSON gốc")
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa ảnh")
    parser.add_argument("--data_out_dir", type=str, default="data/processed")
    parser.add_argument("--adapter_out_dir", type=str, default="checkpoints/lora_adapters")
    parser.add_argument("--final_model_dir", type=str, default="checkpoints/qwen_merged")
    
    # Params
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="vqa-qlora", name=f"{args.task}_run")

    train_json, val_json = prepare_data(args)
    adapter_path = train_qlora(args, train_json, val_json)
    merge_model(args, adapter_path)

    if args.use_wandb:
        wandb.finish()
    print("finish")

if __name__ == "__main__":
    main()