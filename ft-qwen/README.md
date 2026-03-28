# ft-qwen

Folder này chứa pipeline finetune cho Qwen2.5-VL trên ViInfographicVQA sau khi dữ liệu đã đi qua preprocessing.

Phiên bản hiện tại là **train-only**:
- load `train_dataset` và `valid_dataset` từ preprocessing
- finetune bằng QLoRA
- chọn best model theo `eval_loss` trên validation
- lưu `final_adapter`
- không merge model
- không predict trên test

## Luồng xử lý

```text
config.yaml
-> cli.py
-> runtime_data.py
-> preprocessing bundle
-> trainer.py
-> checkpoints/final_adapter
```

## Cấu trúc folder

```text
ft-qwen/
├── main.py
├── cli.py
├── bootstrap.py
├── config.py
├── config.yaml
├── runtime_data.py
├── trainer.py
├── wandb_utils.py
└── .wandb.env.example
```

## Chức năng từng file

### `main.py`
- entrypoint của training pipeline
- load config
- build runtime data
- init WandB nếu bật
- gọi train

### `cli.py`
- parse CLI override
- merge override vào YAML config

### `config.py`
- định nghĩa `QwenFinetuneConfig`
- load/save YAML config
- load `.wandb.env` nếu cần

### `config.yaml`
- file cấu hình train chính

### `runtime_data.py`
- build `train_dataset` và `valid_dataset`
- adapt output từ preprocessing sang format trainer/collator dùng được

### `trainer.py`
- load processor + model
- setup QLoRA
- setup `TrainingArguments`
- train bằng `SFTTrainer`
- lưu `final_adapter`
- log metrics / artifacts

### `wandb_utils.py`
- naming run
- runtime metadata
- artifact logging

## Cách chạy

### Chạy với YAML mặc định

```bash
python ft-qwen/main.py --config ft-qwen/config.yaml
```

### Chạy `single`

```bash
python ft-qwen/main.py --config ft-qwen/config.yaml --data-mode single
```

### Chạy `multi`

```bash
python ft-qwen/main.py --config ft-qwen/config.yaml --data-mode multi
```

### Chạy `single_and_multi`

```bash
python ft-qwen/main.py --config ft-qwen/config.yaml --data-mode single_and_multi
```

### Override nhanh một số tham số

```bash
python ft-qwen/main.py ^
  --config ft-qwen/config.yaml ^
  --dataset-root D:/Downloads/BTL_DLA/ViInfographicVQA_dataset ^
  --data-mode single_and_multi ^
  --epochs 3 ^
  --batch-size 4 ^
  --lr 2e-4 ^
  --seed 42 ^
  --adapter-out-dir checkpoints/lora_adapters ^
  --use-wandb
```

## Các option CLI quan trọng

- `--config`: path tới file YAML
- `--data-mode`: `single`, `multi`, `single_and_multi`
- `--dataset-root`: path tới `ViInfographicVQA_dataset`
- `--disable-subdataset`: tắt subdataset mode
- `--epochs`
- `--batch-size`
- `--lr`
- `--seed`
- `--adapter-out-dir`
- `--use-wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-group`
- `--wandb-job-type`
- `--wandb-run-name`
- `--wandb-notes`
- `--wandb-mode`
- `--wandb-env-file`
- `--wandb-tags`
- `--attn-implementation`
- `--dataloader-num-workers`
- `--pin-memory` / `--no-pin-memory`
- `--persistent-workers` / `--no-persistent-workers`

## Setting trong `config.yaml`

### Dataset
- `dataset_root`
- `data_mode`
- `use_subdataset`

### Output
- `adapter_out_dir`

### Model
- `model_id`
- `system_prompt`
- `attn_implementation`
- `min_pixels`
- `max_pixels`

### Training
- `seed`
- `epochs`
- `batch_size`
- `grad_accum`
- `lr`
- `save_steps`
- `save_total_limit`
- `logging_steps`
- `max_seq_length`

### LoRA
- `lora_r`
- `lora_alpha`
- `lora_dropout`

### Quantization
- `bf16`
- `load_in_4bit`
- `bnb_4bit_use_double_quant`
- `bnb_4bit_quant_type`
- `bnb_4bit_compute_dtype`

### DataLoader
- `dataloader_num_workers`
- `pin_memory`
- `persistent_workers`

### WandB
- `use_wandb`
- `wandb_env_file`
- `wandb_project`
- `wandb_entity`
- `wandb_group`
- `wandb_job_type`
- `wandb_run_name`
- `wandb_notes`
- `wandb_mode`
- `wandb_tags`
- `wandb_save_code`
- `wandb_log_artifacts`
- `wandb_log_final_checkpoint`

## Output sau khi train

Trong `adapter_out_dir` sẽ có các artifact chính:
- `effective_config.yaml`
- `training_metrics.json`
- `final_adapter/`
- các checkpoint do Hugging Face Trainer lưu theo `save_steps`

## WandB

File mẫu:

```text
ft-qwen/.wandb.env.example
```

Bạn có thể copy thành `.wandb.env` và điền:

```env
WANDB_API_KEY=...
WANDB_ENTITY=your-team
WANDB_PROJECT=your-project
```

## Ghi chú

- Best model hiện tại được chọn theo `eval_loss` trên validation.
- `ft-qwen` không thực hiện test-time inference nữa.
- Muốn predict trên test, dùng `src/inference`.
