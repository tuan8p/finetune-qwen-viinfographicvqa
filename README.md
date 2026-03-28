# finetune-qwen-viinfographicvqa

Repo này tổ chức lại pipeline cho bài toán Vietnamese Infographic VQA theo 3 tầng rõ ràng:

1. `src/preprocessing`
   chuẩn bị dữ liệu on-the-fly từ raw JSON tới `Dataset` / `DataLoader`
2. `ft-qwen`
   finetune Qwen2.5-VL bằng train/valid, lưu adapter checkpoint rồi dừng
3. `src/inference`
   chạy inference trên test bằng checkpoint/model đã có và lưu prediction ra file JSON

Repo hiện không dùng một pipeline “train xong tự merge rồi tự test” nữa. Luồng đúng là:

`raw dataset -> preprocessing -> finetune -> adapter checkpoint -> inference -> prediction json -> calculate_scores`

## Cấu trúc chính

```text
finetune-qwen-viinfographicvqa/
├── ft-qwen/
│   ├── main.py
│   ├── cli.py
│   ├── config.py
│   ├── config.yaml
│   ├── runtime_data.py
│   ├── trainer.py
│   ├── wandb_utils.py
│   └── .wandb.env.example
├── src/
│   ├── common/
│   ├── inference/
│   │   ├── single/
│   │   ├── multi/
│   │   └── single_and_multi/
│   ├── inference_core/
│   └── preprocessing/
├── tests/
├── run_preprocessing.py
└── requirements.txt
```

## Thành phần nào làm gì

### `ft-qwen`
- Chỉ lo training.
- Nhận dữ liệu đã qua preprocessing.
- Hỗ trợ `data_mode = single | multi | single_and_multi`.
- Chọn best model theo `eval_loss` trên tập validation.
- Lưu `final_adapter` và `training_metrics.json`.
- Có tích hợp WandB cho run train.

### `src/preprocessing`
- Đọc raw split `single_train`, `single_test`, `multi_train`, `multi_test`.
- Tạo logical split `train`, `valid`, `test`.
- Hỗ trợ `use_subdataset=True/False`.
- Filter sample có `answer > 20 token`.
- Normalize text, enrich metadata, dựng `Dataset` / `DataLoader`.

### `src/inference`
- Mỗi folder con là một mode inference:
  - `single`
  - `multi`
  - `single_and_multi`
- Dùng chung config/runtime/wandb/model helpers từ `src/inference_core`.
- Nạp test data qua preprocessing pipeline, không đọc raw test thủ công nữa.
- Có thể load trực tiếp adapter checkpoint của `ft-qwen` bằng `model_path`.

### `src/inference_core`
- Chứa phần dùng chung cho inference:
  - config YAML
  - runtime data bundle
  - wandb helpers
  - runner helpers
  - Qwen adapter-loading utilities

## Quick Start

### 1. Preview preprocessing

```bash
python run_preprocessing.py --data-mode single_and_multi --preview-batch
```

### 2. Finetune

```bash
python ft-qwen/main.py --config ft-qwen/config.yaml
```

Override nhanh vài option quan trọng:

```bash
python ft-qwen/main.py ^
  --config ft-qwen/config.yaml ^
  --data-mode single_and_multi ^
  --dataset-root D:/Downloads/BTL_DLA/ViInfographicVQA_dataset ^
  --epochs 3 ^
  --batch-size 4 ^
  --use-wandb
```

### 3. Inference single

```bash
python -m src.inference.single qwenvl --config src/inference_core/config.yaml
```

### 4. Inference multi

```bash
python -m src.inference.multi qwenvl --config src/inference_core/config.yaml
```

### 5. Inference single + multi

```bash
python -m src.inference.single_and_multi qwenvl --config src/inference_core/config.yaml
```

Mode `single_and_multi` sẽ lưu:
- output cho `test_all`
- output cho `test_single`

### 6. Tính score

```bash
python -m src.inference.single.calculate_scores --results-dir results/single
python -m src.inference.multi.calculate_scores --results-dir results/multi
python -m src.inference.single_and_multi.calculate_scores --results-dir results/single_and_multi
```

## Cấu hình quan trọng

### Train
- file chính: `ft-qwen/config.yaml`
- các nhóm setting:
  - dataset / mode
  - model / attention
  - qlora / quantization
  - training hyperparameters
  - dataloader
  - wandb

### Inference
- file chính: `src/inference_core/config.yaml`
- các nhóm setting:
  - dataset root / data mode
  - model key / model path / model registry
  - batch size / save interval
  - wandb

## WandB

Train và inference đều có thể dùng `.env` riêng cho WandB:
- file mẫu: `ft-qwen/.wandb.env.example`

Thường chỉ cần:

```env
WANDB_API_KEY=...
WANDB_ENTITY=your-team
WANDB_PROJECT=your-project
```

Model checkpoint path không cần đưa qua `.env`; với inference bạn chỉnh trực tiếp trong `src/inference_core/config.yaml`.

## Test

Repo hiện có unit test cho preprocessing/training bundle:

```bash
python -m unittest discover tests -v
```

## Đọc thêm

- `ft-qwen/README.md`
- `src/preprocessing/README.md`
- `src/inference/README.md`
