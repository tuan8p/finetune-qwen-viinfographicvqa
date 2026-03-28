# inference

Folder này chứa các runner inference theo từng mode dữ liệu.

Nó không chứa phần config/runtime shared nữa; phần đó đã được tách ra `src/inference_core`.

Vai trò của `src/inference` là:
- chạy model trên tập test
- lưu prediction JSON
- tính score từ file prediction đã lưu

## Các mode inference

```text
src/inference/
├── single/
├── multi/
└── single_and_multi/
```

### `single`
- chạy inference trên tập `single_test`
- lưu 1 file prediction

### `multi`
- chạy inference trên tập `multi_test`
- lưu 1 file prediction

### `single_and_multi`
- chạy inference trên `test_all`
- chạy thêm lần nữa trên `test_single`
- lưu 2 output prediction riêng

## Shared layer dùng chung

Các runner trong folder này dùng chung utility ở:

```text
src/inference_core/
├── config.py
├── config.yaml
├── runtime_data.py
├── runner_utils.py
├── wandb_utils.py
└── model_utils.py
```

Điều này có nghĩa là:
- test data luôn được nạp qua preprocessing
- config inference được quản lý tập trung
- WandB cho inference được setup thống nhất
- Qwen có thể load trực tiếp adapter checkpoint từ `ft-qwen`

## Cấu trúc từng mode

Mỗi folder mode thường có:

```text
<mode>/
├── run_inference.py
├── calculate_scores.py
├── __main__.py
└── models/
```

### `run_inference.py`
- parse CLI
- load config từ `src/inference_core/config.yaml`
- build test dataset qua preprocessing
- load model
- chạy predict
- lưu output JSON
- log WandB nếu bật

### `calculate_scores.py`
- đọc prediction JSON trong thư mục kết quả
- tính score local
- lưu `final_scores.json`
- lưu `detailed_analysis.json`

### `models/`
- wrapper riêng cho từng backend:
  - `qwenvl`
  - `internvl`
  - `ovis`

## Cách chạy inference

### Single

```bash
python -m src.inference.single qwenvl --config src/inference_core/config.yaml
```

### Multi

```bash
python -m src.inference.multi qwenvl --config src/inference_core/config.yaml
```

### Single + Multi

```bash
python -m src.inference.single_and_multi qwenvl --config src/inference_core/config.yaml
```

## Các option CLI dùng chung

3 runner đều hỗ trợ:
- positional `model`
- `--config`
- `--dataset-root`
- `--model-path`
- `--output-dir`
- `--batch-size`
- `--save-interval`
- `--seed`
- `--test-loading`
- `--use-wandb`
- `--disable-subdataset`

## Cấu hình inference

File chính:

```text
src/inference_core/config.yaml
```

Các nhóm setting:

### Dataset
- `dataset_root`
- `data_mode`
- `use_subdataset`
- `seed`

### Model
- `model_key`
- `model_path`
- `model_registry`
- `attn_implementation`
- `test_loading`

### Runtime
- `batch_size`
- `save_interval`
- `output_dir`
- `dataloader_num_workers`

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

## Cách chỉ định checkpoint finetune

Nếu muốn dùng adapter vừa train từ `ft-qwen`, chỉ cần sửa:

```yaml
model_key: "qwenvl"
model_path: "D:/.../checkpoints/lora_adapters/final_adapter"
```

Trong trường hợp đó:
- inference sẽ load trực tiếp adapter checkpoint
- không cần merge model trước
- không cần truyền checkpoint path qua `.env`

## Output prediction

### `single`
- `results/single/<model_name>.json`

### `multi`
- `results/multi/<model_name>.json`

### `single_and_multi`
- `results/single_and_multi/test_all/<model_name>.json`
- `results/single_and_multi/test_single/<model_name>.json`

## Chạy score

### Single

```bash
python -m src.inference.single.calculate_scores --results-dir results/single
```

### Multi

```bash
python -m src.inference.multi.calculate_scores --results-dir results/multi
```

### Single + Multi

```bash
python -m src.inference.single_and_multi.calculate_scores --results-dir results/single_and_multi
```

## Ghi chú

- Inference dùng test data đã đi qua preprocessing.
- Với scope project hiện tại, test cũng đã bị filter `answer > 20 token`.
- `single_and_multi` luôn lưu riêng prediction cho `test_all` và `test_single`.
