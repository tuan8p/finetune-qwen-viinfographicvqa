# EDA_Preprocessing cho ViInfographicVQA

## Mục tiêu

Thư mục này hiện có 2 phần chính:

- pipeline EDA để sinh CSV / JSON / PNG artifact từ dataset local `ViInfographicVQA_dataset`
- module preprocessing on-the-fly để đi từ raw dataset tới `Dataset` / `DataLoader` cho finetune

Dataset nguồn:

- `ViInfographicVQA_dataset/data/*.json`
- `ViInfographicVQA_dataset/images/*.jpg`

Lưu ý:

- phần EDA có đọc metadata ảnh để phân tích
- phần preprocessing cho finetune **không chỉnh sửa gì liên quan tới image**

## Những gì đã được hiện thực

### 1. EDA pipeline

Các case EDA hiện có:

- `Case 1.1` Answer length distribution
- `Case 1.2` Answer semantic type distribution (`number`, `entity_like`, `text_span_like`, `other`)
- `Case 1.3` Question type distribution + `lookup` vs `reasoning`
- `Case 2.1` Answer-source distribution
- `Case 3.1` Image resolution distribution
- `Case 3.2` Text density proxy bằng `edge density`, `entropy`, `connected components`
- `Case 4.1` Number of images per multi-image sample
- `Case 4.2` Cross-image dependency heuristic
- `Case 5.1` Vietnamese diacritics consistency
- `Case 5.2` Tokenization ambiguity flags

Ngoài ra pipeline EDA còn sinh:

- `dataset_overview.json`
- `heuristic_rules.json`
- `run_manifest.json`

### 2. Preprocessing tới dataloader

Đã thêm module mới cho luồng finetune:

- `subdataset/`: tạo `train / valid / test` logic từ raw dataset
- `preprocessing/`: normalize text, filter sample, enrich metadata
- `training/`: `PreprocessedTrainingDataset`, `collate_fn`, helper `build_finetune_datasets(...)`

Rule đang dùng cho preprocessing:

- nếu `use_subdataset=True` thì lấy `10%` từ `single_train` và `multi_train` để tạo `sub train`
- nếu `use_subdataset=True` thì `sub val` lấy `20%` từ `sub train`
- nếu `use_subdataset=False` thì không tạo `sub train`, mà tách `20%` valid trực tiếp từ train gốc
- test giữ nguyên raw
- hỗ trợ `data_mode`:
  `single`, `multi`, `single_and_multi`
- loại sample có `answer > 20 token`
- không resize ảnh
- không OCR
- không validate hay biến đổi `image_paths`

## Thiết kế code

```text
EDA_Preprocessing/
├── README.md
├── run_eda.py
├── outputs/
├── src/
│   └── eda_preprocessing/
│       ├── analyzers/
│       ├── core/
│       ├── io/
│       ├── pipeline/
│       ├── preprocessing/
│       ├── subdataset/
│       ├── training/
│       └── visualization/
└── tests/
```

Một vài điểm chính:

- `io/dataset_loader.py` đọc raw JSON thành typed sample
- `subdataset/builder.py` tạo logical split `train / valid / test`
- `preprocessing/sample_preprocessor.py` xử lý text/sample metadata
- `training/dataset.py` bọc thành `PyTorch-style Dataset`
- `training/collate.py` tạo `collate_fn`
- EDA runner cũ vẫn hoạt động độc lập, không bị buộc dùng preprocessing mới

## Cách chạy EDA

Chạy full dataset:

```bash
python EDA_Preprocessing/run_eda.py --mode full
```

Chạy sample mode để debug nhanh:

```bash
python EDA_Preprocessing/run_eda.py --mode sample --sample-size 16
```

Chạy riêng một số split:

```bash
python EDA_Preprocessing/run_eda.py --mode full --splits single_train,multi_train
```

Chạy riêng một số case:

```bash
python EDA_Preprocessing/run_eda.py --cases case_1_1_answer_length,case_3_1_image_resolution
```

Đổi thư mục output:

```bash
python EDA_Preprocessing/run_eda.py --output-dir EDA_Preprocessing/outputs
```

### EDA đã làm gì

Khi chạy `run_eda.py`, pipeline sẽ:

- đọc raw sample từ `ViInfographicVQA_dataset/data/*.json`
- đọc metadata ảnh để phục vụ các case phân tích ảnh
- chạy toàn bộ hoặc một phần các analyzer trong `src/eda_preprocessing/analyzers/`
- tổng hợp số liệu theo `overall`, `split`, `task_family`
- sinh artifact ra `outputs/csv`, `outputs/json`, `outputs/figures`, `outputs/logs`

EDA hiện phân tích các nhóm chính:

- phân bố độ dài answer
- loại answer và loại question
- `lookup` vs `reasoning`
- nguồn answer (`answer_source`)
- độ phân giải ảnh và text density proxy
- số ảnh trong multi-image sample
- kiểu phụ thuộc liên ảnh
- độ nhất quán dấu tiếng Việt
- tokenization ambiguity

## Cách chạy preprocessing

Preprocessing bây giờ có entrypoint riêng:

```bash
python EDA_Preprocessing/run_preprocessing.py
```

Lệnh này sẽ:

- load raw dataset
- tạo logical split `train / valid / test`
- chạy preprocessing on-the-fly
- in ra số lượng sample sau preprocessing

### Chạy preprocessing với subdataset mặc định

```bash
python EDA_Preprocessing/run_preprocessing.py --seed 42
```

Khi đó:

- `sub train` lấy `10%` từ `single_train` và `multi_train`
- `sub val` lấy `20%` từ `sub train`
- `test` giữ nguyên raw

### Chạy theo từng data mode

```bash
python EDA_Preprocessing/run_preprocessing.py --data-mode single
python EDA_Preprocessing/run_preprocessing.py --data-mode multi
python EDA_Preprocessing/run_preprocessing.py --data-mode single_and_multi
```

### Chạy mixed mode, mode này sẽ có thêm `test_single`

```bash
python EDA_Preprocessing/run_preprocessing.py --data-mode single_and_multi --preview-batch
```

### Dataloader có những mode nào

`data_mode` hiện có 3 chế độ:

- `single`
  chỉ dùng dữ liệu `single image question`
- `multi`
  chỉ dùng dữ liệu `multi image question`
- `single_and_multi`
  dùng chung cả `single + multi`

Với từng mode, dataloader sẽ được tạo như sau:

- `single`
  có `train_loader`, `valid_loader`, `test_loader`
  trong đó `test_loader` chỉ chứa `single_test`
- `multi`
  có `train_loader`, `valid_loader`, `test_loader`
  trong đó `test_loader` chỉ chứa `multi_test`
- `single_and_multi`
  có `train_loader`, `valid_loader`, `test_loader`, `extra_test_loader`
  trong đó:
  `test_loader` = `test_all` = `single_test + multi_test`
  `extra_test_loader` = `test_single` = chỉ `single_test`

Nếu dùng `build_finetune_dataset_bundle(...)` hoặc `build_finetune_dataloaders(...)`:

- `train_dataset` / `train_loader` luôn là tập train theo mode đã chọn
- `valid_dataset` / `valid_loader` luôn là tập validation theo mode đã chọn
- `test_dataset` / `test_loader` là test chính của mode
- riêng `single_and_multi` sẽ có thêm:
  `extra_test_dataset` / `extra_test_loader` với tên `test_single`

### Chạy preprocessing và preview dataloader batch

```bash
python EDA_Preprocessing/run_preprocessing.py --preview-batch --batch-size 4
```

### Chạy preprocessing nhưng tắt subdataset mode

```bash
python EDA_Preprocessing/run_preprocessing.py --disable-subdataset
```

Khi đó:

- train dùng phần còn lại `80%` từ `single_train` và `multi_train`
- valid lấy `20%` trực tiếp từ train gốc
- test vẫn giữ nguyên raw

### Tách riêng lệnh chạy analysis và preprocessing

```bash
python EDA_Preprocessing/run_eda.py --mode full
python EDA_Preprocessing/run_preprocessing.py --preview-batch
```

### Preprocessing đã làm gì

Khi chạy `run_preprocessing.py`, pipeline sẽ:

- đọc raw sample từ `ViInfographicVQA_dataset/data/*.json`
  - tạo subdataset logic:
    sub train lấy `10%` từ `single_train` và `multi_train`, sub val lấy `20%` từ `sub train`, test giữ nguyên raw
- nếu `use_subdataset=False`:
  train lấy `80%` từ train gốc, valid lấy `20%` trực tiếp từ train gốc, test vẫn giữ nguyên raw
- lọc train / valid / test theo `data_mode`:
  chỉ single-image question, chỉ multi-image question, hoặc mixed
- riêng `single_and_multi` sẽ có:
  `train`, `valid`, `test_all`, `test_single`
- normalize text cho `question` và `answer`:
  Unicode `NFC`, trim, collapse whitespace, lowercase, ascii-folded
- lọc sample có `answer > 20 token`
- gắn metadata heuristic:
  `answer_type`, `question_type`, `reasoning_mode`, `cross_image_dependency`, `diacritic_consistency`, `tokenization_flags`
- chuẩn hóa nhẹ `answer` theo `answer_type`
- tách `answer_segments` nếu answer có delimiter `;`
- bọc dữ liệu đã xử lý thành `PreprocessedTrainingDataset`
- tạo `collate_fn` để đưa tiếp vào `DataLoader`

Lưu ý:

- preprocessing không resize ảnh
- preprocessing không OCR
- preprocessing không validate hay thay đổi `image_path` / `image_paths`

## Cách import preprocessing vào finetune pipeline

Hiện tại preprocessing vẫn có thể được import trực tiếp vào pipeline train.

Trước khi chạy các ví dụ import bên dưới từ repo root, cần trỏ `PYTHONPATH` vào thư mục `src`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\EDA_Preprocessing\src)
```

### Cách dùng nhanh nhất

```python
from pathlib import Path

from torch.utils.data import DataLoader

from eda_preprocessing.training import build_collate_fn, build_finetune_datasets

train_dataset, valid_dataset, test_dataset = build_finetune_datasets(
    dataset_root=Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"),
    use_subdataset=True,
    seed=42,
    data_mode="single_and_multi",
)

collate_fn = build_collate_fn()

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
```

### Nếu muốn tắt subdataset mode

```python
from pathlib import Path

train_dataset, valid_dataset, test_dataset = build_finetune_datasets(
    dataset_root=Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"),
    use_subdataset=False,
    seed=42,
    data_mode="single",
)
```

Khi đó:

- `train_dataset` = `80%` từ train gốc
- `valid_dataset` = `20%` từ train gốc
- `test_dataset` = `single_test + multi_test`

### Nếu muốn mixed mode và thêm `test_single`

```python
from pathlib import Path

from torch.utils.data import DataLoader

from eda_preprocessing.training import build_collate_fn, build_finetune_dataset_bundle

bundle = build_finetune_dataset_bundle(
    dataset_root=Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"),
    use_subdataset=True,
    seed=42,
    data_mode="single_and_multi",
)

collate_fn = build_collate_fn()

train_loader = DataLoader(bundle.train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(bundle.valid_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(bundle.test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_single_loader = DataLoader(
    bundle.extra_test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
```

Khi đó:

- `test_loader` = `test_all` = mixed `single_test + multi_test`
- `test_single_loader` = chỉ `single_test`

### Batch trả ra từ `collate_fn`

`build_collate_fn()` hiện trả batch dict với các key chính:

- `question_id`
- `split_name`
- `subdataset_split`
- `task_family`
- `image_type`
- `answer_source`
- `image_paths`
- `image_count`
- `raw_question`
- `raw_answer`
- `question_text`
- `answer_text`
- `answer_type`
- `question_type`
- `reasoning_mode`
- `cross_image_dependency`
- `diacritic_consistency`
- `tokenization_flags`
- `answer_segments`

## Nếu muốn tự gọi preprocessing mà không qua helper

```python
from pathlib import Path

from eda_preprocessing.io.dataset_loader import DatasetLoader, SUPPORTED_SPLITS
from eda_preprocessing.preprocessing import preprocess_samples
from eda_preprocessing.subdataset import build_subdataset_splits

loader = DatasetLoader(Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"))
raw_samples = loader.load_splits(SUPPORTED_SPLITS)

logical_splits = build_subdataset_splits(raw_samples, enabled=True, seed=42)
train_samples = preprocess_samples(logical_splits.train_samples, subdataset_split="train")
valid_samples = preprocess_samples(logical_splits.valid_samples, subdataset_split="valid")
test_samples = preprocess_samples(logical_splits.test_samples, subdataset_split="test")
```

## Chạy test

Chạy toàn bộ unit test + smoke test:

```bash
python -m unittest discover EDA_Preprocessing/tests -v
```

## Ghi chú heuristic

Một số nhãn không có sẵn trong JSON nên được suy diễn bằng rule-based heuristic:

- `answer_type`
- `question_type`
- `lookup` vs `reasoning`
- `cross_image_dependency`
- `diacritic_consistency`
- `tokenization_flags`

EDA pipeline sẽ export rule ra `outputs/json/heuristic_rules.json` để dễ audit và chỉnh sửa.

## Output chính của EDA

Sau mỗi lần chạy EDA, bạn sẽ thấy:

- `outputs/csv/<case_id>_summary.csv`
- `outputs/csv/<case_id>_details.csv`
- `outputs/json/<case_id>_metadata.json`
- `outputs/figures/<case_id>__*.png`
- `outputs/json/dataset_overview.json`
- `outputs/json/heuristic_rules.json`
- `outputs/json/run_manifest.json`

## Assumptions đang dùng

- Local JSON là source of truth nếu lệch với paper
- Không dùng `results_vietocr.jsonl`
- Token length dùng whitespace tokenization đơn giản
- Preprocessing không chỉnh sửa gì liên quan tới image
- Test chạy bằng `unittest`
