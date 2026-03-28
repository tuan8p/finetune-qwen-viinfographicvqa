# Preprocessing cho `finetune-qwen-viinfographicvqa`

Repo này hiện chỉ giữ phần preprocessing phục vụ finetune ViInfographicVQA.

Phạm vi hiện tại:
- load raw dataset từ `ViInfographicVQA_dataset/data/*.json`
- tạo `train / valid / test` logic bằng subdataset builder
- preprocess text on-the-fly
- bọc thành `Dataset` và `DataLoader`

Namespace import vẫn là `eda_preprocessing` để không làm gãy code train đã viết trước đó.

Không còn giữ EDA runner, analyzer, artifact writer, hay image analysis trong repo này. Thư mục `outputs/` được giữ nguyên nếu bạn cần tham chiếu artifact cũ, nhưng không nằm trong preprocessing pipeline.

## Cấu trúc chính

```text
finetune-qwen-viinfographicvqa/
├── outputs/
├── run_preprocessing.py
├── src/
│   └── eda_preprocessing/
│       ├── core/
│       ├── io/
│       ├── preprocessing/
│       ├── subdataset/
│       └── training/
└── tests/
```

## Pipeline dữ liệu

Luồng xử lý:

```text
raw dataset
-> subdataset builder
-> preprocessing
-> PyTorch-style Dataset
-> collate_fn
-> DataLoader
```

## Subdataset builder làm gì

Module ở [builder.py](d:/Downloads/BTL_DLA/finetune-qwen-viinfographicvqa/src/eda_preprocessing/subdataset/builder.py).

Quy tắc:
- `use_subdataset=True`
  lấy `10%` từ `single_train` và `multi_train`, sau đó tách `20%` của phần sampled này làm `valid`
- `use_subdataset=False`
  không tạo sub-train, mà lấy `20%` `valid` trực tiếp từ train gốc
- `single_test` và `multi_test` luôn giữ nguyên từ raw
- sampling theo từng raw split train, không gộp train trước rồi mới sample
- seed mặc định là `42`

## Preprocessing làm gì

Module ở [preprocessing](d:/Downloads/BTL_DLA/finetune-qwen-viinfographicvqa/src/eda_preprocessing/preprocessing).

Các bước đang áp dụng:
- chuẩn hóa text về `NFC`
- trim và collapse whitespace
- tạo `lowercase`
- tạo `ascii_folded`
- đếm token của answer bằng whitespace tokenization
- loại sample có `answer > 20 token`
- gắn metadata heuristic:
  `answer_type`, `question_type`, `reasoning_mode`, `cross_image_dependency`, `diacritic_consistency`, `tokenization_flags`
- chuẩn hóa nhẹ answer theo type
- tách `answer_segments` khi answer có `;`

Ràng buộc:
- không resize image
- không OCR
- không validate image
- không đọc metadata image
- giữ nguyên `image_path` / `image_paths`

## Dataloader có những mode nào

`data_mode` hiện có 3 mode:

- `single`
  chỉ dùng sample từ `single_train` / `single_test`
- `multi`
  chỉ dùng sample từ `multi_train` / `multi_test`
- `single_and_multi`
  dùng chung cả single và multi

Các loader tương ứng:

- `single`
  có `train_loader`, `valid_loader`, `test_loader`
- `multi`
  có `train_loader`, `valid_loader`, `test_loader`
- `single_and_multi`
  có `train_loader`, `valid_loader`, `test_loader`, `extra_test_loader`

Ý nghĩa của test loader trong mode `single_and_multi`:
- `test_loader`
  là `test_all = single_test + multi_test`
- `extra_test_loader`
  là `test_single = single_test`

Nếu bạn dùng dataset bundle thay vì dataloader bundle thì mapping tương tự:
- `train_dataset`
- `valid_dataset`
- `test_dataset`
- `extra_test_dataset` chỉ có trong mode `single_and_multi`

## Cách chạy preprocessing

Chạy summary:

```bash
python finetune-qwen-viinfographicvqa/run_preprocessing.py
```

Preview một batch:

```bash
python finetune-qwen-viinfographicvqa/run_preprocessing.py --preview-batch --batch-size 4
```

Tắt subdataset:

```bash
python finetune-qwen-viinfographicvqa/run_preprocessing.py --disable-subdataset
```

Chạy theo mode dữ liệu:

```bash
python finetune-qwen-viinfographicvqa/run_preprocessing.py --data-mode single
python finetune-qwen-viinfographicvqa/run_preprocessing.py --data-mode multi
python finetune-qwen-viinfographicvqa/run_preprocessing.py --data-mode single_and_multi
```

## Cách import vào code train

Thiết lập `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\finetune-qwen-viinfographicvqa\src)
```

Dùng dataset bundle:

```python
from pathlib import Path

from eda_preprocessing.training import build_finetune_dataset_bundle

bundle = build_finetune_dataset_bundle(
    dataset_root=Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"),
    use_subdataset=True,
    seed=42,
    data_mode="single_and_multi",
)

train_dataset = bundle.train_dataset
valid_dataset = bundle.valid_dataset
test_all_dataset = bundle.test_dataset
test_single_dataset = bundle.extra_test_dataset
```

Dùng dataloader bundle:

```python
from pathlib import Path

from eda_preprocessing.training import build_finetune_dataloaders

loaders = build_finetune_dataloaders(
    dataset_root=Path("D:/Downloads/BTL_DLA/ViInfographicVQA_dataset"),
    batch_size=4,
    use_subdataset=True,
    seed=42,
    data_mode="single_and_multi",
)

train_loader = loaders.train_loader
valid_loader = loaders.valid_loader
test_all_loader = loaders.test_loader
test_single_loader = loaders.extra_test_loader
```

## Test

Chạy toàn bộ test:

```bash
python -m unittest discover finetune-qwen-viinfographicvqa/tests -v
```

Các nhóm test chính:
- loader raw dataset
- subdataset builder
- preprocessing rule
- training dataset / dataloader
