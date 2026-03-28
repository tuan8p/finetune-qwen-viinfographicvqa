# preprocessing

Folder này chứa module preprocessing on-the-fly cho ViInfographicVQA, dùng chung cho:
- preview dữ liệu qua `run_preprocessing.py`
- finetune trong `ft-qwen`
- inference test trong `src/inference` thông qua `src/inference_core`

Preprocessing hiện dừng ở mức `Dataset` / `DataLoader`, không OCR và không chỉnh sửa pixel image.

## Mục tiêu của preprocessing

- đọc raw splits từ `ViInfographicVQA_dataset/data`
- tạo logical split `train`, `valid`, `test`
- hỗ trợ `single`, `multi`, `single_and_multi`
- hỗ trợ `use_subdataset=True/False`
- normalize text
- filter sample có `answer > 20 token`
- enrich metadata heuristic
- trả về `Dataset` / `DataLoader` cho train và test

## Luồng xử lý

```text
raw JSON
-> io/dataset_loader.py
-> subdataset/builder.py
-> preprocessing/sample_preprocessor.py
-> training/dataset.py
-> training/collate.py
-> DataLoader
```

## Cấu trúc folder

```text
src/preprocessing/
├── core/
│   ├── contracts.py
│   ├── heuristics.py
│   ├── rules.py
│   └── text_utils.py
├── io/
│   └── dataset_loader.py
├── preprocessing/
│   ├── answer_processor.py
│   ├── contracts.py
│   ├── sample_filter.py
│   ├── sample_preprocessor.py
│   └── text_normalizer.py
├── subdataset/
│   ├── builder.py
│   └── contracts.py
├── training/
│   ├── collate.py
│   ├── contracts.py
│   ├── dataloaders.py
│   └── dataset.py
└── __init__.py
```

## Các chức năng chính

### `io/dataset_loader.py`
- load 4 raw split:
  - `single_train`
  - `single_test`
  - `multi_train`
  - `multi_test`

### `subdataset/builder.py`
- tạo logical split theo rule hiện tại:
  - `use_subdataset=True`
    - lấy sub train từ train gốc
    - tách validation từ sub train
  - `use_subdataset=False`
    - lấy validation trực tiếp từ train gốc
- test giữ nguyên raw split rồi mới đi qua preprocessing

### `preprocessing/sample_preprocessor.py`
- normalize text
- đếm `answer_tokens`
- filter `answer > 20 token`
- classify heuristic:
  - `answer_type`
  - `question_type`
  - `reasoning_mode`
  - `cross_image_dependency`
  - `diacritic_consistency`
  - `tokenization_flags`
- parse `answer_segments`

### `training/dataset.py`
- build dataset bundle cho:
  - `single`
  - `multi`
  - `single_and_multi`
- với mode `single_and_multi` có thêm `extra_test_dataset = test_single`

### `training/dataloaders.py`
- build `DataLoader` tương ứng từ dataset bundle

## Data mode

Có 3 mode:
- `single`
- `multi`
- `single_and_multi`

Riêng `single_and_multi` sẽ có:
- `train`
- `valid`
- `test_all`
- `test_single`

## Cách chạy preview preprocessing

Từ root repo:

```bash
python run_preprocessing.py
```

### Mixed mode + preview batch

```bash
python run_preprocessing.py --data-mode single_and_multi --preview-batch
```

### Single mode

```bash
python run_preprocessing.py --data-mode single
```

### Multi mode

```bash
python run_preprocessing.py --data-mode multi
```

### Tắt subdataset

```bash
python run_preprocessing.py --disable-subdataset
```

### Override seed và batch size preview

```bash
python run_preprocessing.py --seed 42 --batch-size 4 --preview-batch
```

## CLI options của `run_preprocessing.py`

- `--dataset-root`
- `--seed`
- `--batch-size`
- `--disable-subdataset`
- `--data-mode`
- `--preview-batch`

## Những rule preprocessing quan trọng

- Chỉ xử lý text và sample metadata.
- Không resize ảnh.
- Không OCR.
- Không validate image content ở mức pixel.
- Giữ nguyên `image_path` / `image_paths`.
- Filter tất cả sample có `answer > 20 token`, gồm cả test.

## Output logic

Preprocessing không export dataset processed ra file mới trong flow chính.
Nó trả trực tiếp:
- `PreprocessedTrainingDataset`
- `FinetuneDatasetBundle`
- `FinetuneDataLoaderBundle`

Đây là tầng mà `ft-qwen` và `inference_core` import vào để dùng lại.

## API chính

Các entrypoint hay dùng:
- `build_finetune_dataset_bundle(...)`
- `build_finetune_datasets(...)`
- `build_finetune_dataloaders(...)`
- `build_collate_fn()`

## Test

```bash
python -m unittest discover tests -v
```

Các test hiện cover:
- loader
- preprocessing filter
- subdataset builder
- dataset bundle
- dataloader bundle
