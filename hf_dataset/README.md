---
language: en
license: mit
tags:
  - handwritten-digits
  - digit-recognition
  - mnist-style
  - grayscale
  - classification
pretty_name: Handwritten Digits Pack
size_categories:
  - 100K<n<1M
task_categories:
  - image-classification
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: int32
  splits:
  - name: train
    num_examples: 503100
  - name: validation
    num_examples: 36758
---

# Handwritten Digits Pack

Unified 28×28 grayscale handwritten digit dataset for training classification models (digits 0–9). White background, dark digits, float32 [0, 1].

## Composition

Merged from 3 public datasets:

| Dataset | Train | Validation | Source |
|---|---|---|---|
| EMNIST Digits | 280,000 | — | [NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) |
| HWD-V1 | 136,920 | 15,218 | [Kaggle](https://www.kaggle.com/datasets/metricasecuador/handwritten-digits-version-1-hwd-v1/data) |
| Handwritten Digits | 86,180 | 21,540 | [HF](https://huggingface.co/datasets/nguyenminh4099/handwritten-digits) |
| **Total** | **503,100** | **36,758** | |

## Format

- `train.parquet` — 503,100 rows (191.8 MB)
- `validation.parquet` — 36,758 rows (11.0 MB)
- `image` column: PNG-encoded 28×28 grayscale (struct with `bytes` + `path` fields)
- `label` column: int32, digits 0–9

## Loading

```python
from datasets import load_dataset
ds = load_dataset("leobottaro/handwritten-digits-pack")
```


## Preprocessing Applied

- **EMNIST**: IDX binary → transpose (fix orientation) → invert (white bg, dark digit) → float32
- **HWD-V1**: 52×52 PNG → LANCZOS resize 28×28 → grayscale → float32. Includes both Standard and Edge Cases
- **Handwritten Digits**: RGBA PNG → composite on white background (preserves dark-on-transparent) → grayscale → float32

## Credits

- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
- Metricas Ecuador — HWD-V1 on Kaggle
- nguyenminh4099 — Handwritten Digits on Hugging Face
