# Digits AI — Handwritten Digit Classification

Training an AI model for handwritten digit classification using multiple datasets, with Apple Metal (MPS) GPU acceleration.

## Datasets

### EMNIST Digits
- **Source**: [NIST EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- **Format**: 28×28 grayscale, IDX binary format (gzipped)
- **Samples**: 240,000 training + 40,000 testing
- **Credit**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.

### HWD-V1 (Handwritten Digits Version 1)
- **Source**: [Kaggle — HWD-V1](https://www.kaggle.com/datasets/metricasecuador/handwritten-digits-version-1-hwd-v1/data)
- **Format**: 52×52 PNG images, organized by digit folder
- **Samples**: 150,000 standard + ~2,138 edge cases across digits 0–9
- **Credit**: Dataset by Metricas Ecuador on Kaggle

### Handwritten Digits
- **Source**: [Hugging Face — nguyenminh4099/handwritten-digits](https://huggingface.co/datasets/nguyenminh4099/handwritten-digits)
- **Format**: 28×28 RGBA PNG (transparent background, dark digits)
- **Samples**: 86,180 train + 10,770 validation + 10,770 test
- **Credit**: nguyenminh4099 on Hugging Face

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Run `dataset_prep.ipynb` to load, preprocess, and merge all datasets
2. Output tensors saved to `dataset/` as `X_train.pt.gz`, `y_train.pt.gz`, `X_val.pt.gz`, `y_val.pt.gz`
