---
language: en
license: mit
tags:
  - handwritten-digit-recognition
  - cnn
  - transformer
  - pytorch
  - onnx
  - classification
  - mnist
  - emnist
datasets:
  - emnist
  - metricasecuador/hwd-v1
  - nguyenminh4099/handwritten-digits
metrics:
  - accuracy
  - precision
  - recall
  - f1
pipeline_tag: image-classification
model-index:
  - name: CNN-Transformer Handwritten Digit Classifier
    results:
      - task:
          type: image-classification
        dataset:
          name: Combined Handwritten Digits
          type: handwritten-digits-classification
        metrics:
          - type: accuracy
            value: 0.9993
          - type: precision
            value: 0.9993
          - type: recall
            value: 0.9993
          - type: f1
            value: 0.9993
---

# Handwritten Digit Classification — CNN + Transformer Hybrid

Based on: *"Handwriting Recognition through Neural Networks: Enhancing Accuracy and Performance"* (Rajest & Regin, 2024)

**Source code**: [github.com/leocb/handwritten-digits-cnn-transformer](https://github.com/leocb/handwritten-digits-cnn-transformer)

## Model Description

Hybrid CNN + Transformer architecture for handwritten digit classification (0-9). CNN extracts low-level features (edges, shapes); Transformer encoder captures spatial dependencies via self-attention. Trained with Apple Metal (MPS) GPU acceleration.

**Input**: 28×28 grayscale image (white background, dark digit), shape `(1, 28, 28)`, float32, range [0, 1].

**Output**: 10-class logits (digits 0-9).

## Architecture

| Component | Layers | Output Shape |
|---|---|---|
| Conv Block 1 | Conv(1→32) → BN → ReLU → Pool/2 | (32, 14, 14) |
| Conv Block 2 | Conv(32→64) → BN → ReLU → Pool/2 | (64, 7, 7) |
| Conv Block 3 | Conv(64→128) → BN → ReLU | (128, 7, 7) |
| Conv Block 4 | Conv(128→256) → BN → ReLU → Pool/2 | (256, 3, 3) |
| Flatten + Positional Encoding | 9×256 learned position embeddings | (9, 256) |
| Transformer Encoder | 4 layers, 8 heads, d=256, ff=512, GELU, Pre-LN | (9, 256) |
| Classification Head | Mean → LayerNorm → FC(128) → ReLU → Dropout → FC(10) | (10,) |

**Parameters**: 2,534,218 (all trainable)

## Performance

Evaluated on held-out validation set (36,758 samples from 3 combined datasets):

| Metric | Value |
|---|---|
| Accuracy | **99.93%** |
| Precision (macro) | 0.9993 |
| Recall (macro) | 0.9993 |
| F1 (macro) | 0.9993 |

### Per-Class

| Digit | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.9997 | 0.9997 | 0.9997 |
| 1 | 0.9995 | 0.9984 | 0.9989 |
| 2 | 0.9995 | 0.9992 | 0.9993 |
| 3 | 0.9992 | 0.9997 | 0.9995 |
| 4 | 0.9989 | 0.9995 | 0.9992 |
| 5 | 0.9997 | 0.9995 | 0.9996 |
| 6 | 0.9986 | 0.9997 | 0.9992 |
| 7 | 0.9995 | 0.9997 | 0.9996 |
| 8 | 0.9992 | 0.9986 | 0.9989 |
| 9 | 0.9995 | 0.9992 | 0.9993 |

## Training Data

Combined from 3 datasets (503,100 train + 36,758 validation, all 28×28 grayscale):

- **EMNIST Digits** (280,000) — [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- **HWD-V1** (136,920 + 15,218) — [Kaggle HWD-V1](https://www.kaggle.com/datasets/metricasecuador/handwritten-digits-version-1-hwd-v1/data) (standard + edge cases)
- **Handwritten Digits** (86,180 + 21,540) — [nguyenminh4099/handwritten-digits](https://huggingface.co/datasets/nguyenminh4099/handwritten-digits)

Preprocessed dataset available at **[leobottaro/handwritten-digits-pack](https://huggingface.co/datasets/leobottaro/handwritten-digits-pack)**.

## Training

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | Categorical Cross-Entropy |
| Batch Size | 256 |
| Epochs | 22 (best), early stopped at 34 |
| Mixed Precision | MPS AMP |
| Hardware | Apple M-series (MPS) |

## Usage

### PyTorch

```python
import torch

# Load model
ckpt = torch.load("model_best.pt", weights_only=True)
model = CNNTransformer(num_classes=10)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    logits = model(image_tensor)  # (1, 1, 28, 28) → (1, 10)
    pred = logits.argmax(1).item()
```

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
logits = session.run(None, {"input": image_array})[0]  # (1, 1, 28, 28) → (1, 10)
pred = logits.argmax(axis=1)[0]
```

## Files

- `model_best.pt` — PyTorch checkpoint (weights + optimizer state + history)
- `model.onnx` — ONNX model (opset 17, dynamic batch)

## Citation

```bibtex
@article{rajest2024handwriting,
  title={Handwriting Recognition through Neural Networks: Enhancing Accuracy and Performance},
  author={Rajest, S. Suman and Regin, R.},
  journal={Central Asian Journal of Medical and Natural Science},
  volume={5},
  number={4},
  pages={1010--1024},
  year={2024}
}
```

## License

MIT
