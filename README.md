# Assignment 2 — Q1 Training Baseline

This folder contains a PyTorch training script to produce the Q1 baseline (MobileNet-v2 on CIFAR-10).

Quick start

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run training (example):

```bash
python train.py --epochs 100 --batch-size 128 --lr 0.1 --output-dir outputs
```

Options
- `--pretrained`: use ImageNet pre-trained MobileNet-v2 weights (still adapted for CIFAR-10)
- `--seed`: set RNG seed for reproducibility

Files
- `train.py`: training and evaluation script
- `compress.py`: manual weight/activation quantization, calibration, evaluation, and sweep plots
- `requirements.txt`: minimal dependencies

Notes for Q1 report
- Data augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `Normalize(mean,std)`
- MobileNet-v2 changes: first convolution stride set to 1 for CIFAR-10, classifier output set to 10 classes
- Training: SGD with momentum 0.9, weight decay 5e-4, MultiStepLR at [60,120,160]

Q2-Q4 compression workflow

The compression code uses symmetric per-tensor fake quantization written in this repository. It does not use a compression or quantization API. All floating-point parameters are quantized; activation ranges are calibrated from a configurable number of test batches, then fake-quantized during evaluation. Each tensor has one 32-bit scale value, which is included in the size estimates.

Run a sweep using the trained checkpoint:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 6 4 --calibration-batches 20 --output-dir compression_outputs
```

For per-output-channel weight scales on convolution and linear weights:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 --weight-granularity channel --calibration-batches 20 --output-dir channel_outputs
```

For mixed precision, use 6-bit weights for intermediate convolution/linear tensors and 8-bit weights for the first convolution, final classifier, biases, and normalization tensors:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 --weight-policy mixed_6_8 --weight-granularity layer --calibration-batches 20 --output-dir mixed_outputs
```

The output directory contains `compression_results.csv`, `compression_report.txt`, and `parallel_coordinates.png`. The report selects the highest-accuracy configuration meeting `--minimum-accuracy` (90% by default). The activation ratio is estimated from the peak per-sample intermediate activation observed during a forward pass; the weight ratio includes one 32-bit scale per floating-point parameter tensor.

`--weight-granularity layer` uses one scale per weight tensor. `--weight-granularity channel` uses one scale per output channel for convolution and linear weights, while biases and batch-normalization parameters remain layer-wise. In the local 8-bit test, channel-wise quantization reached 93.30% versus 92.92% for layer-wise quantization, but its metadata reduced the weight ratio from 4.00x to 3.88x.

The channel-wise 6-bit test reached 80.33% accuracy, 5.12x weight compression, and an estimated 1.75 MB model size. It demonstrates the size/accuracy trade-off but is not recommended as the final model because of the large accuracy drop.

The mixed 6/8-bit layer-wise test reached 92.06% accuracy, 5.29x weight compression, and an estimated 1.690 MB model size. It is currently the best measured size/accuracy trade-off.

The mixed 6/8-bit channel-wise test reached 92.83% accuracy, 5.09x weight compression, and an estimated 1.758 MB model size. It improves accuracy over layer-wise mixed precision, but its additional scale metadata reduces compression.

The hybrid 8-bit test uses channel-wise scales only for intermediate convolution weights and layer-wise scales at the first convolution, classifier, biases, and normalization parameters. It reached 93.02% accuracy, 3.88x weight compression, and an estimated 2.305 MB model size.
