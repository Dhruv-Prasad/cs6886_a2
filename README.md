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
- `section3_results.csv`: complete measured configuration table for W&B visualization
- `wandb_section3.py`: uploads Section 3 results and logs a W&B parallel-coordinates chart
- `generate_section3_plot.py`: generates the complete local Section 3 plot
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

For the smaller 90%-target model:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 --weight-policy mixed_5_6_8 --weight-granularity layer --calibration-batches 20 --output-dir mixed_5_6_8_outputs
```

The output directory contains `compression_results.csv`, `compression_report.txt`, and `parallel_coordinates.png`. The report selects the highest-accuracy configuration meeting `--minimum-accuracy` (90% by default). The activation ratio is estimated from the peak per-sample intermediate activation observed during a forward pass; the weight ratio includes one 32-bit scale per floating-point parameter tensor.

`--weight-granularity layer` uses one scale per weight tensor. `--weight-granularity channel` uses one scale per output channel for convolution and linear weights, while biases and batch-normalization parameters remain layer-wise. In the local 8-bit test, channel-wise quantization reached 93.30% versus 92.92% for layer-wise quantization, but its metadata reduced the weight ratio from 4.00x to 3.88x.

Magnitude pruning can be combined with mixed quantization. The `--sparsity` value zeros the smallest-magnitude weights independently in each convolution/linear weight tensor before quantization. The size estimate includes one mask bit per model parameter, plus scale metadata:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 --weight-policy mixed_5_6_8 --weight-granularity layer --sparsity 0.30 --calibration-batches 20 --output-dir pruned_mixed_outputs
```

To estimate per-layer Huffman storage after mixed quantization:

```bash
python compress.py --checkpoint path/to/best_model.pth --bits 8 --weight-policy mixed_5_6_8 --weight-granularity layer --encoding huffman --calibration-batches 20 --output-dir huffman_outputs
```

Huffman coding is estimated independently for each parameter tensor from its quantized symbol frequencies. The estimate includes one codebook entry per observed symbol, scale metadata, and pruning masks when pruning is enabled. The current evaluation still uses fake-quantized tensors; an actual packed deployment file would require a serializer using the reported codebooks
