# CS6886 Assignment 2: MobileNet-v2 Compression on CIFAR-10

**Student:** Dhruv Prasad  
**Repository:** https://github.com/Dhruv-Prasad/cs6886_a2  
**Date:** September 8, 2026

## Abstract

This project trains MobileNet-v2 on CIFAR-10 and evaluates a manually implemented, configurable quantization method for model weights and activations. The uncompressed pretrained baseline reaches **93.62%** test top-1 accuracy after 150 epochs. The selected 8-bit configuration reaches **92.92%**, while reducing the estimated weight storage by **4.00x** and the peak activation storage by **3.99x**. The estimated compressed model size is **2.237 MB**, compared with an approximately **9.192 MB** floating-point checkpoint.

## 1. Training Baseline

### 1.1 Dataset preparation and transforms

The experiment uses CIFAR-10, containing 50,000 training images and 10,000 test images across 10 classes. The CIFAR-10 channel statistics used for normalization are:

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

Training transforms:

1. `RandomCrop(32, padding=4)`
2. `RandomHorizontalFlip()`
3. `ToTensor()`
4. `Normalize(mean, std)`

Test transforms:

1. `ToTensor()`
2. `Normalize(mean, std)`

The random crop and horizontal flip provide spatial augmentation while normalization improves optimization stability. No random augmentation is applied to the test set.

### 1.2 MobileNet-v2 configuration and training strategy

The implementation starts from torchvision MobileNet-v2 with the default width multiplier of 1.0 and the default inverted-residual configuration. The following changes adapt it to CIFAR-10:

- The first convolution uses a 3x3 kernel, stride 1, and padding 1 instead of the ImageNet-oriented downsampling stride.
- The classifier is replaced with a linear layer producing 10 logits.
- The default MobileNet-v2 dropout setting is retained.
- Batch-normalization layers are retained and trained normally.

Training configuration:

| Setting | Value |
|---|---|
| Initialization | ImageNet-pretrained MobileNet-v2 for the attached run |
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Learning-rate schedule | MultiStepLR, milestones 60, 120, 160; gamma 0.2 |
| Random seed | 42 in the reproducible script configuration |
| Device | GPU used for the attached long run; local validation used CPU |

The repository supports both fresh and pretrained runs through `train.py`. The checkpoint and history supplied with this report are from the 150-epoch pretrained run.

### 1.3 Baseline results

The best and final test top-1 accuracy are both **93.62%** at epoch 150. The attached training history contains the corresponding loss and accuracy curves.

![Baseline loss and accuracy curves](pretrained_150-20260908T091452Z-1-001/pretrained_150/training_curves.png)

The baseline checkpoint is approximately **9.192 MB** on disk. This file is a PyTorch checkpoint and includes model-state serialization overhead, so the analytical 32-bit parameter size is used for compression-ratio calculations.

### 1.4 Failure modes

The main failure modes observed during low-bit evaluation are:

- A single per-tensor scale can be dominated by outliers, reducing the effective resolution for most values.
- Quantization error accumulates across the many MobileNet-v2 blocks.
- Activation quantization is especially sensitive because an early activation error is propagated through later depthwise and pointwise convolutions.
- 4-bit quantization causes near-chance behavior in this implementation, reaching 11.19% accuracy, which is close to the 10-class random baseline.
- The pretrained ImageNet initialization is useful for optimization, but CIFAR-10 has a very different image resolution and distribution; the first convolution therefore needs the CIFAR-specific stride adjustment.

## 2. Compression Method

### 2.1 Design

The compression implementation is in `compress.py`. It uses manual symmetric uniform quantization and does not call a quantization or compression library API.

For a tensor $x$ and signed quantization width $b$, the integer range is approximately:

$$
q_{min} = -2^{b-1}, \qquad q_{max} = 2^{b-1}-1
$$

The scale is calculated as:

$$
s = \frac{\max_i |x_i|}{2^{b-1}-1}
$$

and quantization is:

$$
q_i = \operatorname{clip}\left(\operatorname{round}\left(\frac{x_i}{s}\right), q_{min}, q_{max}\right).
$$

For evaluation, the integer value is dequantized as $\hat{x}_i=sq_i$. This is implemented as fake quantization in floating-point tensors so the normal PyTorch model can still execute.

### 2.2 Layers covered

**Weights:** every floating-point parameter tensor returned by `model.named_parameters()` is quantized, including convolution weights, linear weights, biases, and batch-normalization parameters.

**Activations:** outputs of `Conv2d`, `Linear`, and `ReLU6` modules are calibrated and fake-quantized. Calibration records the maximum absolute activation over a configurable number of batches. The default sweep uses 20 calibration batches.

The method is intentionally simple and reproducible. It uses one scale per tensor rather than a more complex per-channel representation.

An optional channel-wise variant is also implemented. With `--weight-granularity channel`, convolution and linear weights receive one scale per output channel; biases and batch-normalization parameters remain layer-wise. This reduces the effect of outlier channels, but increases metadata storage.

### 2.3 Storage overheads

For each floating-point parameter tensor, one 32-bit scale is stored. For each calibrated activation tensor, one 32-bit scale is stored. These scale values are included in the estimates.

The model contains **2,236,682 parameters**. For a $b$-bit weight configuration, the estimated weight storage is:

$$
S_w = \frac{N_w b + 32T_w}{8 \times 10^6}\text{ MB},
$$

where $N_w$ is the number of weight values and $T_w$ is the number of weight tensors. The activation estimate uses the largest observed per-sample intermediate activation and includes one scale per calibrated activation module:

$$
S_a = \frac{N_a b + 32T_a}{8 \times 10^6}\text{ MB}.
$$

These are packed-storage estimates. The evaluation checkpoint itself remains a floating-point PyTorch model; an integer-packed deployment serializer would be needed to realize the estimated file size physically.

## 3. Compression Sweep

The sweep evaluates 8-bit, 6-bit, and 4-bit weights and activations using the same baseline checkpoint and 20 calibration batches. The full measured results are stored in `compression_full/compression_results.csv`.

| Weight/activation bits | Baseline accuracy | Quantized accuracy | Accuracy drop | Weight ratio | Activation ratio | Estimated model size |
|---:|---:|---:|---:|---:|---:|---:| 
| 8 | 93.62% | **92.92%** | 0.70 pp | **4.00x** | **3.99x** | **2.237 MB** |
| 6 | 93.62% | 74.01% | 19.61 pp | 5.33x | 5.31x | 1.678 MB |
| 4 | 93.62% | 11.19% | 82.43 pp | 8.00x | 7.94x | 1.119 MB |

An additional local 8-bit comparison using channel-wise weight scales reached **93.30%** accuracy, compared with **92.92%** for layer-wise weights. Its metadata-aware weight ratio was **3.88x**, compared with **4.00x** for layer-wise weights. Thus channel-wise quantization improves accuracy by 0.38 percentage points relative to the current layer-wise 8-bit result, but does not improve the storage ratio.

The channel-wise 6-bit experiment reached **80.33%** accuracy with a **5.12x** metadata-aware weight ratio and an estimated **1.75 MB** model size. This is smaller than 8-bit storage, but the 13.29 percentage-point drop from the channel-wise 8-bit result is too large for the selected final configuration.

### 3.1 Mixed-precision experiment

The mixed policy uses 6-bit weights for intermediate convolution and linear tensors, while protecting the first convolution, final classifier, biases, and batch-normalization parameters with 8 bits. With layer-wise scales, it reached **92.06%** accuracy, a **5.29x** metadata-aware weight ratio, and an estimated **1.690 MB** model size. Activations remained 8-bit with a **3.99x** ratio.

The channel-wise mixed variant reached **92.83%** accuracy, a **5.09x** weight ratio, and an estimated **1.758 MB** model size. It improves accuracy by 0.77 percentage points over layer-wise mixed precision, but its additional scale metadata costs 0.20 MB and reduces the compression ratio.

### 3.2 Hybrid layer/channel experiment

The hybrid 8-bit policy uses channel-wise scales only for intermediate convolution weights. The first convolution, final classifier, biases, and normalization parameters use layer-wise scales. It reached **93.02%** accuracy, a **3.881x** metadata-aware weight ratio, and an estimated **2.305 MB** model size. This is nearly the same accuracy as fully channel-wise 8-bit quantization, while explicitly protecting the input and output boundaries, but it does not beat mixed 6/8-bit quantization for size reduction.

![Compression sweep parallel coordinates](compression_full/parallel_coordinates.png)

The plot above is generated locally by `compress.py` from the measured sweep CSV. It is a parallel-coordinates figure suitable for the report. For a hosted Weights & Biases version, upload the rows of `compression_results.csv` as a W&B table and create a parallel-coordinates visualization using the columns `bits`, `accuracy`, `weight_ratio`, `activation_ratio`, and `model_size_mb`.

## 4. Compression Analysis and Selected Configuration

The selected configuration is **mixed 6/8-bit weights with 8-bit activations and layer-wise weight scales**. It is the smallest measured configuration that retains accuracy above the 90% selection threshold.

For maximum accuracy rather than maximum compression ratio, the channel-wise 8-bit variant is a viable alternative: it reaches 93.30% test accuracy, only 0.32 percentage points below the floating-point baseline. The final selection is mixed 6/8-bit layer-wise quantization because it provides stronger size reduction while retaining 92.06% accuracy.

### 4.1 Weight compression ratio

The measured estimated weight ratio is:

$$
\text{weight ratio} = \frac{\text{32-bit floating-point storage}}{\text{8-bit storage plus scale metadata}} = \mathbf{3.999x}.
$$

### 4.2 Activation compression ratio

Activation storage was measured using the largest per-sample intermediate activation observed during one forward pass through the model. The 8-bit activation estimate includes one 32-bit scale per calibrated activation module. The resulting ratio is:

$$
\text{activation ratio} = \mathbf{3.986x}.
$$

The peak activation size used in the estimate was 98,304 elements per sample.

### 4.3 Accuracy

The selected 8-bit model reaches **92.92%** test top-1 accuracy, a decrease of only **0.70 percentage points** from the 93.62% baseline.

### 4.4 Final estimated model size

The estimated 8-bit weight/model size, including weight scale metadata, is **2.237 MB**. The corresponding peak activation storage estimate, including activation scale metadata, is **0.0987 MB per sample**. The original floating-point checkpoint is approximately 9.192 MB on disk.

## 5. Reproducibility and Repository

Repository:

https://github.com/Dhruv-Prasad/cs6886_a2

Clone and install:

```bash
git clone https://github.com/Dhruv-Prasad/cs6886_a2.git
cd cs6886_a2
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Train a baseline:

```bash
python train.py --epochs 150 --batch-size 128 --lr 0.1 --output-dir outputs
```

Train with pretrained MobileNet-v2 initialization:

```bash
python train.py --pretrained --epochs 150 --batch-size 128 --lr 0.01 --output-dir pretrained_outputs
```

Run the compression sweep using a best-model checkpoint:

```bash
python compress.py \
  --checkpoint pretrained_outputs/best_model.pth \
  --bits 8 6 4 \
  --calibration-batches 20 \
  --batch-size 128 \
  --num-workers 2 \
  --minimum-accuracy 90 \
  --output-dir compression_outputs
```

The code fixes the random seed to 42 by default. The dataset downloads automatically to `data/`, which is excluded from Git because the CIFAR-10 archive is larger than GitHub's file-size limit. Generated checkpoints and result directories are also excluded from the public repository.

Environment used for local validation:

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.1 |

The repository separates baseline training/evaluation (`train.py`) from compression and sweep evaluation (`compress.py`). The output artifacts used for the numerical results are `history.npy`, `training_curves.png`, `compression_results.csv`, `compression_report.txt`, and `parallel_coordinates.png`.

## Conclusion

Mixed-precision symmetric quantization provides the best measured accuracy-compression trade-off in this experiment. It reduces estimated weight storage by 5.29 times and preserves 92.06% CIFAR-10 test accuracy, 1.56 percentage points below the uncompressed baseline. Uniform 6-bit quantization loses too much accuracy, while protecting sensitive layers with 8 bits makes a useful intermediate policy possible.

The implemented channel-wise experiment shows the expected trade-off: finer weight scales improve 8-bit accuracy to 93.30%, but additional scale metadata reduces the weight ratio to 3.88x. It is therefore useful when accuracy is the primary objective, while layer-wise 8-bit quantization remains the selected compression configuration.