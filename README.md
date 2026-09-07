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
- `requirements.txt`: minimal dependencies

Notes for Q1 report
- Data augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `Normalize(mean,std)`
- MobileNet-v2 changes: first convolution stride set to 1 for CIFAR-10, classifier output set to 10 classes
- Training: SGD with momentum 0.9, weight decay 5e-4, MultiStepLR at [60,120,160]
