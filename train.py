import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(batch_size, num_workers=4):
    # CIFAR-10 stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def build_mobilenet_v2(pretrained=False, num_classes=10):
    # Load torchvision MobileNetV2 and adapt for CIFAR-10
    model = torchvision.models.mobilenet_v2(pretrained=pretrained)
    # Adjust first conv stride/kernel for 32x32 inputs to preserve spatial dims
    # features[0][0] is the initial Conv2d
    try:
        orig = model.features[0][0]
        model.features[0][0] = torch.nn.Conv2d(3, orig.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    except Exception:
        pass
    # Replace classifier for CIFAR-10
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return running_loss / total, 100.0 * correct / total


def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    trainloader, testloader = get_dataloaders(args.batch_size, num_workers=args.num_workers)

    model = build_mobilenet_v2(pretrained=args.pretrained, num_classes=10)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix(loss=running_loss / total, acc=100.0 * correct / total)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, testloader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'Epoch {epoch}: Train loss {train_loss:.4f}, Train acc {train_acc:.2f}%, Test loss {test_loss:.4f}, Test acc {test_acc:.2f}%')

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'acc': best_acc}, out_dir / 'best_model.pth')

    # Save final history and plot
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'history.npy', history)

    # Plot curves
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['test_loss'], label='test')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['test_acc'], label='test')
    plt.title('Top-1 Acc')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png')
    print(f'Best test accuracy: {best_acc:.2f}%')


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNet-v2 CIFAR10 baseline')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
