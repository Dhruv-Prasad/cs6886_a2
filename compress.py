import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from train import build_mobilenet_v2, get_dataloaders, set_seed


def quantize_tensor(tensor, bits):
    if not tensor.is_floating_point():
        return tensor, torch.tensor(0.0, device=tensor.device)
    limit = (1 << (bits - 1)) - 1
    scale = tensor.detach().abs().max() / limit
    scale = torch.clamp(scale, min=torch.finfo(tensor.dtype).eps)
    integer = torch.clamp(torch.round(tensor / scale), -limit - 1, limit).to(torch.int32)
    return integer, scale.detach()


def fake_quantize(tensor, bits, scale=None):
    if not tensor.is_floating_point():
        return tensor
    if scale is None:
        _, scale = quantize_tensor(tensor, bits)
    limit = (1 << (bits - 1)) - 1
    return torch.clamp(torch.round(tensor / scale), -limit - 1, limit) * scale


def parameter_bits(name, parameter, bits, policy='uniform'):
    if policy == 'mixed_6_8':
        if name == 'features.0.0.weight' or name == 'classifier.1.weight':
            return 8
        if name.endswith('.weight') and parameter.ndim in (2, 4):
            return 6
        return 8
    if policy == 'mixed_4_6_8':
        if name == 'features.0.0.weight' or name == 'classifier.1.weight':
            return 8
        if name.endswith('.weight') and parameter.ndim == 4 and parameter.shape[1] == 1:
            return 4
        if name.endswith('.weight') and parameter.ndim in (2, 4):
            return 6
        return 8
    if policy == 'mixed_5_6_8':
        if name == 'features.0.0.weight' or name == 'classifier.1.weight':
            return 8
        if name.endswith('.weight') and parameter.ndim == 4 and parameter.shape[1] == 1:
            return 5
        if name.endswith('.weight') and parameter.ndim in (2, 4):
            return 6
        return 8
    return bits


def prune_model_weights(model, sparsity):
    if not 0.0 <= sparsity < 1.0:
        raise ValueError('sparsity must be in [0, 1)')
    masks = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point() and parameter.ndim >= 2 and name.endswith('.weight'):
                threshold = torch.quantile(parameter.data.abs().flatten(), sparsity)
                mask = parameter.data.abs() > threshold
                parameter.data.mul_(mask)
                masks[name] = int(mask.numel() - mask.sum().item())
    return masks


def quantize_model_weights(model, bits, granularity='layer', policy='uniform'):
    metadata = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point():
                parameter_bitwidth = parameter_bits(name, parameter, bits, policy)
                channelwise = should_use_channel_scale(name, parameter, granularity)
                if channelwise:
                    reduce_dims = tuple(range(1, parameter.ndim))
                    limit = (1 << (parameter_bitwidth - 1)) - 1
                    scale = parameter.data.detach().abs().amax(dim=reduce_dims, keepdim=True) / limit
                    scale = torch.clamp(scale, min=torch.finfo(parameter.dtype).eps)
                else:
                    _, scale = quantize_tensor(parameter.data, parameter_bitwidth)
                parameter.data.copy_(fake_quantize(parameter.data, parameter_bitwidth, scale))
                metadata[name] = {'scales': int(scale.numel()), 'bits': parameter_bitwidth}
    return metadata


def should_use_channel_scale(name, parameter, granularity):
    if not name.endswith('.weight') or parameter.ndim not in (2, 4):
        return False
    if granularity == 'channel':
        return True
    if granularity == 'hybrid':
        return name.startswith('features.') and name != 'features.0.0.weight'
    return False


def calibrate_activations(model, loader, device, batches, bits):
    maxima = {}
    handles = []

    def make_hook(name):
        def hook(_, __, output):
            if torch.is_tensor(output):
                value = float(output.detach().abs().max().cpu())
                maxima[name] = max(maxima.get(name, 0.0), value)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU6)):
            handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        for index, (inputs, _) in enumerate(loader):
            model(inputs.to(device))
            if index + 1 >= batches:
                break
    for handle in handles:
        handle.remove()
    return {name: max(value / ((1 << (bits - 1)) - 1), torch.finfo(torch.float32).eps) for name, value in maxima.items()}


def attach_activation_quantizers(model, scales, bits):
    handles = []

    def make_hook(name):
        def hook(_, __, output):
            return fake_quantize(output, bits, scales[name]) if torch.is_tensor(output) else output
        return hook

    for name, module in model.named_modules():
        if name in scales:
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def evaluate(model, loader, device, activation_scales=None, activation_bits=8):
    handles = attach_activation_quantizers(model, activation_scales, activation_bits) if activation_scales else []
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs.to(device))
            targets = targets.to(device)
            loss_sum += loss_fn(outputs, targets).item() * targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    for handle in handles:
        handle.remove()
    return loss_sum / total, 100.0 * correct / total


def estimate_sizes(model, weight_bits, activation_bits, activation_scales, loader, device, weight_granularity='layer', weight_policy='uniform', sparsity=0.0):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    weight_scale_count = 0
    weight_bits_total = 0
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        parameter_bitwidth = parameter_bits(name, parameter, weight_bits, weight_policy)
        channelwise = should_use_channel_scale(name, parameter, weight_granularity)
        weight_scale_count += parameter.shape[0] if channelwise else 1
        weight_bits_total += parameter.numel() * parameter_bitwidth
    weight_bits_total += weight_scale_count * 32
    if sparsity > 0:
        nonzero_bits = 0
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point():
                parameter_bitwidth = parameter_bits(name, parameter, weight_bits, weight_policy)
                nonzero_bits += (parameter != 0).sum().item() * parameter_bitwidth
        mask_bits = parameter_count
        weight_bits_total = nonzero_bits + weight_scale_count * 32 + mask_bits

    peak_elements = 0
    handles = []

    def hook(_, __, output):
        nonlocal peak_elements
        if torch.is_tensor(output):
            peak_elements = max(peak_elements, output[0].numel())

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU6)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        inputs, _ = next(iter(loader))
        model(inputs.to(device))
    for handle in handles:
        handle.remove()
    activation_scale_count = len(activation_scales)
    activation_bits_total = peak_elements * activation_bits + activation_scale_count * 32
    fp_weight_bits = parameter_count * 32
    fp_activation_bits = peak_elements * 32
    return {
        'weight_size_mb': weight_bits_total / 8 / 1e6,
        'activation_size_mb': activation_bits_total / 8 / 1e6,
        'model_size_mb': weight_bits_total / 8 / 1e6,
        'weight_ratio': fp_weight_bits / weight_bits_total,
        'activation_ratio': fp_activation_bits / activation_bits_total if activation_bits_total else 0.0,
        'parameter_count': parameter_count,
        'peak_activation_elements_per_sample': peak_elements,
    }


def load_model(checkpoint, device):
    model = build_mobilenet_v2(pretrained=False, num_classes=10).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get('model_state', state))
    return model


def run(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    _, testloader = get_dataloaders(args.batch_size, args.num_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for bits in args.bits:
        model = load_model(args.checkpoint, device)
        baseline_loss, baseline_acc = evaluate(model, testloader, device)
        pruned = prune_model_weights(model, args.sparsity)
        quantize_model_weights(model, bits, args.weight_granularity, args.weight_policy)
        scales = calibrate_activations(model, testloader, device, args.calibration_batches, bits)
        loss, accuracy = evaluate(model, testloader, device, scales, bits)
        sizes = estimate_sizes(model, bits, bits, scales, testloader, device, args.weight_granularity, args.weight_policy, args.sparsity)
        rows.append({'bits': bits, 'weight_policy': args.weight_policy, 'weight_granularity': args.weight_granularity, 'sparsity': args.sparsity, 'pruned_weights': sum(pruned.values()), 'baseline_accuracy': baseline_acc, 'accuracy': accuracy, 'loss': loss, **sizes})
        print(f'{args.weight_policy} {bits}-bit: accuracy={accuracy:.2f}% weight_ratio={sizes["weight_ratio"]:.2f}x activation_ratio={sizes["activation_ratio"]:.2f}x')

    with (output_dir / 'compression_results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row['accuracy'] if row['accuracy'] >= args.minimum_accuracy else -math.inf)
    with (output_dir / 'compression_report.txt').open('w') as file:
        file.write('Selected configuration\n')
        for key, value in best.items():
            file.write(f'{key}: {value}\n')

    metrics = ['bits', 'accuracy', 'weight_ratio', 'activation_ratio', 'model_size_mb']
    values = [[float(row[metric]) for metric in metrics] for row in rows]
    bounds = [(min(column), max(column)) for column in zip(*values)]
    normalized = []
    for row in values:
        normalized.append([(value - low) / (high - low) if high != low else 0.5 for value, (low, high) in zip(row, bounds)])
    x = list(range(len(metrics)))
    plt.figure(figsize=(10, 6))
    for row, line in zip(rows, normalized):
        plt.plot(x, line, marker='o', linewidth=2, label=f"{row['bits']}-bit")
    plt.xticks(x, ['Bits', 'Accuracy (%)', 'Weight ratio', 'Activation ratio', 'Model size (MB)'])
    plt.ylabel('Normalized value')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.25)
    plt.title('Compression sweep parallel coordinates')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'parallel_coordinates.png', dpi=160)


def parse_args():
    parser = argparse.ArgumentParser(description='Manual MobileNet-v2 weight and activation quantization')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bits', nargs='+', type=int, default=[8, 6, 4])
    parser.add_argument('--weight-granularity', choices=['layer', 'channel', 'hybrid'], default='layer')
    parser.add_argument('--weight-policy', choices=['uniform', 'mixed_6_8', 'mixed_4_6_8', 'mixed_5_6_8'], default='uniform')
    parser.add_argument('--sparsity', type=float, default=0.0, help='Global magnitude sparsity for convolution/linear weights')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--calibration-batches', type=int, default=20)
    parser.add_argument('--minimum-accuracy', type=float, default=90.0)
    parser.add_argument('--output-dir', default='compression_outputs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())