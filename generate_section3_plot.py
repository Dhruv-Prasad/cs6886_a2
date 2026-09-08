import csv
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ["bits", "accuracy", "weight_ratio", "activation_ratio", "model_size_mb"]
LABELS = ["Bits", "Accuracy (%)", "Weight ratio", "Activation ratio", "Model size (MB)"]


def main():
    rows = []
    with Path("section3_results.csv").open(newline="") as file:
        for item in csv.DictReader(file):
            rows.append({metric: float(item[metric]) for metric in METRICS} | {"configuration": item["configuration"]})

    bounds = [(min(row[metric] for row in rows), max(row[metric] for row in rows)) for metric in METRICS]
    x_values = range(len(METRICS))

    plt.figure(figsize=(13, 7))
    for row in rows:
        normalized = [
            (row[metric] - low) / (high - low) if high != low else 0.5
            for metric, (low, high) in zip(METRICS, bounds)
        ]
        style = {"linewidth": 2.8, "color": "#c0392b", "zorder": 3} if row["configuration"].startswith("20% prune + mixed 5/6/8-bit + Huffman") else {"alpha": 0.55, "linewidth": 1.2, "color": "#2c7fb8"}
        plt.plot(x_values, normalized, marker="o", label=row["configuration"], **style)

    plt.xticks(list(x_values), LABELS)
    plt.ylabel("Normalized value (0–1)")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.title("CS6886 Assignment 2: Compression Sweep")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    output = Path("section3_parallel_coordinates.png")
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
