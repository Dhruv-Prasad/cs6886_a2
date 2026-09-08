import csv

import wandb


COLUMNS = [
    "configuration",
    "accuracy",
    "bits",
    "weight_ratio",
    "activation_ratio",
    "model_size_mb",
]


def main():
    run = wandb.init(project="cs6886-a2", name="section3-compression-sweep")
    rows = []
    with open("section3_results.csv", newline="") as file:
        for item in csv.DictReader(file):
            rows.append([
                item["configuration"],
                float(item["accuracy"]),
                int(item["bits"]),
                float(item["weight_ratio"]),
                float(item["activation_ratio"]),
                float(item["model_size_mb"]),
            ])

    table = wandb.Table(columns=COLUMNS, data=rows)
    run.log({
        "section3_results": table,
        "parallel_coordinates": wandb.plot.parallel_coordinates(
            table,
            "accuracy",
            ["bits", "accuracy", "weight_ratio", "activation_ratio", "model_size_mb"],
        ),
    })
    run.finish()


if __name__ == "__main__":
    main()
