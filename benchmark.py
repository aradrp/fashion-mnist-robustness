"""Evaluate corruption robustness and train a matched mixed-augmentation CNN."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from experiment import (
    AUG_SIGMA, CLASSES, NOISE_SEED, ROOT, batches, gaussian, load_split,
    make_model, pixels, train, train_val_indices,
)

MODELS = {
    "linear": "Linear",
    "cnn": "CNN",
    "cnn_aug": "CNN + Gaussian",
    "cnn_mixed": "CNN + Gaussian / shift",
}
SEEDS = (11, 22, 33)
SEVERITIES = {
    "gaussian": (0.1, 0.2, 0.3, 0.4),
    "shift": (1, 2, 3, 4),
    "blur": (3, 5, 7, 9),
}
CONDITIONS = [("clean", 0)] + [
    (family, severity) for family, levels in SEVERITIES.items() for severity in levels
]


def shift_horizontal(images, offsets):
    """Shift each image by a signed integer offset, zero-filling newly exposed pixels."""
    width = images.shape[-1]
    columns = torch.arange(width, device=images.device)[None, :] - offsets[:, None]
    indices = columns.clamp(0, width - 1)[:, None, None, :].expand_as(images)
    valid = ((columns >= 0) & (columns < width))[:, None, None, :]
    return images.gather(-1, indices) * valid


def mixed_augmentation(images, generator):
    # 50% clean, 25% Gaussian, 25% shift. Blur is reserved for evaluation.
    choice = torch.rand((len(images), 1, 1, 1), generator=generator)
    offsets = torch.randint(1, 4, (len(images),), generator=generator)
    direction = torch.randint(0, 2, (len(images),), generator=generator) * 2 - 1
    noisy = gaussian(images, AUG_SIGMA, generator)
    shifted = shift_horizontal(images, offsets * direction)
    return torch.where(choice < 0.5, images, torch.where(choice < 0.75, noisy, shifted))


def corrupt(images, family, severity, generator):
    if family == "gaussian":
        return gaussian(images, severity, generator)
    if family == "shift":
        return shift_horizontal(images, torch.full((len(images),), severity))
    if family == "blur":
        return F.avg_pool2d(images, severity, stride=1, padding=severity // 2)
    if family == "clean":
        return images
    raise ValueError(f"Unknown corruption: {family}")


@torch.inference_mode()
def evaluate(model, loader, family, severity):
    model.eval()
    generator = torch.Generator().manual_seed(NOISE_SEED)
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    for images, labels in loader:
        inputs = corrupt(pixels(images), family, severity, generator)
        prediction = model(inputs).argmax(1)
        pairs = len(CLASSES) * labels + prediction
        confusion += pairs.bincount(minlength=confusion.numel()).reshape_as(confusion)
    true_positive = confusion.diag()
    denominator = confusion.sum(0) + confusion.sum(1)
    return {
        "accuracy": true_positive.sum().item() / confusion.sum().item(),
        "macro_f1": (2 * true_positive / denominator.clamp_min(1)).mean().item(),
    }


def summarize(rows):
    """Average each corruption family within a seed before combining seeds."""
    result = {}
    for name in MODELS:
        by_seed = {}
        for seed in SEEDS:
            group = [row for row in rows if row["model"] == name and row["seed"] == seed]
            families = {
                family: mean(row["accuracy"] for row in group if row["family"] == family)
                for family in ("clean", *SEVERITIES)
            }
            by_seed[seed] = {
                **families,
                "all_corruptions": mean(families[family] for family in SEVERITIES),
                "worst_condition": min(row["accuracy"] for row in group if row["family"] != "clean"),
            }
        result[name] = {
            key: {"mean": mean(run[key] for run in by_seed.values()),
                  "sd": stdev(run[key] for run in by_seed.values())}
            for key in by_seed[SEEDS[0]]
        }
    return result


def save_plot(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharey=True, layout="constrained")
    colors = ("#686E76", "#386CB0", "#D06827", "#008479")
    for ax, (family, levels), title, unit in zip(
        axes, SEVERITIES.items(), ("Gaussian noise", "Right shift", "Unseen box blur"),
        ("Standard deviation", "Pixels", "Kernel width (pixels)"),
    ):
        for (name, label), color, marker, line in zip(MODELS.items(), colors, "os^D", (":", "--", "-.", "-")):
            values = [[100 * row["accuracy"] for row in rows
                       if (row["model"], row["family"], row["severity"]) == (name, family, level)]
                      for level in levels]
            ax.errorbar(levels, list(map(mean, values)), yerr=list(map(stdev, values)),
                        color=color, marker=marker, linestyle=line, capsize=3, label=label)
        ax.set(title=title, xlabel=unit, xticks=levels, ylim=(0, 100))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--baselines", type=Path, default=ROOT / "results" / "checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    # Match the architecture, split, optimizer, budget and selection rule of the original run.
    args.batch_size, args.epochs = 256, 8
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        for name in ("linear", "cnn", "cnn_aug"):
            if not (args.baselines / f"{name}_{seed}.pt").is_file():
                parser.error(f"missing baseline checkpoint: {name}_{seed}.pt")
    images, labels = load_split("train", args.data_dir)
    test_loader = batches(*load_split("t10k", args.data_dir), args.batch_size)
    training, validation = train_val_indices(labels)
    train_data = images[training], labels[training]
    val_loader = batches(images[validation], labels[validation], args.batch_size)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    for seed in SEEDS:
        mixed, epoch, history = train("cnn_mixed", seed, train_data, val_loader, args, checkpoints,
                                      augment=mixed_augmentation)
        histories[seed] = {"selected_epoch": epoch, "epochs": history}
        for name in MODELS:
            if name == "cnn_mixed":
                model = mixed
            else:
                checkpoint = torch.load(args.baselines / f"{name}_{seed}.pt", weights_only=True,
                                        map_location="cpu")
                if (checkpoint["model"], checkpoint["seed"]) != (name, seed):
                    raise ValueError(f"Unexpected model or seed in baseline {name}_{seed}.pt")
                model = make_model(name)
                model.load_state_dict(checkpoint["state_dict"])
            for family, severity in CONDITIONS:
                metrics = evaluate(model, test_loader, family, severity)
                rows.append({"model": name, "seed": seed, "family": family,
                             "severity": severity, **metrics})
        print(f"Finished all benchmark conditions for seed {seed}", flush=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seeds": SEEDS, "epochs": args.epochs, "batch_size": args.batch_size,
        "torch": str(torch.__version__), "threads": args.threads,
        "test_noise_seed": NOISE_SEED, "test_severities": SEVERITIES,
        "mixed_training": {"clean_probability": 0.5, "gaussian_probability": 0.25,
                           "shift_probability": 0.25, "gaussian_sigma": AUG_SIGMA,
                           "shift_offsets": [-3, -2, -1, 1, 2, 3]},
        "histories": histories,
    }
    (args.output / "summary.json").write_text(
        json.dumps({"protocol": metadata, "models": summarize(rows)}, indent=2) + "\n", encoding="utf-8",
    )
    save_plot(rows, args.output / "corruption-benchmark.png")


if __name__ == "__main__":
    main()
