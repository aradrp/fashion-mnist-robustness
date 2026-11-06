"""Generate the report figures and summary from the saved experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from experiment import CLASSES, MODELS, ROOT, SIGMAS, gaussian, load_split, pixels, translate_right

LABELS = {"linear": "Linear", "cnn": "CNN", "cnn_aug": "CNN + noise"}
COLORS = ("#525B68", "#1A5783", "#B64E20")


def summarize(runs):
    summary = {}
    for name in MODELS:
        group = [run for run in runs if run["model"] == name]
        summary[name] = {}
        for i, test in enumerate(group[0]["tests"]):
            key = f'{test["corruption"]}_{test["sigma"]:.1f}'
            summary[name][key] = {
                f"{metric}_{stat}": float(fn([run["tests"][i][metric] for run in group]))
                for metric in ("accuracy", "macro_f1")
                for stat, fn in (("mean", np.mean), ("sd", lambda x: np.std(x, ddof=1) if len(x) > 1 else 0))
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    directory = args.results
    record = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    runs, config = record["runs"], record["config"]
    summary = summarize(runs)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(4.4, 2.6), layout="constrained")
    keys = ["clean_0.0"] + [f"gaussian_{sigma:.1f}" for sigma in SIGMAS[1:]]
    for name, color, marker, line in zip(MODELS, COLORS, ("o", "s", "^"), (":", "--", "-")):
        mean = [100 * summary[name][key]["accuracy_mean"] for key in keys]
        sd = [100 * summary[name][key]["accuracy_sd"] for key in keys]
        ax.errorbar(SIGMAS, mean, yerr=sd, color=color, marker=marker, ls=line,
                    lw=1.5, ms=4, capsize=3, label=LABELS[name])
    ax.set(xlabel="Gaussian noise standard deviation", ylabel="Test accuracy (%)", xticks=SIGMAS)
    ax.grid(axis="y", color="#DFE4E9")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(directory / "robustness.png", dpi=300)
    plt.close(fig)

    x, y = load_split("t10k", args.data_dir)
    index = int((y == 6).nonzero()[0, 0])
    clean = pixels(x[index:index + 1])
    noisy = gaussian(clean, 0.3, torch.Generator().manual_seed(config["noise_seed"]))
    fig, axes = plt.subplots(1, 3, figsize=(4.2, 1.45), layout="constrained")
    for ax, image, title in zip(axes, (clean, noisy, translate_right(clean)),
                                ("Clean shirt", "Gaussian 0.3", "Shift +3 px")):
        ax.imshow(image[0, 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(directory / "corruptions.png", dpi=300)
    plt.close(fig)
    Image.fromarray(x[index, 0].numpy()).save(directory / "sample_shirt.png")

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.05), layout="constrained")
    for ax, name, title in zip(axes, ("cnn", "cnn_aug"), ("CNN", "CNN + noise")):
        matrices = [test["confusion"] for run in runs if run["model"] == name
                    for test in run["tests"] if test["corruption"] == "gaussian" and test["sigma"] == 0.3]
        confusion = np.sum(matrices, axis=0)
        fractions = confusion / confusion.sum(axis=1, keepdims=True)
        heatmap = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        ax.set(title=title, xlabel="Predicted class ID", xticks=range(len(CLASSES)),
               yticks=range(len(CLASSES)))
        ax.tick_params(length=0, labelsize=7)
    axes[0].set_ylabel("True class ID")
    axes[1].set_yticklabels([])
    fig.colorbar(heatmap, ax=axes, shrink=0.77, ticks=[0, 0.5, 1])
    fig.savefig(directory / "confusion.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
