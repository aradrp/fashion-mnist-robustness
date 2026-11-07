"""Compare four trained classifiers before and after corrupting one image."""
import argparse
import json
from pathlib import Path

import torch

from benchmark import MODELS, SEEDS, SEVERITIES, corrupt, plt
from experiment import CLASSES, ROOT
from predict import load_classifier, load_image


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=ROOT / "results/sample_shirt.png")
    parser.add_argument("--family", choices=SEVERITIES, default="shift")
    parser.add_argument("--severity", type=float)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=11)
    parser.add_argument("--noise-seed", type=int, default=420)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sample-comparison.png"))
    args = parser.parse_args()
    levels = SEVERITIES[args.family]
    severity = levels[2] if args.severity is None else args.severity
    if severity not in levels:
        parser.error(f"severity for {args.family} must be one of {', '.join(map(str, levels))}")
    # Read the canonical level so that pixel offsets and kernel widths stay integers.
    severity = levels[levels.index(severity)]
    clean = load_image(args.image, invert=args.invert)
    changed = corrupt(clean, args.family, severity, torch.Generator().manual_seed(args.noise_seed))
    images = torch.cat((clean, changed))
    predictions = []
    for name, label in MODELS.items():
        directory = "benchmark_results" if name == "cnn_mixed" else "results"
        checkpoint = ROOT / directory / "checkpoints" / f"{name}_{args.seed}.pt"
        model = load_classifier(checkpoint)
        scores, indices = model(images).softmax(1).max(dim=1)
        result = [{"class": CLASSES[i], "softmax": round(s, 4)}
                  for i, s in zip(indices.tolist(), scores.tolist())]
        predictions.append({"model": label, "clean": result[0], "corrupted": result[1]})

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), layout="constrained")
    for ax, image, key, title in zip(
        axes, images, ("clean", "corrupted"), ("Clean", f"{args.family.title()} ({severity:g})"),
    ):
        ax.imshow(image[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("\n".join(
            f'{row["model"]}: {row[key]["class"]} ({row[key]["softmax"]:.1%})'
            for row in predictions
        ), loc="left")
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(json.dumps({"comparison_image": str(args.output), "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
