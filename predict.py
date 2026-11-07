"""Classify a grayscale, Fashion-MNIST-style image using a saved checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from experiment import CLASSES, ROOT, make_model


def load_classifier(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def load_image(path, *, invert=False):
    with Image.open(path) as image:
        gray = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(gray, dtype=np.float32)).reshape(1, 1, 28, 28) / 255
    return 1 - pixels if invert else pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "results/checkpoints/cnn_aug_11.pt")
    parser.add_argument("--invert", action="store_true", help="Invert dark foreground on light background.")
    args = parser.parse_args()
    model = load_classifier(args.checkpoint)
    x = load_image(args.image, invert=args.invert)
    with torch.inference_mode():
        scores, labels = model(x).softmax(1).topk(3)
    print(json.dumps([{"label": CLASSES[label], "score": score}
                      for label, score in zip(labels[0].tolist(), scores[0].tolist())], indent=2))


if __name__ == "__main__":
    main()
