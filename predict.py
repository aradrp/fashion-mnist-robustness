"""Classify a grayscale, Fashion-MNIST-style image using a saved checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from experiment import CLASSES, ROOT, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "results/checkpoints/cnn_aug_11.pt")
    parser.add_argument("--invert", action="store_true", help="Invert dark foreground on light background.")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(args.image) as image:
        gray = image.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(gray, dtype=np.float32)).reshape(1, 1, 28, 28) / 255
    if args.invert:
        x = 1 - x
    with torch.inference_mode():
        scores, labels = model(x).softmax(1).topk(3)
    print(json.dumps([{"label": CLASSES[label], "score": score}
                      for label, score in zip(labels[0].tolist(), scores[0].tolist())], indent=2))


if __name__ == "__main__":
    main()
