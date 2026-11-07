"""Train and evaluate classifiers for Fashion-MNIST on CPU."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import struct
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
    "Shirt", "Sneaker", "Bag", "Ankle boot",
]
MODELS = ("linear", "cnn", "cnn_aug")
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.4)
SHIFT = 3
SPLIT_SEED = 645
NOISE_SEED = 420
AUG_SIGMA = 0.3


def dataset_file(name, directory):
    path = directory / name
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == FILES[name]:
        return path
    print(f"Downloading {name}", flush=True)
    with urlopen(URL + name, timeout=120) as response:
        content = response.read()
    if hashlib.md5(content).hexdigest() != FILES[name]:
        raise ValueError(f"Checksum mismatch for {name}")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_split(prefix, directory):
    images, labels = [
        gzip.decompress(dataset_file(name, directory).read_bytes())
        for name in (f"{prefix}-images-idx3-ubyte.gz", f"{prefix}-labels-idx1-ubyte.gz")
    ]
    magic, count, rows, cols = struct.unpack_from(">4I", images)
    label_magic, label_count = struct.unpack_from(">2I", labels)
    if (magic, rows, cols, label_magic, label_count) != (2051, 28, 28, 2049, count):
        raise ValueError(f"Invalid IDX headers for {prefix}")
    x = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, 1, 28, 28)
    y = np.frombuffer(labels, dtype=np.uint8, offset=8)
    if len(y) != count or np.any(y >= len(CLASSES)):
        raise ValueError(f"Invalid labels for {prefix}")
    return torch.from_numpy(x.copy()), torch.from_numpy(y.astype(np.int64))


def make_model(name):
    if name == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if name in ("cnn", "cnn_aug", "cnn_mixed"):
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    raise ValueError(f"Unknown model {name}")


def train_val_indices(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    validation, training = [], []
    for label in range(len(CLASSES)):
        indices = rng.permutation(np.flatnonzero(labels.numpy() == label))
        validation.extend(indices[:500])
        training.extend(indices[500:])
    return rng.permutation(training), rng.permutation(validation)


def batches(images, labels, size, seed=None):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(images, labels), batch_size=size,
                      shuffle=seed is not None, generator=generator)


def pixels(images):
    return images.float().div(255)


def gaussian(images, sigma, generator):
    noise = torch.randn(images.shape, generator=generator)
    return (images + sigma * noise).clamp(0, 1)


def translate_right(images):
    shifted = torch.zeros_like(images)
    shifted[..., SHIFT:] = images[..., :-SHIFT]
    return shifted


@torch.inference_mode()
def evaluate(model, loader, corruption="clean", sigma=0.0):
    model.eval()
    noise_rng = torch.Generator().manual_seed(NOISE_SEED)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for images, labels in loader:
        x = pixels(images)
        if corruption == "gaussian":
            x = gaussian(x, sigma, noise_rng)
        elif corruption == "shift":
            x = translate_right(x)
        predictions = model(x).argmax(1)
        confusion += torch.bincount(10 * labels + predictions, minlength=100).reshape(10, 10)
    tp = confusion.diag()
    denom = confusion.sum(0) + confusion.sum(1)
    return {
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_f1": float((2 * tp / denom.clamp_min(1)).mean()),
        "confusion": confusion.tolist(),
    }


def train(name, seed, train_data, val_loader, args, checkpoint_dir, *, augment=None):
    torch.manual_seed(seed)
    model = make_model(name)
    train_loader = batches(*train_data, args.batch_size, seed=seed)
    aug_rng = torch.Generator().manual_seed(10000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            x = pixels(images)
            if name == "cnn_aug":
                # Mix clean and noisy examples in the same batch.
                noisy = gaussian(x, AUG_SIGMA, aug_rng)
                mask = torch.rand((len(x), 1, 1, 1), generator=aug_rng) < 0.5
                x = torch.where(mask, noisy, x)
            if augment is not None:
                x = augment(x, aug_rng)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        accuracy = evaluate(model, val_loader)["accuracy"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset),
                        "val_accuracy": accuracy})
        print(f"{name} seed={seed} epoch={epoch} val={accuracy:.4f}", flush=True)
        if best is None or accuracy > best["val_accuracy"]:
            best = {
                "model": name, "seed": seed, "epoch": epoch, "val_accuracy": accuracy,
                "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
            }
    torch.save(best, checkpoint_dir / f"{name}_{seed}.pt")
    model.load_state_dict(best["state_dict"])
    return model, best["epoch"], history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs, batch size and threads must be positive, with distinct seeds")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    images, labels = load_split("train", args.data_dir)
    test_images, test_labels = load_split("t10k", args.data_dir)
    training, validation = train_val_indices(labels)
    train_data = images[training], labels[training]
    val_loader = batches(images[validation], labels[validation], args.batch_size)
    test_loader = batches(test_images, test_labels, args.batch_size)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "epochs": args.epochs, "batch_size": args.batch_size, "threads": args.threads,
        "seeds": args.seeds, "split_seed": SPLIT_SEED, "noise_seed": NOISE_SEED,
        "augmentation_sigma": AUG_SIGMA, "augmentation_probability": 0.5,
        "shift_pixels": SHIFT, "n_train": len(training), "n_val": len(validation),
        "n_test": len(test_labels), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "platform": platform.platform(), "classes": CLASSES,
    }
    rows, runs = [], []
    conditions = [("clean", 0.0)] + [("gaussian", sigma) for sigma in SIGMAS[1:]] + [("shift", 0.0)]
    for seed in args.seeds:
        for name in MODELS:
            model, epoch, history = train(name, seed, train_data, val_loader, args, checkpoint_dir)
            record = {"model": name, "seed": seed, "selected_epoch": epoch, "history": history,
                      "parameters": sum(p.numel() for p in model.parameters()), "tests": []}
            for corruption, sigma in conditions:
                metrics = evaluate(model, test_loader, corruption, sigma)
                row = {"model": name, "seed": seed, "corruption": corruption, "sigma": sigma,
                       "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
                rows.append(row)
                record["tests"].append({**row, "confusion": metrics["confusion"]})
            runs.append(record)
    (args.output / "experiment.json").write_text(json.dumps({"config": config, "runs": runs}, indent=2),
                                                 encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
