# Fashion-MNIST Corruption Robustness

An ENEL 645 deep learning project that measures how Gaussian noise and small image shifts affect clothing classification on Fashion-MNIST. It compares a linear classifier, a compact convolutional neural network (CNN), and the same CNN trained with Gaussian noise augmentation.

## Results

Mean test accuracy across three training seeds on the official 10,000-image test split.

| Model | Clean | Gaussian noise (σ = 0.3) | 3-pixel right shift |
| --- | ---: | ---: | ---: |
| Linear classifier | 83.59% | 63.25% | 29.34% |
| CNN | **88.46%** | 50.28% | 46.12% |
| CNN with noise augmentation | 87.53% | **83.43%** | **47.48%** |

Adding Gaussian noise during training improved CNN accuracy by 33.15 percentage points at σ = 0.3, with a 0.93-point drop on clean images. Accuracy on the right-shift test rose by 1.36 points. The evaluation covers synthetic corruptions of grayscale catalog images.

![Test accuracy for three models across increasing Gaussian noise levels](results/robustness.png)

## Setup on Windows

Open PowerShell in the repository folder. These commands use Python 3.12 and the CPU build of PyTorch.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

## Predict with a trained model

~~~powershell
.\.venv\Scripts\python.exe predict.py results/sample_shirt.png
~~~

The script returns the three classes with the highest softmax scores. The sample's top prediction is `Shirt`, with a score of about 0.727. By default, it loads the noise-augmented CNN checkpoint for seed 11. Use `--checkpoint` to select a different checkpoint. For a dark foreground on a light background, pass `--invert`. Inputs work best when they resemble the centered garment crops in Fashion-MNIST.

## Reproduce the experiment

~~~powershell
.\.venv\Scripts\python.exe experiment.py --output reproduced
.\.venv\Scripts\python.exe figures.py --results reproduced
~~~

The first run downloads the official dataset and verifies the file checksums. It uses 55,000 images for training, 5,000 for validation, and 10,000 for testing. Each of the three models trains for eight epochs with seeds 11, 22, and 33. For each run, the script saves the checkpoint with the highest clean validation accuracy. It then evaluates that checkpoint on clean images, four Gaussian noise levels, and a three-pixel right shift.

These commands write to `reproduced/` so you can compare your run with the bundled results. Small numerical differences can occur across software versions or hardware.

## Repository contents

| Path | Contents |
| --- | --- |
| `experiment.py` | Dataset download, data splits, model training, and evaluation |
| `predict.py` | Single-image inference with a saved checkpoint |
| `figures.py` | Summary statistics and plots |
| `results/checkpoints/` | Nine checkpoints for three models and three seeds |
| `results/metrics.csv` | Accuracy and macro-F1 for all 54 evaluations |
| `results/experiment.json` | Run settings, validation histories, and confusion matrices |
| `results/summary.json` | Means and sample standard deviations across training seeds |
| `results/*.png` | Experiment plots and the sample inference image |

[Official Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
