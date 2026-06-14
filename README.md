# DeepFashion2 DETR

Research implementation of a DETR-style model for garment detection and
landmark prediction on the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)
dataset.

The model predicts:

- one of 13 garment categories, plus the background class;
- a normalized bounding box for each object query;
- up to 294 packed garment landmarks.

## Requirements

The project uses Python 3.10+ and the following main dependencies:

```text
albumentations
numpy
opencv-python
pytorch-lightning
scipy
tensorboard
torch
torchmetrics
torchvision
```

Install PyTorch and torchvision for the required CPU or CUDA environment, then
install the remaining packages:

```bash
python -m pip install albumentations numpy opencv-python pytorch-lightning \
    scipy tensorboard torchmetrics
```

## Dataset Layout

Download DeepFashion2 and organize each split as follows:

```text
DeepFashion2/
├── train/
│   ├── image/
│   │   ├── 000001.jpg
│   │   └── ...
│   └── annos/
│       ├── 000001.json
│       └── ...
├── validation/
│   ├── image/
│   └── annos/
└── test/
    └── image/
```

Image and annotation filenames must have matching six-digit identifiers.

## Configuration

Data, model, matcher, loss, and optimizer settings are defined in
[`config.py`](config.py). The main configuration classes are:

- `DataConfig`: image size, number of landmarks, and maximum object count;
- `ModelConfig`: transformer dimensions, matching weights, loss weights, and
  learning rates.

Run commands from the repository root so imports resolve against the root
configuration module.

## Training

Update the train and validation paths in [`train.py`](train.py), then run:

```bash
python train.py
```

Training uses PyTorch Lightning. TensorBoard logs and checkpoints are written
under `lightning_logs/` by default.

To inspect the logs:

```bash
tensorboard --logdir lightning_logs
```

## Inference

Run inference for one image using a Lightning checkpoint:

```bash
python test_file.py \
    --image-path /path/to/DeepFashion2/test/image/000001.jpg \
    --checkpoint-path /path/to/model.ckpt \
    --device cpu
```

For CUDA, use `--device cuda --device-index 0`. The rendered image is written
to `/tmp` using the input filename.

## Module Checks

The modules contain small executable checks. Run them from the repository root
with Python's module syntax:

```bash
python -m data.data_pt
python -m data.data_pl
python -m models.model_pt
python -m models.model_pl
```

The data checks currently contain local dataset paths; update those paths
before running them.

## Project Structure

```text
.
├── config.py                 # Shared project configuration
├── data/
│   ├── data_pt.py            # Dataset and Albumentations transforms
│   └── data_pl.py            # Lightning data module
├── models/
│   ├── match.py              # Hungarian matching costs
│   ├── model_pt.py           # Transformer model
│   ├── model_pl.py           # Lightning training and validation module
│   ├── object_queries.py     # Learned object queries
│   ├── positional_encoding.py
│   └── utils.py              # Backbone builders
├── train.py                  # Training entry point
└── test_file.py              # Single-image inference
```
