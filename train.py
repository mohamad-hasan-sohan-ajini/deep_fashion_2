"""Train the DeepFashion2 transformer model."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import DataConfig, ModelConfig
from data.data_pl import DeepFashion2DataModule
from models.model_pl import TransformerModelPL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/aj/data/DeepFashion2"),
        help="Dataset directory containing train/ and validation/",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--log-every-n-steps", type=int, default=100)
    parser.add_argument(
        "--scalar-log-every-n-batches",
        type=int,
        default=100,
        help="Write batch-level loss metrics every N mini-batches; 0 disables",
    )
    parser.add_argument(
        "--image-log-every-n-batches",
        type=int,
        default=1000,
        help="Write training prediction images every N mini-batches; 0 disables",
    )
    parser.add_argument(
        "--tensorboard-num-images",
        type=int,
        default=ModelConfig.tensorboard_num_images,
        help="Number of images to render at each image logging event",
    )
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs"),
        help="Root directory for TensorBoard logs and checkpoints",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Checkpoint from which to resume training",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one training and validation batch for debugging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_root / "train"
    val_path = args.data_root / "validation"

    seed_everything(args.seed, workers=True)

    datamodule = DeepFashion2DataModule(
        train_path,
        val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="fit")
    num_tensorboard_images = min(
        args.tensorboard_num_images,
        len(datamodule.val_dataset),
    )
    fixed_val_images = torch.stack(
        [
            datamodule.val_dataset[index][0]
            for index in range(num_tensorboard_images)
        ]
    )
    # check defaults in model_pl.py for default constructors and parameters
    model = TransformerModelPL(
        scalar_log_every_n_batches=args.scalar_log_every_n_batches,
        image_log_every_n_batches=args.image_log_every_n_batches,
        tensorboard_num_images=args.tensorboard_num_images,
    )
    model.set_tensorboard_images(fixed_val_images)
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="",
        version="",
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="class_accuracy/val",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        filename="{epoch:04d}-{step}",
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
    )
    run_config = {
        "data": asdict(DataConfig()),
        "model": asdict(ModelConfig()),
        "training": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    logger.experiment.add_text(
        "device",
        str(trainer.strategy.root_device),
        0,
    )
    logger.experiment.add_text(
        "config",
        "\n".join(
            f"{section}.{key}: {value}"
            for section, values in run_config.items()
            for key, value in values.items()
        ),
        0,
    )
    logger.experiment.flush()
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
