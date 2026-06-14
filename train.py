"""Train the DeepFashion2 transformer model."""

import argparse
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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
    parser.add_argument("--log-every-n-steps", type=int, default=1000)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("lightning_logs"),
        help="Root directory for TensorBoard logs and checkpoints",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint from which to resume training",
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
    # check defaults in model_pl.py for default constructors and parameters
    model = TransformerModelPL()
    logger = TensorBoardLogger(save_dir=args.log_dir, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="class_accuracy_w0",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
        filename="{epoch:04d}-{class_accuracy_w0:.4f}",
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        logger=logger,
        fast_dev_run=True,  # Run one training and validation batch for debugging
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path else None,
    )


if __name__ == "__main__":
    main()
