"""Pytorch dataLoader and pytorch lightning DataModule"""

from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_pt import DeepFashion2Dataset, train_transforms, val_transforms


class DeepFashion2DataModule(LightningDataModule):
    def __init__(
            self,
            train_base_path: str,
            val_base_path: str,
            max_objects: int = 10,
            batch_size: int = 64,
            num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_base_path = train_base_path
        self.val_base_path = val_base_path
        self.max_objects = max_objects
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DeepFashion2Dataset(
            self.train_base_path,
            train_transforms,
            self.max_objects,
        )
        self.val_dataset = DeepFashion2Dataset(
            self.val_base_path,
            val_transforms,
            self.max_objects,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            # pin_memory_device=torch.device('cuda', index=0),
        )

    def val_dataloader(self) -> DataLoader:
        DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            # pin_memory_device=torch.device('cuda', index=0),
        )


if __name__ == '__main__':
    dm = DeepFashion2DataModule(
       '/home/aj/data/DeepFashion2/train',
       '/home/aj/data/DeepFashion2/validation',
    )
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    from tqdm import tqdm
    for image, classes, bboxes, keypoints, visibilities in tqdm(train_dl):
        print(classes)
        break
