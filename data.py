"""Dataset, DataLoader, and pytorch lightning DataModule"""

import json
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from config import DataConfig, keypoint_indices

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=DataConfig.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=DataConfig.IMAGE_SIZE,
            min_width=DataConfig.IMAGE_SIZE,
            border_mode=cv2.BORDER_REPLICATE,
        ),
        A.Normalize(),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['classes']),
)

train_transforms = A.Compose(
    [
        # spatial
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Affine(mode=cv2.BORDER_REPLICATE),
        A.Perspective(pad_mode=cv2.BORDER_REPLICATE),
        A.Rotate(limit=45, border_mode=cv2.BORDER_REPLICATE),
        A.SmallestMaxSize(max_size=320),
        A.RandomScale(scale_limit=.15),
        A.RandomCrop(
            height=DataConfig.IMAGE_SIZE,
            width=DataConfig.IMAGE_SIZE,
        ),
        # pixel level
        A.RandomBrightnessContrast(p=.15),
        A.AdvancedBlur(p=.15),
        A.ChannelShuffle(p=.15),
        A.MedianBlur(p=.15),
        A.Posterize(p=.15),
        A.Solarize(p=.015),
        # format data
        A.Normalize(),
        ToTensorV2(),
    ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['classes']),
)


class DeepFashion2Dataset(Dataset):
    def __init__(
            self,
            base_path: str,
            transforms: A.Compose,
            max_objects: int,
    ) -> None:
        super().__init__()
        base_path = Path(base_path)
        self._base_path = Path(base_path)
        self._length = len(glob(str(self._base_path / 'image/*.jpg')))
        self._transforms = transforms
        self._max_objects = max_objects

    def __len__(self) -> int:
        return self._length

    def _pad_classes(self, classes: list[int]) -> Tensor:
        classes = torch.LongTensor(classes)
        classes = torch.cat(
            [
                classes,
                torch.zeros(
                    self._max_objects - classes.size(0),
                    dtype=torch.int32,
                ),
            ],
        )
        return classes

    def _pad_bboxes(self, bboxes: list[tuple[float]]) -> Tensor:
        bboxes = torch.FloatTensor(bboxes).clip(0, DataConfig.IMAGE_SIZE)
        bboxes /=  DataConfig.IMAGE_SIZE
        bboxes = torch.cat(
            [
                bboxes,
                torch.zeros(
                    (self._max_objects - bboxes.size(0), 4),
                    dtype=torch.float32,
                ),
            ],
        )
        return bboxes

    def _pad_keypoints(
            self,
            keypoints: list[list[tuple[float]]],
            classes: Tensor,
    ) -> Tensor:
        keypoints = [
            (
                torch.FloatTensor(keypoint).clip(0, DataConfig.IMAGE_SIZE)
                / DataConfig.IMAGE_SIZE
            )
            for keypoint
            in keypoints
        ]
        result = torch.zeros(
            (self._max_objects, DataConfig.NUM_KEYPOINTS, 2),
            dtype=torch.float32,
        )
        for i, (class_, keypoint) in enumerate(zip(classes, keypoints)):
            class_ = class_.item()
            if class_ == 0:
                break
            start, end = keypoint_indices[class_]
            result[i, start:end] = keypoint
        return result

    def _pad_visibilities(
        self,
        visibilities: list[np.ndarray],
        classes: Tensor,
    ) -> Tensor:
        visibilities = [
            torch.FloatTensor(visibility).reshape(-1, 1) / 2.
            for visibility
            in visibilities
        ]
        result = torch.zeros(
            (self._max_objects, DataConfig.NUM_KEYPOINTS, 1),
            dtype=torch.float32,
        )
        for i, (class_, visibility) in enumerate(zip(classes, visibilities)):
            class_ = class_.item()
            if class_ == 0:
                break
            start, end = keypoint_indices[class_]
            result[i, start:end] = visibility
        return result

    def __getitem__(self, index: int) -> tuple[Tensor]:
        # create paths
        image_path = self._base_path / f'image/{index + 1:06d}.jpg'
        annotation_path = self._base_path / f'annos/{index + 1:06d}.json'
        # load image and annotation
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        with open(annotation_path) as f:
            annotation = json.load(f)
        # restructure annotation
        annotation = [
            {
                'bbox': v['bounding_box'],
                'class': v['category_id'],
                'keypoints': np.array(v['landmarks']).reshape(-1, 3)[:, :2],
                'visibilities': np.array(v['landmarks']).reshape(-1, 3)[:, 2],
            }
            for k, v in annotation.items()
            if k.startswith('item')
        ]
        # create keypoint, bbox, and classes lists. (pack keypoints)
        bboxes = [item['bbox'] for item in annotation]
        keypoints = np.concatenate([item['keypoints'] for item in annotation])
        keypoints_border = [item['keypoints'].shape[0] for item in annotation]
        classes = [item['class'] for item in annotation]
        visibilities = [item['visibilities'] for item in annotation]
        # apply transform
        transformed = self._transforms(
            image=image,
            bboxes=bboxes,
            keypoints=keypoints,
            classes=classes,
        )
        # separate transformed results
        image = transformed['image']
        bboxes = transformed['bboxes']
        keypoints = transformed['keypoints']
        classes = transformed['classes']
        # unpack keypoints
        keypoints_border = np.cumsum([0] + keypoints_border)
        iterator = zip(keypoints_border[:-1], keypoints_border[1:])
        keypoints = [keypoints[start:end] for start, end in iterator]
        # normalize and fix length of classes, bboxes, keypoints,
        # and visibilities
        classes = self._pad_classes(classes)
        bboxes = self._pad_bboxes(bboxes)
        keypoints = self._pad_keypoints(keypoints, classes)
        visibilities = self._pad_visibilities(visibilities, classes)
        return image, classes, bboxes, keypoints, visibilities


if __name__ == '__main__':
    ds = DeepFashion2Dataset(
        base_path='/home/aj/data/DeepFashion2/validation',
        transforms=train_transforms,
        # transforms=val_transforms,
        max_objects=10,
    )
    image, classes, bboxes, keypoints, visibilities = ds[0]
    from torchvision.utils import save_image
    save_image(image, '/tmp/tmp.png')
