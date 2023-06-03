"""Dataset, DataLoader, and pytorch lightning DataModule"""

import json
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

val_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=256),
        A.PadIfNeeded(
            min_height=256,
            min_width=256,
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
        A.RandomCrop(height=256, width=256),
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
    def __init__(self, base_path: str, transforms: A.Compose) -> None:
        super().__init__()
        base_path = Path(base_path)
        self._base_path = Path(base_path)
        self._length = len(glob(str(self._base_path / 'image/*.jpg')))
        self._transforms = transforms

    def __len__(self) -> int:
        return self._length

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
                'class': v['category_name'],
                'keypoints': np.array(v['landmarks']).reshape(-1, 3)[:, :2],
                'visibility': np.array(v['landmarks']).reshape(-1, 3)[:, 2],
            }
            for k, v in annotation.items()
            if k.startswith('item')
        ]
        # create keypoint, bbox, and classes lists. (pack keypoints)
        bboxes = [item['bbox'] for item in annotation]
        keypoints = np.concatenate([item['keypoints'] for item in annotation])
        keypoints_border = [item['keypoints'].shape[0] for item in annotation]
        classes = [item['class'] for item in annotation]
        visibility = [item['visibility'] for item in annotation]
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
        return image, bboxes, keypoints, classes, visibility


if __name__ == '__main__':
    ds = DeepFashion2Dataset(
        '/home/aj/data/DeepFashion2/validation',
        train_transforms,
        # val_transforms,
    )
    image, bboxes, keypoints, classes, visibility = ds[0]
    from torchvision.utils import save_image
    save_image(image, '/tmp/tmp.png')
