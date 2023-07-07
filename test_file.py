import argparse
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

from data.config import DataConfig, keypoint_indices
from models.model_pl import TransformerModelPL

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', default='/home/aj/data/DeepFashion2/test/image/000001.jpg')
parser.add_argument('--checkpoint-path', default='lightning_logs/version_3/checkpoints/epoch=5-step=42000.ckpt')
parser.add_argument('--device', default='cpu')
parser.add_argument('--device-index', type=int, default=0)
args = parser.parse_args()

if args.device == 'cpu':
    device = torch.device(args.device)
else:
    device = torch.device(args.device, index=args.device_index)

# data
image_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=DataConfig.IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=DataConfig.IMAGE_SIZE,
            min_width=DataConfig.IMAGE_SIZE,
            border_mode=cv2.BORDER_REPLICATE,
        ),
    ],
)
pytorch_transforms = A.Compose(
    [
        A.Normalize(),
        ToTensorV2(),
    ],
)
image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
transformed = image_transforms(image=image)
image = transformed['image']
transformed = pytorch_transforms(image=image)
x = transformed['image']
x = x.view(1, 3, DataConfig.IMAGE_SIZE, DataConfig.IMAGE_SIZE)
x = x.to(device)

# model
model = TransformerModelPL.load_from_checkpoint(
    args.checkpoint_path,
    map_location=device,
)

# inference
with torch.inference_mode():
    predicted_classes, predicted_bboxes, predicted_keypoints = model(x)
class_probs, class_indices = (
    predicted_classes
    .squeeze(0)
    .softmax(dim=-1)
    .max(dim=-1)
)
bboxes = (predicted_bboxes.squeeze(0) * DataConfig.IMAGE_SIZE).long().tolist()
keypoints = (predicted_keypoints.squeeze(0) * DataConfig.IMAGE_SIZE).long().tolist()

# visualize
for class_prob, class_index, box, keypoint in zip(class_probs, class_indices, bboxes, keypoints):
    if class_index.item() == 0:
        continue
    cv2.rectangle(image, box[:2], box[2:4], color=(255, 0, 0), thickness=1)
    start_index, end_index = keypoint_indices[class_index.item()]
    for x, y in keypoint[start_index:end_index]:
        image[x, y] = (0, 0, 255)
        image[x + 1, y + 1] = (0, 0, 255)
        image[x + 1, y - 1] = (0, 0, 255)
        image[x - 1, y + 1] = (0, 0, 255)
        image[x - 1, y - 1] = (0, 0, 255)

# save result
base_path = Path('/tmp')
image_path = Path(args.image_path)
cv2.imwrite(str(base_path / image_path.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
