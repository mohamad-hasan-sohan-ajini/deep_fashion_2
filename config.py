"""Project configuration."""

from dataclasses import dataclass

keypoint_indices = {
    1: (0, 25),
    2: (25, 58),
    3: (58, 89),
    4: (89, 128),
    5: (128, 143),
    6: (143, 158),
    7: (158, 168),
    8: (168, 182),
    9: (182, 190),
    10: (190, 219),
    11: (219, 256),
    12: (256, 275),
    13: (275, 294),
}

class_names = [
    "background",
    "short sleeve top",
    "long sleeve top",
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress",
]


@dataclass
class DataConfig:
    IMAGE_SIZE: int = 256
    NUM_KEYPOINTS: int = 294
    MAX_OBJECTS: int = 10


@dataclass
class ModelConfig:
    d_model: int = 1024
    dropout: float = 0.15
    num_layers: int = 6
    height: int = 32
    width: int = 32
    max_objects: int = DataConfig.MAX_OBJECTS
    num_classes: int = len(keypoint_indices) + 1  # 0: no object, 1-13: object classes
    num_keypoints: int = DataConfig.NUM_KEYPOINTS
    # matcher parameters
    bbox_matching_weight: float = 1.0
    class_matching_weight: float = 1.0
    keypoint_matching_weight: float = 0.0
    # loss function parameters
    ce_class_loss_weight: float = 1
    giou_bbox_loss_weight: float = 1
    mse_keypoints_loss_weight: float = 1
    class0_weight: float = 0.25
    # optimizer and scheduler parameters
    feature_lr: float = 1e-4
    transformer_lr: float = 1e-5
    prediction_score_threshold: float = 0.5
    tensorboard_num_images: int = 4
    tensorboard_max_predictions: int = 10
