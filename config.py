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


@dataclass
class DataConfig:
    IMAGE_SIZE: int = 256
    NUM_KEYPOINTS: int = 294
    MAX_OBJECTS: int = 100


@dataclass
class ModelConfig:
    d_model: int = 1024
    dropout: float = 0.1
    num_layers: int = 6
    height: int = 32
    width: int = 32
    max_objects: int = DataConfig.MAX_OBJECTS
    num_classes: int = len(keypoint_indices) + 1
    num_keypoints: int = DataConfig.NUM_KEYPOINTS
    # matcher parameters
    bbox_matching_weight: float = 1e-2
    class_matching_weight: float = 1.0
    keypoint_matching_weight: float = 1e-2
    # loss function parameters
    ce_class_loss_weight: float = 1
    mse_bbox_loss_weight: float = 1
    giou_bbox_loss_weight: float = 1
    mse_keypoints_loss_weight: float = 1
    class0_weight: float = 1e-1
    # optimizer and scheduler parameters
    feature_lr: float = 1e-4
    transformer_lr: float = 1e-5


IMAGE_SIZE = 128

# object parameters
MIN_NUM_OBJS = 1
MAX_NUM_OBJS = 2
NUM_QUERIES = 10
ROT_DEG_AUG = 15
MIN_OBJS_SIZE = 10
MAX_OBJS_SIZE = 40

# model parameters
D_MODEL = 512
BACKBONE_NUM_LAYERS = 19
FEATURE_HEIGHT = IMAGE_SIZE // 8
FEATURE_WIDTH = IMAGE_SIZE // 8
NUM_CLASSES = 3  # background, rectangle, circle
CLASS_FREQUENCIES = [30.0, 1.0, 1.0]
CLASS_NAMES = ["background", "rectangle", "circle"]
DROPOUT = 0.15

# training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
CLASS_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
MATCHER_CLASS_WEIGHT = 1.0
MATCHER_BBOX_WEIGHT = 5.0
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY_N_EPOCHS = 1
LOG_DIR = "runs"
TENSORBOARD_NUM_IMAGES = 4
TENSORBOARD_MAX_PREDICTIONS = MAX_NUM_OBJS
TENSORBOARD_IMAGE_SCALE = 4
TENSORBOARD_BOX_FONT = "DejaVuSans-Bold.ttf"
TENSORBOARD_BOX_FONT_SIZE = 30
PREDICTION_SCORE_THRESHOLD = 0.25
