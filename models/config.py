"""Model configuration"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    dropout: float = .1
    num_layers: int = 6
    height: int = 8
    width: int = 8
    max_objects: int = 10
    num_classes: int = 13 + 1
    num_keypoints: int = 294
    # matcher parameters
    bbox_matching_weight: float = 5.
    class_matching_weight: float = 1.
    keypoint_matching_weight: float = 1e-2
    # loss function parameters
    ce_class_loss_weight: float = 1
    mse_bbox_loss_weight: float = 2
    giou_bbox_loss_weight: float = 2
    mse_keypoints_loss_weight: float = 1e-2
    class0_weight: float = 3e-2
    # optimizer and scheduler parameters
    feature_lr: float = 1e-4
    transformer_lr: float = 1e-5
