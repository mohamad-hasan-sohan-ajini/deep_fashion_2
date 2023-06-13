"""Match algorithm to find optimal pred-to-target assignment"""

from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


def hard_classification_cost_function(
        pred_logits: Tensor,
        target_classes: Tensor,
        _: int,
) -> Tensor:
    pred_classes = pred_logits.argmax(-1).float().unsqueeze(-1)
    target_classes = target_classes.float().unsqueeze(-1)
    cost = (torch.cdist(pred_classes, target_classes) != 0).float()
    return cost


def soft_classification_cost_function(
        pred_logits: Tensor,
        target_classes: Tensor,
        num_classes: int,
) -> Tensor:
    pred_classes = pred_logits.softmax(-1)
    target_classes = F.one_hot(target_classes, num_classes).float()
    cost = torch.cdist(pred_classes, target_classes)
    return cost


def bbox_cost_function(pred_bboxes: Tensor, target_bboxes: Tensor) -> Tensor:
    cost = torch.cdist(pred_bboxes, target_bboxes)
    return cost


def keypoint_cost_function(
        pred_keypoints: Tensor,
        target_keypoints: Tensor,
        visibilities: Tensor,
) -> Tensor:
    batch_size, num_objects, *_ = pred_keypoints.size()
    pred_keypoints = (pred_keypoints * visibilities)
    pred_keypoints = pred_keypoints.view(batch_size, num_objects, -1)
    target_keypoints = target_keypoints * visibilities
    target_keypoints = target_keypoints.view(batch_size, num_objects, -1)
    cost = torch.cdist(pred_keypoints, target_keypoints)
    return cost


class Matcher:
    def __init__(
            self,
            classification_cost_function: Callable[[Tensor, Tensor], Tensor],
            classification_weight: float,
            bbox_cost_function: Callable[[Tensor, Tensor], Tensor],
            bbox_weight: float,
            keypoint_cost_function: Callable[[Tensor, Tensor], Tensor],
            keypoint_weight: float,
    ) -> None:
        """Initialize

        :param classification_cost_function: A python callable that
            takes model class logits and ground truth classes of shapes
            (N, T) and (N, T, C) respectively and returns a (N, T, T)
            Tensor that contains class matching costs, where T is
            maximum number of objects in dataset images.
        :param classification_weight: Weight of classification errors.
        :param bbox_cost_function: A python callable that takes model
            bbox predictions and ground truth bboxes, both having shape
            of (N, T, 4), and returns a (N, T, T) Tensor that
            contains bbox matching costs. T is maximum number of
            objects in dataset images.
        :param bbox_weight: Weight of bbox matching errors.
        :param keypoint_cost_function: A python callable that takes
            model keypoint predictions, ground truth keypoints
            predictions, and their corresponding visibilities of shape
            (N, T, 294, 2), (N, T, 294, 2), (N, T, 294, 1) respectively
            and returns a (N, T, T) Tensor that contains bbox matching
            costs. T is maximum number of objects in dataset images.
        :param keypoint_weight: Weight of keypoint matching errors.
        """
        self.classification_cost_function = classification_cost_function
        self.classification_weight = classification_weight
        self.bbox_cost_function = bbox_cost_function
        self.bbox_weight = bbox_weight
        self.keypoint_cost_function = keypoint_cost_function
        self.keypoint_weight = keypoint_weight

    def __call__(
            self,
            pred_logits: Tensor,
            target_classes: Tensor,
            pred_bboxes: Tensor,
            target_bboxes: Tensor,
            pred_keypoints: Tensor,
            target_keypoints: Tensor,
            visibilities: Tensor,
    ) -> list:
        costs = (
            (
                self.classification_weight
                * self.classification_cost_function(
                    pred_logits,
                    target_classes,
                )
            )
            + (
                self.bbox_weight
                * self.bbox_cost_function(pred_bboxes, target_bboxes)
            )
            + (
                self.keypoint_weight
                * self.keypoint_cost_function(
                    pred_keypoints,
                    target_keypoints,
                    visibilities,
                )
            )
        )
        pred_indices, target_indices = [], []
        for cost in costs:
            pred_index, target_index = linear_sum_assignment(cost)
            pred_indices.append(pred_index)
            target_indices.append(target_index)
        return pred_indices, target_indices


if __name__ == '__main__':
    cost_mat = np.array(
        [
            [10, 12, 19, 11],
            [5, 10, 7, 8],
            [12, 14, 13, 11],
            [8, 12, 11, 9],
        ]
    )
    cost_mat = np.random.randn(1024, 1024)
    x_indices, y_indices = linear_sum_assignment(cost_mat)

    # test class cost functions
    target_classes = torch.tensor(
        [
            [3, 1, 0, 0],
            [5, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    print(target_classes)
    pred_logits = torch.randn(2, 4, 6)
    print(pred_logits.softmax(-1))
    print(pred_logits.argmax(-1))
    # call functions
    hard_cost = hard_classification_cost_function(
        pred_logits.clone(),
        target_classes.clone(),
    )
    print(f'{hard_cost = }')
    soft_cost = soft_classification_cost_function(
        pred_logits.clone(),
        target_classes.clone(),
        num_classes=6,
    )
    print(f'{soft_cost = }')

    # test bbox cost function
    target_bboxes = torch.randn(2, 4, 4)
    pred_bboxes = torch.randn(2, 4, 4)
    bbox_cost = bbox_cost_function(pred_bboxes, target_bboxes)
    print(f'{bbox_cost.shape = }')

    # test keypoints cost function
    target_keypoints = torch.randn(2, 4, 294, 2)
    pred_keypoints = torch.randn(2, 4, 294, 2)
    visibilities = torch.randn(2, 4, 294, 1)
    keypoints_cost = keypoint_cost_function(
        pred_keypoints,
        target_keypoints,
        visibilities,
    )
    print(f'{keypoints_cost.shape = }')
