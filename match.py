"""Match algorithm to find optimal pred-to-target assignment"""

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch import Tensor


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
            pred_classes: Tensor,
            target_classes: Tensor,
            pred_bboxes: Tensor,
            target_bboxes: Tensor,
            pred_keypoints: Tensor,
            target_keypoints: Tensor,
            visibilities: Tensor,
    ) -> list:
        pass


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
