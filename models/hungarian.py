from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """Match predicted boxes to target boxes with a weighted Hungarian cost."""

    def __init__(self, class_weight: float = 1.0, bbox_weight: float = 1.0) -> None:
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight

    def forward(
        self,
        pred_class_probs: torch.Tensor,
        pred_boxes: torch.Tensor,
        target: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Return class targets and fancy-index tuples for matched bboxes.

        Args:
            pred_class_probs: Tensor of shape ``[batch, num_preds, num_classes]``.
            pred_boxes: Tensor of shape ``[batch, num_preds, 4]``. These boxes
                must use the same coordinate scale as ``target["boxes"]``.
            target: Dict containing ``boxes`` and ``labels`` tensors. If present,
                ``object_mask`` marks real target slots and padded targets are ignored.

        Returns:
            ``target_classes`` has shape ``[batch, num_preds]`` and uses class ``0``
            for unmatched/background predictions. ``pred_bbox_indices`` indexes
            ``pred_boxes`` and ``target_bbox_indices`` indexes ``target["boxes"]``.
        """

        target_boxes = target["boxes"]
        target_labels = target["labels"]
        object_mask = target.get(
            "object_mask",
            torch.ones_like(target_labels, dtype=torch.bool),
        )

        batch_size, num_preds, num_classes = pred_class_probs.shape
        pred_device = pred_class_probs.device
        target_index_device = target_boxes.device

        target_boxes_for_cost = target_boxes.to(device=pred_device)
        target_labels_for_cost = target_labels.to(device=pred_device)
        object_mask_for_cost = object_mask.to(device=pred_device)

        bbox_cost = torch.cdist(pred_boxes, target_boxes_for_cost, p=2)
        target_labels_one_hot = F.one_hot(
            target_labels_for_cost,
            num_classes=num_classes,
        )
        target_labels_one_hot = target_labels_one_hot.float()
        class_cost = torch.cdist(pred_class_probs, target_labels_one_hot, p=2)
        cost = bbox_cost * self.bbox_weight + class_cost * self.class_weight

        target_classes = torch.zeros(
            (batch_size, num_preds),
            dtype=torch.long,
            device=pred_device,
        )

        batch_indices = []
        pred_indices = []
        target_indices = []

        for batch_idx in range(batch_size):
            valid_target_indices = object_mask_for_cost[batch_idx].nonzero(
                as_tuple=True
            )[0]

            valid_cost = cost[batch_idx, :, valid_target_indices]
            matched_pred_indices, matched_target_positions = linear_sum_assignment(
                valid_cost.detach().cpu().numpy()
            )

            matched_pred_indices = torch.as_tensor(
                matched_pred_indices,
                dtype=torch.long,
                device=pred_device,
            )
            matched_target_positions = torch.as_tensor(
                matched_target_positions,
                dtype=torch.long,
                device=pred_device,
            )
            matched_target_indices = valid_target_indices[matched_target_positions]

            target_classes[batch_idx, matched_pred_indices] = target_labels_for_cost[
                batch_idx,
                matched_target_indices,
            ]

            batch_indices.append(torch.full_like(matched_pred_indices, batch_idx))
            pred_indices.append(matched_pred_indices)
            target_indices.append(matched_target_indices)

        if batch_indices:
            pred_batch_indices = torch.cat(batch_indices)
            pred_match_indices = torch.cat(pred_indices)
            target_match_indices = torch.cat(target_indices)
        else:
            pred_batch_indices = torch.empty(0, dtype=torch.long, device=pred_device)
            pred_match_indices = torch.empty(0, dtype=torch.long, device=pred_device)
            target_match_indices = torch.empty(0, dtype=torch.long, device=pred_device)

        pred_bbox_indices = (pred_batch_indices, pred_match_indices)
        target_bbox_indices = (
            pred_batch_indices.to(device=target_index_device),
            target_match_indices.to(device=target_index_device),
        )

        return target_classes, pred_bbox_indices, target_bbox_indices


if __name__ == "__main__":
    torch_seed = 42
    batch_size = 32
    num_classes = 3
    max_num_objs = 10
    image_size = 128

    _, target = torch.load(
        Path(__file__).resolve().parent / "batch.pt",
        weights_only=False,
    )
    target = {
        "boxes": target["boxes"] / image_size,
        "labels": target["labels"],
        "object_mask": target["object_mask"],
    }

    torch.manual_seed(torch_seed)
    pred_boxes = torch.rand(batch_size, max_num_objs, 4)
    pred_class_scores = torch.randn(batch_size, max_num_objs, num_classes)
    pred_class_probs = torch.softmax(pred_class_scores, dim=-1)

    matcher = HungarianMatcher(class_weight=2.0, bbox_weight=1.0)
    target_classes, pred_bbox_indices, target_bbox_indices = matcher(
        pred_class_probs,
        pred_boxes,
        target,
    )

    matched_pred_boxes = pred_boxes[pred_bbox_indices]
    matched_target_boxes = target["boxes"][target_bbox_indices]
    bbox_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes)

    assert target_classes.shape == (batch_size, max_num_objs)
    assert matched_pred_boxes.shape == matched_target_boxes.shape
    assert matched_pred_boxes.shape == (int(target["object_mask"].sum()), 4)
    assert torch.isfinite(bbox_loss)

    print(target_classes.shape)
    print(matched_pred_boxes.shape, matched_target_boxes.shape)
    print(bbox_loss)

    print(target_classes[1])
    print(pred_bbox_indices)
    print(target_bbox_indices)
