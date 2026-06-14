"""Transformer model with pytorch lightning (+criterion +metrics)"""

from itertools import chain
from typing import Callable

from pytorch_lightning import LightningModule
from torch import Tensor, nn, ones, optim
from torchmetrics import Accuracy
from torchvision.ops import generalized_box_iou_loss

from config import ModelConfig
from models.hungarian import HungarianMatcher
from models.model_pt import TransformerModel
from models.positional_encoding import (
    FixedPositionalEncoding2D,
    PositionalEncoding2D,
)
from models.utils import get_resnet_backbone


class TransformerModelPL(LightningModule):
    def __init__(
        self,
        backbone_builder: Callable = get_resnet_backbone,
        feature_num_layers: int = 18,
        positional_encoding_builder: PositionalEncoding2D = FixedPositionalEncoding2D,
    ) -> None:
        super().__init__()
        self.model = TransformerModel(
            backbone_builder,
            feature_num_layers,
            positional_encoding_builder,
            d_model=ModelConfig.d_model,
            height=ModelConfig.height,
            width=ModelConfig.width,
            max_objects=ModelConfig.max_objects,
            num_classes=ModelConfig.num_classes,
            dropout=ModelConfig.dropout,
        )
        self.matcher = HungarianMatcher(
            class_weight=ModelConfig.class_matching_weight,
            bbox_weight=ModelConfig.bbox_matching_weight,
        )
        class_weights = ones(ModelConfig.num_classes)
        class_weights[0] = ModelConfig.class0_weight
        self.class_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.point_criterion = nn.SmoothL1Loss(reduction="mean")
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=ModelConfig.num_classes,
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(images)

    def configure_optimizers(self) -> dict[str, optim.Optimizer]:
        optimizer = optim.Adam(
            [
                {
                    "params": chain(
                        self.model.positional_encoder.parameters(),
                        self.model.object_queries.parameters(),
                        self.model.class_ffn.parameters(),
                        self.model.bbox_ffn.parameters(),
                    ),
                    "lr": ModelConfig.feature_lr,
                },
                {
                    "params": chain(
                        self.model.feature_extractor.parameters(),
                        self.model.encoder1.parameters(),
                        self.model.encoder2.parameters(),
                        self.model.encoder3.parameters(),
                        self.model.decoder1.parameters(),
                        self.model.decoder2.parameters(),
                        self.model.decoder3.parameters(),
                    ),
                    "lr": ModelConfig.transformer_lr,
                },
            ],
        )
        return {"optimizer": optimizer}

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_index: int,
    ) -> dict[str, Tensor]:
        images, gt_classes, gt_bboxes, _, _, object_mask = batch
        pred_classes, pred_bboxes = self(images)
        target = {
            "labels": gt_classes,
            "boxes": gt_bboxes,
            "object_mask": object_mask,
        }
        target_classes, pred_bbox_indices, target_bbox_indices = self.matcher(
            pred_classes.softmax(dim=-1),
            pred_bboxes,
            target,
        )
        matched_pred_bboxes = pred_bboxes[pred_bbox_indices]
        matched_gt_bboxes = gt_bboxes[target_bbox_indices]

        class_loss = self.class_criterion(
            pred_classes.flatten(0, 1),
            target_classes.flatten(),
        )
        giou_bbox_loss = generalized_box_iou_loss(
            matched_pred_bboxes,
            matched_gt_bboxes,
            reduction="mean",
        )
        loss = (
            class_loss * ModelConfig.ce_class_loss_weight
            + giou_bbox_loss * ModelConfig.giou_bbox_loss_weight
        )
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_index: int,
    ) -> dict[str, Tensor]:
        images, gt_classes, gt_bboxes, _, _, object_mask = batch
        pred_classes, pred_bboxes = self(images)
        target = {
            "labels": gt_classes,
            "boxes": gt_bboxes,
            "object_mask": object_mask,
        }
        target_classes, pred_bbox_indices, target_bbox_indices = self.matcher(
            pred_classes.softmax(dim=-1),
            pred_bboxes,
            target,
        )
        matched_pred_bboxes = pred_bboxes[pred_bbox_indices]
        matched_gt_bboxes = gt_bboxes[target_bbox_indices]
        predicted_classes = pred_classes.argmax(dim=-1)
        object_indices = target_classes > 0

        class_accuracy_w0 = self.accuracy(predicted_classes, target_classes)
        if object_indices.any():
            class_accuracy_wo0 = (
                (predicted_classes[object_indices] == target_classes[object_indices])
                .float()
                .mean()
            )
            bbox_giou = generalized_box_iou_loss(
                matched_pred_bboxes,
                matched_gt_bboxes,
                reduction="mean",
            )
            bbox_l1 = self.point_criterion(
                matched_pred_bboxes,
                matched_gt_bboxes,
            ).mean()
        else:
            class_accuracy_wo0 = pred_classes.new_zeros(())
            bbox_giou = pred_bboxes.new_zeros(())
            bbox_l1 = pred_bboxes.new_zeros(())
        self.log("class_accuracy_w0", class_accuracy_w0)
        self.log("class_accuracy_wo0", class_accuracy_wo0)
        self.log("bbox_giou", bbox_giou)
        self.log("bbox_l1", bbox_l1)
        result_dict = {
            "class_accuracy_w0": class_accuracy_w0,
            "class_accuracy_wo0": class_accuracy_wo0,
            "bbox_giou": bbox_giou,
            "bbox_l1": bbox_l1,
        }
        return result_dict


if __name__ == "__main__":
    import torch

    batch_dict = torch.load("batch.pt")
    batch = (
        batch_dict["images"],
        batch_dict["classes"],
        batch_dict["bboxes"],
        batch_dict["keypoints"],
        batch_dict["visibilities"],
        batch_dict["classes"] != 0,
    )
    pl_model = TransformerModelPL()
    pl_model.training_step(batch, 0)
