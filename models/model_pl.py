"""Transformer model with pytorch lightning (+criterion +metrics)"""

from itertools import chain
from typing import Callable

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn, ones, optim
from torchvision.ops import generalized_box_iou_loss
from torchvision.utils import draw_bounding_boxes

from config import ModelConfig, class_names
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
        scalar_log_every_n_batches: int = 100,
        image_log_every_n_batches: int = 800,
        tensorboard_num_images: int = ModelConfig.tensorboard_num_images,
    ) -> None:
        super().__init__()
        self.scalar_log_every_n_batches = scalar_log_every_n_batches
        self.image_log_every_n_batches = image_log_every_n_batches
        self.tensorboard_num_images = tensorboard_num_images
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

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.model(images)

    @staticmethod
    def _ordered_boxes(boxes: Tensor) -> Tensor:
        xy_min = torch.minimum(boxes[..., :2], boxes[..., 2:])
        xy_max = torch.maximum(boxes[..., :2], boxes[..., 2:])
        return torch.cat((xy_min, xy_max), dim=-1)

    def _log_predictions(
        self,
        tag_prefix: str,
        step: int,
        images: Tensor,
        pred_classes: Tensor,
        pred_bboxes: Tensor,
    ) -> None:
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return

        images = images.detach().cpu()
        pred_classes = pred_classes.detach().cpu()
        pred_bboxes = pred_bboxes.detach().cpu()
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        num_images = min(self.tensorboard_num_images, images.size(0))

        for image_index in range(num_images):
            image = ((images[image_index] * std + mean).clamp(0, 1) * 255).to(
                torch.uint8
            )
            probabilities = pred_classes[image_index].softmax(dim=-1)
            scores, labels = probabilities.max(dim=-1)
            keep = (labels != 0) & (
                scores >= ModelConfig.prediction_score_threshold
            )
            scores = scores[keep]
            labels = labels[keep]
            boxes = pred_bboxes[image_index][keep]

            if scores.numel():
                num_predictions = min(
                    ModelConfig.tensorboard_max_predictions,
                    scores.numel(),
                )
                scores, indices = scores.topk(num_predictions)
                labels = labels[indices]
                boxes = self._ordered_boxes(boxes[indices].clamp(0, 1))
                height, width = image.shape[-2:]
                scale = boxes.new_tensor((width, height, width, height))
                boxes = boxes * scale
                valid = (boxes[:, 2] > boxes[:, 0]) & (
                    boxes[:, 3] > boxes[:, 1]
                )
                boxes = boxes[valid]
                scores = scores[valid]
                labels = labels[valid]

            if boxes.numel():
                box_labels = [
                    f"{class_names[int(label)]} {float(score):.2f}"
                    for label, score in zip(labels, scores)
                ]
                rendered_image = draw_bounding_boxes(
                    image,
                    boxes,
                    labels=box_labels,
                    colors="red",
                    width=2,
                )
            else:
                rendered_image = image
            experiment.add_image(
                f"{tag_prefix}/predictions/{image_index}",
                rendered_image,
                step,
            )
        experiment.flush()

    def _log_batch_metrics(
        self,
        step: int,
        loss: Tensor,
        class_loss: Tensor,
        bbox_loss: Tensor,
        class_accuracy: Tensor,
    ) -> None:
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_scalar"):
            return

        experiment.add_scalar("batch/loss", loss.detach(), step)
        experiment.add_scalar("batch/class_loss", class_loss.detach(), step)
        experiment.add_scalar("batch/bbox_loss", bbox_loss.detach(), step)
        experiment.add_scalar(
            "batch/class_accuracy",
            class_accuracy.detach(),
            step,
        )
        experiment.add_scalar(
            "batch/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            step,
        )
        experiment.flush()

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
        matched_pred_bboxes = self._ordered_boxes(
            pred_bboxes[pred_bbox_indices]
        )
        matched_gt_bboxes = self._ordered_boxes(gt_bboxes[target_bbox_indices])

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
        predicted_classes = pred_classes.argmax(dim=-1)
        class_accuracy = (
            predicted_classes == target_classes
        ).float().mean()
        log_options = {
            "on_step": False,
            "on_epoch": True,
            "batch_size": images.size(0),
        }
        self.log("loss/train", loss, prog_bar=True, **log_options)
        self.log("class_loss/train", class_loss, **log_options)
        self.log("bbox_loss/train", giou_bbox_loss, **log_options)
        self.log("class_accuracy/train", class_accuracy, **log_options)
        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            **log_options,
        )

        mini_batch_step = (
            self.current_epoch * int(self.trainer.num_training_batches)
            + batch_index
            + 1
        )
        if (
            self.trainer.is_global_zero
            and self.scalar_log_every_n_batches > 0
            and mini_batch_step % self.scalar_log_every_n_batches == 0
        ):
            self._log_batch_metrics(
                mini_batch_step,
                loss,
                class_loss,
                giou_bbox_loss,
                class_accuracy,
            )
        if (
            self.trainer.is_global_zero
            and self.image_log_every_n_batches > 0
            and mini_batch_step % self.image_log_every_n_batches == 0
        ):
            self._log_predictions(
                "training",
                mini_batch_step,
                images,
                pred_classes,
                pred_bboxes,
            )
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
        matched_pred_bboxes = self._ordered_boxes(
            pred_bboxes[pred_bbox_indices]
        )
        matched_gt_bboxes = self._ordered_boxes(gt_bboxes[target_bbox_indices])
        predicted_classes = pred_classes.argmax(dim=-1)
        object_indices = target_classes > 0

        class_loss = self.class_criterion(
            pred_classes.flatten(0, 1),
            target_classes.flatten(),
        )
        class_accuracy_w0 = (
            predicted_classes == target_classes
        ).float().mean()
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
        loss = (
            class_loss * ModelConfig.ce_class_loss_weight
            + bbox_giou * ModelConfig.giou_bbox_loss_weight
        )
        log_options = {
            "on_step": False,
            "on_epoch": True,
            "batch_size": images.size(0),
        }
        self.log("loss/val", loss, prog_bar=True, **log_options)
        self.log("class_loss/val", class_loss, **log_options)
        self.log("bbox_loss/val", bbox_giou, **log_options)
        self.log(
            "class_accuracy/val",
            class_accuracy_w0,
            prog_bar=True,
            **log_options,
        )
        self.log(
            "class_accuracy_without_background/val",
            class_accuracy_wo0,
            **log_options,
        )
        self.log("bbox_l1/val", bbox_l1, **log_options)

        if batch_index == 0 and not self.trainer.sanity_checking:
            self._log_predictions(
                "validation",
                self.current_epoch + 1,
                images,
                pred_classes,
                pred_bboxes,
            )

        result_dict = {
            "loss": loss,
            "class_loss": class_loss,
            "bbox_loss": bbox_giou,
            "class_accuracy": class_accuracy_w0,
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
