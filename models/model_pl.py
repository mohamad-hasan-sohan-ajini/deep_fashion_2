"""Transformer model with pytorch lightning (+criterion +metrics)"""

from itertools import chain
from typing import Any, Callable

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, ones, optim
from torchmetrics import Accuracy
from torchvision.ops import generalized_box_iou_loss

from models.config import ModelConfig
from models.match import (
    Matcher,
    bbox_cost_function,
    keypoint_cost_function,
    soft_classification_cost_function,
)
from models.model_pt import TransformerModel
from models.positional_encoding import(
    FixedPositionalEncoding2D,
    LearnablePositionalEncoding2D,
    PositionalEncoding2D,
)
from models.utils import get_vgg_backbone, get_resnet_backbone


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
        )
        self.matcher = Matcher(
            soft_classification_cost_function,
            ModelConfig.class_matching_weight,
            bbox_cost_function,
            ModelConfig.bbox_matching_weight,
            keypoint_cost_function,
            ModelConfig.keypoint_matching_weight,
            ModelConfig.num_classes,
        )
        class_weights = ones(ModelConfig.num_classes)
        class_weights[0] = ModelConfig.class0_weight
        self.class_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.point_criterion = nn.SmoothL1Loss(reduction='none')
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=ModelConfig.num_classes,
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def configure_optimizers(self) -> dict[str, optim.Optimizer]:
        optimizer = optim.Adam(
            [
                {
                    'params': chain(
                        self.model.positional_encoder.parameters(),
                        self.model.object_queries.parameters(),
                        self.model.class_ffn.parameters(),
                        self.model.bbox_ffn.parameters(),
                        self.model.keypoints_ffn.parameters(),
                    ),
                    'lr': ModelConfig.feature_lr,
                },
                {
                    'params': chain(
                        self.model.feature_extractor.parameters(),
                        self.model.encoder.parameters(),
                        self.model.decoder.parameters(),
                    ),
                    'lr': ModelConfig.transformer_lr,
                },
            ],
        )
        return {'optimizer': optimizer}

    def training_step(
            self,
            batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
            batch_index: int,
    ) -> dict[str, Tensor]:
        images, gt_classes, gt_bboxes, gt_keypoints, gt_visibilities = batch
        pred_classes, pred_bboxes, pred_keypoints = self(images)
        # find best matchings
        target_indices = self.matcher(
            pred_classes,
            gt_classes,
            pred_bboxes,
            gt_bboxes,
            pred_keypoints,
            gt_keypoints,
            gt_visibilities,
        )
        # collapse batch and objects dims in predictions
        pred_classes = pred_classes.view(-1, ModelConfig.num_classes)
        pred_bboxes = pred_bboxes.view(-1, 4)
        pred_keypoints = pred_keypoints.view(-1, ModelConfig.num_keypoints, 2)
        # shuffle GT indices
        gt_classes = gt_classes.view(-1)[target_indices]
        gt_bboxes = gt_bboxes.view(-1, 4)[target_indices]
        gt_keypoints = gt_keypoints.view(-1, ModelConfig.num_keypoints, 2)
        gt_keypoints = gt_keypoints[target_indices]
        gt_visibilities = gt_visibilities.view(-1, ModelConfig.num_keypoints)
        gt_visibilities = gt_visibilities[target_indices]
        # create class0 mask to be used in loss computation of bboxes
        # and keypoints
        class0_mask = (gt_classes > 0).float()
        # compute loss
        class_loss = self.class_criterion(pred_classes, gt_classes)
        giou_bbox_loss = (
            generalized_box_iou_loss(pred_bboxes, gt_bboxes, reduction='none')
            * class0_mask
        )
        mse_bbox_loss = (
            self.point_criterion(pred_bboxes, gt_bboxes).sum(dim=1)
            * class0_mask
        )
        keypoint_loss = self.point_criterion(pred_keypoints, gt_keypoints)
        keypoint_loss = (keypoint_loss.sum(dim=2) * gt_visibilities).sum(dim=1)
        keypoint_loss = keypoint_loss * class0_mask
        # sum up losses
        loss = (
            (class_loss * ModelConfig.ce_class_loss_weight)
            + (giou_bbox_loss.sum() * ModelConfig.giou_bbox_loss_weight)
            + (mse_bbox_loss.sum() * ModelConfig.mse_bbox_loss_weight)
            + (keypoint_loss.sum() * ModelConfig.mse_keypoints_loss_weight)
        )
        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(
            self,
            batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
            batch_index: int,
    ) -> dict[str, Tensor]:
        images, gt_classes, gt_bboxes, gt_keypoints, gt_visibilities = batch
        pred_classes, pred_bboxes, pred_keypoints = self(images)
        # find best matchings
        target_indices = self.matcher(
            pred_classes,
            gt_classes,
            pred_bboxes,
            gt_bboxes,
            pred_keypoints,
            gt_keypoints,
            gt_visibilities,
        )
        # collapse batch and objects dims in predictions
        pred_classes = pred_classes.view(-1, ModelConfig.num_classes)
        pred_bboxes = pred_bboxes.view(-1, 4)
        pred_keypoints = pred_keypoints.view(-1, ModelConfig.num_keypoints, 2)
        # shuffle GT indices
        gt_classes = gt_classes.view(-1)[target_indices]
        gt_bboxes = gt_bboxes.view(-1, 4)[target_indices]
        gt_keypoints = gt_keypoints.view(-1, ModelConfig.num_keypoints, 2)
        gt_keypoints = gt_keypoints[target_indices]
        gt_visibilities = gt_visibilities.view(-1, ModelConfig.num_keypoints)
        gt_visibilities = gt_visibilities[target_indices]
        # create class0 mask to be used in loss computation of bboxes
        # and keypoints
        other_class_indices = (gt_classes > 0).float().nonzero().view(-1)
        # compute metrics
        class_accuracy_w0 = self.accuracy(pred_classes.argmax(dim=1), gt_classes)
        class_accuracy_wo0 = self.accuracy(pred_classes[other_class_indices].argmax(dim=1), gt_classes[other_class_indices])
        bbox_giou_w0 = generalized_box_iou_loss(pred_bboxes, gt_bboxes, reduction='none')
        bbox_giou_wo0 = generalized_box_iou_loss(pred_bboxes[other_class_indices], gt_bboxes[other_class_indices], reduction='none')
        bbox_iou_w0 = self.point_criterion(pred_bboxes, gt_bboxes).sum(dim=1)
        bbox_iou_wo0 = self.point_criterion(pred_bboxes[other_class_indices], gt_bboxes[other_class_indices]).sum(dim=1)
        self.log('class_accuracy_w0', class_accuracy_w0.sum())
        self.log('class_accuracy_wo0', class_accuracy_wo0.sum())
        self.log('bbox_giou_w0', bbox_giou_w0.sum())
        self.log('bbox_giou_wo0', bbox_giou_wo0.sum())
        self.log('bbox_iou_w0', bbox_iou_w0.sum())
        self.log('bbox_iou_wo0', bbox_iou_wo0.sum())
        result_dict = {
            'class_accuracy_w0': class_accuracy_w0.sum(),
            'class_accuracy_wo0': class_accuracy_wo0.sum(),
            'bbox_giou_w0': bbox_giou_w0.sum(),
            'bbox_giou_wo0': bbox_giou_wo0.sum(),
            'bbox_iou_w0': bbox_iou_w0.sum(),
            'bbox_iou_wo0': bbox_iou_wo0.sum(),
        }
        return result_dict


if __name__ == '__main__':
    import torch
    batch_dict = torch.load('batch.pt')
    batch = (
        batch_dict['images'],
        batch_dict['classes'],
        batch_dict['bboxes'],
        batch_dict['keypoints'],
        batch_dict['visibilities'],
    )
    pl_model = TransformerModelPL()
    pl_model.training_step(batch, 0)
