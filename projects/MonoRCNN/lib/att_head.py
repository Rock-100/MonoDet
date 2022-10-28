# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import math
import numpy as np

from fvcore.nn import smooth_l1_loss
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm, cat
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

__all__ = ["ATTHead", "build_att_head", "ROI_ATT_HEAD_REGISTRY"]

ROI_ATT_HEAD_REGISTRY = Registry("ROI_ATT_HEAD")
ROI_ATT_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

def build_att_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_ATT_HEAD.NAME
    return ROI_ATT_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_ATT_HEAD_REGISTRY.register()
class ATTHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", \
        smooth_l1_beta=0, num_classes=1, num_regions=1, \
        kpt_loss_weight=1.0, \
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        fc_dim_final = 0
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim
            fc_dim_final = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        self.smooth_l1_beta = smooth_l1_beta
        self.kpt_loss_weight = kpt_loss_weight

        self.num_classes = num_classes
        self.num_regions = num_regions

        self.dim_layer = Linear(fc_dim_final, self.num_classes * (3 + 1))
        nn.init.normal_(self.dim_layer.weight, std=0.001)
        nn.init.constant_(self.dim_layer.bias, 0)

        self.yaw_layer = Linear(fc_dim_final, self.num_classes * (2 + 1))
        nn.init.normal_(self.yaw_layer.weight, std=0.001)
        nn.init.constant_(self.yaw_layer.bias, 0)

        self.num_kpts = 9
        self.kpt_layer = Linear(fc_dim_final, self.num_classes * (2 * self.num_kpts + 1))
        nn.init.normal_(self.kpt_layer.weight, std=0.001)
        nn.init.constant_(self.kpt_layer.bias, 0)

        
    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_ATT_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_ATT_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_ATT_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_ATT_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_ATT_HEAD.NORM,
            "smooth_l1_beta": cfg.MODEL.ROI_ATT_HEAD.SMOOTH_L1_BETA,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_regions": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE * cfg.SOLVER.IMS_PER_BATCH,
            "kpt_loss_weight": cfg.MODEL.ROI_ATT_HEAD.KPT_LOSS_WEIGHT,
        }

    def layers(self, x):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        pred_dims = self.dim_layer(x)
        pred_yaws = self.yaw_layer(x)
        pred_kpts = self.kpt_layer(x)
        if self.training:
            loss_cen, loss_kpt = self.kpt_rcnn_loss(pred_kpts, instances)
            return {
                "loss_dim": self.dim_rcnn_loss(pred_dims, instances),
                "loss_yaw": self.yaw_rcnn_loss(pred_yaws, instances),
                "loss_cen": loss_cen,
                "loss_kpt": loss_kpt * self.kpt_loss_weight,
            }
        else:
            self.dim_rcnn_inference(pred_dims, instances)
            self.yaw_rcnn_inference(pred_yaws, instances)
            self.kpt_rcnn_inference(pred_kpts, instances)
            return instances

    def dim_rcnn_loss(self, pred_dims, instances):
        gt_dims = []
        gt_classes = []
        for instances_per_image in instances:
            gt_dims.append(instances_per_image.gt_dims)
            gt_classes.append(instances_per_image.gt_classes)
        gt_dims = cat(gt_dims, dim=0)
        gt_classes = cat(gt_classes, dim=0)

        pred_dims_trans = torch.cuda.FloatTensor(gt_dims.shape[0], 4)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_dims_trans[index, :] = pred_dims[index, 4 * i:4 * (i + 1)]
        pred_dims_trans_uncer = pred_dims_trans[:, 3:]
        pred_dims_trans = pred_dims_trans[:, :3]

        loss_dim = smooth_l1_loss(
            pred_dims_trans,
            gt_dims,
            self.smooth_l1_beta,
            reduction="none",
        )
        loss_dim = loss_dim * ((-pred_dims_trans_uncer).exp())
        return (loss_dim.sum() + pred_dims_trans_uncer.sum()) / self.num_regions

    def dim_rcnn_inference(self, pred_dims, pred_instances):
        num_instances_per_image = [len(i) for i in pred_instances]
        pred_dims = pred_dims.split(num_instances_per_image)
        for dims_per_image, instances_per_image in zip(pred_dims, pred_instances):
            classes_per_image = instances_per_image.pred_classes
            pred_dims_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], 4)
            for i in range(self.num_classes):
                index = classes_per_image == i
                pred_dims_per_image[index, :] = dims_per_image[index, 4 * i:4 * (i + 1)]
            instances_per_image.pred_dims = pred_dims_per_image[:, :3]

    def yaw_rcnn_loss(self, pred_yaws, instances):
        gt_yaws = []
        gt_classes = []
        for instances_per_image in instances:
            gt_yaws.append(instances_per_image.gt_alphas)
            gt_classes.append(instances_per_image.gt_classes)
        gt_classes = cat(gt_classes, dim=0)
        gt_yaws = cat(gt_yaws, dim=0)
        gt_yaws_trans = torch.cuda.FloatTensor(gt_yaws.shape[0], 2)
        gt_yaws_trans[:, 0] = torch.sin(gt_yaws)
        gt_yaws_trans[:, 1] = torch.cos(gt_yaws)

        pred_yaws_trans = torch.cuda.FloatTensor(gt_yaws_trans.shape[0], 3)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_yaws_trans[index, :] = pred_yaws[index, 3 * i:3 * (i + 1)]
        pred_yaws_trans_uncer = pred_yaws_trans[:, 2:]
        pred_yaws_trans = pred_yaws_trans[:, :2]

        loss_yaw = smooth_l1_loss(
            pred_yaws_trans,
            gt_yaws_trans,
            self.smooth_l1_beta,
            reduction="none",
        )
        loss_yaw = loss_yaw * ((-pred_yaws_trans_uncer).exp())
        return (loss_yaw.sum() + pred_yaws_trans_uncer.sum()) / self.num_regions

    def yaw_rcnn_inference(self, pred_yaws, pred_instances):
        num_instances_per_image = [len(i) for i in pred_instances]
        pred_yaws = pred_yaws.split(num_instances_per_image)
        for yaws_per_image, instances_per_image in zip(pred_yaws, pred_instances):
            classes_per_image = instances_per_image.pred_classes
            pred_yaws_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], 3)
            for i in range(self.num_classes):
                index = classes_per_image == i
                pred_yaws_per_image[index, :] = yaws_per_image[index, 3 * i:3 * (i + 1)]
            instances_per_image.pred_yaws = torch.atan2(pred_yaws_per_image[:, 0:1], pred_yaws_per_image[:, 1:2])

    def convert_kpts(self, gt_kpts, proposal_boxes):
        gt_kpts_trans = torch.cuda.FloatTensor(gt_kpts.shape[0], self.num_kpts * 2)
        for i in range(gt_kpts.shape[0]):
            sx = proposal_boxes[i][0]
            sy = proposal_boxes[i][1]
            w = proposal_boxes[i][2] - proposal_boxes[i][0]
            h = proposal_boxes[i][3] - proposal_boxes[i][1]
            gt_kpts_trans[i, 0::2] = (gt_kpts[i, :, 0] - sx) / w
            gt_kpts_trans[i, 1::2] = (gt_kpts[i, :, 1] - sy) / h
        return gt_kpts_trans

    def convert_kpts_inv(self, pred_kpts, proposal_boxes):
        pred_kpts_inv = torch.cuda.FloatTensor(pred_kpts.shape[0], self.num_kpts, 2)
        for i in range(pred_kpts.shape[0]):
            sx = proposal_boxes[i][0]
            sy = proposal_boxes[i][1]
            w = proposal_boxes[i][2] - proposal_boxes[i][0]
            h = proposal_boxes[i][3] - proposal_boxes[i][1]
            pred_kpts_inv[i, :, 0] = pred_kpts[i, 0::2] * w + sx
            pred_kpts_inv[i, :, 1] = pred_kpts[i, 1::2] * h + sy
        return pred_kpts_inv

    def kpt_rcnn_loss(self, pred_kpts, instances):
        gt_kpts = []
        gt_classes = []
        proposal_boxes = []
        for instances_per_image in instances:
            gt_kpts.append(instances_per_image.gt_proj_kpts)
            gt_classes.append(instances_per_image.gt_classes)
            proposal_boxes.append(instances_per_image.proposal_boxes.tensor)
        gt_kpts = cat(gt_kpts, dim=0)
        gt_classes = cat(gt_classes, dim=0)
        proposal_boxes = cat(proposal_boxes, dim=0)
        gt_kpts_trans = self.convert_kpts(gt_kpts, proposal_boxes)

        pred_kpts_trans = torch.cuda.FloatTensor(pred_kpts.shape[0], self.num_kpts * 2 + 1)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_kpts_trans[index, :] = pred_kpts[index, (self.num_kpts * 2 + 1) * i:(self.num_kpts * 2 + 1) * (i + 1)]
        pred_cens_trans_uncer = pred_kpts_trans[:, self.num_kpts * 2:]
        pred_kpts_trans = pred_kpts_trans[:, :self.num_kpts * 2]
        
        loss_kpt = smooth_l1_loss(
            pred_kpts_trans[:, :16],
            gt_kpts_trans[:, :16],
            self.smooth_l1_beta,
            reduction="none",
        )
        loss_cen = smooth_l1_loss(
            pred_kpts_trans[:, 16:],
            gt_kpts_trans[:, 16:],
            self.smooth_l1_beta,
            reduction="none",
        )
        loss_cen = loss_cen * ((-pred_cens_trans_uncer).exp())
        return (loss_cen.sum() + pred_cens_trans_uncer.sum()) / self.num_regions, loss_kpt.sum() / (self.num_regions * self.num_kpts)

    def kpt_rcnn_inference(self, pred_kpts, pred_instances):
        num_instances_per_image = [len(i) for i in pred_instances]
        pred_kpts = pred_kpts.split(num_instances_per_image)
        for kpts_per_image, instances_per_image in zip(pred_kpts, pred_instances):
            classes_per_image = instances_per_image.pred_classes
            pred_kpts_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], self.num_kpts * 2 + 1)
            for i in range(self.num_classes):
                index = classes_per_image == i
                pred_kpts_per_image[index, :] = kpts_per_image[index, (self.num_kpts * 2 + 1) * i:(self.num_kpts * 2 + 1) * (i + 1)]
            instances_per_image.pred_proj_kpts = self.convert_kpts_inv(pred_kpts_per_image[:, :self.num_kpts * 2], instances_per_image.pred_boxes.tensor)


    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])