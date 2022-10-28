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

__all__ = ["DISHead", "build_dis_head", "ROI_DIS_HEAD_REGISTRY"]

ROI_DIS_HEAD_REGISTRY = Registry("ROI_DIS_HEAD")
ROI_DIS_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

from lib.bivariate_Laplace_distribution import(
    bivariate_Laplace_loss,
    bivariate_Laplace_cov,
)

def build_dis_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIS_HEAD.NAME
    return ROI_DIS_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_DIS_HEAD_REGISTRY.register()
class DISHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", \
        smooth_l1_beta=0, num_classes=1, num_regions=1,
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
        self.num_classes = num_classes
        self.num_regions = num_regions
        
        self.H_layer = Linear(fc_dim_final, self.num_classes)
        nn.init.normal_(self.H_layer.weight, std=0.001)
        nn.init.constant_(self.H_layer.bias, 0)

        self.scale = 700.0
        self.hrec_layer = Linear(fc_dim_final, self.num_classes)
        nn.init.normal_(self.hrec_layer.weight, std=0.001)
        nn.init.constant_(self.hrec_layer.bias, 0)

        self.cov_layer = Linear(fc_dim_final, self.num_classes * 3)
        nn.init.normal_(self.cov_layer.weight, std=0.001)
        nn.init.constant_(self.cov_layer.bias, 0)
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_DIS_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_DIS_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_DIS_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_DIS_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_DIS_HEAD.NORM,
            "smooth_l1_beta": cfg.MODEL.ROI_DIS_HEAD.SMOOTH_L1_BETA,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "num_regions": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE * cfg.SOLVER.IMS_PER_BATCH,
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
        pred_Hs = self.H_layer(x)
        pred_hrecs = self.hrec_layer(x)
        pred_covs = self.cov_layer(x)
        if self.training:
            return {
                "loss_dis": self.dis_rcnn_loss(pred_Hs, pred_hrecs, pred_covs, instances),
            }
        else:
            self.dis_rcnn_inference(pred_Hs, pred_hrecs, pred_covs, instances)
            return instances

    def dis_rcnn_loss(self, pred_Hs, pred_hrecs, pred_covs, instances):
        gt_dims = []
        gt_proj_hs = []
        gt_classes = []
        for instances_per_image in instances:
            gt_dims.append(instances_per_image.gt_dims)
            gt_proj_hs.append(instances_per_image.gt_proj_hs)
            gt_classes.append(instances_per_image.gt_classes)
        gt_dims = cat(gt_dims, dim=0)
        gt_Hs = gt_dims[:, 0:1]
        gt_proj_hs = cat(gt_proj_hs, dim=0).unsqueeze(-1)
        gt_hrecs = self.scale / gt_proj_hs
        gt_classes = cat(gt_classes, dim=0)

        pred_Hs_trans = torch.cuda.FloatTensor(gt_Hs.shape[0], 1)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_Hs_trans[index, :] = pred_Hs[index, i:i + 1]

        pred_hrecs_trans = torch.cuda.FloatTensor(gt_Hs.shape[0], 1)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_hrecs_trans[index, :] = pred_hrecs[index, i:i + 1]

        pred_covs_trans = torch.cuda.FloatTensor(gt_Hs.shape[0], 3)
        for i in range(self.num_classes):
            index = gt_classes == i
            pred_covs_trans[index, :] = pred_covs[index, 3 * i:3 * (i + 1)]

        loss_dis = bivariate_Laplace_loss(
            pred_Hs_trans,
            pred_hrecs_trans,
            pred_covs_trans,
            gt_Hs,
            gt_hrecs,
        )
        return loss_dis.sum() / self.num_regions

    def dis_rcnn_inference(self, pred_Hs, pred_hrecs, pred_covs, pred_instances):
        num_instances_per_image = [len(i) for i in pred_instances]
        pred_Hs = pred_Hs.split(num_instances_per_image)
        pred_hrecs = pred_hrecs.split(num_instances_per_image)
        pred_covs = pred_covs.split(num_instances_per_image)

        for Hs_per_image, hrecs_per_image, covs_per_image, instances_per_image in zip(pred_Hs, pred_hrecs, pred_covs, pred_instances):
            classes_per_image = instances_per_image.pred_classes
            pred_Hs_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], 1)
            pred_hrecs_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], 1)
            pred_covs_per_image = torch.cuda.FloatTensor(classes_per_image.shape[0], 3)
            for i in range(self.num_classes):
                index = classes_per_image == i
                pred_Hs_per_image[index, :] = Hs_per_image[index, i:i + 1]
                pred_hrecs_per_image[index, :] = hrecs_per_image[index, i:i + 1]
                pred_covs_per_image[index, :] = covs_per_image[index, 3 * i:3 * (i + 1)]
                
            instances_per_image.pred_Hs = pred_Hs_per_image
            instances_per_image.pred_hrecs = pred_hrecs_per_image / self.scale
            pred_sigmas_per_image = bivariate_Laplace_cov(pred_covs_per_image)
            instances_per_image.pred_hrec_uncers = (0.5 * pred_sigmas_per_image[:, 1, 1])**0.5 / self.scale
            
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