# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..classifier_head.classifier_head import build_classifier_head


class GeneralizedClassifier(nn.Module):

    def __init__(self, cfg):
        super(GeneralizedClassifier, self).__init__()

        self.backbone = build_backbone(cfg)
        self.classifier_head = build_classifier_head(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, classifier_losses = self.classifier_head(features, targets)

        if self.training:
            losses = {}
            losses.update(classifier_losses)
            return losses

        return result
