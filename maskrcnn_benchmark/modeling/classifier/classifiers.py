# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .ODIR_classifier import GeneralizedClassifier


_CLASSIFICATION_META_ARCHITECTURES = {"GeneralizedClassifier": GeneralizedClassifier}


def build_classification_model(cfg):
    meta_arch = _CLASSIFICATION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
