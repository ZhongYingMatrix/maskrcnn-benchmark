# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

def build_classifier_head(cfg, in_channels):
    return ClassifierModule(cfg, in_channels)

class ClassifierModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ClassifierModule, self).__init__()

        self.cfg = cfg.clone()

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        # TODO
        #self.fc = nn.Linear(in_channels, cfg.MODEL.CLASSIFIER.NUM_CLASSES)
        self.fc = nn.Linear(2048, cfg.MODEL.CLASSIFIER.NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()
        if cfg.MODEL.CLASSIFIER.LOSS == "BCE":
            self.loss = nn.BCELoss()
        elif cfg.MODEL.CLASSIFIER.LOSS == "MultiLabelMarginLoss":
            self.loss = nn.MultiLabelMarginLoss()
        else:
            raise ValueError('Wrong loss type %s'%cfg.MODEL.CLASSIFIER.LOSS)
            
    def _prepare_target(self, target):
        if self.cfg.MODEL.CLASSIFIER.LOSS == "BCE":
            return target
        elif self.cfg.MODEL.CLASSIFIER.LOSS == "MultiLabelMarginLoss":
            target_idx = []
            for idx, label in enumerate(target):
                if label>0: target_idx.append(idx)
            while len(target_idx) < len(target):
                target_idx.append(-1)
            return torch.tensor(target_idx, dtype=torch.long, device=target.device)
        else:
            raise ValueError('Wrong loss type %s'%self.cfg.MODEL.CLASSIFIER.LOSS)

    def forward(self, features, targets=None):
        classifier_losses = {}
        feature = features[0]
        AvgPool = self.GlobalAvgPool(feature)
        AvgPool = AvgPool.squeeze(2).squeeze(2) 

        logits = self.fc(AvgPool)
        output = self.sigmoid(logits)
        if self.training:
            assert len(output) == len(targets), 'Length of result %d and targets %d must be equal' %(len(output), len(targets))
            loss = []
            for o, target in zip(output,targets):
                loss.append(self.loss(o, self._prepare_target(target)))
            classifier_losses['classifier_losses'] = sum(loss)/len(loss)

        return output, classifier_losses
        