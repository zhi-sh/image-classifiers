# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 下午3:07
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from torch import nn, Tensor
from image_classifiers import ImageClassifier


class SoftmaxLoss(nn.Module):
    def __init__(self, model: ImageClassifier):
        super(SoftmaxLoss, self).__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features: Tensor, labels: Tensor):
        output = self.model(features)
        loss = self.loss_fct(output, labels.view(-1))
        return loss
