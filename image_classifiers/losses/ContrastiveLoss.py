# -*- coding: utf-8 -*-
# @DateTime :2021/3/12
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from torch import nn
from image_classifiers import ImageClassifier


class ContrastiveLoss(nn.Module):
    r''' 适用于 Simanese Networks '''

    def __init__(self, model: ImageClassifier):
        super(ContrastiveLoss, self).__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, feature, target, need_metric=False):
        output_positive = self.model.forward_siamense(feature[:2])
        output_negative = self.model.forward_siamense(feature[0:3:2])
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])
        loss_positive = self.loss_fct(output_positive, target_positive)
        loss_negative = self.loss_fct(output_negative, target_negative)
        loss = loss_positive + loss_negative
        if need_metric:
            labels_positive = torch.sum(torch.argmax(output_positive, dim=1) == target_positive).cpu()
            labels_negative = torch.sum(torch.argmax(output_negative, dim=1) == target_negative).cpu()
            accurate_labels = labels_positive.item() + labels_negative.item()
            return loss, accurate_labels
        return loss
