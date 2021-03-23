# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 下午3:07
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from torch import nn, Tensor


class SoftmaxLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features: Tensor, labels: Tensor, need_metric=False):
        output = self.model(features)
        loss = self.loss_fct(output, labels.view(-1))
        if need_metric:
            pred = output.data.max(1, keepdim=True)[1]
            correct_items = pred.eq(labels.data.view_as(pred)).cpu().sum()
            return loss, correct_items.item()
        return loss
