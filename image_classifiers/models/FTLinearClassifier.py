# -*- coding: utf-8 -*-
# @DateTime :2021/3/15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from torch import nn
from image_classifiers.models import AbstractModel


class FTLinearClassifier(nn.Module, AbstractModel):
    def __init__(self, base_model: nn.Module, output_size: int = None):
        super(FTLinearClassifier, self).__init__()
        self.base_model = base_model

        # 固定参数
        for param in self.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(base_model.dimension, out_features=output_size)

        # 固定BaseModel的参数，只训练最后一层

    def forward(self, x):
        feature = self.base_model(x)
        out = self.fc(feature)
        return out

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        # 只更新需要更新的权重
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def recover_model(self, input_path: str):
        self.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location='cpu'))

    def get_config_dict(self):
        return {}

    @staticmethod
    def load(input_path):
        pass
