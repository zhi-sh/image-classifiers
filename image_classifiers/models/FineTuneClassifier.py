# -*- coding: utf-8 -*-
# @DateTime :2021/3/15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from torch import nn


class FineTuneClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, output_size: int = None):
        super(FineTuneClassifier, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.dimension, out_features=output_size)

    def forward(self, x):
        feature = self.base_model(x)
        out = self.fc(feature)
        return out

    def save(self, output_path: str):
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def recover_model(self, input_path: str):
        self.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location='cpu'))
