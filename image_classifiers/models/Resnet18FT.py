# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 上午11:20
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import json, os
import torch
from torch import nn
from torchvision import models


class Resnet18FT(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Resnet18FT, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        model = models.resnet18(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(input_size, output_size)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'config.json'), 'w') as fout:
            # 这里需要保存的是__init__的参数列表，以便后续构造模型
            json.dump({'input_size': self.input_size, 'output_size': self.output_size}, fout)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'config.json')) as fin:
            config = json.load(fin)

        model = Resnet18FT(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model
