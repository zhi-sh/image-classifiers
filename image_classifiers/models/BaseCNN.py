# -*- coding: utf-8 -*-
# @DateTime :2021/3/12
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os, json
import torch
from torch import nn
from torch.nn import functional as F


class BaseCNN(nn.Module):
    def __init__(self, input_size=512, output_size=4):
        super(BaseCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        assert input_size % 64 == 0
        self.out_channels = input_size // 64
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 2^9
        self.pool1 = nn.MaxPool2d(4, 4)  # 2^7
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)  # 2^5
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(4, 4)  # 2^3
        self.fc1 = nn.Linear(64 * self.out_channels * self.out_channels, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * self.out_channels * self.out_channels)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        out = self.fc2(x)
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

        model = BaseCNN(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model