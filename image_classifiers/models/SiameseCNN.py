# -*- coding: utf-8 -*-
# @DateTime :2021/3/12
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os, json
import torch
from torch import nn
from torch.nn import functional as F


class SiameseCNN(nn.Module):
    r'''孪生网络的输出一般是 2（相似，不相似）'''

    def __init__(self, input_size, feature_dim=512):
        super(SiameseCNN, self).__init__()

        self.input_size = input_size
        self.feature_dim = feature_dim
        assert input_size % 64 == 0
        self.out_channels = input_size // 64
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 2^9
        self.pool1 = nn.MaxPool2d(4, 4)  # 2^7
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)  # 2^5
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(4, 4)  # 2^3
        self.fc = nn.Linear(64 * self.out_channels * self.out_channels, feature_dim)
        self.linear = nn.Linear(512, 2)

    def forward_once(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * self.out_channels * self.out_channels)
        out = F.relu(self.fc(x))
        return out

    def forward(self, data):
        feats = []
        for i in range(2):  # Siamese nets: 同样的结构，共享权重
            x = self.forward_once(data[i])
            feats.append(x)
        out = self.linear(torch.abs(feats[1] - feats[0]))
        return out

    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'config.json'), 'w') as fout:
            # 这里需要保存的是__init__的参数列表，以便后续构造模型
            json.dump({'input_size': self.input_size, 'feature_dim': self.feature_dim}, fout)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'config.json')) as fin:
            config = json.load(fin)

        model = SiameseCNN(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model

    @property
    def dimension(self):
        return self.feature_dim
