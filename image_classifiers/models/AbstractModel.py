# -*- coding: utf-8 -*-
# @DateTime :2021/3/18
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import json
from abc import ABC, abstractmethod

import torch


class AbstractModel(ABC):
    def save(self, output_path: str):
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'config.json'), 'w') as fout:
            # 这里需要保存的是__init__的参数列表，以便后续构造模型
            json.dump(self.get_config_dict(), fout, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    @abstractmethod
    def load(input_path):
        pass

    @abstractmethod
    def get_config_dict(self):
        pass

    @abstractmethod
    def get_optimizer(self, lr, weight_decay):
        pass
