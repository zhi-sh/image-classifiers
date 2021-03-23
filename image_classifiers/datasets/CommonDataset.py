# -*- coding: utf-8 -*-
# @DateTime :2021/3/12
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from PIL import Image
from typing import Union
import torch
from torch.utils.data import Dataset


class CommonDataset(Dataset):
    def __init__(self, root_or_list: Union[str, list], transform=None, target_transform=None):
        if isinstance(root_or_list, list):
            self.has_label = False
            self.data = root_or_list
        else:
            self.has_label = True
            self.root = root_or_list
            self.label2path_dict = self._label2path_dict()
            self.label2ix = {v: i for i, v in enumerate(list(self.label2path_dict.keys()))}
            self.data = self._gather_data_list()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        if self.has_label:
            img_path, target = self.data[index]
            img = Image.open(img_path).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, torch.tensor(target)
        else:
            img_path = self.data[index]
            img = Image.open(img_path).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            return img

    def __len__(self):
        return len(self.data)

    def _label2path_dict(self):
        r'''
                返回目录下 类别对应的所有文件路径列表

                :return: eg: {'left': [file1.png, file2.png], 'down': [file1.png, file2.png]}
                '''
        # 获取顶级目录下的子目录，视为类别
        categories = []
        for item in os.listdir(self.root):
            path = os.path.join(self.root, item)
            if os.path.isdir(path):
                categories.append(str(item))

        ret = {}
        # 获取每个类别下的所有图片，组成列表
        for cat in categories:
            sub_path = os.path.join(self.root, cat)
            for fn in os.listdir(sub_path):
                img_path = os.path.join(sub_path, fn)
                if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
                    ret.setdefault(cat, []).append(img_path)

        return ret

    def _gather_data_list(self):
        ret = []
        for k, v in self.label2path_dict.items():
            for e in v:
                ret.append((e, self.label2ix[k]))
        return ret
