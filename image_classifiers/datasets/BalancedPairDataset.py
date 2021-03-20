# -*- coding: utf-8 -*-
# @DateTime :2021/3/12
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import copy
import random
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset


class BalancedPairDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, single_size=500, need_mean_picture=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.single_size = single_size

        self.label2path_dict = self._label2path_dict()
        self.data = self._balanced_sample()
        if need_mean_picture:
            self.generate_mean_picture()

    def __getitem__(self, index):
        img_paths, target = self.data[index]
        img_array = []
        for img_path in img_paths:
            img = Image.open(img_path).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            img_array.append(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_array, torch.tensor(target)

    def generate_mean_picture(self):
        pic_root = os.path.split(self.root)[0]
        pic = {}
        for cate, paths in self.label2path_dict.items():
            for path in paths:
                img = Image.open(path).convert('L')
                img = torch.from_numpy(np.asarray(img, dtype=float))
                pic.setdefault(cate, []).append(img)

        for cate, image_array in pic.items():
            img_tensor = torch.stack(image_array, dim=0)
            mean_tensor = torch.mean(img_tensor, dim=0)
            mean_image = Image.fromarray(mean_tensor.numpy().astype('uint8'))
            mean_image.save(f'{pic_root}/{cate}.png')

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
                if img_path.endswith('.png'):
                    ret.setdefault(cat, []).append(img_path)

        return ret

    def _balanced_sample(self):
        data = []
        categories = list(self.label2path_dict.keys())
        for pos in categories:
            cans = copy.deepcopy(categories)
            cans.remove(pos)
            for j in range(min(self.single_size, len(self.label2path_dict[pos]))):
                neg = random.choice(cans)
                anchor = random.choice(self.label2path_dict[pos])
                pos_item = random.choice(self.label2path_dict[pos])
                neg_item = random.choice(self.label2path_dict[neg])
                data.append([(anchor, pos_item, neg_item), (1, 0)])
        return data
