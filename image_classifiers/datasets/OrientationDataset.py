# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 上午9:55
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset
from image_classifiers.tools import tools


class OrientationDataset(Dataset):
    def __init__(self, img_df: pd.DataFrame, transform: transforms = None):
        self.img_df = img_df
        self.transform = transform

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, index):
        img = Image.open(self.img_df.iloc[index]['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor([self.img_df.iloc[index]['label']]).long()

    @classmethod
    def load_dataset_from_path(cls, path: str, dev_size: int = 0):
        r'''
        从目录加载数据, 目录结构为 Path/class_no./xxx.jpg
        :param path: 图像类别所在主目录
        :param dev_size: 开发集比例
        :return: le, train_dataset, dev_dataset
        '''

        assert os.path.isdir(path)
        image_paths = tools.gather_files_by_ext(path, ext='.png')
        df = pd.DataFrame({'path': image_paths})
        df['label'] = df['path'].apply(lambda x: x.split(r'/')[-2])
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        if 0 < dev_size < 1:
            train, dev = train_test_split(df, test_size=dev_size, stratify=df['label'])
            return le, train, dev
        else:
            return le, df, None
