# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 上午11:28
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, json, os

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from image_classifiers import __version__
from image_classifiers.tools import tools, utils
from image_classifiers.models import AbstractModel
from image_classifiers.evaluations import AbstractEvaluator

logger = logging.getLogger(__name__)


class ImageClassifier:
    def __init__(self, model_path: str = None, model: AbstractModel = None, device: str = None):
        # 加载已训练的模型
        if model_path is not None:
            logger.info(f'load Image Classifier model from : {model_path}')
            # 确认版本一致
            with open(os.path.join(model_path, 'config.json')) as fin:
                config = json.load(fin)
                if config['__version__'] != __version__:
                    logger.error(f'version of image classifier should be same!')

            module_class = utils.import_from_string(config['type'])
            model = module_class.load(os.path.join(model_path, config['path']))

        self.model = model

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'use pytorch device: {device}')
        self._target_device = torch.device(device)
        self.model.to(self._target_device)
        self.round_counter = 0

    # -------------------------------------- 训练方法 ------------------------------------------
    def fit(self, train_loader: DataLoader, loss_model: nn.Module, optimizer: torch.optim.Optimizer = None, evaluator: AbstractEvaluator = None, output_path: str = None, epochs: int = 10000, early_stopped_thresh=10):
        self.min_loss = 1e9
        tools.ensure_path_exist(output_path)

        loss_model = loss_model.to(self._target_device)

        if optimizer is None:
            optimizer = self.model.get_optimizer()

        score, eval_loss = self._eval_during_training(evaluator, output_path, -1)
        print(f"Random evaluation: valid loss: {eval_loss}, valid score: {score}\n")
        for epoch in range(epochs):
            losses = []

            loss_model.train()
            loss_model.zero_grad()
            progress_bar = tqdm(train_loader)
            for batch in progress_bar:
                features, labels = self.batch_to_device(batch)
                loss_value = loss_model(features, labels)
                loss_value.backward()  # 反向传播
                optimizer.step()  # 更新权重
                optimizer.zero_grad()  # 清空梯度
                losses.append(loss_value.cpu().detach().numpy())
                progress_bar.set_description(f"Epoch: {epoch} train loss: {np.sum(losses) / len(train_loader.dataset)}")

            score, eval_loss = self._eval_during_training(evaluator, output_path, epoch)
            print(f"Fine tune model at Epoch: {epoch}, train loss: {np.sum(losses) / len(train_loader.dataset):.6f}, valid loss: {eval_loss:.6f}, valid metrics: {score:.6f}\n")

            if self.round_counter > early_stopped_thresh:
                break

    def save(self, path: str):
        r'''保存模型'''
        if path is None:
            return

        tools.ensure_path_exist(path)
        logger.info(f'save model to {path}')

        model_path = os.path.join(path, f'{type(self.model).__name__}')
        base_model_config = {'path': model_path, 'type': type(self.model).__module__, '__version__': __version__}

        with open(os.path.join(path, 'config.json'), 'w') as fout:
            json.dump(base_model_config, fout, indent=2)

        self.model.save(model_path)

    def batch_to_device(self, data):
        features, labels = data
        if isinstance(features, list):  # for simanse network
            for i in range(len(features)):
                features[i] = features[i].to(self._target_device)  # CPU or GPU
        else:  # for common network
            features = features.to(self._target_device)
        labels = labels.to(self._target_device)
        return features, labels

    # --------------------------------------------- 模型属性 ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        try:
            return self.model.device
        except StopIteration:
            # TODO nn.DataParaParallel compatibility
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # --------------------------------------------- 内部函数 ----------------------------------------------------------
    def _eval_during_training(self, evaluator, output_path, epoch):
        if evaluator is not None:  # 如果存在评估实体，则进行评估模型的过程
            score, loss = evaluator(self, output_path=output_path, epoch=epoch)
            if loss <= self.min_loss:
                self.min_loss = loss
                self.round_counter = 0
                self.save(output_path)
            else:
                self.round_counter += 1
            return score, loss
