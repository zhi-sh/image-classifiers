# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 上午11:28
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, json, os
from typing import Dict, Iterable, Tuple, Type
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import trange
from image_classifiers import __version__
from image_classifiers.tools import tools
from image_classifiers.evaluations import AbstractEvaluation

logger = logging.getLogger(__name__)


class ImageClassifier(nn.Sequential):
    def __init__(self, model_path: str = None, modules: Iterable[nn.Module] = None, device: str = None):
        r'''
        分类模型包装器，包装训练和预测过程
        :param model_path: 已训练的模型路径
        :param modules: 分层模型列表
        :param device: CPU or GPU, 默认自动探测
        '''
        if model_path is not None:
            raise ValueError(r'not implement!')

        if (modules is not None) and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'use pytorch device: {device}')
        self._target_device = torch.device(device)

    # -------------------------------------- 训练方法 ------------------------------------------
    def fit(self, train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: AbstractEvaluation = None,
            epochs: int = 1,
            steps_per_epoch: int = 0,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_params: Dict[str, object] = {'lr': 1e-3},
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            ):
        self.to(self._target_device)
        self.best_score = -99999999  # 默认得分，用以寻找最优模型

        # 若需要，确保输出目录存在, None则跳过
        tools.ensure_path_exist(output_path)

        # 获取所有数据加载器
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # 获取所有包含模型模型损失类
        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        # 计算所有数据集，一轮训练最小的步数
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        # 为不同的训练对，提供不同的优化器
        optimizers = []
        for loss_model in loss_models:
            optimizer = optimizer_class(loss_model.parameters(), **optimizer_params)
        optimizers.append(optimizer)

        # 封装各个dataloader为iter
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)

        # 构造训练过程
        self._eval_during_training(evaluator, output_path, save_best_model, -1, -1)  # 初始基准
        for epoch in trange(epochs, desc='Epoch'):
            training_steps = 0  # 训练统计，用以计算何时评估模型

            # 每轮先更改模型模式，并清空梯度
            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc='Iteration', smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    # 循环遍历每个数据集的dataloader
                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    features = features.to(self._target_device)
                    labels = labels.to(self._target_device)

                    loss_value = loss_model(features, labels)
                    loss_value.backward()  # 反向传播
                    optimizer.step()  # 更新权重
                    optimizer.zero_grad()  # 清空梯度

                training_steps += 1
                if (evaluation_steps > 0) and (training_steps % evaluation_steps == 0):
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            # 每轮最后评估一次模型
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def save(self, path: str):
        r'''保存模型'''
        if path is None:
            return

        tools.ensure_path_exist(path)
        logger.info(f'save model to {path}')

        contained_modules = []
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, f'{str(idx)}_{type(module).__name__}')
            tools.ensure_path_exist(model_path)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fout:
            json.dump(contained_modules, fout, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fout:
            json.dump({'__version__': __version__}, fout, indent=2)

    # --------------------------------------------- 模型属性 ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            # TODO nn.DataParaParallel compatibility
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # --------------------------------------------- 内部函数 ----------------------------------------------------------
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        if evaluator is not None:  # 如果存在评估实体，则进行评估模型的过程
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
