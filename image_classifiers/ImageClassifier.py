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
from tqdm import tqdm
from tensorboardX import SummaryWriter
from image_classifiers import __version__
from image_classifiers.tools import tools, utils
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

        # 加载已训练的模型
        if model_path is not None:
            logger.info(f'load Image Classifier model from : {model_path}')
            # 确认版本一致
            with open(os.path.join(model_path, 'config.json')) as fin:
                config = json.load(fin)
                if config['__version__'] != __version__:
                    logger.error(f'version of image classifier should be same!')

            # 依次加载各层模型
            with open(os.path.join(model_path, 'modules.json')) as fin:
                contained_modules = json.load(fin)

            modules = OrderedDict()
            for module_config in contained_modules:
                module_class = utils.import_from_string(module_config['type'])
                module = module_class.load(os.path.join(model_path, module_config['path']))
                modules[module_config['name']] = module

        # 确保参数传递、或路径加载的modules为OrderedDict
        if (modules is not None) and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        # 使用父类，构造模型
        super().__init__(modules)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'use pytorch device: {device}')
        self._target_device = torch.device(device)
        self.round_counter = 0

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
            early_stopped_thresh: int = 20,  # 早停法阈值
            ):
        self.to(self._target_device)
        self.min_loss = 99999999  # 默认得分，用以寻找最优模型
        self.writer = None

        # 若需要，确保输出目录存在, None则跳过
        tools.ensure_path_exist(output_path)
        if output_path is not None:
            log_path = os.path.join(output_path, 'logs')
            tools.ensure_path_exist(log_path)
            self.writer = SummaryWriter(logdir=log_path)

        # 获取所有数据加载器
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # 获取所有包含模型模型损失类
        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        # 计算所有数据集，一轮训练最小的步数
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        # 不同的训练对，提供不同的优化器
        optimizers = []
        for loss_model in loss_models:
            optimizer = optimizer_class(loss_model.parameters(), **optimizer_params)
        optimizers.append(optimizer)

        # 封装各个dataloader为iter
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)

        # 构造训练过程
        self._eval_during_training(evaluator, output_path, save_best_model, -1, -1)  # 初始基准
        losses = []
        for epoch in tqdm(range(epochs), desc='Epoch'):
            training_steps = 0  # 训练统计，用以计算何时评估模型

            # 每轮先更改模型模式，并清空梯度
            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in tqdm(range(steps_per_epoch), desc='Iteration'):
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

                    # 数据
                    features, labels = data
                    if isinstance(features, list):  # for simanse network
                        for i in range(len(features)):
                            features[i] = features[i].to(self._target_device)  # CPU or GPU
                    else:  # for common network
                        features = features.to(self._target_device)
                    labels = labels.to(self._target_device)

                    loss_value = loss_model(features, labels)
                    loss_value.backward()  # 反向传播
                    optimizer.step()  # 更新权重
                    optimizer.zero_grad()  # 清空梯度
                    losses.append(loss_value.item())  # 收集训练损失

                training_steps += 1
                if (evaluation_steps > 0) and (training_steps % evaluation_steps == 0):
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            # 每轮最后评估一次模型
            score, valid_loss = self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

            # TensorBoard记录训练数据
            if self.writer is not None:
                if losses:
                    self.writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)
                if score is not None:
                    self.writer.add_scalar('valid_score', score, epoch)
                    self.writer.add_scalar('valid_loss', valid_loss, epoch)

            # 探测是否需要停止训练（早停法）
            if self.round_counter > early_stopped_thresh:
                logger.info(f'early stopped at epoch {epoch + 1} / {epochs}')
                break

        # 关闭TensorBoard
        if self.writer is not None:
            self.writer.close()

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

    def predict(self, dataloader: DataLoader):
        r'''模型推导'''
        predictions = []
        for batch in dataloader:
            features, _ = batch
            features = features.to(self.device)
            output = self(features)
            preds = torch.argmax(output, dim=1).numpy()
            predictions.extend(list(preds))
        return predictions

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
            score, loss = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if loss <= self.min_loss:
                self.min_loss = loss
                self.round_counter = 0
                if save_best_model:
                    self.save(output_path)
            else:
                self.round_counter += 1
            return score
