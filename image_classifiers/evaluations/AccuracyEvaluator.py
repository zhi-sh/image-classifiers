# -*- coding: utf-8 -*-
# @DateTime :2021/3/15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from image_classifiers.evaluations import AbstractEvaluator

logger = logging.getLogger(__name__)


class AccuracyEvaluator(AbstractEvaluator):
    def __init__(self, dataloader: DataLoader, loss_model, name: str = ''):
        self.dataloader = dataloader
        self.loss_model = loss_model
        self.name = name

        self.csv_file = f'evaluation_accuracy_{name}_results.csv'
        self.csv_headers = ['epoch', 'loss', 'accuracy']

    def __call__(self, clf, output_path: str = None, epoch: int = -1):
        logger.info(f"evaluation on the {self.name} dataset :")
        clf.model.eval()
        correct = 0
        total = len(self.dataloader.dataset)

        # 前向过程
        losses = []
        for step, batch in enumerate(tqdm(self.dataloader, desc='Evaluating')):
            features, labels = clf.batch_to_device(batch)
            with torch.no_grad():
                loss, preds = self.loss_model.forward(features, labels, need_metric=True)
                losses.append(loss.item())
                correct += preds

        accuracy = correct / total
        eval_loss = sum(losses) / total
        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        self._write_to_csv(output_path, [epoch, eval_loss, accuracy], epoch)  # format with self.csv_headers
        return accuracy, eval_loss
