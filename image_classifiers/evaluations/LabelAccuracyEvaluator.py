# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 下午4:01
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import csv, logging, os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from image_classifiers.evaluations import AbstractEvaluation

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(AbstractEvaluation):
    def __init__(self, dataloader: DataLoader, name: str = ''):
        self.dataloader = dataloader
        self.name = name
        self.loss_fct = nn.CrossEntropyLoss()

        self.csv_file = f'accuracy_evaluation_{name}_results.csv'
        self.csv_headers = ['epoch', 'steps', 'accuracy', 'loss']

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps=-1) -> float:
        model.eval()
        total = 0
        correct = 0

        # 构造日志输出格式
        if epoch == -1:
            if steps == -1:
                out_txt = "after epoch {}:".format(epoch)
            else:
                out_txt = "in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ':'
        logger.info(f"evaluation on the {self.name} dataset {out_txt}")

        # 前向过程
        losses = []
        for step, batch in enumerate(tqdm(self.dataloader, desc='Evaluating')):
            features, label_ids = batch
            features = features.to(model.device)
            label_ids = label_ids.to(model.device)

            with torch.no_grad():
                prediction = model(features)
                loss = self.loss_fct(prediction, label_ids.view(-1))
                losses.append(loss.item())

                total += prediction.size(0)
                correct += torch.argmax(prediction, dim=1).eq(label_ids.view(-1)).sum().item()  # eq(a, b) 应确保 a, b 维度一致

        accuracy = correct / total
        eval_loss = sum(losses) / len(losses)
        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        # 评估结果保存至文件
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode='w', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, eval_loss])
            else:
                with open(csv_path, mode='a', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow([epoch, steps, accuracy, eval_loss])
        print("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        return accuracy, eval_loss
