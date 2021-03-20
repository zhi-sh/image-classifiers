# -*- coding: utf-8 -*-
# @DateTime :2021/2/22 下午3:57
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import csv, os
from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    @abstractmethod
    def __call__(self, model, output_path: str = None, epoch: int = -1):
        # return score, loss
        pass

    def _write_to_csv(self, output_path, data_list, epoch):
        # 评估结果保存至文件
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if (not os.path.isfile(csv_path)) or (epoch == 0):
                with open(csv_path, mode='w', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow(self.csv_headers)
                    writer.writerow(data_list)
            else:
                with open(csv_path, mode='a', encoding='utf-8') as fout:
                    writer = csv.writer(fout)
                    writer.writerow(data_list)
