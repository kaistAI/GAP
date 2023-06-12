
from pytorch_lightning import Callback
from utils import DIALOG_DATASETS, CLASSIFICATION_DATASETS
import logging


class MetricManager(Callback):

    def __init__(self):
        self.early_stopping_count = 0
        self.max_dialog_f1 = 0
        self.max_classification_acc = 0

    def average_values(self, dict, dataset_list, metric):
        values = []
        for k, v in dict.items():
            if any(d in k.lower() for d in dataset_list):
                if metric in k.lower():
                    values.append(v)
        return sum(values) / len(values)

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics  # access it here
        elogs = {k: v.item() for k, v in elogs.items()}
        dialog_loss = self.average_values(elogs, DIALOG_DATASETS, 'loss')
        dialog_f1 = self.average_values(elogs, DIALOG_DATASETS, 'f1')
        module.log('average/dialog_loss', dialog_loss,
                   logger=True, sync_dist=True)
        module.log('average/dialog_f1', dialog_f1, logger=True, sync_dist=True)

        classification_acc = self.average_values(
            elogs, CLASSIFICATION_DATASETS, 'acc')
        module.log('average/classification_acc',
                   classification_acc, logger=True, sync_dist=True)

        if module.current_epoch >= 5:
            if dialog_f1 < self.max_dialog_f1 and classification_acc < self.max_classification_acc:
                self.early_stopping_count += 1
                logging.info('Early Stopping Patience -1')

            if self.early_stopping_count >= 2:
                trainer.should_stop = True
                logging.info('Early Stopping as criteria was met')

        self.max_dialog_f1 = max(dialog_f1, self.max_dialog_f1)
        self.max_classification_acc = max(
            classification_acc, self.max_classification_acc)
