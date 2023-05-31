import subprocess
from pathlib import Path
from typing import List
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

import os
import pickle
import logging


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class SaveLogits(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, save_dir):
        self.logits = []
        self.y_true = []
        self.save_dir = save_dir
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.logits += list(outputs["logits"].cpu().numpy())
            self.y_true += list(batch[1].cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate mean/std of entropy."""
        if self.ready:
            self.logits.clear()
            self.y_true.clear()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.logits += list(outputs["logits"].cpu().numpy())
            self.y_true += list(batch[1].cpu().numpy())

    def on_test_end(self, trainer, pl_module):
        """Generate mean/std of entropy."""
        if self.ready:
            with open(os.path.join(self.save_dir, "test_logits.pkl"), "wb") as handle:
                pickle.dump(self.logits, handle)
            with open(os.path.join(self.save_dir, "test_y_true.pkl"), "wb") as handle:
                pickle.dump(self.y_true, handle)

            logging.info(f"Eval data is stored at: {self.save_dir}")

            self.logits.clear()
            self.y_true.clear()


class MeanStdEntropy(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self):
        self.entropy = []
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def get_entropy(self, outputs, dim=1):
        outputs = torch.nn.functional.softmax(outputs, dim=dim)
        en = (torch.log(outputs) * outputs).sum(dim=dim) * -1
        return en

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.entropy += list(
                self.get_entropy(outputs["logits"], dim=1).cpu().numpy()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate mean/std of entropy."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            mean = np.mean(self.entropy)
            var = np.var(self.entropy)

            experiment.log({f"entropy-mean/val": mean})
            experiment.log({f"entropy-var/val": var})

            self.entropy.clear()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.entropy += list(
                self.get_entropy(outputs["logits"], dim=1).cpu().numpy()
            )

    def on_test_end(self, trainer, pl_module):
        """Generate mean/std of entropy."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            mean = np.mean(self.entropy)
            var = np.var(self.entropy)

            experiment.log({f"entropy-mean/test": mean})
            experiment.log({f"entropy-var/test": var})

            self.entropy.clear()
