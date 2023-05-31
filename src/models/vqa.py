from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from transformers import ViltProcessor, ViltForQuestionAnswering
import logging


class ViLTModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.test_acc = Accuracy()

    def forward(self, x: torch.Tensor):
        return self.model(**x)

    def on_train_start(self):
        pass

    def step(self, batch: Any):
        x, y = batch
        x = {k: v[0] for k, v in x.items()}
        logits = self.forward(x).logits
        preds = torch.argmax(logits, dim=1)
        return preds, y, logits

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            preds, targets, logits = self.step(batch)
            acc = self.test_acc(preds, targets)

        return {"preds": preds, "targets": targets, "logits": logits}

    def test_epoch_end(self, outputs: List[Any]):
        logging.info(f"The test-accuracy is: {self.test_acc.compute()}")

    def on_epoch_end(self):
        self.test_acc.reset()

    def configure_optimizers(self):
        pass
