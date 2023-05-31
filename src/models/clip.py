from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

import clip

import sklearn
from packaging import version
import numpy as np

import logging


class CLIPModel:
    def __init__(self, model="ViT-B/32", is_cuda=True):        
        self.device="cpu"
        if torch.cuda.is_available() and is_cuda:
            self.device="cuda"
        elif is_cuda:
            logging.warn("Attempted to use device \{cuda\} but it is not available.")

        
        

    def inference(self, image, texts):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        texts = clip.tokenize(texts).to(self.device)
        image_feats = self.model.encode_image(image).detach().cpu().numpy()
        text_feats = self.model.encode_text(texts).detach().cpu().numpy()


        ## This snippet is taken from reference free clip score paper.
        if version.parse(np.__version__) < version.parse('1.21'):
            images = sklearn.preprocessing.normalize(image_feats, axis=1)
            candidates = sklearn.preprocessing.normalize(text_feats, axis=1)
        else:
            # logging.warn(
            #     'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            #     'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
            images = image_feats / np.sqrt(np.sum(image_feats**2, axis=1, keepdims=True))
            candidates = text_feats / np.sqrt(np.sum(text_feats**2, axis=1, keepdims=True))
        
        w=2.5
        per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)

        # logging.debug(f"Probability is: {per}")
        return per

class CLIPModule(LightningModule):
    def __init__(
        self,
        device="cuda"
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.test_score = []

        self.clipmodel, _ = clip.load("ViT-B/32", device=device)

    def forward(self, batch: Any):
        image = batch[0]#self.preprocess(image).unsqueeze(0).to(self.device)
        texts = batch[1]#clip.tokenize(texts).to(self.device)
        image_feats = self.clipmodel.encode_image(image[0]).detach().cpu().numpy()
        text_feats = self.clipmodel.encode_text(texts[0]).detach().cpu().numpy()

        if version.parse(np.__version__) < version.parse('1.21'):
            images = sklearn.preprocessing.normalize(image_feats, axis=1)
            candidates = sklearn.preprocessing.normalize(text_feats, axis=1)
        else:
            images = image_feats / np.sqrt(np.sum(image_feats**2, axis=1, keepdims=True))
            candidates = text_feats / np.sqrt(np.sum(text_feats**2, axis=1, keepdims=True))
        
        w=2.5
        per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
        return per

    def on_train_start(self):
        pass

    def step(self, batch: Any):
        score = self.forward(batch)
        return score

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        score = self.step(batch)
        self.test_score.append(score)
        return {"score": score}

    def test_epoch_end(self, outputs: List[Any]):
        logging.info(f"The mean CLIPScore is: {np.mean(self.test_score)}")

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        pass
