from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers import ViltProcessor
import nonechucks as nc
from PIL import Image

import logging
import json
import os
import clip

from src.datamodules.components.transforms import (
    GreyToColor,
    IdentityTransform,
    ToGrayScale,
    LaplacianOfGaussianFiltering,
)


class ImagenetImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        json_dir: str,
        classes: list,
        categories: list,
    ):

        self.samples = []
        self.targets = []
        self.classes = classes
        self.syn_to_class = {k: en for en, k in enumerate(classes)}

        cap2cat = {}
        with open(
            f"{json_dir.replace('proposed', '')}/captions_category_gpt3_raw.json",
            "r",
        ) as h:
            cap_cat = json.load(h)
        for d_ in cap_cat:
            cap2cat[d_["sentence"]] = []
            for k in ["attribute", "relation", "counting", "action"]:
                if k.lower() in d_["gpt3_response"].lower():
                    cap2cat[d_["sentence"]].append(k.lower())

        self.data = []
        ext = "jpeg"
        for jname in os.listdir(json_dir):
            with open(os.path.join(json_dir, jname), "r") as h:
                json_data = json.load(h)
            data_path = os.path.join(os.path.join(data_dir, jname.split(".")[0]))
            if not os.path.exists(data_path):  ## TODO: missing images
                continue
            data_files = os.listdir(data_path)
            for d_ in json_data:
                if f"{d_['index']}_0.{ext}" not in data_files: ## TODO: remove the assumption that there are only one instance per prompt
                    continue
                flag_test = False
                for k in cap2cat[d_["caption"]]:
                    if k not in categories:
                        continue
                    flag_test = True
                if not flag_test:
                    continue
                for ques in d_["question"]:
                    self.data.append(
                        (os.path.join(data_path, f"{d_['index']}_0.{ext}"), ques)
                    )

        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

        logging.info(f"Total dataset size is: {len(self.data)}.")

    def __getitem__(self, index):
        path, question = self.data[index]
        img = Image.open(path)
        encoding = self.processor(img, question, return_tensors="pt")

        return encoding, torch.tensor(3)

    def __len__(self):
        return len(self.data)


class VQADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/data/matt/learning-to-learn-concepts/outputs/composition/CUBs/bird",
        json_dir: str = "/data/data/matt/learning-to-learn-concepts/concept_bed/dataset/proposed",
        categories: list = ["attribute", "relation", "counting", "action"],
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_test: Optional[Dataset] = None

        self.selected_classes = sorted(
            os.listdir(
                data_dir
            )
        )
        # self.all_classes = sorted(
        #     os.listdir(
        #         "/data/data/matt/datasets/tiered-imagenet-tools/ILSVRC/Data/CLS-LOC/train"
        #     )
        # )
        # self.removed_classes = sorted(
        #     list(set(self.all_classes) - set(self.selected_classes))
        # )

    @property
    def num_classes(self) -> int:
        return self.hparams.num_cls

    def prepare_data(self):
        return

    def setup(self, stage: Optional[str] = None):
        if not self.data_test:
            self.data_test = ImagenetImageDataset(
                self.hparams.data_dir,
                self.hparams.json_dir,
                self.selected_classes,
                self.hparams.categories,
            )

            self.data_test = nc.SafeDataset(self.data_test)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
