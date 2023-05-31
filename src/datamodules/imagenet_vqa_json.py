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
        transform=None,
    ):

        self.samples = []
        self.targets = []
        self.classes = classes
        self.transform = transform
        self.syn_to_class = {k: en for en, k in enumerate(classes)}

        ldm = False
        ext = "jpeg"
        if "ldm" in data_dir:
            ldm = True
            ext = "jpg"

        cap2cat = {}
        with open(
            "/data/data/matt/learning-to-learn-concepts/concept_bed/dataset/imagenet/captions_category_gpt3_raw.json",
            "r",
        ) as h:
            cap_cat = json.load(h)
        for d_ in cap_cat:
            cap2cat[d_["sentence"]] = []
            for k in ["attribute", "relation", "counting", "action"]:
                if k.lower() in d_["gpt3_response"].lower():
                    cap2cat[d_["sentence"]].append(k.lower())

        self.data = []
        total_removed = 0
        for jname in os.listdir(json_dir):
            with open(os.path.join(json_dir, jname), "r") as h:
                json_data = json.load(h)
            data_path = os.path.join(os.path.join(data_dir, jname.split(".")[0]))
            if ldm:
                data_path = os.path.join(data_path, "samples")
            if not os.path.exists(data_path):  ## TODO: missing images
                continue
            data_files = os.listdir(data_path)
            for d_ in json_data:
                if f"{d_['index']}_0.{ext}" not in data_files:
                    continue
                # flag_test = True
                # for k in cap2cat[d_["caption"]]:
                #     if k == "attribute":
                #         flag_test = False
                #         break

                # if flag_test or len(cap2cat[d_["caption"]]) != 1:
                #     total_removed += 1
                #     continue
                self.data.append(
                    (
                        os.path.join(data_path, f"{d_['index']}_0.{ext}"),
                        self.syn_to_class[jname.split(".")[0]],
                    )
                )

        logging.info(f"Total dataset size is: {len(self.data)}.")
        logging.info(f"Total attribute related remvoed images are: {total_removed}.")

    def __getitem__(self, index):
        path, target = self.data[index]
        x = Image.open(path)
        if self.transform:
            x = self.transform(x)
        return x, target

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
                "/data/data/matt/learning-to-learn-concepts/concept_bed/dataset/imagenet/eval"
            )
        )
        self.all_classes = sorted(
            os.listdir(
                "/data/data/matt/datasets/tiered-imagenet-tools/ILSVRC/Data/CLS-LOC/train"
            )
        )
        self.removed_classes = sorted(
            list(set(self.all_classes) - set(self.selected_classes))
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.custom_test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

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
                self.custom_test_transforms,
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
