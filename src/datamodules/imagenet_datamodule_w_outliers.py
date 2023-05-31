from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision import datasets

from PIL import Image
import json

import logging
import os



class ImageNetGen(Dataset):
    def __init__(self, root, classes, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.classes = classes
        self.syn_to_class = {k: en for en, k in enumerate(classes)}

        for entry in os.listdir(root):
            assert entry in self.syn_to_class
            syn_dir = os.path.join(root, entry)
            target = self.syn_to_class[entry]
            for sample in os.listdir(syn_dir):
                self.samples.append(os.path.join(syn_dir, sample))
                self.targets.append(target)

        logging.info(f"Total EVAL selected classes: {len(self.classes)}")
        logging.info(f"Total EVAl selected images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, classes, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.classes = classes
        self.syn_to_class = {k: en for en, k in enumerate(classes)}

        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                if syn_id not in self.classes:
                    continue
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                if syn_id not in self.classes:
                    continue
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

        logging.info(f"Total selected classes: {len(self.classes)}")
        logging.info(f"Total selected images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/ImageNet",
        test_data_dir: str = "data/ImageNet",
        train_val_split: Tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_cls: int = -1,
        test_only: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.selected_classes = sorted(
            os.listdir(
                data_dir
            )
        )
        self.removed_classes = []
        # self.all_classes = sorted(
        #     os.listdir(
        #         "/data/data/matt/datasets/tiered-imagenet-tools/ILSVRC/Data/CLS-LOC/train"
        #     )
        # )
        # self.removed_classes = sorted(
        #     list(set(self.all_classes) - set(self.selected_classes))
        # )

        # data transformations
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.custom_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.custom_test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        logging.info(
            f"`n` class classification problem with: `n={len(self.selected_classes)}`"
        )

    @property
    def num_classes(self) -> int:
        return len(self.selected_classes)

    def prepare_data(self):
        """Download data if needed."""
        return

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            logging.warn("If you are attempting to train the model then make sure ImageNet dataset is available for outlier exposure.")
            
            if not self.hparams.test_only:
                self.data_train = ImageNetKaggle(
                    self.hparams.data_dir,
                    split="train",
                    classes=self.selected_classes,
                    transform=self.custom_transforms,
                )
                self.out_dataset = ImageNetKaggle(
                    self.hparams.data_dir,
                    split="train",
                    classes=self.removed_classes,
                    transform=self.custom_transforms,
                )

                self.data_val = ImageNetKaggle(
                    self.hparams.data_dir,
                    split="val",
                    classes=self.selected_classes,
                    transform=self.custom_test_transforms,
                )

            self.data_test = ImageNetGen(
                self.hparams.test_data_dir,
                classes=self.selected_classes,
                transform=self.custom_test_transforms,
            )

    def train_dataloader(self):
        in_dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        out_dataloader = DataLoader(
            dataset=self.out_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        return {"in_batch": in_dataloader, "out_batch": out_dataloader}

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
