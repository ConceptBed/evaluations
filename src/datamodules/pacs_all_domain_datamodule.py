from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from PIL import Image

import logging
import os

from src.datamodules.components.transforms import (
    GreyToColor,
    IdentityTransform,
    ToGrayScale,
    LaplacianOfGaussianFiltering,
)


class DomainImageDataset(Dataset):
    def __init__(
        self, data_dir: str, domain: str, custom_transforms: transforms, test: False
    ):

        logging.warning(
            "Right now `ImageDataset` only supports single domain problem type."
        )
        self.custom_transforms = custom_transforms
        self.files = []

        domains = ['sketch', 'photo', 'art_painting', 'cartoon']
        self.mapping = {}
        for en, ac in enumerate(domains):
            self.mapping[ac] = en
        if test:
            domains = [domain]

        all_classes = sorted(os.listdir(os.path.join(data_dir, domains[0])))
        logging.info(f"Here are the sorted all classes: `{all_classes}`.")

        for domain in domains:
            for fn in all_classes:
                for ig in os.listdir(os.path.join(data_dir, domain, fn)):
                    self.files.append(
                        (os.path.join(data_dir, domain, fn, ig), self.mapping[domain])
                    )

        logging.info(f"Total dataset size is: {len(self.files)}.")

    def __getitem__(self, index):
        path, num_cls = self.files[index]
        img = self.custom_transforms(Image.open(path))

        return img, num_cls

    def __len__(self):
        return len(self.files)


class PACsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/PACs",
        test_data_dir: str = "data/PACs",
        train_val_split: Tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_cls: int = 4,
        domain: list = "sketch",
        test: bool = True,
    ):
        super().__init__()
        assert num_cls == 4

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.custom_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224), scale=(0.5, 1)
                ),  # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.custom_test_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                IdentityTransform(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_cls

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
            dataset = DomainImageDataset(
                self.hparams.data_dir,
                domain=self.hparams.domain,
                custom_transforms=self.custom_transforms,
                test=self.hparams.test,
            )

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[
                    int(self.hparams.train_val_split[0] * len(dataset)),
                    len(dataset) - int(self.hparams.train_val_split[0] * len(dataset)),
                ],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_val.custom_transforms = self.custom_test_transforms

            if self.hparams.data_dir == self.hparams.test_data_dir:
                self.data_test = self.data_val
            else:
                self.data_test = DomainImageDataset(
                    self.hparams.test_data_dir,
                    domain=self.hparams.domain,
                    custom_transforms=self.custom_test_transforms,
                    test=self.hparams.test,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

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
