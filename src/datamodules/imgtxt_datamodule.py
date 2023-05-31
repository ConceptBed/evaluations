from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from PIL import Image

import logging
import os
import clip

from src.datamodules.components.transforms import (
    GreyToColor,
    IdentityTransform,
    ToGrayScale,
    LaplacianOfGaussianFiltering,
)


class DomainImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
    ):

        logging.warning("Right now `ImageDataset` only supports single domain problem type.")
        self.files = os.listdir(data_dir)
        self.data_dir = data_dir

        _, self.preprocess = clip.load("ViT-B/32", device='cuda')


        logging.info(f"Total dataset size is: {len(self.files)}.")

    def __getitem__(self, index):
        path = self.files[index]
        img = Image.open(os.path.join(self.data_dir, path))
        text = ' '.join(path.split(".")[0].split("_")[:-1])

        return self.preprocess(img).unsqueeze(0), clip.tokenize([text])


    def __len__(self):
        return len(self.files)


class IMGTXTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/data/matt/learning-to-learn-concepts/outputs/composition/CUBs/bird",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

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
        if not self.data_test:
            self.data_test = DomainImageDataset(
                self.hparams.data_dir,
            )

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