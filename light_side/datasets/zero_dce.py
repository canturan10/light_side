import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import extract_archive

from ..core import _download_file_from_url
from .base import BaseDataset, Identity


class Zero_DCE(BaseDataset):

    __phases__ = (
        "train",
        "val",
    )

    def __init__(
        self,
        root_dir: str = None,
        phase: str = "train",
        transforms=None,
        **kwargs,
    ):
        if root_dir is None:
            root_dir = "TODO"

        self.phase = phase
        self.root_dir = root_dir
        self.transforms = Identity() if transforms is None else transforms
        self.download()

        ids, targets = self._split_dataset(phase=phase)
        super().__init__(ids, targets, transforms=transforms, **kwargs)

    def __getitem__(self, idx: int) -> Tuple:
        img = self._load_image(self.ids[idx])
        img = torch.from_numpy(img).float().permute(2, 0, 1)

        # apply transforms
        if self.transforms:
            img = self.transforms(img)

        return (img, img)

    def _split_dataset(self, phase) -> Tuple:
        filenames = []

        data_dir = os.path.join(self.root_dir)

        for item in os.listdir(data_dir):
            f = os.path.join(data_dir, item)
            if os.path.isfile(f):
                filenames.append(f)
            else:
                for subitem in os.listdir(f):
                    sub_f = os.path.join(f, subitem)
                    filenames.append(sub_f)

        filenames = np.asarray(filenames)
        filenames = filenames[filenames.argsort()]
        idxs = range(len(filenames))

        # split into a is_train and test set as provided data is not presplit
        x_train, x_test, y_train, y_test = train_test_split(
            filenames,
            idxs,
            test_size=0.2,
            random_state=1,
        )

        if phase == "train":
            return x_train.tolist(), y_train
        elif phase == "val":
            return x_test.tolist(), y_test
        else:
            raise ValueError("Unknown phase")

    def _check_exists(self) -> bool:
        """
        Check the Root directory is exists
        """
        return os.path.exists(self.root_dir)

    def download(self) -> None:
        """
        Download the dataset from the internet
        """

        if self._check_exists():
            return

        os.makedirs(self.root_dir, exist_ok=True)
        _download_file_from_url(
            "https://drive.google.com/u/0/uc?id=1PCesRqeXYINcsulnTixVjR15xFNXropZ&export=download&confirm=t",
            os.path.join(self.root_dir, "zero_dce.zip"),
        )
        extract_archive(
            os.path.join(self.root_dir, "zero_dce.zip"),
            self.root_dir,
            remove_finished=True,
        )


if __name__ == "__main__":
    data = Zero_DCE("light_side/datas/zero_dce")
    print(data[0])
    print(data.classes)
    print(len(data.classes))
