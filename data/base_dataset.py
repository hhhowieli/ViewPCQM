import os
import abc
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(self, data_info=None, transform=None, lazy=False, labeled=True, db="wpc"):
        super().__init__()

        self.transform = transform
        self.lazy = lazy
        self.labeled = labeled

        self.db = db

        self.INFO = pd.read_csv(data_info).to_numpy()

        if not lazy:
            self.datas = []
            l = self.INFO.shape[0]

            for i in tqdm(range(l)):
                data = self._load_data(i)
                self.datas.append(data)

            print("Sucessfully load {} in pre-load mode!".format(data_info))
        else:
            print("Sucessfully load {} in lazy mode!".format(data_info))

    def __len__(self):
        return self.INFO.shape[0]

    def __getitem__(self, index):
        if not self.lazy:
            data = self.datas[index]
        else:
            data = self._load_data(index)

        if not self.labeled:
            return data

        mos = self.INFO[index, -1]

        if self.db == "wpc":
            mos = (mos / 100) * 4 + 1
        else:
            mos = ((mos - 1) / 9) * 4  + 1
        return data, mos

    abc.abstractmethod
    def _load_data(self, index):
        raise NotImplementedError


# Template
def get_loader(phase, labeled=True, lazy=False, transform=None, args=None):

    data = {
        "train": (args.train_info, args.train_dir),
        "val": (args.val_info, args.val_dir),
        "test": (args.test_info, args.test_dir)
    }[phase]

    if not labeled:
        data = args.unlabel_info, args.unlabel_dir

    data_info, data_dir = data


    dataset = BaseDataset(data_info, data_dir, transform, lazy=lazy, labeled=labeled)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if phase=="train" or phase=="val" else 1,
        shuffle=(phase=="train"),
        pin_memory=True,
        num_workers=4
    )

    return loader
