import torch
from torchvision.transforms import functional as F
import os
from PIL import Image
import numpy as np
import random as rd
from torchvision import transforms as T

from torch.utils.data import DataLoader

from .base_dataset import BaseDataset

def random_crop(ims, size):
    _, rh, rw = ims[0].shape

    new_ims = []
    for i in range(3):
        x = rd.randint(0, rh-size)
        y = rd.randint(0, rw-size)

        im1, im2 = ims[2*i], ims[2*i + 1]

        n_im1 = im1[:, x:x+size, y:y+size]
        n_im2 = im2[:, x:x+size, y:y+size]

        new_ims.append(n_im1)
        new_ims.append(n_im2)
    
    new_ims = torch.stack(new_ims, dim=0)
    return new_ims


class QADataset(BaseDataset):
    def __init__(self, flip=True, data_info=None, data_dir=None, transform=None, lazy=False, labeled=True, args=None):
        self.image_dir = data_dir

        self.flip = flip

        self.hflip_trans = T.RandomHorizontalFlip(p=1)
        self.vflip_trans = T.RandomVerticalFlip(p=1)

        self.args = args

        super().__init__(data_info, transform, lazy, labeled, db=args.db)

    def _load_data(self, index):
        # ply_name = self.INFO[index, 0]
        name = self.INFO[index, 0]

        ims = []

        for i in range(6):

            s_name = name[:-4] + "_" + str(i+1)
            im = Image.open(os.path.join(self.image_dir, s_name+".png")).convert("RGB")

            if i % 2 == 0:
                hfilp = np.random.random()
                vfilp = np.random.random()

            if self.transform:
                if self.flip:
                    if hfilp < 0.5:
                        im = self.hflip_trans(im)
                    if vfilp < 0.5:
                        im = self.vflip_trans(im)
                im = self.transform(im)

            ims.append(im)

        ims = torch.stack(ims, dim=0)
        # if self.flip == True:
        #     ims = random_crop(ims, size=224)
        # else:
        #     ims = torch.stack(ims, dim=0)

        return (ims)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    rd.seed(worker_seed)

def get_loader(phase, flip=True, labeled=True, lazy=False, transform=None, args=None):

    if phase == "train":
        data = args.train_info if labeled else args.unlabel_info
    else:
        data = {
            "val": args.val_info,
            "test": args.test_info
        }[phase]

    data_info = data

    data_dir = args.image_dir

    dataset = QADataset(flip, data_info, data_dir, transform, lazy=lazy, labeled=labeled, args=args)

    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if phase=="train" or phase=="val" else 1,
        shuffle=(phase=="train"),
        pin_memory=True,
        num_workers=8,
        drop_last=(phase=="train"),
        worker_init_fn=seed_worker,
        generator=g
    )


    return loader

