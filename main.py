import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3, 7"

import torch
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)
import torch.optim as optim
from torchvision import transforms as T

import argparse
import torch.nn as nn
import numpy as np
import random
import json

from matplotlib import pyplot as plt

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from scipy.stats import pearsonr, spearmanr, kendalltau

from model.baseline import baseline_resnet50
from model.QAModel import qamodel_resnet34
from data.dataloader import get_loader
from model.QAloss import QALoss
from lib import BaseTrainer

class QATrainer(BaseTrainer):
    def __init__(self, cls_criterion, **kargs):
        super().__init__(**kargs)

        self.cls_criterion = cls_criterion

        self.db = self.args.db
        self.db2 = self.args.db2 if self.args.db2 != "none" else self.db

    def train_step(self, batch, batch_idx):
        image = batch[0]

        mos = batch[-1]
        # mos_cls = torch.floor(mos*2).long()
        mos = mos.float()

        qual = self.model(image)

        qual_pre = qual.squeeze(-1)

        if self.db == "wpc":
            mos = (mos / 100) * 4 + 1
        elif self.db == "sjtu":
            mos = ((mos - 1) / 9) * 4  + 1

        loss = 5 * self.criterion(qual_pre, mos) + self.cls_criterion(qual_pre, mos)

        self.logger("Train/Loss", loss, self.current_epoch*batch_idx)
        return loss

    def val_step(self, batch, batch_idx):
        image = batch[0]

        mos = batch[-1]

        mos = mos.double()

        qual = self.model(image)

        qual_pre = qual.squeeze(-1)


        if self.db == "wpc":
            mos = (mos / 100) * 4 + 1
        elif self.db == "sjtu":
            mos = ((mos - 1) / 9) * 4  + 1

        loss = 5 * self.criterion(qual_pre, mos) + self.cls_criterion(qual_pre, mos)

        return {
            "loss": loss.item(),
            "pred": qual_pre.detach().cpu().numpy(),
            "label": mos.detach().cpu().numpy(),
        }

    def val_epoch_end(self, info):
        pred = np.array([], dtype=np.float64)
        mos = np.array([], dtype=np.float64)
        losses = np.array([], dtype=np.float64)

        for o in info:
            losses = np.append(losses, o["loss"])
            pred = np.append(pred, o["pred"])
            mos = np.append(mos, o["label"])

        # if self.db2 == "sjtu":
        #     pred = ((pred - 1) / 4) * 9 + 1
        #     # mos = ((mos - 1) / 4) * 9 + 1
        # elif self.db2 == "wpc":
        #     pred = ((pred - 1) / 4) * 100
        #     # mos = ((mos - 1) / 4) * 100

        plcc, srocc, krocc, rmse = pearsonr(pred, mos)[0], spearmanr(pred, mos)[0], kendalltau(pred, mos)[0], np.sqrt(np.mean((pred-mos)**2))

        Loss = np.mean(losses)

        # self.logger("Val/PLCC", plcc, self.current_epoch)
        # self.logger("Val/SROCC", srocc, self.current_epoch)
        # self.logger("Val/KROCC", krocc, self.current_epoch)
        # self.logger("Val/LOSS", Loss, self.current_epoch)

        # self.logger.log("EPOCH[{}] -> PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, RMSE: {:.6f}, LOSS: {:.6f}".format(self.current_epoch, plcc, srocc, krocc, rmse, Loss))

        print("PLCC: {:.6f}, SROCC: {:.6f}, KROCC: {:.6f}, RMSE: {:.6f}, LOSS: {:.6f}".format(plcc, srocc, krocc, rmse, Loss))

        res = {
            "pred": pred,
            "mos": mos
        }

        return srocc + plcc + krocc, res


def main(args):
    model = qamodel_resnet34(
        norm_layer=nn.BatchNorm2d,
        pretrained=args.pretrained
    )
    # model = baseline_resnet50()

    if args.pretrained:
        learning_rates = {
            # "regressor": args.lr * 5,
            # "single_regressors": args.lr * 5
        }
        param_groups = []
        for name, param in model.named_parameters():
            if name in learning_rates:
                param_groups.append({'params': param, 'lr': learning_rates[name]})
            else:
                param_groups.append({'params': param})
        optimizer = optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        print("Edit LR!")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    transform_t = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = get_loader("train", flip=True, transform=transform_t, lazy=True, args=args)
    val_loader = get_loader("val", flip=False, transform=transform_t, lazy=True, args=args)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 70, 90], gamma=0.5)

    # trainer = QATrainer(model=model, c_loss=ContrastiveLoss(args.batch_size), optimizer=optimizer,  accelerator="dp", **vars(args))
    trainer = QATrainer(model=model, optimizer=optimizer, accelerator="dp", criterion=nn.HuberLoss(delta=0.01), cls_criterion=QALoss(), **vars(args))

    if args.train:
        trainer.fit(train_loader, val_loader)
    else:
        trainer.load_checkpoints(ckpt_path=args.ckpt_path)
        _, result = trainer.test(val_loader)

        x = result["mos"]
        y = result["pred"]

        print(len(x))

        with open("prediction_sjtu_7_.json", "w") as f:
            json.dump(list(y), f, ensure_ascii=False, indent=4)
            print("prediction dumped!")

        line = np.linspace(0, 10, 10)

        # plt.figure()

        color = []

        c = ["red", "purple", "gray", "blue"]
        j= 0
        for i in range(1, x.shape[0]+1):
            color.append(c[j])

            # if i % 37 == 0:
            #     j += 1

        plt.plot(line, line+0, c="black")
        plt.scatter(y, x, c=color)

        plt.ylabel("MOS")
        plt.xlabel("Prediction")

        plt.xlim(0, 10)
        plt.ylim(0, 10)

        # plt.savefig("./res.png")
        plt.savefig("./res_sjtu.pdf", bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=False)
    # parser.add_argument("--ckpt_path", type=str, default="/home/lhh/codes/ViewPCQA/version_1/checkpoints.pth.tar")
    # parser.add_argument("--ckpt_path", type=str, default="/home/lhh/codes/ViewPCQA/logs/version_23/checkpoints.pth.tar")
    parser.add_argument("--ckpt_path", type=str, default="/home/lhh/codes/ViewPCQA/logs_bak/version_20/checkpoints.pth.tar")
    # parser.add_argument("--ckpt_path", type=str, default="/home/lhh/codes/ViewPCQA/logs/version_2/checkpoints.pth.tar")

    # DATA
    parser.add_argument("--train_info", type=str, default="./csvfiles/wpc_data_info/train_5.csv")
    parser.add_argument("--val_info", type=str, default="./csvfiles/sjtu_data_info/test_7.csv")
    parser.add_argument("--test_info", type=str, default="./data_cfgs/wpc/test.csv")
    parser.add_argument("--image_dir", type=str, default="/public/DATA/lhh/Projected_PC/SJTU/pixel_with_point/384_375/image/")

    # Model
    parser.add_argument("--loss", type=str, default="huber")

    # Hyper Parameters
        # Optimizer
    # parser.add_argument("--optimizer", type=str, default="None")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)

    parser.add_argument("--pretrained", type=bool, default=True)

        # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="stepLR")
    parser.add_argument("--lr_decay_steps", type=float, default=15)
    parser.add_argument("--tmax", type=int, default=5)
    parser.add_argument("--lr_decay_min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_decay_rate", type=float, default=-1)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--db", type=str, default="sjtu")
    parser.add_argument("--db2", type=str, default="sjtu")

    # Training
    parser.add_argument("--batch_size", type=int, default=1)

    parser.set_defaults(max_epochs=60)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    main(args)
