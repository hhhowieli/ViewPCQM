import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
import torch.optim as optim
import abc
import json

from tqdm import tqdm
from easydict import EasyDict as edict

from .logger import TBLogger, CSVLogger

class BaseTrainer(object):

    def __init__(self, **kargs):

        self.args = edict(kargs)
        assert "model" in kargs.keys() or "models" in kargs.keys()
        self._check_parameters()

        if self.model:
            self.model = self.model.to(self.device)
        elif self.models:
            for i in range(len(self.models)):
                self.models[i] = self.models.to(self.device)

        if not self.optimizer:
            self.optimizer = self.configure_optimizers()
        if not self.scheduler:
            self.scheduler = self.configure_schedulers()
        if not self.criterion:
            self.criterion = self.configure_criterions()

    def fit(self, train_loader, val_loader=None):
        self.init_log()
        self.logger.log(str(self))
        self.logger.log(str(self.model))
        self._load_accelerator()

        start_epoch = self.current_epoch

        train_info = []
        for epoch in range(start_epoch, self.max_epochs+1):
            self.current_epoch = epoch
            train_epoch_info = self.train_epoch(train_loader)
            if train_epoch_info:
                train_info.append(train_epoch_info)

            if epoch % self.val_interval == 0 and val_loader:
                val_info = self.val_epoch(val_loader)

                if val_info >= self.best_metric:
                    self.best_metric = val_info
                    self.best_epoch = epoch
                    self.save_checkpoints()

            if self.scheduler:
                self.scheduler.step()

        self.logger.log("Best Epoch: {:d}".format(self.best_epoch))
        return train_info

    def test(self, test_loader):
        self._load_accelerator()
        return self.val_epoch(test_loader)

    def train_epoch(self, train_loader):
        self.model.train()
        train_epoch_info = []
        descrip = "Epoch {}".format(self.current_epoch)
        tqdmitem = tqdm(enumerate(train_loader), desc=descrip, total=len(train_loader), leave=False)
        for step, (batch) in tqdmitem:
            self.optimizer.zero_grad()
            # To Cuda
            for i in range(1, len(batch)):
                batch[i] = batch[i].to(self.device)
            for i in range(len(batch[0])):
                batch[0][i] = batch[0][i].to(self.device)

            train_step_info = self.train_step(batch, step)

            if isinstance(train_step_info, dict):
                loss = train_step_info["loss"]
            else:
                loss = train_step_info
            loss.backward()

            for name, param in self.model.named_parameters():
                if "p_norm" in name:
                    torch.nn.utils.clip_grad_norm_(param, max_norm=0.1)
                    # print("Cliped")

            self.optimizer.step()
            self.train_step_end(train_step_info)
            if train_step_info:
                train_epoch_info.append(train_step_info)

            if self.scheduler:
                tqdmitem.set_postfix(loss=loss.item(), lr="{:e}".format(self.scheduler.get_last_lr()[0]))
            else:
                tqdmitem.set_postfix(loss=loss.item())

        self.train_epoch_end(train_epoch_info)
        # torch.cuda.empty_cache()

        return train_epoch_info

    def val_epoch(self, val_loader):
        self.model.eval()
        val_info = []
        for step, (batch) in tqdm(enumerate(val_loader)):

            # To Cuda
            # batch = batch["image"].to(self.device)

            # To Cuda
            for i in range(1, len(batch)):
                batch[i] = batch[i].to(self.device)
            for i in range(len(batch[0])):
                if isinstance(batch[0][i], list):
                    continue
                batch[0][i] = batch[0][i].to(self.device)

            val_step_info = self.val_step(batch, step)
            self.val_step_end(val_step_info)
            if val_step_info:
                val_info.append(val_step_info)

        info = self.val_epoch_end(val_info)

        if info != None:
            val_info = info

        return val_info

    @abc.abstractmethod
    def train_step(self, batch, batch_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def val_step(self, batch, batch_idx):
        raise NotImplementedError

    def train_step_end(self, info):
        return info

    def val_step_end(self, info):
        return info

    def train_epoch_end(self, info):
        return info

    def val_epoch_end(self, info):
        return info

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    # @abc.abstractmethod
    # def configure_schedulers(self):
    #     pass

    # @abc.abstractmethod
    # def configure_criterions(self):
    #     pass

    def configure_criterions(self):
        if self.args.loss == "mse":
            return nn.MSELoss()
        if self.args.loss == "huber":
            return nn.HuberLoss()

    def configure_schedulers(self):
        if self.args.lr_scheduler == "cosine":
            min_lr = self.args.lr_decay_min_lr
            tmax = self.args.tmax
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tmax, eta_min=min_lr)
        if self.args.lr_scheduler == "exp":
            gamma = self.args.gamma
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        if self.args.lr_scheduler == "stepLR":
            decay_steps = self.args.lr_decay_steps
            gamma = self.args.gamma
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=gamma)

        return scheduler


    def save_checkpoints(self):

        if self.accelerator == "dp":
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        checkpoints = {
            "state_dict": state_dict,
            "opt_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch
        }
        torch.save(checkpoints, os.path.join(self.version_path, "checkpoints.pth.tar"))

    def load_checkpoints(self, version_num=-1, ckpt_path=None, test_loader=None):
        if version_num != -1:
            version_path = "./logs/version_"+str(version_num)
            ckpt_path = os.path.join(version_path, "checkpoints.pth.tar")

        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["state_dict"])
        print("Successfully load model from {} !\t".format(ckpt_path))
        if test_loader:
            self.val_epoch(test_loader)

    def init_log(self):
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
            self.v_num = 0
        else:
            self.v_num = len(os.listdir("./logs"))

        self.best_epoch = 0

        self.version_path = "./logs/version_"+str(self.v_num)
        os.mkdir(self.version_path)

        self._dump_args2json()

        self.logger = TBLogger(self.version_path)

        print("Log dir in " + str(self.version_path))

    def _load_accelerator(self):
        if self.accelerator == "gpu":
            return 
        elif self.accelerator == "dp":
            self.model = DP(self.model)

    def _check_parameters(self):
        self.optimizer = self.args.get("optimizer", None)
        self.scheduler = self.args.get("scheduler", None)
        self.criterion = self.args.get("criterion", None)
        self.model = self.args.get("model", None)
        if self.model == None:
            self.models = self.args.get("models", None)

        self.logger = self.args.get("logger", "TensorboardX")
        self.val_interval = self.args.get("val_interval", 1)

        self.device = ("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.max_epochs = self.args.get("max_epochs")

        self.accelerator = self.args.get("accelerator", "gpu")

        self.current_epoch = 1
        self.best_metric = -1


    def __str__(self) -> str:
        total = 0
        if self.model:
            total = sum([param.nelement() for param in self.model.parameters()])

        printstr = "Number of parameter: {:.2f}M\n Optimizer: {}\n Version: {}"
        printstr = printstr.format(total/1e6, type(self.optimizer).__name__, self.v_num)

        return printstr

    def _dump_args2json(self):
        params = {
            "model": type(self.model).__name__,
            "optimizer": type(self.optimizer).__name__,
            "scheduler": type(self.scheduler).__name__,
            "batch_size": self.args.batch_size,
            "max_epochs": self.args.max_epochs,
            "lr": self.args.lr,
            "gamma": self.args.gamma,
            "val_info": self.args.val_info,
            "pretrained": self.args.pretrained
        }

        with open(os.path.join(self.version_path, "params.json"), "w") as f:
            json.dump(params, fp=f)






if __name__=="__main__":

    net = nn.Linear(10, 1)

    trainer = BaseTrainer(net, optimizer=None)
