import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from typing import *
import datetime
import os

from .utils import *
from .datasets import DATASETS
from .models import MODELS
from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedulers import SCHEDULERS
from .config import Config


class Solver:
    def __init__(self, cfg: Config):
        self.cfg = cfg.build()

        self.device = torch.device(cfg.device)
        self.output_dir = cfg.output_dir

        self.train_loader = None
        if cfg.train_data_name is not None:
            train_data = DATASETS[cfg.train_data_name](**cfg.train_data_args)
            self.train_loader = DataLoader(train_data, **cfg.train_loader_args)

        self.val_loader = None
        if cfg.val_data_name is not None and cfg.do_val:
            val_data = DATASETS[cfg.val_data_name](**cfg.val_data_args)
            self.val_loader = DataLoader(val_data, **cfg.val_data_args)

        self.model = None
        if cfg.model_name is not None:
            self.model = MODELS[cfg.model_name](**cfg.model_args).to(self.device)

        self.loss = None
        if cfg.loss_name is not None:
            self.loss = LOSSES[cfg.loss_name](**cfg.loss_args).to(self.device)

        self.scheduler = None
        if cfg.scheduler_name is not None:
            self.scheduler = SCHEDULERS[cfg.scheduler_name](**cfg.scheduler_args)
            cfg.optimizer_args.lr = self.scheduler

        self.optimizer = None
        if cfg.optimizer_name is not None:
            self.optimizer = OPTIMIZERS[cfg.optimizer_name](**cfg.optimizer_args).to(self.device)

        if cfg.pretrained_file is not None and os.path.exists(cfg.pretrained_file):
            self.load_pretrained(cfg.pretrained_file)

        self.start_epoch = 1
        self.epochs = cfg.epochs
        self.output_dir = cfg.output_dir
        self.log_dir = os.path.join(self.output_dir, "log")
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.best_direction = "high"  # high or low
        self.best_score = 0
        self.global_step = 0
        if self.cfg.num_device > 1 and self.model is not None:
            self.model = torch.nn.DataParallel(self.model)
        self.load_checkpoint()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_epoch(epoch)

            score, is_best = None, False
            if self.val_loader is not None:
                score = self.val_epoch(epoch)
                if self.best_direction == "high":
                    if score >= self.best_score:
                        self.best_score = score
                        is_best = True
                else:
                    if score <= self.best_score:
                        self.best_score = score
                        is_best = True

            if self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint(epoch, score=score, is_best=is_best)

    def train_epoch(self, epoch):
        self.model.eval()

        t, c = Timer(), Counter()
        t.start()
        for step, data in enumerate(self.train_loader):
            data = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in data]
            reader_time = t.elapsed_time()

            loss, metrics = self.train_step(*data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time = t.elapsed_time()
            for k, v in metrics.items():
                if isinstance(v, float):
                    c[k] = v

            c.append(batch_time=batch_time)
            eta = calculate_eta(len(self.train_loader) - step, c.batch_time)

            msg = "".join(f"{k}={v:.4f}{c[k]:.4f} " for k, v in metrics.items() if isinstance(v, float))
            self.log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                     f"[epoch={epoch}/{self.epochs}] "
                     f"step={step + 1}/{len(self.train_loader)} "
                     f"loss={loss:.4f}/{c.loss:.4f} "
                     f"{msg}"
                     f"batch_time={c.batch_time:.4f}+{c.reader_time:.4f} "
                     f"| ETA {eta}",
                     end="\r",
                     to_file=step % 10 == 0)

    def train_step(self, *args) -> Tuple[torch.Tensor, Dict]:
        pass

    @torch.no_grad()
    def val_epoch(self, epoch) -> float:
        pass

    def load_pretrained(self, pretrained_file):
        pass

    def save_checkpoint(self, epoch, score=None, is_best=False):
        state = {
            "epoch": epoch,
            "score": score,
            "best_score": self.best_score,
            "global_step": self.global_step
        }

        if self.model is not None and isinstance(self.model, nn.Module):
            state["model"] = self.model.state_dict()

        if self.loss is not None and isinstance(self.loss, nn.Module):
            state["loss"] = self.loss.state_dict()

        if self.optimizer is not None and isinstance(self.optimizer, nn.Module):
            state["optimizer"] = self.optimizer.state_dict()

        if self.scheduler is not None and isinstance(self.scheduler, nn.Module):
            state["scheduler"] = self.scheduler.state_dict()

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(state, os.path.join(self.output_dir, "latest.pth"))
        if is_best:
            torch.save(state, os.path.join(self.output_dir, f"{score:.4f}_{epoch:04}_model.pth"))

    def load_checkpoint(self):
        file = os.path.join(self.output_dir, "latest.pth")
        if not os.path.exists(file):
            return

        state = torch.load(file)
        self.start_epoch = state["epoch"] + 1
        self.best_score = state["best_score"]
        self.global_step = state["global_step"]

        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.load_state_dict(state["model"])

        if self.loss is not None and isinstance(self.loss, nn.Module):
            self.loss.load_state_dict(state["loss"])

        if self.optimizer is not None and isinstance(self.optimizer, nn.Module):
            self.optimizer.load_state_dict(state["optimizer"])

        if self.scheduler is not None and isinstance(self.scheduler, nn.Module):
            self.scheduler.load_state_dict(state["scheduler"])

        print(f"load checkpoint {file}")

    def log(self, msg, end='\n', to_file=True):
        print(msg, end=end, flush=True)
        if to_file and self.log_file is not None:
            print(msg, end='\n', flush=True, file=open(self.log_file, "a+"))
