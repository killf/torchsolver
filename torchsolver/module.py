import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from typing import *
import datetime
import os

from .utils import *


class Module(nn.Module):
    def __init__(self, task_name=None, epochs=20, batch_size=32, device=None, output_dir="output", pretrained_file=None, num_device=None):
        super(Module, self).__init__()

        self.task_name = task_name
        if task_name is not None:
            output_dir = os.path.join(output_dir, str(task_name))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                if num_device is None:
                    num_device = torch.cuda.device_count()
            else:
                num_device = 1

        self.device = torch.device(device)
        self.num_device = num_device

        self.start_epoch = 1
        self.epochs = epochs
        self.batch_size = batch_size

        self.output_dir = output_dir
        self.log_dir = os.path.join(self.output_dir, "log")
        self.log_file = os.path.join(self.output_dir, "log.txt")
        self.logger = SummaryWriter(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.pretrained_file = pretrained_file

        self.best_direction = "high"  # high or low
        self.best_score = 0
        self.global_step = 0

    def fit(self, train_loader=None, train_data=None, train_batch_size=None, val_loader=None, val_data=None, val_batch_size=None, do_val=True, num_workers=4):
        if train_batch_size is None:
            train_batch_size = self.batch_size

        if val_batch_size is None:
            val_batch_size = self.batch_size

        if train_loader is None and train_data is not None:
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        if val_loader is None and val_data is not None:
            val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        self.to(self.device)

        if self.pretrained_file is not None and os.path.exists(self.pretrained_file):
            self.load_pretrained(self.pretrained_file)

        self.load_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_epoch(epoch, train_loader)

            score, is_best = None, False
            if do_val:
                score = self.val_epoch(val_loader)
                if score is not None:
                    if self.best_direction == "high":
                        if score >= self.best_score:
                            self.best_score = score
                            is_best = True
                    else:
                        if score <= self.best_score:
                            self.best_score = score
                            is_best = True

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            self.save_checkpoint(epoch, score=score, is_best=is_best)

    def train_epoch(self, epoch, train_loader):
        self.train(True)

        t, c = Timer(), Counter()
        t.start()
        for step, inputs in enumerate(train_loader):
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs.to(self.device),)
            else:
                inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
            reader_time = t.elapsed_time()

            metrics = self.train_step(*inputs)

            for k in metrics:
                if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0:
                    metrics[k] = float(metrics[k])

            batch_time = t.elapsed_time()
            for k, v in metrics.items():
                if isinstance(v, float):
                    c.set(k, v)

            c.append(batch_time=batch_time, reader_time=reader_time)
            eta = calculate_eta(len(train_loader) - step, c.batch_time)

            msg = "".join(f"{k}={v:.4f}/{c.get(k):.4f} " for k, v in metrics.items() if isinstance(v, float))
            self.log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
                     f"[epoch={epoch}/{self.epochs}] "
                     f"step={step + 1}/{len(train_loader)} "
                     f"{msg}"
                     f"batch_time={c.batch_time:.4f}+{c.reader_time:.4f} "
                     f"| ETA {eta}",
                     end="\r",
                     to_file=step % 10 == 0)

            self.log_tb(metrics, "train")
            self.global_step += 1
            t.restart()
        print()

    def train_step(self, *inputs) -> Dict:
        loss, metrics = self(*inputs)
        self.backward(loss)
        return metrics

    def backward(self, loss, update_optimizer=True):
        if hasattr(self, "optimizer") and update_optimizer:
            self.optimizer.zero_grad()
        loss.backward()
        if hasattr(self, "optimizer") and update_optimizer:
            self.optimizer.step()

    def val_epoch(self, val_loader) -> Union[float, None]:
        if val_loader is None:
            return

        self.train(False)

        c = Counter()
        for step, inputs in enumerate(val_loader):
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs.to(self.device),)
            else:
                inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]

            score, metrics = self.forward(*inputs)

            for k in metrics:
                if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0:
                    metrics[k] = float(metrics[k])

            c.append(score=float(score))
            for k, v in metrics.items():
                if isinstance(v, float):
                    c.set(k, v)

            print(f"[VAL] {step + 1}/{len(val_loader)}", end="\r", flush=True)

        msg = "".join(f"{k}={c.get(k):.4f} " for k in c.data.keys())
        self.log(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [VAL] {msg}")
        self.log_tb({k: c.get(k) for k in c.data}, "val")

        print()
        return c.score

    def val_step(self, *inputs) -> Tuple[float, Dict]:
        score, metrics = self(*inputs)
        return score, metrics

    def load_pretrained(self, pretrained_file: str):
        pass

    def save_checkpoint(self, epoch, score=None, is_best=False):
        state_dict = dict()
        for k in dir(self):
            if k.startswith("_") or not hasattr(self, k):
                continue

            v = getattr(self, k)
            if hasattr(v, "state_dict"):
                if isinstance(v, torch.nn.DataParallel):
                    state_dict[k] = v.module.state_dict()
                else:
                    state_dict[k] = v.state_dict()

        state_dict.update({
            "epoch": epoch,
            "score": score,
            "best_score": self.best_score,
            "global_step": self.global_step
        })

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(self.output_dir, "latest.pth"))
        if is_best:
            torch.save(state_dict, os.path.join(self.output_dir, f"{score:.4f}_{epoch:04}_model.pth"))

    def load_checkpoint(self):
        file = os.path.join(self.output_dir, "latest.pth")
        if not os.path.exists(file):
            return

        state_dict = torch.load(file)
        self.start_epoch = state_dict["epoch"] + 1
        self.best_score = state_dict["best_score"]
        self.global_step = state_dict["global_step"]

        for k in dir(self):
            if k.startswith("_") or not hasattr(self, k):
                continue

            v = getattr(self, k)
            if hasattr(v, "load_state_dict"):
                if isinstance(v, torch.nn.DataParallel):
                    v.module.load_state_dict(state_dict[k])
                else:
                    v.load_state_dict(state_dict[k])

        print(f"load checkpoint {file}")

    def log(self, msg, end='\n', to_file=True):
        print(msg, end=end, flush=True)
        if to_file and self.log_file is not None:
            print(msg, end='\n', flush=True, file=open(self.log_file, "a+"))

    def log_tb(self, infos: Dict, mode: str):
        for k, v in infos.items():
            if isinstance(v, float):
                self.logger.add_scalar(f"{mode}/{k}", v, global_step=self.global_step)
            elif isinstance(v, torch.Tensor) and v.dim == 4 and int(v.shape[1]) in [1, 3, 4]:
                self.logger.add_images(f"{mode}/{k}", v, global_step=self.global_step)
            elif isinstance(v, torch.Tensor) and v.dim == 3 and int(v.shape[0]) in [1, 3, 4]:
                self.logger.add_image(f"{mode}/{k}", v, global_step=self.global_step)
        self.logger.flush()


class GANModule(Module):
    def forward_d(self, *inputs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError()

    def forward_g(self, *inputs) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError()

    def train_step(self, *inputs) -> Dict:
        d_metrics = self.train_step_d(*inputs)
        g_metrics = self.train_step_g(*inputs)

        d_metrics.update(g_metrics)
        return d_metrics

    def train_step_d(self, *inputs) -> Dict:
        d_loss, d_metrics = self.forward_d(*inputs)

        if hasattr(self, "d_optimizer"):
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        return d_metrics

    def train_step_g(self, *inputs) -> Dict:
        g_loss, g_metrics = self.forward_g(*inputs)

        if hasattr(self, "g_optimizer"):
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

        return g_metrics

    @torch.no_grad()
    def val_epoch(self, *args):
        raise NotImplementedError()
