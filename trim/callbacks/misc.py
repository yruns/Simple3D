"""
Misc Callbacks

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.utils.data

from trim.callbacks.default import CallbackBase
from trim.utils import dist
from trim.utils.dist import is_main_process
from trim.utils.timer import Timer


class IterationTimer(CallbackBase):

    def __init__(self, warmup_iter=2):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def on_training_phase_start(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def on_training_epoch_start(self):
        self._iter_timer.reset()

    def on_training_step_start(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def on_training_step_end(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


class InformationWriter(CallbackBase):
    def __init__(self, log_interval=10):
        self.curr_iter = 0
        self.log_interval = log_interval
        self.model_output_keys = []

    @property
    def epoch_width(self):
        return len(str(self.trainer.max_epoch))

    @property
    def iter_width(self):
        return len(str(len(self.trainer.train_loader)))

    def on_training_phase_start(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.trainer.logger.info(self.trainer.args)

    def on_training_step_start(self):
        self.curr_iter += 1
        info = "Train: [{epoch:>{e_w}}/{max_epoch}][{iter:>{i_w}}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
            e_w=self.epoch_width,
            i_w=self.iter_width
        )
        self.trainer.comm_info["iter_info"] += info

    def on_training_step_end(self):
        current_iter = self.trainer.epoch * len(self.trainer.train_loader) + self.trainer.comm_info["iter"]

        # Anything you want to log in terminal and file
        if "terminal_log" in self.trainer.comm_info.keys():
            terminal_log = self.trainer.comm_info["terminal_log"]
            self.model_output_keys = terminal_log.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, terminal_log[key])
                # self.trainer.wandb.log({
                #     key: terminal_log[key],
                # })

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)

        # log in terminal and file
        if (self.curr_iter + 1) % self.log_interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])

        # Anything you want to log in wandb
        self.trainer.wandb.log({"epoch": self.trainer.epoch})
        if "wandb_log" in self.trainer.comm_info.keys():
            wandb_log = self.trainer.comm_info["wandb_log"]
            for key in wandb_log.keys():
                self.trainer.wandb.log({
                    key: wandb_log[key],
                })

        self.trainer.comm_info["iter_info"] = ""  # reset iter info


class CheckpointSaver(CallbackBase):
    """
    CheckpointSaver

    If you are using this callback, be sure to set `self.trainer.comm_info["current_metric_value"]` and
    `self.trainer.comm_info["current_metric_name"]` before executing this callback.
    It is recommended to set these values in the `Evaluator` callback.
    """

    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save models last

    def on_training_epoch_end(self):
        if self.trainer.args.eval:
            return

        is_best = False
        current_metric_value = self.trainer.comm_info.get("current_metric_value", 0)
        current_metric_name = self.trainer.comm_info.get("current_metric_name", 0)
        if current_metric_value > self.trainer.best_metric_value:
            self.trainer.best_metric_value = current_metric_value
            self.trainer.best_metric_epoch = self.trainer.epoch + 1
            is_best = True
            self.trainer.logger.info(
                "Best validation {} updated to: {:.4f}".format(
                    current_metric_name, current_metric_value
                )
            )
        self.trainer.logger.info(
            "Currently Best {}: {:.4f} at epoch {}".format(
                current_metric_name, self.trainer.best_metric_value, self.trainer.best_metric_epoch
            )
        )

        self.trainer.wandb.update({
            f"best_{current_metric_name}": self.trainer.best_metric_value,
            f"best_{current_metric_name}_epoch": self.trainer.best_metric_epoch
        })

        filename = os.path.join(
            self.trainer.output_dir, "model", "model_last.pth"
        )
        os.makedirs(os.path.join(self.trainer.output_dir, "model"), exist_ok=True)

        self.trainer.logger.info("Saving checkpoint to: " + filename)
        if is_main_process():
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict()
                    if self.trainer.scheduler else None,
                    "scaler": self.trainer.scaler.state_dict()
                    if self.trainer.scaler else None,
                    "best_metric_value": self.trainer.best_metric_value,
                    "best_metric_epoch": self.trainer.best_metric_epoch,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
        if is_best and is_main_process():
            shutil.copyfile(
                filename,
                os.path.join(self.trainer.output_dir, "model", "model_best.pth"),
            )
        if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0 and is_main_process():
            shutil.copyfile(
                filename,
                os.path.join(
                    self.trainer.output_dir,
                    "model",
                    f"epoch_{self.trainer.epoch + 1}.pth",
                ),
            )


class CheckpointLoader(CallbackBase):
    def __init__(self, state_path, resume=False, keywords="", replacement=None, strict=True):
        self.state_path = state_path
        self.resume = resume
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def on_training_phase_start(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.state_path is None:
            self.trainer.logger.info("No weight found, training from scratch.")
            return

        if os.path.isdir(self.state_path):
            self.state_path = os.path.join(self.state_path, "model/model_best.pth")
        if os.path.isfile(self.state_path):
            self.trainer.logger.info(f"Loading weight at: {self.state_path}")
            checkpoint = torch.load(
                self.state_path,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )

            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    if dist.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if dist.get_world_size() == 1 and key.startswith("module."):
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value

            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.best_metric_epoch = checkpoint["epoch"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.scaler:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            raise FileNotFoundError(f"No weight found at: {self.state_path}")
