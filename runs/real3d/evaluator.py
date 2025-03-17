import os
import pickle
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trim.callbacks.default import CallbackBase
from trim.utils import dist, comm
from utils.metrics import compute_metrics


class ClsEvaluator(CallbackBase):

    def __init__(self):
        self.best_acc = 0.0

    def on_training_epoch_end(self):
        # eval per 4 epochs
        interval = 1
        cur_epoch = self.trainer.epoch
        if (cur_epoch + 1) % interval != 0:
            return

        torch.cuda.empty_cache()
        self.trainer.model.eval()
        self.eval()
        self.trainer.model.train()

    def eval_step(self, pointcloud):
        model = self.trainer.model

        features, _, voxel_indices = model.embed_pointmae(pointcloud, trianing=False)
        if model.pre_proj > 0 and model.pre_projection is not None:
            features = model.pre_projection(features)
        logits = model.discriminator(features).squeeze(-1)

        return logits.detach().cpu().numpy(), voxel_indices

    @torch.no_grad()
    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        logits, mask_pred, label_gt, mask_gt = [], [], [], []
        val_iter = tqdm(self.trainer.val_loader, total=len(self.trainer.val_loader))
        for i, batch_data in enumerate(val_iter):
            batch_data = comm.move_tensor_to_device(batch_data)

            pointcloud, mask, label, path = batch_data
            logit, voxel_indices = self.eval_step(pointcloud)

            logits.append(logit)
            mask_pred.append(logit)
            mask_gt.append(mask.squeeze(0).cpu().numpy()[voxel_indices])
            label_gt.append(label.cpu().numpy())

        # To numpy
        logits = np.concatenate(logits)
        mask_pred = np.concatenate(mask_pred)
        mask_gt = np.concatenate(mask_gt)
        label_gt = np.concatenate(label_gt)

        p_ap, p_auroc, p_true, p_fake, f1 = compute_metrics(logits, mask_pred, mask_gt, label_gt)


        self.trainer.wandb.log({
            "val_pixel_ap": p_ap,
            "val_pixel_auroc": p_auroc,
            "val_p_true": p_true,
            "val_p_fake": p_fake,
            "val_f1": f1
        })

        if p_auroc >= self.best_acc:
            self.best_acc = p_auroc

        self.trainer.comm_info["current_metric_value"] = p_auroc
        self.trainer.comm_info["current_metric_name"] = "p_auroc"

        self.trainer.logger.info("P-AUROC: %f" % p_auroc)
        self.trainer.logger.info("P-AP: %f" % p_ap)
        self.trainer.logger.info("P-True: %f" % p_true)
        self.trainer.logger.info("P-Fake: %f" % p_fake)
        self.trainer.logger.info("F1: %f" % f1)

        self.trainer.logger.info("Best P-AUROC: %f" % self.best_acc)
        self.trainer.logger.info("<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")


class Visualizer(CallbackBase):

    def __init__(self):
        self.best_acc = 0.0

    def on_training_epoch_end(self):
        # eval per 4 epochs
        interval = 1
        cur_epoch = self.trainer.epoch
        if (cur_epoch + 1) % interval != 0:
            return

        torch.cuda.empty_cache()
        self.trainer.model.eval()
        self.eval()
        self.trainer.model.train()

    def eval_step(self, pointcloud):
        model = self.trainer.model

        features, _, voxel_indices = model.embed_pointmae(pointcloud, trianing=False)
        if model.pre_proj > 0 and model.pre_projection is not None:
            features = model.pre_projection(features)
        logits = model.discriminator(features).squeeze(-1)

        return logits.detach().cpu().numpy(), voxel_indices

    @torch.no_grad()
    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        logits, mask_pred, label_gt, mask_gt = [], [], [], []
        val_iter = tqdm(self.trainer.val_loader, total=len(self.trainer.val_loader))
        for i, batch_data in enumerate(val_iter):
            batch_data = comm.move_tensor_to_device(batch_data)

            pointcloud, mask, label, path = batch_data
            logit, voxel_indices = self.eval_step(pointcloud)

            logits.append(logit)
            mask_pred.append(logit)
            mask_gt.append(mask.squeeze(0).cpu().numpy()[voxel_indices])
            label_gt.append(label.cpu().numpy())

        # To numpy
        logits = np.concatenate(logits)
        mask_pred = np.concatenate(mask_pred)
        mask_gt = np.concatenate(mask_gt)
        label_gt = np.concatenate(label_gt)

        p_ap, p_auroc, p_true, p_fake, f1 = compute_metrics(logits, mask_pred, mask_gt, label_gt)


        self.trainer.wandb.log({
            "val_pixel_ap": p_ap,
            "val_pixel_auroc": p_auroc,
            "val_p_true": p_true,
            "val_p_fake": p_fake,
            "val_f1": f1
        })

        if p_auroc >= self.best_acc:
            self.best_acc = p_auroc

        self.trainer.comm_info["current_metric_value"] = p_auroc
        self.trainer.comm_info["current_metric_name"] = "p_auroc"

        self.trainer.logger.info("P-AUROC: %f" % p_auroc)
        self.trainer.logger.info("P-AP: %f" % p_ap)
        self.trainer.logger.info("P-True: %f" % p_true)
        self.trainer.logger.info("P-Fake: %f" % p_fake)
        self.trainer.logger.info("F1: %f" % f1)

        self.trainer.logger.info("Best P-AUROC: %f" % self.best_acc)
        self.trainer.logger.info("<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
