import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from trim.callbacks.default import CallbackBase
from trim.utils import comm
from utils.metrics import compute_metrics
from utils.point_ops import upsample


class ClsEvaluator(CallbackBase):

    def __init__(self, interval, test):
        self.best_acc = 0.0
        self.interval = interval
        self.test = test

    def on_training_epoch_end(self):
        # eval per 4 epochs
        cur_epoch = self.trainer.epoch
        if (cur_epoch + 1) % self.interval != 0 and not self.test:
            return

        torch.cuda.empty_cache()
        self.trainer.model.eval()
        self.eval()
        self.trainer.model.train()

    def eval_step(self, pointcloud, gt_mask):
        model = self.trainer.model

        return model.eval_step(pointcloud, gt_mask)

    @torch.no_grad()
    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        save_dict = {
            "metrics": None,
            "epoch": self.trainer.epoch,
            "data": []
        }
        obj_pred, mask_pred, obj_gt, mask_gt = [], [], [], []
        val_iter = tqdm(self.trainer.val_loader, total=len(self.trainer.val_loader))
        for i, batch_data in enumerate(val_iter):
            batch_data = comm.move_tensor_to_device(batch_data)

            pointcloud, mask, label, path = batch_data
            scores = self.eval_step(pointcloud, mask)

            obj_pred.append(np.max(scores, axis=0))
            mask_pred.append(scores)
            mask_gt.append(mask[0].cpu().numpy())
            obj_gt.append(label[0].cpu().numpy())

            save_dict["data"].append({
                # "pointcloud": pointcloud.squeeze(0).cpu().numpy()[voxel_indices],
                "pred": scores,
                # "mask": voxel_labels,
                "obj_pred": obj_pred,
                "mask": mask[0].cpu().numpy(),
                "label": label.squeeze(0).cpu().numpy(),
                "path": path[0]
            })

        # To numpy
        obj_pred = np.array(obj_pred)
        obj_pred = torch.sigmoid(torch.from_numpy(obj_pred)).numpy()
        mask_pred = np.concatenate(mask_pred)
        mask_pred = torch.sigmoid(torch.from_numpy(mask_pred)).numpy()
        mask_gt = np.concatenate(mask_gt)
        obj_gt = np.array(obj_gt)

        # normalize
        # mask_pred = (mask_pred - mask_pred.min()) / (mask_pred.max() - mask_pred.min())

        p_ap, p_auroc, o_ap, o_auroc, p_true, p_fake, f1, threshold = compute_metrics(obj_pred, mask_pred, mask_gt, obj_gt)

        metrics = {
            "val_obj_ap": o_ap,
            "val_obj_auroc": o_auroc,
            "val_pixel_ap": p_ap,
            "val_pixel_auroc": p_auroc,
            "val_p_true": p_true,
            "val_p_fake": p_fake,
            "val_f1": f1,
            "threshold": threshold,
            "epoch": self.trainer.epoch,
        }
        self.trainer.wandb.log(metrics)
        save_dict["metrics"] = metrics

        if p_auroc >= self.best_acc:
            self.best_acc = p_auroc

        trainer_metrics = self.trainer.metrics_history
        # update trainer's metrics history
        if p_auroc > trainer_metrics["best_pauroc"]:
            trainer_metrics["best_pauroc"] = p_auroc
            trainer_metrics["best_pauroc_epoch"] = self.trainer.epoch
            trainer_metrics["oauroc_when_p_best"] = o_auroc

        if o_auroc > trainer_metrics["best_oauroc"]:
            trainer_metrics["best_oauroc"] = o_auroc
            trainer_metrics["best_oauroc_epoch"] = self.trainer.epoch
            trainer_metrics["pauroc_when_o_best"] = p_auroc
        self.trainer.metrics_history = trainer_metrics

        os.makedirs(os.path.join(self.trainer.output_dir, "eval"), exist_ok=True)
        with open(os.path.join(
                self.trainer.output_dir, "eval",
                f"{self.trainer.cls_name}_{self.trainer.epoch}_{'best' if p_auroc >= self.best_acc else ''}.pkl"
        ), "wb") as f:
            pickle.dump(save_dict, f)
            self.trainer.logger.info("Save evaluation results to %s" % f.name)

        self.trainer.comm_info["current_metric_value"] = p_auroc
        self.trainer.comm_info["current_metric_name"] = "p_auroc"

        self.trainer.logger.info("P-AUROC: %f" % p_auroc)
        self.trainer.logger.info("P-AP: %f" % p_ap)
        self.trainer.logger.info("O-AUROC: %f" % o_auroc)
        self.trainer.logger.info("O-AP: %f" % o_ap)
        self.trainer.logger.info("P-True: %f" % p_true)
        self.trainer.logger.info("P-Fake: %f" % p_fake)
        self.trainer.logger.info("F1: %f" % f1)

        self.trainer.logger.info("Best P-AUROC: %f" % self.best_acc)
        self.trainer.logger.info("<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")


