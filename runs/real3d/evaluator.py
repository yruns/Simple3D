import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from trim.callbacks.default import CallbackBase
from trim.utils import comm
from utils.metrics import compute_metrics
from utils.point_ops import upsample
from sklearn.metrics import precision_recall_curve


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
            "data": []
        }
        obj_pred, mask_pred, obj_gt, mask_gt = [], [], [], []
        val_iter = tqdm(self.trainer.val_loader, total=len(self.trainer.val_loader))
        for i, batch_data in enumerate(val_iter):
            batch_data = comm.move_tensor_to_device(batch_data)

            pointcloud, mask, label, path = batch_data
            scores, ori_score = self.eval_step(pointcloud, mask)

            obj_pred.append(np.max(ori_score, axis=0))
            mask_pred.append(scores)
            mask_gt.append(mask[0].cpu().numpy())
            obj_gt.append(label[0].cpu().numpy())

            save_dict["data"].append({
                # "pointcloud": pointcloud.squeeze(0).cpu().numpy()[voxel_indices],
                "pred": scores,
                # "mask": voxel_labels,
                "obj_pred": np.max(ori_score, axis=0),
                "mask": mask[0].cpu().numpy(),
                "label": label.squeeze(0).cpu().numpy(),
                "path": path[0],
                "epoch": self.trainer.epoch,
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

            os.makedirs(os.path.join(self.trainer.output_dir, "eval"), exist_ok=True)
            with open(os.path.join(
                    self.trainer.output_dir, "eval",
                    f"{self.trainer.cls_name}.pkl"
            ), "wb") as f:
                pickle.dump(save_dict, f)
                self.trainer.logger.info("Save evaluation results to %s" % f.name)

        if o_auroc > trainer_metrics["best_oauroc"]:
            trainer_metrics["best_oauroc"] = o_auroc
            trainer_metrics["best_oauroc_epoch"] = self.trainer.epoch
            trainer_metrics["pauroc_when_o_best"] = p_auroc
            error_metrics = self.analye_imgL_metrics(obj_pred, obj_gt)
            trainer_metrics["error_metrics"] = error_metrics

        self.trainer.metrics_history = trainer_metrics

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

    def analye_imgL_metrics(self, obj_pred, obj_gt):
        # Compute metrics
        obj_pred = torch.sigmoid(torch.from_numpy(obj_pred)).numpy()
        obj_gt = np.array(obj_gt)

        # Compute metrics
        precision, recall, thresholds = precision_recall_curve(obj_gt, obj_pred)
        # 计算每个阈值对应的F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        pred_labels = (obj_pred >= best_threshold).astype(np.int32)
        # 获取各样本的真实标签(假设0为正常,1为异常)
        gt_labels = obj_gt.astype(int)

        # 统计四类情况
        true_mask = (pred_labels == gt_labels)
        fp_mask = (gt_labels == 0) & (pred_labels == 1)  # 正常样本误判为异常
        fn_mask = (gt_labels == 1) & (pred_labels == 0)  # 异常样本误判为正常
        tp_mask = (gt_labels == 1) & (pred_labels == 1)  # 正确识别异常
        tn_mask = (gt_labels == 0) & (pred_labels == 0)  # 正确识别正常

        # 计算各类数量
        n_fp = fp_mask.sum()
        n_fn = fn_mask.sum()
        n_tp = tp_mask.sum()
        n_tn = tn_mask.sum()

        # 计算误判样本的预测值统计
        fp_scores = obj_pred[fp_mask] if n_fp > 0 else []
        fn_scores = obj_pred[fn_mask] if n_fn > 0 else []

        # 构建统计指标
        error_metrics = {
            "val_fp_count": n_fp,
            "val_fn_count": n_fn,
            "val_fp_mean": np.mean(fp_scores) if n_fp > 0 else 0,
            "val_fn_mean": np.mean(fn_scores) if n_fn > 0 else 0,
            "val_tp_count": n_tp,
            "val_tn_count": n_tn,
            "val_fp_max": np.max(fp_scores) if n_fp > 0 else 0,
            "val_fn_max": np.max(fn_scores) if n_fn > 0 else 0,
            "val_accuracy": (n_tp + n_tn) / len(gt_labels)
        }

        return error_metrics


