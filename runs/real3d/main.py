"""
File: main.py
Date: 2025/3/17
Author: yruns

"""
import argparse
from glob import glob

import numpy as np
import torch.nn.parallel
import torch.optim
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torch.utils.data

from trim.callbacks.misc import *
from trim.engine.launch import launch
from trim.engine.trainer import TrainerBase
from trim.thirdparty.logging import WandbWrapper
from trim.thirdparty.logging import logger
from trim.utils import comm, dist
from utils import metrics


class Trainer(TrainerBase):

    def __init__(self, cls_name, args, logger, debug=False, callbacks=None):
        super().__init__()
        self.all_f1 = None
        self.all_loss = []
        self.all_pixel_ap = []
        self.all_pixel_auroc = []
        self.all_p_true = []
        self.all_p_fake = []

        self.dsc_schl = None
        self.optimizer = None
        self.proj_opt = None
        self.cls_name = cls_name
        self.args = args
        self.logger = logger
        self.max_epoch = 0
        self.output_dir = args.output_dir
        self.callbacks = callbacks or []
        self.debug = debug

    def configure_model(self):
        logger.info("=> creating model ...")
        from patchcore.common import FaissNN
        nn_method = FaissNN(self.args.faiss_on_gpu, self.args.faiss_num_workers)
        from models.simplenet import SimpleNet
        self.model = SimpleNet(
            device=torch.device("cuda"),
            layers_to_extract_from=None,
            target_embed_dimension=1152,
            anomaly_score_num_nn=self.args.anomaly_scorer_num_nn,
            nn_method=nn_method,
            basic_template=None,
            voxel_size=self.args.voxel_size,
            noise_radius_range=(0.05, 0.10),

            embedding_size=256,
            meta_epochs=40,  # 40
            aed_meta_epochs=1,
            gan_epochs=4,  # 4
            dsc_layers=2,  # 2
            dsc_hidden=None,  # 1024
            dsc_margin=0.2,  # 0.5
            dsc_lr=1e-3,
            train_backbone=False,
            cos_lr=True,
            lr=1e-3,
            pre_proj=1,  # 1
            proj_layer_type=0,

            defect_ratio=self.args.defect_ratio,
            S=self.args.S,
            num_defects=self.args.num_defects,
        )
        self.max_epoch = self.args.max_epoch

        num_parameters = comm.count_parameters(self.model, trainable=True)
        self.wandb.update({"model_size": f"{num_parameters / 1e6}M"})
        logger.info('Number of params: %.4f M' % (num_parameters / 1e6))

    def configure_dataloader(self):
        logger.info("=> creating data loader ...")

        self.args.batch_size = 1
        self.args.batch_size_val = 1
        self.args.workers = 0
        assert not dist.is_distritubed(), "Not support distributed training yet."
        from datasets.real3d import Real3DDataset
        self.train_loader = torch.utils.data.DataLoader(
            Real3DDataset(dataset_dir=self.args.data, split="train",
                          norm=True, cls_name=self.cls_name, aug=self.args.aug_pointcloud),
            batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers, pin_memory=True,
            drop_last=False
        )
        self.val_loader = torch.utils.data.DataLoader(
            Real3DDataset(dataset_dir=self.args.data, split="test", norm=True, cls_name=self.cls_name),
            batch_size=self.args.batch_size_val, shuffle=True, num_workers=self.args.workers, pin_memory=True,
            drop_last=False
        )

        # set basic template
        for data, mask, label, path in self.train_loader:
            self.model.basic_template = data.squeeze(0).cpu().numpy()
            break
        self.logger.info(f"length of training dataset: {len(self.train_loader.dataset)}")
        self.logger.info(f"length of validation dataset: {len(self.val_loader.dataset)}")

    def configure_optimizers(self):
        logger.info("=> creating optimizer ...")
        # Get all parameters except those from pre_projection and deep_feature_extractor
        exclude_modules = ['pre_projection', 'deep_feature_extractor']
        params = []
        for name, param in self.model.named_parameters():
            if not any(exclude_module in name for exclude_module in exclude_modules):
                params.append(param)

        # Create a single parameter group with the filtered parameters
        param_groups = [{
            'params': params,
            'lr': self.model.lr,  # Using the base learning rate
            'weight_decay': 1e-5
        }]

        if self.model.pre_proj > 0 and self.model.pre_projection is not None:
            param_groups.append({
                'params': self.model.pre_projection.parameters(),
                'lr': self.model.lr * 0.1,
                'weight_decay': 1e-5
            })

        # 创建统一优化器
        self.optimizer = torch.optim.AdamW(param_groups)
        if self.model.cos_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                self.model.meta_epochs * self.model.gan_epochs,
                self.model.dsc_lr * 0.4
            )

    def configure_wandb(self):
        # When debugging, we don't need to log anything.
        self.wandb = WandbWrapper(
            project=self.args.log_project,
            name=self.args.log_tag,
            config={"log_tag": self.args.log_tag, "config": vars(self.args)},
            save_code=False,
            resume=False,
            file_prefix=os.path.join(self.output_dir, "codebase"),
            save_files=[__file__, *glob("models/*.py"), *glob("runs/real3d/evaluator.py")],
            debug=self.debug or self.args.no_wandb
        )
        self.wandb.update({"output_dir": self.output_dir})

    def on_training_epoch_start(self):
        if not self.args.eval:
            self.all_loss = []
            self.all_pixel_ap = []
            self.all_pixel_auroc = []
            self.all_p_true = []
            self.all_p_fake = []
            self.all_f1 = []
            self.model.deep_feature_extractor.eval()
        super().on_training_epoch_start()

    def on_training_epoch_end(self):
        if not self.args.eval:
            if self.model.cos_lr:
                self.scheduler.step()

            # Trace the loss, pixel_ap, pixel_auroc, p_true, p_fake
            loss = np.array(self.all_loss).mean()
            p_ap = np.array(self.all_pixel_ap).mean()
            p_auroc = np.array(self.all_pixel_auroc).mean()
            p_true = np.array(self.all_p_true).mean()
            p_fake = np.array(self.all_p_fake).mean()
            # f1 = np.array(self.all_f1).mean()
            self.logger.info(
                f"Epoch {self.epoch}: "
                f"loss={loss:.3f}, "
                f"p_ap={p_ap:.3f}, "
                f"p_auroc={p_auroc:.3f}, "
                f"p_true={p_true:.3f}, "
                f"p_fake={p_fake:.3f}, "
                # f"f1={f1:.3f}"
            )
            self.wandb.log({
                "train_loss": loss,
                "train_pixel_ap": p_ap,
                "train_pixel_auroc": p_auroc,
                "train_p_true": p_true,
                "train_p_fake": p_fake,
                # "train_f1": f1
            })
        super().on_training_epoch_end()

    def training_step(self, batch_data, batch_index):
        batch_data = comm.move_tensor_to_device(batch_data)
        logits, full_mask, corse_logits, voxel_labels = self.model(batch_data)

        full_loss = metrics.focal_loss(
            logits, full_mask, alpha=self.args.focal_loss_a, gamma=self.args.focal_loss_g, eps=1e-7)
        voxel_loss = metrics.focal_loss(
            corse_logits, voxel_labels, alpha=self.args.focal_loss_a, gamma=self.args.focal_loss_g, eps=1e-7)
        logits = torch.sigmoid(logits)
        p_ap, p_auroc, p_true, p_fake, f1, best_threshold = metrics.compute_metrics(None, logits, full_mask, None)

        loss = full_loss + voxel_loss * self.args.alpha

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log_dict = {
            "loss": loss.item(),
            "p_ap": p_ap,
            "p_auroc": p_auroc,
            "p_true": p_true,
            "p_fake": p_fake,
            # "f1": f1
        }
        self.all_loss.append(log_dict["loss"])
        self.all_pixel_ap.append(log_dict["p_ap"])
        self.all_pixel_auroc.append(log_dict["p_auroc"])
        self.all_p_true.append(log_dict["p_true"])
        self.all_p_fake.append(log_dict["p_fake"])
        # self.all_f1.append(log_dict["f1"])
        self.comm_info["terminal_log"] = log_dict
        self.comm_info["wandb_log"] = {}


def main_worker(args):
    from torch.backends import cudnn
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s'
        % (torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)
    )
    import time
    now = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    args.output_dir = f"output/Simple3D-{now}"
    args.log_project = "Simple3DUpSampledV4"


    comm.seed_everything(args.manual_seed)
    comm.copy_codebase(args.output_dir)

    debug = args.eval or args.debug
    from runs.real3d.evaluator import ClsEvaluator

    real_3d_classes = sorted(os.listdir(args.data))
    # real_3d_classes = ["candybar"]
    auroc_list = {}

    voxel_size = args.voxel_size
    for cls_name in real_3d_classes:
        args.log_tag = (f"vs{args.voxel_size}-{cls_name}-noise"
                        f"-alpha{args.alpha}-focal{args.focal_loss_a}-{args.focal_loss_g}-aug{args.aug_pointcloud}")
        if cls_name in ["duck"]:
            args.voxel_size = 0.7

        trainer = Trainer(cls_name, args, logger, debug=debug, callbacks=[
            CheckpointLoader(state_path=args.model_path, resume=not args.eval),
            IterationTimer(warmup_iter=2),
            InformationWriter(log_interval=1),
            ClsEvaluator(interval=args.eval_interval if not args.debug else 1, test=args.eval),
            # Visualizer(),
            CheckpointSaver(),
        ])

        if not args.eval:
            trainer.fit()
        else:
            trainer.test()

        args.voxel_size = voxel_size

        pauroc = trainer.best_metric_value
        auroc_list[cls_name] = pauroc

    # Statistics
    for cls_name, pauroc in auroc_list.items():
        print(f"Class: {cls_name}, AUROC: {pauroc:.4f}")
        logger.info(f"Class: {cls_name}, AUROC: {pauroc:.4f}")
    print("Mean AUROC: ", np.mean(list(auroc_list.values())))
    logger.info("Mean AUROC: %f" % np.mean(list(auroc_list.values())))
    # trainer.wandb.update({"mean_auroc": np.mean(list(auroc_list.values()))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple3D')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--manual_seed', type=int, default=2025)
    parser.add_argument('--faiss_on_gpu', default=True, type=bool)
    parser.add_argument('--faiss_num_workers', default=8, type=int)
    parser.add_argument('--anomaly_scorer_num_nn', default=1, type=int)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--voxel_size', type=float, default=0.5)
    parser.add_argument('--aug_pointcloud', type=bool, default=False)
    parser.add_argument('--defect_ratio', type=float, default=0.004)
    parser.add_argument('--S', type=float, default=0.03)
    parser.add_argument('--num_defects', type=int, default=6)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--focal_loss_a', type=float, default=0.5)
    parser.add_argument('--focal_loss_g', type=float, default=2)

    parser.add_argument('--eval', action="store_true", help='is evaluation')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--debug', action="store_true", help='is evaluation')
    parser.add_argument('--no_wandb', action="store_true", help='is evaluation')
    args = parser.parse_args()

    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(",")
    launch(
        main_worker,
        num_gpus_per_machine=len(visible_device),
        args=(args,),
    )