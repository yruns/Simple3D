# -*- coding: utf-8 -*-
"""
SimpleNet: A Simple Network for Image Anomaly Detection and Localization
Reference:
    https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf
Source:
    https://github.com/DonaldRR/SimpleNet
Licensed under the MIT License.
The script is based on the code of PatchCore:
    https://github.com/amazon-science/patchcore-inspection
"""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import patchcore
from utils.point_ops import simulate_realistic_industrial_anomaly, voxel_downsample_with_anomalies

LOGGER = logging.getLogger(__name__)


def init_weight(module: torch.nn.Module) -> None:
    """
    Initialize weights of a given module using Xavier initialization.

    Args:
        module (torch.nn.Module): The module to initialize.
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(module.weight)


class Discriminator(torch.nn.Module):
    """
    Discriminator network for anomaly detection.
    """

    def __init__(self, in_planes: int, n_layers: int = 1, hidden: int = None):
        """
        Initialize the Discriminator.

        Args:
            in_planes (int): Input feature dimension.
            n_layers (int): Number of layers.
            hidden (int): Hidden layer dimension. Defaults to None,
                          which sets hidden dimension to in_planes or a scaled value.
        """
        super().__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()

        for i in range(n_layers - 1):
            in_dim = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            block = torch.nn.Sequential(
                torch.nn.Linear(in_dim, _hidden),
                torch.nn.BatchNorm1d(_hidden),
                torch.nn.LeakyReLU(0.2)
            )
            self.body.add_module(f'block{i + 1}', block)

        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input features of shape (N, C).

        Returns:
            torch.Tensor: Discriminator output of shape (N, 1).
        """
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    """
    Projection network used to transform feature dimensions if necessary.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int = None,
        n_layers: int = 1,
        layer_type: int = 0
    ):
        super().__init__()

        if out_planes is None:
            out_planes = in_planes

        self.layers = torch.nn.Sequential()
        out_dim = None

        for i in range(n_layers):
            in_dim = in_planes if i == 0 else out_dim
            out_dim = out_planes

            self.layers.add_module(f"{i}fc", torch.nn.Linear(in_dim, out_dim))

            if layer_type > 1 and i < n_layers - 1:
                self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))

        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleNet(torch.nn.Module):
    """
    SimpleNet for anomaly detection and localization.
    """

    def __init__(
            self,
            device,
            layers_to_extract_from,
            target_embed_dimension,  # 1536
            anomaly_score_num_nn,
            nn_method,
            basic_template,
            voxel_size,
            noise_radius_range,
            embedding_size=None,  # 256
            meta_epochs=1,  # 40
            aed_meta_epochs=1,
            gan_epochs=1,  # 4
            dsc_layers=2,  # 2
            dsc_hidden=1024,  # 1024
            dsc_margin=0.5,  # 0.5
            dsc_lr=0.0002,
            train_backbone=False,
            cos_lr=False,
            lr=1e-3,
            pre_proj=1,  # 1
            proj_layer_type=0,
    ) -> None:
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = voxel_size

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn,
            nn_method=nn_method
        )
        self.noise_radius_range = noise_radius_range

        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None

        # Example usage of M3DM or other backbones:
        from M3DM.models import Model1  # pylint: disable=import-outside-toplevel

        self.deep_feature_extractor = Model1(
            device="cuda",
            rgb_backbone_name="vit_base_patch8_224_dino",
            xyz_backbone_name="Point_MAE",
            group_size=128,
            num_group=16384
        ).cuda()
        self.deep_feature_extractor.eval()

        self.embedding_size = embedding_size or self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        assert train_backbone is False, "Training the backbone is not supported."

        # Autoencoder/Similar stage meta-epochs.
        self.aed_meta_epochs = aed_meta_epochs
        self.pre_proj = pre_proj

        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_proj,
                proj_layer_type
            ).to(self.device)
            self.proj_opt = torch.optim.AdamW(
                self.pre_projection.parameters(),
                self.lr * 0.1
            )

        # Discriminator configuration.
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs

        self.discriminator = Discriminator(
            self.target_embed_dimension,
            n_layers=dsc_layers,
            hidden=dsc_hidden
        ).to(self.device)

        self.dsc_opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.dsc_lr,
            weight_decay=1e-5
        )

        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.dsc_opt,
            (meta_epochs - aed_meta_epochs) * gan_epochs,
            self.dsc_lr * 0.4
        )
        self.dsc_margin = dsc_margin

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

        self.forward_modules.eval()
        if self.pre_proj > 0 and self.pre_projection is not None:
            self.pre_projection.train()

        self.discriminator.train()


    def embed_pointmae(
            self,
            point_cloud: torch.Tensor,
            gt_mask=None,
    ):
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        training = gt_mask is None
        if training:     # Training
            # point_cloud = augment_point_cloud(point_cloud)
            point_cloud, gt_mask = simulate_realistic_industrial_anomaly(
                point_cloud)
        voxel_points, voxel_indices, voxel_labels = voxel_downsample_with_anomalies(
            point_cloud, gt_mask, voxel_size=self.voxel_size)

        # Registration
        # point_cloud = get_registration_refine_np(
        #     point_cloud,
        #     self.basic_template
        # )
        # Alternatively:
        # reg_data = point_cloud.squeeze(0).cpu().numpy()

        pointcloud_data = torch.from_numpy(point_cloud).permute(1, 0).unsqueeze(0)
        pointcloud_data = pointcloud_data.cuda().float()

        with torch.no_grad():
            pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(
                pointcloud_data,
                voxel_indices
            )

        if training:
            return pmae_features.squeeze(0).detach().permute(1, 0).contiguous(), center_idx, voxel_labels
        return pmae_features.squeeze(0).detach().permute(1, 0).contiguous(), center_idx, voxel_indices

    def forward(self, batch_data):
        """
        Train the discriminator for a certain number of epochs.

        Args:
            batch_data
        """
        input_pointcloud, mask, label, path = batch_data
        assert input_pointcloud.shape[0] == 1, "Batch size must be 1."

        # True features
        features, _, voxel_labels = self.embed_pointmae(input_pointcloud)
        if self.pre_proj > 0 and self.pre_projection is not None:
            features = self.pre_projection(features)

        # Discriminator forward
        voxel_labels = torch.from_numpy(voxel_labels).float().cuda()
        logits = self.discriminator(features).squeeze(-1)
        logits = torch.sigmoid(logits)

        return logits, voxel_labels

