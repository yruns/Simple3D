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

import torch
from torch import nn
import patchcore
import torch_scatter
from utils.point_ops import simulate_realistic_industrial_anomaly, voxel_downsample_with_anomalies, upsample

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

            defect_ratio=0.004,
            S=0.03,
            num_defects=6,
            upsample="v0"
    ) -> None:
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.defect_ratio = defect_ratio
        self.S = S
        self.num_defects = num_defects
        self.upsample_m = upsample

        # self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = voxel_size

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        preadapt_aggregator.to(self.device)
        # self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
        #     n_nearest_neighbours=anomaly_score_num_nn,
        #     nn_method=nn_method
        # )
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
                self.embedding_size,
                pre_proj,
                proj_layer_type
            ).to(self.device)

        if upsample == "v1":
            self.block_mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.2),
                nn.Linear(16, 8)
            ).to(device)

        # Discriminator configuration.
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embedding_size + (0 if upsample == "v0" else 8))
        ).to(self.device)

        self.discriminator = Discriminator(
            self.embedding_size + (0 if upsample == "v0" else 8),
            n_layers=dsc_layers,
            hidden=dsc_hidden
        ).to(self.device)

        self.dsc_margin = dsc_margin

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

        # self.forward_modules.eval()
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
        if training:  # Training
            # point_cloud = augment_point_cloud(point_cloud)
            # point_cloud = get_registration_refine_np(
            #     point_cloud,
            #     self.basic_template
            # )
            point_cloud, gt_mask = simulate_realistic_industrial_anomaly(
                point_cloud, defect_ratio=self.defect_ratio, S=self.S, num_defects=self.num_defects)

        voxel_points, voxel_indices, voxel_labels = voxel_downsample_with_anomalies(
            point_cloud, gt_mask, voxel_size=self.voxel_size)

        pointcloud_data = torch.from_numpy(point_cloud).permute(1, 0).unsqueeze(0)
        pointcloud_data = pointcloud_data.cuda().float()

        with torch.no_grad():
            pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(
                pointcloud_data,
                voxel_indices
            )

        if training:
            return pmae_features.squeeze(0).detach().permute(1, 0).contiguous(), ori_idx, gt_mask, \
                    torch.from_numpy(voxel_indices).cuda(), torch.from_numpy(voxel_labels).cuda(), center_idx
        return pmae_features.squeeze(0).detach().permute(1, 0).contiguous(), ori_idx, gt_mask, center_idx

    def forward(self, batch_data):
        """
        Train the discriminator for a certain number of epochs.

        Args:
            batch_data
        """
        input_pointcloud, mask, label, path = batch_data
        assert input_pointcloud.shape[0] == 1, "Batch size must be 1."

        # True features
        features, ori_idx, gt_mask, voxel_indices, voxel_labels, center_idx = self.embed_pointmae(input_pointcloud)
        if self.pre_proj > 0 and self.pre_projection is not None:
            features = self.pre_projection(features)

        # Add noise to features
        # noise = torch.randn_like(features) * 0.01
        # features = features + noise

        # coarse_logits = self.coarse_discriminator(features.squeeze(0)).squeeze(-1)

        # upsample
        features = self.upsample_forward(features, ori_idx, input_pointcloud, center_idx)

        # Discriminator forward
        gt_mask = torch.from_numpy(gt_mask).float().cuda()
        logits = self.discriminator(features.squeeze(0)).squeeze(-1)

        return logits, gt_mask, voxel_indices, voxel_labels

    def eval_step(self, pointcloud, gt_mask):
        features, ori_idx, gt_mask, center_idx = self.embed_pointmae(pointcloud, gt_mask.squeeze(0).cpu().numpy())
        if self.pre_proj > 0 and self.pre_projection is not None:
            features = self.pre_projection(features)

        features = self.upsample_forward(features, ori_idx, pointcloud, center_idx)
        scores = self.discriminator(features.squeeze(0)).squeeze(-1)

        return scores.detach().cpu().numpy()

    def upsample_forward(self, features, ori_idx, input_pointcloud, center_idx=None):
        # upsample
        if self.upsample_m == "v0":
            features = upsample(features.unsqueeze(0), ori_idx, input_pointcloud.shape[1])
        elif self.upsample_m == "v1":
            features = self.upsample(
                center_features=features.unsqueeze(0),
                ori_idx=ori_idx,
                original_num_points=input_pointcloud.shape[1],
                input_coords=input_pointcloud,
                center_idx=center_idx.to(torch.int64)
            )
        else:
            raise ValueError(f"Unknown upsample method: {self.upsample_m}")
        xyz = self.pos_embed(input_pointcloud)
        # features = torch.cat([features, xyz], dim=-1)
        features = features + xyz

        return features

    def upsample(self, center_features, ori_idx, original_num_points, input_coords, center_idx):
        """
        改进后的上采样方法，利用邻域分块和局部几何特征。

        Args:
            center_features: [B, G, C] 中心点特征
            ori_idx: [B, G, M] 邻域点索引
            original_num_points: 原始点云点数N
            input_coords: [B, N, 3] 原始点云坐标
            center_idx: [B, G] 中心点在原始点云中的索引
        Returns:
            [B, N, C+8] 上采样后的特征（包含分块几何特征）
        """
        B, G, M = ori_idx.shape
        C = center_features.size(2)
        device = center_features.device

        # 1. 获取中心点坐标 [B, G, 3]
        center_coords = torch.gather(input_coords, 1, center_idx.unsqueeze(-1).expand(-1, -1, 3))

        # 2. 获取邻域点坐标 [B, G, M, 3]
        neighbor_coords = torch.gather(
            input_coords.unsqueeze(1).expand(-1, G, -1, -1),
            2,
            ori_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )

        # 3. 计算相对坐标 [B, G, M, 3]
        relative_xyz = neighbor_coords - center_coords.unsqueeze(2)

        # 4. 计算分块索引 (0-7)
        x_sign = (relative_xyz[..., 0] >= 0).long()
        y_sign = (relative_xyz[..., 1] >= 0).long()
        z_sign = (relative_xyz[..., 2] >= 0).long()
        block_idx = x_sign * 4 + y_sign * 2 + z_sign  # [B, G, M]

        # 5. 生成全局分块标识符
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, G, M)
        center_indices = torch.arange(G, device=device).view(1, G, 1).expand(B, -1, M)
        global_block = batch_indices * G * 8 + center_indices * 8 + block_idx

        # 6. 展平处理
        flat_relative_xyz = relative_xyz.view(-1, 3)  # [B*G*M, 3]
        flat_global_block = global_block.view(-1)  # [B*G*M]

        # 7. 计算分块均值
        block_mean = torch_scatter.scatter_mean(flat_relative_xyz, flat_global_block, dim=0)

        # 8. 归一化坐标
        normalized_xyz = flat_relative_xyz - block_mean[flat_global_block]

        # 9. 通过MLP提取特征
        point_features = self.block_mlp(normalized_xyz)  # [B*G*M, 8]

        # 10. 分块最大池化
        block_features, _ = torch_scatter.scatter_max(point_features, flat_global_block, dim=0)

        # 11. 组合特征
        expanded_center = center_features.unsqueeze(2).expand(-1, -1, M, -1).reshape(-1, C)
        combined_features = torch.cat([expanded_center, block_features[flat_global_block]], dim=1)

        # 12. 聚合到原始点
        original_indices = ori_idx.view(-1)
        output = torch.zeros(B * original_num_points, C + 8, device=device)
        output = torch_scatter.scatter_mean(combined_features, original_indices, dim=0, out=output)

        return output.view(B, original_num_points, C + 8)

