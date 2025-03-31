import torch
import torch.nn as nn
import torch_scatter
import numpy as np
import timm
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from timm.models.layers import DropPath

# from knn_cuda import KNN
from M3DM.cpu_knn import KNN

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group  # 1024
        self.group_size = group_size  # 128
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, sample_idx):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        # center, center_idx_offical = fps(xyz.contiguous().float(), self.num_group)  # B G 3   B 1024 ,B 1024 3
        center_idx = np.array(sample_idx)
        center_idx = center_idx.astype(np.int32)
        center_idx = torch.from_numpy(center_idx).to(xyz.device)
        center_idx = torch.reshape(center_idx, ((1, center_idx.shape[0])))
        center = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), center_idx).transpose(1,
                                                                                                          2).contiguous()
        # knn to get the neighborhood
        center = torch.where(torch.isnan(center), torch.full_like(center, 0), center)
        _, idx = self.knn(xyz, center)  # B G M    idx:B 1024 128 xyz:B N 3  center:B 1024 3
        idx = idx.to(device=xyz.device)
        # assert idx.size(1) == self.num_group
        # assert idx.size(2) == self.group_size
        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, ori_idx.size(1), ori_idx.size(2), 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx

class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, center_features, ori_idx, original_num_points):
        """
        center_features: [B, G, C] 下采样后的特征（如中心点特征）
        ori_idx: [B, G, M] 邻域索引，表示每个中心点的邻域点在原始点云中的索引
        original_num_points: 原始点云的点数 N
        返回: [B, N, C] 上采样后的特征，每个原始点的特征
        """
        B, G, M = ori_idx.shape
        C = center_features.size(2)

        # 扩展中心点特征到每个邻域点 [B, G, M, C] -> [B*G*M, C]
        expanded_features = center_features.unsqueeze(2).expand(-1, -1, M, -1).reshape(B*G*M, C)

        # 获取全局索引并展平 [B*G*M]
        indices = ori_idx.reshape(B*G*M)

        # 初始化输出张量 [B*N, C]
        output = torch.zeros(B * original_num_points, C, device=center_features.device)

        # 使用scatter_mean聚合特征
        output = torch_scatter.scatter_mean(expanded_features, indices, dim=0, out=output)

        # 调整形状为 [B, N, C]
        upsampled_features = output.view(B, original_num_points, C)

        return upsampled_features

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.group = Group(num_group=1024, group_size=128)
        self.upsample = Upsample()

        # 示例网络：处理邻域得到中心点特征
        self.center_feature_net = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        # 最终MLP处理特征和坐标
        self.mlp = nn.Sequential(
            nn.Conv1d(128 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, xyz, sample_idx, sampled_point_features):
        """
        xyz: [B, N, 3] 原始点云坐标
        sample_idx: [B, G] 采样点索引
        sampled_point_features: [B, G, C] 采样点的多模态特征（如图像特征）
        """
        # 获取邻域和中心点信息
        neighborhood, center, ori_idx, _ = self.group(xyz, sample_idx)
        B, G, M, _ = neighborhood.shape

        # 修改3: 直接使用传入的采样点特征（不再通过center_feature_net）
        # 将采样点特征上采样回原始点数 [B, G, C] -> [B, N, C]
        upsampled_features = self.upsample(sampled_point_features, ori_idx, xyz.size(1))

        # 修改4: 特征融合（上采样特征 + 几何邻域特征）
        # 提取几何特征（原center_feature_net流程）
        neighborhood = neighborhood.permute(0, 3, 1, 2)  # [B, 3, G, M]
        geo_features = self.center_feature_net(neighborhood)  # [B, 128, G, 1]
        geo_features = geo_features.squeeze(-1).squeeze(-1)  # [B, 128, G]
        geo_features = geo_features.permute(0, 2, 1)  # [B, G, 128]

        # 将几何特征也上采样到原始点数
        upsampled_geo = self.upsample(geo_features, ori_idx, xyz.size(1))  # [B, N, 128]

        # 特征拼接：多模态特征 + 几何特征
        combined = torch.cat([upsampled_features, upsampled_geo], dim=-1)  # [B, N, C+128]

        # 修改5: 增加特征融合层
        combined = combined.permute(0, 2, 1)  # [B, C+128, N]
        combined = self.fusion_net(combined)  # [B, 128, N]

        # 拼接坐标信息
        final_input = torch.cat([combined, xyz.permute(0, 2, 1)], dim=1)  # [B, 128+3, N]

        # 最终MLP
        score = self.mlp(final_input)  # [B, 1, N]
        return score.permute(0, 2, 1)  # [B, N, 1]

if __name__ == '__main__':
    model = MyModel().cuda()
    xyz = torch.randn(1, 10240, 3).cuda()
    # sample_idx [K]
    sample_idx = torch.randint(0, 10240, (2048,)).numpy()
    sampled_feat = torch.randn(1, 2048, 128).cuda()
    score = model(xyz, sample_idx, sampled_feat)
    print(score.shape)  # torch.Size([2, 1024, 1])
