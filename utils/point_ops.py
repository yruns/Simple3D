"""
File: utils.py
Date: 2025/3/22
Author: yruns

Description: This file contains ...
"""

import numpy as np
import open3d as o3d
from matplotlib.colors import to_rgb
import torch
import torch_scatter


def normalize_cube(P):
    """阶段一：立方体归一化 (对应Localized cube)"""
    centroid = np.mean(P, axis=0)
    P_centered = P - centroid
    scale = np.max(np.linalg.norm(P_centered, axis=1))
    return P_centered / (scale + 1e-8), centroid, scale


def viewpoint_selection(Pa, N_defect):
    """改进的视点采样，包含边界检查"""
    # 在单位球面生成视点（限制在点云包围盒内）
    bbox_min = np.min(Pa, axis=0)
    bbox_max = np.max(Pa, axis=0)

    while True:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        Pv = np.array([np.sin(phi) * np.cos(theta),
                       np.sin(phi) * np.sin(theta),
                       np.cos(phi)])

        # 检查视点是否在有效范围内
        if np.all((Pv >= bbox_min - 0.1) & (Pv <= bbox_max + 0.1)):
            break

    # 选择最近邻点块（带距离衰减权重）
    distances = np.linalg.norm(Pa - Pv, axis=1)
    indices = np.argpartition(distances, N_defect)[:N_defect]
    return indices, Pv


def apply_deformation(Pa, indices, Pv, mode=None, S=0.3):
    """阶段三：多模式变形 (对应Deformation solution)"""
    direction = Pa[indices] - Pv
    norms = np.linalg.norm(direction, axis=1, keepdims=True)

    if mode is None:
        mode = np.random.choice(['bulge', 'sink', 'damage'])

    # 不同变形模式的位移矩阵
    if mode == 'bulge':
        T = -np.linspace(1, 0, len(indices))[:, np.newaxis]
    elif mode == 'sink':
        T = np.linspace(1, 0, len(indices))[:, np.newaxis] * 1.2
    elif mode == 'damage':
        T = np.random.randn(len(indices), 3) * 0.3

    Pa_trans = Pa.copy()
    Pa_trans[indices] += S * (direction / (norms + 1e-8)) * T
    return Pa_trans


def simulate_realistic_industrial_anomaly(P, defect_ratio=0.004, S=0.03, num_defects=6):
    """模拟多个工业异常区域（简化版）
    Args:
        P: 原始点云，形状为 (N, 3)
        defect_ratio: 每个缺陷区域占点数的比例（总缺陷点数 = num_defects * defect_ratio * N）
        S: 变形强度
        num_defects: 生成的缺陷区域数量，随机 1 ~ num_defects
    """
    # 立方体归一化
    Pa_normalized, centroid, scale = normalize_cube(P)
    total_points = len(Pa_normalized)
    mask = np.zeros(total_points)
    Pa_deformed = Pa_normalized.copy()

    # 生成多个独立缺陷
    num_defects = np.random.randint(1, num_defects + 1)
    for _ in range(num_defects):
        # 随机选择视点和缺陷点（允许区域重叠）
        N_defect = int(total_points * defect_ratio)
        indices, Pv = viewpoint_selection(Pa_normalized, N_defect)

        # 应用随机模式变形
        mode = np.random.choice(['bulge', 'sink', 'damage'])
        Pa_deformed = apply_deformation(Pa_deformed, indices, Pv, mode, S)

        # 合并掩码
        mask[indices] = 1

    # 恢复坐标系
    restored_deformed = Pa_deformed * scale + centroid
    return restored_deformed, mask


def voxel_downsample_with_anomalies(points, modified_mask=None, voxel_size=0.5):
    # 计算每个点所属的voxel坐标
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)

    # 获取唯一的voxel坐标及其索引
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)

    # 计算每个voxel的中心点坐标（所有点的平均值）
    voxel_points = np.zeros((unique_voxels.shape[0], 3))
    np.add.at(voxel_points, inverse_indices, points)
    counts = np.bincount(inverse_indices)
    voxel_points /= counts[:, np.newaxis]

    # 计算每个点到其对应voxel中心的欧几里得距离
    distances = np.linalg.norm(points - voxel_points[inverse_indices], axis=1)

    # 获取每个voxel的代表点索引（距离voxel中心最近的点）
    voxel_representative_indices = np.zeros(len(unique_voxels), dtype=int)
    unique_inverse_argmin = np.argsort(inverse_indices, kind='stable')
    sorted_inverse_indices = inverse_indices[unique_inverse_argmin]

    first_occurrences = np.concatenate(([True], sorted_inverse_indices[1:] != sorted_inverse_indices[:-1]))
    voxel_representative_indices[sorted_inverse_indices[first_occurrences]] = unique_inverse_argmin[first_occurrences]

    # 确定每个voxel是否含有异常点（modified_mask中不为0）
    voxel_labels = np.zeros(len(unique_voxels), dtype=int)
    np.maximum.at(voxel_labels, inverse_indices, modified_mask.astype(int))

    return voxel_points, voxel_representative_indices, voxel_labels


def upsample(center_features, ori_idx, original_num_points):
    """
        center_features: [B, G, C] 下采样后的特征（如中心点特征）
        ori_idx: [B, G, M] 邻域索引，表示每个中心点的邻域点在原始点云中的索引
        original_num_points: 原始点云的点数 N
        返回: [B, N, C] 上采样后的特征，每个原始点的特征
    """
    B, G, M = ori_idx.shape
    C = center_features.size(2)

    # 扩展中心点特征到每个邻域点 [B, G, M, C] -> [B*G*M, C]
    expanded_features = center_features.unsqueeze(2).expand(-1, -1, M, -1).reshape(B * G * M, C)

    # 获取全局索引并展平 [B*G*M]
    indices = ori_idx.reshape(B * G * M)

    # 初始化输出张量 [B*N, C]
    output = torch.zeros(B * original_num_points, C, device=center_features.device)

    # 使用scatter_mean聚合特征
    output = torch_scatter.scatter_mean(expanded_features, indices, dim=0, out=output)

    # 调整形状为 [B, N, C]
    upsampled_features = output.view(B, original_num_points, C)

    return upsampled_features


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("/Users/yruns/Downloads/Real3D-AD-PCD/airplane/train/287_template.pcd")
    points = np.array(pcd.points)

    points, mask = simulate_realistic_industrial_anomaly(points)

    colors = np.zeros((len(points), 3))
    colors[mask == 1] = to_rgb('red')
    colors[mask == 0] = to_rgb('gray')

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
